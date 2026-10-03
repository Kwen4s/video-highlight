from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol

import httpx
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from .config import Settings
from .editing import EditOutcome, edit_highlights, validate_highlight_range
from .models import DetectionResult, Highlight

EditOperation = Literal[
    "move_start",
    "move_end",
    "set_range",
    "delete",
    "rename",
    "update_reason",
    "split",
    "merge",
]
QueryOperation = Literal["list", "details", "compare", "history", "help"]


class EditPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["edit"]
    operation: EditOperation
    highlight_ids: list[str] = Field(default_factory=list)
    delta_sec: float | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    split_sec: float | None = None
    text: str | None = None

    @model_validator(mode="after")
    def validate_arguments(self) -> EditPlan:
        if not self.highlight_ids:
            raise ValueError("edit plan requires at least one highlight_id")
        if len(set(self.highlight_ids)) != len(self.highlight_ids):
            raise ValueError("highlight_ids must be unique")
        if self.operation in {"move_start", "move_end"} and (
            self.delta_sec is None or self.delta_sec == 0
        ):
            raise ValueError("boundary movement requires a non-zero delta_sec")
        if self.operation == "set_range" and (self.start_sec is None or self.end_sec is None):
            raise ValueError("set_range requires start_sec and end_sec")
        if self.operation in {"rename", "update_reason"} and not (self.text and self.text.strip()):
            raise ValueError(f"{self.operation} requires text")
        if self.operation == "split" and self.split_sec is None:
            raise ValueError("split requires split_sec")
        if self.operation in {"rename", "update_reason", "split"} and len(self.highlight_ids) != 1:
            raise ValueError(f"{self.operation} requires exactly one highlight")
        if self.operation == "merge" and len(self.highlight_ids) < 2:
            raise ValueError("merge requires at least two highlights")
        return self


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["query"]
    operation: QueryOperation
    highlight_ids: list[str] = Field(default_factory=list)


class UndoPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["undo"]


class ClarifyPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["clarify"]
    reply: str = Field(min_length=1, max_length=300)


CommandPlan = Annotated[
    EditPlan | QueryPlan | UndoPlan | ClarifyPlan,
    Field(discriminator="kind"),
]
COMMAND_PLAN_ADAPTER = TypeAdapter(CommandPlan)


class CommandPlanner(Protocol):
    def plan(
        self,
        *,
        result: DetectionResult,
        message: str,
        selected_highlight_id: str | None,
        conversation: list[dict[str, Any]],
    ) -> CommandPlan: ...


class ConversationAgentError(RuntimeError):
    pass


class PlanExecutionError(ValueError):
    pass


@dataclass
class ConversationOutcome:
    result: DetectionResult
    reply: str
    changed: bool = False
    undo: bool = False
    action: dict[str, Any] | None = None


PLANNER_SYSTEM_PROMPT = """你是 Video Highlight 的命令规划器。
你只能把请求编译成一个 JSON 计划，不能直接修改数据。

可用计划：
- edit: move_start、move_end、set_range、delete、rename、update_reason、split、merge。
- query: list、details、compare、history、help。
- undo: 撤销最近一次结果修改。
- clarify: 缺少目标、方向、秒数或含义不完整时先澄清。

硬性规则：
1. 只能使用输入中出现的稳定 highlight_id，不能把“第几段”的序号直接作为 ID。
2. move_start/move_end 的 delta_sec 为有符号秒数：向后为正、向前为负。
3. set_range 使用绝对秒数。split_sec 也是原片上的绝对秒数。
4. “缩短”“延长”等未说明调整哪一端的请求必须 clarify，不能猜测。
5. 不得声称看过视频、字幕、帧或 Agent trace；只能依据给出的公开高光摘要回答。
6. 不支持重新检测、生成新高光、导出、视频内容检索时，用 clarify 说明当前能力边界。
7. 结合当前选中片段和最近对话解析“这段”“上一段”“再短一点”等指代；仍有歧义则 clarify。
8. 只输出符合给定 schema 的 JSON，不输出 Markdown 或解释。
"""


class OpenAICommandPlanner:
    def __init__(self, settings: Settings) -> None:
        if not settings.chat_api_key:
            raise ValueError("VH_CHAT_API_KEY is empty")
        self.model = settings.chat_model
        self.client = OpenAI(
            api_key=settings.chat_api_key,
            base_url=settings.chat_base_url,
            timeout=settings.chat_timeout_sec,
            max_retries=settings.chat_max_retries,
            http_client=httpx.Client(trust_env=False),
        )

    def plan(
        self,
        *,
        result: DetectionResult,
        message: str,
        selected_highlight_id: str | None,
        conversation: list[dict[str, Any]],
    ) -> CommandPlan:
        highlights = [
            {
                "index": index,
                "highlight_id": item.highlight_id,
                "start_sec": item.start_sec,
                "end_sec": item.end_sec,
                "score": item.score,
                "highlight_type": item.highlight_type,
                "description": item.description,
                "reason": item.reason,
                "review_status": item.review_status,
            }
            for index, item in enumerate(result.highlights, start=1)
        ]
        recent_turns = [
            {
                "user": turn.get("user_message", ""),
                "assistant": turn.get("assistant_reply", ""),
                "action": turn.get("action"),
            }
            for turn in conversation[-6:]
        ]
        prompt = {
            "video": result.video.model_dump(mode="json"),
            "highlights": highlights,
            "selected_highlight_id": selected_highlight_id,
            "recent_conversation": recent_turns,
            "user_message": message,
            "output_schema": COMMAND_PLAN_ADAPTER.json_schema(),
        }
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
                temperature=0,
                max_tokens=900,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("model returned an empty plan")
            return COMMAND_PLAN_ADAPTER.validate_json(content)
        except Exception as error:
            raise ConversationAgentError("对话 Agent 规划失败，请稍后重试。") from error


class HighlightConversationAgent:
    def __init__(
        self,
        settings: Settings,
        planner: CommandPlanner | None = None,
    ) -> None:
        self.planner = planner
        if self.planner is None and settings.chat_api_key:
            self.planner = OpenAICommandPlanner(settings)

    def respond(
        self,
        *,
        result: DetectionResult,
        message: str,
        selected_highlight_id: str | None,
        conversation: list[dict[str, Any]],
    ) -> ConversationOutcome:
        if self.planner is None or _is_explicit_local_command(message):
            local = edit_highlights(result, message, selected_highlight_id)
            return _from_local_outcome(local, message, selected_highlight_id)

        plan = self.planner.plan(
            result=result,
            message=message,
            selected_highlight_id=selected_highlight_id,
            conversation=conversation,
        )
        try:
            return execute_plan(plan, result, conversation)
        except PlanExecutionError as error:
            return ConversationOutcome(
                result=result.model_copy(deep=True),
                reply=str(error),
                action={"kind": "clarify", "reply": str(error)},
            )


def execute_plan(
    plan: CommandPlan,
    result: DetectionResult,
    conversation: list[dict[str, Any]],
) -> ConversationOutcome:
    action = plan.model_dump(mode="json")
    if isinstance(plan, UndoPlan):
        return ConversationOutcome(
            result=result.model_copy(deep=True),
            reply="正在撤销上一次高光修改。",
            undo=True,
            action=action,
        )
    if isinstance(plan, ClarifyPlan):
        return ConversationOutcome(
            result=result.model_copy(deep=True),
            reply=plan.reply,
            action=action,
        )
    if isinstance(plan, QueryPlan):
        return ConversationOutcome(
            result=result.model_copy(deep=True),
            reply=_execute_query(plan, result, conversation),
            action=action,
        )
    return _execute_edit(plan, result, action)


def _execute_query(
    plan: QueryPlan,
    result: DetectionResult,
    conversation: list[dict[str, Any]],
) -> str:
    if plan.operation == "help":
        return (
            "我可以查询候选及其理由，调整入点、出点或绝对时间范围，修改标题和说明，"
            "删除、拆分、合并片段，以及撤销最近一次修改。"
        )
    if plan.operation == "history":
        if not conversation:
            return "当前会话还没有之前的操作。"
        rows = [
            f"{index}. {turn.get('user_message', '')} → {turn.get('assistant_reply', '')}"
            for index, turn in enumerate(conversation[-5:], start=1)
        ]
        return "最近的操作：" + "；".join(rows)
    if plan.operation == "list":
        if not result.highlights:
            return "当前没有高光候选。"
        rows = [
            f"第{index}段 {item.start_sec:.1f}–{item.end_sec:.1f}秒"
            f"「{item.description}」（{round(item.score * 100)}分）"
            for index, item in enumerate(result.highlights, start=1)
        ]
        return f"当前共有 {len(rows)} 段：" + "；".join(rows) + "。"

    targets = _resolve_ids(result, plan.highlight_ids)
    if not targets:
        raise PlanExecutionError("请说明要查询哪一段高光。")
    if plan.operation == "details":
        rows = [
            f"「{item.description}」位于 {item.start_sec:.1f}–{item.end_sec:.1f} 秒，"
            f"类型为 {item.highlight_type}，评分 {round(item.score * 100)}。{item.reason}"
            for item in targets
        ]
        return "\n".join(rows)

    ordered = sorted(targets, key=lambda item: item.score, reverse=True)
    rows = [
        f"「{item.description}」{round(item.score * 100)}分，理由：{item.reason}"
        for item in ordered
    ]
    return "按当前检测评分从高到低：" + "；".join(rows) + "。"


def _execute_edit(
    plan: EditPlan,
    result: DetectionResult,
    action: dict[str, Any],
) -> ConversationOutcome:
    working = result.model_copy(deep=True)
    targets = _resolve_ids(working, plan.highlight_ids)
    if len(targets) != len(plan.highlight_ids):
        known = {item.highlight_id for item in targets}
        missing = [item for item in plan.highlight_ids if item not in known]
        raise PlanExecutionError(f"找不到这些高光：{', '.join(missing)}，请重新选择后再试。")

    if plan.operation == "delete":
        target_ids = set(plan.highlight_ids)
        working.highlights = [
            item for item in working.highlights if item.highlight_id not in target_ids
        ]
        return _changed(working, f"已删除 {len(target_ids)} 段高光。你可以发送“撤销”恢复。", action)

    if plan.operation in {"move_start", "move_end", "set_range"}:
        updates: list[tuple[Highlight, float, float]] = []
        for item in targets:
            if plan.operation == "move_start":
                start_sec = max(0.0, item.start_sec + (plan.delta_sec or 0.0))
                end_sec = item.end_sec
            elif plan.operation == "move_end":
                start_sec = item.start_sec
                end_sec = min(
                    working.video.duration_sec,
                    item.end_sec + (plan.delta_sec or 0.0),
                )
            else:
                start_sec = plan.start_sec if plan.start_sec is not None else item.start_sec
                end_sec = plan.end_sec if plan.end_sec is not None else item.end_sec
            error = validate_highlight_range(start_sec, end_sec, working.video.duration_sec)
            if error:
                raise PlanExecutionError(error)
            updates.append((item, round(start_sec, 3), round(end_sec, 3)))
        for item, start_sec, end_sec in updates:
            item.start_sec = start_sec
            item.end_sec = end_sec
            item.review_status = "revised"
        ranges = "、".join(f"{start:.1f}–{end:.1f}秒" for _, start, end in updates)
        return _changed(working, f"已调整为 {ranges}。", action)

    target = targets[0]
    if plan.operation == "rename":
        target.description = (plan.text or "").strip()[:120]
        target.review_status = "revised"
        return _changed(working, f"已把标题改为「{target.description}」。", action)
    if plan.operation == "update_reason":
        target.reason = (plan.text or "").strip()[:300]
        target.review_status = "revised"
        return _changed(working, f"已更新「{target.description}」的高光说明。", action)
    if plan.operation == "split":
        return _split_highlight(working, target, plan.split_sec or 0.0, action)
    return _merge_highlights(working, targets, plan.text, action)


def _split_highlight(
    result: DetectionResult,
    target: Highlight,
    split_sec: float,
    action: dict[str, Any],
) -> ConversationOutcome:
    left_error = validate_highlight_range(target.start_sec, split_sec, result.video.duration_sec)
    right_error = validate_highlight_range(split_sec, target.end_sec, result.video.duration_sec)
    if left_error or right_error:
        raise PlanExecutionError("拆分点必须在片段内部，并保证拆分后的每段至少 0.5 秒。")

    existing_ids = {item.highlight_id for item in result.highlights}
    right_id = _new_child_id(target.highlight_id, existing_ids)
    left = target.model_copy(
        update={
            "end_sec": round(split_sec, 3),
            "description": _with_suffix(target.description, "（上）"),
            "review_status": "revised",
        }
    )
    right = target.model_copy(
        update={
            "highlight_id": right_id,
            "start_sec": round(split_sec, 3),
            "description": _with_suffix(target.description, "（下）"),
            "review_status": "revised",
        }
    )
    index = next(
        index
        for index, item in enumerate(result.highlights)
        if item.highlight_id == target.highlight_id
    )
    result.highlights[index : index + 1] = [left, right]
    return _changed(
        result,
        f"已在 {split_sec:.1f} 秒把「{target.description}」拆成两段。",
        action,
    )


def _merge_highlights(
    result: DetectionResult,
    targets: list[Highlight],
    title: str | None,
    action: dict[str, Any],
) -> ConversationOutcome:
    timeline = sorted(result.highlights, key=lambda item: (item.start_sec, item.end_sec))
    target_ids = {item.highlight_id for item in targets}
    positions = sorted(
        index for index, item in enumerate(timeline) if item.highlight_id in target_ids
    )
    if positions != list(range(positions[0], positions[-1] + 1)):
        raise PlanExecutionError("只能合并时间轴上相邻的高光片段。")

    ordered = sorted(targets, key=lambda item: (item.start_sec, item.end_sec))
    start_sec = ordered[0].start_sec
    end_sec = max(item.end_sec for item in ordered)
    error = validate_highlight_range(start_sec, end_sec, result.video.duration_sec)
    if error:
        raise PlanExecutionError(error)

    strongest = max(ordered, key=lambda item: item.score)
    anchor = ordered[0]
    default_title = " / ".join(dict.fromkeys(item.description for item in ordered))
    merged_title = (title or default_title).strip()
    merged_reason = "；".join(dict.fromkeys(item.reason for item in ordered))
    merged = anchor.model_copy(
        update={
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
            "score": strongest.score,
            "highlight_type": strongest.highlight_type,
            "description": merged_title[:120],
            "reason": merged_reason[:300],
            "review_status": "revised",
        }
    )
    first_index = min(
        index for index, item in enumerate(result.highlights) if item.highlight_id in target_ids
    )
    result.highlights = [item for item in result.highlights if item.highlight_id not in target_ids]
    result.highlights.insert(first_index, merged)
    return _changed(
        result,
        f"已把 {len(targets)} 段合并为 {start_sec:.1f}–{end_sec:.1f} 秒的"
        f"「{merged.description}」。",
        action,
    )


def _resolve_ids(result: DetectionResult, highlight_ids: list[str]) -> list[Highlight]:
    by_id = {item.highlight_id: item for item in result.highlights}
    return [by_id[item_id] for item_id in highlight_ids if item_id in by_id]


def _changed(
    result: DetectionResult,
    reply: str,
    action: dict[str, Any],
) -> ConversationOutcome:
    return ConversationOutcome(result=result, reply=reply, changed=True, action=action)


def _new_child_id(parent_id: str, existing_ids: set[str]) -> str:
    base = f"{parent_id[:68]}_split"
    candidate = base
    counter = 2
    while candidate in existing_ids:
        candidate = f"{base[:74]}_{counter}"
        counter += 1
    return candidate


def _with_suffix(value: str, suffix: str) -> str:
    return f"{value[: 120 - len(suffix)]}{suffix}"


def _from_local_outcome(
    outcome: EditOutcome,
    message: str,
    selected_highlight_id: str | None,
) -> ConversationOutcome:
    if outcome.undo:
        action: dict[str, Any] = {"kind": "undo"}
    elif outcome.changed:
        action = {
            "kind": "edit",
            "operation": "explicit_command",
            "highlight_ids": [selected_highlight_id] if selected_highlight_id else [],
        }
    elif _is_local_query(message):
        action = {"kind": "query", "operation": "list", "highlight_ids": []}
    else:
        action = {"kind": "clarify", "reply": outcome.reply}
    return ConversationOutcome(
        result=outcome.result,
        reply=outcome.reply,
        changed=outcome.changed,
        undo=outcome.undo,
        action=action,
    )


def _is_explicit_local_command(message: str) -> bool:
    text = message.strip()
    if _is_local_query(text) or re.search(r"撤销|恢复上一步|undo", text, re.IGNORECASE):
        return True
    if re.search(r"删除|移除|去掉|不要这段", text):
        return True
    if re.search(r"(?:标题|名称|描述|说明|理由|原因).*(?:改为|改成|修改|改写)", text):
        return True
    if re.search(r"(?:改为|改成|修改|改写).*(?:标题|名称|描述|说明|理由|原因)", text):
        return True
    return bool(
        re.search(r"开头|入点|结尾|尾部|出点|前后|两端|缩短|延长|收紧", text)
        and re.search(r"\d+(?:\.\d+)?\s*秒", text)
    ) or bool(re.search(r"\d+(?:\.\d+)?\s*秒?\s*(?:到|至|[-—~])\s*\d+(?:\.\d+)?\s*秒", text))


def _is_local_query(message: str) -> bool:
    return bool(re.search(r"列出|有哪些|几段|多少.*高光", message))
