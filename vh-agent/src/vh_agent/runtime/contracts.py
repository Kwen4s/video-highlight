"""Validated arguments and declarations exposed to the video agent."""

from typing import Literal

from openai import pydantic_function_tool
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .finalization import EventContent, SelectionChoice


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class InspectInput(ToolInput):
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    question: str = Field(min_length=1)
    sampling_fps: float | None = Field(default=None, gt=0)
    event_id: str | None = None


class EventInput(EventContent):
    id: str = Field(min_length=1, description="本看点的稳定 ID；跨观察补充或修订时沿用它。")
    expected_version: int | None = Field(
        default=None, ge=1, description="新事件填 null；修改时原样填写当前 event.version。"
    )

    @model_validator(mode="after")
    def validate_merge_identity(self):
        if self.status == "merged" and self.merged_into == self.id:
            raise ValueError("merged_into 不能指向当前事件。")
        return self


class ObservationRecord(ToolInput):
    observation_id: str
    findings: list[EventInput] = Field(
        default_factory=list,
        description="本观察形成或补充的看点，引用已观看的原片依据；同一看点跨观察只提交一次，放在最后相关的观察中。",
    )
    no_event_reason: str | None = Field(
        default=None, description="本观察没有形成或补充候选时，简短说明原因。"
    )

    @model_validator(mode="after")
    def validate_record(self):
        if bool(self.findings) == bool(self.no_event_reason and self.no_event_reason.strip()):
            raise ValueError("填写 findings；没有发现时仅填写 no_event_reason。")
        return self


class RecordInput(ToolInput):
    observations: list[ObservationRecord] = Field(default_factory=list)
    story_so_far: str | None = Field(
        default=None,
        min_length=1,
        description="简短更新剧情记忆，保留人物关系、当前变化与待确认处；无需更新时填 null。",
    )

    @model_validator(mode="after")
    def validate_observations(self):
        if not (self.observations or self.story_so_far):
            raise ValueError("请提交观察或简短剧情记忆。")
        ids = [item.observation_id for item in self.observations]
        if len(ids) != len(set(ids)):
            raise ValueError("每个 observation_id 只能登记一次。")
        explicit_event_ids = [event.id for item in self.observations for event in item.findings]
        if len(explicit_event_ids) != len(set(explicit_event_ids)):
            raise ValueError("同一批次中每个已有事件只能更新一次。")
        return self


class UpdateInput(ToolInput):
    event: EventInput

    @model_validator(mode="after")
    def validate_existing_event(self):
        if self.event.expected_version is None:
            raise ValueError("update_event 需要当前 expected_version。")
        return self


class ReadInput(ToolInput):
    collection: Literal[
        "events",
        "candidates",
        "observations",
        "transcript",
        "queries",
        "pages",
        "readings",
    ] = "events"
    event_id: str | None = None
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=100)

    @model_validator(mode="after")
    def validate_lookup(self):
        if self.event_id is not None and self.collection != "events":
            raise ValueError("event_id 只用于读取 events")
        if self.event_id is not None and (self.offset != 0 or self.limit != 20):
            raise ValueError("按 event_id 读取时不接受 offset 或 limit")
        return self


class SearchInput(ToolInput):
    mode: Literal["semantic", "exact"] = "semantic"
    query: str = ""
    start_sec: float = Field(default=0.0, ge=0)
    end_sec: float | None = Field(default=None, gt=0)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=100)


class FramesInput(ToolInput):
    times: list[float] = Field(min_length=1, max_length=32)
    region: list[float] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="可选归一化区域 [左,上,右,下]；默认整帧。",
    )

    @model_validator(mode="after")
    def validate_region(self):
        if self.region and not (
            0 <= self.region[0] < self.region[2] <= 1 and 0 <= self.region[1] < self.region[3] <= 1
        ):
            raise ValueError("region 必须是非空的归一化矩形。")
        return self


class ProposalInput(ToolInput):
    action: Literal["list", "resolve"] = "list"
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=100)
    proposal_id: str | None = None
    event_id: str | None = None
    observation_ids: list[str] = Field(default_factory=list)
    reason: str | None = None

    @model_validator(mode="after")
    def validate_action(self):
        allowed = {
            "list": {"action", "offset", "limit"},
            "resolve": {
                "action",
                "proposal_id",
                "event_id",
                "observation_ids",
                "reason",
            },
        }[self.action]
        unexpected = {
            key
            for key in self.model_fields_set - allowed
            if getattr(self, key) not in (None, [], type(self).model_fields[key].default)
        }
        if unexpected:
            raise ValueError(f"{self.action} 不接受字段 {sorted(unexpected)}")
        if self.action == "resolve" and (
            not self.proposal_id
            or not self.observation_ids
            or not self.reason
            or not self.reason.strip()
        ):
            raise ValueError("resolve 需要 proposal_id、observation_ids 和 reason")
        return self


class SelectInput(ToolInput):
    decisions: list[SelectionChoice]


TOOL_INPUTS = {
    "search_video": SearchInput,
    "propose_highlights": ProposalInput,
    "inspect_video": InspectInput,
    "read_frames": FramesInput,
    "read_text": FramesInput,
    "record_observations": RecordInput,
    "update_event": UpdateInput,
    "read_state": ReadInput,
    "select_highlights": SelectInput,
}
TOOL_DESCRIPTIONS = {
    "select_highlights": "从完整候选池提交取舍，每个候选一条 decision，保留项按采用价值降序排列。初选后系统只制作并复核拟采用片段；阅读复核结果、处理必要修订后，再提交最终取舍确认交付。重复项用 duplicate_of 引用保留项，也可以全部舍弃。",
    "search_video": "按问题定位原片材料：semantic 查询画面与台词的相关片段；exact 按原文子串或时间查询字幕，空 query 浏览全文。返回分页和来源；相关性不是高光价值，观看原片确认。",
    "propose_highlights": "读取本地模型提供的位置线索。list 分页浏览；观看对应视频后，用 resolve 提交 proposal_id、覆盖该位置的 observation_ids 和判断 reason。确认有关联事件时填写 event_id，排除这条线索时省略 event_id。",
    "inspect_video": "需要连续动作、声音或前后文时，观看指定原片区间并回答 question。返回原始帧和带时间的音画观察；sampling_fps 可增加视频观察密度。",
    "read_frames": "直接查看指定原片时刻的实际帧，可裁剪 region，检查短暂动作、道具或表情。返回真实帧时间，不调用视频观察模型。",
    "read_text": "识别指定原帧或 region 的文字，保留原图、文字位置、置信度和真实帧时间。结合视频理解含义。",
    "record_observations": "保存本轮全部已看观察，并按需更新剧情记忆。每个 observation_id 填 findings 或 no_event_reason；事件参数定义看点、状态和原片依据。",
    "update_event": "用原 event.id 和当前 event.version 修订已有候选，提交完整 event；系统重新制作并复核该片段。只合并重复的同一看点，目标必须保留核心证据；也可在最终选择时舍弃重复项。",
    "read_state": "按 collection 分页读取事件、观察、已读字幕、查询历史或页面；event_id 可定位一个事件。返回 total 和 next_offset。最终候选池会直接出现在选择状态中。",
}


def tool_declarations() -> list[dict]:
    return [
        {
            "type": "function",
            **pydantic_function_tool(schema, name=name, description=TOOL_DESCRIPTIONS[name])[
                "function"
            ],
        }
        for name, schema in TOOL_INPUTS.items()
    ]
