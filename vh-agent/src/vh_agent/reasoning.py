import base64
import json
import re
from io import BytesIO

from openai import OpenAI
from PIL import Image, ImageDraw

from .config import Settings
from .models import (
    CandidateWindow,
    ChapterContext,
    EventCard,
    FrameSample,
    GlobalRanking,
    HighlightHypothesis,
    JudgeDecision,
    RankedHighlight,
    StoryMemory,
    VideoInfo,
)

MAP_SYSTEM_PROMPT = """你是短剧章节事件映射模型。根据章节内带时间戳的关键帧、ASR、OCR 和声音信息，抽取所有有原始证据的独立原子事件。
只记录人物行动、事实披露、身份/关系/目标变化、冲突升级、情绪转折和因果结果，不做最终高光裁决，不续写剧情。
每个事件必须给出最小必要时间范围并引用输入中的画面、字幕或声音证据；没有具体证据的推测不得输出。
无法确认说话人时 actors 写“未知说话者”并提高 uncertainty；state_before 或 state_after 不确定时留空。
普通对话也可以作为事件输出，但 salience 必须校准。不得只输出章节中最显著的一个事件，也不得把同一事件拆成重复项。
只输出一个包含 events 数组的 JSON 对象，不要 Markdown。"""

JUDGE_SYSTEM_PROMPT = """你是短剧高光叙事裁决模型。根据关键帧、ASR/OCR、声音事件、待验证假设和剧情记忆作最终判断。
高光包括冲突、反转、真相或身份揭露、伏笔回收、强情绪、关键动作、感情推进和集尾悬念。
输入来自高召回预筛选，其中相当一部分应判为非高光；不得因事件已被章节 Map 抽取而默认通过。
把待验证假设视为不可信命题：先寻找原始材料中的支持证据和反证，再判断事件是否真实发生、是否造成叙事状态变化。证据不足时必须否决，不得用假设本身证明假设。
原假设不成立但原始材料支持另一个高光事件时，hypothesis_supported 为 false，is_highlight 仍可为 true，并在 description 中给出证据支持的修正事件；counter_evidence 说明原假设为何不成立。
反转必须存在“此前认知 -> 新证据 -> 新认知”；冲突必须升级风险、权力或关系；动作必须改变后续因果。普通大声对话、信息重复、日常安慰和无后果动作均判 false。不得使用材料之外的剧情常识。
分数必须校准：普通内容 0.1-0.4，有意义但不够独立成段 0.4-0.6，明确高光 0.65-0.85，罕见的全片核心事件才可超过 0.9。is_highlight 仅在 score >= 0.65 且存在具体证据时为 true。
时间边界必须依据输入字幕或关键帧的秒数选择，不得照抄候选窗；只包含最短必要铺垫、核心事件和紧随其后的反应。
只输出一个 JSON 对象，不要 Markdown。"""

LISTWISE_SYSTEM_PROMPT = """你是短剧高光的全局 listwise 排序模型。输入均已通过多模态 Judge 验证并完成时间边界精修。
你只能比较相对叙事价值、独立性、剧情覆盖和重复程度，不能修改事实、类型或时间边界。
召回优先于强行精简，但每个入选片段必须能独立理解，拥有自己的触发证据，并产生不同于其他候选的叙事状态变化。反转需要旧认知、新证据和新认知；揭露需要改变已有认知；冲突需要改变风险、权力、关系或目标。只有称呼、立场重述、过场、普通反应或没有后果的升级不单独入选。
核心证据和状态变化被另一候选完整覆盖时才算重复。时间相邻或因果相连本身既不代表重复，也不代表独立；因果链中的行动、转折和后果仅在各自造成不同状态变化时分别保留。
选择数量由满足判据的独立高光数量决定，不为填满预算保留弱片段，也不因其弱于全片最强高潮而删除。只输出 JSON，不要 Markdown。"""


class SiliconFlowReasoner:
    """Cloud reasoner coordinating chapter mapping, judging, and global ranking."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if not settings.siliconflow_api_key:
            raise ValueError("SILICONFLOW_API_KEY is empty")
        self.client = OpenAI(
            api_key=settings.siliconflow_api_key,
            base_url=settings.siliconflow_base_url,
            timeout=settings.request_timeout_sec,
            max_retries=settings.request_max_retries,
        )

    def map_chapter(self, video: VideoInfo, chapter: ChapterContext) -> list[EventCard]:
        response = self.client.chat.completions.create(
            model=self.settings.siliconflow_map_model,
            messages=[
                {"role": "system", "content": MAP_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _multimodal_content(
                        chapter, _chapter_prompt(video, chapter), max_frames=16, detail="low"
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=5000,
        )
        payload = _parse_json_object(response.choices[0].message.content or "")
        raw_events = payload.get("events")
        if not isinstance(raw_events, list) or not raw_events:
            raise ValueError("Chapter map response must contain a non-empty events list")
        events: list[EventCard] = []
        for raw_event in raw_events:
            if not isinstance(raw_event, dict):
                raise TypeError("Chapter event must be a JSON object")
            event = EventCard.model_validate(_normalize_event_payload(raw_event))
            if (
                event.start_sec < chapter.start_sec - 0.5
                or event.end_sec > chapter.end_sec + 0.5
                or event.start_sec >= event.end_sec
            ):
                raise ValueError("Chapter event timestamps are outside the chapter")
            event = event.model_copy(
                update={
                    "start_sec": max(chapter.start_sec, event.start_sec),
                    "end_sec": min(chapter.end_sec, event.end_sec),
                }
            )
            if not event.evidence:
                raise ValueError("Chapter event must cite original evidence")
            events.append(event)
        return events

    def judge(
        self,
        video: VideoInfo,
        candidate: CandidateWindow,
        hypothesis: HighlightHypothesis,
        story_memory: StoryMemory,
        context_before: str,
        context_core: str,
        context_after: str,
    ) -> JudgeDecision:
        response = self.client.chat.completions.create(
            model=self.settings.siliconflow_judge_model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _multimodal_content(
                        candidate,
                        _judge_prompt(
                            video,
                            candidate,
                            hypothesis,
                            story_memory,
                            context_before,
                            context_core,
                            context_after,
                        ),
                        max_frames=8,
                        detail="high",
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=1200,
        )
        return JudgeDecision.model_validate(
            _normalize_decision_payload(
                _parse_json_object(response.choices[0].message.content or "")
            )
        )

    def rank_highlights(
        self,
        video: VideoInfo,
        highlights: list[RankedHighlight],
        max_selected: int,
    ) -> GlobalRanking:
        candidates = [
            {
                "highlight_id": item.highlight_id,
                "time": [item.start_sec, item.end_sec],
                "judge_score": item.judge_score,
                "confidence": item.confidence,
                "highlight_type": item.highlight_type,
                "description": item.description,
                "reason": item.reason,
                "evidence": item.evidence,
            }
            for item in highlights
        ]
        prompt = (
            f"视频：{video.title}，时长 {video.duration_sec:.2f}s\n"
            f"最多选择 {max_selected} 条。候选："
            + json.dumps(candidates, ensure_ascii=False)
            + "\n输出 JSON：ranked_highlight_ids 必须包含全部候选 ID 的完整排序；"
            "selected_highlight_ids 是最终选择；rationale 是简短全局理由。"
        )
        response = self.client.chat.completions.create(
            model=self.settings.siliconflow_judge_model,
            messages=[
                {"role": "system", "content": LISTWISE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1200,
        )
        ranking = GlobalRanking.model_validate(
            _parse_json_object(response.choices[0].message.content or "")
        )
        expected = [item.highlight_id for item in highlights]
        if len(ranking.ranked_highlight_ids) != len(expected) or set(
            ranking.ranked_highlight_ids
        ) != set(expected):
            raise ValueError("Listwise ranking must be a complete candidate permutation")
        if (
            not ranking.selected_highlight_ids
            or len(ranking.selected_highlight_ids) > max_selected
            or len(set(ranking.selected_highlight_ids)) != len(ranking.selected_highlight_ids)
            or not set(ranking.selected_highlight_ids).issubset(expected)
        ):
            raise ValueError("Listwise selection is invalid")
        return ranking


def _chapter_prompt(video: VideoInfo, chapter: ChapterContext) -> str:
    return f"""视频：{video.title}
章节：{chapter.chapter_id}，{chapter.start_sec:.2f}s - {chapter.end_sec:.2f}s
带时间戳字幕（ASR+OCR）：
{chapter.transcript or "[无字幕]"}
声音理解：{chapter.audio_context or "[无声音事件]"}

逐一输出本章节内所有有证据的独立原子事件：
{{
  "events": [
    {{
      "start_sec": 数字, "end_sec": 数字, "actors": ["人物"],
      "action": "可观察的核心行动",
      "event_type": ["conflict|reversal|reveal|payoff|emotion|action|romance|cliffhanger|other"],
      "state_before": "事件前已知状态", "new_evidence": "新出现的证据或台词",
      "state_after": "事件后状态", "relationship_change": "关系或权力变化",
      "emotion": "主要情绪", "salience": 0到1, "uncertainty": 0到1,
      "evidence": ["F编号或带时间戳的字幕/声音证据"]
    }}
  ]
}}"""


def _judge_prompt(
    video: VideoInfo,
    candidate: CandidateWindow,
    hypothesis: HighlightHypothesis,
    story_memory: StoryMemory,
    context_before: str,
    context_core: str,
    context_after: str,
) -> str:
    hypothesis_payload = hypothesis.model_dump(mode="json")
    return f"""视频：{video.title}
候选窗：{candidate.start_sec:.2f}s - {candidate.end_sec:.2f}s
事件前字幕：{context_before or "[无前文字幕]"}
事件核心字幕：{context_core or "[无候选字幕]"}
事件后字幕：{context_after or "[无后文字幕]"}
声音理解：{candidate.audio_context or "[无声音事件]"}
内容过滤提示：{candidate.filter_reasons or "[无]"}
待验证假设（不是事实）：{json.dumps(hypothesis_payload, ensure_ascii=False)}
候选发生前的有证据剧情记忆（仍以原始材料为准）：{json.dumps(story_memory.model_dump(mode="json"), ensure_ascii=False)}

关键帧左上角的 F 编号和秒数是证据标识。只引用确实支持结论的帧、字幕或声音，不得把事件抽取模型的判断当作证据。

输出：
{{
  "hypothesis_supported": true, "is_highlight": true, "score": 0到1,
  "highlight_type": "conflict|reversal|reveal|payoff|emotion|action|romance|cliffhanger|other",
  "description": "发生了什么", "reason": "为何构成高光；反转需写明认知变化",
  "confidence": 0到1, "start_sec": 数字,
  "end_sec": 数字, "evidence": ["F编号、带时间台词或声音证据"],
  "decisive_evidence_times_sec": [真正使该事件构成高光的证据秒数],
  "counter_evidence": ["不支持假设的证据；没有则为空"]
}}"""


def _multimodal_content(
    candidate: CandidateWindow | ChapterContext, prompt: str, *, max_frames: int, detail: str
) -> list[dict[str, object]]:
    content: list[dict[str, object]] = []
    frame_samples = candidate.frame_samples
    if len(frame_samples) > max_frames:
        indices = [
            round(index * (len(frame_samples) - 1) / (max_frames - 1))
            for index in range(max_frames)
        ]
        frame_samples = [frame_samples[index] for index in indices]
    for index, frame in enumerate(frame_samples, start=1):
        label = f"F{index:02d}  {frame.timestamp_sec:.1f}s"
        content.append({"type": "text", "text": label})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(frame, label), "detail": detail},
            }
        )
    content.append({"type": "text", "text": prompt})
    return content


def _image_data_url(frame: FrameSample, label: str) -> str:
    with Image.open(frame.path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = draw.textbbox((0, 0), label)
    draw.rectangle((0, 0, right - left + 12, bottom - top + 10), fill="black")
    draw.text((6, 5), label, fill="white")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _parse_json_object(raw: str) -> dict[str, object]:
    fence = chr(96) * 3
    cleaned = raw.strip().removeprefix(fence + "json").removeprefix(fence).removesuffix(fence)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match is None:
            raise ValueError(f"Model returned invalid JSON: {raw[:200]}")
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise TypeError("Model response is not a JSON object")
    return payload


def parse_judge_decision(raw: str) -> JudgeDecision:
    return JudgeDecision.model_validate(_normalize_decision_payload(_parse_json_object(raw)))


_EVENT_TYPES = {
    "conflict",
    "reversal",
    "reveal",
    "payoff",
    "emotion",
    "action",
    "romance",
    "cliffhanger",
    "other",
}


def _normalize_event_payload(payload: dict[str, object]) -> dict[str, object]:
    normalized = dict(payload)
    for key in (
        "action",
        "state_before",
        "new_evidence",
        "state_after",
        "relationship_change",
        "emotion",
    ):
        normalized[key] = _as_text(normalized.get(key, ""))
    normalized["actors"] = _as_string_list(normalized.get("actors", []))
    normalized["evidence"] = _as_string_list(normalized.get("evidence", []))
    event_types = _as_string_list(normalized.get("event_type", []))
    normalized["event_type"] = [item for item in event_types if item in _EVENT_TYPES] or ["other"]
    score_keys = ("salience", "uncertainty")
    missing = [key for key in score_keys if key not in normalized]
    if missing:
        raise ValueError(f"Event response is missing required fields: {missing}")
    for key in score_keys:
        normalized[key] = _clamp_score(normalized[key])
    return normalized


def _normalize_decision_payload(payload: dict[str, object]) -> dict[str, object]:
    normalized = dict(payload)
    normalized["description"] = _as_text(normalized.get("description", ""))
    normalized["reason"] = _as_text(normalized.get("reason", ""))
    normalized["evidence"] = _as_string_list(normalized.get("evidence", []))
    times = normalized.get("decisive_evidence_times_sec", [])
    if not isinstance(times, list):
        raise TypeError("decisive_evidence_times_sec must be a list")
    normalized["decisive_evidence_times_sec"] = [float(value) for value in times]
    normalized["counter_evidence"] = _as_string_list(normalized.get("counter_evidence", []))
    kinds = _as_string_list(normalized.get("highlight_type", "other"))
    normalized["highlight_type"] = next((item for item in kinds if item in _EVENT_TYPES), "other")
    required = ("hypothesis_supported", "is_highlight", "score", "confidence")
    missing = [key for key in required if key not in normalized]
    if missing:
        raise ValueError(f"Judge response is missing required fields: {missing}")
    normalized["score"] = _clamp_score(normalized["score"])
    normalized["confidence"] = _clamp_score(normalized["confidence"])
    for key in ("hypothesis_supported", "is_highlight"):
        value = normalized[key]
        if isinstance(value, str):
            normalized[key] = value.strip().lower() in {"true", "1", "yes", "是"}
        elif not isinstance(value, bool):
            normalized[key] = bool(value)
    return normalized


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "；".join(_as_text(item) for item in value if _as_text(item))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _as_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [text for item in value if (text := _as_text(item))]
    text = _as_text(value)
    return [text] if text else []


def _clamp_score(value: object) -> float:
    return max(0.0, min(1.0, float(value)))


def build_highlight_hypothesis(event: EventCard) -> HighlightHypothesis:
    gaps: list[str] = []
    for field_name, value in (
        ("state_before", event.state_before),
        ("trigger", event.new_evidence),
        ("state_after", event.state_after or event.relationship_change),
        ("evidence", event.evidence),
    ):
        if not value:
            gaps.append(field_name)
    statement = event.action or event.new_evidence or "候选窗可能包含叙事状态变化"
    return HighlightHypothesis(
        statement=statement,
        event_type=event.event_type,
        state_before=event.state_before,
        trigger=event.new_evidence,
        state_after=event.state_after,
        relationship_effect=event.relationship_change,
        expected_evidence=event.evidence,
        verification_gaps=gaps,
    )


class StoryMemoryStore:
    """Compact narrative state available before each candidate event."""

    def __init__(self) -> None:
        self.memory = StoryMemory()

    def snapshot(self) -> StoryMemory:
        return self.memory.model_copy(deep=True)

    def update(self, event: EventCard) -> None:
        if not event.evidence:
            return
        stamp = f"[{event.start_sec:.2f}-{event.end_sec:.2f}s]"
        source = "；".join(event.evidence[:2])
        citation = f"（证据：{source}）"

        observed_characters = [actor for actor in event.actors if actor and actor != "未知说话者"]
        self.memory.characters = _unique(self.memory.characters + observed_characters)[-30:]

        fact = event.new_evidence or event.action
        if fact:
            self.memory.known_facts = _unique(
                self.memory.known_facts + [f"{stamp} {fact}{citation}"]
            )[-30:]

        if event.relationship_change:
            self.memory.relationship_states = _unique(
                self.memory.relationship_states + [f"{stamp} {event.relationship_change}{citation}"]
            )[-20:]

        if "cliffhanger" in event.event_type:
            question = event.action or event.new_evidence
            if question:
                self.memory.open_questions = _unique(
                    self.memory.open_questions + [f"{stamp} {question}{citation}"]
                )[-8:]

        summary = event.action or event.new_evidence
        if summary:
            self.memory.recent_summaries = _unique(
                self.memory.recent_summaries + [f"{stamp} {summary}"]
            )[-5:]


def build_story_memories(events: list[EventCard]) -> tuple[dict[int, StoryMemory], StoryMemory]:
    store = StoryMemoryStore()
    before: dict[int, StoryMemory] = {}
    for event in sorted(events, key=lambda item: item.start_sec):
        before[id(event)] = store.snapshot()
        store.update(event)
    return before, store.snapshot()


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))
