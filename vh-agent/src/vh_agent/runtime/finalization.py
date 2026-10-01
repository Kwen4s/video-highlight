"""Deterministic event-to-clip contracts for the ReAct detector.

The main agent owns event identity and the meaning of required evidence. This
module owns clip boundaries, review acceptance, and selection over the full pool.
"""

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class RequiredSpan(_Contract):
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    evidence_id: str = Field(
        min_length=1, description="实际看过的视频所返回的 observation_id；时间范围使用原片秒数。"
    )
    role: Literal["setup", "decisive", "reaction"] = Field(
        description="该段对当前单一看点的作用：setup 必要铺垫，decisive 决定性变化，reaction 直接反应。"
    )

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.end_sec <= self.start_sec:
            raise ValueError("Required evidence must have a positive duration")
        return self


class EventContent(_Contract):
    description: str = Field(
        min_length=1,
        description="只描述一个可独立取舍的核心看点，说明人物、行动和决定性变化；另一个可单独采用的看点应新建事件。",
    )
    reason: str = Field(
        min_length=1, description="结合用户目标说明看点、待查问题，或排除、合并的理由。"
    )
    required_spans: list[RequiredSpan] = Field(default_factory=list)
    status: Literal["pending", "supported", "rejected", "merged"] = Field(
        default="pending",
        description="pending 待查；supported 已有视频依据、进入成片验证；rejected 仅表示事件证据无效、明确超出用户任务或无法形成合格片段；merged 已并入 merged_into 指定的事件。观看价值留到最终选择。",
    )
    rejection_category: (
        Literal["not_observed", "contradicted", "outside_request", "clip_infeasible"] | None
    ) = Field(
        default=None,
        description="rejected 时必填：未在视频中发生、被后续证据否定、明确超出用户任务，或已生成片段但因时长/复核缺陷无法修复。低价值、开放悬念和剧情未结束不属于拒绝原因。",
    )
    merged_into: str | None = None
    proposed_start_sec: float | None = Field(default=None, ge=0)
    proposed_end_sec: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if self.status == "supported" and not any(
            span.role == "decisive" for span in self.required_spans
        ):
            raise ValueError("Supported events require decisive evidence")
        if self.status == "merged":
            if not self.merged_into:
                raise ValueError("Merged events require a target event")
        elif self.merged_into is not None:
            raise ValueError("Only merged events may have a merge target")
        if self.status == "rejected" and self.rejection_category is None:
            raise ValueError("Rejected events require a rejection category")
        if self.status != "rejected" and self.rejection_category is not None:
            raise ValueError("Only rejected events may have a rejection category")
        if (
            self.proposed_start_sec is not None
            and self.proposed_end_sec is not None
            and self.proposed_end_sec <= self.proposed_start_sec
        ):
            raise ValueError("Proposed clip must have a positive duration")
        return self


class EventRecord(EventContent):
    id: str = Field(default_factory=lambda: f"evt_{uuid4().hex}", min_length=1)
    version: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_merge_identity(self) -> Self:
        if self.status == "merged" and self.merged_into == self.id:
            raise ValueError("Merged events require a different target event")
        return self


class ReviewIssue(_Contract):
    category: Literal["missing_context", "cutoff", "mixed_focus", "technical"] = Field(
        description="缺少必要上下文、台词/动作被截断、捆绑多个可独立采用的看点，或音画技术问题。"
    )
    description: str = Field(min_length=1)
    at_sec: float | None = Field(default=None, ge=0, description="可定位时填写片段内秒数。")


class ReviewResult(_Contract):
    """Observation of the proposed clip in an independent model context."""

    visible_event: str = Field(
        min_length=1,
        description="描述本段的一个核心看点及实际变化，关键台词用原话；以看到、听到的内容为准。",
    )
    highlight_type: str = Field(
        min_length=1, description="根据片段实际呈现的看点，用简短词语概括类型。"
    )
    blocking_issues: list[ReviewIssue] = Field(
        default_factory=list,
        description="只列阻碍采用的具体缺陷，包括多个独立看点被捆绑；剧情继续、悬念未揭晓、看点较弱不属于缺陷。",
    )


class ClipPlan(_Contract):
    id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    event_version: int = Field(ge=1)
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    required_spans: list[RequiredSpan] = Field(min_length=1)
    status: Literal["draft", "ready", "infeasible_duration"] = "draft"
    review: ReviewResult | None = None
    media_path: Path | None = None
    issues: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_clip(self) -> Self:
        if self.end_sec <= self.start_sec:
            raise ValueError("Clip must have a positive duration")
        if not _contains_required_spans(self):
            raise ValueError("Clip must contain every complete required evidence span")
        if self.status == "ready" and (
            self.review is None or not _review_passed(self.review) or self.issues
        ):
            raise ValueError("Ready clips require an accepted review with no open issues")
        return self


class SelectionChoice(_Contract):
    event_id: str = Field(min_length=1)
    selected: bool
    score: float = Field(
        ge=0, le=1, description="结合用户目标和完整候选池评价采用价值，0 最低、1 最高。"
    )
    reason: str = Field(
        min_length=1,
        description="简短说明该段独立看点及采用价值，或舍弃原因；说明背景或重复内容与其他候选的关系。",
    )
    duplicate_of: str | None = Field(
        default=None, description="内容重复时，引用本次保留的 event_id。"
    )

    @model_validator(mode="after")
    def validate_choice(self) -> Self:
        if not self.reason.strip():
            raise ValueError("Selection requires a nonblank reason")
        if self.duplicate_of and (self.selected or self.duplicate_of == self.event_id):
            raise ValueError("Only an omitted event may reference another selected event")
        return self


class SelectionResult(_Contract):
    selected: list[ClipPlan]
    decisions: list[SelectionChoice]


def draft_plan(
    event: EventRecord,
    video_duration: float,
    max_clip_sec: float = 24.0,
    min_clip_sec: float = 3.0,
) -> ClipPlan:
    """Make one boundary proposal without dropping any required evidence.

    Proposed context is retained. If it makes the clip too long, the agent must
    reconsider that context explicitly; this function does not silently trim it.
    """
    if not math.isfinite(video_duration) or video_duration <= 0:
        raise ValueError("Video duration must be finite and positive")
    if not all(math.isfinite(value) and value > 0 for value in (min_clip_sec, max_clip_sec)):
        raise ValueError("Clip duration limits must be finite and positive")
    if min_clip_sec > max_clip_sec:
        raise ValueError("Minimum clip duration cannot exceed maximum clip duration")
    if event.status != "supported":
        raise ValueError("Only supported events can become clips")
    if not event.required_spans or not any(
        span.role == "decisive" for span in event.required_spans
    ):
        raise ValueError("Supported events require decisive evidence")
    if any(span.end_sec > video_duration for span in event.required_spans):
        raise ValueError("Required evidence is outside the video")
    for boundary in (event.proposed_start_sec, event.proposed_end_sec):
        if boundary is not None and not 0 <= boundary <= video_duration:
            raise ValueError("Proposed boundary is outside the video")

    core_start = min(span.start_sec for span in event.required_spans)
    core_end = max(span.end_sec for span in event.required_spans)
    start = min(
        core_start,
        event.proposed_start_sec if event.proposed_start_sec is not None else core_start,
    )
    end = max(
        core_end,
        event.proposed_end_sec if event.proposed_end_sec is not None else core_end,
    )
    if end - start < min_clip_sec:
        padding = (min_clip_sec - (end - start)) / 2.0
        start = max(0.0, start - padding)
        end = min(video_duration, max(end, start + min_clip_sec))
        start = max(0.0, min(start, end - min_clip_sec))

    issues: list[str] = []
    if end - start > max_clip_sec:
        issues.append("必要证据和建议上下文的总时长超过单片上限；请重新判断所需范围并更新事件。")
    if end - start < min_clip_sec:
        issues.append("原视频长度不足以满足单片最短时长。")
    return ClipPlan(
        id=f"clip_{event.id}",
        event_id=event.id,
        event_version=event.version,
        start_sec=start,
        end_sec=end,
        required_spans=[span.model_copy(deep=True) for span in event.required_spans],
        status="infeasible_duration" if issues else "draft",
        issues=issues,
    )


def attach_media(
    plan: ClipPlan,
    *,
    start_sec: float,
    end_sec: float,
    media_path: Path,
    video_duration: float,
    max_clip_sec: float = 24.0,
) -> ClipPlan:
    """Accept actual frame-aligned boundaries and invalidate any previous review.

    Media preparation may expand to source frame boundaries, but may not remove
    requested context or required evidence. Review must observe this exact asset.
    """
    if not math.isfinite(video_duration) or video_duration <= 0:
        raise ValueError("Video duration must be finite and positive")
    if not math.isfinite(max_clip_sec) or max_clip_sec <= 0:
        raise ValueError("Maximum clip duration must be finite and positive")
    if not (
        math.isfinite(start_sec)
        and math.isfinite(end_sec)
        and 0 <= start_sec < end_sec <= video_duration
    ):
        raise ValueError("Actual media boundaries must be inside the video")
    if start_sec > plan.start_sec or end_sec < plan.end_sec:
        raise ValueError("Frame alignment must not discard proposed clip content")
    issues = list(plan.issues) if plan.status == "infeasible_duration" else []
    if end_sec - start_sec > max_clip_sec:
        issues.append("对齐视频帧后的片段超过单片时长上限；请调整建议边界后重新制作。")
    return ClipPlan.model_validate(
        {
            **plan.model_dump(),
            "start_sec": start_sec,
            "end_sec": end_sec,
            "media_path": media_path,
            "review": None,
            "status": "infeasible_duration" if issues else "draft",
            "issues": list(dict.fromkeys(issues)),
        }
    )


def complete_review(plan: ClipPlan, event: EventRecord, review: ReviewResult) -> ClipPlan:
    """Freeze a reviewed clip as part of the main agent's atomic selection.

    The caller bases its editorial decisions on the reviewed ``visible_event``.
    Reviews with concrete blocking defects remain drafts so the main agent can
    obtain more evidence, revise the event, or explicitly reject it.
    """
    mismatch = _event_mismatch(plan, event)
    if mismatch is not None:
        raise ValueError(f"Cannot review clip: {mismatch}")
    if plan.status == "infeasible_duration":
        raise ValueError("An infeasible clip must be redrafted before review")
    if plan.status == "ready":
        raise ValueError("A frozen clip cannot be reviewed again without a new draft")
    duration = plan.end_sec - plan.start_sec
    if any(
        issue.at_sec is not None and issue.at_sec > duration for issue in review.blocking_issues
    ):
        raise ValueError("Review issue timestamp is outside the clip")
    issues = [issue.description for issue in review.blocking_issues]
    return ClipPlan.model_validate(
        {
            **plan.model_dump(),
            "review": review.model_dump(),
            "status": "ready" if _review_passed(review) else "draft",
            "issues": list(dict.fromkeys(issues)),
        }
    )


def select_clips(
    plans: Sequence[ClipPlan],
    events: Sequence[EventRecord] | Mapping[str, EventRecord],
    max_highlights: int | None,
    *,
    decisions: Sequence[SelectionChoice],
    total_duration: float | None = None,
    allow_overlap: bool = True,
) -> SelectionResult:
    """Validate an explicit editorial selection over the entire ready pool."""
    if max_highlights is not None and (
        isinstance(max_highlights, bool)
        or not isinstance(max_highlights, int)
        or max_highlights <= 0
    ):
        raise ValueError("Highlight count budget must be a positive integer")
    if total_duration is not None and (not math.isfinite(total_duration) or total_duration <= 0):
        raise ValueError("Total duration budget must be finite and positive")
    event_values = list(events.values()) if isinstance(events, Mapping) else list(events)
    event_by_id = {e.id: e for e in event_values}
    if len(event_by_id) != len(event_values):
        raise ValueError("The event pool must contain one current version per event ID")
    pool = {p.event_id: p for p in plans if p.status == "ready"}
    if len({p.id for p in plans}) != len(plans):
        raise ValueError("Duplicate plan IDs")
    if len(decisions) != len(pool) or {d.event_id for d in decisions} != set(pool):
        raise ValueError("Decisions must contain every candidate event_id exactly once")
    for plan in pool.values():
        event = event_by_id.get(plan.event_id)
        mismatch = "unknown_event" if event is None else _event_mismatch(plan, event)
        if mismatch:
            raise ValueError(mismatch)
        if plan.review is None or not _review_passed(plan.review) or plan.issues:
            raise ValueError("review_not_passed")
        if not _contains_required_spans(plan):
            raise ValueError("invalid_boundaries")
    selected = [pool[d.event_id] for d in decisions if d.selected]
    selected_ids = {p.event_id for p in selected}
    if len({p.event_id for p in selected}) != len(selected):
        raise ValueError("An event may only be selected once")
    for decision in decisions:
        if decision.duplicate_of is not None and decision.duplicate_of not in selected_ids:
            raise ValueError("duplicate_of must reference a selected plan")
    if max_highlights is not None and len(selected) > max_highlights:
        raise ValueError(f"Selection exceeds count budget by {len(selected) - max_highlights}")
    duration = sum(p.end_sec - p.start_sec for p in selected)
    if total_duration is not None and duration > total_duration + 1e-9:
        raise ValueError(
            f"Selection exceeds total duration budget by {duration - total_duration:.2f}s"
        )
    if not allow_overlap:
        chronological = sorted(selected, key=lambda p: p.start_sec)
        if any(a.end_sec > b.start_sec for a, b in pairwise(chronological)):
            raise ValueError("Selected clips violate overlap constraint")
    return SelectionResult(
        selected=selected,
        decisions=list(decisions),
    )


def _review_passed(review: ReviewResult) -> bool:
    return not review.blocking_issues


def _contains_required_spans(plan: ClipPlan) -> bool:
    return bool(plan.required_spans) and all(
        plan.start_sec <= span.start_sec < span.end_sec <= plan.end_sec
        for span in plan.required_spans
    )


def _event_mismatch(plan: ClipPlan, event: EventRecord) -> str | None:
    if plan.event_id != event.id:
        return "wrong_event"
    if event.status != "supported":
        return "event_not_supported"
    if plan.event_version != event.version:
        return "stale_event_version"
    if plan.required_spans != event.required_spans:
        return "stale_required_spans"
    return None
