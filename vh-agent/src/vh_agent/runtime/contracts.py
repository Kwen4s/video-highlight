"""Validated arguments and declarations exposed to the video agent."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .finalization import EventContent, SelectionChoice


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class InspectInput(ToolInput):
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    question: str = Field(min_length=1)
    sampling_fps: float | None = Field(default=None, gt=0)


class EventInput(EventContent):
    id: str | None = Field(
        default=None,
        min_length=1,
        description="新事件省略；修改已有事件时填写状态返回的 event.id。",
    )
    expected_version: int | None = Field(
        default=None,
        ge=1,
        description="新事件省略；修改已有事件时原样填写状态返回的 event.version，不要加一。",
    )

    @model_validator(mode="after")
    def validate_merge_identity(self):
        if self.status == "merged" and self.id is not None and self.merged_into == self.id:
            raise ValueError("merged_into 不能指向当前事件。")
        return self


class ObservationRecord(ToolInput):
    observation_id: str
    findings: list[EventInput] = Field(
        default_factory=list,
        description="本轮形成或补充的事件判断，required_spans 引用已经观看且需要保留在成片中的证据。每条只包含一个可独立采用的核心看点；不同看点分开登记，可以共享证据和重叠。修改已有事件时填写 id 和当前 expected_version。",
    )
    no_event_reason: str | None = None

    @model_validator(mode="after")
    def validate_record(self):
        if bool(self.findings) == bool(self.no_event_reason and self.no_event_reason.strip()):
            raise ValueError("填写 findings；没有发现时仅填写 no_event_reason。")
        return self


class RecordInput(ToolInput):
    observations: list[ObservationRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_observations(self):
        ids = [item.observation_id for item in self.observations]
        if len(ids) != len(set(ids)):
            raise ValueError("每个 observation_id 只能登记一次。")
        explicit_event_ids = [
            event.id
            for item in self.observations
            for event in item.findings
            if event.id is not None
        ]
        if len(explicit_event_ids) != len(set(explicit_event_ids)):
            raise ValueError("同一批次中每个已有事件只能更新一次。")
        return self


class UpdateInput(ToolInput):
    event: EventInput

    @model_validator(mode="after")
    def validate_existing_event(self):
        if self.event.id is None or self.event.expected_version is None:
            raise ValueError("update_event 需要 event.id 和当前 expected_version。")
        return self


class ReadInput(ToolInput):
    collection: Literal[
        "events", "candidates", "observations", "transcript", "queries", "pages"
    ] = "events"
    event_id: str | None = None
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=100)

    @model_validator(mode="after")
    def validate_lookup(self):
        if self.event_id is not None and self.collection != "events":
            raise ValueError("event_id 只用于读取 events")
        if self.event_id is not None and self.model_fields_set & {"offset", "limit"}:
            raise ValueError("按 event_id 读取时不接受 offset 或 limit")
        return self


class SearchInput(ToolInput):
    query: str = ""
    start_sec: float = Field(default=0.0, ge=0)
    end_sec: float | None = Field(default=None, gt=0)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=100)


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
        unexpected = self.model_fields_set - allowed
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
    "search_transcript": SearchInput,
    "propose_highlights": ProposalInput,
    "inspect_interval": InspectInput,
    "record_observations": RecordInput,
    "update_event": UpdateInput,
    "read_state": ReadInput,
    "select_highlights": SelectInput,
}
TOOL_DESCRIPTIONS = {
    "select_highlights": "根据完整候选池的 visible_event，在预算内保留不同的可用看点，一次提交全部 decisions，逐项说明取舍。舍弃背景、重复或不符合用户目标的片段时说明原因，重复项可用 duplicate_of 引用保留项。需要核对时先补看或修订。此动作接受复核并冻结候选，保留项按价值降序排列；允许全部舍弃，无需填满预算。",
    "search_transcript": "按原文 query 子串或时间区间检索字幕/ASR，空 query 浏览全文；返回带时间的定位线索，结合视频判断。",
    "propose_highlights": "读取本地模型提供的位置线索。list 分页浏览；观看对应视频后，用 resolve 提交 proposal_id、覆盖该位置的 observation_ids 和判断 reason。确认有关联事件时填写 event_id，排除这条线索时省略 event_id。",
    "inspect_interval": "按原片秒数读取区间原生音视频；可提高 sampling_fps 检查短暂动作，补看文字、台词及前后文。",
    "record_observations": "一次登记本轮收到的全部视频观察。每个 observation_id 各填写 findings 或 no_event_reason；同一事件跨多个观察时合并成一条 finding，并放在最后相关的观察中。每条事件只围绕一个可独立采用的核心看点，不同看点分开登记。清楚发生的事件设为 supported；rejected 仅用于未发生、被证据否定、明确超出用户任务或已有无法修复的成片缺陷。新事件省略 id 和 expected_version；修改已有事件时原样使用当前 event.version。",
    "update_event": "修订已有事件判断、证据或边界，提交完整 event，并把状态返回的 event.version 原样填入 expected_version；不要自行加一。只有重复描述同一核心看点时才用 merged_into 合并，不因时间相邻而合并。",
    "read_state": "按 collection 分页读取事件、观察、已读字幕、查询历史或页面；event_id 可定位一个事件。返回 total 和 next_offset。最终候选池会直接出现在选择状态中。",
}


def tool_declarations() -> list[dict]:
    return [
        {
            "name": name,
            "description": TOOL_DESCRIPTIONS[name],
            "parametersJsonSchema": schema.model_json_schema(),
        }
        for name, schema in TOOL_INPUTS.items()
    ]
