"""The seven actions available to the video editor agent."""

from typing import Literal

from openai import pydantic_function_tool
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .finalization import RequiredSpan


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TimeRange(ToolInput):
    start_sec: float = Field(ge=0, description="负责范围的原片开始秒数。")
    end_sec: float = Field(gt=0, description="负责范围的原片结束秒数，晚于 start_sec。")

    @model_validator(mode="after")
    def ordered(self):
        if self.end_sec <= self.start_sec:
            raise ValueError("end_sec 必须晚于 start_sec")
        return self


class DetectInput(TimeRange):
    offset: int = Field(
        default=0, ge=0, description="结果分页起点；继续读取同一范围时使用 next_offset。"
    )
    limit: int = Field(default=20, ge=1, le=100, description="本页返回数量，不限制实际检测数量。")


class WatchInput(TimeRange):
    start_sec: float = Field(ge=0, description="观看区间的原片开始秒数。")
    end_sec: float = Field(gt=0, description="观看区间的原片结束秒数。")
    question: str = Field(
        min_length=1, description="观看要回答的具体问题；连续阅读时描述人物行动和剧情变化。"
    )
    sampling_fps: float | None = Field(
        default=None, gt=0, le=24, description="可选观察密度；短暂动作看不清时提高。"
    )


class SearchInput(ToolInput):
    mode: Literal["semantic", "exact"] = Field(
        default="semantic",
        description="semantic 按意思检索；exact 查找台词原文，空 query 按时间浏览字幕。",
    )
    query: str = Field(default="", description="想找的事情或台词原文。")
    start_sec: float = Field(default=0, ge=0, description="原片搜索起点。")
    end_sec: float | None = Field(default=None, gt=0, description="原片搜索终点；null 表示片尾。")
    offset: int = Field(default=0, ge=0, description="分页起点。")
    limit: int = Field(default=20, ge=1, le=100, description="本页材料数量。")


class FramesInput(ToolInput):
    times: list[float] = Field(min_length=1, max_length=32, description="要查看的原片秒数。")
    region: list[float] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="可选裁剪区域 [左,上,右,下]，归一化到 0–1；null 表示整帧。",
    )
    with_text: bool = Field(
        default=False, description="需要识别画面文字时设为 true，同时返回文字位置。"
    )

    @model_validator(mode="after")
    def valid_frames(self):
        if any(t < 0 for t in self.times):
            raise ValueError("帧时间必须为非负数")
        if self.region and not (
            0 <= self.region[0] < self.region[2] <= 1 and 0 <= self.region[1] < self.region[3] <= 1
        ):
            raise ValueError("region 必须是有效的归一化矩形")
        return self


class CandidateInput(TimeRange):
    start_sec: float = Field(ge=0, description="给编辑的粗剪起点，原片秒数，包含必要前情。")
    end_sec: float = Field(gt=0, description="粗剪终点，原片秒数，保留核心与直接反应。")
    id: str | None = Field(default=None, description="新候选填 null；修订同一看点时沿用返回的 id。")
    description: str = Field(min_length=1, description="一句话说清人物做了什么、产生什么变化。")
    reason: str = Field(min_length=1, description="这段为什么值得采用，结合用户的提取要求。")
    highlight_type: str = Field(
        min_length=1, description="自然短语概括看点，如身份揭晓、情感摊牌。"
    )
    score: float = Field(
        ge=0, le=1, description="相对于本片其他候选的采用价值，0–1；用于交付排序参考。"
    )
    evidence: list[RequiredSpan] = Field(
        min_length=1,
        description="引用已观看的 observation_id 和原片区间，至少一段 decisive 指明核心；可补充必要铺垫及反应。",
    )


class SaveInput(ToolInput):
    candidates: list[CandidateInput] = Field(
        default_factory=list, description="本次新增或修订的候选；原有其他候选继续保留。"
    )
    story_so_far: str | None = Field(
        default=None,
        min_length=1,
        description="可选更新完整的简短剧情记忆，保留人物关系、当前变化和待查问题。",
    )

    @model_validator(mode="after")
    def useful(self):
        if not self.candidates and self.story_so_far is None:
            raise ValueError("提交候选或剧情记忆")
        ids = [c.id for c in self.candidates if c.id is not None]
        if len(ids) != len(set(ids)):
            raise ValueError("一次只修订同一候选一次")
        return self


class MemoryInput(ToolInput):
    collection: Literal["candidates", "observations", "frames", "transcript", "detections"] = Field(
        default="candidates", description="回查候选、音画观察、原帧及 OCR、已读字幕或本地检测结果。"
    )
    id: str | None = Field(default=None, description="候选或观察的稳定 ID；按 ID 查询时忽略分页。")
    offset: int = Field(default=0, ge=0, description="分页起点。")
    limit: int = Field(default=20, ge=1, le=100, description="本页记录数量。")


class ReviewConfirmation(ToolInput):
    candidate_id: str = Field(description="本次采用的候选 ID。")
    review_id: str = Field(description="本轮复核返回的 review_id，原样填写。")
    reason: str = Field(
        min_length=1,
        description="核对实际看点与候选说明后，简短解释采用判断；复核有疑问时说明为何仍可采用。",
    )


class SubmitInput(ToolInput):
    candidate_ids: list[str] = Field(
        description="按希望交付的顺序填写候选 id，允许空列表；未列出的候选保留在记忆中。"
    )
    confirmations: list[ReviewConfirmation] = Field(
        default_factory=list,
        description="首次提交留空。收到实际复核后，对确认采用的片段填写候选 ID、review_id 和理由；有误则先修订或改选。",
    )


TOOL_INPUTS = {
    "detect_highlights": DetectInput,
    "search_video": SearchInput,
    "watch_video": WatchInput,
    "read_frames": FramesInput,
    "save_candidates": SaveInput,
    "read_memory": MemoryInput,
    "submit_highlights": SubmitInput,
}
TOOL_DESCRIPTIONS = {
    "detect_highlights": "用本地模型寻找指定原片范围的精彩核心。长范围内部按 20 秒负责窗口检测，保留训练所用的前后文。返回原片秒数的 segments、total、next_offset 和无效区间数量；运行细节保存到 trace。空结果表示模型没有发现有效位置。位置用于帮助定位，结合原片观察决定是否成为候选。分页只改变展示，完整结果保留。",
    "search_video": "定位某件事或某句台词出现的位置。semantic 查相关画面与台词，exact 查台词原文或浏览指定时间的字幕。返回带来源的 matches、total 和 next_offset。需要确认连续动作、声音或因果时，用 watch_video 查看对应位置。",
    "watch_video": "观看指定原片区间，回答 question，返回 observation_id、实际观看范围、音画观察及原帧。观察自动保存，成功送达后计入全片观看进度。用于连续理解剧情、核实检测线索或补充前后文；返回的 observation_id 可作为候选依据。",
    "read_frames": "查看指定原片时刻的真实帧，可裁剪 region；with_text=true 同时识别画面文字及位置。返回 frames 和实际时间。适合核对短暂动作、表情、道具或文字；判断连续过程和声音使用 watch_video。",
    "save_candidates": "新增或修订有原片依据的高光候选，可同时更新简短剧情记忆。新候选 id=null，修订沿用 id，提交完整候选；返回已保存候选及稳定 id。范围是给编辑的粗剪，evidence 标出必须保留的核心和上下文。保存不生成视频；重复看点沿用同一 id。",
    "read_memory": "回查本任务已保存的候选、音画观察、原帧及 OCR、已读字幕或检测结果。返回 items、total 和 next_offset，按 id 可读取单条候选或观察。已有依据先回查，需要新证据再观看视频。",
    "submit_highlights": "看完全片后，按采用顺序提交 candidate_ids，首次 confirmations 留空。系统检查预算并制作、独立观看粗剪，返回 awaiting_confirmation 与实际复核。核对 visible_event 是否支持候选说明，判断 issues 是否影响核心看点；确认采用时原样填写 review_id 和理由再次提交。需要修订时用 save_candidates，也可改选或全部舍弃。硬性边界问题返回 needs_revision。原样媒体和观察复用，确认绑定当前候选与成片。",
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
