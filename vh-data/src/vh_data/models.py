import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Span(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    start_sec: float = Field(ge=0, description="片段开始位置，使用原视频秒数")
    end_sec: float = Field(gt=0, description="片段结束位置，使用原视频秒数，晚于开始位置")

    @model_validator(mode="after")
    def ordered(self):
        if self.end_sec <= self.start_sec:
            raise ValueError("结束时间必须晚于开始时间")
        return self


class Label(Span):
    kind: Literal["positive", "negative", "uncertain"] = Field(
        description="positive=值得单独看的精彩瞬间；negative=已看清的普通剧情；uncertain=证据不足"
    )
    reason: str = Field(min_length=1, description="一句说明具体发生了什么、为什么这样判断")


class PageLabels(BaseModel):
    model_config = ConfigDict(extra="forbid")
    story_so_far: str = Field(description="几句话记住题材、人物关系和当前事情，供下一段理解")
    segments: list[Label] = Field(description="实际观看范围中的片段标注，未标记区域保持未知")


class Annotation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    duration_sec: float = Field(gt=0)
    segments: list[Label]

    @model_validator(mode="after")
    def validate_ranges(self):
        for segment in self.segments:
            if segment.end_sec > self.duration_sec:
                raise ValueError("标注超出原视频时间范围")
        if any(
            a.kind != b.kind and overlap(a, b) > 0
            for i, a in enumerate(self.segments)
            for b in self.segments[i + 1 :]
        ):
            raise ValueError("不同判断的时间范围不能冲突")
        return self


class Prediction(Span):
    score: float = Field(ge=0, le=1)


class VideoPredictions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    video_id: str
    segments: list[Prediction]


def overlap(a: Span, b: Span) -> float:
    return max(0, min(a.end_sec, b.end_sec) - max(a.start_sec, b.start_sec))


def iou(a: Span, b: Span) -> float:
    return overlap(a, b) / (max(a.end_sec, b.end_sec) - min(a.start_sec, b.start_sec))
