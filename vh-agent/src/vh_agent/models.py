from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

ReviewStatus = Literal["pending", "accepted", "rejected", "revised"]


class DetectionTask(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    video_path: Path
    video_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    job_id: str = Field(
        default_factory=lambda: f"job_{uuid4().hex[:16]}",
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$",
    )
    language: str | None = None
    instruction: str = "挑选有看点、能独立看懂的短剧片段，保留必要铺垫和反应，剪辑简洁流畅。"
    subtitle_path: Path | None = None
    max_highlights: int | None = Field(default=12, gt=0)
    min_clip_sec: float = Field(default=3, gt=0)
    max_clip_sec: float = Field(default=24, gt=0)
    total_duration_sec: float | None = Field(default=None, gt=0)
    allow_overlap: bool = True
    resume: bool = False

    @model_validator(mode="after")
    def durations(self):
        if self.min_clip_sec > self.max_clip_sec:
            raise ValueError("min_clip_sec must not exceed max_clip_sec")
        return self


class VideoSummary(BaseModel):
    video_id: str
    title: str
    duration_sec: float


class Highlight(BaseModel):
    highlight_id: str
    start_sec: float
    end_sec: float
    score: float = Field(ge=0, le=1)
    highlight_type: str
    description: str
    reason: str
    clip_url: str
    review_status: ReviewStatus = "pending"


class AnalysisSummary(BaseModel):
    scan_coverage: float = Field(ge=0, le=1)
    pending_event_count: int = Field(ge=0)
    pending_observation_count: int = Field(ge=0)
    pending_proposal_count: int = Field(ge=0)
    pending_review_count: int = Field(ge=0)
    stop_reason: str
    model_calls: int = Field(ge=0)


class DetectionResult(BaseModel):
    schema_version: Literal["2.0"] = "2.0"
    job_id: str
    video: VideoSummary
    completion: Literal["complete", "partial"]
    message: str
    analysis: AnalysisSummary
    highlights: list[Highlight]

    @model_validator(mode="after")
    def publish_only_complete_selection(self):
        if self.completion == "partial" and self.highlights:
            raise ValueError("partial results cannot publish unselected highlights")
        return self


class VideoInfo(BaseModel):
    path: Path
    duration_sec: float
    width: int
    height: int
    fps: float
    has_audio: bool


class TranscriptSegment(BaseModel):
    start_sec: float
    end_sec: float
    text: str
    source: Literal["asr", "ocr"] = "asr"
    confidence: float | None = None


class AudioEvent(BaseModel):
    start_sec: float = 0.0
    end_sec: float = 0.0
    emotion: str = "neutral"
    event: str = "speech"


class SceneSegment(BaseModel):
    start_sec: float
    end_sec: float


class FrameSample(BaseModel):
    timestamp_sec: float
    path: Path
