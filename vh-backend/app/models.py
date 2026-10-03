import json
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

JobStatus = Literal["queued", "processing", "completed", "failed"]
ReviewStatus = Literal["pending", "accepted", "rejected", "revised"]


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
    clip_url: str = ""
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


def parse_detection_result(value: str | dict[str, Any]) -> DetectionResult:
    """Load current results while keeping results written by schema 1.0 readable."""
    data = json.loads(value) if isinstance(value, str) else dict(value)
    if data.get("schema_version") == "1.0":
        data = {
            "schema_version": "2.0",
            "job_id": data["job_id"],
            "video": data["video"],
            "completion": "complete",
            "message": "历史分析结果（schema 1.0）",
            "analysis": {
                "scan_coverage": 1.0,
                "pending_event_count": 0,
                "pending_observation_count": 0,
                "pending_proposal_count": 0,
                "pending_review_count": 0,
                "stop_reason": "legacy_schema_1.0",
                "model_calls": 0,
            },
            "highlights": data.get("highlights", []),
        }
    return DetectionResult.model_validate(data)


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    original_name: str
    content_type: str
    size_bytes: int
    language: Literal["zh", "en"]
    created_at: str
    updated_at: str
    revision: int = 0
    attempt: int = Field(default=0, ge=0)
    progress: dict | None = None
    source_url: str | None = None
    error_message: str | None = None
    result: DetectionResult | None = None


class JobDeletionRequest(BaseModel):
    confirmed: Literal[True]
    job_id: str = Field(min_length=1, max_length=52)


class EditMessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    revision: int = Field(ge=0)
    selected_highlight_id: str | None = Field(default=None, max_length=80)


class EditMessageResponse(BaseModel):
    job: JobResponse
    reply: str
    changed: bool


class HighlightRangeEditRequest(BaseModel):
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    revision: int = Field(ge=0)


class DetectionOptions(BaseModel):
    model_config = {"extra": "forbid", "allow_inf_nan": False}
    instruction: str = Field(
        default="挑选有看点、能独立看懂的短剧片段，保留必要铺垫和反应，剪辑简洁流畅。", min_length=1
    )
    max_highlights: int | None = Field(default=12, gt=0)
    min_clip_sec: float = Field(default=3, gt=0)
    max_clip_sec: float = Field(default=24, gt=0)
    total_duration_sec: float | None = Field(default=None, gt=0)
    allow_overlap: bool = True
