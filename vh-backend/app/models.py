from typing import Literal

from pydantic import BaseModel, Field

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
    clip_url: str
    review_status: ReviewStatus = "pending"


class DetectionResult(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    job_id: str
    video: VideoSummary
    highlights: list[Highlight]


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    original_name: str
    content_type: str
    size_bytes: int
    language: Literal["zh", "en"]
    created_at: str
    updated_at: str
    source_url: str
    error_message: str | None = None
    result: DetectionResult | None = None


class ReviewRequest(BaseModel):
    status: ReviewStatus
