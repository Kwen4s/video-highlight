from typing import Literal

from pydantic import BaseModel, Field

JobStatus = Literal["queued", "processing", "completed", "failed"]
JobStage = Literal[
    "orchestration",
    "preprocessing",
    "perception",
    "fusion",
    "reasoning",
    "delivery",
]
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
    review_status: ReviewStatus = "pending"


class AgentHighlight(Highlight):
    clip_url: str = ""


class DetectionResult(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    job_id: str
    video: VideoSummary
    highlights: list[Highlight]


class AgentDetectionResult(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    job_id: str
    video: VideoSummary
    highlights: list[AgentHighlight]

    def to_public_result(self) -> DetectionResult:
        return DetectionResult(
            schema_version=self.schema_version,
            job_id=self.job_id,
            video=self.video,
            highlights=[
                Highlight.model_validate(item.model_dump(exclude={"clip_url"}))
                for item in self.highlights
            ],
        )


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    current_stage: JobStage
    original_name: str
    content_type: str
    size_bytes: int
    language: Literal["zh", "en"]
    created_at: str
    updated_at: str
    revision: int = 0
    attempt: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=3, ge=1, le=3)
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


class DemoSessionRequest(BaseModel):
    original_name: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(ge=0)
    language: Literal["zh", "en"] = "zh"
    result: DetectionResult
