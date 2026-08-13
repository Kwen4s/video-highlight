from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

HighlightType = Literal[
    "conflict",
    "reversal",
    "reveal",
    "payoff",
    "emotion",
    "action",
    "romance",
    "cliffhanger",
    "other",
]
ReviewStatus = Literal["pending", "accepted", "rejected", "revised"]


def _new_job_id() -> str:
    return f"job_{uuid4().hex[:16]}"


class DetectionTask(BaseModel):
    """Internal backend-to-agent task; never exposed as a frontend contract."""

    video_path: Path
    video_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    job_id: str = Field(default_factory=_new_job_id, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    language: Literal["zh", "en"] | None = None


class VideoSummary(BaseModel):
    video_id: str
    title: str
    duration_sec: float


class Highlight(BaseModel):
    highlight_id: str
    start_sec: float
    end_sec: float
    score: float = Field(ge=0, le=1)
    highlight_type: HighlightType
    description: str
    reason: str
    clip_url: str
    review_status: ReviewStatus = "pending"


class DetectionResult(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    job_id: str
    video: VideoSummary
    highlights: list[Highlight]


class VideoInfo(BaseModel):
    path: Path
    duration_sec: float
    width: int
    height: int
    fps: float
    has_audio: bool
    title: str = ""
    language: str | None = None


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
    confidence: float | None = None


class SceneSegment(BaseModel):
    start_sec: float
    end_sec: float


class FrameSample(BaseModel):
    timestamp_sec: float
    path: Path
    change_score: float = 0.0
    semantic_change_score: float = 0.0


class ChapterContext(BaseModel):
    chapter_id: str
    start_sec: float
    end_sec: float
    transcript: str
    audio_context: str = ""
    frame_samples: list[FrameSample] = Field(default_factory=list)


class CandidateWindow(BaseModel):
    start_sec: float
    end_sec: float
    local_score: float = Field(ge=0, le=1)
    audio_score: float = Field(default=0, ge=0, le=1)
    visual_score: float = Field(default=0, ge=0, le=1)
    semantic_score: float = Field(default=0, ge=0, le=1)
    scene_score: float = Field(default=0, ge=0, le=1)
    cue_score: float = Field(default=0, ge=0, le=1)
    filter_penalty: float = Field(default=0, ge=0, le=1)
    filter_reasons: list[str] = Field(default_factory=list)
    transcript: str = ""
    audio_context: str = ""
    frame_samples: list[FrameSample] = Field(default_factory=list)


class EventCard(BaseModel):
    start_sec: float
    end_sec: float
    actors: list[str] = Field(default_factory=list)
    action: str = ""
    event_type: list[HighlightType] = Field(default_factory=list)
    state_before: str = ""
    new_evidence: str = ""
    state_after: str = ""
    relationship_change: str = ""
    emotion: str = ""
    salience: float = Field(default=0.5, ge=0, le=1)
    uncertainty: float = Field(default=0.5, ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)


class HighlightHypothesis(BaseModel):
    statement: str
    event_type: list[HighlightType] = Field(default_factory=list)
    state_before: str = ""
    trigger: str = ""
    state_after: str = ""
    relationship_effect: str = ""
    expected_evidence: list[str] = Field(default_factory=list)
    verification_gaps: list[str] = Field(default_factory=list)


class StoryMemory(BaseModel):
    characters: list[str] = Field(default_factory=list)
    known_facts: list[str] = Field(default_factory=list)
    relationship_states: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    recent_summaries: list[str] = Field(default_factory=list)


class JudgeDecision(BaseModel):
    hypothesis_supported: bool
    is_highlight: bool
    score: float = Field(ge=0, le=1)
    highlight_type: HighlightType = "other"
    description: str = ""
    reason: str = ""
    confidence: float = Field(default=0.5, ge=0, le=1)
    start_sec: float | None = None
    end_sec: float | None = None
    evidence: list[str] = Field(default_factory=list)
    decisive_evidence_times_sec: list[float] = Field(default_factory=list)
    counter_evidence: list[str] = Field(default_factory=list)


class RankedHighlight(BaseModel):
    """Internal scored highlight before conversion to the public result."""

    highlight_id: str
    start_sec: float
    end_sec: float
    score: float = Field(ge=0, le=1)
    local_score: float = Field(ge=0, le=1)
    judge_score: float = Field(ge=0, le=1)
    highlight_type: HighlightType
    description: str
    reason: str
    transcript: str = ""
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class GlobalRanking(BaseModel):
    ranked_highlight_ids: list[str]
    selected_highlight_ids: list[str]
    rationale: str = ""


class DetectionStats(BaseModel):
    sampled_frames: int
    detected_scenes: int
    transcript_segments: int
    ocr_segments: int
    audio_events: int
    candidate_windows: int
    chapter_map_calls: int
    judge_calls: int
    listwise_calls: int


class PreprocessTrace(BaseModel):
    video: VideoInfo
    scenes: list[SceneSegment]
    transcript: list[TranscriptSegment]
    audio_events: list[AudioEvent]
    frame_samples: list[FrameSample]
    saliency_per_second: list[float]


class DecisionTrace(BaseModel):
    candidate: CandidateWindow
    event: EventCard
    hypothesis: HighlightHypothesis
    decision: JudgeDecision


class ReasoningTrace(BaseModel):
    events: list[EventCard]
    decisions: list[DecisionTrace]
    ranking: GlobalRanking


class DetectionTrace(BaseModel):
    pipeline_version: Literal["0.13.0"] = "0.13.0"
    preprocess: PreprocessTrace
    candidates: list[CandidateWindow]
    reasoning: ReasoningTrace
    stats: DetectionStats
    notes: list[str] = Field(default_factory=list)
