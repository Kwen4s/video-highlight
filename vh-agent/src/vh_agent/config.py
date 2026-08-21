from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Playable highlight and scene-construction limits. These are product constraints,
# not per-video knobs: a SceneCard is one dramatic beat, a highlight is a clip.
MAX_HIGHLIGHT_SEC = 24.0
MIN_SCENE_SEC = 8.0
MAX_SCENE_SEC = 40.0
SETUP_LEAD_SEC = 8.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    reasoning_provider: Literal["gemini", "siliconflow"] = Field(
        default="gemini", validation_alias="VH_REASONING_PROVIDER"
    )
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    gemini_base_url: str = Field(
        default="https://yetoken.vip/v1", validation_alias="GEMINI_BASE_URL"
    )
    gemini_map_model: str = Field(
        default="gemini-3.1-flash-lite", validation_alias="GEMINI_MAP_MODEL"
    )
    gemini_judge_model: str = Field(
        default="gemini-3.7-flash", validation_alias="GEMINI_JUDGE_MODEL"
    )
    siliconflow_api_key: str = Field(default="", validation_alias="SILICONFLOW_API_KEY")
    siliconflow_base_url: str = Field(
        default="https://api.siliconflow.cn/v1",
        validation_alias="SILICONFLOW_BASE_URL",
    )
    siliconflow_map_model: str = Field(
        default="Qwen/Qwen3-VL-8B-Instruct",
        validation_alias="SILICONFLOW_MAP_MODEL",
    )
    siliconflow_judge_model: str = Field(
        default="Qwen/Qwen3-VL-32B-Instruct",
        validation_alias="SILICONFLOW_JUDGE_MODEL",
    )
    model_cache_dir: Path = Field(
        default=Path("/data1/video-highlight-models"),
        validation_alias="VH_MODEL_CACHE_DIR",
    )
    asr_model: Path = Field(
        default=Path(
            "/data1/video-highlight-models/faster-whisper/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478"
        ),
        validation_alias="VH_ASR_MODEL",
    )
    asr_device: str = Field(default="cuda:1", validation_alias="VH_ASR_DEVICE")
    asr_compute_type: str = Field(default="float16", validation_alias="VH_ASR_COMPUTE_TYPE")
    sensevoice_model: Path = Field(
        default=Path(
            "/data1/video-highlight-models/modelscope/models/iic--SenseVoiceSmall/snapshots/master"
        ),
        validation_alias="VH_SENSEVOICE_MODEL",
    )
    sensevoice_vad_model: Path = Field(
        default=Path(
            "/data1/video-highlight-models/modelscope/models/iic--speech_fsmn_vad_zh-cn-16k-common-pytorch/snapshots/master"
        ),
        validation_alias="VH_SENSEVOICE_VAD_MODEL",
    )
    sensevoice_device: str = Field(default="cuda:0", validation_alias="VH_SENSEVOICE_DEVICE")
    ocr_device: str = Field(default="cpu", validation_alias="VH_OCR_DEVICE")
    ocr_version: str = Field(default="PP-OCRv6", validation_alias="VH_OCR_VERSION")
    embedding_model: str = Field(
        default="/data1/modelscope_models/Qwen3-VL-Embedding-2B",
        validation_alias="VH_EMBEDDING_MODEL",
    )
    embedding_device: str = Field(default="cuda:0", validation_alias="VH_EMBEDDING_DEVICE")

    job_output_dir: Path = Field(default=Path("outputs/jobs"), validation_alias="VH_JOB_OUTPUT_DIR")
    media_cache_dir: Path = Field(
        default=Path("/data1/video-highlight-cache"), validation_alias="VH_MEDIA_CACHE_DIR"
    )
    write_trace: bool = Field(default=False, validation_alias="VH_WRITE_TRACE")
    write_result_file: bool = Field(default=True, validation_alias="VH_WRITE_RESULT_FILE")
    export_clips: bool = Field(default=True, validation_alias="VH_EXPORT_CLIPS")
    window_sec: float = Field(default=20.0, validation_alias="VH_WINDOW_SEC", gt=2)
    stride_sec: float = Field(default=4.0, validation_alias="VH_STRIDE_SEC", gt=1)
    sample_fps: float = Field(default=1.0, validation_alias="VH_SAMPLE_FPS", gt=0)
    frame_width: int = Field(default=480, validation_alias="VH_FRAME_WIDTH", ge=128)
    local_coverage_sec: float = Field(
        default=90.0, validation_alias="VH_LOCAL_COVERAGE_SEC", ge=30
    )
    candidates_per_segment: int = Field(
        default=4, validation_alias="VH_CANDIDATES_PER_SEGMENT", ge=1
    )
    min_local_candidates: int = Field(default=6, validation_alias="VH_MIN_LOCAL_CANDIDATES", ge=1)
    max_local_candidates: int = Field(default=72, validation_alias="VH_MAX_LOCAL_CANDIDATES", ge=1)
    max_judge_candidates: int = Field(default=18, validation_alias="VH_MAX_JUDGE_CANDIDATES", ge=1)
    ocr_max_frames: int = Field(default=600, validation_alias="VH_OCR_MAX_FRAMES", ge=1)
    candidate_nms_iou: float = Field(
        default=0.60, validation_alias="VH_CANDIDATE_NMS_IOU", ge=0, lt=1
    )
    map_workers: int = Field(default=6, validation_alias="VH_MAP_WORKERS", ge=1)
    judge_workers: int = Field(default=6, validation_alias="VH_JUDGE_WORKERS", ge=1)
    max_highlights: int = Field(default=12, validation_alias="VH_MAX_HIGHLIGHTS", ge=1)
    final_nms_iou: float = Field(default=0.40, validation_alias="VH_FINAL_NMS_IOU", ge=0, lt=1)
    final_threshold: float = Field(default=0.65, validation_alias="VH_FINAL_THRESHOLD")
    request_timeout_sec: float = Field(default=300.0, validation_alias="VH_REQUEST_TIMEOUT_SEC")
    request_max_retries: int = Field(default=2, validation_alias="VH_REQUEST_MAX_RETRIES", ge=0)

    @property
    def reasoning_api_key(self) -> str:
        return (
            self.gemini_api_key if self.reasoning_provider == "gemini" else self.siliconflow_api_key
        )

    @property
    def reasoning_base_url(self) -> str:
        return (
            self.gemini_base_url
            if self.reasoning_provider == "gemini"
            else self.siliconflow_base_url
        )

    @property
    def reasoning_map_model(self) -> str:
        return (
            self.gemini_map_model
            if self.reasoning_provider == "gemini"
            else self.siliconflow_map_model
        )

    @property
    def reasoning_judge_model(self) -> str:
        return (
            self.gemini_judge_model
            if self.reasoning_provider == "gemini"
            else self.siliconflow_judge_model
        )
