from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    gemini_base_url: str = Field(
        default="https://yetoken.vip/v1", validation_alias="GEMINI_BASE_URL"
    )
    gemini_agent_model: str = Field(
        default="gemini-3.7-flash", validation_alias="GEMINI_AGENT_MODEL"
    )
    generation_seed: int | None = Field(default=7, ge=0, validation_alias="VH_GENERATION_SEED")
    request_timeout_sec: float = Field(default=300, gt=0, validation_alias="VH_REQUEST_TIMEOUT_SEC")
    job_output_dir: Path = Field(default=Path("outputs/jobs"), validation_alias="VH_JOB_OUTPUT_DIR")
    media_cache_dir: Path = Field(
        default=Path("/data1/video-highlight-cache"), validation_alias="VH_MEDIA_CACHE_DIR"
    )
    page_sec: float = Field(default=30, gt=0, validation_alias="VH_PAGE_SEC")
    video_fps: float = Field(default=4, gt=0, validation_alias="VH_VIDEO_FPS")
    context_token_budget: int = Field(
        default=100_000, gt=8192, validation_alias="VH_CONTEXT_TOKEN_BUDGET"
    )
    context_byte_budget: int = Field(
        default=18_000_000, gt=0, lt=20_000_000, validation_alias="VH_CONTEXT_BYTE_BUDGET"
    )
    max_request_attempts: int = Field(default=3, gt=0, validation_alias="VH_MAX_REQUEST_ATTEMPTS")
    max_stagnant_steps: int = Field(default=8, gt=0, validation_alias="VH_MAX_STAGNANT_STEPS")
    local_checkpoint: Path | None = Field(default=None, validation_alias="VH_LOCAL_CHECKPOINT")
    local_device: str = Field(default="cpu", validation_alias="VH_LOCAL_DEVICE")
    local_feature_cache_dir: Path = Field(
        default=Path("/data1/video-highlight-model-features"),
        validation_alias="VH_LOCAL_FEATURE_CACHE_DIR",
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
