from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore", env_prefix="VH_DATA_")

    api_key: SecretStr = SecretStr("")
    base_url: str = "https://api.fishkj.site/v1"
    model: str = ""
    reasoning_effort: str = "high"
    root: Path = ROOT / "outputs/data"
    media_cache: Path = ROOT.parent / "vh-agent/outputs/data-media"
    preprocess_cache: Path = Path("/data1/video-highlight-cache")
    asr_model: Path = Path(
        "/data1/video-highlight-models/faster-whisper/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478"
    )
    asr_device: str = "cpu"
    asr_compute_type: str = "int8"
    page_sec: float = Field(default=30, gt=0)
    preferred_highlight_sec: float = Field(default=15, gt=0)
    frame_fps: float = Field(default=1, gt=0, le=24)
    agreement_iou: float = Field(default=0.5, gt=0, le=1)
    request_timeout_sec: float = Field(default=300, gt=0)
    request_attempts: int = Field(default=3, ge=1, le=10)
    workers: int = Field(default=2, ge=1, le=8)

    @field_validator("root", "media_cache", "preprocess_cache", "asr_model", mode="after")
    @classmethod
    def resolve_path(cls, value: Path) -> Path:
        value = value.expanduser()
        return (ROOT / value if not value.is_absolute() else value).resolve()
