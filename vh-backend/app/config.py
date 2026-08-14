from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    storage_dir: Path = Field(
        default=REPOSITORY_ROOT / "vh-frontend" / "video-data",
        validation_alias="VH_STORAGE_DIR",
    )
    agent_root: Path = Field(
        default=REPOSITORY_ROOT / "vh-agent",
        validation_alias="VH_AGENT_ROOT",
    )
    agent_uv_executable: str = Field(default="uv", validation_alias="VH_AGENT_UV_EXECUTABLE")
    agent_timeout_sec: int = Field(
        default=6 * 60 * 60,
        ge=60,
        validation_alias="VH_AGENT_TIMEOUT_SEC",
    )
    max_upload_bytes: int = Field(
        default=20 * 1024 * 1024 * 1024,
        ge=1,
        validation_alias="VH_MAX_UPLOAD_BYTES",
    )
    allowed_origins: str = Field(
        default="http://127.0.0.1:5173,http://localhost:5173,null",
        validation_alias="VH_ALLOWED_ORIGINS",
    )

    @field_validator("storage_dir", "agent_root", mode="after")
    @classmethod
    def resolve_path(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @property
    def jobs_dir(self) -> Path:
        return self.storage_dir / "jobs"

    @property
    def database_path(self) -> Path:
        return self.storage_dir / "tasks.sqlite3"

    @property
    def allowed_origin_list(self) -> list[str]:
        return [item.strip() for item in self.allowed_origins.split(",") if item.strip()]
