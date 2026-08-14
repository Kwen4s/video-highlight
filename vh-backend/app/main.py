import json
import re
import shutil
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from .agent import AgentJobRunner
from .config import Settings
from .models import DetectionResult, JobResponse, ReviewRequest
from .repository import JobRepository

JOB_ID_PATTERN = re.compile(r"^job_[A-Za-z0-9_-]{8,48}$")
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
CHUNK_SIZE = 1024 * 1024
DELETE_RETRY_DELAYS_SEC = (0.0, 0.1, 0.25, 0.5, 1.0)


class JobFilesLockedError(RuntimeError):
    pass


def move_job_to_trash(job_dir: Path, tombstone: Path) -> None:
    last_error: PermissionError | None = None
    for delay in DELETE_RETRY_DELAYS_SEC:
        if delay:
            time.sleep(delay)
        try:
            job_dir.replace(tombstone)
            return
        except PermissionError as error:
            last_error = error
    raise JobFilesLockedError("job directory is in use") from last_error


def create_app(
    settings: Settings | None = None,
    repository: JobRepository | None = None,
    runner: Any | None = None,
) -> FastAPI:
    app_settings = settings or Settings()
    app_repository = repository or JobRepository(app_settings.database_path)
    app_runner = runner or AgentJobRunner(app_settings, app_repository)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        app_settings.jobs_dir.mkdir(parents=True, exist_ok=True)
        app_repository.initialize()
        application.state.settings = app_settings
        application.state.repository = app_repository
        application.state.runner = app_runner
        yield
        shutdown = getattr(app_runner, "shutdown", None)
        if shutdown:
            shutdown()

    application = FastAPI(
        title="Video Highlight API",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.allowed_origin_list,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["*"],
    )

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.post("/api/jobs", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
    async def create_job(
        file: Annotated[UploadFile, File()],
        job_id: Annotated[str, Form()],
        language: Annotated[str, Form()] = "zh",
    ) -> JobResponse:
        if not JOB_ID_PATTERN.fullmatch(job_id):
            raise HTTPException(status_code=422, detail="无效的任务编号")
        if language not in {"zh", "en"}:
            raise HTTPException(status_code=422, detail="language 必须是 zh 或 en")
        if app_repository.get(job_id):
            raise HTTPException(status_code=409, detail="任务已存在")

        original_name = Path(file.filename or "video.mp4").name
        extension = Path(original_name).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=415, detail="不支持的视频格式")

        job_dir = app_settings.jobs_dir / job_id
        source_dir = job_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        stored_name = f"original{extension}"
        target_path = source_dir / stored_name
        partial_path = source_dir / f".{stored_name}.uploading"
        size_bytes = 0
        try:
            with partial_path.open("wb") as sink:
                while chunk := await file.read(CHUNK_SIZE):
                    size_bytes += len(chunk)
                    if size_bytes > app_settings.max_upload_bytes:
                        raise HTTPException(status_code=413, detail="视频文件超过上传大小限制")
                    sink.write(chunk)
            partial_path.replace(target_path)
        except Exception:
            partial_path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()

        row = app_repository.create(
            job_id=job_id,
            original_name=original_name,
            stored_name=stored_name,
            content_type=file.content_type or "application/octet-stream",
            size_bytes=size_bytes,
            language=language,
        )
        app_runner.enqueue(job_id)
        return serialize_job(row)

    @application.get("/api/jobs", response_model=list[JobResponse])
    def list_jobs(limit: int = 50) -> list[JobResponse]:
        safe_limit = max(1, min(limit, 100))
        return [serialize_job(row) for row in app_repository.list(safe_limit)]

    @application.get("/api/jobs/{job_id}", response_model=JobResponse)
    def get_job(job_id: str) -> JobResponse:
        return serialize_job(require_job(app_repository, job_id))

    @application.delete("/api/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_job(job_id: str) -> Response:
        row = require_job(app_repository, job_id)
        if row["status"] in {"queued", "processing"}:
            raise HTTPException(status_code=409, detail="正在分析的任务不能删除")

        jobs_root = app_settings.jobs_dir.resolve()
        job_dir = (jobs_root / job_id).resolve()
        if job_dir.parent != jobs_root:
            raise HTTPException(status_code=422, detail="无效的任务编号")

        tombstone: Path | None = None
        if job_dir.exists():
            trash_root = app_settings.storage_dir / ".trash"
            trash_root.mkdir(parents=True, exist_ok=True)
            tombstone = trash_root / f"{job_id}-{uuid4().hex}"
            try:
                move_job_to_trash(job_dir, tombstone)
            except JobFilesLockedError as error:
                raise HTTPException(
                    status_code=423,
                    detail="任务文件仍被播放器或其他程序占用，请等待几秒后重试",
                ) from error

        try:
            app_repository.delete(job_id)
        except Exception:
            if tombstone and tombstone.exists():
                tombstone.replace(job_dir)
            raise

        if tombstone and tombstone.exists():
            shutil.rmtree(tombstone)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @application.get("/api/jobs/{job_id}/source")
    def stream_source(job_id: str) -> FileResponse:
        row = require_job(app_repository, job_id)
        video_path = app_settings.jobs_dir / job_id / "source" / row["stored_name"]
        if not video_path.is_file():
            raise HTTPException(status_code=404, detail="原视频不存在")
        return FileResponse(video_path, media_type=row["content_type"])

    @application.get("/api/jobs/{job_id}/highlights/{highlight_id}/video")
    def stream_highlight(job_id: str, highlight_id: str) -> FileResponse:
        row = require_job(app_repository, job_id)
        result = parse_result(row)
        item = next(
            (value for value in result.highlights if value.highlight_id == highlight_id),
            None,
        )
        if item is None:
            raise HTTPException(status_code=404, detail="高光片段不存在")
        relative_path = Path(item.clip_url)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise HTTPException(status_code=500, detail="高光结果路径无效")
        clip_path = (app_settings.jobs_dir / job_id / relative_path).resolve()
        job_root = (app_settings.jobs_dir / job_id).resolve()
        if job_root not in clip_path.parents or not clip_path.is_file():
            raise HTTPException(status_code=404, detail="高光视频尚不可用")
        return FileResponse(clip_path, media_type="video/mp4")

    @application.patch(
        "/api/jobs/{job_id}/highlights/{highlight_id}",
        response_model=JobResponse,
    )
    def review_highlight(job_id: str, highlight_id: str, review: ReviewRequest) -> JobResponse:
        row = require_job(app_repository, job_id)
        result = parse_result(row)
        item = next(
            (value for value in result.highlights if value.highlight_id == highlight_id),
            None,
        )
        if item is None:
            raise HTTPException(status_code=404, detail="高光片段不存在")
        item.review_status = review.status
        result_data = result.model_dump(mode="json")
        result_path = app_settings.jobs_dir / job_id / "result.json"
        temporary_path = result_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(result_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(result_path)
        app_repository.replace_result(job_id, result_data)
        return serialize_job(require_job(app_repository, job_id))

    return application


def require_job(repository: JobRepository, job_id: str) -> dict[str, Any]:
    row = repository.get(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return row


def parse_result(row: dict[str, Any]) -> DetectionResult:
    if not row.get("result_json"):
        raise HTTPException(status_code=409, detail="任务结果尚未生成")
    return DetectionResult.model_validate_json(row["result_json"])


def serialize_job(row: dict[str, Any]) -> JobResponse:
    result = (
        DetectionResult.model_validate_json(row["result_json"])
        if row.get("result_json")
        else None
    )
    if result:
        for highlight in result.highlights:
            highlight.clip_url = (
                f"/api/jobs/{row['job_id']}/highlights/{highlight.highlight_id}/video"
            )
    return JobResponse(
        job_id=row["job_id"],
        status=row["status"],
        original_name=row["original_name"],
        content_type=row["content_type"],
        size_bytes=row["size_bytes"],
        language=row["language"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        source_url=f"/api/jobs/{row['job_id']}/source",
        error_message=row["error_message"],
        result=result,
    )


app = create_app()
