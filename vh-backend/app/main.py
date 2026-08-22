import asyncio
import json
import re
import shutil
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from .agent import AgentJobRunner
from .config import Settings
from .conversation import ConversationAgentError, HighlightConversationAgent
from .editing import validate_highlight_range
from .models import (
    DemoSessionRequest,
    DetectionResult,
    EditMessageRequest,
    EditMessageResponse,
    JobResponse,
)
from .repository import JobRepository, utc_after, utc_now

JOB_ID_PATTERN = re.compile(r"^job_[A-Za-z0-9_-]{8,48}$")
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
CHUNK_SIZE = 1024 * 1024
DELETE_RETRY_DELAYS_SEC = (0.0, 0.1, 0.25, 0.5, 1.0)
DEMO_JOB_IDS = frozenset({"job_demo_citypulse", "job_demo_launchfilm"})
STREAM_REPLY_CHUNK_SIZE = 4
STREAM_REPLY_DELAY_SEC = 0.02


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


def purge_job(settings: Settings, repository: JobRepository, job_id: str) -> None:
    jobs_root = settings.jobs_dir.resolve()
    job_dir = (jobs_root / job_id).resolve()
    if job_dir.parent != jobs_root:
        raise ValueError("invalid job id")

    tombstone: Path | None = None
    if job_dir.exists():
        trash_root = settings.storage_dir / ".trash"
        trash_root.mkdir(parents=True, exist_ok=True)
        tombstone = trash_root / f"{job_id}-{uuid4().hex}"
        move_job_to_trash(job_dir, tombstone)

    try:
        repository.delete(job_id)
    except Exception:
        if tombstone and tombstone.exists():
            tombstone.replace(job_dir)
        raise
    if tombstone and tombstone.exists():
        shutil.rmtree(tombstone)


def purge_expired_jobs(settings: Settings, repository: JobRepository) -> None:
    for job_id in repository.list_expired(
        utc_now(),
        utc_after(-settings.orphan_job_ttl_sec),
    ):
        try:
            purge_job(settings, repository, job_id)
        except (JobFilesLockedError, OSError):
            continue


async def cleanup_loop(settings: Settings, repository: JobRepository) -> None:
    while True:
        await asyncio.sleep(settings.cleanup_interval_sec)
        await asyncio.to_thread(purge_expired_jobs, settings, repository)


def create_app(
    settings: Settings | None = None,
    repository: JobRepository | None = None,
    runner: Any | None = None,
    conversation_agent: Any | None = None,
) -> FastAPI:
    app_settings = settings or Settings()
    app_repository = repository or JobRepository(app_settings.database_path)
    app_runner = runner or AgentJobRunner(app_settings, app_repository)
    app_conversation_agent = conversation_agent or HighlightConversationAgent(app_settings)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        app_settings.jobs_dir.mkdir(parents=True, exist_ok=True)
        app_repository.initialize()
        purge_expired_jobs(app_settings, app_repository)
        cleanup_task = asyncio.create_task(cleanup_loop(app_settings, app_repository))
        application.state.settings = app_settings
        application.state.repository = app_repository
        application.state.runner = app_runner
        yield
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task
        shutdown = getattr(app_runner, "shutdown", None)
        if shutdown:
            shutdown()

    application = FastAPI(
        title="Video Highlight API",
        version="0.2.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.allowed_origin_list,
        allow_methods=["GET", "POST", "DELETE"],
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
            row = app_repository.create(
                job_id=job_id,
                original_name=original_name,
                stored_name=stored_name,
                content_type=file.content_type or "application/octet-stream",
                size_bytes=size_bytes,
                language=language,
            )
        except Exception:
            partial_path.unlink(missing_ok=True)
            if app_repository.get(job_id) is None:
                resolved_job_dir = job_dir.resolve()
                if resolved_job_dir.parent == app_settings.jobs_dir.resolve():
                    shutil.rmtree(resolved_job_dir, ignore_errors=True)
            raise
        finally:
            await file.close()

        app_runner.enqueue(job_id)
        return serialize_job(row)

    @application.get("/api/jobs", response_model=list[JobResponse])
    def list_jobs(limit: int = 50) -> list[JobResponse]:
        purge_expired_jobs(app_settings, app_repository)
        safe_limit = max(1, min(limit, 100))
        return [serialize_job(row) for row in app_repository.list(safe_limit)]

    @application.post("/api/demo-jobs/{job_id}/session", response_model=JobResponse)
    def open_demo_session(job_id: str, request: DemoSessionRequest) -> JobResponse:
        if job_id not in DEMO_JOB_IDS:
            raise HTTPException(status_code=404, detail="演示任务不存在")
        if request.result.job_id != job_id or request.result.video.video_id != job_id:
            raise HTTPException(status_code=422, detail="演示结果与任务编号不匹配")
        if len(request.result.highlights) > 50:
            raise HTTPException(status_code=422, detail="演示高光数量超出限制")
        for item in request.result.highlights:
            range_error = validate_highlight_range(
                item.start_sec,
                item.end_sec,
                request.result.video.duration_sec,
            )
            if range_error:
                raise HTTPException(status_code=422, detail=range_error)
        if request.size_bytes > app_settings.max_upload_bytes:
            raise HTTPException(status_code=422, detail="演示视频大小超出限制")

        row = app_repository.open_demo_session(
            job_id=job_id,
            original_name=Path(request.original_name).name,
            size_bytes=request.size_bytes,
            language=request.language,
            result=request.result.model_dump(mode="json"),
            session_expires_at=utc_after(app_settings.edit_session_ttl_sec),
        )
        return serialize_job(row)

    @application.get("/api/jobs/{job_id}", response_model=JobResponse)
    def get_job(job_id: str) -> JobResponse:
        return serialize_job(require_job(app_repository, job_id))

    @application.post(
        "/api/jobs/{job_id}/messages",
        response_model=EditMessageResponse,
    )
    def edit_job(job_id: str, request: EditMessageRequest) -> EditMessageResponse:
        row = require_job(app_repository, job_id)
        validate_edit_request(row, request)

        conversation = app_repository.get_conversation(job_id)
        try:
            outcome = app_conversation_agent.respond(
                result=parse_result(row),
                message=request.message,
                selected_highlight_id=request.selected_highlight_id,
                conversation=conversation,
            )
        except ConversationAgentError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        expires_at = utc_after(app_settings.edit_session_ttl_sec)
        if outcome.undo:
            undone = app_repository.undo_result(
                job_id,
                expected_revision=request.revision,
                session_expires_at=expires_at,
            )
            if undone is False:
                raise HTTPException(status_code=409, detail="结果版本已更新，请重新载入后再编辑")
            if undone is None:
                app_repository.touch_session(
                    job_id,
                    expected_revision=request.revision,
                    session_expires_at=expires_at,
                )
                reply = "没有可以撤销的高光修改。"
                changed = False
            else:
                reply = "已撤销上一次高光修改。"
                changed = True
        elif outcome.changed:
            replaced = app_repository.replace_result(
                job_id,
                outcome.result.model_dump(mode="json"),
                expected_revision=request.revision,
                session_expires_at=expires_at,
            )
            if not replaced:
                raise HTTPException(status_code=409, detail="结果版本已更新，请重新载入后再编辑")
            reply = outcome.reply
            changed = True
        else:
            touched = app_repository.touch_session(
                job_id,
                expected_revision=request.revision,
                session_expires_at=expires_at,
            )
            if not touched:
                raise HTTPException(status_code=409, detail="结果版本已更新，请重新载入后再编辑")
            reply = outcome.reply
            changed = False

        resulting_revision = request.revision + (1 if changed else 0)
        app_repository.append_conversation(
            job_id,
            expected_revision=resulting_revision,
            user_message=request.message,
            assistant_reply=reply,
            action=outcome.action,
        )

        return EditMessageResponse(
            job=serialize_job(require_job(app_repository, job_id)),
            reply=reply,
            changed=changed,
        )

    @application.post("/api/jobs/{job_id}/messages/stream")
    async def stream_edit_job(
        job_id: str,
        request: EditMessageRequest,
    ) -> StreamingResponse:
        validate_edit_request(require_job(app_repository, job_id), request)

        async def events() -> AsyncIterator[bytes]:
            yield encode_stream_event({"type": "start"})
            try:
                response = await asyncio.to_thread(edit_job, job_id, request)
            except HTTPException as error:
                detail = error.detail if isinstance(error.detail, str) else "高光编辑失败"
                yield encode_stream_event(
                    {"type": "error", "status": error.status_code, "detail": detail}
                )
                return
            except Exception:
                yield encode_stream_event(
                    {"type": "error", "status": 500, "detail": "高光编辑失败，请稍后重试"}
                )
                return

            for index in range(0, len(response.reply), STREAM_REPLY_CHUNK_SIZE):
                yield encode_stream_event(
                    {
                        "type": "delta",
                        "delta": response.reply[index : index + STREAM_REPLY_CHUNK_SIZE],
                    }
                )
                await asyncio.sleep(STREAM_REPLY_DELAY_SEC)

            yield encode_stream_event(
                {
                    "type": "complete",
                    "job": response.job.model_dump(mode="json"),
                    "changed": response.changed,
                }
            )

        return StreamingResponse(
            events(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    @application.delete("/api/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_job(job_id: str) -> Response:
        row = require_job(app_repository, job_id)
        if row["status"] in {"queued", "processing"}:
            raise HTTPException(status_code=409, detail="正在分析的任务不能清理")
        try:
            purge_job(app_settings, app_repository, job_id)
        except JobFilesLockedError as error:
            raise HTTPException(
                status_code=423,
                detail="临时任务文件仍被占用，请等待几秒后重试",
            ) from error
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return application


def require_job(repository: JobRepository, job_id: str) -> dict[str, Any]:
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise HTTPException(status_code=422, detail="无效的任务编号")
    row = repository.get(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="临时任务不存在或已清理")
    return row


def validate_edit_request(row: dict[str, Any], request: EditMessageRequest) -> None:
    if row["status"] != "completed":
        raise HTTPException(status_code=409, detail="任务尚未完成，不能编辑高光")
    if not row.get("session_expires_at") or _is_expired(row["session_expires_at"]):
        raise HTTPException(status_code=410, detail="编辑会话已结束，本地结果仍可继续审阅")
    if row["revision"] != request.revision:
        raise HTTPException(status_code=409, detail="结果版本已更新，请重新载入后再编辑")


def encode_stream_event(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


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
    return JobResponse(
        job_id=row["job_id"],
        status=row["status"],
        original_name=row["original_name"],
        content_type=row["content_type"],
        size_bytes=row["size_bytes"],
        language=row["language"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        session_expires_at=row.get("session_expires_at"),
        revision=row.get("revision", 0),
        error_message=row["error_message"],
        result=result,
    )


def _is_expired(value: str) -> bool:
    return datetime.fromisoformat(value) <= datetime.now(UTC)


app = create_app()
