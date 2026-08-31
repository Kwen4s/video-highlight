import json
import sqlite3

from fastapi.testclient import TestClient

from app import main as main_module
from app.config import Settings
from app.main import create_app
from app.repository import JobRepository


class FakeRunner:
    def __init__(self) -> None:
        self.enqueued: list[str] = []

    def enqueue(self, job_id: str) -> None:
        self.enqueued.append(job_id)


def make_client(tmp_path):
    settings = Settings(
        VH_STORAGE_DIR=tmp_path / "runtime",
        VH_AGENT_ROOT=tmp_path / "agent",
    )
    repository = JobRepository(settings.database_path)
    runner = FakeRunner()
    app = create_app(settings=settings, repository=repository, runner=runner)
    return TestClient(app), settings, repository, runner


def sample_result(job_id: str = "job_abcdefgh") -> dict:
    return {
        "schema_version": "1.0",
        "job_id": job_id,
        "video": {"video_id": job_id, "title": "demo", "duration_sec": 60},
        "highlights": [
            {
                "highlight_id": "hl_1",
                "start_sec": 2,
                "end_sec": 8,
                "score": 0.91,
                "highlight_type": "emotion",
                "description": "情绪转折",
                "reason": "人物情绪发生明显变化",
                "review_status": "pending",
            }
        ],
    }


def create_completed_job(settings, repository, job_id: str = "job_abcdefgh") -> None:
    source_dir = settings.jobs_dir / job_id / "source"
    source_dir.mkdir(parents=True)
    (source_dir / "original.mp4").write_bytes(b"source")
    repository.create(
        job_id=job_id,
        original_name="demo.mp4",
        stored_name="original.mp4",
        content_type="video/mp4",
        size_bytes=6,
        language="zh",
    )
    repository.save_result(
        job_id,
        sample_result(job_id),
    )


def test_upload_returns_public_media_url_without_exposing_server_path(tmp_path) -> None:
    client, settings, _repository, runner = make_client(tmp_path)

    with client:
        response = client.post(
            "/api/jobs",
            data={"job_id": "job_12345678", "language": "zh"},
            files={"file": ("片段.mp4", b"video-bytes", "video/mp4")},
        )
        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "queued"
        assert body["current_stage"] == "orchestration"
        assert body["attempt"] == 0
        assert body["max_attempts"] == 3
        assert body["source_url"] == "/api/jobs/job_12345678/source"
        assert str(settings.jobs_dir) not in json.dumps(body)
        assert runner.enqueued == ["job_12345678"]
        assert (
            settings.jobs_dir / "job_12345678" / "source" / "original.mp4"
        ).read_bytes() == b"video-bytes"


def test_cors_allows_electron_origins_and_message_preflight(tmp_path) -> None:
    client, _settings, _repository, _runner = make_client(tmp_path)

    with client:
        development = client.get(
            "/health",
            headers={"Origin": "http://127.0.0.1:5173"},
        )
        assert development.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"

        packaged = client.options(
            "/api/jobs/job_12345678/messages",
            headers={
                "Origin": "null",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert packaged.status_code == 200
        assert packaged.headers["access-control-allow-origin"] == "null"
        assert "POST" in packaged.headers["access-control-allow-methods"]

        media_preflight = client.options(
            "/api/jobs/job_12345678/source",
            headers={
                "Origin": "http://127.0.0.1:5173",
                "Access-Control-Request-Method": "HEAD",
            },
        )
        assert media_preflight.status_code == 200
        assert "HEAD" in media_preflight.headers["access-control-allow-methods"]

        rejected = client.get(
            "/health",
            headers={"Origin": "http://unconfigured.example"},
        )
        assert "access-control-allow-origin" not in rejected.headers


def test_demo_job_can_open_a_conversation_session(tmp_path) -> None:
    client, _settings, repository, _runner = make_client(tmp_path)
    job_id = "job_demo_citypulse"

    with client:
        opened = client.post(
            f"/api/demo-jobs/{job_id}/session",
            json={
                "original_name": "城市节拍_原片.mp4",
                "size_bytes": 1024,
                "language": "zh",
                "result": sample_result(job_id),
            },
        )
        assert opened.status_code == 200
        body = opened.json()
        assert body["status"] == "completed"
        assert body["current_stage"] == "delivery"
        assert body["source_url"] is None
        assert body["revision"] == 0
        assert "session_expires_at" not in body

        edited = client.post(
            f"/api/jobs/{job_id}/messages",
            json={
                "message": "入点后移 1 秒",
                "revision": 0,
                "selected_highlight_id": "hl_1",
            },
        )
        assert edited.status_code == 200
        assert edited.json()["job"]["result"]["highlights"][0]["start_sec"] == 3
        assert repository.get(job_id)["stored_name"] == ""


def test_demo_session_bootstrap_rejects_non_demo_job(tmp_path) -> None:
    client, _settings, _repository, _runner = make_client(tmp_path)

    with client:
        response = client.post(
            "/api/demo-jobs/job_not_a_demo/session",
            json={
                "original_name": "fake.mp4",
                "size_bytes": 1,
                "language": "zh",
                "result": sample_result("job_not_a_demo"),
            },
        )
        assert response.status_code == 404


def test_result_contains_only_intervals_and_editing_is_versioned(tmp_path) -> None:
    client, settings, repository, _runner = make_client(tmp_path)

    with client:
        create_completed_job(settings, repository)
        response = client.get("/api/jobs/job_abcdefgh")
        body = response.json()
        highlight = body["result"]["highlights"][0]
        assert "clip_url" not in highlight
        assert body["source_url"] == "/api/jobs/job_abcdefgh/source"
        assert body["revision"] == 0
        assert "session_expires_at" not in body

        edited = client.post(
            "/api/jobs/job_abcdefgh/messages",
            json={
                "message": "入点后移 1 秒",
                "revision": 0,
                "selected_highlight_id": "hl_1",
            },
        )
        assert edited.status_code == 200
        assert edited.json()["changed"] is True
        assert edited.json()["job"]["revision"] == 1
        assert edited.json()["job"]["result"]["highlights"][0]["start_sec"] == 3
        conversation = repository.get_conversation("job_abcdefgh")
        assert conversation[0]["user_message"] == "入点后移 1 秒"
        assert conversation[0]["revision"] == 1

        ambiguous = client.post(
            "/api/jobs/job_abcdefgh/messages",
            json={
                "message": "缩短 1 秒",
                "revision": 1,
                "selected_highlight_id": "hl_1",
            },
        )
        assert ambiguous.status_code == 200
        assert ambiguous.json()["changed"] is False
        assert "哪一端" in ambiguous.json()["reply"]

        undone = client.post(
            "/api/jobs/job_abcdefgh/messages",
            json={"message": "撤销", "revision": 1},
        )
        assert undone.status_code == 200
        assert undone.json()["job"]["revision"] == 2
        assert undone.json()["job"]["result"]["highlights"][0]["start_sec"] == 2

        stale = client.post(
            "/api/jobs/job_abcdefgh/messages",
            json={"message": "撤销", "revision": 1},
        )
        assert stale.status_code == 409


def test_highlight_range_edit_is_validated_and_versioned(tmp_path) -> None:
    client, settings, repository, _runner = make_client(tmp_path)

    with client:
        create_completed_job(settings, repository)
        edited = client.post(
            "/api/jobs/job_abcdefgh/highlights/hl_1/range",
            json={"start_sec": 3.25, "end_sec": 9.5, "revision": 0},
        )
        assert edited.status_code == 200
        body = edited.json()
        assert body["revision"] == 1
        assert body["result"]["highlights"][0]["start_sec"] == 3.25
        assert body["result"]["highlights"][0]["end_sec"] == 9.5
        assert body["result"]["highlights"][0]["review_status"] == "revised"

        stale = client.post(
            "/api/jobs/job_abcdefgh/highlights/hl_1/range",
            json={"start_sec": 4, "end_sec": 10, "revision": 0},
        )
        assert stale.status_code == 409

        invalid = client.post(
            "/api/jobs/job_abcdefgh/highlights/hl_1/range",
            json={"start_sec": 9.4, "end_sec": 9.5, "revision": 1},
        )
        assert invalid.status_code == 422


def test_source_supports_inline_head_and_byte_ranges(tmp_path) -> None:
    client, settings, repository, _runner = make_client(tmp_path)

    with client:
        create_completed_job(settings, repository)

        head = client.head(
            "/api/jobs/job_abcdefgh/source",
            headers={"Origin": "http://127.0.0.1:5173"},
        )
        assert head.status_code == 200
        assert head.content == b""
        assert head.headers["content-type"] == "video/mp4"
        assert head.headers["content-length"] == "6"
        assert head.headers["accept-ranges"] == "bytes"
        assert head.headers["cache-control"] == "private, no-store"
        assert head.headers["content-disposition"].startswith("inline;")
        assert head.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"

        partial = client.get(
            "/api/jobs/job_abcdefgh/source",
            headers={"Range": "bytes=1-3"},
        )
        assert partial.status_code == 206
        assert partial.content == b"our"
        assert partial.headers["content-range"] == "bytes 1-3/6"
        assert partial.headers["content-length"] == "3"


def test_source_returns_404_when_media_file_is_missing(tmp_path) -> None:
    client, settings, repository, _runner = make_client(tmp_path)

    with client:
        create_completed_job(settings, repository)
        (settings.jobs_dir / "job_abcdefgh" / "source" / "original.mp4").unlink()

        response = client.get("/api/jobs/job_abcdefgh/source")
        assert response.status_code == 404


def test_source_infers_video_type_when_upload_type_is_generic(tmp_path) -> None:
    client, settings, repository, _runner = make_client(tmp_path)

    with client:
        source_dir = settings.jobs_dir / "job_generic123" / "source"
        source_dir.mkdir(parents=True)
        (source_dir / "original.m4v").write_bytes(b"source")
        repository.create(
            job_id="job_generic123",
            original_name="demo.m4v",
            stored_name="original.m4v",
            content_type="application/octet-stream",
            size_bytes=6,
            language="zh",
        )

        response = client.head("/api/jobs/job_generic123/source")
        assert response.status_code == 200
        assert response.headers["content-type"] == "video/x-m4v"


def test_message_stream_emits_incremental_reply_and_final_job(tmp_path) -> None:
    client, settings, repository, _runner = make_client(tmp_path)

    with client:
        create_completed_job(settings, repository)
        response = client.post(
            "/api/jobs/job_abcdefgh/messages/stream",
            json={
                "message": "入点后移 1 秒",
                "revision": 0,
                "selected_highlight_id": "hl_1",
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        events = [json.loads(line) for line in response.text.splitlines()]
        assert events[0] == {"type": "start"}
        assert "".join(
            event["delta"] for event in events if event["type"] == "delta"
        )
        completed = events[-1]
        assert completed["type"] == "complete"
        assert completed["changed"] is True
        assert completed["job"]["revision"] == 1
        assert completed["job"]["result"]["highlights"][0]["start_sec"] == 3


def test_completed_job_remains_available_and_editable_without_time_limit(tmp_path) -> None:
    client, settings, repository, _runner = make_client(tmp_path)

    with client:
        create_completed_job(settings, repository)
        with sqlite3.connect(settings.database_path) as connection:
            connection.execute("ALTER TABLE jobs ADD COLUMN session_expires_at TEXT")
            connection.execute(
                """
                UPDATE jobs
                SET updated_at = '2000-01-01T00:00:00+00:00',
                    session_expires_at = '2000-01-01T00:00:00+00:00'
                WHERE job_id = ?
                """,
                ("job_abcdefgh",),
            )

        listed = client.get("/api/jobs")
        assert listed.status_code == 200
        assert any(job["job_id"] == "job_abcdefgh" for job in listed.json())

        response = client.post(
            "/api/jobs/job_abcdefgh/messages",
            json={"message": "入点后移 1 秒", "revision": 0, "selected_highlight_id": "hl_1"},
        )
        assert response.status_code == 200
        assert response.json()["job"]["result"]["highlights"][0]["start_sec"] == 3


def test_stale_queued_job_is_not_deleted_automatically(tmp_path) -> None:
    client, settings, repository, _runner = make_client(tmp_path)

    with client:
        source_dir = settings.jobs_dir / "job_stalequeue" / "source"
        source_dir.mkdir(parents=True)
        (source_dir / "original.mp4").write_bytes(b"source")
        repository.create(
            job_id="job_stalequeue",
            original_name="demo.mp4",
            stored_name="original.mp4",
            content_type="video/mp4",
            size_bytes=6,
            language="zh",
        )
        with sqlite3.connect(settings.database_path) as connection:
            connection.execute(
                "UPDATE jobs SET updated_at = '2000-01-01T00:00:00+00:00' WHERE job_id = ?",
                ("job_stalequeue",),
            )

        listed = client.get("/api/jobs")

        assert listed.status_code == 200
        assert any(job["job_id"] == "job_stalequeue" for job in listed.json())
        assert repository.get("job_stalequeue") is not None
        assert (settings.jobs_dir / "job_stalequeue").exists()


def test_list_jobs_returns_every_submitted_job_without_hidden_limit(tmp_path) -> None:
    client, _settings, repository, _runner = make_client(tmp_path)

    with client:
        for index in range(125):
            repository.create(
                job_id=f"job_bulk{index:04d}",
                original_name=f"video-{index}.mp4",
                stored_name="original.mp4",
                content_type="video/mp4",
                size_bytes=index,
                language="zh",
            )

        listed = client.get("/api/jobs")

        assert listed.status_code == 200
        assert len(listed.json()) == 125
        assert {job["job_id"] for job in listed.json()} == {
            f"job_bulk{index:04d}" for index in range(125)
        }


def test_delete_cleans_finished_temporary_job_and_reports_locked_files(
    tmp_path, monkeypatch
) -> None:
    client, settings, repository, _runner = make_client(tmp_path)

    with client:
        create_completed_job(settings, repository, "job_finished1")
        unconfirmed = client.delete("/api/jobs/job_finished1")
        assert unconfirmed.status_code == 422
        assert repository.get("job_finished1") is not None
        assert (settings.jobs_dir / "job_finished1").exists()

        mismatched = client.request(
            "DELETE",
            "/api/jobs/job_finished1",
            json={"confirmed": True, "job_id": "job_different1"},
        )
        assert mismatched.status_code == 409
        assert repository.get("job_finished1") is not None
        assert (settings.jobs_dir / "job_finished1").exists()

        deleted = client.request(
            "DELETE",
            "/api/jobs/job_finished1",
            json={"confirmed": True, "job_id": "job_finished1"},
        )
        assert deleted.status_code == 204
        assert repository.get("job_finished1") is None
        assert not (settings.jobs_dir / "job_finished1").exists()

        create_completed_job(settings, repository, "job_locked123")

        def locked_replace(_source, _target):
            raise PermissionError("locked")

        monkeypatch.setattr(main_module.time, "sleep", lambda _delay: None)
        monkeypatch.setattr(main_module.Path, "replace", locked_replace)
        locked = client.request(
            "DELETE",
            "/api/jobs/job_locked123",
            json={"confirmed": True, "job_id": "job_locked123"},
        )
        assert locked.status_code == 423
        assert repository.get("job_locked123") is not None
        assert (settings.jobs_dir / "job_locked123").exists()
