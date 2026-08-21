from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from app import main as main_module
from app.config import Settings
from app.main import create_app
from app.repository import JobRepository, utc_after


class FakeRunner:
    def __init__(self) -> None:
        self.enqueued: list[str] = []

    def enqueue(self, job_id: str) -> None:
        self.enqueued.append(job_id)


def make_client(tmp_path):
    settings = Settings(
        VH_STORAGE_DIR=tmp_path / "runtime",
        VH_AGENT_ROOT=tmp_path / "agent",
        VH_EDIT_SESSION_TTL_SEC=600,
        VH_CLEANUP_INTERVAL_SEC=3600,
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
        session_expires_at=utc_after(600),
    )


def test_upload_is_temporary_and_response_has_no_server_media_url(tmp_path) -> None:
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
        assert "source_url" not in body
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

        rejected = client.get(
            "/health",
            headers={"Origin": "http://unconfigured.example"},
        )
        assert "access-control-allow-origin" not in rejected.headers


def test_demo_job_can_open_a_temporary_conversation_session(tmp_path) -> None:
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
        assert body["revision"] == 0
        assert datetime.fromisoformat(body["session_expires_at"]) > datetime.now(UTC)

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


def test_result_contains_only_intervals_and_edit_session_is_versioned(tmp_path) -> None:
    client, settings, repository, _runner = make_client(tmp_path)

    with client:
        create_completed_job(settings, repository)
        response = client.get("/api/jobs/job_abcdefgh")
        body = response.json()
        highlight = body["result"]["highlights"][0]
        assert "clip_url" not in highlight
        assert "source_url" not in body
        assert body["revision"] == 0
        assert body["session_expires_at"]

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


def test_expired_session_rejects_edits_but_local_result_can_survive(tmp_path) -> None:
    client, settings, repository, _runner = make_client(tmp_path)

    with client:
        create_completed_job(settings, repository)
        expired = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
        repository.touch_session(
            "job_abcdefgh",
            expected_revision=0,
            session_expires_at=expired,
        )
        response = client.post(
            "/api/jobs/job_abcdefgh/messages",
            json={"message": "入点后移 1 秒", "revision": 0, "selected_highlight_id": "hl_1"},
        )
        assert response.status_code == 410


def test_delete_cleans_finished_temporary_job_and_reports_locked_files(
    tmp_path, monkeypatch
) -> None:
    client, settings, repository, _runner = make_client(tmp_path)

    with client:
        create_completed_job(settings, repository, "job_finished1")
        deleted = client.delete("/api/jobs/job_finished1")
        assert deleted.status_code == 204
        assert repository.get("job_finished1") is None
        assert not (settings.jobs_dir / "job_finished1").exists()

        create_completed_job(settings, repository, "job_locked123")

        def locked_replace(_source, _target):
            raise PermissionError("locked")

        monkeypatch.setattr(main_module.time, "sleep", lambda _delay: None)
        monkeypatch.setattr(main_module.Path, "replace", locked_replace)
        locked = client.delete("/api/jobs/job_locked123")
        assert locked.status_code == 423
        assert repository.get("job_locked123") is not None
        assert (settings.jobs_dir / "job_locked123").exists()
