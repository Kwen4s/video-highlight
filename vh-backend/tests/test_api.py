import json

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
        VH_STORAGE_DIR=tmp_path / "data",
        VH_AGENT_ROOT=tmp_path / "agent",
    )
    repository = JobRepository(settings.database_path)
    runner = FakeRunner()
    app = create_app(settings=settings, repository=repository, runner=runner)
    return TestClient(app), settings, repository, runner


def test_upload_streams_video_and_enqueues_job(tmp_path) -> None:
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
        assert body["original_name"] == "片段.mp4"
        assert body["source_url"] == "/api/jobs/job_12345678/source"
        assert runner.enqueued == ["job_12345678"]
        assert (
            settings.jobs_dir / "job_12345678" / "source" / "original.mp4"
        ).read_bytes() == b"video-bytes"
        assert client.get(body["source_url"]).content == b"video-bytes"


def test_completed_result_is_mapped_to_public_video_url_and_reviewed(tmp_path) -> None:
    client, settings, repository, _runner = make_client(tmp_path)
    result = {
        "schema_version": "1.0",
        "job_id": "job_abcdefgh",
        "video": {"video_id": "job_abcdefgh", "title": "demo", "duration_sec": 12},
        "highlights": [
            {
                "highlight_id": "hl_1",
                "start_sec": 2,
                "end_sec": 8,
                "score": 0.91,
                "highlight_type": "emotion",
                "description": "情绪转折",
                "reason": "人物情绪发生明显变化",
                "clip_url": "clips/hl_1.mp4",
                "review_status": "pending",
            }
        ],
    }

    with client:
        source_dir = settings.jobs_dir / "job_abcdefgh" / "source"
        source_dir.mkdir(parents=True)
        (source_dir / "original.mp4").write_bytes(b"source")
        repository.create(
            job_id="job_abcdefgh",
            original_name="demo.mp4",
            stored_name="original.mp4",
            content_type="video/mp4",
            size_bytes=6,
            language="zh",
        )
        repository.save_result("job_abcdefgh", result)
        result_path = settings.jobs_dir / "job_abcdefgh" / "result.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        clip_path = settings.jobs_dir / "job_abcdefgh" / "clips" / "hl_1.mp4"
        clip_path.parent.mkdir()
        clip_path.write_bytes(b"clip")

        response = client.get("/api/jobs/job_abcdefgh")
        clip_url = response.json()["result"]["highlights"][0]["clip_url"]
        assert clip_url == "/api/jobs/job_abcdefgh/highlights/hl_1/video"
        assert client.get(clip_url).content == b"clip"

        reviewed = client.patch(
            "/api/jobs/job_abcdefgh/highlights/hl_1",
            json={"status": "accepted"},
        )
        assert reviewed.status_code == 200
        assert reviewed.json()["result"]["highlights"][0]["review_status"] == "accepted"


def test_delete_removes_finished_job_and_files_but_rejects_active_job(tmp_path) -> None:
    client, settings, repository, _runner = make_client(tmp_path)

    with client:
        finished_dir = settings.jobs_dir / "job_finished1"
        finished_dir.mkdir(parents=True)
        (finished_dir / "agent.log").write_text("done", encoding="utf-8")
        repository.create(
            job_id="job_finished1",
            original_name="finished.mp4",
            stored_name="original.mp4",
            content_type="video/mp4",
            size_bytes=4,
            language="zh",
        )
        repository.set_status("job_finished1", "failed", error_message="failed")

        deleted = client.delete("/api/jobs/job_finished1")
        assert deleted.status_code == 204
        assert repository.get("job_finished1") is None
        assert not finished_dir.exists()

        active_dir = settings.jobs_dir / "job_active123"
        active_dir.mkdir(parents=True)
        repository.create(
            job_id="job_active123",
            original_name="active.mp4",
            stored_name="original.mp4",
            content_type="video/mp4",
            size_bytes=4,
            language="zh",
        )

        conflict = client.delete("/api/jobs/job_active123")
        assert conflict.status_code == 409
        assert repository.get("job_active123") is not None
        assert active_dir.exists()


def test_delete_reports_locked_job_without_removing_record(tmp_path, monkeypatch) -> None:
    client, settings, repository, _runner = make_client(tmp_path)

    with client:
        job_dir = settings.jobs_dir / "job_locked123"
        job_dir.mkdir(parents=True)
        repository.create(
            job_id="job_locked123",
            original_name="locked.mp4",
            stored_name="original.mp4",
            content_type="video/mp4",
            size_bytes=4,
            language="zh",
        )
        repository.set_status("job_locked123", "failed", error_message="failed")

        def locked_replace(_source, _target):
            raise PermissionError("locked")

        monkeypatch.setattr(main_module.time, "sleep", lambda _delay: None)
        monkeypatch.setattr(main_module.Path, "replace", locked_replace)

        response = client.delete("/api/jobs/job_locked123")
        assert response.status_code == 423
        assert response.json()["detail"] == "任务文件仍被播放器或其他程序占用，请等待几秒后重试"
        assert repository.get("job_locked123") is not None
        assert job_dir.exists()
