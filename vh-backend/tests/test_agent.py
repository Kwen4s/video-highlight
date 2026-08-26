from types import SimpleNamespace

from app.agent import AgentInvoker, AgentJobRunner
from app.config import Settings
from app.models import AgentDetectionResult
from app.repository import JobRepository


def test_agent_invocation_does_not_leak_backend_virtualenv(tmp_path, monkeypatch) -> None:
    settings = Settings(
        VH_STORAGE_DIR=tmp_path / "runtime",
        VH_AGENT_ROOT=tmp_path / "vh-agent",
    )
    settings.agent_root.mkdir()
    job_dir = settings.jobs_dir / "job_abcdefgh"
    job_dir.mkdir(parents=True)
    video_path = job_dir / "source.mp4"
    video_path.write_bytes(b"video")
    (job_dir / "result.json").write_text(
        '{"job_id":"job_abcdefgh","video":{"video_id":"job_abcdefgh",'
        '"title":"demo","duration_sec":1},"highlights":[]}',
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_run(command, *, cwd, env, **kwargs):
        captured.update(command=command, cwd=cwd, env=env, kwargs=kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "backend-venv"))
    monkeypatch.setattr("app.agent.subprocess.run", fake_run)

    result = AgentInvoker(settings).run(
        job_id="job_abcdefgh",
        video_path=video_path,
        language="zh",
    )

    assert result.job_id == "job_abcdefgh"
    assert "VIRTUAL_ENV" not in captured["env"]
    assert captured["cwd"] == settings.agent_root


class FlakyInvoker:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def run(self, *, job_id, video_path, language):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("temporary failure")
        return AgentDetectionResult.model_validate(
            {
                "job_id": job_id,
                "video": {"video_id": job_id, "title": "demo", "duration_sec": 10},
                "highlights": [],
            }
        )


def make_retry_runner(tmp_path, failures: int):
    settings = Settings(
        VH_STORAGE_DIR=tmp_path / "runtime",
        VH_AGENT_ROOT=tmp_path / "vh-agent",
        VH_AGENT_RETRY_DELAY_SEC=0,
        VH_AGENT_MAX_ATTEMPTS=3,
    )
    repository = JobRepository(settings.database_path)
    repository.initialize()
    source_dir = settings.jobs_dir / "job_abcdefgh" / "source"
    source_dir.mkdir(parents=True)
    (source_dir / "original.mp4").write_bytes(b"video")
    repository.create(
        job_id="job_abcdefgh",
        original_name="demo.mp4",
        stored_name="original.mp4",
        content_type="video/mp4",
        size_bytes=5,
        language="zh",
        max_attempts=3,
    )
    invoker = FlakyInvoker(failures)
    runner = AgentJobRunner(settings, repository, invoker=invoker)
    return runner, repository, invoker


def test_agent_job_retries_until_third_attempt_succeeds(tmp_path) -> None:
    runner, repository, invoker = make_retry_runner(tmp_path, failures=2)
    try:
        runner._execute("job_abcdefgh")
    finally:
        runner.shutdown()

    job = repository.get("job_abcdefgh")
    assert invoker.calls == 3
    assert job["status"] == "completed"
    assert job["attempt"] == 3
    assert job["error_message"] is None


def test_agent_job_stops_after_three_failed_attempts(tmp_path) -> None:
    runner, repository, invoker = make_retry_runner(tmp_path, failures=3)
    try:
        runner._execute("job_abcdefgh")
    finally:
        runner.shutdown()

    job = repository.get("job_abcdefgh")
    assert invoker.calls == 3
    assert job["status"] == "failed"
    assert job["attempt"] == 3
    assert "3/3" in job["error_message"]
