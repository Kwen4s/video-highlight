import json
from types import SimpleNamespace

import pytest

from app.agent import AgentInvoker, AgentJobRunner
from app.config import Settings
from app.repository import JobRepository


@pytest.mark.parametrize("partial", [False, True])
def test_native_invocation_resumes_and_preserves_completion(tmp_path, monkeypatch, partial):
    settings = Settings(VH_STORAGE_DIR=tmp_path / "runtime", VH_AGENT_ROOT=tmp_path / "agent")
    settings.agent_root.mkdir()
    job = settings.jobs_dir / "job_abcdefgh"
    job.mkdir(parents=True)
    (job / "state.json").write_text("{}")
    (job / "task.json").write_text("{}")
    (job / "result.json").write_text(
        json.dumps(
            dict(
                schema_version="2.0",
                job_id=job.name,
                video=dict(video_id="v", title="demo", duration_sec=10),
                highlights=[],
                completion="partial" if partial else "complete",
                message="status",
                analysis=dict(
                    scan_coverage=0.5 if partial else 1,
                    pending_event_count=0,
                    pending_observation_count=0,
                    pending_proposal_count=0,
                    pending_review_count=0,
                    stop_reason="test",
                    model_calls=2,
                ),
            )
        )
    )

    def run(command, **kwargs):
        assert "--resume" in command and "--task-file" in command
        assert "VIRTUAL_ENV" not in kwargs["env"]
        return SimpleNamespace(returncode=2 if partial else 0)

    monkeypatch.setenv("VIRTUAL_ENV", "backend")
    monkeypatch.setattr("app.agent.subprocess.run", run)
    result = AgentInvoker(settings).run(
        job_id=job.name, video_path=job / "source.mp4", language="zh"
    )
    assert (result.completion == "partial") == partial


def test_failure_does_not_restart_whole_analysis(tmp_path):
    settings = Settings(VH_STORAGE_DIR=tmp_path)
    repository = JobRepository(settings.database_path)
    repository.initialize()
    repository.create(
        job_id="job_abcdefgh",
        original_name="a.mp4",
        stored_name="a.mp4",
        content_type="video/mp4",
        size_bytes=1,
        language="zh",
    )

    class Failing:
        calls = 0

        def run(self, **kwargs):
            self.calls += 1
            raise RuntimeError("service unavailable")

    invoker = Failing()
    runner = AgentJobRunner(settings, repository, invoker)
    try:
        runner._execute("job_abcdefgh")
    finally:
        runner.shutdown()
    assert invoker.calls == 1
    assert repository.get("job_abcdefgh")["status"] == "failed"
    assert repository.get("job_abcdefgh")["attempt"] == 1
