from types import SimpleNamespace

from app.agent import AgentInvoker
from app.config import Settings


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
