"""The backend uses the CLI exit status to distinguish failure from saved results."""

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from vh_agent import cli
from vh_agent.models import DetectionResult


@pytest.mark.parametrize(
    "completion,stop_reason,exit_code",
    [
        ("complete", "complete", 0),
        ("partial", "execution_error", 1),
        ("partial", "user_stopped", 2),
    ],
)
def test_saved_result_exit_status(tmp_path, monkeypatch, completion, stop_reason, exit_code):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    tasks = []

    class Service:
        def __init__(self, *args, **kwargs):
            pass

        def detect(self, task):
            tasks.append(task)
            result = DetectionResult(
                job_id=task.job_id,
                video={"video_id": "video", "title": source.name, "duration_sec": 10},
                completion=completion,
                message="saved",
                highlights=[],
                analysis={
                    "scan_coverage": 1 if completion == "complete" else 0,
                    "pending_event_count": 0,
                    "pending_observation_count": 0,
                    "pending_proposal_count": 0,
                    "pending_review_count": 0,
                    "stop_reason": stop_reason,
                    "model_calls": 3,
                },
            )
            output = tmp_path / task.job_id
            output.mkdir()
            (output / "state.json").write_text('{"checkpoint":"saved"}')
            (output / "result.json").write_text(result.model_dump_json())
            return result

    monkeypatch.setattr(cli, "Settings", lambda: SimpleNamespace(job_output_dir=tmp_path))
    monkeypatch.setattr(cli, "HighlightDetectionService", Service)
    result = CliRunner().invoke(cli.app, ["run", str(source), "--job-id", "test", "--resume"])
    assert result.exit_code == exit_code, result.output
    assert len(tasks) == 1 and tasks[0].resume
    assert (tmp_path / "test" / "state.json").is_file()
    saved = DetectionResult.model_validate_json((tmp_path / "test" / "result.json").read_text())
    assert saved.completion == completion and saved.analysis.stop_reason == stop_reason
