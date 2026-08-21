import json

from vh_agent.evaluation import decision_segments, run_evaluation, score_evaluation
from vh_agent.models import DetectionResult, VideoSummary


def test_decision_segments_reads_scene_bounds() -> None:
    trace = {
        "reasoning": {
            "decisions": [
                {
                    "scene": {"start_sec": 12.0, "end_sec": 24.0},
                    "decision": {"is_highlight": True, "start_sec": None, "end_sec": None},
                }
            ]
        }
    }

    assert decision_segments(trace, "decision") == [{"start_sec": 12.0, "end_sec": 24.0}]


def test_score_evaluation_reads_scene_trace_contract(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    run_dir = tmp_path / "outputs" / "scene-contract"
    jobs_dir = run_dir / "jobs" / "video_1"
    dataset_dir.mkdir()
    jobs_dir.mkdir(parents=True)
    (dataset_dir / "manifest.jsonl").write_text(
        json.dumps(
            {
                "video_id": "video_1",
                "title": "Episode 1",
                "path": "/tmp/video.mp4",
                "language": "zh",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "annotations.json").write_text(
        json.dumps(
            [
                {
                    "video_id": "video_1",
                    "annotation_status": "labeled",
                    "highlights": [{"start_sec": 10.0, "end_sec": 20.0}],
                }
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "predictions.jsonl").write_text(
        json.dumps({"video": {"video_id": "video_1"}, "highlights": []}) + "\n",
        encoding="utf-8",
    )
    (jobs_dir / "trace.json").write_text(
        json.dumps(
            {
                "local_candidates": [{"start_sec": 10.0, "end_sec": 20.0}],
                "reasoning": {
                    "decisions": [
                        {
                            "scene": {"start_sec": 10.0, "end_sec": 20.0},
                            "decision": {
                                "is_highlight": True,
                                "start_sec": None,
                                "end_sec": None,
                            },
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    metrics = score_evaluation(
        "scene-contract", dataset_dir=dataset_dir, output_dir=tmp_path / "outputs"
    )

    assert metrics["iou_metrics"]["0.50"]["candidate_recall"] == 1.0
    assert metrics["iou_metrics"]["0.50"]["verification_recall"] == 1.0


def test_run_evaluation_records_one_failure_and_continues(tmp_path, monkeypatch) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    rows = [
        {"video_id": "bad", "title": "Bad", "path": "/tmp/bad.mp4", "language": "zh"},
        {"video_id": "good", "title": "Good", "path": "/tmp/good.mp4", "language": "zh"},
    ]
    (dataset_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    fail_bad = True

    class FakeService:
        def __init__(self, _settings) -> None:
            pass

        def detect(self, task):
            if task.video_id == "bad" and fail_bad:
                raise RuntimeError("blocked")
            return DetectionResult(
                job_id=task.job_id,
                video=VideoSummary(
                    video_id=task.video_id,
                    title=task.video_id.title(),
                    duration_sec=10,
                ),
                highlights=[],
            )

    monkeypatch.setattr("vh_agent.evaluation.HighlightDetectionService", FakeService)
    output_dir = tmp_path / "outputs"
    predictions = run_evaluation("run", dataset_dir=dataset_dir, output_dir=output_dir)

    load = json.loads(predictions.read_text(encoding="utf-8").strip())
    assert load["video"]["video_id"] == "good"
    error = json.loads((output_dir / "run" / "errors.jsonl").read_text().strip())
    assert error["video_id"] == "bad"

    fail_bad = False
    resumed = run_evaluation(
        "run",
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        resume=True,
    )
    resumed_ids = {
        item["video"]["video_id"]
        for item in (
            json.loads(line) for line in resumed.read_text(encoding="utf-8").splitlines() if line
        )
    }
    assert resumed_ids == {"bad", "good"}
    assert (output_dir / "run" / "errors.jsonl").read_text() == ""
