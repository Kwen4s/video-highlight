import json
from types import SimpleNamespace

import pytest

from vh_agent import evaluation
from vh_agent.evaluation import (
    matching_hits,
    run_evaluation,
    score_evaluation,
)
from vh_agent.storage import write_json


def test_matching_counts_each_annotation_only_once():
    target = [{"start_sec": 0, "end_sec": 10}]
    assert matching_hits(target * 3, target, 0.5) == 1
    assert matching_hits(target, target * 3, 0.5) == 1


@pytest.fixture
def dataset(tmp_path):
    folder = tmp_path / "dataset"
    folder.mkdir()
    video = folder / "video.mp4"
    video.write_bytes(b"video")
    rows = [
        {
            "video_id": f"v{i}",
            "title": f"Video {i}",
            "path": str(video),
            "language": "zh",
            "duration_sec": 20,
        }
        for i in range(3)
    ]
    (folder / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    labels = [
        {
            "video_id": f"v{i}",
            "annotation_status": "labeled" if i < 2 else "silver",
            "highlights": [{"start_sec": 0, "end_sec": 10}],
        }
        for i in range(3)
    ]
    (folder / "annotations.json").write_text(json.dumps(labels))
    return folder


@pytest.fixture
def fake_service(monkeypatch):
    calls = []

    class Service:
        def __init__(self, settings):
            self.settings = settings

        def detect(self, task):
            calls.append(task)
            span = {"start_sec": 0, "end_sec": 10}
            result = {
                "completion": "complete" if task.video_id == "v0" else "partial",
                "video": {"video_id": task.video_id},
                "highlights": [span, span],
                "analysis": {
                    "model_calls": 2,
                    "stop_reason": "complete" if task.video_id == "v0" else "max_turns",
                },
            }
            job = self.settings.job_output_dir / task.job_id
            write_json(job / "result.json", result)
            write_json(
                job / "state.json",
                {
                    "tools": {
                        "events": [{"required_spans": [span], "status": "supported"}],
                        "plans": [{**span, "status": "ready"}],
                    }
                },
            )
            return SimpleNamespace(
                completion=result["completion"], analysis=SimpleNamespace(**result["analysis"])
            )

    monkeypatch.setattr(evaluation, "HighlightDetectionService", Service)
    return calls


def test_frozen_labels_partial_denominator_and_offline_rescore(dataset, tmp_path, fake_service):
    output = tmp_path / "outputs"
    report = run_evaluation("test", dataset_dir=dataset, output_dir=output)
    assert report.exists()
    assert len(fake_service) == 2  # Silver is not evaluated by default.
    metrics = score_evaluation("test", output_dir=output)
    assert metrics["completion_rate"] == 0.5
    assert metrics["iou_metrics"]["0.50"]["precision"] == 0.5
    assert metrics["iou_metrics"]["0.50"]["recall"] == 0.5
    assert metrics["completed_only"]["0.50"]["recall"] == 1
    assert metrics["cohort_metrics"]["unspecified"]["videos"] == 2
    assert metrics["runtime_metrics"]["model_calls"] == 4
    assert metrics["runtime_metrics"]["invalid_tool_requests"] == 0
    assert metrics["iou_metrics"]["0.50"]["recall_interval_95"][0] < 0.5
    assert metrics["iou_metrics"]["0.50"]["recall_interval_95"][1] > 0.5
    assert metrics["details"][1]["predictions"] == []
    assert len(metrics["details"][1]["partial_predictions"]) == 2
    (dataset / "annotations.json").write_text("[]")
    assert score_evaluation("test", output_dir=output) == metrics
    assert "gemini_api_key" not in (output / "test/protocol.json").read_text()
    assert "openai_api_key" not in (output / "test/protocol.json").read_text()
    assert (output / "test/source/runtime/agent.py").exists()


@pytest.mark.parametrize("first_elapsed,expected", [(12.0, 15.0), (None, None)])
def test_runtime_preserves_failed_attempt_cost_and_unknown_durations(
    dataset, tmp_path, fake_service, first_elapsed, expected
):
    run_evaluation("timing", dataset_dir=dataset, output_dir=tmp_path, limit=1)
    write_json(
        tmp_path / "timing/jobs/v0/execution.json",
        {
            "attempts": [
                {"status": "partial", "elapsed_sec": first_elapsed},
                {"status": "complete", "elapsed_sec": 3.0},
            ]
        },
    )
    metrics = score_evaluation("timing", output_dir=tmp_path)
    assert metrics["details"][0]["runtime"]["elapsed_sec"] == expected
    assert metrics["runtime_metrics"]["video_elapsed_total_sec"] == expected
    assert metrics["runtime_metrics"]["first_attempt_completed"] == 0
    assert metrics["runtime_metrics"]["videos_with_multiple_attempts"] == 1
    assert metrics["runtime_metrics"]["timed_videos"] == (0 if expected is None else 1)


def test_resume_only_retries_unfinished_and_never_overwrites_run(dataset, tmp_path, fake_service):
    run_evaluation("run", dataset_dir=dataset, output_dir=tmp_path)
    with pytest.raises(ValueError, match="already exists"):
        run_evaluation("run", dataset_dir=dataset, output_dir=tmp_path)
    run_evaluation("run", output_dir=tmp_path, resume=True)
    assert [t.video_id for t in fake_service] == ["v0", "v1", "v1"]
    assert fake_service[-1].resume
    attempts = json.loads((tmp_path / "run/jobs/v1/execution.json").read_text())["attempts"]
    assert [attempt["stop_reason"] for attempt in attempts] == ["max_turns", "max_turns"]
    monkey = pytest.MonkeyPatch()
    monkey.setattr(evaluation, "_code", lambda: {"changed": "code"})
    try:
        with pytest.raises(ValueError, match="same code"):
            run_evaluation("run", output_dir=tmp_path, resume=True)
    finally:
        monkey.undo()


def test_failed_video_is_recorded_and_other_videos_continue(dataset, tmp_path, monkeypatch):
    class Broken:
        def __init__(self, settings):
            pass

        def detect(self, task):
            raise ValueError("bad media")

    monkeypatch.setattr(evaluation, "HighlightDetectionService", Broken)
    run_evaluation("run", dataset_dir=dataset, output_dir=tmp_path)
    metrics = score_evaluation("run", output_dir=tmp_path)
    assert metrics["completed"] == 0
    assert metrics["iou_metrics"]["0.50"]["targets"] == 2
    assert all(d["status"] == "failed" for d in metrics["details"])


def test_dataset_validation_rejects_duplicate_labels_and_invalid_optional_intervals(
    dataset, tmp_path, fake_service
):
    path = dataset / "annotations.json"
    labels = json.loads(path.read_text())
    path.write_text(json.dumps([*labels, labels[0]]))
    with pytest.raises(ValueError, match="duplicate video IDs"):
        run_evaluation("duplicates", dataset_dir=dataset, output_dir=tmp_path)

    labels[0]["optional_highlights"] = [{"start_sec": 19, "end_sec": 21}]
    path.write_text(json.dumps(labels))
    with pytest.raises(ValueError, match="Invalid annotation interval"):
        run_evaluation("interval", dataset_dir=dataset, output_dir=tmp_path)


def test_shared_constraints_are_saved_and_not_inferred_from_gold(dataset, tmp_path, fake_service):
    task = tmp_path / "task.json"
    task.write_text('{"max_clip_sec":30,"max_highlights":null}')
    run_evaluation("run", dataset_dir=dataset, output_dir=tmp_path, task_file=task, limit=1)
    protocol = json.loads((tmp_path / "run/protocol.json").read_text())
    assert len(protocol["items"]) == 1
    assert protocol["items"][0]["task"]["max_clip_sec"] == 30
    assert protocol["items"][0]["task"]["max_highlights"] is None
    assert score_evaluation("run", output_dir=tmp_path)["videos"] == 1


def test_event_scoring_alternatives_optional_and_duplicates():
    from vh_agent.evaluation import score_events

    required = [
        {"start_sec": 40, "end_sec": 52, "alternative_clips": [{"start_sec": 21, "end_sec": 52}]}
    ]
    optional = [{"start_sec": 60, "end_sec": 70}]
    predictions = [{"start_sec": 21, "end_sec": 52}, {"start_sec": 60, "end_sec": 70}]
    result = score_events(predictions, required, optional, 0.7)
    assert result["hits"] == result["optional_hits"] == 1
    assert result["precision"] == result["recall"] == 1
    assert result["output_count"] == 2
    result = score_events([*predictions, predictions[0]], required, optional, 0.7)
    assert result["hits"] == 1
    assert result["duplicate_count"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == 1


def test_recommended_matching_precedes_optional_and_does_not_waive_false_positives():
    from vh_agent.evaluation import score_events

    target = {"start_sec": 0, "end_sec": 10}
    false_positive = {"start_sec": 40, "end_sec": 50}
    result = score_events([target, false_positive], [target], [target], 0.5)
    assert result["hits"] == 1
    assert result["optional_hits"] == 0
    assert result["precision"] == 0.5
    assert result["duplicate_count"] == 0


def test_temporal_match_can_still_miss_part_of_core_evidence():
    from vh_agent.evaluation import score_events

    target = {
        "start_sec": 20,
        "end_sec": 40,
        "evidence": [{"start_sec": 25, "end_sec": 35, "role": "decisive"}],
    }
    result = score_events([{"start_sec": 20, "end_sec": 30}], [target], [], 0.5)
    assert result["hits"] == 1
    assert result["decisive_evidence_coverage"][0]["coverage"] == 0.5
    assert result["output_duration_sec"] == 10


def test_event_discovery_separates_core_evidence_from_boundary_iou():
    from vh_agent.evaluation import score_events

    target = {
        "start_sec": 38,
        "end_sec": 50,
        "evidence": [{"start_sec": 41, "end_sec": 50, "role": "decisive"}],
    }
    prediction = {"start_sec": 32, "end_sec": 60}
    assert matching_hits([prediction], [target], 0.5) == 0
    result = score_events([prediction], [target], [], 0.5)
    assert result["hits"] == 1
    assert result["matches"][0]["event_coverage"] == 1
    assert result["decisive_evidence_coverage"][0]["coverage"] == 1


def test_textual_evidence_does_not_block_temporal_scoring():
    from vh_agent.evaluation import score_events

    target = {
        "start_sec": 0,
        "end_sec": 10,
        "evidence": ["The older silver labels contain prose rather than timed evidence."],
    }
    result = score_events([{"start_sec": 0, "end_sec": 10}], [target], [], 0.5)
    assert result["hits"] == 1
    assert result["decisive_evidence_coverage"] == []
