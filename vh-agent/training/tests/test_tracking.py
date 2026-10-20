"""Training telemetry preserves resumes, partial writes and validation updates."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "training_track", Path(__file__).resolve().parents[1] / "track.py"
)
tracking = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracking)


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_resumed_history_and_same_step_evaluation(tmp_path):
    path = tmp_path / "metrics.jsonl"
    write_rows(
        path,
        [
            {"step": 1, "epoch": 0.1, "loss": 0.5, "elapsed_sec": 10},
            {"step": 2, "epoch": 0.2, "loss": 0.4, "elapsed_sec": 20},
            {"step": 1, "epoch": 0.1, "loss": 0.3, "elapsed_sec": 40},
            {"step": 2, "epoch": 0.2, "loss": 0.2, "elapsed_sec": 45},
            {"step": 2, "epoch": 0.2, "eval_loss": 0.25, "elapsed_sec": 50},
        ],
    )
    history = tracking.read_metrics(path, 100)
    assert list(history) == [1, 2]
    assert history[1]["train/loss"] == 0.3
    assert "speed/seconds_per_update" not in history[1]
    assert history[2]["speed/seconds_per_update"] == 5
    assert history[2]["speed/windows_per_second"] == pytest.approx(2)
    assert history[2]["train/loss"] == 0.2
    assert history[2]["validation/loss"] == 0.25


def test_partial_line_and_incremental_upload(tmp_path):
    class Run:
        def __init__(self):
            self.logs = []

        def log(self, values, step):
            self.logs.append((step, values))

    path = tmp_path / "metrics.jsonl"
    write_rows(path, [{"step": 1, "loss": 0.3, "epoch": 0.5, "elapsed_sec": 5}])
    with path.open("a") as stream:
        stream.write('{"step": 2')
    run, sent = Run(), {}
    assert tracking.sync_once(run, path, 20, sent) == 1
    tracking.sync_once(run, path, 20, sent)
    assert len(run.logs) == 1
    with path.open("a") as stream:
        stream.write(', "loss": 0.2, "epoch": 1.0, "elapsed_sec": 15}\n')
    assert tracking.sync_once(run, path, 20, sent) == 2
    assert len(run.logs) == 2
    assert run.logs[-1][1]["speed/windows_per_second"] == 1


def test_short_final_batch_throughput(tmp_path):
    path = tmp_path / "metrics.jsonl"
    write_rows(
        path,
        [
            {"step": 298, "epoch": 4768 / 4778, "loss": 0.3, "elapsed_sec": 10},
            {"step": 299, "epoch": 1, "loss": 0.2, "elapsed_sec": 15},
        ],
    )
    history = tracking.read_metrics(path, 4778)
    assert history[299]["speed/windows_per_second"] == pytest.approx(2)


def test_test_metrics_keep_final_validation_and_training(tmp_path):
    path = tmp_path / "metrics.jsonl"
    write_rows(
        path,
        [
            {"step": 299, "loss": 0.3, "epoch": 1, "elapsed_sec": 10},
            {"step": 299, "eval_loss": 0.25, "epoch": 1, "elapsed_sec": 20},
            {"step": 299, "test_loss": 0.27, "test_runtime": 30},
            {"step": 299, "effect_recall": 0.4, "effect_precision": 0.5},
        ],
    )
    values = tracking.read_metrics(path, 4778)[299]
    assert values["train/loss"] == 0.3
    assert values["validation/loss"] == 0.25
    assert values["test/loss"] == 0.27
    assert values["test/seconds"] == 30
    assert values["time/training_seconds"] == 20
    assert values["effect/recall_iou50"] == 0.4
    assert values["effect/precision_iou50"] == 0.5


def test_gspo_metrics_merge_attempts_and_keep_only_numeric_allowlist(tmp_path):
    folder = tmp_path / "metrics"
    folder.mkdir()
    write_rows(
        folder / "001.jsonl",
        [
            {"step": 0, "data": {"validation/f1_iou50": 0.4}},
            {"step": 1, "data": {"sampling/groups_with_signal": 0.75}},
            {
                "step": 1,
                "data": {
                    "actor/pg_loss": -0.1,
                    "config/path": "/private",
                    "validation/text": "private",
                    "validation/bad": float("nan"),
                },
            },
        ],
    )
    write_rows(folder / "002.jsonl", [{"step": 1, "data": {"actor/pg_loss": -0.2}}])
    with (folder / "002.jsonl").open("a") as stream:
        stream.write('{"step": 2')
    values = tracking.read_gspo_metrics(tmp_path)
    assert values == {
        0: {"validation/f1_iou50": 0.4},
        1: {"sampling/groups_with_signal": 0.75, "actor/pg_loss": -0.2},
    }
