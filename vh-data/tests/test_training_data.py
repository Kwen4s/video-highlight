import hashlib
import json

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from vh_data.cli import app
from vh_data.models import Annotation, digest
from vh_data.sampling import select_gspo
from vh_data.store import ConflictError, Store
from vh_data.training_data import VLWindow, build_windows, prepare_training


def make_source(tmp_path):
    store = Store(tmp_path / "data")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"unique source")
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "video_id": "v1",
                "drama_id": "one-drama",
                "path": str(source),
                "duration_sec": 80,
                "language": "zh",
            }
        )
    )
    store.ingest([manifest])
    (store.root / "dialogues").mkdir()
    (store.root / "readings").mkdir()
    (store.root / "dialogues/v1.json").write_text(
        json.dumps(
            {
                "input": {"source_sha256": digest(source), "source_start_time_sec": 0},
                "segments": [
                    {"start_sec": 0, "end_sec": 2, "text": "前情", "source": "asr"},
                    {"start_sec": 18, "end_sec": 22, "text": "揭晓", "source": "ocr"},
                    {"start_sec": 56, "end_sec": 58, "text": "后续对白", "source": "asr"},
                ],
            },
            ensure_ascii=False,
        )
    )
    scan = {
        "input": {"pass": "scan"},
        "result": {
            "segments": [
                {"start_sec": 30, "end_sec": 34, "kind": "positive"},
                {"start_sec": 31, "end_sec": 35, "kind": "positive"},
            ]
        },
    }
    (store.root / "readings/scan.json").write_text(json.dumps(scan))
    labels = [
        (0, 18, "negative"),
        (0, 10, "negative"),
        (18, 22, "positive"),
        (22, 40, "negative"),
        (40, 42, "uncertain"),
        (42, 62, "negative"),
        (62, 64, "positive"),
        (64, 80, "negative"),
    ]
    store.save(
        "v1",
        Annotation(
            duration_sec=80,
            segments=[
                {
                    "start_sec": a,
                    "end_sec": b,
                    "kind": kind,
                    "reason": "标签理由不能进入输入",
                }
                for a, b, kind in labels
            ],
        ),
        "model_reviewed",
        {
            "dialogue": "dialogues/v1.json",
            "calls": ["readings/scan.json"],
            "review_calls": [],
            "model": "teacher",
        },
        expected_label_id=None,
    )
    return store


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_fixed_windows_preserve_ownership_unknown_regions_and_history(tmp_path):
    store = make_source(tmp_path)
    dest = tmp_path / "dataset"
    report = prepare_training(store, dest, gspo_size=1)
    rows = read_rows(dest / "sft/train.jsonl")
    assert report["window_counts"]["train"] == {
        "total": 4,
        "usable": 3,
        "empty": 1,
        "positive": 2,
        "unknown": 1,
    }
    first, second, last = [VLWindow.model_validate(r) for r in rows]
    assert first.segments == []  # The event centered exactly at 20 belongs to the next window.
    assert second.video.start_sec == 5
    assert second.detect_range.model_dump() == {"start_sec": 15, "end_sec": 35}
    assert [s.model_dump() for s in second.segments] == [{"start_sec": 13, "end_sec": 17}]
    assert second.dialogue[0].start_sec == -5
    assert "后续对白" not in [t.text for t in second.dialogue]
    assert "后续对白" in [
        t["text"] for t in read_rows(dest / "source/dialogues.jsonl")[0]["segments"]
    ]
    assert last.video.end_sec == 80
    assert [s.model_dump() for s in last.segments] == [{"start_sec": 17, "end_sec": 19}]
    assert "标签理由" not in (dest / "sft/train.jsonl").read_text()
    assert digest(dest / "sft/train.jsonl") == report["files"]["sft/train.jsonl"]
    assert report["sampling"]["focus_sample_ids"] == [second.sample_id]
    assert report["sampling"]["history_regions"] == 1
    assert read_rows(dest / "gspo/train.jsonl") == [second.model_dump()]
    assert not (dest / "train.jsonl").exists()
    assert not (dest / "review_queue.jsonl").exists()
    with pytest.raises(FileExistsError):
        prepare_training(store, dest, gspo_size=1)
    other = tmp_path / "repeat"
    report2 = prepare_training(store, other, gspo_size=1)
    assert report2["files"] == report["files"]


def test_incomplete_event_is_not_clipped_into_a_different_label(tmp_path):
    store = make_source(tmp_path)
    report = prepare_training(store, tmp_path / "dataset", context_sec=0, gspo_size=1)
    assert report["window_counts"]["train"]["incomplete_core"] == 1
    assert report["window_counts"]["train"]["usable"] == 2


@pytest.mark.parametrize("problem", ["source_changed", "dialogue_changed", "split_leak"])
def test_invalid_sources_abort_without_publishing_a_partial_dataset(tmp_path, problem):
    store = make_source(tmp_path)
    if problem == "source_changed":
        (tmp_path / "source.mp4").write_bytes(b"changed")
    elif problem == "dialogue_changed":
        p = store.root / "dialogues/v1.json"
        d = json.loads(p.read_text())
        d["input"]["source_sha256"] = "0" * 64
        p.write_text(json.dumps(d))
    else:
        original = store.get("v1")
        p = tmp_path / "different.mp4"
        p.write_bytes(b"different")
        with store.connect() as db:
            db.execute(
                "INSERT INTO videos VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "v2",
                    original["drama_id"],
                    str(p),
                    hashlib.sha256(p.read_bytes()).hexdigest(),
                    80,
                    "zh",
                    "test" if original["split"] != "test" else "train",
                    "ready",
                    original["label_id"],
                    0,
                    None,
                ),
            )
    destination = tmp_path / "bad"
    with pytest.raises(ConflictError):
        prepare_training(store, destination, gspo_size=1)
    assert not destination.exists()
    assert not list(tmp_path.glob(".training-*"))


def test_schema_rejects_future_dialogue_and_wrong_event_ownership(tmp_path):
    store = make_source(tmp_path)
    dest = tmp_path / "dataset"
    prepare_training(store, dest, gspo_size=1)
    row = read_rows(dest / "sft/train.jsonl")[1]
    row["segments"] = [{"start_sec": 40, "end_sec": 42}]
    with pytest.raises(ValidationError, match="归属"):
        VLWindow.model_validate(row)
    row["segments"] = []
    row["dialogue"] = [{"start_sec": 49, "end_sec": 51, "text": "未来信息", "source": "asr"}]
    with pytest.raises(ValidationError, match="之后"):
        VLWindow.model_validate(row)


def test_prepare_training_cli_uses_the_existing_store(tmp_path, monkeypatch):
    store = make_source(tmp_path)
    monkeypatch.setenv("VH_DATA_ROOT", str(store.root))
    result = CliRunner().invoke(
        app, ["prepare-training", str(tmp_path / "dataset"), "--gspo-size", "1"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["format"] == "vh-vl-training-v1"


@pytest.mark.parametrize("duration", [0.5, 1583.5])
def test_subsecond_highlight_and_short_tail_are_preserved(duration):
    version = {
        "label_id": 1,
        "annotation": {
            "duration_sec": duration,
            "segments": [
                {"start_sec": 0, "end_sec": 0.1, "kind": "negative", "reason": "普通"},
                {"start_sec": 0.1, "end_sec": 0.4, "kind": "positive", "reason": "关键动作"},
                {"start_sec": 0.4, "end_sec": duration, "kind": "negative", "reason": "普通"},
            ],
        },
    }
    windows, stats = build_windows(
        {"video_id": "short", "drama_id": "drama", "path": "/video.mp4", "sha256": "0" * 64},
        version,
        [],
        window_sec=20,
        context_sec=15,
        dialogue_history_sec=90,
    )
    assert [s.model_dump() for s in windows[0].segments] == [{"start_sec": 0.1, "end_sec": 0.4}]
    assert windows[-1].video.end_sec == duration
    assert stats["positive"] == 1


def test_gspo_selection_preserves_distribution_and_current_labels():
    rows = []
    for count, size in enumerate([12, 8, 4]):
        for _ in range(size):
            index = len(rows)
            rows.append(
                VLWindow.model_validate(
                    {
                        "sample_id": str(index),
                        "video_id": str(index),
                        "drama_id": str(index % 4),
                        "label_id": 1,
                        "video": {
                            "path": "/source.mp4",
                            "sha256": "0" * 64,
                            "start_sec": 0,
                            "end_sec": 40,
                        },
                        "detect_range": {"start_sec": 0, "end_sec": 20},
                        "dialogue": [],
                        "segments": [
                            {"start_sec": i * 5 + 1, "end_sec": i * 5 + 2} for i in range(count)
                        ],
                    }
                )
            )
    focus = {"0", "1", "12", "20"}
    selected = select_gspo(rows, focus, 8, 42)
    assert len(selected) == len({r.sample_id for r in selected}) == 8
    assert focus <= {r.sample_id for r in selected}
    assert [sum(len(r.segments) == n for r in selected) for n in range(3)] == [4, 3, 1]
    assert all(row is rows[int(row.sample_id)] for row in selected)
    assert selected == select_gspo(rows, focus, 8, 42)
    with pytest.raises(ValueError, match="配额"):
        select_gspo(rows, focus, 1, 42)
    with pytest.raises(ValueError, match="训练集之外"):
        select_gspo(rows, {"val"}, 8, 42)
