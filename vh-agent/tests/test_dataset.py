import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets/test"


def load_manifest() -> list[dict]:
    return [
        json.loads(line)
        for line in (DATASET_ROOT / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_dataset_contains_short_and_long_real_videos() -> None:
    rows = load_manifest()

    assert len(rows) == 15
    assert len({row["video_id"] for row in rows}) == len(rows)
    assert {row["language"] for row in rows} == {"zh", "en"}
    assert {row["split"] for row in rows} == {"train", "test"}
    assert min(row["duration_sec"] for row in rows) < 60
    assert max(row["duration_sec"] for row in rows) >= 20 * 60
    assert sum(row["duration_sec"] for row in rows) >= 55 * 60
    assert all(Path(row["path"]).is_file() for row in rows)


def test_annotations_match_manifest() -> None:
    rows = load_manifest()
    annotations = json.loads((DATASET_ROOT / "annotations.json").read_text(encoding="utf-8"))

    assert {item["video_id"] for item in annotations} == {row["video_id"] for row in rows}
    assert {item["annotation_status"] for item in annotations} == {"labeled", "silver"}
    assert all(item["highlights"] for item in annotations)
    allowed_types = {
        "conflict",
        "reversal",
        "reveal",
        "payoff",
        "emotion",
        "action",
        "romance",
        "cliffhanger",
        "other",
    }
    for row in rows:
        item = next(value for value in annotations if value["video_id"] == row["video_id"])
        for highlight in item["highlights"]:
            assert 0 <= highlight["start_sec"] < highlight["end_sec"] <= row["duration_sec"]
            assert highlight["highlight_type"] in allowed_types
