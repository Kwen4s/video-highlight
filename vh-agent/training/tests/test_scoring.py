"""Effect metrics count duplicates, malformed outputs and missed windows honestly."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predict import committed_records
from score import covered_seconds, match_segments, parse_prediction, reference_coverage, score_rows


def span(a, b):
    return {"start_sec": a, "end_sec": b}


def test_one_to_one_maximum_cardinality_prevents_double_credit():
    assert len(match_segments([span(0, 10)] * 3, [span(0, 10)], 0.5)) == 1
    assert len(match_segments([span(0, 10)], [span(0, 10)] * 3, 0.5)) == 1
    # The first prediction fits either reference; the second fits only the first.
    result = match_segments([span(0, 20), span(0, 8)], [span(0, 10), span(10, 20)], 0.3)
    assert len(result) == 2


@pytest.mark.parametrize(
    "text",
    [
        "bad",
        '{"segments":[{"start_sec":true,"end_sec":2}]}',
        '{"segments":[{"start_sec":NaN,"end_sec":2}]}',
        '{"segments":[{"start_sec":2,"end_sec":1}]}',
    ],
)
def test_invalid_outputs_are_not_silently_repaired(text):
    parsed, error = parse_prediction(text)
    assert parsed == [] and error


def test_empty_errors_and_duplicates_keep_correct_denominators():
    rows = []
    for i, reference in enumerate([[span(1, 3)], [], [span(4, 6)], []]):
        rows.append(
            {
                "sample_id": str(i),
                "video_id": str(i),
                "drama_id": "d",
                "video": {"path": "video.mp4", "start_sec": 0, "end_sec": 10},
                "detect_range": span(0, 10),
                "segments": reference,
                "dialogue": [],
            }
        )
    answers = [[span(1, 3)] * 2, [span(5, 7)], [], None]
    records = [
        {
            "sample_id": str(i),
            "text": json.dumps({"segments": a}) if a is not None else "bad",
            "generation_sec": 1,
            "truncated": False,
        }
        for i, a in enumerate(answers)
    ]
    report = score_rows(rows, records)
    m = report["metrics"]
    assert m["iou"]["0.5"]["hits"] == 1
    assert m["iou"]["0.5"]["predictions"] == 3
    assert m["iou"]["0.5"]["targets"] == 2
    assert m["iou"]["0.5"]["precision"] == pytest.approx(1 / 3)
    assert m["iou"]["0.5"]["recall"] == 0.5
    assert m["valid_output_rate"] == 0.75
    assert m["empty_window_false_positive_rate"] == 0.5
    assert m["empty_window_correct_rate"] == 0
    with pytest.raises(ValueError, match="完整覆盖"):
        score_rows(rows, records[:-1])


def test_duration_counts_union_inside_detection_range():
    assert covered_seconds([span(0, 6), span(4, 15)], span(2, 10)) == 8
    assert covered_seconds([span(20, 22)], span(2, 10)) == 0


def test_interrupted_tail_is_removed_without_repeating_completed_record(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text('{"sample_id":"完成"}\n{"sample_id":')
    with path.open("a+") as stream:
        records = committed_records(stream)
        stream.write('{"sample_id":"下一条"}\n')
    assert records == [{"sample_id": "完成"}]
    assert [json.loads(line) for line in path.read_text().splitlines()] == records + [
        {"sample_id": "下一条"}
    ]


def test_core_inside_overlong_prediction_is_not_a_missing_event():
    target, prediction = span(27, 31.5), span(20.5, 36)
    assert match_segments([prediction], [target], 0.3) == []
    matches = match_segments([prediction], [target], 0.8, reference_coverage)
    assert len(matches) == 1 and matches[0]["coverage"] == 1
    assert matches[0]["iou"] < 0.3


def test_real_inference_input_cannot_see_reference_answers():
    import torch
    from data import InputConfig, VideoInputs, read_rows
    from predict import PromptInputs
    from transformers import AutoProcessor

    root = Path("/data0/wkw/video-highlight")
    processor = AutoProcessor.from_pretrained(root / "models/Qwen3.5-9B", local_files_only=True)
    builder = PromptInputs(VideoInputs(processor, InputConfig(visual_tokens=128)))
    row = read_rows(root / "datasets/vl-001", "test.jsonl")[0]
    row["video"].update(start_sec=0, end_sec=2)
    row["detect_range"] = span(0, 2)
    row["dialogue"] = []
    row["segments"] = [span(0.12345, 1.98765)]
    _, first = builder(row)
    row["segments"] = []
    _, other = builder(row)
    assert "labels" not in first and "shift_labels" not in first
    assert first.keys() == other.keys()
    assert all(torch.equal(value, other[key]) for key, value in first.items())
