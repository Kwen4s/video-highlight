"""The interval reward must favor useful sets rather than empty or long answers."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reward import compute_score


def span(start, end):
    return {"start_sec": start, "end_sec": end}


def reward(predictions, targets):
    truth = {"segments": targets, "duration_sec": 50, "detect_range": span(0, 50)}
    return compute_score("short_drama", json.dumps({"segments": predictions}), truth)


def test_missing_extra_duplicate_and_long_intervals_reduce_reward():
    targets = [span(5, 10), span(20, 25)]
    assert reward(targets, targets)["score"] == 1
    assert reward(targets[:1], targets)["score"] == pytest.approx(2 / 3)
    assert reward(targets + [span(35, 40)], targets)["score"] == pytest.approx(4 / 5)
    assert reward(targets + targets, targets)["score"] == pytest.approx(2 / 3)
    assert reward([span(0, 50)], targets)["score"] == pytest.approx(1 / 15)
    assert reward([], targets)["score"] == 0


def test_continuous_boundary_credit_and_empty_windows():
    assert reward([span(5, 9)], [span(5, 10)])["score"] == pytest.approx(0.8)
    assert reward([span(5, 10)], [span(5, 10)])["hits50"] == 1
    assert reward([], [])["score"] == 1
    assert reward([span(5, 10)], [])["score"] == 0
    assert reward([], [span(5, 10)])["empty_positive"] == 1
    assert reward([span(5, 10)], [])["empty_negative_false_alarm"] == 1


@pytest.mark.parametrize("bad", [None, {}, span(-1, 10), span(1, 51), span(True, 5), span(5, 5)])
def test_invalid_intervals_remain_in_denominator(bad):
    result = reward([span(5, 10), bad], [span(5, 10)])
    assert result["score"] == pytest.approx(2 / 3)
    assert result["invalid_segments"] == 1


@pytest.mark.parametrize("text", ["null", "[]", "{}", '{"segments":null}', "not JSON"])
def test_malformed_empty_answer_never_gets_negative_window_reward(text):
    result = compute_score("short_drama", text, {"segments": []})
    assert result["score"] == 0 and result["format_valid"] == 0


def test_responsibility_range_excludes_neighboring_event():
    truth = {"segments": [span(20, 25)], "duration_sec": 50, "detect_range": span(15, 35)}
    result = compute_score(
        "short_drama", json.dumps({"segments": [span(20, 25), span(2, 5)]}), truth
    )
    assert result["score"] == pytest.approx(2 / 3)
    assert result["invalid_segments"] == 1
