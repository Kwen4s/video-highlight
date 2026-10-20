"""One task reward: maximum-weight temporal matching, normalized by set sizes."""

import json
import math

import networkx as nx


def iou(left, right):
    intersection = max(
        0.0, min(left["end_sec"], right["end_sec"]) - max(left["start_sec"], right["start_sec"])
    )
    union = max(left["end_sec"], right["end_sec"]) - min(left["start_sec"], right["start_sec"])
    return intersection / union


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    truth = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    targets = truth["segments"]
    result = {
        "score": 0.0,
        "format_valid": 0,
        "invalid_segments": 0,
        "predicted_count": 0,
        "target_count": len(targets),
        "hits30": 0,
        "hits50": 0,
        "empty_positive": 0,
        "empty_negative_false_alarm": 0,
        "positive": int(bool(targets)),
    }
    try:
        spans = json.loads(solution_str)["segments"]
        if not isinstance(spans, list):
            raise TypeError("segments 必须是列表")
    except (ValueError, TypeError, KeyError):
        return result
    result["format_valid"] = 1
    result["predicted_count"] = len(spans)
    result["empty_positive"] = int(bool(targets) and not spans)
    result["empty_negative_false_alarm"] = int(not targets and bool(spans))
    valid = []
    for span in spans:
        try:
            start, end = span["start_sec"], span["end_sec"]
            if any(
                isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                for x in (start, end)
            ):
                raise ValueError("无效时间")
            midpoint = (start + end) / 2
            core = truth["detect_range"]
            if not (
                0 <= start < end <= truth["duration_sec"]
                and core["start_sec"] <= midpoint < core["end_sec"]
            ):
                raise ValueError("区间越界或不属于当前负责范围")
        except (ValueError, TypeError, KeyError, OverflowError):
            result["invalid_segments"] += 1
            continue
        valid.append(span)
    if not spans and not targets:
        result["score"] = 1.0
        return result
    graph = nx.Graph()
    for p, prediction in enumerate(valid):
        for g, target in enumerate(targets):
            weight = iou(prediction, target)
            if weight > 0:
                graph.add_edge(("p", p), ("g", g), weight=weight)
    pairs = nx.max_weight_matching(graph, maxcardinality=False)
    total = sum(graph.edges[p, g]["weight"] for p, g in pairs)
    denominator = len(spans) + len(targets)
    result["score"] = 2 * total / denominator if denominator else 0.0
    for threshold, key in ((0.3, "hits30"), (0.5, "hits50")):
        above = nx.Graph(
            (p, g, attrs) for p, g, attrs in graph.edges(data=True) if attrs["weight"] >= threshold
        )
        result[key] = len(nx.max_weight_matching(above, maxcardinality=True))
    return result
