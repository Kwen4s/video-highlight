"""Score raw model intervals and build a browsable, offline video review."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median

import networkx as nx
from data import dialogue_text, read_rows

THRESHOLDS = (0.3, 0.5, 0.7)
CORE_COVERAGE = 0.8


def parse_prediction(text):
    try:
        value = json.loads(text)
        segments = value["segments"]
        if not isinstance(segments, list):
            raise TypeError("segments 不是列表")
        for span in segments:
            start, end = span["start_sec"], span["end_sec"]
            if (
                any(
                    isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x)
                    for x in (start, end)
                )
                or start >= end
            ):
                raise ValueError("时间不是有限数值或起止颠倒")
        return segments, None
    except (ValueError, KeyError, TypeError, OverflowError) as error:
        return [], str(error)


def temporal_iou(left, right):
    overlap = max(
        0, min(left["end_sec"], right["end_sec"]) - max(left["start_sec"], right["start_sec"])
    )
    union = max(left["end_sec"], right["end_sec"]) - min(left["start_sec"], right["start_sec"])
    return overlap / union if union > 0 else 0


def reference_coverage(prediction, target):
    overlap = max(
        0,
        min(prediction["end_sec"], target["end_sec"])
        - max(prediction["start_sec"], target["start_sec"]),
    )
    return overlap / (target["end_sec"] - target["start_sec"])


def match_segments(predictions, targets, threshold, similarity=temporal_iou):
    graph = nx.Graph()
    for i, predicted in enumerate(predictions):
        for j, target in enumerate(targets):
            iou = similarity(predicted, target)
            if iou >= threshold and iou > 0:
                graph.add_edge(("p", i), ("g", j), weight=iou)
    matched = nx.max_weight_matching(graph, maxcardinality=True)
    pairs = []
    for left, right in matched:
        p, g = (left, right) if left[0] == "p" else (right, left)
        pairs.append(
            {
                "prediction": p[1],
                "target": g[1],
                "iou": temporal_iou(predictions[p[1]], targets[g[1]]),
                "coverage": reference_coverage(predictions[p[1]], targets[g[1]]),
            }
        )
    return sorted(pairs, key=lambda pair: pair["prediction"])


def rates(hits, predicted, targets):
    precision = hits / predicted if predicted else 0
    recall = hits / targets if targets else 0
    return {
        "hits": hits,
        "predictions": predicted,
        "targets": targets,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
    }


def covered_seconds(segments, core):
    intervals = sorted(
        (max(s["start_sec"], core["start_sec"]), min(s["end_sec"], core["end_sec"]))
        for s in segments
    )
    covered, end = 0, core["start_sec"]
    for start, stop in intervals:
        if stop > max(start, end):
            covered += stop - max(start, end)
            end = stop
    return covered


def score_rows(rows, records):
    predicted = {item["sample_id"]: item for item in records}
    expected = {row["sample_id"] for row in rows}
    if set(predicted) != expected or len(predicted) != len(records):
        raise ValueError("预测必须完整覆盖当前测试集，且每个窗口只有一条")
    details = []
    for row in rows:
        record = predicted[row["sample_id"]]
        segments, error = parse_prediction(record["text"])
        matches = {str(t): match_segments(segments, row["segments"], t) for t in THRESHOLDS}
        pairs = match_segments(segments, row["segments"], CORE_COVERAGE, reference_coverage)
        missed = [i for i in range(len(row["segments"])) if i not in {m["target"] for m in pairs}]
        false = [i for i in range(len(segments)) if i not in {m["prediction"] for m in pairs}]
        if error:
            category = "输出错误"
        elif missed and false:
            category = "漏选和误选"
        elif missed:
            category = "漏选"
        elif false:
            category = "误选"
        elif pairs and len(matches["0.5"]) < len(pairs):
            category = "边界偏差"
        else:
            category = "命中" if pairs else "正确空窗"
        duration = row["video"]["end_sec"] - row["video"]["start_sec"]
        details.append(
            {
                "sample_id": row["sample_id"],
                "video_id": row["video_id"],
                "drama_id": row["drama_id"],
                "video": row["video"],
                "detect_range": row["detect_range"],
                "reference": row["segments"],
                "predicted": segments,
                "matches": matches,
                "coverage_matches": pairs,
                "missed": missed,
                "false": false,
                "category": category,
                "error": error,
                "out_of_bounds": sum(
                    s["start_sec"] < 0 or s["end_sec"] > duration for s in segments
                ),
                "out_of_scope": sum(
                    not row["detect_range"]["start_sec"]
                    <= (s["start_sec"] + s["end_sec"]) / 2
                    < row["detect_range"]["end_sec"]
                    for s in segments
                ),
                "raw_text": record["text"],
                "truncated": record["truncated"],
                "generation_sec": record["generation_sec"],
                "dialogue": dialogue_text(row),
            }
        )
    count = len(details)
    n_pred = sum(len(d["predicted"]) for d in details)
    n_ref = sum(len(d["reference"]) for d in details)
    metrics = {
        "windows": count,
        "videos": len({r["video_id"] for r in rows}),
        "dramas": len({r["drama_id"] for r in rows}),
    }
    metrics["iou"] = {
        str(t): rates(sum(len(d["matches"][str(t)]) for d in details), n_pred, n_ref)
        for t in THRESHOLDS
    }
    metrics["core_capture"] = rates(sum(len(d["coverage_matches"]) for d in details), n_pred, n_ref)
    lengths = [s["end_sec"] - s["start_sec"] for d in details for s in d["predicted"]]
    metrics["predicted_median_seconds"] = median(lengths) if lengths else None
    metrics["predicted_max_seconds"] = max(lengths) if lengths else None
    reference_lengths = [s["end_sec"] - s["start_sec"] for d in details for s in d["reference"]]
    metrics["reference_median_seconds"] = median(reference_lengths) if reference_lengths else None
    empty = [d for d in details if not d["reference"]]
    metrics.update(
        {
            "valid_output_rate": sum(d["error"] is None for d in details) / count,
            "invalid_outputs": sum(d["error"] is not None for d in details),
            "truncated_outputs": sum(d["truncated"] for d in details),
            "empty_windows": len(empty),
            "empty_window_false_positive_rate": sum(bool(d["predicted"]) for d in empty)
            / len(empty)
            if empty
            else 0,
            "empty_window_correct_rate": sum(
                not d["predicted"] and d["error"] is None for d in empty
            )
            / len(empty)
            if empty
            else 0,
            "out_of_bounds_segments": sum(d["out_of_bounds"] for d in details),
            "out_of_scope_segments": sum(d["out_of_scope"] for d in details),
            "mean_generation_sec": mean(d["generation_sec"] for d in details),
            "median_generation_sec": median(d["generation_sec"] for d in details),
        }
    )
    all_pairs = [(d, m) for d in details for m in d["matches"]["0.5"]]
    metrics["matched_mean_iou"] = mean(m["iou"] for _, m in all_pairs) if all_pairs else None
    for key in ("start_sec", "end_sec"):
        metrics[f"matched_{key}_mae"] = (
            mean(
                abs(d["predicted"][m["prediction"]][key] - d["reference"][m["target"]][key])
                for d, m in all_pairs
            )
            if all_pairs
            else None
        )
    core_seconds = sum(r["detect_range"]["end_sec"] - r["detect_range"]["start_sec"] for r in rows)
    metrics["predictions_per_minute"] = n_pred / (core_seconds / 60)
    metrics["predicted_duration_fraction"] = (
        sum(covered_seconds(d["predicted"], d["detect_range"]) for d in details) / core_seconds
    )
    metrics["reference_duration_fraction"] = (
        sum(covered_seconds(d["reference"], d["detect_range"]) for d in details) / core_seconds
    )
    metrics["by_drama"] = {
        drama: rates(
            sum(len(d["matches"]["0.5"]) for d in details if d["drama_id"] == drama),
            sum(len(d["predicted"]) for d in details if d["drama_id"] == drama),
            sum(len(d["reference"]) for d in details if d["drama_id"] == drama),
        )
        for drama in sorted({r["drama_id"] for r in rows})
    }
    return {"metrics": metrics, "details": details}


def build_report(output):
    metadata = json.loads((output / "run.json").read_text())
    rows = read_rows(Path(metadata["dataset"]), "test.jsonl")
    folder = output / "effect"
    records = [
        json.loads(line)
        for path in sorted(folder.glob("predictions-*.jsonl"))
        for line in path.read_text().splitlines()
        if line
    ]
    report = score_rows(rows, records)
    report["protocol"] = json.loads((folder / "protocol.json").read_text())
    from examples import export_examples

    export_examples(report, folder)
    media = folder / "media"
    media.mkdir(exist_ok=True)
    for row in rows:
        link = media / f"{row['video_id']}.mp4"
        if not link.is_symlink():
            link.symlink_to(row["video"]["path"])
    (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    template = Path(__file__).with_name("templates") / "effect.html"
    data = json.dumps(report, ensure_ascii=False).replace("<", "\\u003c")
    (folder / "index.html").write_text(template.read_text().replace("__REPORT_DATA__", data))
    metrics = report["metrics"]
    numeric = {f"effect_{key}": metrics["iou"]["0.5"][key] for key in ("precision", "recall", "f1")}
    for key in ("precision", "recall"):
        numeric[f"effect_{key}_iou30"] = metrics["iou"]["0.3"][key]
    numeric["effect_core_recall"] = metrics["core_capture"]["recall"]
    numeric.update(
        {
            f"effect_{key}": metrics[key]
            for key in (
                "valid_output_rate",
                "empty_window_false_positive_rate",
                "predicted_duration_fraction",
            )
        }
    )
    if metadata["stage"] == "gspo":
        from track import METRICS

        step = json.loads((output / "completed.json").read_text())["best"]["step"]
        values = {METRICS[key]: value for key, value in numeric.items()}
        (output / "metrics/effect.jsonl").write_text(
            json.dumps({"step": step, "data": values}) + "\n"
        )
    else:
        step = json.loads((output / "trainer_state.json").read_text())["global_step"]
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": step, **numeric}) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    build_report(parser.parse_args().output.resolve())
