"""Frozen evaluation runs, resumable execution and offline temporal scoring."""

import hashlib
import json
import re
import time
from datetime import UTC, datetime
from enum import StrEnum
from itertools import combinations
from math import ceil, isfinite, sqrt
from pathlib import Path
from statistics import median

from .config import Settings
from .models import DetectionTask
from .pipeline import HighlightDetectionService
from .storage import read_jsonl, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "datasets" / "test"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluations"
THRESHOLDS = (0.3, 0.5, 0.7)


class ComparisonKind(StrEnum):
    repeat = "repeat"
    revision = "revision"


def segment_iou(left: dict, right: dict) -> float:
    overlap = max(
        0.0, min(left["end_sec"], right["end_sec"]) - max(left["start_sec"], right["start_sec"])
    )
    union = max(left["end_sec"], right["end_sec"]) - min(left["start_sec"], right["start_sec"])
    return overlap / union if union else 0.0


def matching_pairs(predicted, targets, threshold, similarity=segment_iou, score_key="iou"):
    """Maximum cardinality one-to-one matching; prefer higher IoU during traversal."""
    matches = {}

    def augment(index, seen):
        for j in sorted(
            range(len(targets)), key=lambda j: -similarity(predicted[index], targets[j])
        ):
            if j in seen or similarity(predicted[index], targets[j]) < threshold:
                continue
            seen.add(j)
            if j not in matches or augment(matches[j], seen):
                matches[j] = index
                return True
        return False

    for i in range(len(predicted)):
        augment(i, set())
    return [
        {"prediction": i, "target": j, score_key: similarity(predicted[i], targets[j])}
        for j, i in sorted(matches.items())
    ]


def matching_hits(predicted, targets, threshold):
    return len(matching_pairs(predicted, targets, threshold))


def event_iou(prediction, target):
    return max(
        segment_iou(prediction, clip) for clip in [target, *target.get("alternative_clips", [])]
    )


def event_coverage(prediction, target):
    """Event discovery score: accepted-boundary IoU or retained decisive evidence."""
    evidence_coverage = decisive_evidence_coverage(prediction, target)
    return max(event_iou(prediction, target), evidence_coverage or 0.0)


def decisive_evidence_coverage(prediction, target) -> float | None:
    spans = [
        span
        for span in target.get("evidence", [])
        if isinstance(span, dict) and span.get("role") == "decisive"
    ]
    if not spans:
        return None
    retained = sum(
        max(
            0,
            min(prediction["end_sec"], span["end_sec"])
            - max(prediction["start_sec"], span["start_sec"]),
        )
        for span in spans
    )
    return retained / sum(span["end_sec"] - span["start_sec"] for span in spans)


def score_events(predictions, targets, optional, threshold):
    """Required events first; optional hits neither increase recall nor penalize precision."""
    required_pairs = matching_pairs(
        predictions, targets, threshold, event_coverage, "event_coverage"
    )
    used = {p["prediction"] for p in required_pairs}
    remaining = [i for i in range(len(predictions)) if i not in used]
    optional_pairs = matching_pairs(
        [predictions[i] for i in remaining],
        optional,
        threshold,
        event_coverage,
        "event_coverage",
    )
    optional_pairs = [{**p, "prediction": remaining[p["prediction"]]} for p in optional_pairs]
    used.update(p["prediction"] for p in optional_pairs)
    unmatched = [i for i in range(len(predictions)) if i not in used]
    matched_targets = [targets[p["target"]] for p in required_pairs] + [
        optional[p["target"]] for p in optional_pairs
    ]
    duplicates = [
        i
        for i in unmatched
        if any(event_coverage(predictions[i], t) >= threshold for t in matched_targets)
    ]
    evidence_coverage = []
    for pair in required_pairs:
        coverage = decisive_evidence_coverage(
            predictions[pair["prediction"]], targets[pair["target"]]
        )
        if coverage is not None:
            evidence_coverage.append(
                {
                    "prediction": pair["prediction"],
                    "target": pair["target"],
                    "coverage": coverage,
                }
            )
    return {
        "decisive_evidence_coverage": evidence_coverage,
        **_rates(len(required_pairs), len(predictions) - len(optional_pairs), len(targets)),
        "output_count": len(predictions),
        "output_duration_sec": sum(p["end_sec"] - p["start_sec"] for p in predictions),
        "optional_hits": len(optional_pairs),
        "duplicate_count": len(duplicates),
        "matches": required_pairs,
        "optional_matches": optional_pairs,
        "duplicate_predictions": duplicates,
        "unmatched_predictions": unmatched,
        "unmatched_targets": [
            i for i in range(len(targets)) if i not in {p["target"] for p in required_pairs}
        ],
    }


def _run_dir(output_dir, run_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run_id must contain only letters, digits, underscores or hyphens")
    return Path(output_dir) / run_id


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code():
    root = Path(__file__).parent
    return {str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob("*.py"))}


def _settings_profile(settings):
    # Credentials never enter the evaluation snapshot.
    return settings.model_dump(mode="json", exclude={"gemini_api_key", "job_output_dir"})


def _validate_dataset(rows: list[dict], annotations: list[dict], allowed: set[str]) -> tuple:
    if not isinstance(annotations, list) or not all(isinstance(row, dict) for row in annotations):
        raise ValueError("annotations.json must contain an array of objects")
    annotation_ids = [row.get("video_id") for row in annotations]
    if any(not isinstance(video_id, str) or not video_id for video_id in annotation_ids):
        raise ValueError("Every annotation needs a nonempty video_id")
    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("annotations.json contains duplicate video IDs")
    labels = {
        row["video_id"]: row for row in annotations if row.get("annotation_status") in allowed
    }
    selected = [row for row in rows if row.get("video_id") in labels]
    selected_ids = [row.get("video_id") for row in selected]
    if not selected or len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Evaluation needs a nonempty dataset with unique video IDs")
    missing = sorted(set(labels) - set(selected_ids))
    if missing:
        raise ValueError(f"Annotations missing from manifest: {missing}")
    for row in selected:
        video_id = row["video_id"]
        duration = row.get("duration_sec")
        if not isinstance(duration, (int, float)) or not isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid video duration: {video_id}")
        annotation = labels[video_id]
        events = []
        for group in ("highlights", "optional_highlights"):
            values = annotation.get(group, [])
            if not isinstance(values, list) or not all(isinstance(value, dict) for value in values):
                raise ValueError(f"Invalid {group}: {video_id}")
            events.extend(values)
        intervals = [
            *(event for event in events),
            *annotation.get("context_events", []),
            *annotation.get("excluded_intervals", []),
            *(clip for event in events for clip in event.get("alternative_clips", [])),
            *(
                span
                for event in events
                for span in event.get("evidence", [])
                if isinstance(span, dict) and {"start_sec", "end_sec"} <= span.keys()
            ),
        ]
        for interval in intervals:
            start, end = interval.get("start_sec"), interval.get("end_sec")
            if (
                not isinstance(start, (int, float))
                or not isinstance(end, (int, float))
                or not isfinite(start)
                or not isfinite(end)
                or not 0 <= start < end <= duration + 0.1
            ):
                raise ValueError(f"Invalid annotation interval: {video_id}")
        event_ids = [event.get("event_id") for event in events if event.get("event_id")]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError(f"Duplicate annotation event IDs: {video_id}")
    return labels, selected


def run_evaluation(
    run_id: str,
    *,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    resume: bool = False,
    limit: int | None = None,
    include_silver: bool = False,
    video_ids: list[str] | None = None,
    task_file: Path | None = None,
) -> Path:
    run_dir = _run_dir(output_dir, run_id).resolve()
    settings = Settings(VH_JOB_OUTPUT_DIR=run_dir / "jobs")
    code = _code()
    implementation = hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest()
    if resume:
        protocol = json.loads((run_dir / "protocol.json").read_text())
        if protocol["implementation"] != implementation or protocol[
            "settings"
        ] != _settings_profile(settings):
            raise ValueError(
                "Evaluation resume requires the same code and settings; use a new run_id"
            )
        if task_file or limit is not None or include_silver or video_ids:
            raise ValueError(
                "Resume uses the frozen protocol; do not supply selection or task overrides"
            )
    else:
        if run_dir.exists():
            raise ValueError("Evaluation already exists; use --resume or a new run_id")
        annotations = json.loads((dataset_dir / "annotations.json").read_text())
        allowed = {"labeled", "silver"} if include_silver else {"labeled"}
        labels, rows = _validate_dataset(
            read_jsonl(dataset_dir / "manifest.jsonl"), annotations, allowed
        )
        if video_ids:
            if len(video_ids) != len(set(video_ids)):
                raise ValueError("video_ids must be unique")
            available = {row["video_id"] for row in rows}
            missing = sorted(set(video_ids) - available)
            if missing:
                raise ValueError(f"Requested videos are not available for evaluation: {missing}")
            selected = set(video_ids)
            rows = [row for row in rows if row["video_id"] in selected]
        if limit is not None:
            rows = rows[:limit]
        constraints = json.loads(task_file.read_text()) if task_file else {}
        forbidden = {"video_path", "video_id", "job_id", "resume", "subtitle_path", "language"}
        if forbidden & constraints.keys():
            raise ValueError(
                "Task file must contain shared task constraints, not video identity or input paths"
            )
        items = []
        for row in rows:
            task = DetectionTask(
                video_path=Path(row["path"]).resolve(),
                video_id=row["video_id"],
                job_id=row["video_id"],
                language=row.get("language"),
                **constraints,
            )
            targets = labels[row["video_id"]]["highlights"]
            items.append(
                {
                    "video_id": row["video_id"],
                    "title": row["title"],
                    "split": row.get("split"),
                    "drama_id": row.get("drama_id"),
                    "annotation_status": labels[row["video_id"]]["annotation_status"],
                    "targets": targets,
                    "annotation": labels[row["video_id"]],
                    "task": task.model_dump(mode="json"),
                    "video_sha256": _digest(task.video_path),
                }
            )
        protocol = {
            "schema_version": 1,
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "dataset_dir": str(dataset_dir.resolve()),
            "implementation": implementation,
            "settings": _settings_profile(settings),
            "items": items,
            "scoring": "event discovery by accepted-boundary IoU or decisive-evidence coverage; raw boundary IoU retained; incomplete outputs excluded",
        }
        write_json(run_dir / "protocol.json", protocol)
        for name, content in code.items():
            path = run_dir / "source" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    service = HighlightDetectionService(settings)
    for index, item in enumerate(protocol["items"], 1):
        video_id = item["video_id"]
        job = run_dir / "jobs" / video_id
        result_path = job / "result.json"
        if result_path.exists() and json.loads(result_path.read_text())["completion"] == "complete":
            continue
        status_path = job / "execution.json"
        prior = json.loads(status_path.read_text()) if status_path.exists() else {"attempts": []}
        attempt = {"started_at": datetime.now(UTC).isoformat(), "status": "running"}
        prior["attempts"].append(attempt)
        write_json(status_path, prior)
        print(f"[{index}/{len(protocol['items'])}] {item['title']}", flush=True)
        started = time.monotonic()
        try:
            task = DetectionTask.model_validate(item["task"])
            if _digest(task.video_path) != item["video_sha256"]:
                raise ValueError("Video changed since evaluation was frozen")
            task.resume = (job / "state.json").exists()
            result = service.detect(task)
            attempt["status"] = result.completion
            attempt["stop_reason"] = result.analysis.stop_reason
        except Exception as exc:
            attempt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        finally:
            attempt["elapsed_sec"] = time.monotonic() - started
            write_json(status_path, prior)
            score_evaluation(run_id, output_dir=output_dir)
        print(f"  {attempt['status']} ({attempt['elapsed_sec']:.1f}s)", flush=True)
    score_evaluation(run_id, output_dir=output_dir)
    return run_dir / "report.md"


def _rates(hits, predictions, targets):
    precision = hits / predictions if predictions else 0.0
    recall = hits / targets if targets else 0.0
    return {
        "hits": hits,
        "predictions": predictions,
        "targets": targets,
        "precision": precision,
        "recall": recall,
        "f1": 2 * hits / (predictions + targets) if predictions + targets else 0.0,
        "precision_interval_95": _wilson_interval(hits, predictions),
        "recall_interval_95": _wilson_interval(hits, targets),
    }


def _wilson_interval(hits: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    probability = hits / total
    denominator = 1 + z**2 / total
    center = (probability + z**2 / (2 * total)) / denominator
    radius = z * sqrt(probability * (1 - probability) / total + z**2 / (4 * total**2)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _annotation_cohort(annotation: dict) -> str:
    if annotation.get("human_verified") is True:
        return "human_verified"
    if annotation.get("annotation_status") == "silver":
        return "silver"
    if annotation.get("prediction_exposure") is False:
        return "ai_blind"
    if annotation.get("prediction_exposure") is True:
        return "ai_prediction_exposed"
    return "unspecified"


def _aggregate_event_metrics(rows: list[dict], key: str) -> dict:
    totals = {
        name: sum(row["event_thresholds"][key][name] for row in rows)
        for name in (
            "hits",
            "predictions",
            "targets",
            "output_count",
            "output_duration_sec",
            "optional_hits",
            "duplicate_count",
            "candidate_hits",
            "verified_hits",
        )
    }
    evidence_coverages = [
        item["coverage"]
        for row in rows
        for item in row["event_thresholds"][key]["decisive_evidence_coverage"]
    ]
    return {
        **totals,
        **_rates(totals["hits"], totals["predictions"], totals["targets"]),
        "candidate_recall": totals["candidate_hits"] / max(totals["targets"], 1),
        "verification_recall": totals["verified_hits"] / max(totals["targets"], 1),
        "mean_decisive_evidence_coverage": (
            sum(evidence_coverages) / len(evidence_coverages) if evidence_coverages else None
        ),
        "decisive_evidence_match_count": len(evidence_coverages),
    }


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, ceil(probability * len(ordered)) - 1)]


def _seconds(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}s"


def _match_text(matches: list[dict], target_prefix: str, score_key: str) -> str:
    if not matches:
        return "无"
    return "，".join(
        f"P{match['prediction']}→{target_prefix}{match['target']} ({match[score_key]:.3f})"
        for match in matches
    )


def _index_text(values: list[int], prefix: str) -> str:
    return "、".join(f"{prefix}{value}" for value in values) if values else "无"


def score_evaluation(run_id: str, *, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict:
    """Score frozen labels offline, retaining failed and not-yet-run videos in the denominator."""
    run_dir = _run_dir(output_dir, run_id)
    protocol = json.loads((run_dir / "protocol.json").read_text())
    details = []
    for item in protocol["items"]:
        annotation = item.get("annotation", {})
        job = run_dir / "jobs" / item["video_id"]
        result = (
            json.loads((job / "result.json").read_text())
            if (job / "result.json").exists()
            else None
        )
        execution = (
            json.loads((job / "execution.json").read_text())
            if (job / "execution.json").exists()
            else {"attempts": []}
        )
        trace_path = job / "trace.jsonl"
        trace = read_jsonl(trace_path) if trace_path.exists() else []
        request_durations = [
            row["elapsed_sec"]
            for row in trace
            if row.get("kind") in {"model", "clip_review"}
            and isinstance(row.get("elapsed_sec"), (int, float))
        ]
        attempt_durations = [attempt.get("elapsed_sec") for attempt in execution["attempts"]]
        runtime = {
            "elapsed_sec": sum(attempt_durations)
            if attempt_durations
            and all(isinstance(value, (int, float)) for value in attempt_durations)
            else None,
            "model_calls": result.get("analysis", {}).get("model_calls", 0) if result else 0,
            "successful_model_requests": len(request_durations),
            "model_request_durations_sec": request_durations,
            "model_request_median_sec": median(request_durations) if request_durations else None,
            "model_request_p90_sec": _percentile(request_durations, 0.9),
            "model_request_max_sec": max(request_durations) if request_durations else None,
            "request_error_count": sum(row.get("kind") == "request_error" for row in trace),
            "missing_tool_call_count": sum(
                row.get("kind") == "model" and row.get("calls") == [] for row in trace
            ),
            "invalid_tool_request_count": sum(
                row.get("kind") == "tool" and row.get("error") is True for row in trace
            ),
        }
        status = execution["attempts"][-1]["status"] if execution["attempts"] else "not_started"
        if result and status not in {"failed", "running"}:
            status = result["completion"]
        complete = status == "complete"
        predictions = result["highlights"] if complete else []
        state = (
            json.loads((job / "state.json").read_text())["tools"]
            if (job / "state.json").exists()
            else {}
        )
        candidates = [
            {
                "start_sec": min(s["start_sec"] for s in e["required_spans"]),
                "end_sec": max(s["end_sec"] for s in e["required_spans"]),
            }
            for e in state.get("events", [])
            if e["required_spans"] and e["status"] not in {"merged", "rejected"}
        ]
        observed_candidates = [
            {
                "start_sec": min(s["start_sec"] for s in e["required_spans"]),
                "end_sec": max(s["end_sec"] for s in e["required_spans"]),
            }
            for e in state.get("events", [])
            if e["required_spans"] and e["status"] != "merged"
        ]
        verified = [p for p in state.get("plans", []) if p["status"] == "ready"]
        thresholds = {}
        event_thresholds = {}
        for threshold in THRESHOLDS:
            pairs = matching_pairs(predictions, item["targets"], threshold)
            event_thresholds[f"{threshold:.2f}"] = {
                **score_events(
                    predictions,
                    item["targets"],
                    annotation.get("optional_highlights", []),
                    threshold,
                ),
                "candidate_hits": len(
                    matching_pairs(observed_candidates, item["targets"], threshold, event_coverage)
                )
                if complete
                else 0,
                "verified_hits": len(
                    matching_pairs(verified, item["targets"], threshold, event_coverage)
                )
                if complete
                else 0,
            }
            thresholds[f"{threshold:.2f}"] = {
                **_rates(len(pairs), len(predictions), len(item["targets"])),
                "matches": pairs,
                "unmatched_targets": [
                    j for j in range(len(item["targets"])) if j not in {p["target"] for p in pairs}
                ],
                "unmatched_predictions": [
                    i for i in range(len(predictions)) if i not in {p["prediction"] for p in pairs}
                ],
                "candidate_hits": matching_hits(candidates, item["targets"], threshold)
                if complete
                else 0,
                "verified_hits": matching_hits(verified, item["targets"], threshold)
                if complete
                else 0,
            }
        details.append(
            {
                "video_id": item["video_id"],
                "title": item["title"],
                "annotation_cohort": _annotation_cohort(annotation),
                "annotation_profile": {
                    key: annotation.get(key)
                    for key in (
                        "annotation_version",
                        "annotation_method",
                        "annotation_status",
                        "human_verified",
                        "prediction_exposure",
                    )
                },
                "status": status,
                "targets": item["targets"],
                "optional_targets": annotation.get("optional_highlights", []),
                "predictions": predictions,
                "partial_predictions": result["highlights"] if result and not complete else [],
                "analysis": result.get("analysis", {}) if result else {},
                "runtime": runtime,
                "attempts": execution["attempts"],
                "thresholds": thresholds,
                "event_thresholds": event_thresholds,
            }
        )
    metrics = {
        "schema_version": 3,
        "run_id": run_id,
        "videos": len(details),
        "completed": sum(d["status"] == "complete" for d in details),
        "details": details,
    }
    metrics["completion_rate"] = metrics["completed"] / len(details)
    metrics["iou_metrics"] = {}
    metrics["completed_only"] = {}
    for key in (f"{t:.2f}" for t in THRESHOLDS):
        for destination, rows in (
            (metrics["iou_metrics"], details),
            (metrics["completed_only"], [d for d in details if d["status"] == "complete"]),
        ):
            totals = {
                k: sum(d["thresholds"][key][k] for d in rows)
                for k in ("hits", "predictions", "targets", "candidate_hits", "verified_hits")
            }
            destination[key] = {
                **_rates(totals["hits"], totals["predictions"], totals["targets"]),
                "candidate_recall": totals["candidate_hits"] / max(totals["targets"], 1),
                "verification_recall": totals["verified_hits"] / max(totals["targets"], 1),
            }
    metrics["event_metrics"] = {}
    for key in (f"{t:.2f}" for t in THRESHOLDS):
        metrics["event_metrics"][key] = _aggregate_event_metrics(details, key)
    metrics["cohort_metrics"] = {}
    for cohort in sorted({detail["annotation_cohort"] for detail in details}):
        rows = [detail for detail in details if detail["annotation_cohort"] == cohort]
        metrics["cohort_metrics"][cohort] = {
            "videos": len(rows),
            "event_metrics": {
                key: _aggregate_event_metrics(rows, key)
                for key in (f"{threshold:.2f}" for threshold in THRESHOLDS)
            },
        }
    elapsed = [
        detail["runtime"]["elapsed_sec"]
        for detail in details
        if isinstance(detail["runtime"]["elapsed_sec"], (int, float))
    ]
    request_durations = [
        duration
        for detail in details
        for duration in detail["runtime"]["model_request_durations_sec"]
    ]
    metrics["runtime_metrics"] = {
        "video_elapsed_total_sec": sum(elapsed) if len(elapsed) == len(details) else None,
        "timed_videos": len(elapsed),
        "first_attempt_completed": sum(
            bool(detail["attempts"]) and detail["attempts"][0]["status"] == "complete"
            for detail in details
        ),
        "videos_with_multiple_attempts": sum(len(detail["attempts"]) > 1 for detail in details),
        "video_elapsed_median_sec": median(elapsed) if elapsed else None,
        "video_elapsed_p90_sec": _percentile(elapsed, 0.9),
        "model_calls": sum(detail["runtime"]["model_calls"] for detail in details),
        "successful_model_requests": len(request_durations),
        "model_request_median_sec": median(request_durations) if request_durations else None,
        "model_request_p90_sec": _percentile(request_durations, 0.9),
        "model_request_max_sec": max(request_durations) if request_durations else None,
        "request_errors": sum(detail["runtime"]["request_error_count"] for detail in details),
        "missing_tool_calls": sum(
            detail["runtime"]["missing_tool_call_count"] for detail in details
        ),
        "invalid_tool_requests": sum(
            detail["runtime"]["invalid_tool_request_count"] for detail in details
        ),
    }
    write_json(run_dir / "metrics.json", metrics)
    lines = [
        f"# Agent 评测：{run_id}",
        "",
        f"完成：{metrics['completed']}/{len(details)}。标注与配置快照见 protocol.json；逐次执行记录保留在 jobs/。",
        "",
        "主指标以全体标注为分母；未完成/失败的输出不作为最终预测。时间重叠匹配不校验剧情身份，也不代表人工采用率。",
        "",
        "事件发现口径：可接受边界 IoU 或决定性证据保留率达到阈值即命中；同一事件只命中一次。先匹配推荐事件，再匹配可选事件。可选命中不增加必选召回分母，也不计为误报。时间覆盖不代表语义确认。",
        "",
        "| 事件覆盖阈值 | Precision | Recall | F1 | 可选命中 | 重复预测 |",
        "|---|---:|---:|---:|---:|---:|",
        *[
            f"| {key} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['optional_hits']} | {v['duplicate_count']} |"
            for key, v in metrics["event_metrics"].items()
        ],
        "",
        "IoU≥0.5 按标注来源拆分；不同来源不应只看混合总分：",
        "",
        "| 标注来源 | 视频 | 命中/必选 | 计入精度的预测 | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[
            (
                f"| {cohort} | {value['videos']} | {score['hits']}/{score['targets']} | "
                f"{score['predictions']} | {score['precision']:.3f} | {score['recall']:.3f} | "
                f"{score['f1']:.3f} |"
            )
            for cohort, value in metrics["cohort_metrics"].items()
            for score in [value["event_metrics"]["0.50"]]
        ],
        "",
        "运行稳定性：",
        "",
        (
            f"首轮完成 {metrics['runtime_metrics']['first_attempt_completed']}/{len(details)}；"
            f"多次尝试 {metrics['runtime_metrics']['videos_with_multiple_attempts']} 条；"
            f"完整计时 {metrics['runtime_metrics']['timed_videos']}/{len(details)}。"
            f"累计尝试耗时 {_seconds(metrics['runtime_metrics']['video_elapsed_total_sec'])}；"
            f"缺失工具调用 {metrics['runtime_metrics']['missing_tool_calls']} 次。"
        ),
        "视频耗时包含全部尝试；任一次耗时缺失时，该视频及全组累计耗时记为未知。中位数和 P90 只使用计时完整的视频。",
        "",
        "| 总模型调用 | 视频耗时中位数 | 视频耗时 P90 | 请求耗时中位数 | 请求耗时 P90 | 请求错误 | 无效工具调用 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {metrics['runtime_metrics']['model_calls']} | "
            f"{_seconds(metrics['runtime_metrics']['video_elapsed_median_sec'])} | "
            f"{_seconds(metrics['runtime_metrics']['video_elapsed_p90_sec'])} | "
            f"{_seconds(metrics['runtime_metrics']['model_request_median_sec'])} | "
            f"{_seconds(metrics['runtime_metrics']['model_request_p90_sec'])} | "
            f"{metrics['runtime_metrics']['request_errors']} | "
            f"{metrics['runtime_metrics']['invalid_tool_requests']} |"
        ),
        "",
        "以下保留推荐边界的原始时间 IoU 口径：",
        "",
        "| IoU | Precision | Recall | F1 | 事件池召回 | 复核后召回 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, m in metrics["iou_metrics"].items():
        lines.append(
            f"| {key} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['candidate_recall']:.3f} | {m['verification_recall']:.3f} |"
        )
    for d in details:
        lines += [
            "",
            f"## {d['title']} ({d['status']})",
            "",
            f"视频 ID：{d['video_id']}。",
            (
                f"累计耗时：{_seconds(d['runtime']['elapsed_sec'])}（{len(d['attempts'])} 次尝试）；"
                f"模型调用：{d['runtime']['model_calls']}；"
                f"无效工具调用：{d['runtime']['invalid_tool_request_count']}；"
                f"请求错误：{d['runtime']['request_error_count']}。"
            ),
            "",
            "标注：",
        ]
        lines += [
            f"- G{i}: {s['start_sec']:.2f}–{s['end_sec']:.2f}s {s.get('description', '')}"
            for i, s in enumerate(d["targets"])
        ]
        lines += ["", "可选标注："]
        lines += [
            f"- O{i}: {s['start_sec']:.2f}–{s['end_sec']:.2f}s {s.get('description', '')}"
            for i, s in enumerate(d["optional_targets"])
        ] or ["- 无"]
        lines += ["", "最终预测："]
        lines += [
            f"- P{i}: {s['start_sec']:.2f}–{s['end_sec']:.2f}s {s.get('description', '')}"
            for i, s in enumerate(d["predictions"])
        ] or ["- 无"]
        event_match = d["event_thresholds"]["0.50"]
        boundary_match = d["thresholds"]["0.50"]
        lines += [
            "",
            (
                "事件发现≥0.5（可接受边界 IoU 或决定性证据覆盖）："
                f"{_match_text(event_match['matches'], 'G', 'event_coverage')}；"
                f"可选匹配：{_match_text(event_match['optional_matches'], 'O', 'event_coverage')}；"
                f"未发现必选标注：{_index_text(event_match['unmatched_targets'], 'G')}；"
                f"未匹配预测：{_index_text(event_match['unmatched_predictions'], 'P')}。"
            ),
            "",
            (
                "原始边界 IoU≥0.5："
                f"{_match_text(boundary_match['matches'], 'G', 'iou')}；"
                f"未贴合必选边界：{_index_text(boundary_match['unmatched_targets'], 'G')}；"
                f"未匹配预测：{_index_text(boundary_match['unmatched_predictions'], 'P')}。"
            ),
        ]
        if d["attempts"] and d["attempts"][-1].get("error"):
            lines += ["", f"执行错误：{d['attempts'][-1]['error']}"]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")
    return metrics


def _comparison_item(item: dict) -> dict:
    return {
        "video_id": item["video_id"],
        "video_sha256": item["video_sha256"],
        "task": item["task"],
        "annotation": item["annotation"],
    }


def _agent_source_digest(run_dir: Path) -> str:
    source = run_dir / "source"
    files = {
        str(path.relative_to(source)): path.read_text()
        for path in sorted(source.rglob("*.py"))
        if path.name not in {"cli.py", "evaluation.py"}
    }
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def compare_evaluations(
    comparison_id: str,
    run_ids: list[str],
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    kind: ComparisonKind = ComparisonKind.repeat,
) -> Path:
    """Compare identical inputs for repeated runs or an implementation revision."""
    kind = ComparisonKind(kind)
    if len(run_ids) < 2 or len(set(run_ids)) != len(run_ids):
        raise ValueError("Comparison needs at least two distinct run IDs")
    comparison_dir = _run_dir(Path(output_dir) / "comparisons", comparison_id)
    if comparison_dir.exists():
        raise ValueError("Comparison already exists; use a new comparison_id")

    protocols = {
        run_id: json.loads((_run_dir(output_dir, run_id) / "protocol.json").read_text())
        for run_id in run_ids
    }
    settings = [protocols[run_id]["settings"] for run_id in run_ids]
    agent_digests = [_agent_source_digest(_run_dir(output_dir, run_id)) for run_id in run_ids]
    if any(value != settings[0] for value in settings[1:]):
        raise ValueError("Comparison requires identical settings")
    if kind == ComparisonKind.repeat and any(
        value != agent_digests[0] for value in agent_digests[1:]
    ):
        raise ValueError("Stability comparison requires identical Agent code")
    protocol_items = {
        run_id: {item["video_id"]: item for item in protocols[run_id]["items"]}
        for run_id in run_ids
    }
    metrics = {run_id: score_evaluation(run_id, output_dir=output_dir) for run_id in run_ids}
    pairwise = []
    for left_id, right_id in combinations(run_ids, 2):
        left = {row["video_id"]: row for row in metrics[left_id]["details"]}
        right = {row["video_id"]: row for row in metrics[right_id]["details"]}
        common_ids = sorted(set(left) & set(right))
        if not common_ids:
            raise ValueError("Comparison runs have no videos in common")
        for video_id in common_ids:
            if _comparison_item(protocol_items[left_id][video_id]) != _comparison_item(
                protocol_items[right_id][video_id]
            ):
                raise ValueError(f"Comparison protocol differs for video {video_id}")
        status_agreements = 0
        completion_agreements = 0
        jointly_complete = 0
        empty_pairs = 0
        count_differences = []
        threshold_totals = {
            f"{threshold:.2f}": {
                "matches": 0,
                "left_predictions": 0,
                "right_predictions": 0,
                "matched_ious": [],
            }
            for threshold in THRESHOLDS
        }
        for video_id in common_ids:
            left_row = left[video_id]
            right_row = right[video_id]
            status_agreements += left_row["status"] == right_row["status"]
            left_complete = left_row["status"] == "complete"
            right_complete = right_row["status"] == "complete"
            completion_agreements += left_complete == right_complete
            if not (left_complete and right_complete):
                continue
            jointly_complete += 1
            left_predictions = left_row["predictions"]
            right_predictions = right_row["predictions"]
            empty_pairs += not left_predictions and not right_predictions
            count_differences.append(abs(len(left_predictions) - len(right_predictions)))
            for threshold in THRESHOLDS:
                key = f"{threshold:.2f}"
                pairs = matching_pairs(left_predictions, right_predictions, threshold)
                threshold_totals[key]["matches"] += len(pairs)
                threshold_totals[key]["left_predictions"] += len(left_predictions)
                threshold_totals[key]["right_predictions"] += len(right_predictions)
                threshold_totals[key]["matched_ious"].extend(pair["iou"] for pair in pairs)
        agreements = {}
        for key, values in threshold_totals.items():
            denominator = values["left_predictions"] + values["right_predictions"]
            agreements[key] = {
                "matches": values["matches"],
                "left_predictions": values["left_predictions"],
                "right_predictions": values["right_predictions"],
                "output_f1": 2 * values["matches"] / denominator if denominator else None,
                "mean_matched_iou": (
                    sum(values["matched_ious"]) / len(values["matched_ious"])
                    if values["matched_ious"]
                    else None
                ),
            }
        left_common = [left[video_id] for video_id in common_ids]
        right_common = [right[video_id] for video_id in common_ids]
        pairwise.append(
            {
                "left": left_id,
                "right": right_id,
                "video_ids": common_ids,
                "videos": len(common_ids),
                "exact_status_agreement": status_agreements / len(common_ids),
                "completion_agreement": completion_agreements / len(common_ids),
                "jointly_complete": jointly_complete,
                "jointly_complete_empty_outputs": empty_pairs,
                "mean_output_count_difference": (
                    sum(count_differences) / len(count_differences) if count_differences else None
                ),
                "left_event_metrics": {
                    key: _aggregate_event_metrics(left_common, key) for key in threshold_totals
                },
                "right_event_metrics": {
                    key: _aggregate_event_metrics(right_common, key) for key in threshold_totals
                },
                "output_agreement": agreements,
            }
        )
    result = {
        "schema_version": 3,
        "comparison_id": comparison_id,
        "kind": kind.value,
        "created_at": datetime.now(UTC).isoformat(),
        "run_ids": run_ids,
        "agent_implementations": dict(zip(run_ids, agent_digests, strict=True)),
        "pairwise": pairwise,
    }
    write_json(comparison_dir / "comparison.json", result)
    purpose = "重复运行稳定性" if kind == ComparisonKind.repeat else "实现改动对照"
    code_requirement = (
        "Agent 代码相同" if kind == ComparisonKind.repeat else "Agent 代码允许不同，逐轮保存指纹"
    )
    lines = [
        f"# Agent {purpose}：{comparison_id}",
        "",
        f"每一对只比较共有且视频、任务、标注、配置完全一致的样本；{code_requirement}。输出一致性不代表正确，事件指标也不替代人工采用率。",
        "",
        "| 运行对 | 完成一致率 | 共同完成 | 事件精度（左/右） | 事件召回（左/右） | 平均数量差 | IoU≥0.5 输出 F1 | 匹配边界 IoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pairwise:
        agreement = row["output_agreement"]["0.50"]
        output_f1 = "—" if agreement["output_f1"] is None else f"{agreement['output_f1']:.3f}"
        mean_iou = (
            "—" if agreement["mean_matched_iou"] is None else f"{agreement['mean_matched_iou']:.3f}"
        )
        mean_difference = (
            "—"
            if row["mean_output_count_difference"] is None
            else f"{row['mean_output_count_difference']:.2f}"
        )
        left_recall = row["left_event_metrics"]["0.50"]["recall"]
        right_recall = row["right_event_metrics"]["0.50"]["recall"]
        left_precision = row["left_event_metrics"]["0.50"]["precision"]
        right_precision = row["right_event_metrics"]["0.50"]["precision"]
        lines.append(
            f"| {row['left']} / {row['right']} | {row['completion_agreement']:.3f} | "
            f"{row['jointly_complete']}/{row['videos']} | "
            f"{left_precision:.3f}/{right_precision:.3f} | {left_recall:.3f}/{right_recall:.3f} | "
            f"{mean_difference} | {output_f1} | {mean_iou} |"
        )
    (comparison_dir / "report.md").write_text("\n".join(lines) + "\n")
    return comparison_dir / "report.md"
