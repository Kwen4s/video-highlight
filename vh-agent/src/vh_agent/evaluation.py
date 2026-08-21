import json
from pathlib import Path

from .config import Settings
from .models import DetectionTask
from .pipeline import HighlightDetectionService

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "datasets" / "test"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluations"


def _write_jsonl(path: Path, rows) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_predictions(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    return {item["video"]["video_id"]: item for item in load_jsonl(path)}


def decision_segments(trace: dict, field: str) -> list[dict[str, float]]:
    segments: list[dict[str, float]] = []
    for item in trace["reasoning"]["decisions"]:
        decision = item[field]
        if not decision["is_highlight"]:
            continue
        scene = item["scene"]
        segments.append(
            {
                "start_sec": decision["start_sec"]
                if decision["start_sec"] is not None
                else scene["start_sec"],
                "end_sec": decision["end_sec"]
                if decision["end_sec"] is not None
                else scene["end_sec"],
            }
        )
    return segments


def segment_iou(left: dict, right: dict) -> float:
    overlap = max(
        0.0,
        min(left["end_sec"], right["end_sec"]) - max(left["start_sec"], right["start_sec"]),
    )
    union = max(left["end_sec"], right["end_sec"]) - min(left["start_sec"], right["start_sec"])
    return overlap / union if union else 0.0


def run_evaluation(
    run_id: str,
    *,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    resume: bool = False,
    limit: int | None = None,
) -> Path:
    run_dir = output_dir / run_id
    predictions_path = run_dir / "predictions.jsonl"
    errors_path = run_dir / "errors.jsonl"
    rows = load_jsonl(dataset_dir / "manifest.jsonl")
    if limit is not None:
        rows = rows[:limit]
    completed = load_predictions(predictions_path) if resume else {}
    errors = (
        {item["video_id"]: item for item in load_jsonl(errors_path)}
        if resume and errors_path.is_file()
        else {}
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        predictions_path.write_text("", encoding="utf-8")
        errors_path.write_text("", encoding="utf-8")

    service = HighlightDetectionService(
        Settings(
            VH_JOB_OUTPUT_DIR=run_dir / "jobs",
            VH_WRITE_TRACE=True,
            VH_WRITE_RESULT_FILE=False,
        )
    )
    for index, row in enumerate(rows, start=1):
        video_id = row["video_id"]
        if video_id in completed:
            print(f"[{index}/{len(rows)}] {video_id}: complete", flush=True)
            continue
        print(f"[{index}/{len(rows)}] {video_id}: {row['title']}", flush=True)
        try:
            result = service.detect(
                DetectionTask(
                    video_path=row["path"],
                    video_id=video_id,
                    job_id=video_id,
                    language=row["language"],
                )
            )
        except Exception as exc:
            errors[video_id] = {
                "video_id": video_id,
                "path": row["path"],
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            _write_jsonl(errors_path, errors.values())
            print(f"  failed: {type(exc).__name__}: {exc}", flush=True)
            continue
        with predictions_path.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(result.model_dump(mode="json"), ensure_ascii=False) + "\n")
        if errors.pop(video_id, None) is not None:
            _write_jsonl(errors_path, errors.values())

    return predictions_path


def score_evaluation(
    run_id: str,
    *,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    include_silver: bool = False,
) -> dict[str, object]:
    run_dir = output_dir / run_id
    rows = load_jsonl(dataset_dir / "manifest.jsonl")
    annotations = {
        item["video_id"]: item
        for item in json.loads((dataset_dir / "annotations.json").read_text(encoding="utf-8"))
        if item["annotation_status"] == "labeled"
        or (include_silver and item["annotation_status"] == "silver")
    }
    predictions = load_predictions(run_dir / "predictions.jsonl")
    missing = [
        row["video_id"]
        for row in rows
        if row["video_id"] in annotations and row["video_id"] not in predictions
    ]
    if missing:
        raise SystemExit(f"Missing predictions for {len(missing)} videos: {', '.join(missing)}")

    thresholds = (0.3, 0.5, 0.7)
    totals = {
        threshold: {
            "ground_truth": 0,
            "predictions": 0,
            "hits": 0,
            "prediction_hits": 0,
            "candidate_hits": 0,
            "verified_hits": 0,
        }
        for threshold in thresholds
    }
    details = []
    for row in rows:
        video_id = row["video_id"]
        if video_id not in annotations:
            continue
        gt = annotations[video_id]["highlights"]
        predicted = predictions[video_id]["highlights"]
        trace = json.loads((run_dir / "jobs" / video_id / "trace.json").read_text(encoding="utf-8"))
        candidates = trace["local_candidates"]
        verified = decision_segments(trace, "decision")
        per_threshold = {}
        for threshold, total in totals.items():
            hit_gt = sum(
                any(segment_iou(item, result) >= threshold for result in predicted) for item in gt
            )
            hit_pred = sum(
                any(segment_iou(result, item) >= threshold for item in gt) for result in predicted
            )
            hit_verified = sum(
                any(segment_iou(item, result) >= threshold for result in verified) for item in gt
            )
            hit_candidates = sum(
                any(segment_iou(item, candidate) >= threshold for candidate in candidates)
                for item in gt
            )
            total["ground_truth"] += len(gt)
            total["predictions"] += len(predicted)
            total["hits"] += hit_gt
            total["prediction_hits"] += hit_pred
            total["candidate_hits"] += hit_candidates
            total["verified_hits"] += hit_verified
            per_threshold[f"{threshold:.2f}"] = {
                "hits": hit_gt,
                "candidate_hits": hit_candidates,
                "verified_hits": hit_verified,
            }
        details.append(
            {
                "video_id": video_id,
                "title": row["title"],
                "ground_truth": len(gt),
                "predictions": len(predicted),
                "thresholds": per_threshold,
            }
        )

    iou_metrics = {}
    for threshold, total in totals.items():
        precision = total["prediction_hits"] / max(total["predictions"], 1)
        recall = total["hits"] / max(total["ground_truth"], 1)
        iou_metrics[f"{threshold:.2f}"] = {
            "candidate_recall": total["candidate_hits"] / max(total["ground_truth"], 1),
            "verification_recall": total["verified_hits"] / max(total["ground_truth"], 1),
            "boundary_recall": recall,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        }
    metrics = {
        "dataset": "test",
        "videos": len(details),
        "iou_metrics": iou_metrics,
        "details": details,
    }
    destination = run_dir / "metrics.json"
    destination.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metrics
