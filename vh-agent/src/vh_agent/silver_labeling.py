"""Resumable Gemini 3.7 silver-label production for the short-drama corpus."""

import json
from pathlib import Path
from typing import Iterable

from .config import PROJECT_ROOT, Settings
from .models import DetectionResult, DetectionTask, DetectionTrace, Highlight
from .pipeline import HighlightDetectionService

DEFAULT_METADATA_DIR = Path("/data1/my_short_drama/metadata")
DEFAULT_SILVER_DATASET_DIR = PROJECT_ROOT / "datasets" / "silver"
SILVER_MODEL = "gemini-3.7-flash"
SILVER_METHOD = "gemini_3_7_flash_dual_judge"


def load_metadata_records(metadata_dir: Path = DEFAULT_METADATA_DIR) -> list[dict[str, object]]:
    """Load each source record once in a stable language/split order."""
    paths = sorted(
        path
        for language in ("en", "zh")
        for split in ("train", "test")
        if (path := metadata_dir / language / f"{split}.jsonl").is_file()
    )
    records: list[dict[str, object]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def run_silver_labeling(
    run_id: str,
    *,
    metadata_dir: Path = DEFAULT_METADATA_DIR,
    output_dir: Path = DEFAULT_SILVER_DATASET_DIR,
    resume: bool = True,
    limit: int | None = None,
) -> Path:
    """Label source records in an append-only run directory, never source metadata."""
    records = load_metadata_records(metadata_dir)
    if limit is not None:
        records = records[:limit]

    run_dir = output_dir / run_id
    annotations_path = run_dir / "annotations.jsonl"
    if not resume and run_dir.exists():
        raise ValueError(f"Run directory already exists: {run_dir}; use --resume or a new run ID")
    run_dir.mkdir(parents=True, exist_ok=True)
    completed = _completed_ids(annotations_path) if resume else set()
    if not resume:
        annotations_path.write_text("", encoding="utf-8")

    pending = [record for record in records if str(record["video_id"]) not in completed]
    _write_run_manifest(run_dir, records, len(completed), status="running")
    service = HighlightDetectionService(
        Settings(
            VH_REASONING_PROVIDER="gemini",
            GEMINI_MAP_MODEL=SILVER_MODEL,
            GEMINI_JUDGE_MODEL=SILVER_MODEL,
            VH_MAX_JUDGE_CANDIDATES=10_000,
            VH_JOB_OUTPUT_DIR=run_dir / "work",
            VH_EXPORT_CLIPS=False,
            VH_WRITE_RESULT_FILE=False,
            VH_WRITE_TRACE=False,
        )
    )
    errors_path = run_dir / "errors.jsonl"
    for index, record in enumerate(pending, start=1):
        video_id = str(record["video_id"])
        print(f"[{index}/{len(pending)}] {video_id}: {record['title']}", flush=True)
        try:
            result, trace = service.detect_with_trace(
                DetectionTask(
                    video_path=Path(str(record["path"])),
                    video_id=video_id,
                    job_id=video_id,
                    language=str(record["language"]),
                )
            )
            _append_jsonl(annotations_path, _silver_record(record, result, trace))
        except Exception as exc:
            _append_jsonl(
                errors_path,
                {
                    "video_id": video_id,
                    "path": record["path"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            print(f"  failed: {type(exc).__name__}: {exc}", flush=True)

    completed_count = len(_completed_ids(annotations_path))
    _write_run_manifest(
        run_dir,
        records,
        completed_count,
        status="complete" if completed_count == len(records) else "incomplete",
    )
    return annotations_path


def _completed_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        str(record["video_id"])
        for record in _read_jsonl(path)
        if "video_id" in record
    }


def _read_jsonl(path: Path) -> Iterable[dict[str, object]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def _append_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as sink:
        sink.write(json.dumps(record, ensure_ascii=False) + "\n")
        sink.flush()


def _silver_record(
    record: dict[str, object],
    result: DetectionResult,
    trace: DetectionTrace,
) -> dict[str, object]:
    evidence_by_highlight = {
        item.highlight_id: _highlight_evidence(item, trace) for item in result.highlights
    }
    labeled = dict(record)
    labeled.update(
        {
            "annotation_method": SILVER_METHOD,
            "annotation_status": "silver",
            "annotator_ids": [SILVER_MODEL],
            "global_quality_score": _quality_score(trace),
            "highlights": [
                {
                    "highlight_id": f"{result.video.video_id}_hl_{index:03d}",
                    "start_sec": item.start_sec,
                    "end_sec": item.end_sec,
                    "duration_sec": round(item.end_sec - item.start_sec, 3),
                    "label": 1,
                    "score": item.score,
                    "highlight_type": item.highlight_type,
                    "description": item.description,
                    "reason": item.reason,
                    "source": SILVER_MODEL,
                    "confidence": evidence_by_highlight[item.highlight_id]["confidence"],
                    "annotator_scores": evidence_by_highlight[item.highlight_id][
                        "annotator_scores"
                    ],
                    "evidence": evidence_by_highlight[item.highlight_id]["evidence"],
                    "review_status": "pending",
                    "notes": "Consensus silver label; requires human review.",
                }
                for index, item in enumerate(result.highlights, start=1)
            ],
            "scene_labels": _scene_labels(result.video.video_id, trace),
            "notes": "Gemini 3.7 dual-Judge silver labels. No human review has been applied.",
        }
    )
    return labeled


def _quality_score(trace: DetectionTrace) -> float | None:
    decisions = trace.reasoning.decisions
    if not decisions:
        return None
    return round(
        sum(item.decision.confidence for item in decisions) / len(decisions),
        4,
    )


def _highlight_evidence(item: Highlight, trace: DetectionTrace) -> dict[str, object]:
    matches = [
        decision
        for decision in trace.reasoning.decisions
        if decision.decision.is_highlight
        and _segment_overlap(
            item.start_sec,
            item.end_sec,
            decision.decision.start_sec,
            decision.decision.end_sec,
        )
        > 0
    ]
    if not matches:
        return {"confidence": 0.0, "annotator_scores": [], "evidence": []}
    confidence = max(match.decision.confidence for match in matches)
    scores = [vote.score for match in matches for vote in match.votes]
    evidence = list(
        dict.fromkeys(value for match in matches for value in match.decision.evidence)
    )
    return {
        "confidence": round(confidence, 4),
        "annotator_scores": scores,
        "evidence": evidence,
    }


def _scene_labels(video_id: str, trace: DetectionTrace) -> list[dict[str, object]]:
    decisions = {item.scene.scene_id: item for item in trace.reasoning.decisions}
    labels = []
    for scene in trace.reasoning.scenes:
        trace_item = decisions.get(scene.scene_id)
        labels.append(
            {
                "video_id": video_id,
                "scene": scene.model_dump(mode="json"),
                "label": (
                    int(trace_item.decision.is_highlight) if trace_item is not None else None
                ),
                "decision": (
                    trace_item.decision.model_dump(mode="json")
                    if trace_item is not None
                    else None
                ),
                "votes": (
                    [vote.model_dump(mode="json") for vote in trace_item.votes]
                    if trace_item is not None
                    else []
                ),
            }
        )
    return labels


def _segment_overlap(
    left_start: float,
    left_end: float,
    right_start: float | None,
    right_end: float | None,
) -> float:
    if right_start is None or right_end is None:
        return 0.0
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def _write_run_manifest(
    run_dir: Path,
    records: list[dict[str, object]],
    completed: int,
    *,
    status: str,
) -> None:
    payload = {
        "run_id": run_dir.name,
        "status": status,
        "annotation_method": SILVER_METHOD,
        "model": SILVER_MODEL,
        "judge_policy": "two_independent_votes_then_disagreement_adjudication",
        "total_records": len(records),
        "completed_records": completed,
        "pending_records": max(len(records) - completed, 0),
        "output": "annotations.jsonl",
        "errors": "errors.jsonl",
    }
    (run_dir / "run.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
