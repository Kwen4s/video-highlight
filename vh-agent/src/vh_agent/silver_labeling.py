"""Resumable agent silver-label production for the short-drama corpus."""

import json
from pathlib import Path

from .config import PROJECT_ROOT, Settings
from .models import DetectionResult, DetectionTask
from .pipeline import HighlightDetectionService
from .storage import append_jsonl, read_jsonl, write_json

DEFAULT_METADATA_DIR = Path("/data1/my_short_drama/metadata")
DEFAULT_SILVER_DATASET_DIR = PROJECT_ROOT / "datasets" / "silver"
SILVER_METHOD = "native_video_react"
SILVER_ANNOTATION_REVISION = "event_v3"


def load_metadata_records(metadata_dir: Path = DEFAULT_METADATA_DIR) -> list[dict[str, object]]:
    """Load each source record once, shortest first for observable throughput."""
    paths = sorted(
        path
        for language in ("en", "zh")
        for split in ("train", "test")
        if (path := metadata_dir / language / f"{split}.jsonl").is_file()
    )
    records: list[dict[str, object]] = []
    for path in paths:
        records.extend(read_jsonl(path))
    return sorted(
        records,
        key=lambda record: (float(record["duration_sec"]), str(record["video_id"])),
    )


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
    completed_count = len(completed)
    service = HighlightDetectionService(Settings(VH_JOB_OUTPUT_DIR=run_dir / "work"))
    _write_run_manifest(
        run_dir,
        records,
        completed_count,
        status="running",
        model=service.settings.openai_agent_model,
    )
    errors_path = run_dir / "errors.jsonl"
    for index, record in enumerate(pending, start=1):
        video_id = str(record["video_id"])
        print(f"[{index}/{len(pending)}] {video_id}: {record['title']}", flush=True)
        try:
            result = service.detect(
                DetectionTask(
                    video_path=Path(str(record["path"])),
                    video_id=video_id,
                    job_id=video_id,
                    language=str(record["language"]),
                    max_highlights=None,
                    resume=(run_dir / "work" / video_id / "state.json").is_file(),
                ),
            )
            if result.completion != "complete":
                raise RuntimeError("Partial analysis is not a silver label")
            state = json.loads((run_dir / "work" / video_id / "state.json").read_text())
            clips = json.loads((run_dir / "work" / video_id / "clips.json").read_text())
            append_jsonl(annotations_path, _silver_record(record, result, state, clips))
            completed_count += 1
        except Exception as exc:
            append_jsonl(
                errors_path,
                {
                    "video_id": video_id,
                    "path": record["path"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            print(f"  failed: {type(exc).__name__}: {exc}", flush=True)
        finally:
            _write_run_manifest(
                run_dir,
                records,
                completed_count,
                status="running",
                model=service.settings.openai_agent_model,
            )

    _write_run_manifest(
        run_dir,
        records,
        completed_count,
        status="complete" if completed_count == len(records) else "incomplete",
        model=service.settings.openai_agent_model,
    )
    return annotations_path


def _completed_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {str(record["video_id"]) for record in read_jsonl(path) if "video_id" in record}


def _silver_record(record: dict, result: DetectionResult, state: dict, clips: dict) -> dict:
    events = {e["id"]: e for e in state["tools"]["events"]}
    highlights = []
    for item in result.highlights:
        event = events[clips[item.highlight_id]["event_id"]]
        spans = event["required_spans"]
        highlights.append(
            {
                **item.model_dump(mode="json"),
                "label": 1,
                "duration_sec": item.end_sec - item.start_sec,
                "source": state["profile"]["model"],
                "evidence": spans,
                **{
                    f"{role}_times_sec": [
                        (s["start_sec"] + s["end_sec"]) / 2 for s in spans if s["role"] == role
                    ]
                    for role in ("setup", "decisive", "reaction")
                },
            }
        )
    return {
        **record,
        "annotation_revision": SILVER_ANNOTATION_REVISION,
        "annotation_method": SILVER_METHOD,
        "annotation_status": "silver",
        "annotator_ids": [state["profile"]["model"]],
        "highlights": highlights,
        "notes": "Agent-generated silver labels; human review pending.",
    }


def _write_run_manifest(
    run_dir: Path,
    records: list[dict[str, object]],
    completed: int,
    *,
    status: str,
    model: str,
) -> None:
    payload = {
        "run_id": run_dir.name,
        "status": status,
        "current_annotation_revision": SILVER_ANNOTATION_REVISION,
        "annotation_method": SILVER_METHOD,
        "model": model,
        "judge_policy": "independent_rendered_clip_review",
        "selection_policy": "explicit_editorial_selection_without_count_budget",
        "ordering": "duration_ascending",
        "total_records": len(records),
        "completed_records": completed,
        "pending_records": max(len(records) - completed, 0),
        "output": "annotations.jsonl",
        "errors": "errors.jsonl",
    }
    write_json(run_dir / "run.json", payload)
