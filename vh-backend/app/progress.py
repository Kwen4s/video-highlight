"""Project existing task/progress artifacts into a stable, path-free UI contract."""

import json
import math
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from .models import DetectionOptions, TaskSpecification, WorkflowSnapshot


def read_object(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def read_task_specification(path: Path) -> TaskSpecification | None:
    data = read_object(path)
    if data is None:
        return None
    options = {key: data[key] for key in DetectionOptions.model_fields if key in data}
    try:
        return TaskSpecification(**options, subtitle_provided=bool(data.get("subtitle_path")))
    except ValidationError:
        return None


def workflow_snapshot(data: dict, path: Path) -> WorkflowSnapshot:
    def number(key, *, integer=False, positive=False):
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if not math.isfinite(value) or value < 0 or (positive and value == 0):
            return None
        if integer and value != int(value):
            return None
        return int(value) if integer else value

    values = {
        key: number(key, integer=True)
        for key in (
            "completed_page_count",
            "page_count",
            "event_count",
            "candidate_count",
            "transcript_rows_read",
            "model_calls",
            "turn",
        )
    }
    values["duration_sec"] = number("duration_sec", positive=True)
    ready = data.get("ready_event_ids")
    values["ready_clip_count"] = (
        len(set(item for item in ready if isinstance(item, str)))
        if isinstance(ready, list)
        else None
    )
    capabilities = data.get("capabilities")
    values["capabilities"] = (
        {
            key: capabilities.get(key) if type(capabilities.get(key)) is bool else None
            for key in (
                "transcript",
                "semantic_search",
                "read_frames",
                "read_text",
                "local_proposals",
            )
        }
        if isinstance(capabilities, dict)
        else {}
    )
    page = data.get("next_page")
    if isinstance(page, dict):
        start, end = page.get("core_start_sec"), page.get("core_end_sec")
        if (
            all(type(value) in (int, float) and math.isfinite(value) for value in (start, end))
            and 0 <= start < end
        ):
            values["next_scan_range"] = {"start_sec": start, "end_sec": end}
    with suppress(OSError):
        values["updated_at"] = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
    return WorkflowSnapshot(**values)


def public_progress(data: dict, snapshot: WorkflowSnapshot) -> dict:
    """Retain public counters and capability flags, excluding opaque Agent state."""
    result = snapshot.model_dump(
        exclude_none=True,
        exclude={"schema_version", "updated_at", "next_scan_range"},
    )
    for key in (
        "scan_coverage",
        "pending_event_count",
        "pending_observation_count",
        "pending_proposal_count",
        "pending_review_count",
    ):
        value = data.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            continue
        if key == "scan_coverage":
            if value <= 1:
                result[key] = value
        elif value == int(value):
            result[key] = int(value)
    return result
