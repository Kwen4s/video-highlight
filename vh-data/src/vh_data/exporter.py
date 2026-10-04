"""Freeze eligible labels and their provenance into a training snapshot."""

import json
import shutil
import tempfile
from pathlib import Path

from .models import Annotation, digest
from .quality import overlapping_highlights, union
from .store import ConflictError, Store


def export(store: Store, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError("导出目录已存在，请给新一轮数据使用新名称")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows, versions = [], []
    counts = {name: 0 for name in ("train", "val", "test")}
    labels = {kind: 0 for kind in ("positive", "negative", "uncertain")}
    total_sec = supervised_sec = highlight_sec = longest_highlight_sec = 0.0
    with store.connect() as db:
        # One database read yields a coherent snapshot even while reviewers save new versions.
        records = db.execute(
            "SELECT v.*,a.source,a.payload,a.provenance FROM videos v "
            "JOIN annotations a ON v.label_id=a.id WHERE v.status='ready' ORDER BY v.video_id"
        ).fetchall()
    for video in records:
        if video["source"] != "model_reviewed":
            continue
        if digest(Path(video["path"])) != video["sha256"]:
            raise ConflictError(f"{video['video_id']} 的源视频已改变，不能导出")
        annotation = Annotation.model_validate_json(video["payload"])
        if overlapping_highlights(annotation.segments):
            raise ConflictError(f"{video['video_id']} 的高光仍有重叠，不能导出")
        positive = [s for s in annotation.segments if s.kind == "positive"]
        negative = [s for s in annotation.segments if s.kind == "negative"]
        if not positive and not negative:
            continue
        for segment in annotation.segments:
            labels[segment.kind] += 1
        total_sec += annotation.duration_sec
        highlight_sec += sum(s.end_sec - s.start_sec for s in positive)
        supervised_sec += sum(s.end_sec - s.start_sec for s in union([*positive, *negative]))
        longest_highlight_sec = max(
            longest_highlight_sec, max((s.end_sec - s.start_sec for s in positive), default=0)
        )
        rows.append(
            {
                "video_id": video["video_id"],
                "drama_id": video["drama_id"],
                "path": video["path"],
                "sha256": video["sha256"],
                "duration_sec": annotation.duration_sec,
                "language": video["language"],
                "split": video["split"],
                "label_source": video["source"],
                "highlights": [{"start_sec": s.start_sec, "end_sec": s.end_sec} for s in positive],
                "negative_intervals": [
                    {"start_sec": s.start_sec, "end_sec": s.end_sec} for s in negative
                ],
            }
        )
        versions.append(
            {
                "video_id": video["video_id"],
                "label_id": video["label_id"],
                "source": video["source"],
                "provenance": json.loads(video["provenance"]),
                "annotation": annotation.model_dump(),
            }
        )
        counts[video["split"]] += 1
    if not rows:
        raise ValueError("没有模型复核通过的标注；先运行标注任务")
    temporary = Path(tempfile.mkdtemp(prefix=".export-", dir=destination.parent))
    try:
        annotations = temporary / "annotations.jsonl"
        annotations.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
        (temporary / "provenance.json").write_text(
            json.dumps({"versions": versions}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report = {
            "format": "vh-data-interval-v1",
            "videos": len(rows),
            "splits": counts,
            "dramas": len({row["drama_id"] for row in rows}),
            "labels": labels,
            "duration_sec": {
                "total": round(total_sec, 3),
                "supervised": round(supervised_sec, 3),
                "highlight": round(highlight_sec, 3),
                "unknown": round(total_sec - supervised_sec, 3),
                "longest_highlight": round(longest_highlight_sec, 3),
            },
            "highlight_ratio": round(highlight_sec / total_sec, 4),
            "label_source": "model_reviewed",
            "annotations_sha256": digest(annotations),
            "provenance_sha256": digest(temporary / "provenance.json"),
            "unmarked_regions": "unknown",
            "selection": "val selects weights; test is evaluated once",
            "reference": "model_reviewed labels",
        }
        (temporary / "manifest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.rename(destination)
        return report
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
