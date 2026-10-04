"""SQLite stores sources, immutable annotations and detector feedback."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .models import Annotation, VideoPredictions, digest, iou
from .quality import union


class ConflictError(ValueError):
    pass


class Store:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "data.sqlite3"
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS videos (
                    video_id TEXT PRIMARY KEY, drama_id TEXT NOT NULL, path TEXT NOT NULL,
                    sha256 TEXT NOT NULL UNIQUE, duration_sec REAL NOT NULL, language TEXT NOT NULL,
                    split TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued',
                    label_id INTEGER, priority REAL NOT NULL DEFAULT 0, error TEXT
                );
                CREATE TABLE IF NOT EXISTS annotations (
                    id INTEGER PRIMARY KEY, video_id TEXT NOT NULL REFERENCES videos(video_id),
                    source TEXT NOT NULL, payload TEXT NOT NULL, provenance TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS predictions (
                    video_id TEXT NOT NULL REFERENCES videos(video_id), model_id TEXT NOT NULL,
                    payload TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(video_id, model_id)
                );
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def ingest(self, manifests: list[Path]) -> dict:
        added = duplicates = 0
        with self.connect() as db:
            for manifest in manifests:
                with manifest.open() as stream:
                    for line in stream:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        path = Path(row["path"]).expanduser()
                        path = (
                            manifest.parent / path if not path.is_absolute() else path
                        ).resolve()
                        if not path.is_file():
                            raise FileNotFoundError(path)
                        sha = digest(path)
                        video_id, drama_id = str(row["video_id"]), str(row["drama_id"])
                        duration = float(row["duration_sec"])
                        if not video_id.strip() or not drama_id.strip():
                            raise ValueError("视频和短剧 ID 不能为空")
                        if not math.isfinite(duration) or duration <= 0:
                            raise ValueError("视频时长必须是有限正数")
                        if row.get("sha256") and row["sha256"] != sha:
                            raise ConflictError(f"{video_id} 的清单指纹与源视频不一致")
                        existing = db.execute(
                            "SELECT * FROM videos WHERE video_id=? OR sha256=?", (video_id, sha)
                        ).fetchone()
                        if existing:
                            if existing["video_id"] == video_id and existing["sha256"] != sha:
                                raise ConflictError(f"{video_id} 的源视频已改变")
                            if existing["drama_id"] != drama_id:
                                raise ConflictError("同一源视频不能属于两个短剧分组")
                            duplicates += 1
                            continue
                        bucket = int(hashlib.sha256(drama_id.encode()).hexdigest()[:8], 16) % 10
                        split = "train" if bucket < 8 else "val" if bucket == 8 else "test"
                        db.execute(
                            "INSERT INTO videos(video_id,drama_id,path,sha256,duration_sec,"
                            "language,split) "
                            "VALUES(?,?,?,?,?,?,?)",
                            (video_id, drama_id, str(path), sha, duration, row["language"], split),
                        )
                        added += 1
        return {"added": added, "duplicates": duplicates}

    def get(self, video_id: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT * FROM videos WHERE video_id=?", (video_id,)).fetchone()
            if row is None:
                raise KeyError(video_id)
            video = dict(row)
            label = db.execute(
                "SELECT * FROM annotations WHERE id=?", (row["label_id"],)
            ).fetchone()
            predictions = db.execute(
                "SELECT model_id,payload FROM predictions WHERE video_id=? "
                "ORDER BY created_at DESC,rowid DESC LIMIT 1",
                (video_id,),
            ).fetchall()
        video["annotation"] = json.loads(label["payload"]) if label else None
        video["source"] = label["source"] if label else None
        video["provenance"] = json.loads(label["provenance"]) if label else None
        video["predictions"] = [
            {"model_id": p["model_id"], "segments": json.loads(p["payload"])["segments"]}
            for p in predictions
        ]
        return video

    def recover(
        self, retry_failed: bool, split: str | None = None, video_ids: list[str] | None = None
    ):
        with self.connect() as db:
            db.execute("UPDATE videos SET status='queued' WHERE status='labeling'")
            if retry_failed:
                selection = ""
                parameters = [split, split]
                if video_ids:
                    selection = f"AND video_id IN ({','.join('?' for _ in video_ids)})"
                    parameters.extend(video_ids)
                db.execute(
                    "UPDATE videos SET status='queued',error=NULL WHERE status='failed' "
                    "AND (? IS NULL OR split=?) " + selection,
                    parameters,
                )

    def requeue(self, video_ids: list[str]) -> dict:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for video_id in set(video_ids):
                row = db.execute(
                    "SELECT status FROM videos WHERE video_id=?", (video_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(video_id)
                if row["status"] == "labeling":
                    raise ConflictError("视频正在标注，不能重复入队")
                db.execute(
                    "UPDATE videos SET status='queued',error=NULL WHERE video_id=?", (video_id,)
                )
        return {"queued": len(set(video_ids))}

    def claim(self, split: str | None = None, video_ids: list[str] | None = None) -> dict | None:
        selection = ""
        parameters = [split, split]
        if video_ids:
            selection = f"AND v.video_id IN ({','.join('?' for _ in video_ids)}) "
            parameters.extend(video_ids)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT v.* FROM videos v WHERE status IN ('queued','review') "
                "AND (? IS NULL OR split=?) "
                + selection
                + "ORDER BY priority DESC, (SELECT COUNT(*) FROM videos x "
                "JOIN annotations a ON a.id=x.label_id WHERE x.drama_id=v.drama_id "
                "AND x.status='ready' AND a.source='model_reviewed'), "
                "duration_sec, video_id LIMIT 1",
                parameters,
            ).fetchone()
            if row is None:
                return None
            db.execute(
                "UPDATE videos SET status='labeling',error=NULL WHERE video_id=?",
                (row["video_id"],),
            )
        return self.get(row["video_id"])

    def fail(self, video_id: str, error: str):
        with self.connect() as db:
            db.execute(
                "UPDATE videos SET status='failed',error=? WHERE video_id=? AND status='labeling'",
                (error, video_id),
            )

    def save(
        self,
        video_id: str,
        annotation: Annotation,
        source: str,
        provenance: dict,
        *,
        expected_label_id: int | None,
    ):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT label_id FROM videos WHERE video_id=?", (video_id,)).fetchone()
            if row is None:
                raise KeyError(video_id)
            if row["label_id"] != expected_label_id:
                raise ConflictError("标注已更新，请重新打开后再保存")
            cursor = db.execute(
                "INSERT INTO annotations(video_id,source,payload,provenance) VALUES(?,?,?,?)",
                (
                    video_id,
                    source,
                    annotation.model_dump_json(),
                    json.dumps(provenance, ensure_ascii=False),
                ),
            )
            db.execute(
                "UPDATE videos SET label_id=?,duration_sec=?,status=?,priority=0,error=NULL "
                "WHERE video_id=?",
                (
                    cursor.lastrowid,
                    annotation.duration_sec,
                    "ready"
                    if any(s.kind != "uncertain" for s in annotation.segments)
                    else "unresolved",
                    video_id,
                ),
            )

    def feedback(
        self, predictions: list[VideoPredictions], model_id: str, threshold: float
    ) -> dict:
        if not model_id.strip():
            raise ValueError("必须记录模型版本")
        if len({p.video_id for p in predictions}) != len(predictions):
            raise ValueError("一次反馈中每个视频只能出现一次")
        queued = unchanged = 0
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for prediction in predictions:
                video = self.get(prediction.video_id)
                if video["split"] != "train":
                    raise ValueError("反馈采样只使用训练组，验证与测试组保持独立")
                if video["status"] == "labeling":
                    raise ConflictError("视频正在标注，请完成后再导入反馈")
                if any(s.end_sec > video["duration_sec"] for s in prediction.segments):
                    raise ValueError("模型预测超出原视频范围")
                if digest(Path(video["path"])) != video["sha256"]:
                    raise ConflictError("模型反馈对应的原视频已改变")
                payload = prediction.model_dump_json()
                previous = db.execute(
                    "SELECT payload FROM predictions WHERE video_id=? AND model_id=?",
                    (prediction.video_id, model_id),
                ).fetchone()
                if previous and previous["payload"] == payload:
                    unchanged += 1
                    continue
                db.execute(
                    "INSERT INTO predictions(video_id,model_id,payload,created_at) VALUES(?,?,?,?) "
                    "ON CONFLICT(video_id,model_id) DO UPDATE SET "
                    "payload=excluded.payload,created_at=excluded.created_at",
                    (prediction.video_id, model_id, payload, datetime.now(UTC).isoformat()),
                )
                if video["annotation"]:
                    labels = Annotation.model_validate(video["annotation"])
                    positives = [s for s in labels.segments if s.kind == "positive"]
                    missed = sum(
                        not any(iou(g, p) >= threshold for p in prediction.segments)
                        for g in positives
                    )
                    extra = sum(
                        not any(iou(p, g) >= threshold for g in positives)
                        for p in prediction.segments
                    )
                    priority = missed + extra
                    status = "review" if priority else video["status"]
                else:
                    priority = len(prediction.segments)
                    status = "queued" if priority else video["status"]
                db.execute(
                    "UPDATE videos SET priority=?,status=? WHERE video_id=?",
                    (priority, status, prediction.video_id),
                )
                queued += bool(priority)
        return {"queued_for_review": queued, "unchanged": unchanged}

    def stats(self) -> dict:
        with self.connect() as db:
            statuses = {
                r["status"]: r["n"]
                for r in db.execute("SELECT status,COUNT(*) n FROM videos GROUP BY status")
            }
            splits = {
                r["split"]: r["n"]
                for r in db.execute("SELECT split,COUNT(*) n FROM videos GROUP BY split")
            }
            sources = {
                r["source"]: r["n"]
                for r in db.execute(
                    "SELECT a.source,COUNT(*) n FROM videos v "
                    "JOIN annotations a ON a.id=v.label_id WHERE v.status='ready' "
                    "AND a.source='model_reviewed' "
                    "GROUP BY a.source"
                )
            }
            ready_splits = dict.fromkeys(("train", "val", "test"), 0)
            ready_splits.update(
                (r["split"], r["n"])
                for r in db.execute(
                    "SELECT v.split,COUNT(*) n FROM videos v JOIN annotations a ON a.id=v.label_id "
                    "WHERE v.status='ready' AND a.source='model_reviewed' GROUP BY v.split"
                )
            )
            dramas = db.execute("SELECT COUNT(DISTINCT drama_id) FROM videos").fetchone()[0]
            annotations = [
                Annotation.model_validate_json(r["payload"])
                for r in db.execute(
                    "SELECT a.payload FROM videos v JOIN annotations a ON a.id=v.label_id "
                    "WHERE v.status='ready' AND a.source='model_reviewed'"
                )
            ]
        highlight_sec = sum(
            s.end_sec - s.start_sec
            for annotation in annotations
            for s in union([s for s in annotation.segments if s.kind == "positive"])
        )
        total_sec = sum(a.duration_sec for a in annotations)
        return {
            "videos": sum(statuses.values()),
            "dramas": dramas,
            "status": statuses,
            "splits": splits,
            "ready_splits": ready_splits,
            "labels": sources,
            "quality": {
                "highlight_sec": round(highlight_sec, 3),
                "total_sec": round(total_sec, 3),
                "highlight_ratio": round(highlight_sec / total_sec, 4) if total_sec else 0,
            },
        }
