from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class JobRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    original_name TEXT NOT NULL,
                    stored_name TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    language TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 0,
                    history_json TEXT NOT NULL DEFAULT '[]',
                    conversation_json TEXT NOT NULL DEFAULT '[]',
                    error_message TEXT,
                    result_json TEXT
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
            }
            additions = {
                "revision": "INTEGER NOT NULL DEFAULT 0",
                "history_json": "TEXT NOT NULL DEFAULT '[]'",
                "conversation_json": "TEXT NOT NULL DEFAULT '[]'",
            }
            for name, declaration in additions.items():
                if name not in columns:
                    connection.execute(f"ALTER TABLE jobs ADD COLUMN {name} {declaration}")

    def create(
        self,
        *,
        job_id: str,
        original_name: str,
        stored_name: str,
        content_type: str,
        size_bytes: int,
        language: str,
    ) -> dict[str, Any]:
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    job_id, status, original_name, stored_name, content_type,
                    size_bytes, language, created_at, updated_at
                ) VALUES (?, 'queued', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    original_name,
                    stored_name,
                    content_type,
                    size_bytes,
                    language,
                    now,
                    now,
                ),
            )
        return self.get(job_id)  # type: ignore[return-value]

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def set_status(
        self,
        job_id: str,
        status: str,
        *,
        error_message: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = ?, error_message = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (status, error_message, utc_now(), job_id),
            )

    def save_result(
        self,
        job_id: str,
        result: dict[str, Any],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = 'completed', result_json = ?, error_message = NULL,
                    revision = 0, history_json = '[]', conversation_json = '[]',
                    updated_at = ?
                WHERE job_id = ?
                """,
                (
                    json.dumps(result, ensure_ascii=False),
                    utc_now(),
                    job_id,
                ),
            )

    def open_demo_session(
        self,
        *,
        job_id: str,
        original_name: str,
        size_bytes: int,
        language: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    job_id, status, original_name, stored_name, content_type,
                    size_bytes, language, created_at, updated_at,
                    revision, history_json, conversation_json, error_message, result_json
                ) VALUES (?, 'completed', ?, '', 'video/mp4', ?, ?, ?, ?, 0, '[]', '[]', NULL, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status = 'completed',
                    original_name = excluded.original_name,
                    stored_name = '',
                    content_type = 'video/mp4',
                    size_bytes = excluded.size_bytes,
                    language = excluded.language,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at,
                    revision = 0,
                    history_json = '[]',
                    conversation_json = '[]',
                    error_message = NULL,
                    result_json = excluded.result_json
                """,
                (
                    job_id,
                    original_name,
                    size_bytes,
                    language,
                    now,
                    now,
                    json.dumps(result, ensure_ascii=False),
                ),
            )
        return self.get(job_id)  # type: ignore[return-value]

    def replace_result(
        self,
        job_id: str,
        result: dict[str, Any],
        *,
        expected_revision: int,
    ) -> bool:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT revision, result_json, history_json FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None or row["revision"] != expected_revision or not row["result_json"]:
                return False
            history = json.loads(row["history_json"] or "[]")
            history.append(json.loads(row["result_json"]))
            history = history[-20:]
            connection.execute(
                """
                UPDATE jobs
                SET result_json = ?, history_json = ?, revision = revision + 1,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (
                    json.dumps(result, ensure_ascii=False),
                    json.dumps(history, ensure_ascii=False),
                    utc_now(),
                    job_id,
                ),
            )
        return True

    def undo_result(
        self,
        job_id: str,
        *,
        expected_revision: int,
    ) -> bool | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT revision, history_json FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None or row["revision"] != expected_revision:
                return False
            history = json.loads(row["history_json"] or "[]")
            if not history:
                return None
            previous = history.pop()
            connection.execute(
                """
                UPDATE jobs
                SET result_json = ?, history_json = ?, revision = revision + 1,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (
                    json.dumps(previous, ensure_ascii=False),
                    json.dumps(history, ensure_ascii=False),
                    utc_now(),
                    job_id,
                ),
            )
        return True

    def get_conversation(self, job_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT conversation_json FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return []
        conversation = json.loads(row["conversation_json"] or "[]")
        return conversation if isinstance(conversation, list) else []

    def append_conversation(
        self,
        job_id: str,
        *,
        expected_revision: int,
        user_message: str,
        assistant_reply: str,
        action: dict[str, Any] | None,
    ) -> bool:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT revision, conversation_json FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None or row["revision"] != expected_revision:
                return False
            conversation = json.loads(row["conversation_json"] or "[]")
            if not isinstance(conversation, list):
                conversation = []
            conversation.append(
                {
                    "user_message": user_message,
                    "assistant_reply": assistant_reply,
                    "action": action,
                    "revision": expected_revision,
                    "created_at": utc_now(),
                }
            )
            conversation = conversation[-20:]
            cursor = connection.execute(
                """
                UPDATE jobs SET conversation_json = ?, updated_at = ?
                WHERE job_id = ? AND revision = ?
                """,
                (
                    json.dumps(conversation, ensure_ascii=False),
                    utc_now(),
                    job_id,
                    expected_revision,
                ),
            )
        return cursor.rowcount == 1

    def delete(self, job_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection
