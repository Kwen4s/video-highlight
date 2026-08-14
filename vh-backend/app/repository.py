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
                    error_message TEXT,
                    result_json TEXT
                )
                """
            )

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
                "UPDATE jobs SET status = ?, error_message = ?, updated_at = ? WHERE job_id = ?",
                (status, error_message, utc_now(), job_id),
            )

    def save_result(self, job_id: str, result: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = 'completed', result_json = ?, error_message = NULL, updated_at = ?
                WHERE job_id = ?
                """,
                (json.dumps(result, ensure_ascii=False), utc_now(), job_id),
            )

    def replace_result(self, job_id: str, result: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET result_json = ?, updated_at = ? WHERE job_id = ?",
                (json.dumps(result, ensure_ascii=False), utc_now(), job_id),
            )

    def delete(self, job_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection
