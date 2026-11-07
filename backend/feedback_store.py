"""SQLite-backed storage for per-answer thumbs up/down feedback."""

import logging
import sqlite3
from contextlib import closing

logger = logging.getLogger(__name__)


class FeedbackStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    message_id TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    session_id TEXT,
                    rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (message_id, user_id)
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        # One connection per call keeps this safe across uvicorn worker threads.
        return sqlite3.connect(self.db_path)

    def set_rating(
        self, message_id: str, user_id: int, rating: str, session_id: str | None = None
    ) -> None:
        """Record a rating; rating the same message again replaces the earlier one."""
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT INTO feedback (message_id, user_id, session_id, rating)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (message_id, user_id) DO UPDATE SET
                    rating = excluded.rating,
                    session_id = excluded.session_id,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (message_id, user_id, session_id, rating),
            )

    def clear_rating(self, message_id: str, user_id: int) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "DELETE FROM feedback WHERE message_id = ? AND user_id = ?",
                (message_id, user_id),
            )

    def get_rating(self, message_id: str, user_id: int) -> str | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT rating FROM feedback WHERE message_id = ? AND user_id = ?",
                (message_id, user_id),
            ).fetchone()
        return row[0] if row else None
