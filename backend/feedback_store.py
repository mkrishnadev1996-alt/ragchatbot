"""SQLite-backed storage for per-answer thumbs up/down feedback."""

import logging
from contextlib import closing

from utils.db import Database, as_database

logger = logging.getLogger(__name__)


def feedback_schema(db: Database) -> str:
    """The one definition of the feedback table (MessageStore creates it too, for its join)"""
    return f"""
        CREATE TABLE IF NOT EXISTS feedback (
            message_id TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            session_id TEXT,
            rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
            updated_at {db.timestamp_column()},
            PRIMARY KEY (message_id, user_id)
        )
    """


class FeedbackStore:
    def __init__(self, db: Database | str):
        self.db = as_database(db)
        self.db.run_ddl([feedback_schema(self.db)])

    def _connect(self):
        return self.db.connect()

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
