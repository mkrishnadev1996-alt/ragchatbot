"""SQLite-backed storage for chat history, so conversations survive restarts and re-logins."""

import json
import logging
import sqlite3
from contextlib import closing

logger = logging.getLogger(__name__)


class MessageStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_id TEXT,
                    message_id TEXT,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    sources TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_user ON messages (user_id, id)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def add_exchange(
        self,
        user_id: int,
        session_id: str,
        question: str,
        answer: str,
        sources: list[str],
        message_id: str,
    ) -> None:
        """Save a question and its answer atomically."""
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT INTO messages (user_id, session_id, role, content)
                VALUES (?, ?, 'user', ?)
                """,
                (user_id, session_id, question),
            )
            conn.execute(
                """
                INSERT INTO messages (user_id, session_id, message_id, role, content, sources)
                VALUES (?, ?, ?, 'assistant', ?, ?)
                """,
                (user_id, session_id, message_id, answer, json.dumps(sources)),
            )

    def list_messages(
        self, user_id: int, limit: int = 200, session_id: str | None = None
    ) -> list[dict]:
        """The user's most recent messages (optionally of one session), oldest first,
        with any thumbs rating attached."""
        with closing(self._connect()) as conn:
            # The feedback table is created by FeedbackStore; make sure it exists for the join.
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
            rows = conn.execute(
                """
                SELECT m.role, m.content, m.sources, m.message_id, f.rating
                FROM messages m
                LEFT JOIN feedback f ON f.message_id = m.message_id AND f.user_id = m.user_id
                WHERE m.user_id = ? AND (? IS NULL OR m.session_id = ?)
                ORDER BY m.id DESC
                LIMIT ?
                """,
                (user_id, session_id, session_id, limit),
            ).fetchall()
        return [
            {
                "role": role,
                "content": content,
                "sources": json.loads(sources) if sources else [],
                "message_id": message_id,
                "rating": rating,
            }
            for role, content, sources, message_id, rating in reversed(rows)
        ]
