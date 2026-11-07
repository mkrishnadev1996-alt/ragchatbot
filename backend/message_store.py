"""SQLite-backed storage for chat history, so conversations survive restarts and re-logins."""

import json
import logging
from contextlib import closing

from feedback_store import feedback_schema
from models import Source
from utils.db import Database, as_database

logger = logging.getLogger(__name__)


def _load_sources(raw: str | None) -> list[dict]:
    """Stored sources; rows saved before links existed hold plain label strings"""
    items = json.loads(raw) if raw else []
    return [{"label": i, "url": None} if isinstance(i, str) else i for i in items]


class MessageStore:
    def __init__(self, db: Database | str):
        self.db = as_database(db)
        self.db.run_ddl(
            [
                f"""
                CREATE TABLE IF NOT EXISTS messages (
                    id {self.db.pk_column()},
                    user_id INTEGER NOT NULL,
                    session_id TEXT,
                    message_id TEXT,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    sources TEXT,
                    created_at {self.db.timestamp_column()}
                )
                """,
                "CREATE INDEX IF NOT EXISTS idx_messages_user ON messages (user_id, id)",
                feedback_schema(self.db),  # list_messages joins against it
            ]
        )

    def _connect(self):
        return self.db.connect()

    def add_exchange(
        self,
        user_id: int,
        session_id: str,
        question: str,
        answer: str,
        sources: list[Source],
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
                (
                    user_id,
                    session_id,
                    message_id,
                    answer,
                    json.dumps([s.model_dump() for s in sources]),
                ),
            )

    def count_questions_since(self, user_id: int, hours: float = 24) -> int:
        """How many questions the user has asked in the last `hours` (for the daily quota)."""
        with closing(self._connect()) as conn:
            (count,) = conn.execute(
                f"""
                SELECT COUNT(*) FROM messages
                WHERE user_id = ? AND role = 'user' AND created_at >= {self.db.seconds_ago_sql()}
                """,
                (user_id, float(hours) * 3600),
            ).fetchone()
        return count

    def purge_older_than(self, days: float) -> int:
        """Delete messages older than `days` days; returns how many rows were removed."""
        with closing(self._connect()) as conn, conn:
            cur = conn.execute(
                f"DELETE FROM messages WHERE created_at < {self.db.seconds_ago_sql()}",
                (float(days) * 86400,),
            )
            return cur.rowcount

    def list_messages(
        self, user_id: int, limit: int = 200, session_id: str | None = None
    ) -> list[dict]:
        """The user's most recent messages (optionally of one session), oldest first,
        with any thumbs rating attached."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT m.role, m.content, m.sources, m.message_id, f.rating
                FROM messages m
                LEFT JOIN feedback f ON f.message_id = m.message_id AND f.user_id = m.user_id
                WHERE m.user_id = ? AND (CAST(? AS TEXT) IS NULL OR m.session_id = ?)
                ORDER BY m.id DESC
                LIMIT ?
                """,
                (user_id, session_id, session_id, limit),
            ).fetchall()
        return [
            {
                "role": role,
                "content": content,
                "sources": _load_sources(sources),
                "message_id": message_id,
                "rating": rating,
            }
            for role, content, sources, message_id, rating in reversed(rows)
        ]
