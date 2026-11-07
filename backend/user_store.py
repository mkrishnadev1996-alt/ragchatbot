"""SQLite-backed user storage."""

import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class UserExistsError(Exception):
    """Raised when registering an email that is already taken."""


@dataclass
class User:
    id: int
    email: str
    password_hash: str


class UserStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        # One connection per call keeps this safe across uvicorn worker threads.
        return sqlite3.connect(self.db_path)

    def create_user(self, email: str, password_hash: str) -> User:
        try:
            with closing(self._connect()) as conn, conn:
                cur = conn.execute(
                    "INSERT INTO users (email, password_hash) VALUES (?, ?)",
                    (email, password_hash),
                )
                user_id = cur.lastrowid
        except sqlite3.IntegrityError as e:
            raise UserExistsError(email) from e
        assert user_id is not None
        return User(id=user_id, email=email, password_hash=password_hash)

    def _get(self, where: str, param: object) -> User | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                f"SELECT id, email, password_hash FROM users WHERE {where} = ?", (param,)
            ).fetchone()
        return User(*row) if row else None

    def get_by_email(self, email: str) -> User | None:
        return self._get("email", email)

    def get_by_id(self, user_id: int) -> User | None:
        return self._get("id", user_id)
