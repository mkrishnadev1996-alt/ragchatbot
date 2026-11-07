"""SQLite-backed user storage."""

import logging
from contextlib import closing
from dataclasses import dataclass

from utils.db import Database, as_database

logger = logging.getLogger(__name__)


class UserExistsError(Exception):
    """Raised when registering an email that is already taken."""


@dataclass
class User:
    id: int
    email: str
    password_hash: str


class UserStore:
    def __init__(self, db: Database | str):
        self.db = as_database(db)
        # Email is unique case-insensitively: NOCASE collation in sqlite, a lower() index in
        # PostgreSQL (lookups use lower(email) in both).
        if self.db.dialect == "sqlite":
            email_column = "email TEXT NOT NULL UNIQUE COLLATE NOCASE"
            extra: list[str] = []
        else:
            email_column = "email TEXT NOT NULL"
            extra = [
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_lower ON users (lower(email))"
            ]
        self.db.run_ddl(
            [
                f"""
                CREATE TABLE IF NOT EXISTS users (
                    id {self.db.pk_column()},
                    {email_column},
                    password_hash TEXT NOT NULL,
                    created_at {self.db.timestamp_column()}
                )
                """,
                *extra,
            ]
        )

    def _connect(self):
        return self.db.connect()

    def create_user(self, email: str, password_hash: str) -> User:
        try:
            with closing(self._connect()) as conn, conn:
                row = conn.execute(
                    "INSERT INTO users (email, password_hash) VALUES (?, ?) RETURNING id",
                    (email, password_hash),
                ).fetchone()
        except self.db.integrity_errors as e:
            raise UserExistsError(email) from e
        if row is None:
            raise RuntimeError("the database returned no id for the new user")
        user_id = row[0]
        return User(id=user_id, email=email, password_hash=password_hash)

    def _get(self, where: str, param: object) -> User | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                f"SELECT id, email, password_hash FROM users WHERE {where}", (param,)
            ).fetchone()
        return User(*row) if row else None

    def get_by_email(self, email: str) -> User | None:
        return self._get("lower(email) = lower(?)", email)

    def get_by_id(self, user_id: int) -> User | None:
        return self._get("id = ?", user_id)
