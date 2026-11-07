"""Versioned schema changes for the shared sqlite file (users, feedback, messages).

Each store still creates its own tables with `CREATE TABLE IF NOT EXISTS`; that is version 0.
Anything added afterwards (indexes, new columns) goes in MIGRATIONS as a new entry at the end.
The applied version is kept in sqlite's `PRAGMA user_version`. Never edit or reorder an entry
that has shipped.
"""

import logging
from contextlib import closing

from utils.db import connect

logger = logging.getLogger(__name__)

# (version, statements). Versions must be 1, 2, 3, ... with no gaps.
MIGRATIONS: list[tuple[int, list[str]]] = [
    (
        1,
        [
            # Daily quota lookups and retention purges
            "CREATE INDEX IF NOT EXISTS idx_messages_user_role_created "
            "ON messages (user_id, role, created_at)",
            "CREATE INDEX IF NOT EXISTS idx_messages_created ON messages (created_at)",
        ],
    ),
]


def migrate(db_path: str, migrations: list[tuple[int, list[str]]] | None = None) -> int:
    """Apply pending migrations in order, each in its own transaction; returns the version."""
    migrations = MIGRATIONS if migrations is None else migrations
    with closing(connect(db_path)) as conn:
        current = conn.execute("PRAGMA user_version").fetchone()[0]
        for version, statements in migrations:
            if version <= current:
                continue
            try:
                conn.execute("BEGIN IMMEDIATE")
                # Another worker may have applied it while we waited for the write lock
                if conn.execute("PRAGMA user_version").fetchone()[0] >= version:
                    conn.execute("ROLLBACK")
                    current = version
                    continue
                for statement in statements:
                    conn.execute(statement)
                conn.execute(f"PRAGMA user_version = {int(version)}")
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                logger.exception("Database migration %d failed", version)
                raise
            logger.info("Applied database migration %d", version)
            current = version
        return current
