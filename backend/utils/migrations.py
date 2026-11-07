"""Versioned schema changes for the shared database (users, feedback, messages).

Each store still creates its own tables with `CREATE TABLE IF NOT EXISTS`; that is version 0.
Anything added afterwards (indexes, new columns) goes in MIGRATIONS as a new entry at the end,
written in SQL that works on both sqlite and PostgreSQL. The applied version is kept in
sqlite's `PRAGMA user_version`, or in a `schema_version` table on PostgreSQL. Never edit or
reorder an entry that has shipped.
"""

import logging
from contextlib import closing

from utils.db import Database, as_database, schema_lock_id

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


def migrate(db: Database | str, migrations: list[tuple[int, list[str]]] | None = None) -> int:
    """Apply pending migrations in order; returns the resulting schema version.

    Safe to call from several workers at once: the first takes a write lock, the others wait
    and then find nothing left to do. A failing migration is rolled back and re-raised.
    """
    database = as_database(db)
    migrations = MIGRATIONS if migrations is None else migrations
    if database.dialect == "postgres":
        return _migrate_postgres(database, migrations)
    return _migrate_sqlite(database, migrations)


def _migrate_sqlite(database: Database, migrations: list[tuple[int, list[str]]]) -> int:
    with closing(database.connect()) as conn:
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


def _migrate_postgres(database: Database, migrations: list[tuple[int, list[str]]]) -> int:
    """One transaction under an advisory lock (PostgreSQL DDL is transactional)."""
    with closing(database.connect()) as conn:
        try:
            with conn:
                conn.execute("SELECT pg_advisory_xact_lock(?)", (schema_lock_id(),))
                conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
                row = conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version")
                current = row.fetchone()[0]
                for version, statements in migrations:
                    if version <= current:
                        continue
                    for statement in statements:
                        conn.execute(statement)
                    conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
                    logger.info("Applied database migration %d", version)
                    current = version
                return current
        except Exception:
            logger.exception("Database migration failed")
            raise
