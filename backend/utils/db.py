"""Database access shared by the user, feedback and message stores.

Two backends behind one small interface: a local sqlite file (default, fine for a single
process) or PostgreSQL (`DATABASE_URL`, needed when several workers/replicas share the data).
Stores write SQL with `?` placeholders and use the dialect helpers below for the few places
where sqlite and PostgreSQL differ (primary keys, timestamps, "N seconds ago").

A connection supports `execute(sql, params)`, `with conn:` (commit, or rollback on error) and
`close()`, so stores use it the same way for both backends:
`with closing(db.connect()) as conn, conn:`.
"""

import logging
import sqlite3
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)

# Defaults only: the app passes config.DB_BUSY_TIMEOUT and the DATABASE_* settings in
BUSY_TIMEOUT_SECONDS = 10
_SCHEMA_LOCK_ID = 727274  # arbitrary app-wide advisory lock id for DDL / migrations


def connect(db_path: str, busy_timeout: float = BUSY_TIMEOUT_SECONDS) -> sqlite3.Connection:
    """Open a sqlite connection (one per call keeps this safe across worker threads).

    WAL lets readers and the writer work at the same time, and the busy timeout makes a
    writer wait for a lock instead of failing with "database is locked".
    """
    conn = sqlite3.connect(db_path, timeout=busy_timeout)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


class Database:
    """Common interface. Use `open_database` / `as_database` rather than instantiating."""

    dialect: str = ""
    integrity_errors: tuple[type[Exception], ...] = ()

    def connect(self) -> Any:
        raise NotImplementedError

    def close(self) -> None:
        """Release pooled resources (no-op for sqlite)."""

    # --- dialect helpers -------------------------------------------------------------------
    def pk_column(self) -> str:
        raise NotImplementedError

    def timestamp_column(self) -> str:
        """A NOT NULL timestamp column type that defaults to the current time."""
        raise NotImplementedError

    def seconds_ago_sql(self) -> str:
        """SQL expression for "now minus ? seconds" (one `?` placeholder, a number)."""
        raise NotImplementedError

    def run_ddl(self, statements: Iterable[str]) -> None:
        """Run schema statements once, safely even if several workers start together."""
        raise NotImplementedError

    def ping(self) -> None:
        """Raise if the database is unreachable."""
        conn = self.connect()
        try:
            conn.execute("SELECT 1").fetchone()
        finally:
            conn.close()


class SqliteDatabase(Database):
    dialect = "sqlite"
    integrity_errors = (sqlite3.IntegrityError,)

    def __init__(self, path: str, busy_timeout: float = BUSY_TIMEOUT_SECONDS):
        self.path = path
        self.busy_timeout = busy_timeout

    def connect(self) -> sqlite3.Connection:
        return connect(self.path, self.busy_timeout)

    def pk_column(self) -> str:
        return "INTEGER PRIMARY KEY AUTOINCREMENT"

    def timestamp_column(self) -> str:
        return "TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP"

    def seconds_ago_sql(self) -> str:
        return "datetime('now', '-' || ? || ' seconds')"

    def run_ddl(self, statements: Iterable[str]) -> None:
        conn = self.connect()
        try:
            with conn:
                for statement in statements:
                    conn.execute(statement)
        finally:
            conn.close()


class _PgConnection:
    """psycopg connection behind the sqlite-style interface (`?` placeholders, `with` commits)."""

    def __init__(self, pool: Any, conn: Any):
        self._pool = pool
        self._conn: Any = conn

    def execute(self, sql: str, params: Iterable[Any] = ()) -> Any:
        return self._conn.execute(sql.replace("?", "%s"), tuple(params))

    def __enter__(self) -> "_PgConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self._conn.commit()
        else:
            self._conn.rollback()

    def close(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:  # closing twice is harmless
            conn.rollback()  # ends the implicit transaction a read-only query opens (else the
            # pool logs a warning on every return); a no-op after commit
            self._pool.putconn(conn)


class PostgresDatabase(Database):
    dialect = "postgres"

    def __init__(
        self,
        url: str,
        max_size: int = 10,
        min_size: int = 1,
        acquire_timeout: float = 10,
        connect_timeout: float = 5,
        startup_wait: float = 15,
    ):
        import psycopg
        from psycopg_pool import ConnectionPool

        self.integrity_errors = (psycopg.IntegrityError,)
        self._pool = ConnectionPool(
            url,
            min_size=min_size,
            max_size=max_size,
            timeout=acquire_timeout,
            kwargs={"connect_timeout": connect_timeout},
            check=ConnectionPool.check_connection,
            open=True,
        )
        self._pool.wait(timeout=startup_wait)  # fail at startup, not on the first request

    def connect(self) -> _PgConnection:
        return _PgConnection(self._pool, self._pool.getconn())

    def close(self) -> None:
        self._pool.close()

    def pk_column(self) -> str:
        return "BIGSERIAL PRIMARY KEY"

    def timestamp_column(self) -> str:
        return "TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP"

    def seconds_ago_sql(self) -> str:
        return "(CURRENT_TIMESTAMP - ? * INTERVAL '1 second')"

    def run_ddl(self, statements: Iterable[str]) -> None:
        # CREATE ... IF NOT EXISTS races when workers start together; serialize on a lock
        conn = self.connect()
        try:
            with conn:
                conn.execute("SELECT pg_advisory_xact_lock(?)", (_SCHEMA_LOCK_ID,))
                for statement in statements:
                    conn.execute(statement)
        finally:
            conn.close()


def schema_lock_id() -> int:
    return _SCHEMA_LOCK_ID


def open_database(
    database_url: str = "",
    sqlite_path: str = "./users.db",
    pool_size: int = 10,
    *,
    busy_timeout: float = BUSY_TIMEOUT_SECONDS,
    pool_min: int = 1,
    pool_timeout: float = 10,
    connect_timeout: float = 5,
    startup_wait: float = 15,
):
    """PostgreSQL when `database_url` is set (postgres:// or postgresql://), else sqlite."""
    if database_url:
        if not database_url.startswith(("postgres://", "postgresql://")):
            raise ValueError("DATABASE_URL must start with postgres:// or postgresql://")
        logger.info("Using PostgreSQL for users, feedback and messages")
        return PostgresDatabase(
            database_url,
            max_size=pool_size,
            min_size=pool_min,
            acquire_timeout=pool_timeout,
            connect_timeout=connect_timeout,
            startup_wait=startup_wait,
        )
    return SqliteDatabase(sqlite_path, busy_timeout)


def open_database_from_config(cfg) -> Database:
    """`open_database` with every value taken from the app's Config (duck-typed)."""
    return open_database(
        cfg.DATABASE_URL,
        cfg.USERS_DB_PATH,
        cfg.DATABASE_POOL_SIZE,
        busy_timeout=cfg.DB_BUSY_TIMEOUT,
        pool_min=cfg.DATABASE_POOL_MIN,
        pool_timeout=cfg.DATABASE_POOL_TIMEOUT,
        connect_timeout=cfg.DATABASE_CONNECT_TIMEOUT,
        startup_wait=cfg.DATABASE_STARTUP_WAIT,
    )


def as_database(db: "Database | str") -> Database:
    """Accept a Database or a sqlite file path (what the stores took before)."""
    return db if isinstance(db, Database) else SqliteDatabase(db)
