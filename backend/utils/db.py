"""SQLite connection helper shared by the user, feedback and message stores."""

import sqlite3

BUSY_TIMEOUT_SECONDS = 10


def connect(db_path: str) -> sqlite3.Connection:
    """Open a connection (one per call keeps this safe across worker threads).

    WAL lets readers and the writer work at the same time, and the busy timeout makes a
    writer wait for a lock instead of failing with "database is locked".
    """
    conn = sqlite3.connect(db_path, timeout=BUSY_TIMEOUT_SECONDS)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn
