"""Phase 0 production-readiness features: migrations, quota, retention, history budget."""

import sqlite3
from contextlib import closing

from message_store import MessageStore
from models import Source
from session_manager import SessionManager
from utils.migrations import MIGRATIONS, migrate


def _version(path):
    with closing(sqlite3.connect(path)) as conn:
        return conn.execute("PRAGMA user_version").fetchone()[0]


def test_migrate_applies_pending_once_and_is_idempotent(tmp_path):
    db = str(tmp_path / "db.sqlite")
    MessageStore(db)  # creates the baseline tables
    assert migrate(db) == MIGRATIONS[-1][0]
    assert migrate(db) == MIGRATIONS[-1][0]
    with closing(sqlite3.connect(db)) as conn:
        names = {r[1] for r in conn.execute("PRAGMA index_list(messages)")}
    assert "idx_messages_user_role_created" in names
    assert _version(db) == MIGRATIONS[-1][0]


def test_failed_migration_rolls_back_and_keeps_version(tmp_path):
    db = str(tmp_path / "db.sqlite")
    MessageStore(db)
    bad = [(1, ["CREATE TABLE t (x)", "THIS IS NOT SQL"])]
    try:
        migrate(db, bad)
    except sqlite3.Error:
        pass
    else:
        raise AssertionError("expected the bad migration to raise")
    assert _version(db) == 0
    with closing(sqlite3.connect(db)) as conn:
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='t'").fetchone() is None


def _add(store, user_id, n=1):
    for i in range(n):
        store.add_exchange(user_id, "s", f"q{i}", "a", [Source(label="L", url=None)], f"m{i}")


def test_count_questions_since_counts_only_user_rows_in_window(tmp_path):
    store = MessageStore(str(tmp_path / "db.sqlite"))
    _add(store, 1, 3)
    _add(store, 2, 1)
    assert store.count_questions_since(1) == 3
    assert store.count_questions_since(2) == 1
    with closing(store._connect()) as conn, conn:
        conn.execute(
            "UPDATE messages SET created_at = datetime('now', '-2 days') WHERE user_id = 1"
        )
    assert store.count_questions_since(1, 24) == 0


def test_purge_older_than_removes_only_old_rows(tmp_path):
    store = MessageStore(str(tmp_path / "db.sqlite"))
    _add(store, 1, 2)
    with closing(store._connect()) as conn, conn:
        conn.execute("UPDATE messages SET created_at = datetime('now', '-10 days') WHERE id <= 4")
    _add(store, 1, 1)
    assert store.purge_older_than(5) == 4
    assert len(store.list_messages(1)) == 2


def _session(max_chars, pairs):
    sm = SessionManager(max_history=10, max_history_chars=max_chars)
    sid = sm.create_session(1)
    for q, a in pairs:
        sm.add_exchange(sid, q, a)
    return sm, sid


def test_history_budget_drops_oldest_and_starts_with_a_question():
    sm, sid = _session(25, [("q1" * 5, "a1" * 5), ("q2" * 5, "a2" * 5)])
    history = sm.get_conversation_history(sid)
    assert [m["content"] for m in history] == ["q2" * 5, "a2" * 5]


def test_history_budget_drops_orphaned_leading_answer():
    sm, sid = _session(15, [("q" * 5, "a" * 10)])
    # 15 chars fit both; shrink so only the answer would fit -> answer alone is dropped
    sm.max_history_chars = 12
    assert sm.get_conversation_history(sid) is None


def test_history_budget_truncates_a_single_oversized_message():
    sm = SessionManager(max_history=10, max_history_chars=5)
    sid = sm.create_session(1)
    sm.add_message(sid, "user", "0123456789")
    assert sm.get_conversation_history(sid) == [{"role": "user", "content": "56789"}]


def test_history_unlimited_when_budget_is_zero():
    sm, sid = _session(0, [("q" * 5000, "a" * 5000)])
    assert len(sm.get_conversation_history(sid)) == 2
