"""Phase 1: stateless backends (PostgreSQL stores, Redis sessions/limits) and the ingest CLI.

The store tests run on sqlite always, and on PostgreSQL too when TEST_DATABASE_URL is set
(e.g. postgresql://postgres:pw@localhost:55432/rag). Redis behaviour is tested with fakeredis.
"""

import os
from contextlib import closing
from unittest.mock import MagicMock, patch

import fakeredis
import pytest
import redis
from auth import RedisRateLimiter
from fastapi import HTTPException
from feedback_store import FeedbackStore
from message_store import MessageStore
from models import Source
from session_manager import RedisSessionManager
from user_store import UserExistsError, UserStore
from utils.db import PostgresDatabase, SqliteDatabase, open_database
from utils.migrations import MIGRATIONS, migrate

PG_URL = os.getenv("TEST_DATABASE_URL", "")


@pytest.fixture(params=["sqlite", "postgres"])
def db(request, tmp_path):
    if request.param == "sqlite":
        yield SqliteDatabase(str(tmp_path / "t.db"))
        return
    if not PG_URL:
        pytest.skip("TEST_DATABASE_URL not set")
    database = PostgresDatabase(PG_URL, max_size=3)
    with closing(database.connect()) as conn, conn:
        conn.execute("DROP TABLE IF EXISTS messages, feedback, users, schema_version CASCADE")
    try:
        yield database
    finally:
        database.close()


# --- stores on both backends ---------------------------------------------------------------


def test_user_store_roundtrip_and_case_insensitive_email(db):
    store = UserStore(db)
    user = store.create_user("a@example.com", "h")
    assert store.get_by_id(user.id) == user
    assert store.get_by_email("A@EXAMPLE.COM") == user
    with pytest.raises(UserExistsError):
        store.create_user("A@Example.com", "h2")
    assert store.get_by_email("nobody@example.com") is None


def test_creating_stores_twice_is_harmless(db):
    UserStore(db)
    UserStore(db)
    MessageStore(db)
    MessageStore(db)


def _exchange(store, user_id, n, session="s"):
    for i in range(n):
        store.add_exchange(
            user_id, session, f"q{i}", f"a{i}", [Source(label="L", url="u")], f"m{i}"
        )


def test_messages_feedback_quota_and_purge(db):
    messages = MessageStore(db)
    feedback = FeedbackStore(db)
    _exchange(messages, 1, 2)
    _exchange(messages, 2, 1, session="other")

    feedback.set_rating("m0", 1, "up", "s")
    feedback.set_rating("m0", 1, "down", "s")  # replaces
    assert feedback.get_rating("m0", 1) == "down"
    listed = messages.list_messages(1)
    assert [m["content"] for m in listed] == ["q0", "a0", "q1", "a1"]
    assert listed[1]["rating"] == "down" and listed[1]["sources"] == [{"label": "L", "url": "u"}]
    feedback.clear_rating("m0", 1)
    assert feedback.get_rating("m0", 1) is None

    assert [m["content"] for m in messages.list_messages(1, limit=2)] == ["q1", "a1"]
    assert len(messages.list_messages(2, session_id="other")) == 2
    assert messages.list_messages(2, session_id="nope") == []

    assert messages.count_questions_since(1) == 2
    assert messages.count_questions_since(2) == 1
    with closing(db.connect()) as conn, conn:
        conn.execute(
            f"UPDATE messages SET created_at = {db.seconds_ago_sql()} WHERE user_id = ?",
            (10 * 86400.0, 1),
        )
    assert messages.count_questions_since(1) == 0
    assert messages.purge_older_than(5) == 4
    assert messages.count_questions_since(2) == 1


def test_migrate_is_idempotent_and_concurrency_safe_to_repeat(db):
    MessageStore(db)
    assert migrate(db) == MIGRATIONS[-1][0]
    assert migrate(db) == MIGRATIONS[-1][0]


def test_open_database_picks_backend(tmp_path):
    assert isinstance(open_database("", str(tmp_path / "x.db")), SqliteDatabase)
    with pytest.raises(ValueError):
        open_database("mysql://nope", "x")


# --- Redis sessions -------------------------------------------------------------------------


@pytest.fixture
def rds():
    return fakeredis.FakeRedis(decode_responses=True)


def _sessions(rds, **kw):
    return RedisSessionManager(rds, **{"max_history": 2, **kw})


def test_redis_session_is_shared_between_manager_instances(rds):
    a, b = _sessions(rds), _sessions(rds)  # two "workers"
    sid = a.create_session(user_id=7)
    assert b.is_owner(sid, 7)
    a.add_exchange(sid, "q1", "a1")
    assert b.get_conversation_history(sid) == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]


def test_redis_session_ownership_and_unknown(rds):
    sm = _sessions(rds)
    sid = sm.create_session(user_id=7)
    assert not sm.is_owner(sid, 8)
    assert not sm.is_owner("session_missing", 7)
    assert sm.get_conversation_history("session_missing") is None
    assert sm.get_conversation_history(None) is None
    assert sm.get_conversation_history(sid) is None  # exists but empty


def test_redis_session_keeps_last_max_history_exchanges(rds):
    sm = _sessions(rds, max_history=2)
    sid = sm.create_session(1)
    for i in range(5):
        sm.add_exchange(sid, f"q{i}", f"a{i}")
    assert [m["content"] for m in sm.get_conversation_history(sid)] == ["q3", "a3", "q4", "a4"]


def test_redis_session_applies_char_budget(rds):
    sm = _sessions(rds, max_history_chars=25)
    sid = sm.create_session(1)
    sm.add_exchange(sid, "q" * 10, "a" * 10)
    sm.add_exchange(sid, "r" * 10, "b" * 10)
    assert [m["content"] for m in sm.get_conversation_history(sid)] == ["r" * 10, "b" * 10]


def test_redis_session_expires_after_ttl(rds):
    sm = _sessions(rds, ttl_seconds=100)
    sid = sm.create_session(1)
    sm.add_exchange(sid, "q", "a")
    assert 0 < rds.ttl(f"rag:sess:{sid}:owner") <= 100
    assert 0 < rds.ttl(f"rag:sess:{sid}:msgs") <= 100
    rds.delete(f"rag:sess:{sid}:owner")  # as if it expired
    assert not sm.is_owner(sid, 1)
    assert sm.get_conversation_history(sid) is None


# --- Redis rate limiter ---------------------------------------------------------------------


def test_redis_limiter_blocks_after_limit_with_retry_after(rds):
    limiter = RedisRateLimiter(rds, "query", max_attempts=3, window_seconds=60, detail="slow down")
    for _ in range(3):
        limiter.check("user:1")
    with pytest.raises(HTTPException) as exc:
        limiter.check("user:1")
    assert exc.value.status_code == 429 and exc.value.detail == "slow down"
    assert 1 <= int((exc.value.headers or {})["Retry-After"]) <= 60
    limiter.check("user:2")  # other keys unaffected


def test_redis_limiter_limit_is_shared_between_instances(rds):
    a = RedisRateLimiter(rds, "query", max_attempts=2, window_seconds=60)
    b = RedisRateLimiter(rds, "query", max_attempts=2, window_seconds=60)
    a.check("k")
    b.check("k")
    with pytest.raises(HTTPException):
        a.check("k")


def test_redis_limiters_with_different_names_do_not_share_counts(rds):
    auth = RedisRateLimiter(rds, "auth", max_attempts=1, window_seconds=60)
    query = RedisRateLimiter(rds, "query", max_attempts=1, window_seconds=60)
    auth.check("k")
    query.check("k")


def test_redis_limiter_fails_open_when_redis_is_down():
    broken = MagicMock()
    broken.pipeline.side_effect = redis.ConnectionError("down")
    RedisRateLimiter(broken, "query", max_attempts=1, window_seconds=60).check("k")


# --- ingest CLI -----------------------------------------------------------------------------


def test_ingest_cli_runs_add_course_folder(tmp_path):
    import ingest

    rag = MagicMock()
    rag.add_course_folder.return_value = (2, 10)
    with patch.object(ingest, "RAGSystem", return_value=rag), patch.object(ingest, "setup_logging"):
        assert ingest.main(["--clear", "--path", str(tmp_path)]) == 0
    rag.add_course_folder.assert_called_once_with(str(tmp_path), clear_existing=True)
