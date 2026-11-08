"""Values that used to be hardcoded in modules now come from Config and reach their users."""

import logging
from logging.handlers import RotatingFileHandler
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import embeddings
import pytest
from auth import create_auth
from config import Config, config, setting
from embeddings import create_embedder
from feedback import FeedbackRequest
from input_guard import InputGuard
from pg_vector_store import PgVectorStore, keyword_tokens, rrf_fuse
from rag_system import RAGSystem
from utils import db as db_module
from utils import redis_client
from utils.db import open_database_from_config
from utils.logging_config import setup_logging

# --- Config itself --------------------------------------------------------------------------


def test_setting_uses_the_object_value_else_the_config_default():
    assert setting(SimpleNamespace(HYBRID_RRF_K=7), "HYBRID_RRF_K") == 7
    assert setting(SimpleNamespace(), "HYBRID_RRF_K") == Config.HYBRID_RRF_K
    with pytest.raises(AttributeError):
        setting(SimpleNamespace(), "NOT_A_SETTING")


def test_moved_defaults_keep_their_previous_values():
    assert (Config.JWT_MIN_SECRET_LENGTH, Config.MAX_SESSION_ID_LENGTH) == (32, 128)
    assert (Config.MAX_MESSAGE_ID_LENGTH, Config.DAILY_QUOTA_RETRY_AFTER) == (64, 3600)
    assert (Config.DB_BUSY_TIMEOUT, Config.DATABASE_POOL_MIN) == (10, 1)
    assert (Config.DATABASE_POOL_TIMEOUT, Config.DATABASE_CONNECT_TIMEOUT) == (10, 5)
    assert Config.DATABASE_STARTUP_WAIT == 15
    assert (Config.REDIS_TIMEOUT, Config.REDIS_HEALTH_CHECK_INTERVAL) == (2, 30)
    assert (Config.LOG_MAX_BYTES, Config.LOG_BACKUP_COUNT) == (5 * 1024 * 1024, 3)
    assert (Config.GUARD_HISTORY_MESSAGES, Config.GUARD_HISTORY_CHARS) == (4, 300)
    assert (Config.HYBRID_RRF_K, Config.HYBRID_CANDIDATE_FACTOR) == (60, 4)
    assert (Config.HYBRID_MIN_CANDIDATES, Config.HYBRID_MAX_KEYWORDS) == (20, 20)
    assert Config.KEYWORD_STATS_CACHE_SECONDS == 300
    assert (Config.EMBEDDING_BATCH_SIZE, Config.EMBEDDING_TIMEOUT) == (64, 30)
    assert Config.EMBEDDING_MAX_ATTEMPTS == 3


# --- auth / request limits ------------------------------------------------------------------


def test_minimum_jwt_secret_length_comes_from_config(monkeypatch):
    store = MagicMock()
    with pytest.raises(RuntimeError, match="at least 32"):
        create_auth(store, "x" * 20, 60)
    monkeypatch.setattr(config, "JWT_MIN_SECRET_LENGTH", 16)
    create_auth(store, "x" * 20, 60)
    with pytest.raises(RuntimeError, match="at least 16"):
        create_auth(store, "x" * 10, 60)


def test_feedback_id_length_limits_follow_config():
    def max_length(field):
        metadata = FeedbackRequest.model_fields[field].metadata
        return next(m.max_length for m in metadata if hasattr(m, "max_length"))

    assert max_length("message_id") == config.MAX_MESSAGE_ID_LENGTH
    assert max_length("session_id") == config.MAX_SESSION_ID_LENGTH


# --- database -------------------------------------------------------------------------------


def db_config(**kw):
    base = dict(
        DATABASE_URL="",
        USERS_DB_PATH="unused.db",
        DATABASE_POOL_SIZE=7,
        DB_BUSY_TIMEOUT=3,
        DATABASE_POOL_MIN=2,
        DATABASE_POOL_TIMEOUT=4,
        DATABASE_CONNECT_TIMEOUT=6,
        DATABASE_STARTUP_WAIT=8,
    )
    return SimpleNamespace(**{**base, **kw})


def test_sqlite_busy_timeout_comes_from_config(tmp_path):
    database = open_database_from_config(db_config(USERS_DB_PATH=str(tmp_path / "t.db")))
    conn = database.connect()
    try:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 3000
    finally:
        conn.close()


def test_postgres_pool_settings_come_from_config():
    with patch.object(db_module, "PostgresDatabase") as pg:
        open_database_from_config(db_config(DATABASE_URL="postgresql://u:p@h/db"))
    pg.assert_called_once_with(
        "postgresql://u:p@h/db",
        max_size=7,
        min_size=2,
        acquire_timeout=4,
        connect_timeout=6,
        startup_wait=8,
    )


# --- redis ----------------------------------------------------------------------------------


def test_redis_timeouts_come_from_config():
    redis_client.get_redis.cache_clear()
    try:
        cfg = SimpleNamespace(
            REDIS_URL="redis://x/0", REDIS_TIMEOUT=9, REDIS_HEALTH_CHECK_INTERVAL=11
        )
        with patch.object(redis_client.redis.Redis, "from_url") as from_url:
            redis_client.redis_from_config(cfg)
        kwargs = from_url.call_args.kwargs
        assert kwargs["socket_timeout"] == 9 and kwargs["socket_connect_timeout"] == 9
        assert kwargs["health_check_interval"] == 11
    finally:
        redis_client.get_redis.cache_clear()


def test_redis_defaults_apply_when_the_config_lacks_the_tuning_values():
    redis_client.get_redis.cache_clear()
    try:
        with patch.object(redis_client.redis.Redis, "from_url") as from_url:
            redis_client.redis_from_config(SimpleNamespace(REDIS_URL="redis://y/0"))
        assert from_url.call_args.kwargs["socket_timeout"] == 2.0
    finally:
        redis_client.get_redis.cache_clear()


# --- logging --------------------------------------------------------------------------------


def test_log_rotation_limits_come_from_the_arguments(tmp_path):
    try:
        setup_logging("INFO", str(tmp_path / "app.log"), max_bytes=1234, backup_count=2)
        handler = next(
            h for h in logging.getLogger().handlers if isinstance(h, RotatingFileHandler)
        )
        assert (handler.maxBytes, handler.backupCount) == (1234, 2)
    finally:
        setup_logging("INFO")  # detach and close the file handler


# --- input guard ----------------------------------------------------------------------------


def test_guard_history_window_comes_from_the_constructor():
    guard = InputGuard("k", "http://g", "m", history_messages=2, history_chars=4)
    history = [{"role": "user", "content": f"msg{i}-long-text"} for i in range(5)]
    state = guard._state("now?", history)
    assert state.count("user: ") == 2  # only the last two messages
    assert "long-text" not in state and "msg3-" not in state  # each cut to 4 characters
    assert "user: msg3\n" in state and "user: msg4\n" in state
    assert state.endswith("now?")


# --- embeddings -----------------------------------------------------------------------------


def test_embedding_call_settings_reach_the_openai_provider():
    provider = create_embedder(
        SimpleNamespace(
            EMBEDDING_PROVIDER="openai",
            EMBEDDING_MODEL="m",
            EMBEDDING_BASE_URL="http://emb/v1",
            EMBEDDING_DIMENSIONS=8,
            EMBEDDING_BATCH_SIZE=5,
            EMBEDDING_TIMEOUT=7,
            EMBEDDING_MAX_ATTEMPTS=2,
        )
    )
    assert provider.batch_size == 5 and provider.max_attempts == 2  # type: ignore[attr-defined]
    assert provider._client.timeout.read == 7  # type: ignore[attr-defined]


def test_embedding_batch_size_reaches_the_local_provider():
    with patch.object(embeddings, "LocalEmbeddings") as local:
        create_embedder(
            SimpleNamespace(EMBEDDING_PROVIDER="local", EMBEDDING_MODEL="m", EMBEDDING_BATCH_SIZE=9)
        )
    local.assert_called_once_with("m", batch_size=9)


# --- hybrid search --------------------------------------------------------------------------


def test_rrf_constant_is_a_parameter_and_changes_the_ranking():
    rankings = [["x", "y"], ["z", "w", "y"]]
    assert rrf_fuse(rankings, k=0)[0] == "x"  # rank differences dominate
    assert rrf_fuse(rankings, k=60)[0] == "y"  # found by both rankings wins


def test_keyword_limit_is_a_parameter():
    assert keyword_tokens("aa bb cc dd", limit=2) == ["aa", "bb"]
    many = " ".join(f"w{i}x" for i in range(50))
    assert len(keyword_tokens(many)) == Config.HYBRID_MAX_KEYWORDS


def test_rag_system_passes_the_configured_retrieval_guard_and_llm_settings():
    cfg = SimpleNamespace(
        CHUNK_SIZE=800,
        CHUNK_OVERLAP=100,
        CHROMA_PATH="x",
        EMBEDDING_MODEL="x",
        MAX_RESULTS=5,
        MAX_HISTORY=2,
        LLM_API_KEY="k",
        LLM_BASE_URL="http://llm",
        LLM_MODEL="m",
        LLM_MAX_TOKENS=1,
        LLM_TEMPERATURE=0,
        LLM_MAX_RETRIES=1,
        VECTOR_BACKEND="pgvector",
        HYBRID_RRF_K=7,
        HYBRID_CANDIDATE_FACTOR=2,
        HYBRID_MIN_CANDIDATES=3,
        HYBRID_MAX_KEYWORDS=5,
        KEYWORD_STATS_CACHE_SECONDS=11,
        GUARD_API_KEY="g",
        GUARD_URL="http://g",
        GUARD_MODEL="gm",
        GUARD_THRESHOLD=0.5,
        GUARD_TIMEOUT=1,
        GUARD_HISTORY_MESSAGES=2,
        GUARD_HISTORY_CHARS=9,
        MAX_TOOL_ROUNDS=4,
        LLM_BREAKER_FAILURES=5,
    )
    database = MagicMock(dialect="postgres")
    with (
        patch("rag_system.AIGenerator") as generator,
        patch("rag_system.PgVectorStore") as store,
        patch("rag_system.create_embedder"),
    ):
        rag = RAGSystem(cfg, database=database)
    kwargs = store.call_args.kwargs
    assert (kwargs["rrf_k"], kwargs["candidate_factor"], kwargs["min_candidates"]) == (7, 2, 3)
    assert (kwargs["max_keywords"], kwargs["stats_cache_seconds"]) == (5, 11)
    assert rag.input_guard is not None
    assert (rag.input_guard.history_messages, rag.input_guard.history_chars) == (2, 9)
    gen_kwargs = generator.call_args.kwargs
    assert gen_kwargs["max_tool_rounds"] == 4 and gen_kwargs["breaker_failures"] == 5
    assert gen_kwargs["breaker_cooldown"] == Config.LLM_BREAKER_COOLDOWN  # default fallback


def test_pgvector_store_keeps_the_given_retrieval_settings():
    with (
        patch.object(PgVectorStore, "_create_schema"),
        patch.object(PgVectorStore, "_check_index_matches_embedder"),
    ):
        store = PgVectorStore(
            MagicMock(dialect="postgres"),
            SimpleNamespace(model="m", dimensions=8),
            rrf_k=7,
            candidate_factor=2,
            min_candidates=3,
            max_keywords=5,
            stats_cache_seconds=11,
        )
    assert (store.rrf_k, store.candidate_factor, store.min_candidates) == (7, 2, 3)
    assert (store.max_keywords, store.stats_cache_seconds) == (5, 11)
