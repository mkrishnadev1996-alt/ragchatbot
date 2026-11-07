"""HTTP-level tests for app.py with a mocked RAGSystem (no LLM, Chroma or network)."""

import importlib
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from ai_generator import LLMRateLimitError
from config import config
from fastapi.testclient import TestClient
from models import Source
from rag_system import QueryResult
from session_manager import SessionManager

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def env(tmp_path, monkeypatch):
    """The real app module, built against a temp database and a mocked RAGSystem"""
    monkeypatch.chdir(BACKEND_DIR)  # app.py mounts ../frontend relative to the cwd
    monkeypatch.setattr(config, "JWT_SECRET", "t" * 32)
    monkeypatch.setattr(config, "USERS_DB_PATH", str(tmp_path / "users.db"))
    monkeypatch.setattr(config, "QUERY_RATE_LIMIT", 100)
    monkeypatch.setattr(config, "AUTH_RATE_LIMIT", 100)

    rag = MagicMock()
    rag.session_manager = SessionManager(max_history=2)
    rag.query.return_value = QueryResult(
        "the answer", [Source(label="Course A - Lesson 1", url="https://a.test/1")]
    )
    rag.get_course_analytics.return_value = {"total_courses": 1, "course_titles": ["Course A"]}

    monkeypatch.delitem(sys.modules, "app", raising=False)
    with patch("rag_system.RAGSystem", return_value=rag):
        app_module = importlib.import_module("app")
    try:
        yield app_module, TestClient(app_module.app), rag
    finally:
        sys.modules.pop("app", None)


def register(client, email="a@example.com"):
    res = client.post("/api/auth/register", json={"email": email, "password": "password123"})
    return {"Authorization": f"Bearer {res.json()['token']}"}


def ask(client, headers, query="What is MCP?", **extra):
    return client.post("/api/query", json={"query": query, **extra}, headers=headers)


def test_query_requires_auth(env):
    _, client, _ = env
    assert client.post("/api/query", json={"query": "hi"}).status_code == 401


def test_query_success_returns_sources_with_links_and_persists(env):
    _, client, rag = env
    headers = register(client)
    res = ask(client, headers)
    assert res.status_code == 200
    body = res.json()
    assert body["answer"] == "the answer"
    assert body["sources"] == [{"label": "Course A - Lesson 1", "url": "https://a.test/1"}]
    assert body["session_id"] and body["message_id"]
    rag.query.assert_called_once()

    history = client.get("/api/messages", headers=headers).json()
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[1]["sources"][0]["url"] == "https://a.test/1"


def test_follow_up_reuses_the_session(env):
    _, client, rag = env
    headers = register(client)
    first = ask(client, headers).json()
    second = ask(client, headers, session_id=first["session_id"])
    assert second.json()["session_id"] == first["session_id"]


def test_foreign_or_unknown_session_is_404(env):
    _, client, _ = env
    owner = register(client, "owner@example.com")
    other = register(client, "other@example.com")
    sid = ask(client, owner).json()["session_id"]
    assert ask(client, other, session_id=sid).status_code == 404
    assert ask(client, owner, session_id="session_nope").status_code == 404


def test_blocked_query_is_not_persisted(env):
    _, client, rag = env
    rag.query.return_value = QueryResult("Sorry, I can't help with that.", [], blocked=True)
    headers = register(client)
    assert ask(client, headers).status_code == 200
    assert client.get("/api/messages", headers=headers).json() == []


def test_llm_rate_limit_maps_to_429_with_retry_after(env):
    _, client, rag = env
    rag.query.side_effect = LLMRateLimitError(retry_after=2.2)
    res = ask(client, register(client))
    assert res.status_code == 429
    assert res.headers["Retry-After"] == "3"


def test_unexpected_error_is_500_without_leaking_details(env):
    _, client, rag = env
    rag.query.side_effect = RuntimeError("secret chroma path /var/db exploded")
    res = ask(client, register(client))
    assert res.status_code == 500
    assert "secret" not in res.text and "exploded" not in res.text
    assert "reference:" in res.json()["detail"]


def test_persistence_failure_does_not_fail_the_answer(env):
    app_module, client, _ = env
    headers = register(client)
    with patch.object(app_module.message_store, "add_exchange", side_effect=OSError("disk")):
        res = ask(client, headers)
    assert res.status_code == 200
    assert res.json()["answer"] == "the answer"


def test_invalid_requests_are_422(env):
    _, client, _ = env
    headers = register(client)
    assert ask(client, headers, query="").status_code == 422
    assert ask(client, headers, query="x" * (config.MAX_QUERY_LENGTH + 1)).status_code == 422
    assert ask(client, headers, session_id="s" * 129).status_code == 422


def test_per_user_query_rate_limit_is_429(env, monkeypatch):
    app_module, client, _ = env
    monkeypatch.setattr(app_module.query_limiter, "max_attempts", 2)
    headers = register(client)
    assert ask(client, headers).status_code == 200
    assert ask(client, headers).status_code == 200
    third = ask(client, headers)
    assert third.status_code == 429
    assert "Retry-After" in third.headers


def test_messages_are_private_and_scoped_by_session(env):
    _, client, _ = env
    a = register(client, "a@example.com")
    b = register(client, "b@example.com")
    sid = ask(client, a).json()["session_id"]
    ask(client, a)  # a second session
    assert client.get("/api/messages", headers=b).json() == []
    scoped = client.get("/api/messages", params={"session_id": sid}, headers=a).json()
    assert len(scoped) == 2
    assert client.get("/api/messages").status_code == 401


def test_healthz_courses_and_security_headers(env):
    _, client, _ = env
    assert client.get("/healthz").json() == {"status": "ok"}
    res = client.get("/api/courses")
    assert res.json() == {"total_courses": 1, "course_titles": ["Course A"]}
    assert res.headers["X-Frame-Options"] == "DENY"
    assert "script-src 'self'" in res.headers["Content-Security-Policy"]


def test_oversized_body_is_413(env):
    _, client, _ = env
    res = client.post("/api/query", content=b"x" * (config.MAX_BODY_BYTES + 1))
    assert res.status_code == 413


def test_readyz_ok_when_dependencies_work(env):
    _, client, rag = env
    rag.vector_store.get_course_count.return_value = 1
    res = client.get("/readyz")
    assert res.status_code == 200
    assert res.json() == {
        "status": "ready",
        "checks": {"vector_store": "ok", "database": "ok"},
    }


def test_readyz_503_when_vector_store_fails(env):
    _, client, rag = env
    rag.vector_store.get_course_count.side_effect = RuntimeError("chroma down")
    res = client.get("/readyz")
    assert res.status_code == 503
    assert res.json()["checks"] == {"vector_store": "failed", "database": "ok"}


def test_startup_ingestion_can_be_disabled(env, monkeypatch):
    app_module, _, rag = env
    rag.input_guard = None
    monkeypatch.setattr(config, "DOCS_PATH", BACKEND_DIR)  # any existing folder
    monkeypatch.setattr(config, "INGEST_ON_STARTUP", False)
    with TestClient(app_module.app):
        pass
    rag.add_course_folder.assert_not_called()
    monkeypatch.setattr(config, "INGEST_ON_STARTUP", True)
    rag.add_course_folder.return_value = (0, 0)
    with TestClient(app_module.app):
        pass
    rag.add_course_folder.assert_called_once()


def test_daily_quota_blocks_with_429(env, monkeypatch):
    _, client, rag = env
    monkeypatch.setattr(config, "DAILY_QUERY_QUOTA", 2)
    headers = register(client)
    assert ask(client, headers).status_code == 200
    assert ask(client, headers).status_code == 200
    res = ask(client, headers)
    assert res.status_code == 429
    assert res.headers["Retry-After"] == "3600"
    assert rag.query.call_count == 2


def test_daily_quota_is_per_user(env, monkeypatch):
    _, client, _ = env
    monkeypatch.setattr(config, "DAILY_QUERY_QUOTA", 1)
    a = register(client, "a@example.com")
    b = register(client, "b@example.com")
    assert ask(client, a).status_code == 200
    assert ask(client, a).status_code == 429
    assert ask(client, b).status_code == 200


def test_api_docs_can_be_disabled(env, monkeypatch):
    app_module, client, _ = env
    assert client.get("/openapi.json").status_code == 200  # default: enabled
    monkeypatch.setattr(config, "ENABLE_API_DOCS", False)
    monkeypatch.setattr(config, "ENABLE_HSTS", True)
    sys.modules.pop("app", None)
    with patch("rag_system.RAGSystem", return_value=MagicMock()):
        off = importlib.import_module("app")
    off_client = TestClient(off.app)
    assert off_client.get("/openapi.json").status_code == 404
    assert off_client.get("/docs").status_code == 404
    assert "max-age=" in off_client.get("/healthz").headers["Strict-Transport-Security"]
