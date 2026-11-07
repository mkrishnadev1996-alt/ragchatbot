import pytest


@pytest.fixture(autouse=True)
def _disable_langsmith_tracing(monkeypatch):
    """Keep unit tests offline: never send traces to LangSmith, even if .env enables tracing"""
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
