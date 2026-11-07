"""Async streaming: AIGenerator.astream_response, RAGSystem.aquery_stream, /api/query/stream."""

import json
import unittest
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

from ai_generator import AIGenerator, LLMRateLimitError
from langchain_core.messages import AIMessageChunk
from models import Source
from rag_system import QueryResult, RAGSystem
from search_tools import ToolResult
from test_ai_generator import make_rate_limit_error
from test_app_endpoints import env, register  # noqa: F401  (env is a fixture)
from test_rag_system import make_config


def chunk(text="", **kwargs):
    return AIMessageChunk(content=text, **kwargs)


async def agen(items) -> AsyncIterator:
    for item in items:
        yield item


async def collect(aiter) -> list:
    return [item async for item in aiter]


def events(res):
    return [json.loads(line[6:]) for line in res.text.split("\n\n") if line.startswith("data: ")]


class TestAstreamResponse(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patcher = patch("ai_generator.ChatOpenAI")
        self.addCleanup(patcher.stop)
        self.llm = patcher.start().return_value
        self.bound = self.llm.bind_tools.return_value
        self.generator = AIGenerator(api_key="k", base_url="http://llm.test", model="m")

    async def test_yields_text_deltas_without_tools(self):
        self.llm.astream.return_value = agen([chunk("Hel"), chunk("lo")])
        self.assertEqual(await collect(self.generator.astream_response("hi")), ["Hel", "lo"])

    async def test_tool_round_then_streamed_answer(self):
        tool_chunk = chunk(
            tool_call_chunks=[{"name": "search", "args": '{"query": "x"}', "id": "c1", "index": 0}]
        )
        self.bound.astream.return_value = agen([tool_chunk])
        self.llm.astream.return_value = agen([chunk("An"), chunk("swer")])
        manager = MagicMock()
        manager.execute_tool.return_value = ToolResult("text", [Source(label="L", url=None)])
        sources: list[Source] = []

        out = await collect(
            self.generator.astream_response(
                "q", tools=[{"type": "function"}], tool_manager=manager, sources_out=sources
            )
        )

        self.assertEqual(out, ["An", "swer"])
        manager.execute_tool.assert_called_once_with("search", query="x")
        self.assertEqual([s.label for s in sources], ["L"])

    async def test_truncation_notice_is_last(self):
        self.llm.astream.return_value = agen(
            [chunk("abc"), chunk(response_metadata={"finish_reason": "length"})]
        )
        out = await collect(self.generator.astream_response("hi"))
        self.assertEqual(out, ["abc", AIGenerator.TRUNCATION_NOTICE])

    async def test_rate_limit_is_translated_on_first_item(self):
        async def boom(_messages):
            raise make_rate_limit_error("3")
            yield  # pragma: no cover

        self.llm.astream.side_effect = boom
        with self.assertRaises(LLMRateLimitError) as ctx:
            await anext(self.generator.astream_response("hi"))
        self.assertEqual(ctx.exception.retry_after, 3)


class TestAqueryStream(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        for target in ("rag_system.VectorStore", "rag_system.AIGenerator"):
            p = patch(target)
            self.addCleanup(p.stop)
            p.start()
        self.rag = RAGSystem(make_config())
        self.ai: Any = self.rag.ai_generator

    async def test_saves_history_only_after_complete_and_ends_with_result(self):
        sid = self.rag.session_manager.create_session(1)
        self.ai.astream_response.side_effect = lambda **kw: agen(["a", "b"])
        stream = self.rag.aquery_stream("q", sid)
        self.assertEqual(await anext(stream), "a")
        self.assertIsNone(self.rag.session_manager.get_conversation_history(sid))
        rest = await collect(stream)
        self.assertEqual(rest[0], "b")
        self.assertEqual(rest[-1], QueryResult("ab", []))
        self.assertEqual(len(self.rag.session_manager.get_conversation_history(sid) or []), 2)

    async def test_blocked_skips_llm_and_history(self):
        self.rag.input_guard = cast(Any, SimpleNamespace(ais_safe=AsyncMock(return_value=False)))
        sid = self.rag.session_manager.create_session(1)
        items = await collect(self.rag.aquery_stream("bad", sid))
        self.assertTrue(cast(QueryResult, items[-1]).blocked)
        self.ai.astream_response.assert_not_called()
        self.assertIsNone(self.rag.session_manager.get_conversation_history(sid))


def stream(client, headers, query="What is MCP?", **extra):
    return client.post("/api/query/stream", json={"query": query, **extra}, headers=headers)


def test_stream_requires_auth(env):  # noqa: F811
    _, client, _ = env
    assert stream(client, {}).status_code == 401


def test_stream_emits_deltas_then_done_and_persists(env):  # noqa: F811
    _, client, rag = env
    final = QueryResult("Hello", [Source(label="A", url="https://a.test")])
    rag.aquery_stream.side_effect = lambda *a, **k: agen(["Hel", "lo", final])
    headers = register(client)
    res = stream(client, headers)
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    assert res.headers["x-content-type-options"] == "nosniff"  # security headers still applied
    evs = events(res)
    assert [e["type"] for e in evs] == ["delta", "delta", "done"]
    assert evs[-1]["answer"] == "Hello"
    assert evs[-1]["sources"] == [{"label": "A", "url": "https://a.test"}]
    assert evs[-1]["session_id"] and evs[-1]["message_id"]
    history = client.get("/api/messages", headers=headers).json()
    assert [m["role"] for m in history] == ["user", "assistant"]


def test_stream_blocked_is_not_persisted(env):  # noqa: F811
    _, client, rag = env
    rag.aquery_stream.side_effect = lambda *a, **k: agen(
        ["no", QueryResult("no", [], blocked=True)]
    )
    headers = register(client)
    assert events(stream(client, headers))[-1]["type"] == "done"
    assert client.get("/api/messages", headers=headers).json() == []


def test_stream_foreign_session_404(env):  # noqa: F811
    _, client, rag = env
    other = rag.session_manager.create_session(user_id=999)
    assert stream(client, register(client), session_id=other).status_code == 404


def test_stream_early_rate_limit_is_http_429(env):  # noqa: F811
    _, client, rag = env

    async def boom(*args, **kwargs):
        raise LLMRateLimitError(7)
        yield  # pragma: no cover

    rag.aquery_stream.side_effect = boom
    res = stream(client, register(client))
    assert res.status_code == 429
    assert res.headers["retry-after"] == "7"


def test_stream_midway_failure_becomes_error_event_without_details(env):  # noqa: F811
    _, client, rag = env

    async def broken(*args, **kwargs):
        yield "partial"
        raise RuntimeError("secret internals")

    rag.aquery_stream.side_effect = broken
    res = stream(client, register(client))
    evs = events(res)
    assert [e["type"] for e in evs] == ["delta", "error"]
    assert "secret internals" not in res.text
    assert evs[-1]["status"] == 500
