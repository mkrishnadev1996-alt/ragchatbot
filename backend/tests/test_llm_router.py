"""LLM fallback chain: circuit breaker, router failover, config parsing, generator wiring."""

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest
from ai_generator import AIGenerator, LLMRateLimitError
from config import KNOWN_LLM_PROVIDERS, llm_provider_settings
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from llm_router import (
    CircuitBreaker,
    Endpoint,
    LLMRouter,
    ProviderSpec,
    classify_failure,
    flatten_tool_turns,
)
from rag_system import RAGSystem

MESSAGES: list[BaseMessage] = [HumanMessage(content="hi")]


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def status_error(cls, status, headers=None):
    response = SimpleNamespace(status_code=status, headers=headers or {}, request=None)
    return cls("boom", response=cast(Any, response), body=None)


def rate_limit(retry_after=None):
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    return status_error(openai.RateLimitError, 429, headers)


def server_error():
    return status_error(openai.InternalServerError, 503)


def bad_request():
    return status_error(openai.BadRequestError, 400)


def make_endpoint(name, clock, outcomes, failures=3, cooldown=60.0, max_cooldown=900.0):
    """An endpoint whose model replays `outcomes` (an exception is raised, else returned)."""
    llm = MagicMock(name=name)
    llm.invoke.side_effect = outcomes
    llm.bind_tools.return_value = llm
    return Endpoint(
        name, f"{name}-model", llm, CircuitBreaker(failures, cooldown, max_cooldown, clock)
    )


def ai(text):
    return AIMessage(content=text)


# --- circuit breaker ------------------------------------------------------------------------


def test_breaker_opens_after_consecutive_failures_and_closes_on_success():
    clock = Clock()
    b = CircuitBreaker(failure_threshold=3, cooldown=60, clock=clock)
    b.record_failure()
    b.record_failure()
    assert b.allow()
    b.record_failure()
    assert not b.allow() and b.reopens_in() == 60
    clock.now += 61
    assert b.allow()  # half-open: the next call is a trial
    b.record_success()
    b.record_failure()
    assert b.allow()  # the failure count started over


def test_breaker_reopens_at_once_when_the_trial_fails():
    clock = Clock()
    b = CircuitBreaker(failure_threshold=2, cooldown=30, clock=clock)
    b.record_failure()
    b.record_failure()
    clock.now += 31
    assert b.allow()
    b.record_failure()
    assert not b.allow()


def test_rate_limit_opens_immediately_for_retry_after_within_bounds():
    clock = Clock()
    b = CircuitBreaker(failure_threshold=5, cooldown=60, max_cooldown=900, clock=clock)
    b.record_failure(rate_limited=True, retry_after=300)
    assert not b.allow() and b.reopens_in() == 300
    b.record_failure(rate_limited=True, retry_after=5000)  # capped
    assert b.reopens_in() == 900
    b.record_failure(rate_limited=True, retry_after=1)  # at least the cooldown
    assert b.reopens_in() == 60
    b.record_failure(rate_limited=True)  # no header: the cooldown
    assert b.reopens_in() == 60


def test_classify_failure():
    assert classify_failure(rate_limit(7)) == (True, True, 7.0)
    assert classify_failure(server_error()) == (True, False, None)
    assert classify_failure(bad_request()) == (False, False, None)  # fail over, never trips
    assert classify_failure(status_error(openai.NotFoundError, 404)) == (True, False, None)
    assert classify_failure(openai.APIConnectionError(request=cast(Any, None))) == (
        True,
        False,
        None,
    )
    assert classify_failure(httpx.ReadTimeout("slow")) == (True, False, None)
    assert classify_failure(RuntimeError("bug")) is None


# --- router: sync ---------------------------------------------------------------------------


def test_first_endpoint_answers_when_healthy():
    clock = Clock()
    a = make_endpoint("a", clock, [ai("from a")])
    b = make_endpoint("b", clock, [ai("from b")])
    assert LLMRouter([a, b]).invoke(MESSAGES).content == "from a"
    b.llm.invoke.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [rate_limit(), server_error(), bad_request(), status_error(openai.AuthenticationError, 401)],
)
def test_failure_falls_through_to_the_next_endpoint(error):
    clock = Clock()
    a = make_endpoint("a", clock, [error])
    b = make_endpoint("b", clock, [ai("from b")])
    assert LLMRouter([a, b]).invoke(MESSAGES).content == "from b"


def test_rate_limited_endpoint_is_skipped_until_its_retry_after_passes():
    clock = Clock()
    a = make_endpoint("a", clock, [rate_limit(120), ai("a is back")])
    b = make_endpoint("b", clock, [ai("b1"), ai("b2"), ai("b3")])
    router = LLMRouter([a, b])

    assert router.invoke(MESSAGES).content == "b1"  # a rate limited -> b
    assert router.invoke(MESSAGES).content == "b2"  # a is cooling down: not even tried
    assert a.llm.invoke.call_count == 1
    clock.now += 121
    assert router.invoke(MESSAGES).content == "a is back"  # trial succeeds, a is first again


def test_server_errors_trip_the_breaker_only_after_the_threshold():
    clock = Clock()
    a = make_endpoint("a", clock, [server_error()] * 3 + [ai("never")], failures=3)
    b = make_endpoint("b", clock, [ai("b")] * 5)
    router = LLMRouter([a, b])
    for _ in range(3):
        router.invoke(MESSAGES)
    assert a.llm.invoke.call_count == 3
    router.invoke(MESSAGES)  # breaker is open now
    assert a.llm.invoke.call_count == 3


def test_bad_request_falls_over_but_never_opens_the_breaker():
    clock = Clock()
    a = make_endpoint("a", clock, [bad_request()] * 10, failures=1)
    b = make_endpoint("b", clock, [ai("b")] * 10)
    router = LLMRouter([a, b])
    for _ in range(4):
        assert router.invoke(MESSAGES).content == "b"
    assert a.llm.invoke.call_count == 4  # a is still tried every time


def test_when_everything_fails_the_last_error_is_raised():
    clock = Clock()
    a = make_endpoint("a", clock, [rate_limit(10)])
    b = make_endpoint("b", clock, [server_error()])
    with pytest.raises(openai.InternalServerError):
        LLMRouter([a, b]).invoke(MESSAGES)


def test_when_every_endpoint_is_cooling_down_the_soonest_is_still_tried():
    clock = Clock()
    a = make_endpoint("a", clock, [rate_limit(500), ai("a again")])
    b = make_endpoint("b", clock, [rate_limit(100), ai("never")])
    router = LLMRouter([a, b])
    with pytest.raises(openai.RateLimitError):
        router.invoke(MESSAGES)  # both rate limited now: a for 500 s, b for 100 s
    b.llm.invoke.side_effect = [ai("b recovered")]
    assert router.invoke(MESSAGES).content == "b recovered"  # b reopens soonest -> tried


def test_errors_that_are_not_provider_errors_are_not_masked():
    clock = Clock()
    a = make_endpoint("a", clock, [RuntimeError("bug")])
    b = make_endpoint("b", clock, [ai("b")])
    with pytest.raises(RuntimeError):
        LLMRouter([a, b]).invoke(MESSAGES)
    b.llm.invoke.assert_not_called()


def test_bind_tools_binds_every_endpoint_and_uses_the_bound_models():
    clock = Clock()
    a = make_endpoint("a", clock, [rate_limit()])
    b = make_endpoint("b", clock, [ai("b")])
    a_bound, b_bound = MagicMock(), MagicMock()
    a_bound.invoke.side_effect = rate_limit()
    b_bound.invoke.return_value = ai("bound b")
    a.llm.bind_tools.return_value = a_bound
    b.llm.bind_tools.return_value = b_bound

    routed = LLMRouter([a, b]).bind_tools([{"type": "function"}], tool_choice="auto")

    assert routed.invoke(MESSAGES).content == "bound b"
    a.llm.bind_tools.assert_called_once_with([{"type": "function"}], tool_choice="auto")
    a.llm.invoke.assert_not_called()  # the unbound model is not used for a tool call


def test_router_needs_an_endpoint():
    with pytest.raises(ValueError):
        LLMRouter([])


# --- router: streaming ----------------------------------------------------------------------


async def chunks(*texts, then_raise=None):
    for text in texts:
        yield AIMessageChunk(content=text)
    if then_raise:
        raise then_raise


def stream_endpoint(name, clock, factory, **kw):
    llm = MagicMock(name=name)
    llm.astream.side_effect = factory
    return Endpoint(name, f"{name}-model", llm, CircuitBreaker(clock=clock, **kw))


async def collect(aiter):
    return [c.content async for c in aiter]


def test_stream_fails_over_before_the_first_token():
    async def scenario():
        clock = Clock()
        a = stream_endpoint("a", clock, lambda m: chunks(then_raise=rate_limit(90)))
        b = stream_endpoint("b", clock, lambda m: chunks("he", "llo"))
        router = LLMRouter([a, b])
        assert await collect(router.astream(MESSAGES)) == ["he", "llo"]
        assert not a.breaker.allow()  # rate limited: skipped from now on
        assert await collect(router.astream(MESSAGES)) == ["he", "llo"]
        assert a.llm.astream.call_count == 1

    asyncio.run(scenario())


def test_stream_error_after_the_first_token_is_raised_not_retried_elsewhere():
    async def scenario():
        clock = Clock()
        a = stream_endpoint("a", clock, lambda m: chunks("partial", then_raise=server_error()))
        b = stream_endpoint("b", clock, lambda m: chunks("other"))
        seen = []
        with pytest.raises(openai.InternalServerError):
            async for chunk in LLMRouter([a, b]).astream(MESSAGES):
                seen.append(chunk.content)
        assert seen == ["partial"]
        b.llm.astream.assert_not_called()

    asyncio.run(scenario())


def test_stream_all_failing_raises_the_last_error():
    async def scenario():
        clock = Clock()
        a = stream_endpoint("a", clock, lambda m: chunks(then_raise=server_error()))
        b = stream_endpoint("b", clock, lambda m: chunks(then_raise=rate_limit()))
        with pytest.raises(openai.RateLimitError):
            await collect(LLMRouter([a, b]).astream(MESSAGES))

    asyncio.run(scenario())


# --- provider settings from the environment -------------------------------------------------


def test_known_providers_use_default_urls_and_models_when_the_key_is_set():
    env = {"GROQ_API_KEY": "g", "NVIDIA_API_KEY": "n"}
    specs = llm_provider_settings("groq, nvidia ,google", env)
    assert [s["name"] for s in specs] == ["groq", "nvidia"]  # google has no key: skipped
    assert specs[0]["base_url"] == KNOWN_LLM_PROVIDERS["groq"]["base_url"]
    assert specs[0]["models"][0] == "openai/gpt-oss-120b" and len(specs[0]["models"]) > 1
    assert specs[1]["api_key"] == "n"


def test_provider_models_and_url_can_be_overridden():
    env = {"GROQ_API_KEY": "g", "GROQ_MODELS": " a , b ", "GROQ_BASE_URL": "http://x/v1"}
    (spec,) = llm_provider_settings("groq", env)
    assert spec["models"] == ["a", "b"] and spec["base_url"] == "http://x/v1"


def test_llm_prefixed_names_are_accepted_and_the_preferred_model_goes_first():
    env = {
        "NVIDIA_LLM_API_KEY": "n",
        "NVIDIA_LLM_BASE_URL": "http://nv/v1",
        "NVIDIA_LLM_MODEL": "nvidia/nemotron-3.5-lightning-30b-a3b",  # also in the defaults
    }
    (spec,) = llm_provider_settings("nvidia", env)
    assert spec["api_key"] == "n" and spec["base_url"] == "http://nv/v1"
    assert spec["models"][0] == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert len(spec["models"]) == len(set(spec["models"]))  # no duplicate of the default
    assert "nvidia/nemotron-3-ultra-550b-a55b" in spec["models"]  # other free models follow


def test_plain_names_win_over_llm_prefixed_ones():
    env = {"GROQ_API_KEY": "plain", "GROQ_LLM_API_KEY": "other", "GROQ_MODEL": "m1"}
    (spec,) = llm_provider_settings("groq", env)
    assert spec["api_key"] == "plain" and spec["models"][0] == "m1"


def test_unknown_provider_needs_url_and_models():
    assert llm_provider_settings("acme", {"ACME_API_KEY": "k"}) == []
    env = {"ACME_API_KEY": "k", "ACME_BASE_URL": "http://acme/v1", "ACME_MODELS": "m1"}
    assert llm_provider_settings("acme", env)[0]["models"] == ["m1"]


def test_grok_is_accepted_as_a_spelling_of_groq():
    env = {"GROK_API_KEY": "g", "GROK_MODEL": "openai/gpt-oss-120b"}
    (spec,) = llm_provider_settings("grok", env)
    assert spec["name"] == "grok"
    assert spec["base_url"] == KNOWN_LLM_PROVIDERS["groq"]["base_url"]
    assert spec["models"][0] == "openai/gpt-oss-120b" and len(spec["models"]) == 3


def test_empty_names_give_no_providers():
    assert llm_provider_settings("", {"GROQ_API_KEY": "g"}) == []


# --- AIGenerator and RAGSystem wiring -------------------------------------------------------


def specs():
    return [
        ProviderSpec("groq", "http://groq/v1", "gk", ["g1", "g2"]),
        ProviderSpec("nvidia", "http://nv/v1", "nk", ["n1"]),
    ]


def test_generator_builds_one_client_per_model_in_order_without_retries_before_the_last():
    with patch("ai_generator.ChatOpenAI") as chat:
        gen = AIGenerator("k", "http://x", "m", max_retries=3, providers=specs())
    calls = [c.kwargs for c in chat.call_args_list]
    assert [(c["base_url"], c["model"], c["api_key"]) for c in calls] == [
        ("http://groq/v1", "g1", "gk"),
        ("http://groq/v1", "g2", "gk"),
        ("http://nv/v1", "n1", "nk"),
    ]
    assert [c["max_retries"] for c in calls] == [0, 0, 3]  # only the last gets the full count
    assert gen.model == "g1"


def test_generator_answers_from_the_next_model_when_the_first_is_rate_limited():
    first, second, third = MagicMock(), MagicMock(), MagicMock()
    first.invoke.side_effect = rate_limit(60)
    second.invoke.return_value = AIMessage(content="from g2")
    with patch("ai_generator.ChatOpenAI", side_effect=[first, second, third]):
        gen = AIGenerator("k", "http://x", "m", providers=specs())
        assert gen.generate_response("q") == "from g2"
    third.invoke.assert_not_called()


def test_generator_translates_a_fully_rate_limited_chain():
    llms = [MagicMock(), MagicMock(), MagicMock()]
    for llm in llms:
        llm.invoke.side_effect = rate_limit(42)
    with patch("ai_generator.ChatOpenAI", side_effect=llms):
        gen = AIGenerator("k", "http://x", "m", providers=specs())
        with pytest.raises(LLMRateLimitError) as exc:
            gen.generate_response("q")
    assert exc.value.retry_after == 42


def rag_config(**kw):
    return SimpleNamespace(**kw)


def test_rag_system_uses_the_single_endpoint_when_no_chain_is_configured():
    assert RAGSystem._llm_providers(rag_config()) is None
    assert RAGSystem._llm_providers(rag_config(LLM_PROVIDERS="")) is None


def test_rag_system_builds_the_chain_from_the_environment(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setenv("NVIDIA_API_KEY", "n")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    result = RAGSystem._llm_providers(rag_config(LLM_PROVIDERS="groq,nvidia,google"))
    assert result is not None
    assert [s.name for s in result] == ["groq", "nvidia"]


def test_rag_system_fails_loudly_when_no_listed_provider_has_a_key(monkeypatch):
    for name in ("GROQ_API_KEY", "NVIDIA_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="API key"):
        RAGSystem._llm_providers(rag_config(LLM_PROVIDERS="groq,nvidia"))


# --- flattening tool turns (Gemini 3 thought signatures) ------------------------------------


def tool_turn(call_id="c1", name="search_course_content", args=None, result="R"):
    call = {"id": call_id, "name": name, "args": args or {"query": "hyde"}, "type": "tool_call"}
    return [
        AIMessage(content="", tool_calls=[call]),
        ToolMessage(content=result, tool_call_id=call_id),
    ]


def test_flatten_turns_a_tool_round_into_plain_text():
    flat = flatten_tool_turns(
        [SystemMessage(content="sys"), HumanMessage(content="q"), *tool_turn(result="HyDE is ...")]
    )
    assert [type(m).__name__ for m in flat] == [
        "SystemMessage",
        "HumanMessage",
        "AIMessage",
        "HumanMessage",
    ]
    assert not any(isinstance(m, ToolMessage) or getattr(m, "tool_calls", None) for m in flat)
    assert 'search_course_content({"query": "hyde"})' in flat[2].content
    assert "[search_course_content]" in flat[3].content and "HyDE is ..." in flat[3].content


def test_flatten_keeps_the_assistants_own_text_and_merges_parallel_results():
    calls = [
        {"id": "a", "name": "search", "args": {"query": "one"}, "type": "tool_call"},
        {"id": "b", "name": "search", "args": {"query": "two"}, "type": "tool_call"},
    ]
    flat = flatten_tool_turns(
        [
            HumanMessage(content="q"),
            AIMessage(content="Let me check.", tool_calls=calls),
            ToolMessage(content="R1", tool_call_id="a"),
            ToolMessage(content="R2", tool_call_id="b"),
        ]
    )
    assert len(flat) == 3  # question, assistant note, ONE results message
    assert str(flat[1].content).startswith("Let me check.")
    assert "R1" in flat[2].content and "R2" in flat[2].content


def test_flatten_merges_adjacent_user_messages_such_as_the_answer_now_notice():
    flat = flatten_tool_turns(
        [HumanMessage(content="q"), *tool_turn(), HumanMessage(content="Answer now.")]
    )
    assert [type(m).__name__ for m in flat] == ["HumanMessage", "AIMessage", "HumanMessage"]
    assert str(flat[-1].content).endswith("Answer now.") and "R" in str(flat[-1].content)


def test_flatten_leaves_plain_conversations_alone_and_does_not_mutate_the_input():
    plain = [SystemMessage(content="s"), HumanMessage(content="q"), AIMessage(content="a")]
    assert flatten_tool_turns(plain) == plain
    original = [HumanMessage(content="q"), *tool_turn()]
    snapshot = list(original)
    flatten_tool_turns(original)
    assert original == snapshot


def test_adapter_is_applied_only_to_its_own_endpoint():
    clock = Clock()
    a = make_endpoint("a", clock, [rate_limit()])
    b = make_endpoint("b", clock, [ai("from b")])
    b.adapt = lambda messages: [HumanMessage(content="adapted")]
    LLMRouter([a, b]).invoke(MESSAGES)
    assert a.llm.invoke.call_args.args[0] == MESSAGES  # sent as is
    assert b.llm.invoke.call_args.args[0] == [HumanMessage(content="adapted")]


def test_adapter_is_applied_when_streaming_too():
    async def scenario():
        clock = Clock()
        seen = []

        def factory(messages):
            seen.append(messages)
            return chunks("ok")

        a = stream_endpoint("a", clock, factory)
        a.adapt = lambda messages: [HumanMessage(content="adapted")]
        await collect(LLMRouter([a]).astream(MESSAGES))
        assert seen == [[HumanMessage(content="adapted")]]

    asyncio.run(scenario())


def test_google_flattens_by_default_and_the_flag_can_be_overridden():
    (google,) = llm_provider_settings("google", {"GOOGLE_API_KEY": "k"})
    (groq,) = llm_provider_settings("groq", {"GROQ_API_KEY": "k"})
    assert google["flatten_tool_messages"] is True and groq["flatten_tool_messages"] is False
    (off,) = llm_provider_settings(
        "google", {"GOOGLE_API_KEY": "k", "GOOGLE_FLATTEN_TOOL_MESSAGES": "false"}
    )
    assert off["flatten_tool_messages"] is False
    (on,) = llm_provider_settings(
        "groq", {"GROQ_API_KEY": "k", "GROQ_FLATTEN_TOOL_MESSAGES": "true"}
    )
    assert on["flatten_tool_messages"] is True


def test_generator_sets_the_adapter_only_for_flattening_providers():
    chain = [
        ProviderSpec("groq", "http://groq/v1", "gk", ["g1"]),
        ProviderSpec("google", "http://g/v1", "ggk", ["m1", "m2"], flatten_tool_messages=True),
    ]
    with patch("ai_generator.ChatOpenAI"):
        gen = AIGenerator("k", "http://x", "m", providers=chain)
    adapters = [e.adapt for e in gen.llm.endpoints]
    assert adapters[0] is None
    assert adapters[1] is flatten_tool_turns and adapters[2] is flatten_tool_turns
