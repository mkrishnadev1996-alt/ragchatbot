"""Try a chain of LLM endpoints in order, skipping ones that are currently failing.

The chain is a list of (provider, model) entries, e.g. groq/gpt-oss-120b, groq/gpt-oss-20b,
nvidia/nemotron-..., google/gemini-.... Each entry has its own circuit breaker, because
quotas are per model (Groq's daily token limit) and outages are per endpoint.

The router exposes the small slice of the LangChain chat-model interface that `AIGenerator`
uses (`invoke`, `astream`, `bind_tools`), so the generator does not know whether it talks to
one model or to a chain.

Failover rules
  * Only errors from the provider side move on to the next entry and count against the
    breaker: rate limits, 5xx, timeouts/connection errors, 401/403/404 (bad key, model gone).
  * Request-shaped errors (400/422) also move on to the next entry for this call, because
    another model may accept what this one rejected, but they do not trip the breaker, so a
    bad input can never take every provider out.
  * Streaming can only fail over before the first token; later errors are raised.
  * If every entry is in cooldown, the one that reopens soonest is tried anyway, so the
    chain never refuses to try.
Breaker state is per process (not shared between workers).
"""

import json
import logging
import threading
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import openai
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from utils.http_utils import parse_retry_after

logger = logging.getLogger(__name__)

# Statuses that mean "the other side is unhealthy or we cannot use it", not "bad request"
_PROVIDER_FAULT_STATUSES = {401, 402, 403, 404, 408, 409, 429}


class CircuitBreaker:
    """Closed -> open (cooldown) -> half-open (next call is a trial) -> closed on success."""

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown: float = 60.0,
        max_cooldown: float = 900.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.failure_threshold = max(1, failure_threshold)
        self.cooldown = cooldown
        self.max_cooldown = max(cooldown, max_cooldown)
        self._clock = clock
        self._failures = 0
        self._open_until = 0.0
        self._lock = threading.Lock()

    def allow(self) -> bool:
        """True when closed, or when the cooldown has passed (the call is a trial)."""
        with self._lock:
            return self._clock() >= self._open_until

    def reopens_in(self) -> float:
        with self._lock:
            return max(0.0, self._open_until - self._clock())

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._open_until = 0.0

    def record_failure(self, rate_limited: bool = False, retry_after: float | None = None) -> None:
        """A rate limit opens the breaker at once (for Retry-After if given, within bounds);
        other failures open it once `failure_threshold` in a row is reached."""
        with self._lock:
            self._failures += 1
            now = self._clock()
            if rate_limited:
                wait = retry_after if retry_after else self.cooldown
                self._open_until = now + min(max(wait, self.cooldown), self.max_cooldown)
            elif self._failures >= self.failure_threshold:
                self._open_until = now + self.cooldown


def classify_failure(exc: BaseException) -> tuple[bool, bool, float | None] | None:
    """(provider_fault, rate_limited, retry_after) for errors that allow failover, else None."""
    if isinstance(exc, openai.RateLimitError):
        headers = getattr(getattr(exc, "response", None), "headers", None)
        return True, True, parse_retry_after(headers)
    if isinstance(exc, openai.APIStatusError):
        status = getattr(exc, "status_code", 0)
        return (status >= 500 or status in _PROVIDER_FAULT_STATUSES), False, None
    if isinstance(exc, openai.APIError | httpx.HTTPError):  # timeouts, connection errors
        return True, False, None
    return None


def flatten_tool_turns(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Rewrite tool-call turns as plain text, for endpoints that cannot take them back.

    Gemini 3 models require the opaque `thought_signature` from their own tool calls to be
    echoed in the next request (OpenAI-compatible field `extra_content`), which LangChain's
    chat model drops, so the second call of a tool round fails with HTTP 400. Flattened, the
    assistant's tool call becomes a text note and the tool results one user message; the
    model still sees the search results, just not as native tool turns. Adjacent user
    messages are merged, since some endpoints reject two in a row.
    """
    names: dict[str, str] = {}
    out: list[BaseMessage] = []
    pending: list[str] = []  # tool results waiting to be emitted as one user message

    def flush() -> None:
        if pending:
            out.append(
                HumanMessage(content="Results of your tool calls:\n\n" + "\n\n".join(pending))
            )
            pending.clear()

    for message in messages:
        if isinstance(message, ToolMessage):
            label = names.get(message.tool_call_id, message.name or "tool")
            pending.append(f"[{label}]\n{message.content}")
            continue
        flush()
        if isinstance(message, AIMessage) and message.tool_calls:
            calls = []
            for call in message.tool_calls:
                names[call["id"] or ""] = call["name"]
                calls.append(f"{call['name']}({json.dumps(call['args'], ensure_ascii=False)})")
            note = f"[I called: {'; '.join(calls)}]"
            text = message.text
            out.append(AIMessage(content=f"{text}\n{note}" if text else note))
        elif isinstance(message, HumanMessage) and out and isinstance(out[-1], HumanMessage):
            out[-1] = HumanMessage(content=f"{out[-1].content}\n\n{message.content}")
        else:
            out.append(message)
    flush()
    # a results message may now follow a user message (e.g. the next question): merge those too
    merged: list[BaseMessage] = []
    for message in out:
        if isinstance(message, HumanMessage) and merged and isinstance(merged[-1], HumanMessage):
            merged[-1] = HumanMessage(content=f"{merged[-1].content}\n\n{message.content}")
        else:
            merged.append(message)
    return merged


@dataclass
class ProviderSpec:
    """One provider in the chain: its endpoint, key and the models to try, best first."""

    name: str
    base_url: str
    api_key: str
    models: list[str]
    flatten_tool_messages: bool = False  # see flatten_tool_turns


@dataclass
class Endpoint:
    """One link of the chain: a provider's model and its client."""

    provider: str
    model: str
    llm: Any  # a LangChain chat model
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    # Rewrites the messages before every call to this endpoint (None = send them as they are)
    adapt: Callable[[list[BaseMessage]], list[BaseMessage]] | None = None

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}"


class RoutedModel:
    """A set of endpoints, each already bound (or not) to tools, tried in chain order."""

    def __init__(self, endpoints: list[Endpoint], runnables: list[Any]):
        self._endpoints = endpoints
        self._runnables = runnables

    def _order(self) -> list[int]:
        """Healthy entries in chain order; if none, just the one that reopens soonest."""
        healthy = [i for i, e in enumerate(self._endpoints) if e.breaker.allow()]
        if healthy:
            skipped = [e.label for e in self._endpoints if not e.breaker.allow()]
            if skipped:
                logger.info("LLM chain skipping (cooling down): %s", ", ".join(skipped))
            return healthy
        soonest = min(
            range(len(self._endpoints)), key=lambda i: self._endpoints[i].breaker.reopens_in()
        )
        logger.warning(
            "Every LLM endpoint is cooling down; trying %s anyway", self._endpoints[soonest].label
        )
        return [soonest]

    @staticmethod
    def _messages_for(endpoint: Endpoint, messages: list[BaseMessage]) -> list[BaseMessage]:
        return endpoint.adapt(messages) if endpoint.adapt else messages

    def _failed(self, index: int, exc: BaseException, last: bool) -> bool:
        """Record a failure; True if the error allows trying the next entry."""
        endpoint = self._endpoints[index]
        info = classify_failure(exc)
        if info is None:
            return False  # not an LLM-provider error (a bug): do not mask it
        provider_fault, rate_limited, retry_after = info
        if provider_fault:
            endpoint.breaker.record_failure(rate_limited, retry_after)
        logger.warning(
            "LLM %s failed (%s)%s",
            endpoint.label,
            type(exc).__name__,
            "" if last else "; trying the next endpoint",
        )
        return True

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        order = self._order()
        for position, index in enumerate(order):
            endpoint = self._endpoints[index]
            try:
                response = self._runnables[index].invoke(self._messages_for(endpoint, messages))
            except Exception as exc:
                if (
                    self._failed(index, exc, last=position == len(order) - 1)
                    and position < len(order) - 1
                ):
                    continue
                raise
            endpoint.breaker.record_success()
            logger.info("LLM answered via %s", endpoint.label)
            return response
        raise RuntimeError("no LLM endpoints configured")  # pragma: no cover

    async def astream(self, messages: list[BaseMessage]) -> AsyncIterator[Any]:
        order = self._order()
        for position, index in enumerate(order):
            endpoint = self._endpoints[index]
            started = False
            try:
                async for chunk in self._runnables[index].astream(
                    self._messages_for(endpoint, messages)
                ):
                    started = True
                    yield chunk
            except Exception as exc:
                if started:  # tokens already sent: a different model cannot continue the answer
                    self._failed(index, exc, last=True)
                    raise
                if (
                    self._failed(index, exc, last=position == len(order) - 1)
                    and position < len(order) - 1
                ):
                    continue
                raise
            endpoint.breaker.record_success()
            logger.info("LLM answered via %s", endpoint.label)
            return


class LLMRouter:
    """Chain of endpoints with the chat-model methods the generator needs."""

    def __init__(self, endpoints: list[Endpoint]):
        if not endpoints:
            raise ValueError("LLMRouter needs at least one endpoint")
        self.endpoints = endpoints

    @property
    def primary(self) -> Endpoint:
        return self.endpoints[0]

    def _routed(self, runnables: list[Any]) -> RoutedModel:
        return RoutedModel(self.endpoints, runnables)

    def bind_tools(self, tools: list, **kwargs: Any) -> RoutedModel:
        return self._routed([e.llm.bind_tools(tools, **kwargs) for e in self.endpoints])

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        return self._routed([e.llm for e in self.endpoints]).invoke(messages)

    def astream(self, messages: list[BaseMessage]) -> AsyncIterator[Any]:
        return self._routed([e.llm for e in self.endpoints]).astream(messages)
