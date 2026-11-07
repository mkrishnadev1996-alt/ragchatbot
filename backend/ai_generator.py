import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import contextmanager
from typing import Any

import openai
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from models import Source
from utils.http_utils import parse_retry_after

logger = logging.getLogger(__name__)


class LLMRateLimitError(Exception):
    """The LLM provider kept rejecting requests with HTTP 429 after the SDK's own retries"""

    USER_MESSAGE = (
        "The AI service is receiving too many requests right now. "
        "Please wait a moment and try again."
    )

    def __init__(self, retry_after: float | None = None):
        self.retry_after = retry_after
        super().__init__(self.USER_MESSAGE)


class AIGenerator:
    """Handles interactions with an OpenAI-compatible chat endpoint (Groq, Google Gemini, etc.)
    through LangChain's ChatOpenAI for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational
content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content
  or detailed educational materials
- Search again when one search is not enough (comparing courses, questions with several
  parts, or a first search that found nothing relevant). Never repeat a search you
  already ran, and stop searching as soon as you can answer
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations,
   or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    # Sent after the last allowed tool round, so the final (tool-less) call answers instead of
    # trying to call a tool again (providers reject a tool call when no tools are offered)
    TOOL_LIMIT_NOTICE = (
        "Searching is no longer available. Answer the question now using only the "
        "information already retrieved."
    )

    TRUNCATION_NOTICE = (
        "\n\n*(This answer was cut off because it reached the length limit. "
        "Ask a narrower question to get the rest.)*"
    )

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        max_tokens: int = 1600,
        temperature: float = 0,
        max_retries: int = 3,
        timeout: float | None = 60,
        max_tool_rounds: int = 3,
    ):
        # max_retries is passed to the OpenAI SDK, which retries 429s and 5xx with backoff
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,  # type: ignore[arg-type]
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,  # type: ignore[call-arg]
            max_retries=max_retries,
            timeout=timeout,
        )
        self.model = model
        # Safety cap on model -> tools -> model rounds (cost and runaway-loop guard);
        # the model may search as often as it needs up to this many rounds.
        self.max_tool_rounds = max(1, max_tool_rounds)

    def generate_response(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
        tools: list | None = None,
        tool_manager=None,
        sources_out: list[Source] | None = None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Prior chat messages as role/content dicts, sent as
                messages between the system prompt and the query
            tools: Available tools the AI can use (nested function-tool format
                {"type": "function", "function": {...}}, as accepted by bind_tools)
            tool_manager: Manager to execute tools
            sources_out: Optional per-request list that tool sources are appended to

        Returns:
            Generated response as string
        """
        messages = self._build_messages(query, conversation_history)

        # Bind tools if available; otherwise call the plain model
        bound = self.llm.bind_tools(tools, tool_choice="auto") if tools else self.llm
        runnable = bound
        response = self._invoke(runnable, messages)

        # Agentic loop: while the model asks for tools, run them and call it again. Up to
        # max_tool_rounds rounds; the call after the last round has no tools.
        rounds = 0
        while response.tool_calls and tool_manager and rounds < self.max_tool_rounds:
            messages = self._run_tools(response, messages, tool_manager, sources_out)
            rounds += 1
            if rounds >= self.max_tool_rounds:
                messages = [*messages, HumanMessage(content=self.TOOL_LIMIT_NOTICE)]
                runnable = self.llm
            response = self._invoke(runnable, messages)

        return self._text(response)

    async def astream_response(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
        tools: list | None = None,
        tool_manager=None,
        sources_out: list[Source] | None = None,
    ) -> AsyncIterator[str]:
        """Async twin of generate_response: yields the answer text as it is produced.

        Same agentic loop (up to max_tool_rounds rounds, the call after the last one has no
        tools). Every model call is streamed; if a call that goes on to use tools also wrote
        some text, the next call's text starts after a blank line. The LLM calls are natively
        async (`astream`); the blocking tool execution (vector search) runs in a worker thread
        so the event loop stays free. A truncation notice is yielded last if the provider
        stopped at max_tokens. LLMRateLimitError is raised from the first `anext()` when the
        provider rejects the initial request.
        """
        messages = self._build_messages(query, conversation_history)
        bound = self.llm.bind_tools(tools, tool_choice="auto") if tools else self.llm
        runnable = bound

        rounds = 0
        spoke = False  # an earlier call already produced text
        while True:
            result: list[AIMessage] = []
            wrote = False
            async for delta in self._astream(runnable, messages, result):
                if spoke and not wrote:
                    yield "\n\n"
                wrote = True
                yield delta
            spoke = spoke or wrote
            final = result[0]

            if not (final.tool_calls and tool_manager and rounds < self.max_tool_rounds):
                break
            messages = await asyncio.to_thread(
                self._run_tools, final, messages, tool_manager, sources_out
            )
            rounds += 1
            if rounds >= self.max_tool_rounds:
                messages = [*messages, HumanMessage(content=self.TOOL_LIMIT_NOTICE)]
                runnable = self.llm
            else:
                runnable = bound

        notice = self._truncation_notice(final)
        if notice:
            yield notice

    def _build_messages(
        self, query: str, conversation_history: list[dict[str, str]] | None
    ) -> list[BaseMessage]:
        messages: list[BaseMessage] = [SystemMessage(content=self.SYSTEM_PROMPT)]
        for item in conversation_history or []:
            if item["role"] == "assistant":
                messages.append(AIMessage(content=item["content"]))
            else:
                messages.append(HumanMessage(content=item["content"]))
        messages.append(HumanMessage(content=query))
        return messages

    def _run_tools(
        self,
        response: AIMessage,
        messages: list[BaseMessage],
        tool_manager,
        sources_out: list[Source] | None = None,
    ) -> list[BaseMessage]:
        """Execute the tool calls in response; return the messages for the follow-up call"""
        follow_up: list[BaseMessage] = [*messages, response]

        # Execute all tool calls and append each result as its own ToolMessage
        for tool_call in response.tool_calls:
            name, arguments = tool_call["name"], tool_call["args"]
            logger.info("Executing tool %s", name)
            logger.debug("Tool %s arguments: %s", name, arguments)  # may hold user text
            try:
                if not isinstance(arguments, dict):
                    raise TypeError("tool arguments must be a JSON object")
                result = tool_manager.execute_tool(name, **arguments)
                text = result.text
                if sources_out is not None:
                    # Several rounds may find the same lesson: list each citation once
                    seen = {(s.label, s.url) for s in sources_out}
                    for source in result.sources:
                        if (source.label, source.url) not in seen:
                            seen.add((source.label, source.url))
                            sources_out.append(source)
            except Exception as e:
                # Bad arguments or a failing tool must not fail the request: hand the error
                # to the model so it can answer without the tool or retry differently.
                logger.exception("Tool %s failed", name)
                text = f"Error: the {name} tool failed ({type(e).__name__}). Answer without it."
            logger.debug("Tool %s returned %d chars", name, len(text))
            follow_up.append(ToolMessage(content=text, tool_call_id=tool_call["id"] or ""))
        return follow_up

    def _truncation_notice(self, response: AIMessage) -> str:
        """The notice to show if the provider stopped at max_tokens, else an empty string"""
        if response.response_metadata.get("finish_reason") == "length":
            logger.warning("LLM answer truncated at max_tokens (model=%s)", self.model)
            return self.TRUNCATION_NOTICE
        return ""

    def _text(self, response: AIMessage) -> str:
        """The answer text, with a visible notice if the provider stopped at max_tokens"""
        return response.text + self._truncation_notice(response)

    @contextmanager
    def _translate_errors(self):
        """Translate provider rate limits into LLMRateLimitError and log API failures"""
        try:
            yield
        except openai.RateLimitError as e:
            headers = getattr(getattr(e, "response", None), "headers", None)
            retry_after = parse_retry_after(headers)
            logger.warning("LLM rate limit hit after SDK retries (retry_after=%s)", retry_after)
            raise LLMRateLimitError(retry_after) from e
        except openai.APIError:
            logger.exception("LLM API request failed (model=%s)", self.model)
            raise

    def _invoke(self, runnable: Any, messages: list[BaseMessage]) -> AIMessage:
        """Invoke the model, translating provider rate limits into LLMRateLimitError"""
        logger.debug("LLM request: model=%s messages=%d", self.model, len(messages))
        with self._translate_errors():
            return runnable.invoke(messages)

    async def _astream(
        self, runnable: Any, messages: list[BaseMessage], result: list[AIMessage]
    ) -> AsyncIterator[str]:
        """Stream the model asynchronously, yielding text deltas.

        The aggregated message is appended to `result` once the stream ends (async generators
        cannot return a value).
        """
        logger.debug("LLM stream request: model=%s messages=%d", self.model, len(messages))
        aggregate: Any = None
        with self._translate_errors():
            async for chunk in runnable.astream(messages):
                aggregate = chunk if aggregate is None else aggregate + chunk
                if chunk.text:
                    yield chunk.text
        result.append(aggregate if aggregate is not None else AIMessage(content=""))
