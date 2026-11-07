import logging
from typing import Any

import openai
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
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
- **One search per query maximum**
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

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        max_tokens: int = 1600,
        temperature: float = 0,
        max_retries: int = 3,
        timeout: float | None = 60,
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

    def generate_response(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
        tools: list | None = None,
        tool_manager=None,
        sources_out: list[str] | None = None,
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
        messages: list[BaseMessage] = [SystemMessage(content=self.SYSTEM_PROMPT)]
        for item in conversation_history or []:
            if item["role"] == "assistant":
                messages.append(AIMessage(content=item["content"]))
            else:
                messages.append(HumanMessage(content=item["content"]))
        messages.append(HumanMessage(content=query))

        # Bind tools if available; otherwise call the plain model
        runnable = self.llm.bind_tools(tools, tool_choice="auto") if tools else self.llm
        response = self._invoke(runnable, messages)

        # Handle tool execution if needed
        if response.tool_calls and tool_manager:
            return self._handle_tool_execution(response, messages, tool_manager, sources_out)

        # Return direct response
        return response.text

    def _handle_tool_execution(
        self,
        response: AIMessage,
        messages: list[BaseMessage],
        tool_manager,
        sources_out: list[str] | None = None,
    ):
        """
        Handle execution of tool calls and get follow-up response.

        Args:
            response: The AI message containing tool call requests
            messages: Messages sent in the initial request (system + history + user query)
            tool_manager: Manager to execute tools
            sources_out: Optional list that sources from tool results are appended to

        Returns:
            Final response text after tool execution
        """
        follow_up: list[BaseMessage] = [*messages, response]

        # Execute all tool calls and append each result as its own ToolMessage
        for tool_call in response.tool_calls:
            name, arguments = tool_call["name"], tool_call["args"]
            logger.info("Executing tool %s", name)
            try:
                if not isinstance(arguments, dict):
                    raise TypeError("tool arguments must be a JSON object")
                result = tool_manager.execute_tool(name, **arguments)
                text = result.text
                if sources_out is not None:
                    sources_out.extend(result.sources)
            except Exception as e:
                # Bad arguments or a failing tool must not fail the request: hand the error
                # to the model so it can answer without the tool or retry differently.
                logger.exception("Tool %s failed", name)
                text = f"Error: the {name} tool failed ({type(e).__name__}). Answer without it."
            logger.debug("Tool %s returned %d chars", name, len(text))
            follow_up.append(ToolMessage(content=text, tool_call_id=tool_call["id"] or ""))

        # Final call on the unbound model: no tools, so this is the last round
        final_response = self._invoke(self.llm, follow_up)
        return final_response.text

    def _invoke(self, runnable: Any, messages: list[BaseMessage]) -> AIMessage:
        """Invoke the model, translating provider rate limits into LLMRateLimitError"""
        logger.debug("LLM request: model=%s messages=%d", self.model, len(messages))
        try:
            return runnable.invoke(messages)
        except openai.RateLimitError as e:
            headers = getattr(getattr(e, "response", None), "headers", None)
            retry_after = parse_retry_after(headers)
            logger.warning("LLM rate limit hit after SDK retries (retry_after=%s)", retry_after)
            raise LLMRateLimitError(retry_after) from e
        except openai.APIError:
            logger.exception("LLM API request failed (model=%s)", self.model)
            raise
