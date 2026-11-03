import json
from typing import Any

import openai
from openai import OpenAI


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
    """Handles interactions with an OpenAI-compatible Responses API (Groq, Google Gemini, etc.)
    for generating responses"""

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

    def __init__(self, api_key: str, base_url: str, model: str):
        # The SDK retries 429s and 5xx with backoff; default is 2, raised to 3 here
        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=3)
        self.model = model

        # Pre-build base API parameters
        self.base_params: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "max_output_tokens": 1600,
        }

    def generate_response(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
        tools: list | None = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Prior chat messages as role/content dicts, sent as
                input items (not folded into the instructions)
            tools: Available tools the AI can use (function-tool format; the nested
                chat-completions shape {"type": "function", "function": {...}} is accepted)
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """
        input_items: list[Any] = []
        if conversation_history:
            input_items.extend(conversation_history)
        input_items.append({"role": "user", "content": query})

        # Prepare API call parameters
        api_params: dict[str, Any] = {
            **self.base_params,
            "instructions": self.SYSTEM_PROMPT,
            "input": input_items,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = [self._to_responses_tool(tool) for tool in tools]
            api_params["tool_choice"] = "auto"

        # Get response from the model
        response = self._create_response(**api_params)

        function_calls = [item for item in response.output if item.type == "function_call"]

        # Handle tool execution if needed
        if function_calls and tool_manager:
            return self._handle_tool_execution(response, input_items, tool_manager)

        # Return direct response
        return response.output_text or ""

    def _handle_tool_execution(self, response, input_items: list[Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.

        Args:
            response: The response containing function call requests
            input_items: Input items sent in the initial request (history + user query)
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Carry over the model's output items (function calls, plus any reasoning items
        # the provider requires to be returned with them)
        follow_up_input = [*input_items, *response.output]

        # Execute all tool calls and append each result as its own function_call_output
        for item in response.output:
            if item.type != "function_call":
                continue
            arguments = json.loads(item.arguments or "{}")
            tool_result = tool_manager.execute_tool(item.name, **arguments)
            follow_up_input.append(
                {"type": "function_call_output", "call_id": item.call_id, "output": tool_result}
            )

        # Prepare final API call without tools
        final_params: dict[str, Any] = {
            **self.base_params,
            "instructions": self.SYSTEM_PROMPT,
            "input": follow_up_input,
        }

        # Get final response
        final_response = self._create_response(**final_params)
        return final_response.output_text or ""

    def _create_response(self, **params):
        """Call the Responses API, translating provider rate limits into LLMRateLimitError"""
        try:
            return self.client.responses.create(**params)
        except openai.RateLimitError as e:
            raise LLMRateLimitError(self._retry_after_seconds(e)) from e

    @staticmethod
    def _retry_after_seconds(error: openai.APIStatusError) -> float | None:
        """Read the Retry-After header (seconds) from a rate-limit error, if sent"""
        headers = getattr(getattr(error, "response", None), "headers", None) or {}
        value = headers.get("retry-after")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_responses_tool(tool: dict[str, Any]) -> dict[str, Any]:
        """Convert a nested function-tool definition to the flat Responses API shape"""
        if "function" not in tool:
            return tool
        function = tool["function"]
        flat: dict[str, Any] = {"type": "function", "name": function["name"]}
        for key in ("description", "parameters", "strict"):
            if key in function:
                flat[key] = function[key]
        return flat
