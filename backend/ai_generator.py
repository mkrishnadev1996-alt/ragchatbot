import json
from typing import Any

from openai import OpenAI


class AIGenerator:
    """Handles interactions with an OpenAI-compatible chat API (Groq, Google Gemini, etc.)
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
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

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
                message turns (not folded into the system prompt)
            tools: Available tools the AI can use (OpenAI function-tool format)
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        messages: list[dict[str, Any]] = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": query})

        # Prepare API call parameters
        api_params: dict[str, Any] = {**self.base_params, "messages": messages}

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = "auto"

        # Get response from the model
        response = self.client.chat.completions.create(**api_params)
        message = response.choices[0].message

        # Handle tool execution if needed
        if message.tool_calls and tool_manager:
            return self._handle_tool_execution(message, messages, tool_manager)

        # Return direct response
        return message.content or ""

    def _handle_tool_execution(
        self, assistant_message, messages: list[dict[str, Any]], tool_manager
    ):
        """
        Handle execution of tool calls and get follow-up response.

        Args:
            assistant_message: The assistant message containing tool call requests
            messages: Messages sent in the initial request (system + user)
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = messages.copy()

        # Add AI's tool call request
        messages.append(
            {
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in assistant_message.tool_calls
                ],
            }
        )

        # Execute all tool calls and append each result as its own tool message
        for tool_call in assistant_message.tool_calls:
            arguments = json.loads(tool_call.function.arguments or "{}")
            tool_result = tool_manager.execute_tool(tool_call.function.name, **arguments)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": tool_result})

        # Prepare final API call without tools
        final_params: dict[str, Any] = {**self.base_params, "messages": messages}

        # Get final response
        final_response = self.client.chat.completions.create(**final_params)
        return final_response.choices[0].message.content or ""
