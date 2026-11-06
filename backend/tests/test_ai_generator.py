import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import openai
from ai_generator import AIGenerator, LLMRateLimitError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


def make_rate_limit_error(retry_after=None):
    """Build the SDK's RateLimitError as it is raised on HTTP 429"""
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    response = SimpleNamespace(status_code=429, headers=headers, request=None)
    # The fake only needs status_code/headers/request, so cast it to the SDK's Response type
    return openai.RateLimitError("rate limited", response=cast(Any, response), body=None)


def make_tool_call(call_id, name, args):
    return {"id": call_id, "name": name, "args": args, "type": "tool_call"}


def make_message(text="", tool_calls=None):
    return AIMessage(content=text, tool_calls=tool_calls or [])


class TestAIGenerator(unittest.TestCase):
    def setUp(self):
        patcher = patch("ai_generator.ChatOpenAI")
        self.chat_cls = patcher.start()
        self.addCleanup(patcher.stop)

        self.llm = self.chat_cls.return_value
        self.bound = self.llm.bind_tools.return_value
        self.generator = AIGenerator(api_key="key", base_url="http://llm.test", model="test-model")

    def test_client_is_built_from_arguments_and_limits(self):
        self.chat_cls.assert_called_once_with(
            model="test-model",
            api_key="key",
            base_url="http://llm.test",
            temperature=0,
            max_tokens=1600,
            max_retries=3,
        )

    def test_direct_answer_without_tools(self):
        self.llm.invoke.return_value = make_message("Hello there")

        result = self.generator.generate_response("hi")

        self.assertEqual(result, "Hello there")
        self.assertEqual(self.llm.invoke.call_count, 1)
        self.llm.bind_tools.assert_not_called()
        messages = self.llm.invoke.call_args.args[0]
        self.assertEqual(
            messages,
            [SystemMessage(content=AIGenerator.SYSTEM_PROMPT), HumanMessage(content="hi")],
        )

    def test_list_content_is_joined_into_text(self):
        self.llm.invoke.return_value = AIMessage(
            content=[{"type": "text", "text": "Hel"}, "lo", {"type": "image_url"}]
        )
        self.assertEqual(self.generator.generate_response("hi"), "Hello")

    def test_tools_are_bound_with_auto_choice(self):
        self.bound.invoke.return_value = make_message("ok")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "t",
                    "description": "d",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        result = self.generator.generate_response("q", tools=tools)

        self.assertEqual(result, "ok")
        self.llm.bind_tools.assert_called_once_with(tools, tool_choice="auto")
        self.llm.invoke.assert_not_called()

    def test_conversation_history_sits_between_system_prompt_and_query(self):
        self.llm.invoke.return_value = make_message("ok")
        history = [
            {"role": "user", "content": "earlier q"},
            {"role": "assistant", "content": "earlier a"},
        ]

        self.generator.generate_response("now", conversation_history=history)

        messages = self.llm.invoke.call_args.args[0]
        self.assertEqual(
            messages,
            [
                SystemMessage(content=AIGenerator.SYSTEM_PROMPT),
                HumanMessage(content="earlier q"),
                AIMessage(content="earlier a"),
                HumanMessage(content="now"),
            ],
        )

    def test_tool_call_runs_tool_and_makes_follow_up_call_without_tools(self):
        tool_call = make_tool_call("call_1", "search_course_content", {"query": "rag"})
        first = make_message(tool_calls=[tool_call])
        self.bound.invoke.return_value = first
        self.llm.invoke.return_value = make_message("Final answer")
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "SEARCH RESULT"

        result = self.generator.generate_response(
            "q",
            tools=[{"type": "function", "function": {"name": "search_course_content"}}],
            tool_manager=tool_manager,
        )

        self.assertEqual(result, "Final answer")
        tool_manager.execute_tool.assert_called_once_with("search_course_content", query="rag")
        self.assertEqual(self.bound.invoke.call_count, 1)
        self.assertEqual(self.llm.invoke.call_count, 1)

        follow_up = self.llm.invoke.call_args.args[0]
        self.assertEqual(follow_up[1], HumanMessage(content="q"))
        self.assertIs(follow_up[2], first)
        self.assertEqual(follow_up[3], ToolMessage(content="SEARCH RESULT", tool_call_id="call_1"))

    def test_multiple_tool_calls_each_get_a_tool_message(self):
        calls = [
            make_tool_call("a", "search_course_content", {"query": "one"}),
            make_tool_call("b", "search_course_content", {"query": "two"}),
        ]
        self.bound.invoke.return_value = make_message(tool_calls=calls)
        self.llm.invoke.return_value = make_message("done")
        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = ["R1", "R2"]

        self.generator.generate_response("q", tools=[{}], tool_manager=tool_manager)

        self.assertEqual(tool_manager.execute_tool.call_count, 2)
        follow_up = self.llm.invoke.call_args.args[0]
        outputs = [m for m in follow_up if isinstance(m, ToolMessage)]
        self.assertEqual([(o.tool_call_id, o.content) for o in outputs], [("a", "R1"), ("b", "R2")])

    def test_empty_tool_arguments_are_treated_as_no_args(self):
        self.bound.invoke.return_value = make_message(tool_calls=[make_tool_call("x", "noop", {})])
        self.llm.invoke.return_value = make_message("ok")
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "r"

        self.generator.generate_response("q", tools=[{}], tool_manager=tool_manager)

        tool_manager.execute_tool.assert_called_once_with("noop")

    def test_tool_calls_without_manager_return_content(self):
        self.bound.invoke.return_value = make_message(
            "partial", tool_calls=[make_tool_call("x", "search_course_content", {})]
        )

        result = self.generator.generate_response("q", tools=[{}], tool_manager=None)

        self.assertEqual(result, "partial")
        self.llm.invoke.assert_not_called()

    def test_rate_limit_becomes_llm_rate_limit_error_with_retry_after(self):
        self.llm.invoke.side_effect = make_rate_limit_error(retry_after="7")

        with self.assertRaises(LLMRateLimitError) as ctx:
            self.generator.generate_response("q")

        self.assertEqual(ctx.exception.retry_after, 7.0)
        self.assertIn("too many requests", str(ctx.exception))

    def test_rate_limit_without_retry_after_header_has_no_delay(self):
        self.llm.invoke.side_effect = make_rate_limit_error()

        with self.assertRaises(LLMRateLimitError) as ctx:
            self.generator.generate_response("q")

        self.assertIsNone(ctx.exception.retry_after)

    def test_unparseable_retry_after_header_is_ignored(self):
        self.llm.invoke.side_effect = make_rate_limit_error(
            retry_after="Wed, 21 Oct 2026 07:28:00 GMT"
        )

        with self.assertRaises(LLMRateLimitError) as ctx:
            self.generator.generate_response("q")

        self.assertIsNone(ctx.exception.retry_after)

    def test_rate_limit_on_tool_follow_up_call_is_also_translated(self):
        self.bound.invoke.return_value = make_message(
            tool_calls=[make_tool_call("c", "search_course_content", {})]
        )
        self.llm.invoke.side_effect = make_rate_limit_error(retry_after="3")
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "r"

        with self.assertRaises(LLMRateLimitError) as ctx:
            self.generator.generate_response("q", tools=[{}], tool_manager=tool_manager)

        self.assertEqual(ctx.exception.retry_after, 3.0)

    def test_other_errors_are_not_translated(self):
        self.llm.invoke.side_effect = RuntimeError("network down")

        with self.assertRaises(RuntimeError):
            self.generator.generate_response("q")


if __name__ == "__main__":
    unittest.main()
