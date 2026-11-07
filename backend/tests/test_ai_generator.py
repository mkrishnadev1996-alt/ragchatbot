import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import openai
from ai_generator import AIGenerator, LLMRateLimitError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from models import Source
from search_tools import ToolResult


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
            timeout=60,
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

    def test_tool_call_runs_tool_then_calls_the_model_again_with_tools_available(self):
        tool_call = make_tool_call("call_1", "search_course_content", {"query": "rag"})
        first = make_message(tool_calls=[tool_call])
        self.bound.invoke.side_effect = [first, make_message("Final answer")]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = ToolResult(
            "SEARCH RESULT", [Source(label="Course A - Lesson 1")]
        )

        sources: list[Source] = []
        result = self.generator.generate_response(
            "q",
            tools=[{"type": "function", "function": {"name": "search_course_content"}}],
            tool_manager=tool_manager,
            sources_out=sources,
        )

        self.assertEqual(result, "Final answer")
        self.assertEqual([x.label for x in sources], ["Course A - Lesson 1"])
        tool_manager.execute_tool.assert_called_once_with("search_course_content", query="rag")
        self.assertEqual(self.bound.invoke.call_count, 2)  # the follow-up may search again
        self.llm.invoke.assert_not_called()

        follow_up = self.bound.invoke.call_args.args[0]
        self.assertEqual(follow_up[1], HumanMessage(content="q"))
        self.assertIs(follow_up[2], first)
        self.assertEqual(follow_up[3], ToolMessage(content="SEARCH RESULT", tool_call_id="call_1"))

    def test_multiple_tool_calls_each_get_a_tool_message(self):
        calls = [
            make_tool_call("a", "search_course_content", {"query": "one"}),
            make_tool_call("b", "search_course_content", {"query": "two"}),
        ]
        self.bound.invoke.side_effect = [make_message(tool_calls=calls), make_message("done")]
        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = [ToolResult("R1"), ToolResult("R2")]

        self.generator.generate_response("q", tools=[{}], tool_manager=tool_manager)

        self.assertEqual(tool_manager.execute_tool.call_count, 2)
        follow_up = self.bound.invoke.call_args.args[0]
        outputs = [m for m in follow_up if isinstance(m, ToolMessage)]
        self.assertEqual([(o.tool_call_id, o.content) for o in outputs], [("a", "R1"), ("b", "R2")])

    def test_empty_tool_arguments_are_treated_as_no_args(self):
        self.bound.invoke.side_effect = [
            make_message(tool_calls=[make_tool_call("x", "noop", {})]),
            make_message("ok"),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = ToolResult("r")

        self.generator.generate_response("q", tools=[{}], tool_manager=tool_manager)

        tool_manager.execute_tool.assert_called_once_with("noop")

    def test_tool_exception_is_returned_to_the_model_not_raised(self):
        self.bound.invoke.side_effect = [
            make_message(tool_calls=[make_tool_call("t", "search_course_content", {"bogus": 1})]),
            make_message("recovered"),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = TypeError("unexpected keyword 'bogus'")

        result = self.generator.generate_response("q", tools=[{}], tool_manager=tool_manager)

        self.assertEqual(result, "recovered")
        follow_up = self.bound.invoke.call_args.args[0]
        tool_message = [m for m in follow_up if isinstance(m, ToolMessage)][0]
        self.assertIn("failed", tool_message.content)
        self.assertNotIn("bogus", tool_message.content)  # internals are not echoed

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
        self.bound.invoke.side_effect = [
            make_message(tool_calls=[make_tool_call("c", "search_course_content", {})]),
            make_rate_limit_error(retry_after="3"),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = ToolResult("r")

        with self.assertRaises(LLMRateLimitError) as ctx:
            self.generator.generate_response("q", tools=[{}], tool_manager=tool_manager)

        self.assertEqual(ctx.exception.retry_after, 3.0)

    def test_the_model_can_search_several_times_before_answering(self):
        first = make_message(tool_calls=[make_tool_call("a", "search", {"query": "one"})])
        second = make_message(tool_calls=[make_tool_call("b", "search", {"query": "two"})])
        self.bound.invoke.side_effect = [first, second, make_message("Combined answer")]
        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = [
            ToolResult("R1", [Source(label="A - Lesson 1"), Source(label="C - Lesson 3")]),
            # B is new; A repeats from the first round and must not be listed twice
            ToolResult("R2", [Source(label="B - Lesson 2"), Source(label="A - Lesson 1")]),
        ]
        sources: list[Source] = []

        result = self.generator.generate_response(
            "q", tools=[{}], tool_manager=tool_manager, sources_out=sources
        )

        self.assertEqual(result, "Combined answer")
        self.assertEqual(
            [c.kwargs for c in tool_manager.execute_tool.call_args_list],
            [{"query": "one"}, {"query": "two"}],
        )
        self.assertEqual(
            [s.label for s in sources], ["A - Lesson 1", "C - Lesson 3", "B - Lesson 2"]
        )
        self.assertEqual(self.bound.invoke.call_count, 3)
        self.llm.invoke.assert_not_called()
        last_call = self.bound.invoke.call_args.args[0]
        self.assertEqual(
            [m for m in last_call if isinstance(m, ToolMessage)],
            [
                ToolMessage(content="R1", tool_call_id="a"),
                ToolMessage(content="R2", tool_call_id="b"),
            ],
        )

    def test_round_cap_makes_the_last_call_tool_less_with_an_answer_now_notice(self):
        generator = AIGenerator(
            api_key="key", base_url="http://llm.test", model="m", max_tool_rounds=2
        )
        tool_msg = make_message(tool_calls=[make_tool_call("a", "search", {"query": "x"})])
        self.bound.invoke.side_effect = [tool_msg, tool_msg]
        self.llm.invoke.return_value = make_message("Best effort answer")
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = ToolResult("R")

        result = generator.generate_response("q", tools=[{}], tool_manager=tool_manager)

        self.assertEqual(result, "Best effort answer")
        self.assertEqual(tool_manager.execute_tool.call_count, 2)
        self.assertEqual(self.bound.invoke.call_count, 2)
        self.assertEqual(self.llm.invoke.call_count, 1)
        final_messages = self.llm.invoke.call_args.args[0]
        self.assertEqual(final_messages[-1], HumanMessage(content=AIGenerator.TOOL_LIMIT_NOTICE))

    def test_a_single_round_cap_matches_the_old_one_search_behaviour(self):
        generator = AIGenerator(
            api_key="key", base_url="http://llm.test", model="m", max_tool_rounds=1
        )
        self.bound.invoke.return_value = make_message(
            tool_calls=[make_tool_call("a", "search", {"query": "x"})]
        )
        self.llm.invoke.return_value = make_message("done")
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = ToolResult("R")

        generator.generate_response("q", tools=[{}], tool_manager=tool_manager)

        self.assertEqual(self.bound.invoke.call_count, 1)
        self.assertEqual(self.llm.invoke.call_count, 1)
        tool_manager.execute_tool.assert_called_once()

    def test_round_cap_is_at_least_one(self):
        generator = AIGenerator(
            api_key="key", base_url="http://llm.test", model="m", max_tool_rounds=0
        )
        self.assertEqual(generator.max_tool_rounds, 1)

    def test_default_round_cap_and_prompt_no_longer_limit_searching(self):
        self.assertEqual(self.generator.max_tool_rounds, 3)
        self.assertNotIn("One search per query", AIGenerator.SYSTEM_PROMPT)
        self.assertIn("Search again", AIGenerator.SYSTEM_PROMPT)

    def test_truncation_notice_comes_from_the_final_response(self):
        first = make_message(tool_calls=[make_tool_call("a", "search", {"query": "x"})])
        final = AIMessage(content="cut", response_metadata={"finish_reason": "length"})
        self.bound.invoke.side_effect = [first, final]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = ToolResult("R")

        result = self.generator.generate_response("q", tools=[{}], tool_manager=tool_manager)

        self.assertEqual(result, "cut" + AIGenerator.TRUNCATION_NOTICE)

    def test_other_errors_are_not_translated(self):
        self.llm.invoke.side_effect = RuntimeError("network down")

        with self.assertRaises(RuntimeError):
            self.generator.generate_response("q")


if __name__ == "__main__":
    unittest.main()
