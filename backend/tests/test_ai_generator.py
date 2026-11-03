import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ai_generator import AIGenerator


def make_response(text=None, function_calls=None):
    """Build a fake Responses API result: output items plus the output_text convenience"""
    output = list(function_calls or [])
    return SimpleNamespace(output=output, output_text=text)


def make_function_call(call_id, name, arguments):
    return SimpleNamespace(
        type="function_call",
        call_id=call_id,
        name=name,
        arguments=arguments,
    )


class TestAIGenerator(unittest.TestCase):
    def setUp(self):
        openai_patcher = patch("ai_generator.OpenAI")
        openai_cls = openai_patcher.start()
        self.addCleanup(openai_patcher.stop)

        self.client = openai_cls.return_value
        self.create = self.client.responses.create
        self.generator = AIGenerator(api_key="key", base_url="http://llm.test", model="test-model")

    def test_client_is_built_from_arguments(self):
        with patch("ai_generator.OpenAI") as openai_cls:
            AIGenerator(api_key="k", base_url="http://u", model="m")
            openai_cls.assert_called_once_with(api_key="k", base_url="http://u")

    def test_base_params_use_model_and_limits(self):
        self.assertEqual(self.generator.base_params["model"], "test-model")
        self.assertEqual(self.generator.base_params["temperature"], 0)
        self.assertEqual(self.generator.base_params["max_output_tokens"], 800)

    def test_direct_answer_without_tools(self):
        self.create.return_value = make_response(text="Hello there")

        result = self.generator.generate_response("hi")

        self.assertEqual(result, "Hello there")
        self.assertEqual(self.create.call_count, 1)
        params = self.create.call_args.kwargs
        self.assertNotIn("tools", params)
        self.assertEqual(params["instructions"], AIGenerator.SYSTEM_PROMPT)
        self.assertEqual(params["input"], [{"role": "user", "content": "hi"}])

    def test_none_text_returns_empty_string(self):
        self.create.return_value = make_response(text=None)
        self.assertEqual(self.generator.generate_response("hi"), "")

    def test_tools_are_attached_with_auto_choice_in_flat_shape(self):
        self.create.return_value = make_response(text="ok")
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

        self.generator.generate_response("q", tools=tools)

        params = self.create.call_args.kwargs
        self.assertEqual(
            params["tools"],
            [
                {
                    "type": "function",
                    "name": "t",
                    "description": "d",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        )
        self.assertEqual(params["tool_choice"], "auto")

    def test_flat_tool_definitions_pass_through_unchanged(self):
        self.create.return_value = make_response(text="ok")
        tools = [{"type": "function", "name": "t", "parameters": {}}]

        self.generator.generate_response("q", tools=tools)

        self.assertEqual(self.create.call_args.kwargs["tools"], tools)

    def test_conversation_history_sits_between_instructions_and_query(self):
        self.create.return_value = make_response(text="ok")
        history = [
            {"role": "user", "content": "earlier q"},
            {"role": "assistant", "content": "earlier a"},
        ]

        self.generator.generate_response("now", conversation_history=history)

        params = self.create.call_args.kwargs
        self.assertEqual(params["input"][:2], history)
        self.assertEqual(params["input"][-1], {"role": "user", "content": "now"})
        self.assertNotIn("system", [item.get("role") for item in params["input"]])

    def test_system_prompt_is_sent_as_instructions(self):
        self.create.return_value = make_response(text="ok")
        self.generator.generate_response("q")
        self.assertEqual(self.create.call_args.kwargs["instructions"], AIGenerator.SYSTEM_PROMPT)

    def test_tool_call_runs_tool_and_makes_follow_up_call_without_tools(self):
        function_call = make_function_call("call_1", "search_course_content", '{"query": "rag"}')
        self.create.side_effect = [
            make_response(function_calls=[function_call]),
            make_response(text="Final answer"),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "SEARCH RESULT"

        result = self.generator.generate_response(
            "q",
            tools=[{"type": "function", "function": {"name": "search_course_content"}}],
            tool_manager=tool_manager,
        )

        self.assertEqual(result, "Final answer")
        tool_manager.execute_tool.assert_called_once_with("search_course_content", query="rag")
        self.assertEqual(self.create.call_count, 2)

        follow_up = self.create.call_args_list[1].kwargs
        self.assertNotIn("tools", follow_up)
        input_items = follow_up["input"]
        self.assertEqual(input_items[0], {"role": "user", "content": "q"})
        self.assertIs(input_items[1], function_call)

        output_item = input_items[2]
        self.assertEqual(output_item["type"], "function_call_output")
        self.assertEqual(output_item["call_id"], "call_1")
        self.assertEqual(output_item["output"], "SEARCH RESULT")

    def test_multiple_tool_calls_each_get_a_function_call_output(self):
        calls = [
            make_function_call("a", "search_course_content", '{"query": "one"}'),
            make_function_call("b", "search_course_content", '{"query": "two"}'),
        ]
        self.create.side_effect = [
            make_response(function_calls=calls),
            make_response(text="done"),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = ["R1", "R2"]

        self.generator.generate_response("q", tools=[{}], tool_manager=tool_manager)

        self.assertEqual(tool_manager.execute_tool.call_count, 2)
        input_items = self.create.call_args_list[1].kwargs["input"]
        outputs = [
            i
            for i in input_items
            if isinstance(i, dict) and i.get("type") == "function_call_output"
        ]
        self.assertEqual([(o["call_id"], o["output"]) for o in outputs], [("a", "R1"), ("b", "R2")])

    def test_empty_tool_arguments_are_treated_as_no_args(self):
        self.create.side_effect = [
            make_response(function_calls=[make_function_call("x", "noop", "")]),
            make_response(text="ok"),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "r"

        self.generator.generate_response("q", tools=[{}], tool_manager=tool_manager)

        tool_manager.execute_tool.assert_called_once_with("noop")

    def test_function_calls_without_manager_return_output_text(self):
        self.create.return_value = make_response(
            text="partial",
            function_calls=[make_function_call("x", "search_course_content", "{}")],
        )
        result = self.generator.generate_response("q", tools=[{}], tool_manager=None)
        self.assertEqual(result, "partial")
        self.assertEqual(self.create.call_count, 1)


if __name__ == "__main__":
    unittest.main()
