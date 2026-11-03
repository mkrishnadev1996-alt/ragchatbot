import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ai_generator import AIGenerator


def make_response(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def make_tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class TestAIGenerator(unittest.TestCase):

    def setUp(self):
        openai_patcher = patch("ai_generator.OpenAI")
        openai_cls = openai_patcher.start()
        self.addCleanup(openai_patcher.stop)

        self.client = openai_cls.return_value
        self.create = self.client.chat.completions.create
        self.generator = AIGenerator(api_key="key", base_url="http://llm.test", model="test-model")

    def test_client_is_built_from_arguments(self):
        with patch("ai_generator.OpenAI") as openai_cls:
            AIGenerator(api_key="k", base_url="http://u", model="m")
            openai_cls.assert_called_once_with(api_key="k", base_url="http://u")

    def test_base_params_use_model_and_limits(self):
        self.assertEqual(self.generator.base_params["model"], "test-model")
        self.assertEqual(self.generator.base_params["temperature"], 0)
        self.assertEqual(self.generator.base_params["max_tokens"], 800)

    def test_direct_answer_without_tools(self):
        self.create.return_value = make_response(content="Hello there")

        result = self.generator.generate_response("hi")

        self.assertEqual(result, "Hello there")
        self.assertEqual(self.create.call_count, 1)
        params = self.create.call_args.kwargs
        self.assertNotIn("tools", params)
        self.assertEqual(params["messages"][0]["role"], "system")
        self.assertEqual(params["messages"][-1], {"role": "user", "content": "hi"})

    def test_none_content_returns_empty_string(self):
        self.create.return_value = make_response(content=None)
        self.assertEqual(self.generator.generate_response("hi"), "")

    def test_tools_are_attached_with_auto_choice(self):
        self.create.return_value = make_response(content="ok")
        tools = [{"type": "function", "function": {"name": "t"}}]

        self.generator.generate_response("q", tools=tools)

        params = self.create.call_args.kwargs
        self.assertEqual(params["tools"], tools)
        self.assertEqual(params["tool_choice"], "auto")

    def test_conversation_history_sits_between_system_and_query(self):
        self.create.return_value = make_response(content="ok")
        history = [
            {"role": "user", "content": "earlier q"},
            {"role": "assistant", "content": "earlier a"},
        ]

        self.generator.generate_response("now", conversation_history=history)

        messages = self.create.call_args.kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1:3], history)
        self.assertEqual(messages[-1], {"role": "user", "content": "now"})

    def test_system_prompt_is_first_message(self):
        self.create.return_value = make_response(content="ok")
        self.generator.generate_response("q")
        messages = self.create.call_args.kwargs["messages"]
        self.assertEqual(messages[0]["content"], AIGenerator.SYSTEM_PROMPT)

    def test_tool_call_runs_tool_and_makes_follow_up_call_without_tools(self):
        tool_call = make_tool_call("call_1", "search_course_content", '{"query": "rag"}')
        self.create.side_effect = [
            make_response(content=None, tool_calls=[tool_call]),
            make_response(content="Final answer"),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "SEARCH RESULT"

        result = self.generator.generate_response(
            "q", tools=[{"type": "function", "function": {"name": "search_course_content"}}],
            tool_manager=tool_manager,
        )

        self.assertEqual(result, "Final answer")
        tool_manager.execute_tool.assert_called_once_with("search_course_content", query="rag")
        self.assertEqual(self.create.call_count, 2)

        follow_up = self.create.call_args_list[1].kwargs
        self.assertNotIn("tools", follow_up)
        roles = [m["role"] for m in follow_up["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool"])

        assistant_msg = follow_up["messages"][2]
        self.assertEqual(assistant_msg["tool_calls"][0]["id"], "call_1")
        self.assertEqual(
            assistant_msg["tool_calls"][0]["function"]["name"], "search_course_content"
        )

        tool_msg = follow_up["messages"][3]
        self.assertEqual(tool_msg["tool_call_id"], "call_1")
        self.assertEqual(tool_msg["content"], "SEARCH RESULT")

    def test_multiple_tool_calls_each_get_a_tool_message(self):
        calls = [
            make_tool_call("a", "search_course_content", '{"query": "one"}'),
            make_tool_call("b", "search_course_content", '{"query": "two"}'),
        ]
        self.create.side_effect = [
            make_response(content=None, tool_calls=calls),
            make_response(content="done"),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = ["R1", "R2"]

        self.generator.generate_response("q", tools=[{}], tool_manager=tool_manager)

        self.assertEqual(tool_manager.execute_tool.call_count, 2)
        messages = self.create.call_args_list[1].kwargs["messages"]
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        self.assertEqual([(m["tool_call_id"], m["content"]) for m in tool_msgs],
                         [("a", "R1"), ("b", "R2")])

    def test_empty_tool_arguments_are_treated_as_no_args(self):
        self.create.side_effect = [
            make_response(content=None, tool_calls=[make_tool_call("x", "noop", "")]),
            make_response(content="ok"),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "r"

        self.generator.generate_response("q", tools=[{}], tool_manager=tool_manager)

        tool_manager.execute_tool.assert_called_once_with("noop")

    def test_tool_calls_without_manager_return_message_content(self):
        self.create.return_value = make_response(
            content="partial", tool_calls=[make_tool_call("x", "search_course_content", "{}")]
        )
        result = self.generator.generate_response("q", tools=[{}], tool_manager=None)
        self.assertEqual(result, "partial")
        self.assertEqual(self.create.call_count, 1)


if __name__ == "__main__":
    unittest.main()
