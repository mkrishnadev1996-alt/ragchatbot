import unittest
from unittest.mock import MagicMock

from models import Source
from search_tools import CourseSearchTool, Tool, ToolManager, ToolResult
from vector_store import SearchResults


def make_results(docs, metas):
    return SearchResults(documents=docs, metadata=metas, distances=[0.1] * len(docs))


class FakeTool(Tool):
    """Minimal Tool implementation for exercising ToolManager"""

    def __init__(self, name="fake_tool"):
        self._name = name
        self.calls = []

    def get_tool_definition(self):
        return {"type": "function", "function": {"name": self._name, "parameters": {}}}

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        return ToolResult(f"ran with {kwargs}")


class TestCourseSearchTool(unittest.TestCase):
    def setUp(self):
        self.store = MagicMock()
        self.store.get_lesson_link.return_value = None
        self.store.get_course_link.return_value = None
        self.tool = CourseSearchTool(self.store)

    def test_definition_exposes_search_function_with_required_query(self):
        definition = self.tool.get_tool_definition()
        self.assertEqual(definition["type"], "function")
        self.assertEqual(definition["function"]["name"], "search_course_content")
        self.assertEqual(definition["function"]["parameters"]["required"], ["query"])

    def test_optional_params_accept_null(self):
        # Models send explicit nulls for unused optional params; the schema must allow them
        # or the provider rejects the whole tool call (surfaced to users as a 500).
        props = self.tool.get_tool_definition()["function"]["parameters"]["properties"]
        self.assertIn("null", props["course_name"]["type"])
        self.assertIn("null", props["lesson_number"]["type"])

    def test_execute_treats_explicit_none_as_unset(self):
        self.store.search.return_value = make_results([], [])
        self.tool.execute(query="embeddings", course_name=None, lesson_number=None)
        self.store.search.assert_called_once_with(
            query="embeddings", course_name=None, lesson_number=None
        )

    def test_execute_passes_filters_to_store(self):
        self.store.search.return_value = make_results([], [])
        self.tool.execute(query="embeddings", course_name="MCP", lesson_number=2)
        self.store.search.assert_called_once_with(
            query="embeddings", course_name="MCP", lesson_number=2
        )

    def test_execute_returns_store_error_verbatim(self):
        self.store.search.return_value = SearchResults.empty("No course found matching 'zzz'")
        result = self.tool.execute(query="x", course_name="zzz")
        self.assertEqual(result.text, "No course found matching 'zzz'")

    def test_empty_results_without_filters(self):
        self.store.search.return_value = make_results([], [])
        result = self.tool.execute(query="x")
        self.assertEqual(result.text, "No relevant content found.")

    def test_empty_results_mention_course_and_lesson_filters(self):
        self.store.search.return_value = make_results([], [])
        result = self.tool.execute(query="x", course_name="Intro", lesson_number=3)
        self.assertEqual(result.text, "No relevant content found in course 'Intro' in lesson 3.")

    def test_empty_results_mention_lesson_zero(self):
        self.store.search.return_value = make_results([], [])
        result = self.tool.execute(query="x", lesson_number=0)
        self.assertEqual(result.text, "No relevant content found in lesson 0.")

    def test_results_are_formatted_with_headers(self):
        self.store.search.return_value = make_results(
            ["Chunk about RAG.", "Chunk about tools."],
            [
                {"course_title": "Intro to AI", "lesson_number": 1},
                {"course_title": "Intro to AI", "lesson_number": 2},
            ],
        )
        result = self.tool.execute(query="rag")
        self.assertEqual(
            result.text,
            "[Intro to AI - Lesson 1]\nChunk about RAG.\n\n"
            "[Intro to AI - Lesson 2]\nChunk about tools.",
        )

    def test_result_without_lesson_number_has_course_only_header(self):
        self.store.search.return_value = make_results(
            ["Course-level text."], [{"course_title": "Plain Course", "lesson_number": None}]
        )
        result = self.tool.execute(query="x")
        self.assertEqual(result.text, "[Plain Course]\nCourse-level text.")
        self.assertEqual(result.sources, [Source(label="Plain Course")])

    def test_sources_track_course_and_lesson(self):
        self.store.search.return_value = make_results(
            ["a", "b"],
            [
                {"course_title": "C1", "lesson_number": 4},
                {"course_title": "C2", "lesson_number": 1},
            ],
        )
        result = self.tool.execute(query="x")
        self.assertEqual([s.label for s in result.sources], ["C1 - Lesson 4", "C2 - Lesson 1"])

    def test_sources_belong_to_each_call_not_the_shared_tool(self):
        self.store.search.side_effect = [
            make_results(["a"], [{"course_title": "C1", "lesson_number": 1}]),
            make_results(["b"], [{"course_title": "C2", "lesson_number": 2}]),
        ]
        first = self.tool.execute(query="x")
        second = self.tool.execute(query="y")
        self.assertEqual([s.label for s in first.sources], ["C1 - Lesson 1"])
        self.assertEqual([s.label for s in second.sources], ["C2 - Lesson 2"])

    def test_sources_carry_the_lesson_link(self):
        self.store.get_lesson_link.return_value = "https://x.test/l4"
        self.store.search.return_value = make_results(
            ["a"], [{"course_title": "C1", "lesson_number": 4}]
        )
        result = self.tool.execute(query="x")
        self.assertEqual(result.sources, [Source(label="C1 - Lesson 4", url="https://x.test/l4")])
        self.store.get_lesson_link.assert_called_once_with("C1", 4)

    def test_source_falls_back_to_the_course_link(self):
        self.store.get_course_link.return_value = "https://x.test/course"
        self.store.search.return_value = make_results(
            ["a"], [{"course_title": "C1", "lesson_number": 4}]
        )
        self.assertEqual(self.tool.execute(query="x").sources[0].url, "https://x.test/course")

    def test_course_level_source_uses_the_course_link_only(self):
        self.store.get_course_link.return_value = "https://x.test/course"
        self.store.search.return_value = make_results(
            ["a"], [{"course_title": "C1", "lesson_number": None}]
        )
        self.assertEqual(self.tool.execute(query="x").sources[0].url, "https://x.test/course")
        self.store.get_lesson_link.assert_not_called()

    def test_repeated_sources_are_listed_once(self):
        self.store.search.return_value = make_results(
            ["a", "b"], [{"course_title": "C1", "lesson_number": 1}] * 2
        )
        self.assertEqual(len(self.tool.execute(query="x").sources), 1)


class TestToolManager(unittest.TestCase):
    def test_register_and_list_definitions(self):
        manager = ToolManager()
        manager.register_tool(FakeTool("alpha"))
        manager.register_tool(FakeTool("beta"))
        names = [d["function"]["name"] for d in manager.get_tool_definitions()]
        self.assertEqual(names, ["alpha", "beta"])

    def test_register_tool_without_name_raises(self):
        class Nameless(FakeTool):
            def get_tool_definition(self):
                return {"type": "function", "function": {}}

        with self.assertRaises(ValueError):
            ToolManager().register_tool(Nameless())

    def test_execute_dispatches_to_named_tool(self):
        manager = ToolManager()
        tool = FakeTool("alpha")
        manager.register_tool(tool)
        result = manager.execute_tool("alpha", query="hi")
        self.assertEqual(result.text, "ran with {'query': 'hi'}")
        self.assertEqual(tool.calls, [{"query": "hi"}])

    def test_execute_unknown_tool_returns_message(self):
        manager = ToolManager()
        self.assertEqual(manager.execute_tool("ghost").text, "Tool 'ghost' not found")


if __name__ == "__main__":
    unittest.main()
