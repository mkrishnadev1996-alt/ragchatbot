import unittest
from unittest.mock import MagicMock

from search_tools import CourseSearchTool, Tool, ToolManager
from vector_store import SearchResults


def make_results(docs, metas):
    return SearchResults(documents=docs, metadata=metas, distances=[0.1] * len(docs))


class FakeTool(Tool):
    """Minimal Tool implementation for exercising ToolManager"""

    def __init__(self, name="fake_tool"):
        self._name = name
        self.last_sources = []
        self.calls = []

    def get_tool_definition(self):
        return {"type": "function", "function": {"name": self._name, "parameters": {}}}

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        return f"ran with {kwargs}"


class TestCourseSearchTool(unittest.TestCase):

    def setUp(self):
        self.store = MagicMock()
        self.tool = CourseSearchTool(self.store)

    def test_definition_exposes_search_function_with_required_query(self):
        definition = self.tool.get_tool_definition()
        self.assertEqual(definition["type"], "function")
        self.assertEqual(definition["function"]["name"], "search_course_content")
        self.assertEqual(definition["function"]["parameters"]["required"], ["query"])

    def test_execute_passes_filters_to_store(self):
        self.store.search.return_value = make_results([], [])
        self.tool.execute(query="embeddings", course_name="MCP", lesson_number=2)
        self.store.search.assert_called_once_with(
            query="embeddings", course_name="MCP", lesson_number=2
        )

    def test_execute_returns_store_error_verbatim(self):
        self.store.search.return_value = SearchResults.empty("No course found matching 'zzz'")
        result = self.tool.execute(query="x", course_name="zzz")
        self.assertEqual(result, "No course found matching 'zzz'")

    def test_empty_results_without_filters(self):
        self.store.search.return_value = make_results([], [])
        result = self.tool.execute(query="x")
        self.assertEqual(result, "No relevant content found.")

    def test_empty_results_mention_course_and_lesson_filters(self):
        self.store.search.return_value = make_results([], [])
        result = self.tool.execute(query="x", course_name="Intro", lesson_number=3)
        self.assertEqual(result, "No relevant content found in course 'Intro' in lesson 3.")

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
            result,
            "[Intro to AI - Lesson 1]\nChunk about RAG.\n\n"
            "[Intro to AI - Lesson 2]\nChunk about tools.",
        )

    def test_result_without_lesson_number_has_course_only_header(self):
        self.store.search.return_value = make_results(
            ["Course-level text."], [{"course_title": "Plain Course", "lesson_number": None}]
        )
        result = self.tool.execute(query="x")
        self.assertEqual(result, "[Plain Course]\nCourse-level text.")
        self.assertEqual(self.tool.last_sources, ["Plain Course"])

    def test_last_sources_track_course_and_lesson(self):
        self.store.search.return_value = make_results(
            ["a", "b"],
            [
                {"course_title": "C1", "lesson_number": 4},
                {"course_title": "C2", "lesson_number": 1},
            ],
        )
        self.tool.execute(query="x")
        self.assertEqual(self.tool.last_sources, ["C1 - Lesson 4", "C2 - Lesson 1"])


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
        self.assertEqual(result, "ran with {'query': 'hi'}")
        self.assertEqual(tool.calls, [{"query": "hi"}])

    def test_execute_unknown_tool_returns_message(self):
        manager = ToolManager()
        self.assertEqual(manager.execute_tool("ghost"), "Tool 'ghost' not found")

    def test_get_last_sources_returns_first_nonempty(self):
        manager = ToolManager()
        empty = FakeTool("a")
        full = FakeTool("b")
        full.last_sources = ["Course - Lesson 1"]
        manager.register_tool(empty)
        manager.register_tool(full)
        self.assertEqual(manager.get_last_sources(), ["Course - Lesson 1"])

    def test_get_last_sources_empty_when_no_tool_tracks_sources(self):
        manager = ToolManager()
        manager.register_tool(FakeTool("a"))
        self.assertEqual(manager.get_last_sources(), [])

    def test_reset_sources_clears_all_tools(self):
        manager = ToolManager()
        tool = FakeTool("a")
        tool.last_sources = ["X"]
        manager.register_tool(tool)
        manager.reset_sources()
        self.assertEqual(tool.last_sources, [])
        self.assertEqual(manager.get_last_sources(), [])


if __name__ == "__main__":
    unittest.main()
