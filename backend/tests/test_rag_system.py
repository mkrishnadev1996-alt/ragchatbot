import contextlib
import io
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rag_system import RAGSystem


def make_config():
    return SimpleNamespace(
        CHUNK_SIZE=800,
        CHUNK_OVERLAP=100,
        CHROMA_PATH="unused",
        EMBEDDING_MODEL="unused",
        MAX_RESULTS=5,
        MAX_HISTORY=2,
        LLM_API_KEY="key",
        LLM_BASE_URL="http://llm.test",
        LLM_MODEL="test-model",
    )


class TestRAGSystem(unittest.TestCase):

    def setUp(self):
        # Replace the heavy components so no ChromaDB, embedding model or LLM is touched.
        vs_patcher = patch("rag_system.VectorStore")
        ai_patcher = patch("rag_system.AIGenerator")
        self.vector_store_cls = vs_patcher.start()
        self.ai_cls = ai_patcher.start()
        self.addCleanup(vs_patcher.stop)
        self.addCleanup(ai_patcher.stop)

        self.rag = RAGSystem(make_config())
        self.vector_store = self.vector_store_cls.return_value
        self.ai = self.ai_cls.return_value

    def test_wires_search_tool_into_tool_manager(self):
        definitions = self.rag.tool_manager.get_tool_definitions()
        names = [d["function"]["name"] for d in definitions]
        self.assertEqual(names, ["search_course_content"])

    def test_query_without_session_returns_answer_and_sources(self):
        def fake_generate(**kwargs):
            # Simulate the search tool recording sources during the AI call
            self.rag.search_tool.last_sources = ["Course A - Lesson 1"]
            return "The answer"

        self.ai.generate_response.side_effect = fake_generate

        answer, sources = self.rag.query("What is MCP?")

        self.assertEqual(answer, "The answer")
        self.assertEqual(sources, ["Course A - Lesson 1"])
        kwargs = self.ai.generate_response.call_args.kwargs
        self.assertIn("What is MCP?", kwargs["query"])
        self.assertIsNone(kwargs["conversation_history"])

    def test_sources_are_reset_after_query(self):
        def fake_generate(**kwargs):
            self.rag.search_tool.last_sources = ["X"]
            return "ans"

        self.ai.generate_response.side_effect = fake_generate
        self.rag.query("q1")

        self.assertEqual(self.rag.tool_manager.get_last_sources(), [])

        self.ai.generate_response.side_effect = None
        self.ai.generate_response.return_value = "ans2"
        _, sources = self.rag.query("q2")
        self.assertEqual(sources, [])

    def test_query_with_session_passes_history_and_records_exchange(self):
        session_id = self.rag.session_manager.create_session()
        self.rag.session_manager.add_exchange(session_id, "earlier", "earlier answer")
        self.ai.generate_response.return_value = "new answer"

        self.rag.query("follow up", session_id)

        kwargs = self.ai.generate_response.call_args.kwargs
        self.assertEqual(kwargs["conversation_history"], [
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "earlier answer"},
        ])

        history = self.rag.session_manager.get_conversation_history(session_id)
        assert history is not None
        self.assertEqual(history[-2:], [
            {"role": "user", "content": "follow up"},
            {"role": "assistant", "content": "new answer"},
        ])

    def test_query_passes_tool_definitions_and_manager(self):
        self.ai.generate_response.return_value = "ok"
        self.rag.query("q")
        kwargs = self.ai.generate_response.call_args.kwargs
        self.assertIs(kwargs["tool_manager"], self.rag.tool_manager)
        self.assertEqual(kwargs["tools"], self.rag.tool_manager.get_tool_definitions())

    def test_add_course_document_failure_returns_none(self):
        with contextlib.redirect_stdout(io.StringIO()):
            course, count = self.rag.add_course_document("does-not-exist.txt")
        self.assertIsNone(course)
        self.assertEqual(count, 0)

    def test_add_course_folder_missing_folder_returns_zero(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.rag.add_course_folder("no-such-folder"), (0, 0))

    def test_add_course_folder_adds_only_new_supported_courses(self):
        self.vector_store.get_existing_course_titles.return_value = ["Already Here"]

        new_doc = (
            "Course Title: Brand New\n"
            "Course Link: https://x.test\n"
            "Course Instructor: Ann\n\n"
            "Lesson 0: Start\nSome lesson text for the new course.\n"
        )
        existing_doc = (
            "Course Title: Already Here\n"
            "Course Link: https://y.test\n"
            "Course Instructor: Bob\n\n"
            "Lesson 0: Start\nText for the existing course.\n"
        )

        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(folder, "new.txt"), "w", encoding="utf-8") as f:
                f.write(new_doc)
            with open(os.path.join(folder, "old.txt"), "w", encoding="utf-8") as f:
                f.write(existing_doc)
            # Unsupported extension should be ignored entirely
            with open(os.path.join(folder, "notes.md"), "w", encoding="utf-8") as f:
                f.write(new_doc)

            with contextlib.redirect_stdout(io.StringIO()):
                courses, chunks = self.rag.add_course_folder(folder)

        self.assertEqual(courses, 1)
        self.assertGreater(chunks, 0)
        self.vector_store.add_course_metadata.assert_called_once()
        added_course = self.vector_store.add_course_metadata.call_args.args[0]
        self.assertEqual(added_course.title, "Brand New")
        self.vector_store.add_course_content.assert_called_once()

    def test_add_course_folder_clear_existing_resets_store(self):
        with tempfile.TemporaryDirectory() as folder:
            with contextlib.redirect_stdout(io.StringIO()):
                self.rag.add_course_folder(folder, clear_existing=True)
        self.vector_store.clear_all_data.assert_called_once()

    def test_get_course_analytics(self):
        self.vector_store.get_course_count.return_value = 2
        self.vector_store.get_existing_course_titles.return_value = ["A", "B"]
        self.assertEqual(self.rag.get_course_analytics(),
                         {"total_courses": 2, "course_titles": ["A", "B"]})


if __name__ == "__main__":
    unittest.main()
