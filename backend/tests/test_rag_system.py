import contextlib
import io
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rag_system import RAGSystem

SMALL_COURSE = "Course Title: C\nCourse Link: x\nCourse Instructor: Y\n\nLesson 0: A\nText here.\n"


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
        LLM_MAX_TOKENS=1600,
        LLM_TEMPERATURE=0,
        LLM_MAX_RETRIES=3,
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

    def test_guard_disabled_without_key(self):
        self.assertIsNone(self.rag.input_guard)

    def test_blocked_query_skips_llm_and_history(self):
        cfg = make_config()
        cfg.GUARD_API_KEY, cfg.GUARD_URL, cfg.GUARD_MODEL = "k", "http://g.test", "g"
        cfg.GUARD_THRESHOLD = 0.5
        cfg.GUARD_TIMEOUT = 10.0
        with patch("rag_system.InputGuard") as guard_cls:
            guard_cls.BLOCKED_MESSAGE = "Sorry, we cannot answer that."
            guard_cls.return_value.is_safe.return_value = False
            rag = RAGSystem(cfg)
            session_id = rag.session_manager.create_session()

            result = rag.query("bad request", session_id)

        self.assertEqual(result.answer, "Sorry, we cannot answer that.")
        self.assertEqual(result.sources, [])
        self.assertTrue(result.blocked)
        self.ai.generate_response.assert_not_called()
        self.assertEqual(rag.session_manager.get_conversation_history(session_id) or "", "")

    def test_safe_query_continues_to_llm(self):
        cfg = make_config()
        cfg.GUARD_API_KEY, cfg.GUARD_URL, cfg.GUARD_MODEL = "k", "http://g.test", "g"
        cfg.GUARD_THRESHOLD = 0.5
        cfg.GUARD_TIMEOUT = 10.0
        with patch("rag_system.InputGuard") as guard_cls:
            guard_cls.return_value.is_safe.return_value = True
            rag = RAGSystem(cfg)
            self.ai.generate_response.return_value = "The answer"

            result = rag.query("What is MCP?")

        self.assertEqual(result.answer, "The answer")
        self.assertFalse(result.blocked)

    def test_wires_search_tool_into_tool_manager(self):
        definitions = self.rag.tool_manager.get_tool_definitions()
        names = [d["function"]["name"] for d in definitions]
        self.assertEqual(names, ["search_course_content"])

    def test_guard_receives_conversation_history(self):
        cfg = make_config()
        cfg.GUARD_API_KEY, cfg.GUARD_URL, cfg.GUARD_MODEL = "k", "http://g.test", "g"
        cfg.GUARD_THRESHOLD = 0.5
        cfg.GUARD_TIMEOUT = 10.0
        with patch("rag_system.InputGuard") as guard_cls:
            guard_cls.return_value.is_safe.return_value = True
            rag = RAGSystem(cfg)
            self.ai.generate_response.return_value = "a"
            session_id = rag.session_manager.create_session()
            rag.session_manager.add_exchange(session_id, "about MCP", "MCP is...")

            rag.query("and lesson 3?", session_id)

        history = guard_cls.return_value.is_safe.call_args.args[1]
        self.assertEqual(history[0], {"role": "user", "content": "about MCP"})

    def test_query_without_session_returns_answer_and_sources(self):
        def fake_generate(**kwargs):
            # The search tool's sources are collected into the per-request list
            kwargs["sources_out"].append("Course A - Lesson 1")
            return "The answer"

        self.ai.generate_response.side_effect = fake_generate

        answer, sources, _ = self.rag.query("What is MCP?")

        self.assertEqual(answer, "The answer")
        self.assertEqual(sources, ["Course A - Lesson 1"])
        kwargs = self.ai.generate_response.call_args.kwargs
        self.assertIn("What is MCP?", kwargs["query"])
        self.assertIsNone(kwargs["conversation_history"])

    def test_each_query_gets_its_own_sources(self):
        def fake_generate(**kwargs):
            kwargs["sources_out"].append("X")
            return "ans"

        self.ai.generate_response.side_effect = fake_generate
        self.rag.query("q1")

        self.ai.generate_response.side_effect = None
        self.ai.generate_response.return_value = "ans2"
        result = self.rag.query("q2")
        self.assertEqual(result.sources, [])

    def test_failed_llm_call_skips_history(self):
        self.ai.generate_response.side_effect = RuntimeError("provider down")
        session_id = self.rag.session_manager.create_session()

        with self.assertRaises(RuntimeError):
            self.rag.query("q1", session_id)

        self.assertEqual(self.rag.session_manager.get_conversation_history(session_id), None)

    def test_query_with_session_passes_history_and_records_exchange(self):
        session_id = self.rag.session_manager.create_session()
        self.rag.session_manager.add_exchange(session_id, "earlier", "earlier answer")
        self.ai.generate_response.return_value = "new answer"

        self.rag.query("follow up", session_id)

        kwargs = self.ai.generate_response.call_args.kwargs
        self.assertEqual(
            kwargs["conversation_history"],
            [
                {"role": "user", "content": "earlier"},
                {"role": "assistant", "content": "earlier answer"},
            ],
        )

        history = self.rag.session_manager.get_conversation_history(session_id)
        assert history is not None
        self.assertEqual(
            history[-2:],
            [
                {"role": "user", "content": "follow up"},
                {"role": "assistant", "content": "new answer"},
            ],
        )

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

    def _write_course(self, folder, name, text):
        with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
            f.write(text)

    def test_course_without_chunks_is_not_added_to_the_catalog(self):
        self.vector_store.get_existing_course_titles.return_value = []
        with tempfile.TemporaryDirectory() as folder:
            self._write_course(
                folder, "empty.txt", "Course Title: Empty\nCourse Link: x\nCourse Instructor: Y\n"
            )
            courses, chunks = self.rag.add_course_folder(folder)

        self.assertEqual((courses, chunks), (0, 0))
        self.vector_store.add_course_metadata.assert_not_called()

    def test_content_is_written_before_the_catalog_entry(self):
        self.vector_store.get_existing_course_titles.return_value = []
        order = []
        self.vector_store.add_course_content.side_effect = lambda *_: order.append("content")
        self.vector_store.add_course_metadata.side_effect = lambda *_: order.append("metadata")
        with tempfile.TemporaryDirectory() as folder:
            self._write_course(
                folder,
                "c.txt",
                SMALL_COURSE,
            )
            self.rag.add_course_folder(folder)

        self.assertEqual(order, ["content", "metadata"])

    def test_failed_ingestion_rolls_back_content_and_leaves_no_catalog_entry(self):
        self.vector_store.get_existing_course_titles.return_value = []
        self.vector_store.add_course_metadata.side_effect = RuntimeError("disk full")
        with tempfile.TemporaryDirectory() as folder:
            self._write_course(
                folder,
                "c.txt",
                SMALL_COURSE,
            )
            courses, _ = self.rag.add_course_folder(folder)

        self.assertEqual(courses, 0)
        self.vector_store.delete_course_content.assert_called_once_with("C")

    def test_add_course_folder_clear_existing_resets_store(self):
        with tempfile.TemporaryDirectory() as folder:
            with contextlib.redirect_stdout(io.StringIO()):
                self.rag.add_course_folder(folder, clear_existing=True)
        self.vector_store.clear_all_data.assert_called_once()

    def test_get_course_analytics(self):
        self.vector_store.get_course_count.return_value = 2
        self.vector_store.get_existing_course_titles.return_value = ["A", "B"]
        self.assertEqual(
            self.rag.get_course_analytics(), {"total_courses": 2, "course_titles": ["A", "B"]}
        )


if __name__ == "__main__":
    unittest.main()
