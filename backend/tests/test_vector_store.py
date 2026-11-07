import json
import unittest
from unittest.mock import MagicMock, patch

from models import Course, CourseChunk, Lesson
from vector_store import SearchResults, VectorStore


class TestSearchResults(unittest.TestCase):
    def test_from_chroma_unpacks_first_query_row(self):
        results = SearchResults.from_chroma(
            {
                "documents": [["doc1", "doc2"]],
                "metadatas": [[{"course_title": "A"}, {"course_title": "B"}]],
                "distances": [[0.1, 0.2]],
            }
        )
        self.assertEqual(results.documents, ["doc1", "doc2"])
        self.assertEqual(results.metadata, [{"course_title": "A"}, {"course_title": "B"}])
        self.assertEqual(results.distances, [0.1, 0.2])
        self.assertIsNone(results.error)
        self.assertFalse(results.is_empty())

    def test_from_chroma_handles_empty_results(self):
        results = SearchResults.from_chroma({"documents": [], "metadatas": [], "distances": []})
        self.assertTrue(results.is_empty())

    def test_empty_carries_error(self):
        results = SearchResults.empty("boom")
        self.assertTrue(results.is_empty())
        self.assertEqual(results.error, "boom")


class TestVectorStore(unittest.TestCase):
    def setUp(self):
        # Patch the whole chromadb module so no database is created and no
        # embedding model is downloaded.
        patcher = patch("vector_store.chromadb")
        self.chromadb = patcher.start()
        self.addCleanup(patcher.stop)
        embedder = patch("vector_store.SentenceTransformerEmbeddingFunction")
        self.embedder = embedder.start()
        self.addCleanup(embedder.stop)

        self.client = self.chromadb.PersistentClient.return_value
        self.catalog = MagicMock(name="course_catalog")
        self.content = MagicMock(name="course_content")
        self.client.get_or_create_collection.side_effect = lambda name, **kwargs: (
            self.catalog if name == "course_catalog" else self.content
        )

        self.store = VectorStore("unused-path", "fake-model", max_results=5)

    def test_creates_both_collections(self):
        self.assertIs(self.store.course_catalog, self.catalog)
        self.assertIs(self.store.course_content, self.content)

    def test_search_without_filters(self):
        self.content.query.return_value = {
            "documents": [["hit"]],
            "metadatas": [[{"course_title": "A", "lesson_number": 1}]],
            "distances": [[0.05]],
        }
        results = self.store.search("anything")

        self.content.query.assert_called_once_with(
            query_texts=["anything"], n_results=5, where=None
        )
        self.assertEqual(results.documents, ["hit"])
        self.catalog.query.assert_not_called()

    def test_search_resolves_course_name_and_filters_by_title(self):
        self.catalog.query.return_value = {
            "documents": [["MCP Course"]],
            "metadatas": [[{"title": "Intro to MCP"}]],
        }
        self.content.query.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        results = self.store.search("q", course_name="MCP")

        self.catalog.query.assert_called_once_with(query_texts=["MCP"], n_results=1)
        self.content.query.assert_called_once_with(
            query_texts=["q"], n_results=5, where={"course_title": "Intro to MCP"}
        )
        self.assertTrue(results.is_empty())

    def test_search_with_lesson_only_filter(self):
        self.content.query.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        self.store.search("q", lesson_number=2)
        self.content.query.assert_called_once_with(
            query_texts=["q"], n_results=5, where={"lesson_number": 2}
        )

    def test_search_with_course_and_lesson_uses_and_filter(self):
        self.catalog.query.return_value = {
            "documents": [["x"]],
            "metadatas": [[{"title": "Course X"}]],
        }
        self.content.query.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        self.store.search("q", course_name="X", lesson_number=3)

        self.content.query.assert_called_once_with(
            query_texts=["q"],
            n_results=5,
            where={"$and": [{"course_title": "Course X"}, {"lesson_number": 3}]},
        )

    def test_unresolvable_course_name_returns_error(self):
        self.catalog.query.return_value = {"documents": [[]], "metadatas": [[]]}
        results = self.store.search("q", course_name="nothing")

        self.assertEqual(results.error, "No course found matching 'nothing'")
        self.content.query.assert_not_called()

    def test_custom_limit_overrides_max_results(self):
        self.content.query.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        self.store.search("q", limit=2)
        self.assertEqual(self.content.query.call_args.kwargs["n_results"], 2)

    def test_chroma_exception_becomes_search_error(self):
        self.content.query.side_effect = RuntimeError("db down")
        results = self.store.search("q")
        self.assertEqual(results.error, "Search error: db down")

    def test_add_course_metadata_serializes_lessons(self):
        course = Course(
            title="Course A",
            course_link="https://a.test",
            instructor="Ann",
            lessons=[Lesson(lesson_number=0, title="Intro", lesson_link="https://a.test/0")],
        )
        self.store.add_course_metadata(course)

        kwargs = self.catalog.add.call_args.kwargs
        self.assertEqual(kwargs["ids"], ["Course A"])
        meta = kwargs["metadatas"][0]
        self.assertEqual(meta["title"], "Course A")
        self.assertEqual(meta["lesson_count"], 1)
        self.assertEqual(
            json.loads(meta["lessons_json"]),
            [{"lesson_number": 0, "lesson_title": "Intro", "lesson_link": "https://a.test/0"}],
        )

    def test_add_course_content_builds_ids_and_metadata(self):
        chunks = [
            CourseChunk(content="one", course_title="My Course", lesson_number=1, chunk_index=0),
            CourseChunk(content="two", course_title="My Course", lesson_number=1, chunk_index=1),
        ]
        self.store.add_course_content(chunks)

        kwargs = self.content.add.call_args.kwargs
        self.assertEqual(kwargs["documents"], ["one", "two"])
        self.assertEqual(len(set(kwargs["ids"])), 2)
        self.assertTrue(all(i.endswith(("_0", "_1")) for i in kwargs["ids"]))
        self.assertEqual(
            kwargs["metadatas"][0],
            {"course_title": "My Course", "lesson_number": 1, "chunk_index": 0},
        )

    def test_chunk_ids_do_not_collide_for_similar_titles(self):
        def ids_for(title):
            chunk = CourseChunk(content="x", course_title=title, chunk_index=0)
            self.store.add_course_content([chunk])
            return self.content.add.call_args.kwargs["ids"]

        self.assertNotEqual(ids_for("A B"), ids_for("A_B"))

    def test_delete_course_content_filters_by_title(self):
        self.store.delete_course_content("My Course")
        self.content.delete.assert_called_once_with(where={"course_title": "My Course"})

    def test_add_empty_chunks_is_noop(self):
        self.store.add_course_content([])
        self.content.add.assert_not_called()

    def test_get_existing_course_titles_returns_ids(self):
        self.catalog.get.return_value = {"ids": ["A", "B"]}
        self.assertEqual(self.store.get_existing_course_titles(), ["A", "B"])

    def test_get_course_count(self):
        self.catalog.get.return_value = {"ids": ["A", "B", "C"]}
        self.assertEqual(self.store.get_course_count(), 3)

    def test_catalog_read_failure_is_raised_not_reported_as_empty(self):
        self.catalog.get.side_effect = RuntimeError("nope")
        with self.assertRaises(RuntimeError):
            self.store.get_existing_course_titles()
        with self.assertRaises(RuntimeError):
            self.store.get_course_count()

    def test_catalog_reads_do_not_fetch_documents(self):
        self.catalog.get.return_value = {"ids": ["A"]}
        self.store.get_existing_course_titles()
        self.catalog.get.assert_called_once_with(include=[])

    def _store_with_threshold(self, threshold):
        return VectorStore("unused-path", "fake-model", max_course_distance=threshold)

    def test_course_name_beyond_distance_threshold_is_not_found(self):
        self.catalog.query.return_value = {
            "documents": [["Some Course"]],
            "metadatas": [[{"title": "Some Course"}]],
            "distances": [[1.8]],
        }
        results = self._store_with_threshold(1.6).search("q", course_name="banana")
        self.assertEqual(results.error, "No course found matching 'banana'")
        self.content.query.assert_not_called()

    def test_course_name_within_distance_threshold_resolves(self):
        self.catalog.query.return_value = {
            "documents": [["Some Course"]],
            "metadatas": [[{"title": "Some Course"}]],
            "distances": [[0.9]],
        }
        self.content.query.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        results = self._store_with_threshold(1.6).search("q", course_name="some")
        self.assertIsNone(results.error)
        self.content.query.assert_called_once()

    def test_get_all_courses_metadata_parses_lessons_json(self):
        self.catalog.get.return_value = {
            "metadatas": [
                {
                    "title": "A",
                    "lessons_json": json.dumps(
                        [{"lesson_number": 1, "lesson_title": "T", "lesson_link": None}]
                    ),
                }
            ]
        }
        meta = self.store.get_all_courses_metadata()
        self.assertEqual(meta[0]["lessons"][0]["lesson_number"], 1)
        self.assertNotIn("lessons_json", meta[0])

    def test_get_course_link(self):
        self.catalog.get.return_value = {"metadatas": [{"course_link": "https://a.test"}]}
        self.assertEqual(self.store.get_course_link("A"), "https://a.test")
        self.catalog.get.assert_called_with(ids=["A"])

    def test_get_course_link_missing_course_returns_none(self):
        self.catalog.get.return_value = {"metadatas": []}
        self.assertIsNone(self.store.get_course_link("missing"))

    def test_get_lesson_link_finds_matching_lesson(self):
        self.catalog.get.return_value = {
            "metadatas": [
                {
                    "lessons_json": json.dumps(
                        [
                            {"lesson_number": 1, "lesson_link": "https://a.test/1"},
                            {"lesson_number": 2, "lesson_link": "https://a.test/2"},
                        ]
                    )
                }
            ]
        }
        self.assertEqual(self.store.get_lesson_link("A", 2), "https://a.test/2")

    def test_get_lesson_link_unknown_lesson_returns_none(self):
        self.catalog.get.return_value = {
            "metadatas": [{"lessons_json": json.dumps([{"lesson_number": 1, "lesson_link": "x"}])}]
        }
        self.assertIsNone(self.store.get_lesson_link("A", 9))


if __name__ == "__main__":
    unittest.main()
