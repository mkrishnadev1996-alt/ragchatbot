import os
import tempfile
import unittest

from document_processor import DocumentProcessor

COURSE_DOC = """Course Title: Test Course
Course Link: https://example.com/course
Course Instructor: Jane Doe

Lesson 0: Introduction
Lesson Link: https://example.com/lesson0
This is the intro. It has two sentences.

Lesson 1: Basics
Basics content lives here. Another sentence follows.
"""

NO_LESSON_DOC = """Course Title: Plain Course
Course Link: https://example.com/plain
Course Instructor: John Roe

This course has no lesson markers. It is one block of text.
"""


class TestChunkText(unittest.TestCase):
    def test_short_text_is_single_chunk(self):
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=0)
        chunks = processor.chunk_text("One sentence. Two sentence.")
        self.assertEqual(len(chunks), 1)

    def test_empty_text_produces_no_chunks(self):
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=0)
        self.assertEqual(processor.chunk_text("   "), [])

    def test_chunks_respect_chunk_size_and_keep_all_sentences(self):
        processor = DocumentProcessor(chunk_size=40, chunk_overlap=0)
        text = "Alpha is first. Beta is second. Gamma is third. Delta is fourth."
        chunks = processor.chunk_text(text)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 40)
        joined = " ".join(chunks)
        sentences = ["Alpha is first.", "Beta is second.", "Gamma is third.", "Delta is fourth."]
        for sentence in sentences:
            self.assertIn(sentence, joined)

    def test_overlap_repeats_last_sentence_in_next_chunk(self):
        processor = DocumentProcessor(chunk_size=40, chunk_overlap=20)
        text = "Alpha is first. Beta is second. Gamma is third. Delta is fourth."
        chunks = processor.chunk_text(text)

        self.assertIn("Beta is second.", chunks[0])
        self.assertTrue(chunks[1].startswith("Beta is second."))

    def test_no_overlap_moves_to_next_sentence(self):
        processor = DocumentProcessor(chunk_size=40, chunk_overlap=0)
        text = "Alpha is first. Beta is second. Gamma is third. Delta is fourth."
        chunks = processor.chunk_text(text)

        self.assertTrue(chunks[1].startswith("Gamma is third."))

    def test_oversized_sentence_still_emitted(self):
        processor = DocumentProcessor(chunk_size=10, chunk_overlap=0)
        long_sentence = "This single sentence is far longer than ten characters."
        chunks = processor.chunk_text(long_sentence)
        self.assertEqual(chunks, [long_sentence])


class TestProcessCourseDocument(unittest.TestCase):
    def setUp(self):
        self.processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write(self, name, content):
        path = os.path.join(self.tmpdir.name, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_parses_course_metadata(self):
        path = self._write("course.txt", COURSE_DOC)
        course, _ = self.processor.process_course_document(path)

        self.assertEqual(course.title, "Test Course")
        self.assertEqual(course.course_link, "https://example.com/course")
        self.assertEqual(course.instructor, "Jane Doe")

    def test_parses_lessons_and_lesson_links(self):
        path = self._write("course.txt", COURSE_DOC)
        course, _ = self.processor.process_course_document(path)

        self.assertEqual([lesson.lesson_number for lesson in course.lessons], [0, 1])
        self.assertEqual(course.lessons[0].title, "Introduction")
        self.assertEqual(course.lessons[0].lesson_link, "https://example.com/lesson0")
        self.assertIsNone(course.lessons[1].lesson_link)

    def test_lesson_link_line_is_not_included_in_content(self):
        path = self._write("course.txt", COURSE_DOC)
        _, chunks = self.processor.process_course_document(path)

        for chunk in chunks:
            self.assertNotIn("Lesson Link", chunk.content)
            self.assertNotIn("https://example.com/lesson0", chunk.content)

    def test_chunks_carry_course_and_lesson_metadata(self):
        path = self._write("course.txt", COURSE_DOC)
        _, chunks = self.processor.process_course_document(path)

        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertEqual(chunk.course_title, "Test Course")
            self.assertIn(chunk.lesson_number, [0, 1])

        lesson0 = [c for c in chunks if c.lesson_number == 0]
        lesson1 = [c for c in chunks if c.lesson_number == 1]
        self.assertIn("Lesson 0 content:", lesson0[0].content)
        self.assertIn("Lesson 1 content:", lesson1[0].content)

    def test_chunk_indexes_are_sequential(self):
        path = self._write("course.txt", COURSE_DOC)
        _, chunks = self.processor.process_course_document(path)
        self.assertEqual([c.chunk_index for c in chunks], list(range(len(chunks))))

    def test_document_without_lessons_becomes_unnumbered_chunks(self):
        path = self._write("plain.txt", NO_LESSON_DOC)
        course, chunks = self.processor.process_course_document(path)

        self.assertEqual(course.title, "Plain Course")
        self.assertEqual(course.lessons, [])
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertIsNone(chunk.lesson_number)
            self.assertEqual(chunk.course_title, "Plain Course")
        self.assertIn("one block of text", " ".join(c.content for c in chunks))

    def test_first_line_without_prefix_is_used_as_title(self):
        content = (
            "Raw Title Line\nCourse Link: https://x.test\nCourse Instructor: Ann\n\n"
            "Lesson 0: Only\nSome text here.\n"
        )
        path = self._write("raw.txt", content)
        course, _ = self.processor.process_course_document(path)
        self.assertEqual(course.title, "Raw Title Line")

    def test_missing_instructor_is_none(self):
        content = (
            "Course Title: No Instructor\nCourse Link: https://x.test\n\n"
            "Lesson 0: Only\nSome text here.\n"
        )
        path = self._write("noinst.txt", content)
        course, _ = self.processor.process_course_document(path)
        self.assertIsNone(course.instructor)


if __name__ == "__main__":
    unittest.main()
