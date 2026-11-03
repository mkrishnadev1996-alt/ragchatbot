import os
import tempfile
import unittest

from utils.file_utils import read_text_file
from utils.http_utils import parse_retry_after
from utils.text_utils import chunk_text


class TestReadTextFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _write_bytes(self, data: bytes) -> str:
        path = os.path.join(self.tmp.name, "doc.txt")
        with open(path, "wb") as f:
            f.write(data)
        return path

    def test_reads_utf8(self):
        path = self._write_bytes("café".encode())
        self.assertEqual(read_text_file(path), "café")

    def test_invalid_utf8_bytes_are_ignored_instead_of_raising(self):
        path = self._write_bytes(b"ok \xff\xfe end")
        self.assertEqual(read_text_file(path), "ok  end")


class TestParseRetryAfter(unittest.TestCase):
    def test_numeric_header_is_parsed(self):
        self.assertEqual(parse_retry_after({"retry-after": "7"}), 7.0)

    def test_missing_header_returns_none(self):
        self.assertIsNone(parse_retry_after({}))
        self.assertIsNone(parse_retry_after(None))

    def test_http_date_is_not_parsed(self):
        self.assertIsNone(parse_retry_after({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}))


class TestChunkText(unittest.TestCase):
    def test_empty_text_gives_no_chunks(self):
        self.assertEqual(chunk_text("   ", 800, 100), [])

    def test_chunks_respect_size_limit(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(50))
        chunks = chunk_text(text, 200, 0)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 200 for c in chunks))

    def test_overlap_repeats_trailing_sentence(self):
        text = "First one here. Second one here. Third one here. Fourth one here."
        chunks = chunk_text(text, 40, 25)
        self.assertGreater(len(chunks), 1)
        last_sentence_of_first = chunks[0].split(". ")[-1].rstrip(".")
        self.assertIn(last_sentence_of_first, chunks[1])


if __name__ == "__main__":
    unittest.main()
