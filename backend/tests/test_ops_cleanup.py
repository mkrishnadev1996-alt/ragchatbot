import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from ai_generator import AIGenerator
from feedback_store import FeedbackStore
from langchain_core.messages import AIMessage
from message_store import MessageStore
from user_store import UserStore
from utils.db import BUSY_TIMEOUT_SECONDS, connect


class TestTruncationNotice(unittest.TestCase):
    def setUp(self):
        patcher = patch("ai_generator.ChatOpenAI")
        self.llm = patcher.start().return_value
        self.addCleanup(patcher.stop)
        self.generator = AIGenerator(api_key="k", base_url="http://x", model="m")

    def test_length_finish_reason_adds_notice(self):
        self.llm.invoke.return_value = AIMessage(
            content="partial", response_metadata={"finish_reason": "length"}
        )
        out = self.generator.generate_response("q")
        self.assertTrue(out.startswith("partial"))
        self.assertIn("cut off", out)

    def test_normal_stop_is_unchanged(self):
        self.llm.invoke.return_value = AIMessage(
            content="done", response_metadata={"finish_reason": "stop"}
        )
        self.assertEqual(self.generator.generate_response("q"), "done")


class TestSqliteSetup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "t.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_connect_uses_wal_and_busy_timeout(self):
        conn = connect(self.path)
        try:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(mode, "wal")
        self.assertEqual(timeout, BUSY_TIMEOUT_SECONDS * 1000)

    def test_stores_share_one_feedback_schema_and_list_messages_works_alone(self):
        UserStore(self.path)
        messages = MessageStore(self.path)  # creates the feedback table itself
        messages.add_exchange(1, "s", "q", "a", [], "m1")
        FeedbackStore(self.path).set_rating("m1", 1, "up", "s")
        self.assertEqual(messages.list_messages(1)[-1]["rating"], "up")
        conn = sqlite3.connect(self.path)
        try:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
        finally:
            conn.close()
        self.assertIn("feedback", tables)


if __name__ == "__main__":
    unittest.main()
