import unittest

from session_manager import SessionManager


class TestSessionManager(unittest.TestCase):
    def test_create_session_returns_unique_ids(self):
        manager = SessionManager(max_history=2)
        first = manager.create_session()
        second = manager.create_session()
        self.assertNotEqual(first, second)
        self.assertIn(first, manager.sessions)
        self.assertIn(second, manager.sessions)

    def test_new_session_has_no_history(self):
        manager = SessionManager(max_history=2)
        session_id = manager.create_session()
        self.assertIsNone(manager.get_conversation_history(session_id))

    def test_add_exchange_records_user_then_assistant(self):
        manager = SessionManager(max_history=2)
        session_id = manager.create_session()
        manager.add_exchange(session_id, "What is RAG?", "Retrieval-augmented generation.")

        history = manager.get_conversation_history(session_id)
        self.assertEqual(
            history,
            [
                {"role": "user", "content": "What is RAG?"},
                {"role": "assistant", "content": "Retrieval-augmented generation."},
            ],
        )

    def test_history_is_trimmed_to_max_history_exchanges(self):
        # max_history=2 keeps the last 2 * 2 = 4 messages
        manager = SessionManager(max_history=2)
        session_id = manager.create_session()
        for i in range(5):
            manager.add_exchange(session_id, f"q{i}", f"a{i}")

        history = manager.get_conversation_history(session_id)
        assert history is not None
        self.assertEqual(len(history), 4)
        self.assertEqual(history[0]["content"], "q3")
        self.assertEqual(history[-1]["content"], "a4")

    def test_add_message_to_unknown_session_creates_it(self):
        manager = SessionManager(max_history=2)
        manager.add_message("custom", "user", "hi")
        self.assertEqual(
            manager.get_conversation_history("custom"), [{"role": "user", "content": "hi"}]
        )

    def test_unknown_or_empty_session_id_returns_none(self):
        manager = SessionManager(max_history=2)
        self.assertIsNone(manager.get_conversation_history("missing"))
        self.assertIsNone(manager.get_conversation_history(None))
        self.assertIsNone(manager.get_conversation_history(""))

    def test_clear_session_removes_messages_but_keeps_session(self):
        manager = SessionManager(max_history=2)
        session_id = manager.create_session()
        manager.add_exchange(session_id, "q", "a")
        manager.clear_session(session_id)

        self.assertIn(session_id, manager.sessions)
        self.assertIsNone(manager.get_conversation_history(session_id))

    def test_clear_unknown_session_is_noop(self):
        manager = SessionManager(max_history=2)
        manager.clear_session("nope")
        self.assertNotIn("nope", manager.sessions)


if __name__ == "__main__":
    unittest.main()
