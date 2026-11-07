import os
import tempfile
import unittest

from user_store import UserExistsError, UserStore


class TestUserStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = UserStore(os.path.join(self.tmp.name, "users.db"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_create_and_lookup(self):
        user = self.store.create_user("a@example.com", "h")
        self.assertEqual(self.store.get_by_id(user.id), user)
        self.assertEqual(self.store.get_by_email("a@example.com"), user)

    def test_duplicate_email_is_case_insensitive(self):
        self.store.create_user("a@example.com", "h")
        with self.assertRaises(UserExistsError):
            self.store.create_user("A@Example.com", "h2")

    def test_lookup_is_case_insensitive(self):
        self.store.create_user("a@example.com", "h")
        self.assertIsNotNone(self.store.get_by_email("A@EXAMPLE.COM"))

    def test_missing_user(self):
        self.assertIsNone(self.store.get_by_id(99))
        self.assertIsNone(self.store.get_by_email("none@example.com"))


if __name__ == "__main__":
    unittest.main()
