import os
import tempfile
import unittest

from auth import RateLimiter, create_auth
from fastapi import FastAPI
from fastapi.testclient import TestClient
from feedback import create_feedback_router
from feedback_store import FeedbackStore
from user_store import UserStore


class TestFeedbackStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = FeedbackStore(os.path.join(self.tmp.name, "f.db"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_rating_is_replaced_not_duplicated(self):
        self.store.set_rating("m1", 1, "up")
        self.store.set_rating("m1", 1, "down", "s1")
        self.assertEqual(self.store.get_rating("m1", 1), "down")

    def test_ratings_are_per_user(self):
        self.store.set_rating("m1", 1, "up")
        self.store.set_rating("m1", 2, "down")
        self.assertEqual(self.store.get_rating("m1", 1), "up")
        self.assertEqual(self.store.get_rating("m1", 2), "down")

    def test_clear_rating(self):
        self.store.set_rating("m1", 1, "up")
        self.store.clear_rating("m1", 1)
        self.assertIsNone(self.store.get_rating("m1", 1))


class TestFeedbackApi(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        db = os.path.join(self.tmp.name, "app.db")
        self.store = FeedbackStore(db)
        router, get_current_user = create_auth(
            UserStore(db), "t" * 32, 60, RateLimiter(max_attempts=100)
        )
        app = FastAPI()
        app.include_router(router)
        app.include_router(create_feedback_router(self.store, get_current_user))
        self.client = TestClient(app)
        res = self.client.post(
            "/api/auth/register", json={"email": "a@example.com", "password": "password123"}
        )
        self.headers = {"Authorization": f"Bearer {res.json()['token']}"}
        self.user_id = res.json()["user"]["id"]

    def tearDown(self):
        self.tmp.cleanup()

    def _post(self, body, headers=True):
        return self.client.post(
            "/api/feedback", json=body, headers=self.headers if headers else None
        )

    def test_requires_auth(self):
        self.assertEqual(
            self._post({"message_id": "m1", "rating": "up"}, headers=False).status_code, 401
        )

    def test_submit_toggle_and_clear(self):
        self.assertEqual(self._post({"message_id": "m1", "rating": "up"}).status_code, 204)
        self.assertEqual(self.store.get_rating("m1", self.user_id), "up")
        self._post({"message_id": "m1", "rating": "down"})
        self.assertEqual(self.store.get_rating("m1", self.user_id), "down")
        self._post({"message_id": "m1", "rating": None})
        self.assertIsNone(self.store.get_rating("m1", self.user_id))

    def test_invalid_rating_rejected(self):
        self.assertEqual(self._post({"message_id": "m1", "rating": "meh"}).status_code, 422)
        self.assertEqual(self._post({"message_id": "", "rating": "up"}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
