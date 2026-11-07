import unittest
from unittest.mock import patch

from auth import RateLimiter
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from session_manager import SessionManager
from utils.body_limit import BodySizeLimitMiddleware


class TestBodyLimit(unittest.TestCase):
    def setUp(self):
        app = FastAPI()

        @app.post("/echo")
        async def echo(request: Request):
            return {"n": len(await request.body())}

        app.add_middleware(BodySizeLimitMiddleware, max_bytes=100)
        self.client = TestClient(app)

    def test_small_body_passes(self):
        self.assertEqual(self.client.post("/echo", content=b"x" * 100).json(), {"n": 100})

    def test_large_content_length_is_413(self):
        res = self.client.post("/echo", content=b"x" * 101)
        self.assertEqual(res.status_code, 413)

    def test_chunked_body_over_limit_is_413(self):
        def chunks():
            yield b"x" * 60
            yield b"x" * 60

        res = self.client.post("/echo", content=chunks())
        self.assertEqual(res.status_code, 413)


class TestSessionExpiry(unittest.TestCase):
    def test_idle_session_expires(self):
        manager = SessionManager(max_history=2, ttl_seconds=10)
        with patch("session_manager.time.monotonic", return_value=1000.0):
            sid = manager.create_session(1)
            self.assertTrue(manager.is_owner(sid, 1))
        with patch("session_manager.time.monotonic", return_value=1011.0):
            self.assertFalse(manager.is_owner(sid, 1))
        self.assertNotIn(sid, manager.sessions)
        self.assertNotIn(sid, manager.owners)

    def test_use_refreshes_expiry(self):
        manager = SessionManager(max_history=2, ttl_seconds=10)
        with patch("session_manager.time.monotonic", return_value=1000.0):
            sid = manager.create_session(1)
        with patch("session_manager.time.monotonic", return_value=1008.0):
            manager.add_exchange(sid, "q", "a")
        with patch("session_manager.time.monotonic", return_value=1015.0):
            self.assertTrue(manager.is_owner(sid, 1))

    def test_expired_sessions_are_purged_on_create(self):
        manager = SessionManager(max_history=2, ttl_seconds=10)
        with patch("session_manager.time.monotonic", return_value=1000.0):
            old = manager.create_session(1)
        with patch("session_manager.time.monotonic", return_value=1100.0):
            manager.create_session(2)
        self.assertNotIn(old, manager.sessions)

    def test_cap_evicts_least_recently_used(self):
        manager = SessionManager(max_history=2, max_sessions=2)
        with patch("session_manager.time.monotonic", return_value=1.0):
            first = manager.create_session(1)
        with patch("session_manager.time.monotonic", return_value=2.0):
            second = manager.create_session(1)
        with patch("session_manager.time.monotonic", return_value=3.0):
            third = manager.create_session(1)
        self.assertEqual(set(manager.sessions), {second, third})
        self.assertNotIn(first, manager.owners)

    def test_no_limits_by_default(self):
        manager = SessionManager(max_history=2)
        sid = manager.create_session(1)
        self.assertTrue(manager.is_owner(sid, 1))


class TestRateLimiterCleanup(unittest.TestCase):
    def test_stale_keys_are_swept(self):
        limiter = RateLimiter(max_attempts=5, window_seconds=10)
        with patch("auth.time.monotonic", return_value=1000.0):
            limiter._last_sweep = 1000.0
            limiter.check("old")
        with patch("auth.time.monotonic", return_value=1100.0):
            limiter.check("new")
        self.assertNotIn("old", limiter._hits)
        self.assertIn("new", limiter._hits)

    def test_limit_still_enforced(self):
        limiter = RateLimiter(max_attempts=2, window_seconds=60)
        limiter.check("k")
        limiter.check("k")
        with self.assertRaises(HTTPException) as ctx:
            limiter.check("k")
        self.assertEqual(ctx.exception.status_code, 429)


if __name__ == "__main__":
    unittest.main()
