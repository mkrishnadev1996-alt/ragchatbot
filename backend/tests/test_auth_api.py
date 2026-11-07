import os
import tempfile
import unittest

from auth import RateLimiter, create_auth
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient
from user_store import User, UserStore


class TestAuthApi(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        store = UserStore(os.path.join(self.tmp.name, "users.db"))
        router, get_current_user = create_auth(store, "t" * 32, 60, RateLimiter(max_attempts=100))
        app = FastAPI()
        app.include_router(router)

        @app.get("/protected")
        def protected(user: User = Depends(get_current_user)):
            return {"id": user.id}

        self.client = TestClient(app)

    def tearDown(self):
        self.tmp.cleanup()

    def _register(self, email="a@example.com", password="password123"):
        return self.client.post("/api/auth/register", json={"email": email, "password": password})

    def test_register_returns_token_and_me_works(self):
        res = self._register()
        self.assertEqual(res.status_code, 201)
        token = res.json()["token"]
        me = self.client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(me.json()["email"], "a@example.com")

    def test_duplicate_register_is_409(self):
        self._register()
        self.assertEqual(self._register("A@example.com").status_code, 409)

    def test_short_password_and_bad_email_rejected(self):
        self.assertEqual(self._register(password="short").status_code, 422)
        self.assertEqual(self._register(email="nope").status_code, 422)

    def test_login_success_and_failures_are_generic(self):
        self._register()
        ok = self.client.post(
            "/api/auth/login", json={"email": "a@example.com", "password": "password123"}
        )
        self.assertEqual(ok.status_code, 200)
        bad_pw = self.client.post(
            "/api/auth/login", json={"email": "a@example.com", "password": "wrongwrong"}
        )
        no_user = self.client.post(
            "/api/auth/login", json={"email": "x@example.com", "password": "password123"}
        )
        self.assertEqual(bad_pw.status_code, 401)
        self.assertEqual(bad_pw.json(), no_user.json())

    def test_protected_route_requires_valid_token(self):
        self.assertEqual(self.client.get("/protected").status_code, 401)
        res = self.client.get("/protected", headers={"Authorization": "Bearer junk"})
        self.assertEqual(res.status_code, 401)
        token = self._register().json()["token"]
        res = self.client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(res.status_code, 200)

    def test_rate_limit(self):
        store = UserStore(os.path.join(self.tmp.name, "rl.db"))
        router, _ = create_auth(store, "s" * 32, 60, RateLimiter(max_attempts=2))
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        body = {"email": "a@example.com", "password": "password123"}
        codes = [client.post("/api/auth/login", json=body).status_code for _ in range(3)]
        self.assertEqual(codes, [401, 401, 429])

    def test_limiter_is_per_key_with_retry_after(self):
        limiter = RateLimiter(max_attempts=1, window_seconds=60, detail="slow down")
        limiter.check("user:1")
        limiter.check("user:2")  # other keys are unaffected
        with self.assertRaises(HTTPException) as ctx:
            limiter.check("user:1")
        self.assertEqual(ctx.exception.status_code, 429)
        self.assertEqual(ctx.exception.detail, "slow down")
        headers = ctx.exception.headers or {}
        self.assertTrue(1 <= int(headers["Retry-After"]) <= 60)

    def test_missing_secret_fails_fast(self):
        store = UserStore(os.path.join(self.tmp.name, "x.db"))
        with self.assertRaises(RuntimeError):
            create_auth(store, "", 60)

    def test_short_secret_fails_fast(self):
        store = UserStore(os.path.join(self.tmp.name, "y.db"))
        with self.assertRaises(RuntimeError):
            create_auth(store, "x" * 31, 60)


if __name__ == "__main__":
    unittest.main()
