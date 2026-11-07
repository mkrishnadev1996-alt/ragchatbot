"""Registration, login and the current-user dependency."""

import logging
import math
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field
from user_store import User, UserExistsError, UserStore
from utils.security import create_token, decode_token, hash_password, verify_password

logger = logging.getLogger(__name__)

MIN_SECRET_LENGTH = 32

# A real hash to verify against when the email is unknown, so timing doesn't leak existence.
_DUMMY_HASH = hash_password("dummy-password")


class Credentials(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserOut(BaseModel):
    id: int
    email: str


class TokenResponse(BaseModel):
    token: str
    user: UserOut


class RateLimiter:
    """In-memory sliding-window limiter keyed by client IP (single-process only)."""

    def __init__(
        self,
        max_attempts: int = 10,
        window_seconds: int = 60,
        detail: str = "Too many attempts. Try again shortly.",
    ):
        self.max_attempts = max_attempts
        self.window = window_seconds
        self.detail = detail
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()  # handlers run in a threadpool

    def check(self, key: str) -> None:
        with self._lock:
            self._check(key)

    def _check(self, key: str) -> None:
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > self.window:
            hits.popleft()
        if len(hits) >= self.max_attempts:
            raise HTTPException(
                status_code=429,
                detail=self.detail,
                headers={"Retry-After": str(max(1, math.ceil(self.window - (now - hits[0]))))},
            )
        hits.append(now)


def create_auth(
    store: UserStore, secret: str, expire_minutes: int, limiter: RateLimiter | None = None
) -> tuple[APIRouter, Callable[..., User]]:
    """Build the auth router and the ``get_current_user`` dependency."""
    if not secret:
        raise RuntimeError("JWT_SECRET is not set; refusing to start without auth configured")
    if len(secret) < MIN_SECRET_LENGTH:
        raise RuntimeError(
            f"JWT_SECRET is too short ({len(secret)} chars); use at least {MIN_SECRET_LENGTH}, "
            'e.g. python -c "import secrets; print(secrets.token_urlsafe(48))"'
        )

    router = APIRouter(prefix="/api/auth", tags=["auth"])
    bearer = HTTPBearer(auto_error=False)
    limiter = limiter or RateLimiter()

    def _client_key(request: Request) -> str:
        return request.client.host if request.client else "unknown"

    def _respond(user: User) -> TokenResponse:
        token = create_token(user.id, secret, expire_minutes)
        return TokenResponse(token=token, user=UserOut(id=user.id, email=user.email))

    def get_current_user(
        creds: HTTPAuthorizationCredentials | None = Depends(bearer),
    ) -> User:
        unauthorized = HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
        if creds is None:
            raise unauthorized
        user_id = decode_token(creds.credentials, secret)
        user = store.get_by_id(user_id) if user_id is not None else None
        if user is None:
            raise unauthorized
        return user

    @router.post("/register", response_model=TokenResponse, status_code=201)
    def register(body: Credentials, request: Request):
        limiter.check(_client_key(request))
        try:
            user = store.create_user(body.email, hash_password(body.password))
        except UserExistsError as e:
            raise HTTPException(status_code=409, detail="Email already registered") from e
        logger.info("Registered user id=%d", user.id)
        return _respond(user)

    @router.post("/login", response_model=TokenResponse)
    def login(body: Credentials, request: Request):
        limiter.check(_client_key(request))
        user = store.get_by_email(body.email)
        ok = verify_password(body.password, user.password_hash if user else _DUMMY_HASH)
        if user is None or not ok:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        return _respond(user)

    @router.get("/me", response_model=UserOut)
    def me(user: User = Depends(get_current_user)):
        return UserOut(id=user.id, email=user.email)

    return router, get_current_user
