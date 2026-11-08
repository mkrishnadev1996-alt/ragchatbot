import warnings

warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")

import asyncio
import json
import logging
import math
import os
import uuid
from contextlib import asynccontextmanager

from ai_generator import LLMRateLimitError
from auth import RateLimiter, RedisRateLimiter, create_auth
from config import config
from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from feedback import create_feedback_router
from feedback_store import FeedbackStore
from langsmith.run_helpers import LangSmithExtra
from message_store import MessageStore
from messages import create_messages_router
from models import Source
from pydantic import BaseModel, ConfigDict, Field
from rag_system import QueryResult, RAGSystem
from user_store import User, UserStore
from utils.body_limit import BodySizeLimitMiddleware
from utils.db import open_database_from_config
from utils.logging_config import setup_logging
from utils.migrations import migrate
from utils.redis_client import redis_from_config
from utils.security_headers import SecurityHeadersMiddleware

setup_logging(config.LOG_LEVEL, config.LOG_FILE, config.LOG_MAX_BYTES, config.LOG_BACKUP_COUNT)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the initial documents on startup"""
    if config.MESSAGE_RETENTION_DAYS:
        try:
            removed = message_store.purge_older_than(config.MESSAGE_RETENTION_DAYS)
            logger.info(
                "Purged %d messages older than %d days", removed, config.MESSAGE_RETENTION_DAYS
            )
        except Exception:
            logger.exception("Message retention purge failed")
    docs_path = config.DOCS_PATH
    if not config.INGEST_ON_STARTUP:
        logger.info("INGEST_ON_STARTUP is off; skipping document ingestion (run ingest.py)")
    elif os.path.exists(docs_path):
        logger.info("Loading initial documents from %s", docs_path)
        try:
            courses, chunks = rag_system.add_course_folder(docs_path, clear_existing=False)
            logger.info("Loaded %d courses with %d chunks", courses, chunks)
        except Exception:
            logger.exception("Error loading initial documents")
    else:
        logger.warning("Docs folder %s not found; skipping initial load", docs_path)
    try:
        yield
    finally:
        # Close the input guard's shared async HTTP client and the database pool
        if rag_system.input_guard:
            await rag_system.input_guard.aclose()
        database.close()


# Initialize FastAPI app
# Interactive docs and the schema are off when ENABLE_API_DOCS=false (production)
app = FastAPI(
    title="Course Materials RAG System",
    root_path="",
    lifespan=lifespan,
    docs_url="/docs" if config.ENABLE_API_DOCS else None,
    redoc_url="/redoc" if config.ENABLE_API_DOCS else None,
    openapi_url="/openapi.json" if config.ENABLE_API_DOCS else None,
)


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


for _name, _value in (
    ("ALLOWED_HOSTS", config.ALLOWED_HOSTS),
    ("ALLOWED_ORIGINS", config.ALLOWED_ORIGINS),
):
    if "*" in _csv(_value):
        logger.warning("%s is '*' (any); set it to explicit values in production", _name)

app.add_middleware(BodySizeLimitMiddleware, max_bytes=config.MAX_BODY_BYTES)

# Trusted hosts and CORS are config-driven (ALLOWED_HOSTS / ALLOWED_ORIGINS, "*" = any).
# Auth uses bearer tokens rather than cookies, so credentials are not allowed cross-origin.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=_csv(config.ALLOWED_HOSTS) or ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=_csv(config.ALLOWED_ORIGINS) or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Retry-After"],
)

# Content-Security-Policy blocks injected scripts even if sanitizing ever misses something.
# Scripts are all local (frontend/vendor); inline styles are used by the UI.
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; "
        "frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}
if config.ENABLE_HSTS:
    SECURITY_HEADERS["Strict-Transport-Security"] = f"max-age={config.HSTS_MAX_AGE}"


# Pure ASGI (not @app.middleware("http")) so streamed responses aren't wrapped and client
# disconnects reach the streaming endpoint. Swagger UI (/docs, /redoc) loads scripts from a
# CDN, so the strict CSP would break it there.
app.add_middleware(
    SecurityHeadersMiddleware, headers=SECURITY_HEADERS, skip_csp_paths=("/docs", "/redoc")
)


# Shared state backends (declared first: the pgvector vector store uses the same PostgreSQL)
database = open_database_from_config(config)

# Initialize RAG system
rag_system = RAGSystem(config, database=database)

# Authentication. The router must be included before the catch-all static mount below.
redis_client = redis_from_config(config) if config.REDIS_URL else None


def make_limiter(name: str, max_attempts: int, window_seconds: int, **kwargs) -> RateLimiter:
    """Rate limiter shared across workers via Redis when REDIS_URL is set, else per process"""
    if redis_client is not None:
        return RedisRateLimiter(redis_client, name, max_attempts, window_seconds, **kwargs)
    return RateLimiter(max_attempts, window_seconds, **kwargs)


user_store = UserStore(database)
auth_router, get_current_user = create_auth(
    user_store,
    config.JWT_SECRET,
    config.JWT_EXPIRE_MINUTES,
    make_limiter("auth", config.AUTH_RATE_LIMIT, config.AUTH_RATE_WINDOW),
)
app.include_router(auth_router)

# Thumbs up/down feedback on answers (same sqlite file as the users table)
app.include_router(create_feedback_router(FeedbackStore(database), get_current_user))

# Persisted chat history, replayed in the UI after login
message_store = MessageStore(database)
app.include_router(
    create_messages_router(message_store, get_current_user, config.MAX_STORED_MESSAGES)
)

# All stores have created their tables; apply versioned schema changes (indexes, ...)
migrate(database)


# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for course queries"""

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=config.MAX_QUERY_LENGTH)
    session_id: str | None = Field(default=None, max_length=config.MAX_SESSION_ID_LENGTH)


class QueryResponse(BaseModel):
    """Response model for course queries"""

    answer: str
    sources: list[Source]
    session_id: str
    message_id: str  # identifies this answer when the client sends feedback on it


class CourseStats(BaseModel):
    """Response model for course statistics"""

    total_courses: int
    course_titles: list[str]


# API Endpoints


query_limiter = make_limiter(
    "query",
    config.QUERY_RATE_LIMIT,
    config.QUERY_RATE_WINDOW,
    detail="Too many questions. Please wait a moment and try again.",
)


def limit_queries(user: User = Depends(get_current_user)) -> User:
    """Authenticate, then apply the daily quota and the per-user query rate limit."""
    if config.DAILY_QUERY_QUOTA:
        used = message_store.count_questions_since(user.id, 24)
        if used >= config.DAILY_QUERY_QUOTA:
            logger.warning("Daily quota reached (user=%s, used=%d)", user.id, used)
            raise HTTPException(
                status_code=429,
                detail="Daily question limit reached. Please try again tomorrow.",
                headers={"Retry-After": str(config.DAILY_QUOTA_RETRY_AFTER)},
            )
    query_limiter.check(f"user:{user.id}")
    return user


# Plain `def` (not `async def`): the RAG pipeline is blocking end to end (sync httpx guard,
# ChromaDB and sentence-transformers have no async API), so FastAPI runs it in its threadpool
# and the event loop stays free for other requests.
def _internal_error(what: str, exc: Exception) -> HTTPException:
    """Log the real error under a short id and give the client only a generic message + id"""
    error_id = uuid.uuid4().hex[:8]
    logger.error("%s (error_id=%s)", what, error_id, exc_info=exc)
    return HTTPException(status_code=500, detail=f"Internal server error (reference: {error_id})")


def _open_session(request: QueryRequest, user: User) -> str:
    """The request's session id (created if missing); 404 for an unknown or foreign session"""
    session_id = request.session_id
    if not session_id:
        return rag_system.session_manager.create_session(user.id)
    if not rag_system.session_manager.is_owner(session_id, user.id):
        # Unknown and foreign sessions look the same, so ids can't be probed
        logger.warning("Session access denied (user=%s, session=%s)", user.id, session_id)
        raise HTTPException(status_code=404, detail="Session not found")
    return session_id


def _langsmith_extra(user: User, session_id: str) -> LangSmithExtra:
    return {"metadata": {"user_id": user.id, "session_id": session_id}, "tags": ["api"]}


def _save_exchange(user: User, session_id: str, question: str, result: QueryResult) -> str:
    """Persist a finished exchange (unless blocked); returns its message id"""
    message_id = uuid.uuid4().hex
    if not result.blocked:  # refused messages are not kept or replayed
        try:
            message_store.add_exchange(
                user.id, session_id, question, result.answer, result.sources, message_id
            )
        except Exception:
            # Losing history is better than failing an answer we already generated
            logger.exception("Failed to persist exchange (session=%s)", session_id)
    return message_id


def _to_http_error(exc: Exception) -> HTTPException:
    """Map an error raised while answering to the HTTP error the client should see"""
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, LLMRateLimitError):
        logger.warning("Returning 429 to client: LLM rate limit (retry_after=%s)", exc.retry_after)
        headers = None
        if exc.retry_after is not None:
            headers = {"Retry-After": str(max(1, math.ceil(exc.retry_after)))}
        return HTTPException(status_code=429, detail=str(exc), headers=headers)
    return _internal_error("Query failed", exc)


@app.post("/api/query", response_model=QueryResponse)
def query_documents(request: QueryRequest, user: User = Depends(limit_queries)):
    """Process a query and return response with sources"""
    try:
        session_id = _open_session(request, user)
        logger.info("Query received (session=%s, %d chars)", session_id, len(request.query))

        result = rag_system.query(
            request.query, session_id, langsmith_extra=_langsmith_extra(user, session_id)
        )
        message_id = _save_exchange(user, session_id, request.query, result)

        return QueryResponse(
            answer=result.answer,
            sources=result.sources,
            session_id=session_id,
            message_id=message_id,
        )
    except Exception as e:
        raise _to_http_error(e) from e


def _sse(event: dict) -> bytes:
    """One server-sent event frame carrying a JSON payload"""
    return b"data: " + json.dumps(event, ensure_ascii=False).encode() + b"\n\n"


@app.post("/api/query/stream")
async def query_documents_stream(request: QueryRequest, user: User = Depends(limit_queries)):
    """Like /api/query, but streams the answer as server-sent events.

    Natively async: the LLM is awaited on the event loop and the blocking steps (input guard,
    vector search, sqlite) are pushed to worker threads, so a slow answer holds no thread.

    Events: {"type": "delta", "text"} while the answer is written, then one
    {"type": "done", answer, sources, session_id, message_id}. Failures before the first
    token (auth, bad session, rate limits, the LLM rejecting the request) are ordinary HTTP
    errors; a failure after that is sent as {"type": "error", "detail"}.
    """
    events = None
    try:
        session_id = await asyncio.to_thread(_open_session, request, user)
        logger.info("Stream query received (session=%s, %d chars)", session_id, len(request.query))
        events = rag_system.aquery_stream(
            request.query, session_id, langsmith_extra=_langsmith_extra(user, session_id)
        )
        # Pull the first item now so an early failure becomes a real HTTP status
        first = await anext(events, None)
    except Exception as e:
        if events is not None:
            await events.aclose()
        raise _to_http_error(e) from e

    async def frames():
        try:
            item = first
            while item is not None:
                if isinstance(item, QueryResult):
                    message_id = await asyncio.to_thread(
                        _save_exchange, user, session_id, request.query, item
                    )
                    yield _sse(
                        {
                            "type": "done",
                            "answer": item.answer,
                            "sources": [s.model_dump() for s in item.sources],
                            "session_id": session_id,
                            "message_id": message_id,
                        }
                    )
                else:
                    yield _sse({"type": "delta", "text": item})
                item = await anext(events, None)
        except Exception as e:
            err = _to_http_error(e)
            yield _sse({"type": "error", "detail": err.detail, "status": err.status_code})
        finally:
            # Also runs when the client disconnects (the task is cancelled): stop the LLM call
            await events.aclose()

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/healthz")
async def healthz():
    """Liveness probe: no auth, touches no dependencies"""
    return {"status": "ok"}


def _check_redis() -> None:
    if redis_client is not None:
        redis_client.ping()


def _readiness_checks() -> list:
    checks = [
        ("vector_store", rag_system.vector_store.get_course_count),
        ("database", database.ping),
    ]
    if redis_client is not None:
        checks.append(("redis", _check_redis))
    return checks


@app.get("/readyz")
def readyz():
    """Readiness probe: no auth; 503 unless the vector store and the sqlite file both work.

    Plain `def` so the blocking checks run in the threadpool. The LLM and guard providers are
    deliberately not probed: it would cost a call per probe, and their outages are reported
    per request instead.
    """
    checks = {}
    for name, check in _readiness_checks():
        try:
            check()
            checks[name] = "ok"
        except Exception:
            logger.exception("Readiness check failed: %s", name)
            checks[name] = "failed"
    ready = all(v == "ok" for v in checks.values())
    return JSONResponse(
        {"status": "ready" if ready else "unavailable", "checks": checks},
        status_code=200 if ready else 503,
    )


@app.get("/api/config")
async def get_client_config(response: Response):
    """Public client-side limits, so the UI never hardcodes them"""
    # Fixed for the life of the process (read from the environment at startup)
    response.headers["Cache-Control"] = f"public, max-age={config.CONFIG_CACHE_SECONDS}"
    return {"max_query_length": config.MAX_QUERY_LENGTH}


@app.get("/api/courses", response_model=CourseStats)
async def get_course_stats(response: Response):
    """Get course analytics and statistics"""
    try:
        analytics = rag_system.get_course_analytics()
        # Courses are only ingested at startup, so a short browser cache is safe
        response.headers["Cache-Control"] = f"public, max-age={config.COURSES_CACHE_SECONDS}"
        return CourseStats(
            total_courses=analytics["total_courses"], course_titles=analytics["course_titles"]
        )
    except Exception as e:
        raise _internal_error("Failed to load course stats", e) from e


# Serve static files for the frontend
app.mount("/", StaticFiles(directory=config.FRONTEND_PATH, html=True), name="static")
