import warnings

warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")

import logging
import math
import os
import uuid

from ai_generator import LLMRateLimitError
from auth import RateLimiter, create_auth
from config import config
from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from feedback import create_feedback_router
from feedback_store import FeedbackStore
from message_store import MessageStore
from messages import create_messages_router
from pydantic import BaseModel, ConfigDict, Field
from rag_system import RAGSystem
from user_store import User, UserStore
from utils.logging_config import setup_logging

setup_logging(config.LOG_LEVEL, config.LOG_FILE)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Course Materials RAG System", root_path="")

# Add trusted host middleware for proxy
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Enable CORS with proper settings for proxy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize RAG system
rag_system = RAGSystem(config)

# Authentication. The router must be included before the catch-all static mount below.
user_store = UserStore(config.USERS_DB_PATH)
auth_router, get_current_user = create_auth(
    user_store, config.JWT_SECRET, config.JWT_EXPIRE_MINUTES
)
app.include_router(auth_router)

# Thumbs up/down feedback on answers (same sqlite file as the users table)
app.include_router(create_feedback_router(FeedbackStore(config.USERS_DB_PATH), get_current_user))

# Persisted chat history, replayed in the UI after login
message_store = MessageStore(config.USERS_DB_PATH)
app.include_router(create_messages_router(message_store, get_current_user))


# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for course queries"""

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=config.MAX_QUERY_LENGTH)
    session_id: str | None = None


class QueryResponse(BaseModel):
    """Response model for course queries"""

    answer: str
    sources: list[str]
    session_id: str
    message_id: str  # identifies this answer when the client sends feedback on it


class CourseStats(BaseModel):
    """Response model for course statistics"""

    total_courses: int
    course_titles: list[str]


# API Endpoints


query_limiter = RateLimiter(
    max_attempts=config.QUERY_RATE_LIMIT,
    window_seconds=config.QUERY_RATE_WINDOW,
    detail="Too many questions. Please wait a moment and try again.",
)


def limit_queries(user: User = Depends(get_current_user)) -> User:
    """Authenticate, then apply the per-user query rate limit."""
    query_limiter.check(f"user:{user.id}")
    return user


@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest, user: User = Depends(limit_queries)):
    """Process a query and return response with sources"""
    try:
        # Create session if not provided
        session_id = request.session_id
        if not session_id:
            session_id = rag_system.session_manager.create_session(user.id)
        elif not rag_system.session_manager.is_owner(session_id, user.id):
            # Unknown and foreign sessions look the same, so ids can't be probed
            raise HTTPException(status_code=404, detail="Session not found")

        logger.info("Query received (session=%s, %d chars)", session_id, len(request.query))
        logger.debug("Query text (session=%s): %s", session_id, request.query)

        # Process query using RAG system
        answer, sources = rag_system.query(
            request.query,
            session_id,
            langsmith_extra={
                "metadata": {"user_id": user.id, "session_id": session_id},
                "tags": ["api"],
            },
        )

        message_id = uuid.uuid4().hex
        try:
            message_store.add_exchange(
                user.id, session_id, request.query, answer, sources, message_id
            )
        except Exception:
            # Losing history is better than failing an answer we already generated
            logger.exception("Failed to persist exchange (session=%s)", session_id)

        return QueryResponse(
            answer=answer, sources=sources, session_id=session_id, message_id=message_id
        )
    except HTTPException:
        raise
    except LLMRateLimitError as e:
        logger.warning("Returning 429 to client: LLM rate limit (retry_after=%s)", e.retry_after)
        headers = None
        if e.retry_after is not None:
            headers = {"Retry-After": str(max(1, math.ceil(e.retry_after)))}
        raise HTTPException(status_code=429, detail=str(e), headers=headers) from e
    except Exception as e:
        logger.exception("Query failed; returning 500")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/config")
async def get_client_config(response: Response):
    """Public client-side limits, so the UI never hardcodes them"""
    # Fixed for the life of the process (read from the environment at startup)
    response.headers["Cache-Control"] = "public, max-age=3600"
    return {"max_query_length": config.MAX_QUERY_LENGTH}


@app.get("/api/courses", response_model=CourseStats)
async def get_course_stats(response: Response):
    """Get course analytics and statistics"""
    try:
        analytics = rag_system.get_course_analytics()
        # Courses are only ingested at startup, so a short browser cache is safe
        response.headers["Cache-Control"] = "public, max-age=60"
        return CourseStats(
            total_courses=analytics["total_courses"], course_titles=analytics["course_titles"]
        )
    except Exception as e:
        logger.exception("Failed to load course stats; returning 500")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.on_event("startup")
async def startup_event():
    """Load initial documents on startup"""
    docs_path = "../docs"
    if os.path.exists(docs_path):
        logger.info("Loading initial documents from %s", docs_path)
        try:
            courses, chunks = rag_system.add_course_folder(docs_path, clear_existing=False)
            logger.info("Loaded %d courses with %d chunks", courses, chunks)
        except Exception:
            logger.exception("Error loading initial documents")
    else:
        logger.warning("Docs folder %s not found; skipping initial load", docs_path)


# Custom static file handler with no-cache headers for development


class DevStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if isinstance(response, FileResponse):
            # Add no-cache headers for development
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


# Serve static files for the frontend
app.mount("/", StaticFiles(directory="../frontend", html=True), name="static")
