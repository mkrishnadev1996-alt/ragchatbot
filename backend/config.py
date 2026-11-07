import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Config:
    """Configuration settings for the RAG system"""

    # LLM settings (any OpenAI-compatible endpoint: Groq, Google Gemini, etc.)
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1600"))  # Max tokens per completion
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))  # SDK retries on 429/5xx
    LLM_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT", "60"))  # Seconds per LLM request

    # Input guard (decision model on OpenRouter). Disabled when GUARD_API_KEY is empty.
    GUARD_API_KEY: str = os.getenv("GUARD_API_KEY", "")
    GUARD_URL: str = os.getenv("GUARD_URL", "https://openrouter.ai/api/alpha/decisions")
    GUARD_MODEL: str = os.getenv("GUARD_MODEL", "respan/span-01-lite:free")
    GUARD_THRESHOLD: float = float(os.getenv("GUARD_THRESHOLD", "0.5"))  # block when P >= this
    GUARD_TIMEOUT: float = float(os.getenv("GUARD_TIMEOUT", "10"))  # Seconds per guard request

    # Authentication (JWT bearer tokens, users stored in SQLite)
    JWT_SECRET: str = os.getenv("JWT_SECRET", "")
    JWT_EXPIRE_MINUTES: int = int(os.getenv("JWT_EXPIRE_MINUTES", str(60 * 24)))
    USERS_DB_PATH: str = os.getenv("USERS_DB_PATH", "./users.db")
    PASSWORD_MIN_LENGTH: int = 8
    PASSWORD_MAX_LENGTH: int = 128
    AUTH_RATE_LIMIT: int = int(
        os.getenv("AUTH_RATE_LIMIT", "10")
    )  # Login/register per window, per IP
    AUTH_RATE_WINDOW: int = int(os.getenv("AUTH_RATE_WINDOW", "60"))  # Window in seconds

    # LangSmith tracing is switched on by LANGSMITH_TRACING / LANGSMITH_API_KEY / LANGSMITH_PROJECT,
    # which LangChain reads straight from the environment. This flag controls whether prompts,
    # answers and tool I/O are sent with the traces (false = traces carry timing/structure only).
    LANGSMITH_CAPTURE_IO: bool = os.getenv("LANGSMITH_CAPTURE_IO", "true").lower() not in (
        "0",
        "false",
        "no",
        "off",
    )

    # Embedding model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Document processing settings
    CHUNK_SIZE: int = 800  # Size of text chunks for vector storage
    CHUNK_OVERLAP: int = 100  # Characters to overlap between chunks
    MAX_RESULTS: int = 5  # Maximum search results to return
    # Course-name lookups farther than this (L2 distance, all-MiniLM-L6-v2) resolve to "not
    # found". Measured on the current catalog: real names/keywords 0.16-1.59, unrelated
    # words 1.61-1.84.
    COURSE_MATCH_MAX_DISTANCE: float = float(os.getenv("COURSE_MATCH_MAX_DISTANCE", "1.6"))
    MAX_HISTORY: int = 2  # Exchanges (question + answer pairs) to remember
    MAX_QUERY_LENGTH: int = int(os.getenv("MAX_QUERY_LENGTH", "1000"))  # Max chars per question

    # Per-user rate limit on /api/query (each query costs an LLM call)
    QUERY_RATE_LIMIT: int = int(os.getenv("QUERY_RATE_LIMIT", "20"))  # Max queries per window
    QUERY_RATE_WINDOW: int = int(os.getenv("QUERY_RATE_WINDOW", "60"))  # Window in seconds

    # Network exposure. Comma-separated lists; "*" allows any. Auth uses bearer tokens (not
    # cookies), so CORS credentials are never enabled.
    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "*")
    ALLOWED_HOSTS: str = os.getenv("ALLOWED_HOSTS", "*")
    MAX_BODY_BYTES: int = int(os.getenv("MAX_BODY_BYTES", str(64 * 1024)))  # Request body cap

    # In-memory session expiry (idle time) and cap on live sessions
    SESSION_TTL_SECONDS: int = int(os.getenv("SESSION_TTL_SECONDS", str(24 * 3600)))
    MAX_SESSIONS: int = int(os.getenv("MAX_SESSIONS", "10000"))

    # Message history and HTTP caching
    MAX_STORED_MESSAGES: int = 200  # Messages returned by GET /api/messages
    CONFIG_CACHE_SECONDS: int = 3600  # Cache-Control max-age for /api/config
    COURSES_CACHE_SECONDS: int = 60  # Cache-Control max-age for /api/courses

    # Database paths
    CHROMA_PATH: str = "./chroma_db"  # ChromaDB storage location

    # Logging settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")  # DEBUG, INFO, WARNING, ERROR
    LOG_FILE: str | None = os.getenv("LOG_FILE") or None  # Optional rotating log file path


config = Config()

# Must be set before the first LangSmith client is created (LangSmith reads these from the env)
if not config.LANGSMITH_CAPTURE_IO:
    os.environ["LANGSMITH_HIDE_INPUTS"] = "true"
    os.environ["LANGSMITH_HIDE_OUTPUTS"] = "true"
