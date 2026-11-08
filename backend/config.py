import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


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
    # Fallback chain. LLM_PROVIDERS="groq,nvidia,google" tries each provider's models in order
    # (see llm_provider_settings); empty = the single LLM_* endpoint above. A rate-limited or
    # failing endpoint is skipped for a cooldown (a 429 uses its Retry-After, within bounds).
    LLM_PROVIDERS: str = os.getenv("LLM_PROVIDERS", "")
    LLM_BREAKER_FAILURES: int = int(os.getenv("LLM_BREAKER_FAILURES", "3"))  # failures in a row
    LLM_BREAKER_COOLDOWN: float = float(os.getenv("LLM_BREAKER_COOLDOWN", "60"))  # seconds
    LLM_BREAKER_MAX_COOLDOWN: float = float(os.getenv("LLM_BREAKER_MAX_COOLDOWN", "900"))
    # The model may call the search tool repeatedly (model -> tool -> model rounds); this
    # caps the rounds per question as a cost/runaway guard. 1 = the old single search.
    MAX_TOOL_ROUNDS: int = int(os.getenv("MAX_TOOL_ROUNDS", "3"))

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
    # PostgreSQL for users/feedback/messages (postgresql://user:pass@host/db). Empty = the sqlite
    # file above. Required when running more than one process that must share this data.
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    DATABASE_POOL_SIZE: int = int(os.getenv("DATABASE_POOL_SIZE", "10"))
    # Redis for sessions and rate limits (redis://host:6379/0). Empty = in-memory, per process.
    REDIS_URL: str = os.getenv("REDIS_URL", "")
    PASSWORD_MIN_LENGTH: int = 8
    PASSWORD_MAX_LENGTH: int = 128
    AUTH_RATE_LIMIT: int = int(
        os.getenv("AUTH_RATE_LIMIT", "10")
    )  # Login/register per window, per IP
    AUTH_RATE_WINDOW: int = int(os.getenv("AUTH_RATE_WINDOW", "60"))  # Window in seconds

    # LangSmith tracing is switched on by LANGSMITH_TRACING / LANGSMITH_API_KEY / LANGSMITH_PROJECT,
    # which LangChain reads straight from the environment. This flag controls whether prompts,
    # answers and tool I/O are sent with the traces (false = traces carry timing/structure only).
    LANGSMITH_CAPTURE_IO: bool = os.getenv("LANGSMITH_CAPTURE_IO", "true").lower() not in _FALSE

    # Vector store backend: "chroma" (embedded ChromaDB, single process) or "pgvector"
    # (PostgreSQL + pgvector, shared by any number of processes; needs DATABASE_URL).
    VECTOR_BACKEND: str = os.getenv("VECTOR_BACKEND", "chroma")
    # Hybrid search (vector + keyword, fused with RRF). pgvector backend only.
    HYBRID_SEARCH: bool = os.getenv("HYBRID_SEARCH", "true").lower() not in _FALSE
    # The keyword side ignores words found in more than this fraction of all chunks.
    KEYWORD_MAX_DOC_FRACTION: float = float(os.getenv("KEYWORD_MAX_DOC_FRACTION", "0.05"))

    # Embeddings. "local" runs a sentence-transformers model in the process; "openai" calls an
    # OpenAI-compatible /embeddings endpoint (pgvector backend only). Changing the model or
    # dimensions requires re-indexing: `python ingest.py --clear`.
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", "")
    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", "")
    EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "0"))  # openai provider

    # Document processing settings
    CHUNK_SIZE: int = 800  # Size of text chunks for vector storage
    CHUNK_OVERLAP: int = 100  # Characters to overlap between chunks
    MAX_RESULTS: int = 5  # Maximum search results to return
    # Course-name lookups farther than this (L2 distance, all-MiniLM-L6-v2) resolve to "not
    # found". Measured on the current catalog: real names/keywords 0.16-1.59, unrelated
    # words 1.61-1.84.
    COURSE_MATCH_MAX_DISTANCE: float = float(os.getenv("COURSE_MATCH_MAX_DISTANCE", "1.6"))
    MAX_HISTORY: int = 2  # Exchanges (question + answer pairs) to remember
    # Character budget for the history sent to the LLM (~4 chars per token). Oldest messages
    # are dropped first; 0 disables the budget.
    MAX_HISTORY_CHARS: int = int(os.getenv("MAX_HISTORY_CHARS", "6000"))
    MAX_QUERY_LENGTH: int = int(os.getenv("MAX_QUERY_LENGTH", "1000"))  # Max chars per question

    # Per-user rate limit on /api/query (each query costs an LLM call)
    QUERY_RATE_LIMIT: int = int(os.getenv("QUERY_RATE_LIMIT", "20"))  # Max queries per window
    QUERY_RATE_WINDOW: int = int(os.getenv("QUERY_RATE_WINDOW", "60"))  # Window in seconds
    # Per-user cap on questions in any rolling 24 h, counted from the persisted messages table
    # (so it holds across restarts and workers). 0 disables it.
    DAILY_QUERY_QUOTA: int = int(os.getenv("DAILY_QUERY_QUOTA", "500"))

    # Network exposure. Comma-separated lists; "*" allows any. Auth uses bearer tokens (not
    # cookies), so CORS credentials are never enabled.
    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "*")
    ALLOWED_HOSTS: str = os.getenv("ALLOWED_HOSTS", "*")
    MAX_BODY_BYTES: int = int(os.getenv("MAX_BODY_BYTES", str(64 * 1024)))  # Request body cap
    # Swagger UI / ReDoc / openapi.json. Set ENABLE_API_DOCS=false in production.
    ENABLE_API_DOCS: bool = os.getenv("ENABLE_API_DOCS", "true").lower() not in _FALSE
    # Strict-Transport-Security. Only enable when the site is served over HTTPS.
    ENABLE_HSTS: bool = os.getenv("ENABLE_HSTS", "false").lower() in _TRUE
    HSTS_MAX_AGE: int = int(os.getenv("HSTS_MAX_AGE", str(365 * 24 * 3600)))

    # In-memory session expiry (idle time) and cap on live sessions
    SESSION_TTL_SECONDS: int = int(os.getenv("SESSION_TTL_SECONDS", str(24 * 3600)))
    MAX_SESSIONS: int = int(os.getenv("MAX_SESSIONS", "10000"))

    # Message history and HTTP caching
    MAX_STORED_MESSAGES: int = 200  # Messages returned by GET /api/messages
    # Stored chat messages older than this many days are deleted at startup. 0 keeps them forever.
    MESSAGE_RETENTION_DAYS: int = int(os.getenv("MESSAGE_RETENTION_DAYS", "0"))
    CONFIG_CACHE_SECONDS: int = 3600  # Cache-Control max-age for /api/config
    COURSES_CACHE_SECONDS: int = 60  # Cache-Control max-age for /api/courses

    # Storage and content locations (relative paths resolve against the working directory,
    # which must be backend/ when started outside Docker)
    CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./chroma_db")  # ChromaDB storage location
    DOCS_PATH: str = os.getenv("DOCS_PATH", "../docs")  # Course documents ingested at startup
    # Ingest DOCS_PATH when the server starts. Turn off when several workers run, and ingest
    # with `python ingest.py` instead (embedded Chroma is not safe for concurrent writers).
    INGEST_ON_STARTUP: bool = os.getenv("INGEST_ON_STARTUP", "true").lower() not in _FALSE
    FRONTEND_PATH: str = os.getenv("FRONTEND_PATH", "../frontend")  # Static UI

    # Logging settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")  # DEBUG, INFO, WARNING, ERROR
    LOG_FILE: str | None = os.getenv("LOG_FILE") or None  # Optional rotating log file path


# Providers known by name: OpenAI-compatible base URL and free-tier models, best first.
# Override per provider with <NAME>_BASE_URL / <NAME>_MODELS; the key is <NAME>_API_KEY.
KNOWN_LLM_PROVIDERS: dict[str, dict[str, str]] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "models": "openai/gpt-oss-120b,openai/gpt-oss-20b,qwen/qwen3.8-27b",
    },
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "models": "nvidia/nemotron-3-ultra-550b-a55b,nvidia/nemotron-3.5-lightning-30b-a3b",
    },
    "google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": "gemini-3.5-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite",
        # Gemini 3 needs its thought_signature echoed on tool calls, which LangChain drops
        "flatten_tool_messages": "true",
    },
}


# Accepted spellings of a provider name (the env variable prefix stays as written)
LLM_PROVIDER_ALIASES = {"grok": "groq"}  # "grok" here means Groq (api.groq.com), not xAI


def _first_set(environ, *keys: str) -> str:
    return next((environ[k] for k in keys if environ.get(k)), "")


def llm_provider_settings(names: str, environ=os.environ) -> list[dict]:
    """Settings for each provider in the comma-separated `names`, in order.

    Per provider NAME (upper-cased), the first of these that is set wins:
      key       NAME_API_KEY | NAME_LLM_API_KEY      (required: no key = provider skipped, so
                                                       one chain works on machines with only
                                                       some of the keys)
      base URL  NAME_BASE_URL | NAME_LLM_BASE_URL    (default from KNOWN_LLM_PROVIDERS)
      models    NAME_MODELS, a comma list, best first (default from KNOWN_LLM_PROVIDERS)
      preferred NAME_LLM_MODEL | NAME_MODEL          (optional single model, tried first,
                                                       then the models above)
      flatten   NAME_FLATTEN_TOOL_MESSAGES=true|false  (send tool turns as plain text; on for
                                                       google, see llm_router.flatten_tool_turns)
    Unknown names need a base URL and models. Returns dicts with name, base_url, api_key and
    models (a list without duplicates).
    """
    out = []
    for raw in names.split(","):
        name = raw.strip().lower()
        if not name:
            continue
        prefix = name.upper().replace("-", "_")
        known = KNOWN_LLM_PROVIDERS.get(LLM_PROVIDER_ALIASES.get(name, name), {})
        api_key = _first_set(environ, f"{prefix}_API_KEY", f"{prefix}_LLM_API_KEY")
        base_url = _first_set(environ, f"{prefix}_BASE_URL", f"{prefix}_LLM_BASE_URL") or known.get(
            "base_url", ""
        )
        listed = (environ.get(f"{prefix}_MODELS") or known.get("models", "")).split(",")
        preferred = _first_set(environ, f"{prefix}_LLM_MODEL", f"{prefix}_MODEL").strip()
        # dict keys: de-duplicate while keeping order
        models = list(dict.fromkeys(m.strip() for m in [preferred, *listed] if m and m.strip()))
        if not api_key or not base_url or not models:
            logging.getLogger(__name__).warning(
                "LLM provider %r skipped: needs %s_API_KEY%s",
                name,
                prefix,
                "" if known else f", {prefix}_BASE_URL and {prefix}_MODELS",
            )
            continue
        flatten = (
            environ.get(f"{prefix}_FLATTEN_TOOL_MESSAGES") or known.get("flatten_tool_messages", "")
        ).lower() in ("1", "true", "yes", "on")
        out.append(
            {
                "name": name,
                "base_url": base_url,
                "api_key": api_key,
                "models": models,
                "flatten_tool_messages": flatten,
            }
        )
    return out


config = Config()

# Must be set before the first LangSmith client is created (LangSmith reads these from the env)
if not config.LANGSMITH_CAPTURE_IO:
    os.environ["LANGSMITH_HIDE_INPUTS"] = "true"
    os.environ["LANGSMITH_HIDE_OUTPUTS"] = "true"
