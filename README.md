# Course Materials RAG System

A Retrieval-Augmented Generation (RAG) system designed to answer questions about course materials using semantic search and AI-powered responses.

## Overview

This application is a full-stack web application that enables users to query course materials and receive intelligent, context-aware responses. It uses ChromaDB for vector storage, any OpenAI-compatible LLM API (Groq, Google Gemini, etc.) for AI generation, and provides a web interface for interaction.

An optional input guard screens every question before it reaches the main LLM. It blocks harmful or illegal requests, jailbreak attempts and out-of-scope questions, and replies "Sorry, we cannot answer that." instead.


## Prerequisites

- Python 3.13 or higher
- uv (Python package manager)
- An API key for an OpenAI-compatible LLM provider, e.g. [Groq](https://console.groq.com/) or [Google AI Studio](https://aistudio.google.com/) (Gemini)
- *(Optional)* An [OpenRouter](https://openrouter.ai/) API key to enable the input guard
- **For Windows**: Use Git Bash to run the application commands - [Download Git for Windows](https://git-scm.com/downloads/win)

## Installation

1. **Install uv** (if not already installed)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install Python dependencies**
   ```bash
   uv sync
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```bash
   LLM_API_KEY=your_api_key_here
   LLM_BASE_URL=https://api.groq.com/openai/v1
   LLM_MODEL=llama-3.3-70b-versatile
   JWT_SECRET=a_long_random_string
   ```
   `JWT_SECRET` signs login tokens and is required (generate one with `python -c "import secrets; print(secrets.token_urlsafe(48))"`). Users register and log in from the web UI; accounts are stored in `backend/users.db` (SQLite, or PostgreSQL if you set `DATABASE_URL`) and the chat requires being logged in.
   For Google Gemini instead, use `LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/` and `LLM_MODEL=gemini-2.5-flash`. See `.env.example`.

   **Input guard (optional).** To enable it, also set:
   ```bash
   GUARD_API_KEY=your_openrouter_api_key_here
   ```
   It uses `respan/span-01-lite:free` through the OpenRouter Decisions API (alpha) and blocks a question when any of its harmful, jailbreak or out-of-scope scores reaches `GUARD_THRESHOLD` (default `0.5`). Optional overrides: `GUARD_URL`, `GUARD_MODEL`, `GUARD_THRESHOLD`. Without `GUARD_API_KEY` the guard is off and every question goes to the LLM. If the guard service errors, the question is allowed through. The out-of-scope check is tuned to the current AI/software courses, so adjust `InputGuard.BEHAVIORS` in `backend/input_guard.py` if you add courses on other subjects.

## Accounts and API

Chat requires an account. Open the app, choose **Register** in the login dialog, enter an email and a password of at least 8 characters, and you are logged in; later visits use **Log in**. Your email and a **Log out** link appear in the top-right corner. Refreshing the page, or logging out and back in, brings back the messages of your current conversation; use the **Clear conversation** button in the left sidebar to start a new one. Logging out only clears the token in your browser; it stays valid until it expires (`JWT_EXPIRE_MINUTES`, default 1440).

| Endpoint | Auth | Purpose |
| --- | --- | --- |
| `POST /api/auth/register` | none | Create an account, returns `{token, user}` (409 if the email exists) |
| `POST /api/auth/login` | none | Returns `{token, user}` (401 on bad credentials) |
| `GET /api/auth/me` | bearer | Current user |
| `POST /api/query` | bearer | Ask a question (401 without a valid token, 429 when rate limited or over the daily quota) |
| `POST /api/query/stream` | bearer | Same, streamed as server-sent events (what the web UI uses) |
| `POST /api/feedback` | bearer | Thumbs up/down on an answer |
| `GET /api/messages` | bearer | Your saved messages; optional `?session_id=` limits to one session |
| `GET /api/courses` | none | Course catalog stats |
| `GET /healthz` | none | Liveness probe |
| `GET /readyz` | none | Readiness probe: 503 unless the vector store, database and (if configured) Redis respond |

Send the token as `Authorization: Bearer <token>`. Login and register are rate-limited per IP, questions per user (`QUERY_RATE_LIMIT` per `QUERY_RATE_WINDOW`) and per day (`DAILY_QUERY_QUOTA`, default 500; `0` = unlimited).

## LangSmith Tracing (optional)

Set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT` in `.env` to send a trace per question to LangSmith: the input guard, the LLM calls, the course search tool and the vector retrieval appear as one nested run, tagged with the user id and session id. Traces include user questions and course text; set `LANGSMITH_CAPTURE_IO=false` to send only timing and structure. Tracing is off by default.

## Running the Application

### Quick Start

Use the provided shell script:
```bash
chmod +x run.sh
./run.sh
```

### Manual Start

```bash
cd backend
uv run uvicorn app:app --reload --port 8000
```

The application will be available at:
- Web Interface: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs` (set `ENABLE_API_DOCS=false` to turn it off)

## Development

```bash
uv run pytest            # unit tests (no network, LLM or model download)
uv run ruff check . && uv run ruff format --check .
uv run mypy backend
```

The store and vector-store tests also run against PostgreSQL when `TEST_DATABASE_URL=postgresql://user:pass@host/db` is set (CI does this); use a throwaway database and a server with the pgvector extension, e.g. the `pgvector/pgvector:pg17` image. Without pgvector the vector-store tests are skipped.

## Production

By default the app keeps its state inside one process (SQLite file, in-memory sessions and rate limits, startup ingestion), which is fine for development and a single small server. To run it more robustly, each piece can move out of the process; set the variable and leave it empty to keep the default:

| Variable | Effect |
| --- | --- |
| `DATABASE_URL` | PostgreSQL (`postgresql://user:pass@host:5432/db`) for users, feedback and messages instead of the SQLite file. `DATABASE_POOL_SIZE` (default 10) sets the connection pool. Existing SQLite data is not copied over. |
| `REDIS_URL` | Redis (`redis://host:6379/0`) for conversation sessions and rate limits, so they are shared and survive restarts. |
| `VECTOR_BACKEND` | `pgvector` stores the course vectors in the same PostgreSQL (needs `DATABASE_URL` and the pgvector extension, e.g. the `pgvector/pgvector:pg17` image) instead of the embedded ChromaDB, so several app processes can share them. Run `python ingest.py --clear` after switching. |
| `HYBRID_SEARCH` | pgvector only, default `true`: combine vector and keyword search, which helps with exact names and rare terms. `KEYWORD_MAX_DOC_FRACTION` (default `0.05`) makes the keyword side ignore words that appear in more than that share of chunks. |
| `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL` | `local` (default, `all-MiniLM-L6-v2` in the process) or `openai` (any OpenAI-compatible embeddings endpoint: also set `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY` and `EMBEDDING_DIMENSIONS`; pgvector only). Changing the model needs `python ingest.py --clear`; the app refuses to search an index built with a different model. |
| `MAX_TOOL_ROUNDS` | How many times the model may search per question (default `3`). It searches as often as it needs, e.g. to compare two courses; this is only a cost guard. `1` gives the old single search. |
| `INGEST_ON_STARTUP` | `false` stops the server from indexing the course documents when it starts; run `cd backend && uv run python ingest.py` (add `--clear` to rebuild) instead. |
| `ENABLE_API_DOCS` | `false` hides `/docs`, `/redoc` and `/openapi.json`. |
| `ENABLE_HSTS` | `true` adds the Strict-Transport-Security header (only when served over HTTPS). |
| `ALLOWED_HOSTS`, `ALLOWED_ORIGINS` | Comma-separated lists. Both default to `*`, and the server logs a warning at startup; set them explicitly in production. |
| `DAILY_QUERY_QUOTA` | Questions per user per rolling 24 hours (default 500, `0` = unlimited). |
| `MAX_HISTORY_CHARS` | Character budget for the chat history sent to the LLM (default 6000, `0` = unlimited). |
| `MESSAGE_RETENTION_DAYS` | Delete stored messages older than this at startup (default `0` = keep forever). |
| `CHROMA_PATH`, `DOCS_PATH`, `FRONTEND_PATH` | Locations of the vector store, the course documents and the web UI. |

See `.env.example` for the full list.

### Docker

```bash
# in .env, in addition to the usual variables: POSTGRES_PASSWORD=<choose one>
docker compose up --build
```

This starts PostgreSQL (with pgvector, holding users, messages and the course vectors), Redis, a one-off `ingest` job that indexes `docs/`, and the app on port 8000, with data kept in Docker volumes. Re-index after adding courses with `docker compose run --rm ingest python ingest.py`. Put a TLS-terminating reverse proxy in front, and set `FORWARDED_ALLOW_IPS` to its address so per-IP rate limits see real client IPs. The image has not been size-optimised: on Linux `sentence-transformers` pulls the CUDA build of PyTorch.

**Scaling out:** with this stack all state is outside the app container (PostgreSQL, Redis), so you can run several app containers behind a load balancer (`docker compose up --scale app=3` after removing the fixed `ports:` mapping). Each container still loads the local embedding model unless you switch `EMBEDDING_PROVIDER`.

**Limits to know about:** with the default `VECTOR_BACKEND=chroma` the vector store is embedded and single-process; hybrid search needs `pgvector`. The input guard fails open when its (free, alpha) service errors or rate limits, so treat it as best-effort.

