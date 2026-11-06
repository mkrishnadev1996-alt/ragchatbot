# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Python 3.13+, dependencies managed with `uv`. Lint and format with ruff (configured in `pyproject.toml`, line length 100): `uv run ruff check .` and `uv run ruff format .`.

```bash
uv sync                                  # install dependencies
./run.sh                                 # start server (cd backend && uvicorn app:app --reload --port 8000)
cd backend && uv run uvicorn app:app --reload --port 8000   # equivalent manual start
uv run pytest                            # run unit tests (backend/tests; no network/LLM/model download)
uv run mypy backend                      # type check (config in pyproject.toml)
```

- Web UI: http://localhost:8000 · API docs: http://localhost:8000/docs
- Requires a `.env` file in the project root containing `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` (`.env.example` is the template). `config.py` loads it with `load_dotenv()`.
- On Windows, run commands from Git Bash (the README requires it; `run.sh` is a bash script).

## Things that break if you ignore them

- **Working directory matters.** The server must be started from `backend/`. Startup loads course documents from the relative path `../docs`, the static frontend is mounted from `../frontend`, and ChromaDB persists to `./chroma_db` (relative, so it lands in `backend/chroma_db/`). Running uvicorn from the root will silently skip the docs and the frontend.
- **Documents are deduplicated by course title, not by content.** `RAGSystem.add_course_folder` skips any file whose parsed title already exists in the `course_catalog` collection. Editing a course file and restarting will NOT re-index it. To rebuild, delete `backend/chroma_db/` or call `add_course_folder(..., clear_existing=True)`.
- **Course file format is parsed positionally.** `DocumentProcessor.process_course_document` expects line 1 `Course Title: ...`, line 2 `Course Link: ...`, line 3 `Course Instructor: ...`, then `Lesson N: Title` markers, optionally followed by a `Lesson Link: ...` line. Files without lesson markers fall back to a single unnumbered chunk stream.
- Supported extensions for folder ingestion: `.pdf`, `.docx`, `.txt`. Note that `DocumentProcessor.read_file` (which delegates to `utils/file_utils.read_text_file`) only reads text, so PDF/DOCX content will not be parsed correctly with the current code.
- **Log, don't print.** Modules use `logger = logging.getLogger(__name__)`; handlers are configured once by `setup_logging()` in `app.py`. Don't add `print()` or call `logging.basicConfig()`. Use `logger.exception(...)` inside `except` blocks so tracebacks are kept.
- The LLM provider is any OpenAI-compatible endpoint, set via `LLM_BASE_URL` / `LLM_MODEL` (defaults in `backend/config.py`: Groq). API limits (`max_tokens=1600`, `temperature=0`, `max_retries=3`) are the `ChatOpenAI(...)` arguments in `AIGenerator.__init__`.

## Architecture

The request flow spans several modules, so it helps to read them together:

1. **`backend/app.py`** — FastAPI app. `POST /api/query` creates a session if the client doesn't send one, then calls `RAGSystem.query`. `GET /api/courses` returns catalog stats. Startup event ingests `../docs`. CORS and TrustedHost are wide open.
2. **`backend/rag_system.py`** — Orchestrator. Owns all components and wires them together. `query()` builds a prompt, asks the AI generator for an answer with the tool definitions attached, then collects sources from the tool manager and records the exchange in session history.
3. **`backend/ai_generator.py`** — LLM calls via LangChain's `ChatOpenAI` (`langchain-openai`, Chat Completions path, not the Responses API) pointed at `LLM_BASE_URL` (Groq and Gemini both expose OpenAI-compatible chat completions). This is an agentic tool loop with a hard limit of **one tool round**: the first call uses `llm.bind_tools(tools)`; if the returned `AIMessage` has `tool_calls`, it executes them, appends each result as a `ToolMessage`, then makes a second call on the unbound `llm` (no tools) to produce the final text. The system prompt (a `SystemMessage`) enforces "one search per query" and that general-knowledge questions are answered without searching. Conversation history is sent as prior `HumanMessage`/`AIMessage` messages between the system prompt and the query. `openai.RateLimitError` still propagates from LangChain and is translated to `LLMRateLimitError`.
4. **`backend/search_tools.py`** — Tool abstraction. `Tool` is an ABC with `get_tool_definition()` (OpenAI function-tool schema: `{"type": "function", "function": {...}}`, which `bind_tools` accepts directly; `ToolManager.register_tool` reads the name from `function.name`) and `execute()`. `ToolManager` registers tools by name and dispatches calls. `CourseSearchTool` (the only tool, `search_course_content`) formats results as `[Course - Lesson N]` headers and stashes `last_sources` on the instance, which `ToolManager.get_last_sources()` later reads. Adding a tool means implementing the ABC and registering it in `RAGSystem.__init__`.
5. **`backend/vector_store.py`** — ChromaDB wrapper with two collections:
   - `course_catalog`: one document per course, keyed by course title (the ID). Metadata holds link, instructor, and a JSON blob of lessons. Used for **fuzzy course-name resolution**: a `course_name` filter is first resolved by vector-searching this collection, and the top hit's title becomes an exact `where` filter on `course_content`.
   - `course_content`: chunked lesson text with metadata `course_title`, `lesson_number`, `chunk_index`. Chunk IDs are `{title with spaces→_}_{chunk_index}`.
   Embeddings use `SentenceTransformerEmbeddingFunction` with `all-MiniLM-L6-v2`, which runs locally and downloads on first use.
6. **`backend/document_processor.py`** — Parses course files and chunks lesson text by sentences (target `CHUNK_SIZE`=800 chars with `CHUNK_OVERLAP`=100). The chunking algorithm itself lives in `utils/text_utils.chunk_text`; `DocumentProcessor.chunk_text` is a thin wrapper that passes the configured sizes. The first chunk of each lesson is prefixed with `Lesson N content:`; the last lesson's chunks are prefixed with the course title and lesson number. Chunk-prefix behavior is inconsistent across lesson branches, so check both before changing it.
7. **`backend/session_manager.py`** — In-memory dict of sessions, keyed `session_N`. Keeps the last `MAX_HISTORY * 2` messages. Lost on restart.
8. **`backend/config.py`** — Single `Config` dataclass instance, loads `.env` via `python-dotenv`. Tunables: chunk size/overlap, `MAX_RESULTS` (search top-k), `MAX_HISTORY` (exchanges remembered), `LOG_LEVEL` (default `INFO`) and `LOG_FILE` (optional rotating log file; console only when unset).
9. **`frontend/`** — Plain HTML/JS/CSS with no build step. `script.js` calls `/api/query` and `/api/courses`, and keeps the `session_id` returned by the server to continue conversations.
10. **`backend/utils/`** — Shared helpers with no dependency on the domain modules. Import from the specific submodule:
   - `file_utils.py` — `read_text_file` (UTF-8 read, falls back to ignoring invalid bytes).
   - `text_utils.py` — `chunk_text(text, chunk_size, chunk_overlap)`, the sentence chunker.
   - `http_utils.py` — `parse_retry_after(headers)`, reads a numeric Retry-After header (HTTP-date values return None).
   - `logging_config.py` — `setup_logging(level, log_file)`. Replaces existing handlers on each call, and quiets chatty third-party loggers (httpx, openai, langchain, urllib3, huggingface_hub, …).
   Put new cross-module helpers here rather than in a domain module.

Data flow for a query: UI → `/api/query` → `RAGSystem.query` → LLM (may call `search_course_content`) → `VectorStore.search` (optional course-name resolution, then filtered Chroma query) → LLM writes the final answer → response includes `sources` from the tool.

## Dependencies of note

Pinned in `pyproject.toml`: `chromadb==1.0.15`, `openai==3.42.1` (still imported for its exception types), `langchain-openai` and `langchain-core` (resolved by `uv.lock`), `sentence-transformers==5.0.0`, `fastapi==0.116.1`. `sentence-transformers` pulls in PyTorch, so the first `uv sync` is heavy.
