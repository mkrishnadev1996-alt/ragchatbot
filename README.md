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
   `JWT_SECRET` signs login tokens and is required (generate one with `python -c "import secrets; print(secrets.token_urlsafe(48))"`). Users register and log in from the web UI; accounts are stored in `backend/users.db` (SQLite) and the chat requires being logged in.
   For Google Gemini instead, use `LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/` and `LLM_MODEL=gemini-2.5-flash`. See `.env.example`.

   **Input guard (optional).** To enable it, also set:
   ```bash
   GUARD_API_KEY=your_openrouter_api_key_here
   ```
   It uses `respan/span-01-lite:free` through the OpenRouter Decisions API (alpha) and blocks a question when any of its harmful, jailbreak or out-of-scope scores reaches `GUARD_THRESHOLD` (default `0.5`). Optional overrides: `GUARD_URL`, `GUARD_MODEL`, `GUARD_THRESHOLD`. Without `GUARD_API_KEY` the guard is off and every question goes to the LLM. If the guard service errors, the question is allowed through. The out-of-scope check is tuned to the current AI/software courses, so adjust `InputGuard.BEHAVIORS` in `backend/input_guard.py` if you add courses on other subjects.

## Accounts and API

Chat requires an account. Open the app, choose **Register** in the login dialog, enter an email and a password of at least 8 characters, and you are logged in; later visits use **Log in**. Your email and a **Log out** link appear in the top-right corner. Logging out only clears the token in your browser; it stays valid until it expires (`JWT_EXPIRE_MINUTES`, default 1440).

| Endpoint | Auth | Purpose |
| --- | --- | --- |
| `POST /api/auth/register` | none | Create an account, returns `{token, user}` (409 if the email exists) |
| `POST /api/auth/login` | none | Returns `{token, user}` (401 on bad credentials) |
| `GET /api/auth/me` | bearer | Current user |
| `POST /api/query` | bearer | Ask a question (401 without a valid token) |
| `GET /api/courses` | none | Course catalog stats |

Send the token as `Authorization: Bearer <token>`. Login and register are rate-limited per IP.

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
- API Documentation: `http://localhost:8000/docs`

