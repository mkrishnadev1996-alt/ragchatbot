# Stage 1: build the virtualenv. uv and pip never reach the final image.
FROM python:3.13-slim AS builder

ENV UV_LINK_MODE=copy

RUN pip install --no-cache-dir uv

WORKDIR /app

# Dependencies only, so code changes don't re-install them. On Linux uv.lock resolves torch to
# the CPU build (see pyproject.toml), which avoids ~6 GB of CUDA wheels.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project --no-cache


# Stage 2: runtime
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# Only /data is writable by the app; the code and the virtualenv stay root-owned (read-only for
# the app user). A recursive chown of /app would copy the whole virtualenv into a new layer.
RUN useradd --create-home --uid 10001 app && mkdir -p /data && chown app /data

WORKDIR /app

# Same path as in the builder (identical base image), so the venv's interpreter links still work
COPY --from=builder /app/.venv ./.venv
COPY backend ./backend
COPY frontend ./frontend
COPY docs ./docs

# Writable state (Chroma index, sqlite, model cache) lives on a volume, not in the image
ENV CHROMA_PATH=/data/chroma_db \
    USERS_DB_PATH=/data/users.db \
    HF_HOME=/data/hf \
    DOCS_PATH=/app/docs \
    FRONTEND_PATH=/app/frontend \
    ENABLE_API_DOCS=false

USER app
VOLUME /data

# The app expects to run from backend/ (relative imports)
WORKDIR /app/backend
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"

# --proxy-headers makes request.client the real client IP (needed by the per-IP rate limiter);
# uvicorn only trusts X-Forwarded-For from FORWARDED_ALLOW_IPS (set it to your proxy's address).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
