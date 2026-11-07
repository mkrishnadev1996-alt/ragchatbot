FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

RUN pip install --no-cache-dir uv

WORKDIR /app

# Dependencies first, so code changes don't re-install them
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

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
RUN useradd --create-home --uid 10001 app && mkdir -p /data && chown -R app /data /app
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
