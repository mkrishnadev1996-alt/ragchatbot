"""Embedding providers for the pgvector store.

`EMBEDDING_PROVIDER=local` (default) runs a sentence-transformers model inside the process.
`EMBEDDING_PROVIDER=openai` calls any OpenAI-compatible `/embeddings` endpoint instead, so the
app container needs no PyTorch at runtime (the `sentence_transformers` import is lazy).

Every provider returns unit-length vectors, so L2 distances mean the same thing whichever
model is used and `COURSE_MATCH_MAX_DISTANCE` stays meaningful. Changing the model or its
dimensions needs a re-index (`python ingest.py --clear`); the store refuses to mix them.
"""

import logging
import math
import time
from typing import Any, Protocol

import httpx

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    model: str
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """One unit-length vector per text, in order."""
        ...


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return vector if norm == 0 else [x / norm for x in vector]


class LocalEmbeddings:
    """sentence-transformers model loaded in this process (downloads on first use)."""

    def __init__(self, model_name: str, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer  # heavy: imports PyTorch

        self.model = model_name
        self.batch_size = batch_size
        self._model: Any = SentenceTransformer(model_name)
        size = self._model.get_sentence_embedding_dimension()
        if not size:
            raise ValueError(f"Cannot determine the embedding size of {model_name!r}")
        self.dimensions: int = int(size)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return [[float(x) for x in v] for v in vectors]


class OpenAICompatibleEmbeddings:
    """Any server exposing POST {base_url}/embeddings in the OpenAI format."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        dimensions: int,
        timeout: float = 30,
        batch_size: int = 64,
        max_attempts: int = 3,
        client: httpx.Client | None = None,
    ):
        if dimensions <= 0:
            raise ValueError("EMBEDDING_DIMENSIONS must be set for the openai provider")
        self.url = base_url.rstrip("/") + "/embeddings"
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_attempts = max_attempts
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(timeout=timeout)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            out.extend(self._embed_batch(texts[start : start + self.batch_size]))
        return out

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": batch}
        resp = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                resp = self._client.post(self.url, json=payload, headers=self._headers)
            except httpx.TransportError:
                if attempt == self.max_attempts:
                    logger.exception("Embedding request failed after %d attempts", attempt)
                    raise
            else:
                retryable = resp.status_code == 429 or resp.status_code >= 500
                if not retryable or attempt == self.max_attempts:
                    resp.raise_for_status()  # 4xx fails at once; a last 429/5xx raises too
                    break
                logger.warning("Embedding request got %d; retrying", resp.status_code)
            time.sleep(2 ** (attempt - 1))
        assert resp is not None
        data = sorted(resp.json()["data"], key=lambda item: item["index"])
        vectors = [item["embedding"] for item in data]
        if len(vectors) != len(batch):
            raise ValueError(f"Expected {len(batch)} embeddings, got {len(vectors)}")
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise ValueError(
                    f"Embedding has {len(vector)} dimensions but EMBEDDING_DIMENSIONS="
                    f"{self.dimensions}"
                )
        return [normalize([float(x) for x in v]) for v in vectors]


def create_embedder(config) -> EmbeddingProvider:
    """Build the provider chosen by EMBEDDING_PROVIDER."""
    provider = getattr(config, "EMBEDDING_PROVIDER", "local")
    if provider == "local":
        return LocalEmbeddings(config.EMBEDDING_MODEL)
    if provider == "openai":
        base_url = getattr(config, "EMBEDDING_BASE_URL", "")
        if not base_url:
            raise ValueError("EMBEDDING_BASE_URL is required when EMBEDDING_PROVIDER=openai")
        return OpenAICompatibleEmbeddings(
            base_url,
            getattr(config, "EMBEDDING_API_KEY", ""),
            config.EMBEDDING_MODEL,
            getattr(config, "EMBEDDING_DIMENSIONS", 0),
        )
    raise ValueError(f"Unknown EMBEDDING_PROVIDER {provider!r} (use 'local' or 'openai')")
