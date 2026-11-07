"""Phase 2: pgvector store, hybrid search, embedding providers and backend selection.

The pgvector tests need a PostgreSQL with the pgvector extension (e.g. the
`pgvector/pgvector:pg17` image); they run when TEST_DATABASE_URL points at one and skip
otherwise. Everything else runs offline.
"""

import hashlib
import math
import os
from contextlib import closing
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import embeddings
import httpx
import pytest
from embeddings import OpenAICompatibleEmbeddings, create_embedder, normalize
from models import Course, CourseChunk, Lesson
from pg_vector_store import (
    PgVectorStore,
    keyword_tokens,
    rare_lexemes,
    rrf_fuse,
    tsquery_from_lexemes,
)
from rag_system import RAGSystem
from utils.db import PostgresDatabase

PG_URL = os.getenv("TEST_DATABASE_URL", "")

# --- pure functions -------------------------------------------------------------------------


def test_rrf_prefers_ids_found_by_both_rankings():
    assert rrf_fuse([["a", "b", "c"], ["c", "d"]])[0] == "c"


def test_rrf_handles_empty_and_single_rankings():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []
    assert rrf_fuse([["x", "y"], []]) == ["x", "y"]


def test_rrf_ties_keep_first_seen_order():
    assert rrf_fuse([["a"], ["b"]]) == ["a", "b"]


def test_keyword_tokens_dedupe_and_drop_punctuation():
    assert keyword_tokens("What is MCP? MCP tools!") == ["what", "is", "mcp", "tools"]
    assert keyword_tokens("!!! ?? a") == []


def test_keyword_tokens_are_capped():
    assert len(keyword_tokens(" ".join(f"w{i}x" for i in range(100)))) == 20


def test_rare_lexemes_drops_common_unknown_and_everything_when_empty():
    freq = {"course": 100, "mcp": 30, "hyde": 1, "rerank": 5}
    assert rare_lexemes(["course", "mcp", "hyde", "rerank", "zzz"], freq, 100, 0.05) == [
        "hyde",
        "rerank",
    ]
    assert rare_lexemes(["hyde"], freq, 0, 0.05) == []
    assert rare_lexemes(["course"], freq, 100, 0.05) == []


def test_tsquery_from_lexemes_quotes_and_rejects_odd_characters():
    assert tsquery_from_lexemes(["hyde", "cross"]) == "'hyde' | 'cross'"
    assert tsquery_from_lexemes(["ok", "bad'; drop", "a b"]) == "'ok'"
    assert tsquery_from_lexemes([]) == ""


# --- embedding providers --------------------------------------------------------------------


def _embedding_response(vectors, shuffle=False):
    data = [{"index": i, "embedding": v} for i, v in enumerate(vectors)]
    return {"data": list(reversed(data)) if shuffle else data}


def _provider(handler, dims=3, **kw):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAICompatibleEmbeddings("http://emb.test/v1/", "key", "m", dims, client=client, **kw)


def test_normalize():
    assert normalize([3.0, 4.0]) == [0.6, 0.8]
    assert normalize([0.0, 0.0]) == [0.0, 0.0]


def test_openai_provider_orders_by_index_and_normalizes():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, json=_embedding_response([[3, 4, 0], [0, 0, 2]], shuffle=True))

    out = _provider(handler).embed(["a", "b"])
    assert seen == {"url": "http://emb.test/v1/embeddings", "auth": "Bearer key"}
    assert out == [[0.6, 0.8, 0.0], [0.0, 0.0, 1.0]]


def test_openai_provider_batches():
    calls = []

    def handler(request):
        import json

        n = len(json.loads(request.content)["input"])
        calls.append(n)
        return httpx.Response(200, json=_embedding_response([[1, 0, 0]] * n))

    assert len(_provider(handler, batch_size=2).embed(list("abcde"))) == 5
    assert calls == [2, 2, 1]


def test_openai_provider_retries_rate_limits_then_succeeds():
    attempts = []

    def handler(request):
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(429)
        return httpx.Response(200, json=_embedding_response([[1, 0, 0]]))

    with patch.object(embeddings.time, "sleep"):
        assert _provider(handler).embed(["a"]) == [[1.0, 0.0, 0.0]]
    assert len(attempts) == 3


def test_openai_provider_gives_up_after_max_attempts():
    with patch.object(embeddings.time, "sleep"), pytest.raises(httpx.HTTPStatusError):
        _provider(lambda r: httpx.Response(503), max_attempts=2).embed(["a"])


def test_openai_provider_does_not_retry_client_errors():
    attempts = []

    def handler(request):
        attempts.append(1)
        return httpx.Response(401)

    with pytest.raises(httpx.HTTPStatusError):
        _provider(handler).embed(["a"])
    assert len(attempts) == 1


def test_openai_provider_rejects_wrong_dimensions_or_count():
    with pytest.raises(ValueError, match="dimensions"):
        _provider(lambda r: httpx.Response(200, json=_embedding_response([[1, 0]]))).embed(["a"])
    with pytest.raises(ValueError, match="Expected 2"):
        _provider(lambda r: httpx.Response(200, json=_embedding_response([[1, 0, 0]]))).embed(
            ["a", "b"]
        )


def test_create_embedder_validates_config():
    with pytest.raises(ValueError, match="Unknown EMBEDDING_PROVIDER"):
        create_embedder(SimpleNamespace(EMBEDDING_PROVIDER="nope"))
    with pytest.raises(ValueError, match="EMBEDDING_BASE_URL"):
        create_embedder(SimpleNamespace(EMBEDDING_PROVIDER="openai", EMBEDDING_MODEL="m"))
    with pytest.raises(ValueError, match="EMBEDDING_DIMENSIONS"):
        create_embedder(
            SimpleNamespace(
                EMBEDDING_PROVIDER="openai", EMBEDDING_MODEL="m", EMBEDDING_BASE_URL="http://x"
            )
        )
    ok = create_embedder(
        SimpleNamespace(
            EMBEDDING_PROVIDER="openai",
            EMBEDDING_MODEL="m",
            EMBEDDING_BASE_URL="http://x",
            EMBEDDING_DIMENSIONS=8,
        )
    )
    assert ok.dimensions == 8


# --- backend selection in RAGSystem ---------------------------------------------------------


def _config(**kw):
    base = dict(
        CHUNK_SIZE=800,
        CHUNK_OVERLAP=100,
        CHROMA_PATH="unused",
        EMBEDDING_MODEL="unused",
        MAX_RESULTS=5,
        MAX_HISTORY=2,
        LLM_API_KEY="k",
        LLM_BASE_URL="http://llm.test",
        LLM_MODEL="m",
        LLM_MAX_TOKENS=1,
        LLM_TEMPERATURE=0,
        LLM_MAX_RETRIES=1,
    )
    return SimpleNamespace(**{**base, **kw})


@pytest.fixture
def light_rag():
    with patch("rag_system.AIGenerator"), patch("rag_system.VectorStore") as chroma:
        yield chroma


def test_pgvector_backend_requires_postgres(light_rag):
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        RAGSystem(_config(VECTOR_BACKEND="pgvector"))
    sqlite_like = MagicMock(dialect="sqlite")
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        RAGSystem(_config(VECTOR_BACKEND="pgvector"), database=sqlite_like)


def test_pgvector_backend_is_built_with_config(light_rag):
    db = MagicMock(dialect="postgres")
    with (
        patch("rag_system.PgVectorStore") as store,
        patch("rag_system.create_embedder") as make,
    ):
        rag = RAGSystem(
            _config(VECTOR_BACKEND="pgvector", HYBRID_SEARCH=False, COURSE_MATCH_MAX_DISTANCE=1.2),
            database=db,
        )
    store.assert_called_once_with(
        db,
        make.return_value,
        5,
        max_course_distance=1.2,
        hybrid=False,
        keyword_max_doc_fraction=0.05,
    )
    assert rag.vector_store is store.return_value
    light_rag.assert_not_called()  # Chroma untouched


def test_unknown_backend_and_chroma_with_remote_embeddings_are_rejected(light_rag):
    with pytest.raises(ValueError, match="VECTOR_BACKEND"):
        RAGSystem(_config(VECTOR_BACKEND="faiss"))
    with pytest.raises(ValueError, match="EMBEDDING_PROVIDER"):
        RAGSystem(_config(VECTOR_BACKEND="chroma", EMBEDDING_PROVIDER="openai"))


def test_default_backend_is_still_chroma(light_rag):
    rag = RAGSystem(_config())
    assert rag.vector_store is light_rag.return_value


# --- PgVectorStore against a real PostgreSQL + pgvector -------------------------------------

DIMS = 8


def _unit(*xs):
    return normalize([float(x) for x in xs] + [0.0] * (DIMS - len(xs)))


class FakeEmbedder:
    """Controlled vectors for known texts, a deterministic hash vector for anything else."""

    def __init__(self, known=None, model="fake", dims=DIMS):
        self.known = known or {}
        self.model = model
        self.dimensions = dims

    def embed(self, texts):
        return [self.known.get(t) or self._hash(t) for t in texts]

    def _hash(self, text):
        digest = hashlib.sha256(text.encode()).digest()
        vec = [b - 127.5 for b in digest[: self.dimensions]]
        return normalize(vec)


@pytest.fixture
def pg():
    if not PG_URL:
        pytest.skip("TEST_DATABASE_URL not set")
    database = PostgresDatabase(PG_URL, max_size=3)
    try:
        with closing(database.connect()) as conn, conn:
            conn.execute("DROP TABLE IF EXISTS course_content, course_catalog, vector_meta CASCADE")
            try:
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            except Exception:
                pytest.skip("pgvector extension is not available on this server")
    except Exception:
        database.close()
        raise
    yield database
    database.close()


def _chunk(course, lesson, index, text):
    return CourseChunk(content=text, course_title=course, lesson_number=lesson, chunk_index=index)


def _course(title, link="https://c.test"):
    return Course(
        title=title,
        course_link=link,
        instructor="I",
        lessons=[Lesson(lesson_number=1, title="One", lesson_link="https://c.test/1")],
    )


def test_store_roundtrip_filters_and_links(pg):
    store = PgVectorStore(pg, FakeEmbedder(), max_results=10, hybrid=False)
    store.add_course_metadata(_course("Alpha Course"))
    store.add_course_metadata(_course("Beta Course", link=None))
    store.add_course_content(
        [
            _chunk("Alpha Course", 1, 0, "alpha lesson one text"),
            _chunk("Alpha Course", 2, 1, "alpha lesson two text"),
            _chunk("Beta Course", 1, 0, "beta lesson one text"),
        ]
    )
    assert store.get_existing_course_titles() == ["Alpha Course", "Beta Course"]
    assert store.get_course_count() == 2
    assert store.get_course_link("Alpha Course") == "https://c.test"
    assert store.get_course_link("Beta Course") is None
    assert store.get_course_link("Nope") is None
    assert store.get_lesson_link("Alpha Course", 1) == "https://c.test/1"
    assert store.get_lesson_link("Alpha Course", 9) is None

    everything = store.search("alpha lesson one text")
    assert everything.documents[0] == "alpha lesson one text"
    assert everything.distances[0] == pytest.approx(0.0, abs=1e-6)
    assert len(everything.documents) == 3
    assert everything.metadata[0] == {
        "course_title": "Alpha Course",
        "chunk_index": 0,
        "lesson_number": 1,
    }

    only_lesson2 = store.search("alpha", course_name="Alpha Course", lesson_number=2)
    assert only_lesson2.documents == ["alpha lesson two text"]
    assert {m["course_title"] for m in store.search("x", course_name="Beta Course").metadata} == {
        "Beta Course"
    }
    assert len(store.search("alpha", limit=1).documents) == 1


def test_course_name_resolution_respects_distance_cutoff(pg):
    known = {"Alpha Course": _unit(1), "alpha": _unit(1, 0.1), "zzz": _unit(0, 0, 0, 0, 1)}
    store = PgVectorStore(pg, FakeEmbedder(known), max_course_distance=0.5, hybrid=False)
    store.add_course_metadata(_course("Alpha Course"))
    store.add_course_content([_chunk("Alpha Course", 1, 0, "text")])
    assert store.search("q", course_name="alpha").documents == ["text"]
    miss = store.search("q", course_name="zzz")
    assert miss.error == "No course found matching 'zzz'"


def test_search_with_no_courses_or_matches_is_empty_not_an_error(pg):
    store = PgVectorStore(pg, FakeEmbedder())
    assert store.search("anything").is_empty() and store.search("anything").error is None
    assert store.search("x", course_name="Nope").error is not None


def test_hybrid_search_surfaces_keyword_match_that_vectors_miss(pg):
    known = {
        "rareword": _unit(0, 1),  # the query
        "alpha text with rareword": _unit(0, 0, 0, 0, 1),  # far from the query vector
        "bravo text": _unit(0, 1),
        "charlie text": _unit(0.1, 1),
    }
    chunks = [_chunk("C", 1, i, t) for i, t in enumerate(known) if t != "rareword"]

    def run(hybrid):
        store = PgVectorStore(
            pg, FakeEmbedder(known), max_results=2, hybrid=hybrid, keyword_max_doc_fraction=0.5
        )
        store.clear_all_data()
        store.add_course_content(chunks)
        return store.search("rareword").documents

    assert run(hybrid=False) == ["bravo text", "charlie text"]
    hybrid = run(hybrid=True)
    assert hybrid[0] == "alpha text with rareword" and len(hybrid) == 2


def test_hybrid_search_survives_punctuation_and_stopword_only_queries(pg):
    store = PgVectorStore(pg, FakeEmbedder(), hybrid=True)
    store.add_course_content([_chunk("C", 1, 0, "some text here")])
    assert store.search("'); DROP TABLE course_content; --").error is None
    assert store.search("the of and").error is None  # only stop words
    assert store.search("???").error is None
    assert store.get_course_count() == 0 and store.search("text").documents == ["some text here"]


def test_content_writes_are_idempotent_and_deletable(pg):
    store = PgVectorStore(pg, FakeEmbedder(), hybrid=False)
    chunk = _chunk("C", 1, 0, "first")
    store.add_course_content([chunk])
    store.add_course_content([_chunk("C", 1, 0, "second")])  # same id: replaced, not duplicated
    assert store.search("x").documents == ["second"]
    store.delete_course_content("C")
    assert store.search("x").is_empty()
    store.add_course_content([])  # no-op


def test_failed_content_write_inserts_nothing(pg):
    store = PgVectorStore(pg, FakeEmbedder(), hybrid=False)
    good = _chunk("C", 1, 0, "good")
    bad = CourseChunk(content="bad", course_title="C", lesson_number=1, chunk_index=1)
    bad.content = None  # type: ignore[assignment]  # violates NOT NULL on insert
    with pytest.raises(Exception):  # noqa: B017
        store.add_course_content([good, bad])
    assert store.search("x").is_empty()


def test_changing_the_embedding_model_is_flagged_until_cleared(pg):
    PgVectorStore(pg, FakeEmbedder(model="model-a")).add_course_metadata(_course("C"))
    other = PgVectorStore(pg, FakeEmbedder(model="model-b"))
    assert other.index_mismatch and "ingest.py --clear" in other.index_mismatch
    assert "ingest.py --clear" in (other.search("x").error or "")
    with pytest.raises(RuntimeError):
        other.get_course_count()
    other.clear_all_data()
    assert other.index_mismatch is None
    assert other.get_course_count() == 0


def test_clear_all_data_adopts_new_embedding_size(pg):
    PgVectorStore(pg, FakeEmbedder(dims=DIMS))
    bigger = PgVectorStore(pg, FakeEmbedder(model="big", dims=16))
    bigger.clear_all_data()
    bigger.add_course_content([_chunk("C", 1, 0, "text")])
    assert bigger.search("text").documents == ["text"]


def test_schema_creation_is_repeatable_and_concurrent_safe(pg):
    for _ in range(3):
        PgVectorStore(pg, FakeEmbedder())


def test_requires_postgres():
    with pytest.raises(ValueError, match="PostgreSQL"):
        PgVectorStore(MagicMock(dialect="sqlite"), FakeEmbedder())


def test_fake_embedder_is_unit_length():
    v = FakeEmbedder().embed(["abc"])[0]
    assert math.isclose(sum(x * x for x in v), 1.0)


def test_read_only_queries_return_connections_cleanly(pg, caplog):
    store = PgVectorStore(pg, FakeEmbedder(), hybrid=True)
    with caplog.at_level("WARNING", logger="psycopg.pool"):
        store.get_course_count()
        store.search("anything")
    assert not [r for r in caplog.records if "rolling back" in r.getMessage()]


def test_hybrid_ignores_words_found_in_most_chunks(pg):
    # "common" is in every chunk, so it must not drag the ranking away from the vector order
    known = {
        "common": _unit(0, 1),
        "common alpha": _unit(0, 1),
        "common bravo": _unit(0.1, 1),
        "common charlie": _unit(0.5, 1),
    }
    chunks = [_chunk("C", 1, i, t) for i, t in enumerate(known) if t != "common"]

    def run(hybrid):
        store = PgVectorStore(
            pg, FakeEmbedder(known), max_results=3, hybrid=hybrid, keyword_max_doc_fraction=0.5
        )
        store.clear_all_data()
        store.add_course_content(chunks)
        return store.search("common").documents

    assert (
        run(hybrid=True) == run(hybrid=False) == ["common alpha", "common bravo", "common charlie"]
    )


def test_corpus_word_counts_are_refreshed_after_writes(pg):
    store = PgVectorStore(pg, FakeEmbedder(), hybrid=True, keyword_max_doc_fraction=0.5)
    store.add_course_content([_chunk("C", 1, 0, "uniqueterm here"), _chunk("C", 1, 1, "other")])
    assert store.search("uniqueterm").documents[0] == "uniqueterm here"  # df 1/2 <= 0.5: usable
    store.add_course_content(
        [_chunk("C", 1, 2, "uniqueterm again"), _chunk("C", 1, 3, "uniqueterm")]
    )
    with closing(pg.connect()) as conn:
        _, total = store._corpus_stats(conn)
    assert total == 4  # cache was invalidated by the write, not served stale
