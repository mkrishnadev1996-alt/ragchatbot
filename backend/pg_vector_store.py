"""Vector store on PostgreSQL + the pgvector extension, with hybrid (vector + keyword) search.

Same public interface as the ChromaDB `VectorStore` (see `vector_backend.VectorStoreBackend`),
but the data lives in the Postgres the app already uses, so any number of app processes can
share it. Needs a Postgres image that ships pgvector (e.g. `pgvector/pgvector:pg17`).

Search runs two queries and fuses their rankings with Reciprocal Rank Fusion:
  * vector: nearest chunks by L2 distance (embeddings are unit length, so this ranks like
    cosine and keeps the same distance scale as the Chroma backend);
  * keyword: Postgres full-text search (`ts_rank_cd`) over the question's *rare* words OR'd
    together. Words found in more than `keyword_max_doc_fraction` of the chunks ("course",
    "lesson", ...) are dropped first, because ts_rank has no IDF weighting and common words
    would otherwise swamp the ranking; if no rare word is left, only the vector ranking is
    used. This is not true BM25, but it rescues exact names, acronyms and rare terms that
    embeddings blur.
"""

import json
import logging
import re
import threading
import time
from typing import Any

from config import Config
from embeddings import EmbeddingProvider
from langsmith import traceable
from models import Course, CourseChunk, chunk_id
from search_results import SearchResults, _retrieval_output
from utils.db import Database

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[A-Za-z0-9]+")


def rrf_fuse(rankings: list[list[str]], k: int = Config.HYBRID_RRF_K) -> list[str]:
    """Merge ranked id lists into one: best combined 1/(k+rank) score first.

    An id found by several rankings beats one found by a single ranking at the same position.
    Ties keep the order in which ids were first seen. Score = sum(1 / (k + rank)).
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + position)
    return sorted(scores, key=lambda item: -scores[item])  # sorted() is stable


def keyword_tokens(query: str, limit: int = Config.HYBRID_MAX_KEYWORDS) -> list[str]:
    """The question's distinct words (letters/digits only, lower-case, capped).

    Only [A-Za-z0-9] tokens survive, so user text can never produce a tsquery syntax error.
    """
    seen: dict[str, None] = {}
    for token in _WORD.findall(query.lower()):
        if len(token) > 1:
            seen.setdefault(token)
    return list(seen)[:limit]


def rare_lexemes(
    lexemes: list[str], doc_freq: dict[str, int], total_docs: int, max_fraction: float
) -> list[str]:
    """Lexemes that occur in the corpus but in at most `max_fraction` of its chunks.

    Unknown lexemes (in no chunk) are dropped too: they can never match.
    """
    if total_docs <= 0:
        return []
    return [lex for lex in lexemes if 0 < doc_freq.get(lex, 0) <= max_fraction * total_docs]


def tsquery_from_lexemes(lexemes: list[str]) -> str:
    """An OR tsquery (for the 'simple' configuration, i.e. no re-stemming) of the lexemes."""
    return " | ".join(f"'{lex}'" for lex in lexemes if re.fullmatch(r"[A-Za-z0-9]+", lex))


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


class PgVectorStore:
    def __init__(
        self,
        database: Database,
        embedder: EmbeddingProvider,
        max_results: int = 5,
        max_course_distance: float | None = None,
        hybrid: bool = True,
        keyword_max_doc_fraction: float = Config.KEYWORD_MAX_DOC_FRACTION,
        rrf_k: int = Config.HYBRID_RRF_K,
        candidate_factor: int = Config.HYBRID_CANDIDATE_FACTOR,
        min_candidates: int = Config.HYBRID_MIN_CANDIDATES,
        max_keywords: int = Config.HYBRID_MAX_KEYWORDS,
        stats_cache_seconds: float = Config.KEYWORD_STATS_CACHE_SECONDS,
    ):
        if database.dialect != "postgres":
            raise ValueError("PgVectorStore needs PostgreSQL (set DATABASE_URL)")
        self.db = database
        self.embedder = embedder
        self.max_results = max_results
        # Course-name matches farther than this (L2 distance) count as "no such course".
        self.max_course_distance = max_course_distance
        self.hybrid = hybrid
        self.keyword_max_doc_fraction = keyword_max_doc_fraction
        self.rrf_k = rrf_k
        self.candidate_factor = candidate_factor
        self.min_candidates = min_candidates
        self.max_keywords = max_keywords
        self.stats_cache_seconds = stats_cache_seconds
        self._doc_freq: tuple[float, dict[str, int], int] | None = None
        self._doc_freq_lock = threading.Lock()
        self._meta = f"{embedder.model}|{embedder.dimensions}"
        self.index_mismatch: str | None = None
        self._create_schema()
        self._check_index_matches_embedder()

    # --- schema ----------------------------------------------------------------------------
    def _create_schema(self) -> None:
        dims = int(self.embedder.dimensions)
        self.db.run_ddl(
            [
                "CREATE EXTENSION IF NOT EXISTS vector",
                f"""CREATE TABLE IF NOT EXISTS course_catalog (
                    title TEXT PRIMARY KEY,
                    instructor TEXT,
                    course_link TEXT,
                    lessons_json TEXT,
                    lesson_count INTEGER,
                    embedding vector({dims}) NOT NULL
                )""",
                f"""CREATE TABLE IF NOT EXISTS course_content (
                    id TEXT PRIMARY KEY,
                    course_title TEXT NOT NULL,
                    lesson_number INTEGER,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector({dims}) NOT NULL,
                    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
                )""",
                "CREATE INDEX IF NOT EXISTS idx_course_content_embedding "
                "ON course_content USING hnsw (embedding vector_l2_ops)",
                "CREATE INDEX IF NOT EXISTS idx_course_content_tsv "
                "ON course_content USING gin (tsv)",
                "CREATE INDEX IF NOT EXISTS idx_course_content_course "
                "ON course_content (course_title, lesson_number)",
                "CREATE TABLE IF NOT EXISTS vector_meta "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL)",
            ]
        )

    def _check_index_matches_embedder(self) -> None:
        """Remember which model built the index; flag a different one (never mix vectors)."""
        conn = self.db.connect()
        try:
            with conn:
                conn.execute(
                    "INSERT INTO vector_meta (key, value) VALUES ('embedding', ?) "
                    "ON CONFLICT (key) DO NOTHING",
                    (self._meta,),
                )
                row = conn.execute("SELECT value FROM vector_meta WHERE key = 'embedding'")
                stored = row.fetchone()[0]
        finally:
            conn.close()
        if stored != self._meta:
            self.index_mismatch = (
                f"The vector index was built with '{stored}' but the app is configured for "
                f"'{self._meta}'. Rebuild it: cd backend && python ingest.py --clear"
            )
            logger.error(self.index_mismatch)

    def _require_ready(self) -> None:
        if self.index_mismatch:
            raise RuntimeError(self.index_mismatch)

    # --- search ----------------------------------------------------------------------------
    @traceable(name="vector_store.search", run_type="retriever", process_outputs=_retrieval_output)
    def search(
        self,
        query: str,
        course_name: str | None = None,
        lesson_number: int | None = None,
        limit: int | None = None,
    ) -> SearchResults:
        try:
            self._require_ready()
            course_title = None
            if course_name:
                course_title = self._resolve_course_name(course_name)
                if not course_title:
                    return SearchResults.empty(f"No course found matching '{course_name}'")
            wanted = limit if limit is not None else self.max_results
            return self._search_content(query, course_title, lesson_number, wanted)
        except Exception as e:
            logger.exception("pgvector content search failed (course_name=%s)", course_name)
            return SearchResults.empty(f"Search error: {e}")

    def _search_content(
        self, query: str, course_title: str | None, lesson_number: int | None, wanted: int
    ) -> SearchResults:
        where, where_params = self._where(course_title, lesson_number)
        pool = max(wanted * self.candidate_factor, self.min_candidates)
        qvec = _vector_literal(self.embedder.embed([query])[0])
        columns = (
            "id, course_title, lesson_number, chunk_index, content, embedding <-> ?::vector AS dist"
        )

        conn = self.db.connect()
        try:
            tsquery = self._keyword_query(conn, query) if self.hybrid else ""
            vector_rows = conn.execute(
                f"SELECT {columns} FROM course_content WHERE {where} ORDER BY dist LIMIT ?",
                (qvec, *where_params, pool),
            ).fetchall()
            keyword_rows = []
            if tsquery:
                keyword_rows = conn.execute(
                    f"SELECT {columns} FROM course_content, to_tsquery('simple', ?) AS q "
                    f"WHERE tsv @@ q AND {where} ORDER BY ts_rank_cd(tsv, q) DESC LIMIT ?",
                    (qvec, tsquery, *where_params, pool),
                ).fetchall()
        finally:
            conn.close()

        rows: dict[str, Any] = {r[0]: r for r in [*keyword_rows, *vector_rows]}
        order = rrf_fuse([[r[0] for r in vector_rows], [r[0] for r in keyword_rows]], self.rrf_k)[
            :wanted
        ]
        documents, metadata, distances = [], [], []
        for row_id in order:
            _, title, lesson, index, content, dist = rows[row_id]
            meta: dict[str, Any] = {"course_title": title, "chunk_index": index}
            if lesson is not None:
                meta["lesson_number"] = lesson
            documents.append(content)
            metadata.append(meta)
            distances.append(float(dist))
        return SearchResults(documents=documents, metadata=metadata, distances=distances)

    def _keyword_query(self, conn: Any, query: str) -> str:
        """tsquery of the question's rare words ("" -> skip the keyword ranking)"""
        tokens = keyword_tokens(query, self.max_keywords)
        if not tokens:
            return ""
        # Stem exactly as the indexed text was (this also drops stop words)
        lexemes = [
            r[0]
            for r in conn.execute(
                "SELECT lexeme FROM unnest(to_tsvector('english', ?))", (" ".join(tokens),)
            )
        ]
        doc_freq, total = self._corpus_stats(conn)
        rare = rare_lexemes(lexemes, doc_freq, total, self.keyword_max_doc_fraction)
        return tsquery_from_lexemes(rare)

    def _corpus_stats(self, conn: Any) -> tuple[dict[str, int], int]:
        """Chunks per lexeme and the chunk count, cached (writes here invalidate it)"""
        with self._doc_freq_lock:
            cached = self._doc_freq
            if cached and time.monotonic() - cached[0] < self.stats_cache_seconds:
                return cached[1], cached[2]
        freq = {
            word: ndoc
            for word, ndoc in conn.execute(
                "SELECT word, ndoc FROM ts_stat('SELECT tsv FROM course_content')"
            )
        }
        total = conn.execute("SELECT COUNT(*) FROM course_content").fetchone()[0]
        with self._doc_freq_lock:
            self._doc_freq = (time.monotonic(), freq, total)
        return freq, total

    def _invalidate_corpus_stats(self) -> None:
        with self._doc_freq_lock:
            self._doc_freq = None

    @staticmethod
    def _where(course_title: str | None, lesson_number: int | None) -> tuple[str, list[Any]]:
        clauses: list[str] = ["TRUE"]
        params: list[Any] = []
        if course_title:
            clauses.append("course_title = ?")
            params.append(course_title)
        if lesson_number is not None:
            clauses.append("lesson_number = ?")
            params.append(lesson_number)
        return " AND ".join(clauses), params

    def _resolve_course_name(self, course_name: str) -> str | None:
        """Nearest course title to the given name, or None if none is close enough"""
        qvec = _vector_literal(self.embedder.embed([course_name])[0])
        conn = self.db.connect()
        try:
            row = conn.execute(
                "SELECT title, embedding <-> ?::vector AS dist FROM course_catalog "
                "ORDER BY dist LIMIT 1",
                (qvec,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        title, dist = row
        if self.max_course_distance is not None and dist > self.max_course_distance:
            logger.info("Course name %r too far from any course (distance %.3f)", course_name, dist)
            return None
        return title

    # --- writes ----------------------------------------------------------------------------
    def add_course_metadata(self, course: Course) -> None:
        self._require_ready()
        lessons = [
            {
                "lesson_number": lesson.lesson_number,
                "lesson_title": lesson.title,
                "lesson_link": lesson.lesson_link,
            }
            for lesson in course.lessons
        ]
        embedding = _vector_literal(self.embedder.embed([course.title])[0])
        conn = self.db.connect()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO course_catalog
                        (title, instructor, course_link, lessons_json, lesson_count, embedding)
                    VALUES (?, ?, ?, ?, ?, ?::vector)
                    ON CONFLICT (title) DO UPDATE SET
                        instructor = EXCLUDED.instructor,
                        course_link = EXCLUDED.course_link,
                        lessons_json = EXCLUDED.lessons_json,
                        lesson_count = EXCLUDED.lesson_count,
                        embedding = EXCLUDED.embedding
                    """,
                    (
                        course.title,
                        course.instructor,
                        course.course_link,
                        json.dumps(lessons),
                        len(course.lessons),
                        embedding,
                    ),
                )
        finally:
            conn.close()

    def add_course_content(self, chunks: list[CourseChunk]) -> None:
        self._require_ready()
        if not chunks:
            return
        vectors = self.embedder.embed([c.content for c in chunks])
        conn = self.db.connect()
        try:
            with conn:  # all chunks or none
                for chunk, vector in zip(chunks, vectors, strict=True):
                    conn.execute(
                        """
                        INSERT INTO course_content
                            (id, course_title, lesson_number, chunk_index, content, embedding)
                        VALUES (?, ?, ?, ?, ?, ?::vector)
                        ON CONFLICT (id) DO UPDATE SET
                            content = EXCLUDED.content, embedding = EXCLUDED.embedding
                        """,
                        (
                            chunk_id(chunk.course_title, chunk.chunk_index),
                            chunk.course_title,
                            chunk.lesson_number,
                            chunk.chunk_index,
                            chunk.content,
                            _vector_literal(vector),
                        ),
                    )
        finally:
            conn.close()
        self._invalidate_corpus_stats()

    def delete_course_content(self, course_title: str) -> None:
        conn = self.db.connect()
        try:
            with conn:
                conn.execute("DELETE FROM course_content WHERE course_title = ?", (course_title,))
        finally:
            conn.close()
        self._invalidate_corpus_stats()

    def clear_all_data(self) -> None:
        """Drop and recreate the tables, so a changed embedding size is picked up too."""
        conn = self.db.connect()
        try:
            with conn:
                conn.execute("DROP TABLE IF EXISTS course_content, course_catalog, vector_meta")
        finally:
            conn.close()
        self._create_schema()
        self.index_mismatch = None
        self._check_index_matches_embedder()
        self._invalidate_corpus_stats()

    # --- reads -----------------------------------------------------------------------------
    def get_existing_course_titles(self) -> list[str]:
        """All course titles; raises on a read failure (an empty list means "no courses")."""
        self._require_ready()
        conn = self.db.connect()
        try:
            return [r[0] for r in conn.execute("SELECT title FROM course_catalog ORDER BY title")]
        finally:
            conn.close()

    def get_course_count(self) -> int:
        return len(self.get_existing_course_titles())

    def _catalog_row(self, course_title: str, column: str) -> Any:
        conn = self.db.connect()
        try:
            row = conn.execute(
                f"SELECT {column} FROM course_catalog WHERE title = ?", (course_title,)
            ).fetchone()
        finally:
            conn.close()
        return row[0] if row else None

    def get_course_link(self, course_title: str) -> str | None:
        try:
            return self._catalog_row(course_title, "course_link")
        except Exception:
            logger.exception("Error getting link for course %r", course_title)
            return None

    def get_lesson_link(self, course_title: str, lesson_number: int) -> str | None:
        try:
            lessons_json = self._catalog_row(course_title, "lessons_json")
            for lesson in json.loads(lessons_json) if lessons_json else []:
                if lesson.get("lesson_number") == lesson_number:
                    return lesson.get("lesson_link")
            return None
        except Exception:
            logger.exception(
                "Error getting link for course %r lesson %s", course_title, lesson_number
            )
            return None
