import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langsmith import traceable
from models import Course, CourseChunk

logger = logging.getLogger(__name__)


@dataclass
class SearchResults:
    """Container for search results with metadata"""

    documents: list[str]
    metadata: list[dict[str, Any]]
    distances: list[float]
    error: str | None = None

    @classmethod
    def from_chroma(cls, chroma_results: dict) -> "SearchResults":
        """Create SearchResults from ChromaDB query results"""
        return cls(
            documents=chroma_results["documents"][0] if chroma_results["documents"] else [],
            metadata=chroma_results["metadatas"][0] if chroma_results["metadatas"] else [],
            distances=chroma_results["distances"][0] if chroma_results["distances"] else [],
        )

    @classmethod
    def empty(cls, error_msg: str) -> "SearchResults":
        """Create empty results with error message"""
        return cls(documents=[], metadata=[], distances=[], error=error_msg)

    def is_empty(self) -> bool:
        """Check if results are empty"""
        return len(self.documents) == 0


def _retrieval_output(results: "SearchResults") -> dict[str, Any]:
    """Shape results for LangSmith's retriever view (page_content + metadata per document)"""
    # Tolerant of None metadata / uneven lists: tracing must never break a search
    documents = [
        {"page_content": doc, "metadata": {**(meta or {}), "distance": dist}}
        for doc, meta, dist in zip(  # noqa: B905
            results.documents, results.metadata, results.distances
        )
    ]
    return {"documents": documents, "error": results.error}


class VectorStore:
    """Vector storage using ChromaDB for course content and metadata"""

    def __init__(
        self,
        chroma_path: str,
        embedding_model: str,
        max_results: int = 5,
        max_course_distance: float | None = None,
    ):
        self.max_results = max_results
        # Course-name matches farther than this (L2 distance) count as "no such course".
        # None disables the check.
        self.max_course_distance = max_course_distance
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=chroma_path, settings=Settings(anonymized_telemetry=False)
        )

        # Set up sentence transformer embedding function
        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name=embedding_model)

        # Create collections for different types of data
        self.course_catalog = self._create_collection("course_catalog")  # Course titles/instructors
        self.course_content = self._create_collection("course_content")  # Actual course material

    def _create_collection(self, name: str):
        """Create or get a ChromaDB collection"""
        # chromadb's type stubs for EmbeddingFunction are narrower than the runtime protocol
        # that SentenceTransformerEmbeddingFunction implements, so mypy rejects this argument.
        return self.client.get_or_create_collection(
            name=name,
            embedding_function=self.embedding_function,  # type: ignore[arg-type]
        )

    @traceable(name="vector_store.search", run_type="retriever", process_outputs=_retrieval_output)
    def search(
        self,
        query: str,
        course_name: str | None = None,
        lesson_number: int | None = None,
        limit: int | None = None,
    ) -> SearchResults:
        """
        Main search interface that handles course resolution and content search.

        Args:
            query: What to search for in course content
            course_name: Optional course name/title to filter by
            lesson_number: Optional lesson number to filter by
            limit: Maximum results to return

        Returns:
            SearchResults object with documents and metadata
        """
        # Step 1: Resolve course name if provided
        course_title = None
        if course_name:
            course_title = self._resolve_course_name(course_name)
            if not course_title:
                return SearchResults.empty(f"No course found matching '{course_name}'")

        # Step 2: Build filter for content search
        filter_dict = self._build_filter(course_title, lesson_number)

        # Step 3: Search course content
        # Use provided limit or fall back to configured max_results
        search_limit = limit if limit is not None else self.max_results

        try:
            results = self.course_content.query(
                query_texts=[query], n_results=search_limit, where=filter_dict
            )
            return SearchResults.from_chroma(results)
        except Exception as e:
            logger.exception("Chroma content search failed (course_title=%s)", course_title)
            return SearchResults.empty(f"Search error: {str(e)}")

    def _resolve_course_name(self, course_name: str) -> str | None:
        """Use vector search to find best matching course by name"""
        try:
            results = self.course_catalog.query(query_texts=[course_name], n_results=1)

            if results["documents"][0] and results["metadatas"][0]:
                distances = results.get("distances") or [[]]
                if (
                    self.max_course_distance is not None
                    and distances[0]
                    and distances[0][0] > self.max_course_distance
                ):
                    logger.info(
                        "Course name %r too far from any course (distance %.3f)",
                        course_name,
                        distances[0][0],
                    )
                    return None
                # Return the title (which is now the ID)
                return results["metadatas"][0][0]["title"]
        except Exception:
            logger.exception("Error resolving course name %r", course_name)

        return None

    def _build_filter(self, course_title: str | None, lesson_number: int | None) -> dict | None:
        """Build ChromaDB filter from search parameters"""
        if not course_title and lesson_number is None:
            return None

        # Handle different filter combinations
        if course_title and lesson_number is not None:
            return {"$and": [{"course_title": course_title}, {"lesson_number": lesson_number}]}

        if course_title:
            return {"course_title": course_title}

        return {"lesson_number": lesson_number}

    def add_course_metadata(self, course: Course):
        """Add course information to the catalog for semantic search"""
        course_text = course.title

        # Build lessons metadata and serialize as JSON string
        lessons_metadata = []
        for lesson in course.lessons:
            lessons_metadata.append(
                {
                    "lesson_number": lesson.lesson_number,
                    "lesson_title": lesson.title,
                    "lesson_link": lesson.lesson_link,
                }
            )

        self.course_catalog.add(
            documents=[course_text],
            metadatas=[
                {
                    "title": course.title,
                    "instructor": course.instructor,
                    "course_link": course.course_link,
                    "lessons_json": json.dumps(lessons_metadata),  # Serialize as JSON string
                    "lesson_count": len(course.lessons),
                }
            ],
            ids=[course.title],
        )

    def add_course_content(self, chunks: list[CourseChunk]):
        """Add course content chunks to the vector store"""
        if not chunks:
            return

        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "course_title": chunk.course_title,
                "lesson_number": chunk.lesson_number,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
        ]
        ids = [self._chunk_id(chunk) for chunk in chunks]

        self.course_content.add(documents=documents, metadatas=metadatas, ids=ids)

    @staticmethod
    def _chunk_id(chunk: CourseChunk) -> str:
        """Unique chunk id: a hash of the title (so "A B" and "A_B" differ) plus the index"""
        digest = hashlib.sha256(chunk.course_title.encode("utf-8")).hexdigest()[:16]
        return f"{digest}_{chunk.chunk_index}"

    def delete_course_content(self, course_title: str) -> None:
        """Remove every content chunk of a course (used to roll back a failed ingestion)"""
        self.course_content.delete(where={"course_title": course_title})

    def clear_all_data(self):
        """Clear all data from both collections"""
        try:
            self.client.delete_collection("course_catalog")
            self.client.delete_collection("course_content")
            # Recreate collections
            self.course_catalog = self._create_collection("course_catalog")
            self.course_content = self._create_collection("course_content")
        except Exception:
            logger.exception("Error clearing course data")

    def get_existing_course_titles(self) -> list[str]:
        """Get all existing course titles from the vector store.

        Raises if the catalog cannot be read: an empty list means "no courses", and returning
        it on a failed read would make startup re-insert courses that already exist.
        """
        # include=[] returns ids only; the catalog documents are not needed here
        results = self.course_catalog.get(include=[])
        return list(results["ids"]) if results and "ids" in results else []

    def get_course_count(self) -> int:
        """Get the total number of courses in the vector store (raises on a read failure)"""
        return len(self.get_existing_course_titles())

    def get_course_link(self, course_title: str) -> str | None:
        """Get course link for a given course title"""
        try:
            # Get course by ID (title is the ID)
            results = self.course_catalog.get(ids=[course_title])
            if results and "metadatas" in results and results["metadatas"]:
                metadata = results["metadatas"][0]
                return metadata.get("course_link")
            return None
        except Exception:
            logger.exception("Error getting link for course %r", course_title)
            return None

    def get_lesson_link(self, course_title: str, lesson_number: int) -> str | None:
        """Get lesson link for a given course title and lesson number"""
        try:
            # Get course by ID (title is the ID)
            results = self.course_catalog.get(ids=[course_title])
            if results and "metadatas" in results and results["metadatas"]:
                metadata = results["metadatas"][0]
                lessons_json = metadata.get("lessons_json")
                if lessons_json:
                    lessons = json.loads(lessons_json)
                    # Find the lesson with matching number
                    for lesson in lessons:
                        if lesson.get("lesson_number") == lesson_number:
                            return lesson.get("lesson_link")
            return None
        except Exception:
            logger.exception(
                "Error getting link for course %r lesson %s", course_title, lesson_number
            )
            return None
