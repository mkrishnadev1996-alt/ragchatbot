import logging
import os
from typing import NamedTuple

from ai_generator import AIGenerator
from document_processor import DocumentProcessor
from input_guard import InputGuard
from langsmith import get_current_run_tree, traceable
from models import Course
from search_tools import CourseSearchTool, ToolManager
from session_manager import SessionManager
from vector_store import VectorStore

logger = logging.getLogger(__name__)


def _annotate_run(**metadata) -> None:
    """Attach metadata to the current LangSmith run; no-op when tracing is off"""
    run = get_current_run_tree()
    if run is not None:
        run.metadata.update(metadata)


class QueryResult(NamedTuple):
    """Outcome of RAGSystem.query"""

    answer: str
    sources: list[str]
    blocked: bool = False  # True when the input guard refused the message


class RAGSystem:
    """Main orchestrator for the Retrieval-Augmented Generation system"""

    def __init__(self, config):
        self.config = config

        # Initialize core components
        self.document_processor = DocumentProcessor(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        self.vector_store = VectorStore(
            config.CHROMA_PATH,
            config.EMBEDDING_MODEL,
            config.MAX_RESULTS,
            max_course_distance=getattr(config, "COURSE_MATCH_MAX_DISTANCE", None),
        )
        self.ai_generator = AIGenerator(
            config.LLM_API_KEY,
            config.LLM_BASE_URL,
            config.LLM_MODEL,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE,
            max_retries=config.LLM_MAX_RETRIES,
            timeout=getattr(config, "LLM_TIMEOUT", 60),
        )
        self.session_manager = SessionManager(
            config.MAX_HISTORY,
            ttl_seconds=getattr(config, "SESSION_TTL_SECONDS", 0),
            max_sessions=getattr(config, "MAX_SESSIONS", 0),
        )

        # Optional pre-LLM safety/scope check; off when no guard API key is configured
        guard_key = getattr(config, "GUARD_API_KEY", "")
        self.input_guard = (
            InputGuard(
                guard_key,
                config.GUARD_URL,
                config.GUARD_MODEL,
                config.GUARD_THRESHOLD,
                config.GUARD_TIMEOUT,
            )
            if guard_key
            else None
        )
        if not self.input_guard:
            logger.warning("GUARD_API_KEY not set; input guard is disabled")

        # Initialize search tools
        self.tool_manager = ToolManager()
        self.search_tool = CourseSearchTool(self.vector_store)
        self.tool_manager.register_tool(self.search_tool)

    def _ingest(self, course: Course, chunks: list) -> bool:
        """Store a course atomically; returns False if it was skipped.

        The catalog entry is what marks a course as ingested (startup skips known titles), so
        it is written last: content first, then metadata. If either step fails the content is
        removed again, and the course is retried on the next start instead of staying
        half-indexed. A file that produced no chunks is not added at all.
        """
        if not chunks:
            logger.warning("Course %r has no content chunks; skipping", course.title)
            return False
        try:
            self.vector_store.add_course_content(chunks)
            self.vector_store.add_course_metadata(course)
        except Exception:
            logger.exception("Failed to store course %r; rolling back its content", course.title)
            try:
                self.vector_store.delete_course_content(course.title)
            except Exception:
                logger.exception("Rollback of course %r failed", course.title)
            raise
        return True

    def add_course_document(self, file_path: str) -> tuple[Course | None, int]:
        """
        Add a single course document to the knowledge base.

        Args:
            file_path: Path to the course document

        Returns:
            Tuple of (Course object, number of chunks created)
        """
        try:
            course, course_chunks = self.document_processor.process_course_document(file_path)
            if not self._ingest(course, course_chunks):
                return None, 0
            return course, len(course_chunks)
        except Exception:
            logger.exception("Error processing course document %s", file_path)
            return None, 0

    def add_course_folder(self, folder_path: str, clear_existing: bool = False) -> tuple[int, int]:
        """
        Add all course documents from a folder.

        Args:
            folder_path: Path to folder containing course documents
            clear_existing: Whether to clear existing data first

        Returns:
            Tuple of (total courses added, total chunks created)
        """
        total_courses = 0
        total_chunks = 0

        # Clear existing data if requested
        if clear_existing:
            logger.info("Clearing existing data for fresh rebuild")
            self.vector_store.clear_all_data()

        if not os.path.exists(folder_path):
            logger.warning("Folder %s does not exist", folder_path)
            return 0, 0

        # Get existing course titles to avoid re-processing
        existing_course_titles = set(self.vector_store.get_existing_course_titles())

        # Process each file in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith((".pdf", ".docx", ".txt")):
                try:
                    # Check if this course might already exist
                    # We'll process the document to get the course ID, but only add if new
                    course, course_chunks = self.document_processor.process_course_document(
                        file_path
                    )

                    if course and course.title not in existing_course_titles:
                        # This is a new course - add it to the vector store
                        if not self._ingest(course, course_chunks):
                            continue
                        total_courses += 1
                        total_chunks += len(course_chunks)
                        logger.info(
                            "Added new course: %s (%d chunks)", course.title, len(course_chunks)
                        )
                        existing_course_titles.add(course.title)
                    elif course:
                        logger.info("Course already exists: %s - skipping", course.title)
                except Exception:
                    logger.exception("Error processing %s", file_name)

        return total_courses, total_chunks

    @traceable(name="rag_query", run_type="chain")
    def query(self, query: str, session_id: str | None = None) -> QueryResult:
        """
        Process a user query using the RAG system with tool-based search.

        Args:
            query: User's question
            session_id: Optional session ID for conversation context

        Returns:
            QueryResult(answer, sources, blocked)
        """
        logger.debug("RAG query (session=%s, %d chars)", session_id, len(query))

        # Get conversation history if session exists
        history = None
        if session_id:
            history = self.session_manager.get_conversation_history(session_id)

        # Reject harmful, jailbreak or out-of-scope input before any main-LLM call.
        # Blocked exchanges are not stored in the session history.
        if self.input_guard and not self.input_guard.is_safe(query, history):
            _annotate_run(blocked=True)
            return QueryResult(InputGuard.BLOCKED_MESSAGE, [], blocked=True)
        _annotate_run(blocked=False, model=self.config.LLM_MODEL)

        # Create prompt for the AI with clear instructions
        prompt = f"""Answer this question about course materials: {query}"""

        # Sources are collected in a per-request list, so concurrent requests stay isolated
        sources: list[str] = []
        response = self.ai_generator.generate_response(
            query=prompt,
            conversation_history=history,
            tools=self.tool_manager.get_tool_definitions(),
            tool_manager=self.tool_manager,
            sources_out=sources,
        )

        # Update conversation history
        if session_id:
            self.session_manager.add_exchange(session_id, query, response)

        # Return response with sources from tool searches
        return QueryResult(response, sources)

    def get_course_analytics(self) -> dict:
        """Get analytics about the course catalog"""
        return {
            "total_courses": self.vector_store.get_course_count(),
            "course_titles": self.vector_store.get_existing_course_titles(),
        }
