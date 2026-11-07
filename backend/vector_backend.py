"""The interface every vector store backend implements (ChromaDB, pgvector)."""

from typing import Protocol

from models import Course, CourseChunk
from search_results import SearchResults


class VectorStoreBackend(Protocol):
    def search(
        self,
        query: str,
        course_name: str | None = None,
        lesson_number: int | None = None,
        limit: int | None = None,
    ) -> SearchResults: ...

    def add_course_metadata(self, course: Course) -> None: ...

    def add_course_content(self, chunks: list[CourseChunk]) -> None: ...

    def delete_course_content(self, course_title: str) -> None: ...

    def clear_all_data(self) -> None: ...

    def get_existing_course_titles(self) -> list[str]: ...

    def get_course_count(self) -> int: ...

    def get_course_link(self, course_title: str) -> str | None: ...

    def get_lesson_link(self, course_title: str, lesson_number: int) -> str | None: ...
