"""Search result container shared by every vector store backend."""

from dataclasses import dataclass
from typing import Any


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
