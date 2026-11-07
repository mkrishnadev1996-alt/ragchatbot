from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from langsmith import traceable
from models import Source
from vector_store import SearchResults, VectorStore


@dataclass
class ToolResult:
    """What a tool returns: text for the LLM plus the sources to show in the UI.

    Sources travel with the result instead of living on the (shared) tool instance, so
    concurrent requests cannot see each other's sources.
    """

    text: str
    sources: list[Source] = field(default_factory=list)


class Tool(ABC):
    """Abstract base class for all tools"""

    @abstractmethod
    def get_tool_definition(self) -> dict[str, Any]:
        """Return OpenAI-compatible function tool definition for this tool"""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store

    def get_tool_definition(self) -> dict[str, Any]:
        """Return OpenAI-compatible function tool definition for this tool"""
        return {
            "type": "function",
            "function": {
                "name": "search_course_content",
                "description": (
                    "Search course materials with smart course name matching and lesson filtering"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for in the course content",
                        },
                        # Optional params are nullable: models sometimes send explicit nulls for
                        # them, and providers (e.g. Groq) reject the call if the schema says
                        # the value must be a string/integer. execute() treats None as "unset".
                        "course_name": {
                            "type": ["string", "null"],
                            "description": (
                                "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                            ),
                        },
                        "lesson_number": {
                            "type": ["integer", "null"],
                            "description": "Specific lesson number to search within (e.g. 1, 2, 3)",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    # Explicit parameters are kept so the LLM-facing schema and the signature stay in sync.
    # This intentionally narrows the base class's **kwargs signature.
    @traceable(name="search_course_content", run_type="tool")
    def execute(
        self, query: str, course_name: str | None = None, lesson_number: int | None = None
    ) -> ToolResult:
        """
        Execute the search tool with given parameters.

        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter

        Returns:
            Formatted search results (with their sources) or an error message
        """

        # Use the vector store's unified search interface
        results = self.store.search(
            query=query, course_name=course_name, lesson_number=lesson_number
        )

        # Handle errors
        if results.error:
            return ToolResult(results.error)

        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number is not None:
                filter_info += f" in lesson {lesson_number}"
            return ToolResult(f"No relevant content found{filter_info}.")

        # Format and return results
        return self._format_results(results)

    def _format_results(self, results: SearchResults) -> ToolResult:
        """Format search results with course and lesson context"""
        formatted = []
        sources: dict[str, Source] = {}  # label -> Source, so each citation appears once

        for doc, meta in zip(results.documents, results.metadata, strict=True):
            course_title = meta.get("course_title", "unknown")
            lesson_num = meta.get("lesson_number")

            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"

            # Track source for the UI, linking to the lesson (or the course) when known
            label = course_title
            if lesson_num is not None:
                label += f" - Lesson {lesson_num}"
            if label not in sources:
                sources[label] = Source(label=label, url=self._link(course_title, lesson_num))

            formatted.append(f"{header}\n{doc}")

        return ToolResult("\n\n".join(formatted), list(sources.values()))

    def _link(self, course_title: str, lesson_num: int | None) -> str | None:
        """The lesson link if there is one, otherwise the course link"""
        url = None
        if lesson_num is not None:
            url = self.store.get_lesson_link(course_title, lesson_num)
        return url or self.store.get_course_link(course_title)


class ToolManager:
    """Manages available tools for the AI"""

    def __init__(self):
        self.tools = {}

    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("function", {}).get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    def get_tool_definitions(self) -> list:
        """Get all tool definitions for OpenAI-compatible tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]

    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return ToolResult(f"Tool '{tool_name}' not found")

        return self.tools[tool_name].execute(**kwargs)
