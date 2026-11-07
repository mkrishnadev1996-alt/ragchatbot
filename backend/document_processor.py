import os
import re

from models import Course, CourseChunk, Lesson
from utils.file_utils import read_document
from utils.text_utils import chunk_text as split_into_chunks


class DocumentProcessor:
    """Processes course documents and extracts structured information"""

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def read_file(self, file_path: str) -> str:
        """Read a .txt, .pdf or .docx file as plain text"""
        return read_document(file_path)

    def chunk_text(self, text: str) -> list[str]:
        """Split text into sentence-based chunks with overlap using config settings"""
        return split_into_chunks(text, self.chunk_size, self.chunk_overlap)

    def process_course_document(self, file_path: str) -> tuple[Course, list[CourseChunk]]:
        """
        Process a course document with expected format:
        Line 1: Course Title: [title]
        Line 2: Course Link: [url]
        Line 3: Course Instructor: [instructor]
        Following lines: Lesson markers and content
        """
        content = self.read_file(file_path)
        filename = os.path.basename(file_path)

        lines = content.strip().split("\n")

        # Extract course metadata from first three lines
        course_title = filename  # Default fallback
        course_link = None
        instructor_name = "Unknown"

        # Parse course title from first line
        if len(lines) >= 1 and lines[0].strip():
            title_match = re.match(r"^Course Title:\s*(.+)$", lines[0].strip(), re.IGNORECASE)
            if title_match:
                course_title = title_match.group(1).strip()
            else:
                course_title = lines[0].strip()

        # Parse remaining lines for course metadata
        for i in range(1, min(len(lines), 4)):  # Check first 4 lines for metadata
            line = lines[i].strip()
            if not line:
                continue

            # Try to match course link
            link_match = re.match(r"^Course Link:\s*(.+)$", line, re.IGNORECASE)
            if link_match:
                course_link = link_match.group(1).strip()
                continue

            # Try to match instructor
            instructor_match = re.match(r"^Course Instructor:\s*(.+)$", line, re.IGNORECASE)
            if instructor_match:
                instructor_name = instructor_match.group(1).strip()
                continue

        # Create course object with title as ID
        course = Course(
            title=course_title,
            course_link=course_link,
            instructor=instructor_name if instructor_name != "Unknown" else None,
        )

        # Process lessons and create chunks
        course_chunks: list[CourseChunk] = []
        current_lesson: int | None = None
        lesson_title: str = ""
        lesson_link: str | None = None
        lesson_content: list[str] = []

        def finish_lesson() -> None:
            """Turn the lesson collected so far into a Lesson entry plus chunks"""
            if current_lesson is None:
                return
            lesson_text = chr(10).join(lesson_content).strip()
            if not lesson_text:
                return
            course.lessons.append(
                Lesson(lesson_number=current_lesson, title=lesson_title, lesson_link=lesson_link)
            )
            # Every chunk carries the same course + lesson context prefix
            prefix = f"Course {course.title} Lesson {current_lesson} content: "
            for chunk in self.chunk_text(lesson_text):
                course_chunks.append(
                    CourseChunk(
                        content=prefix + chunk,
                        course_title=course.title,
                        lesson_number=current_lesson,
                        chunk_index=len(course_chunks),
                    )
                )

        # Start processing from line 4 (after metadata)
        start_index = 3
        if len(lines) > 3 and not lines[3].strip():
            start_index = 4  # Skip empty line after instructor

        i = start_index
        while i < len(lines):
            line = lines[i]

            # Check for lesson markers (e.g., "Lesson 0: Introduction")
            lesson_match = re.match(r"^Lesson\s+(\d+):\s*(.+)$", line.strip(), re.IGNORECASE)

            if lesson_match:
                finish_lesson()

                # Start new lesson
                current_lesson = int(lesson_match.group(1))
                lesson_title = lesson_match.group(2).strip()
                lesson_link = None

                # Check if next line is a lesson link
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    link_match = re.match(r"^Lesson Link:\s*(.+)$", next_line, re.IGNORECASE)
                    if link_match:
                        lesson_link = link_match.group(1).strip()
                        i += 1  # Skip the link line so it's not added to content

                lesson_content = []
            else:
                # Add line to current lesson content
                lesson_content.append(line)

            i += 1

        finish_lesson()  # the last lesson

        # If no lessons found, treat entire content as one document
        if not course_chunks and len(lines) > 2:
            remaining_content = "\n".join(lines[start_index:]).strip()
            if remaining_content:
                chunks = self.chunk_text(remaining_content)
                for chunk in chunks:
                    course_chunks.append(
                        CourseChunk(
                            content=f"Course {course.title} content: {chunk}",
                            course_title=course.title,
                            chunk_index=len(course_chunks),
                        )
                    )

        return course, course_chunks
