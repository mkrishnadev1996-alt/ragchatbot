def read_text_file(file_path: str) -> str:
    """Read a text file as UTF-8, replacing undecodable bytes if the file is not valid UTF-8"""
    try:
        with open(file_path, encoding="utf-8") as file:
            return file.read()
    except UnicodeDecodeError:
        # If UTF-8 fails, try with error handling
        with open(file_path, encoding="utf-8", errors="ignore") as file:
            return file.read()


def read_pdf_file(file_path: str) -> str:
    """Extract the text of every page of a PDF, one page per paragraph"""
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def read_docx_file(file_path: str) -> str:
    """Extract the paragraphs of a Word document, one per line"""
    from docx import Document

    return "\n".join(paragraph.text for paragraph in Document(file_path).paragraphs)


def read_document(file_path: str) -> str:
    """Read a .txt, .pdf or .docx course file as plain text (chosen by extension)"""
    extension = file_path.lower().rsplit(".", 1)[-1] if "." in file_path else ""
    if extension == "pdf":
        return read_pdf_file(file_path)
    if extension == "docx":
        return read_docx_file(file_path)
    return read_text_file(file_path)
