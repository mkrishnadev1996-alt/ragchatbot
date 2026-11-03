def read_text_file(file_path: str) -> str:
    """Read a text file as UTF-8, replacing undecodable bytes if the file is not valid UTF-8"""
    try:
        with open(file_path, encoding="utf-8") as file:
            return file.read()
    except UnicodeDecodeError:
        # If UTF-8 fails, try with error handling
        with open(file_path, encoding="utf-8", errors="ignore") as file:
            return file.read()
