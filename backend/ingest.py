"""Ingest course documents into the vector store, then exit.

Run this as a one-off job (or before starting the server with INGEST_ON_STARTUP=false), so
web workers never write to the embedded Chroma store while serving requests.

    cd backend && uv run python ingest.py            # add courses not yet indexed
    cd backend && uv run python ingest.py --clear    # wipe the index and rebuild it
"""

import argparse
import logging
import sys

from config import config
from rag_system import RAGSystem
from utils.logging_config import setup_logging

logger = logging.getLogger("ingest")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clear", action="store_true", help="delete the existing index first")
    parser.add_argument("--path", default=config.DOCS_PATH, help="folder with course documents")
    args = parser.parse_args(argv)

    setup_logging(config.LOG_LEVEL, config.LOG_FILE)
    rag = RAGSystem(config)
    courses, chunks = rag.add_course_folder(args.path, clear_existing=args.clear)
    logger.info("Ingested %d new courses (%d chunks) from %s", courses, chunks, args.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
