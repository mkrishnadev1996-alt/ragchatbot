"""Central logging setup for the RAG backend.

Modules obtain a logger with ``logging.getLogger(__name__)``; ``setup_logging`` is called once
at application startup to attach the handlers and set the level.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Third-party loggers that are chatty even at INFO/DEBUG (per-HTTP-request lines, model
# download checks). Their warnings and errors still come through.
NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "openai",
    "urllib3",
    "huggingface_hub",
    "filelock",
    "sentence_transformers",
    "chromadb.telemetry",
)

# Handlers installed by setup_logging, so repeated calls replace them instead of stacking
_installed_handlers: list[logging.Handler] = []


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """
    Configure the root logger with a console handler and an optional rotating file handler.

    Args:
        level: Logging level name, e.g. "DEBUG", "INFO", "WARNING"
        log_file: Optional path to a log file; parent directories are created as needed

    Raises:
        ValueError: If ``level`` is not a valid logging level name
    """
    level_value = logging.getLevelNamesMapping().get(level.upper())
    if level_value is None:
        raise ValueError(f"Invalid log level: {level!r}")

    root = logging.getLogger()

    for handler in _installed_handlers:
        root.removeHandler(handler)
        handler.close()
    _installed_handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    _installed_handlers.append(console)

    if log_file:
        log_dir = os.path.dirname(os.path.abspath(log_file))
        os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        _installed_handlers.append(file_handler)

    for handler in _installed_handlers:
        root.addHandler(handler)
    root.setLevel(level_value)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
