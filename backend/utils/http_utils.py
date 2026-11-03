from collections.abc import Mapping
from typing import Any


def parse_retry_after(headers: Mapping[str, Any] | None) -> float | None:
    """Read a Retry-After header as a number of seconds.

    Returns None if the header is missing, or is not a plain number (HTTP-date values are
    not parsed).
    """
    value = (headers or {}).get("retry-after")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
