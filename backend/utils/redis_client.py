"""Shared Redis client for sessions and rate limiting (optional: used when REDIS_URL is set)."""

from functools import lru_cache

import redis


@lru_cache(maxsize=4)
def get_redis(url: str) -> "redis.Redis":
    """One client (and connection pool) per URL for the whole process.

    Short timeouts keep a dead Redis from hanging requests; `decode_responses` gives str values.
    """
    return redis.Redis.from_url(
        url,
        decode_responses=True,
        socket_timeout=2,
        socket_connect_timeout=2,
        health_check_interval=30,
    )
