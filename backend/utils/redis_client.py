"""Shared Redis client for sessions and rate limiting (optional: used when REDIS_URL is set)."""

from functools import lru_cache

import redis


@lru_cache(maxsize=4)
def get_redis(url: str, timeout: float = 2.0, health_check_interval: int = 30) -> "redis.Redis":
    """One client (and connection pool) per settings for the whole process.

    Short timeouts keep a dead Redis from hanging requests; `decode_responses` gives str values.
    The defaults only apply to direct callers: the app passes REDIS_TIMEOUT and
    REDIS_HEALTH_CHECK_INTERVAL through `redis_from_config`.
    """
    return redis.Redis.from_url(
        url,
        decode_responses=True,
        socket_timeout=timeout,
        socket_connect_timeout=timeout,
        health_check_interval=health_check_interval,
    )


def redis_from_config(cfg) -> "redis.Redis":
    """The shared client for cfg.REDIS_URL, with the timeouts from the app's Config.

    Tolerates a config object without the tuning attributes (the defaults above are used).
    """
    return get_redis(
        cfg.REDIS_URL,
        getattr(cfg, "REDIS_TIMEOUT", 2.0),
        getattr(cfg, "REDIS_HEALTH_CHECK_INTERVAL", 30),
    )
