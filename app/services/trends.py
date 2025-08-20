from __future__ import annotations

from typing import List

from pytrends.request import TrendReq
import time
from app.utils.rate_limit import get_limiter
from hashlib import sha1
from app.utils.cache import trends_cache


def fetch_related_queries(keywords: List[str], geo: str = "KR") -> dict[str, list[str]]:
    limiter = get_limiter("pytrends", max_calls=10, per_seconds=60.0)
    # simple retry with exponential backoff
    delay = 1.0
    last_exc: Exception | None = None
    for _ in range(3):
        try:
            while not limiter.allow():
                time.sleep(limiter.sleep_for_next_allowed())
            pytrends = TrendReq(hl="ko-KR", tz=540)
            pytrends.build_payload(keywords, timeframe="today 12-m", geo=geo)
            related = pytrends.related_queries()
            result: dict[str, list[str]] = {}
            for kw, tables in related.items():
                top_df = tables.get("top")
                if top_df is not None:
                    result[kw] = list(top_df["query"].head(10).astype(str))
                else:
                    result[kw] = []
            return result
        except Exception as e:
            last_exc = e
            time.sleep(delay)
            delay *= 2.0
    # Exceeded retries
    raise last_exc if last_exc else RuntimeError("pytrends unknown error")


def fetch_related_queries_cached(keywords: List[str], geo: str = "KR") -> dict[str, list[str]]:
    key = sha1(("|".join(sorted(keywords)) + f"|{geo}").encode("utf-8")).hexdigest()
    cached = trends_cache.get(key)
    if cached is not None:
        return cached
    result = fetch_related_queries(keywords, geo=geo)
    trends_cache.set(key, result, ttl_sec=3600)
    return result


