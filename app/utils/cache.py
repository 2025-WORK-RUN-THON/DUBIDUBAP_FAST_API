from __future__ import annotations

import time
from typing import Any, Dict, Tuple


class TTLCache:
    def __init__(self, default_ttl_sec: float = 600.0):
        self._store: Dict[str, Tuple[float, Any]] = {}
        self._default_ttl = float(default_ttl_sec)

    def get(self, key: str) -> Any | None:
        now = time.time()
        item = self._store.get(key)
        if not item:
            return None
        expires_at, value = item
        if now >= expires_at:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any, ttl_sec: float | None = None) -> None:
        ttl = float(ttl_sec) if ttl_sec is not None else self._default_ttl
        self._store[key] = (time.time() + ttl, value)

    def clear(self) -> None:
        self._store.clear()


trends_cache = TTLCache(default_ttl_sec=3600)


