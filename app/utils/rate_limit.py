from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict


class SimpleRateLimiter:
    """In-process sliding-window rate limiter.

    Example: limiter = SimpleRateLimiter(max_calls=60, per_seconds=60)
    if limiter.allow():
        ...
    else:
        time.sleep(limiter.sleep_for_next_allowed())
    """

    def __init__(self, max_calls: int, per_seconds: float) -> None:
        self.max_calls = max_calls
        self.per_seconds = per_seconds
        self._events: Deque[float] = deque()

    def allow(self) -> bool:
        now = time.time()
        while self._events and (now - self._events[0]) > self.per_seconds:
            self._events.popleft()
        if len(self._events) < self.max_calls:
            self._events.append(now)
            return True
        return False

    def sleep_for_next_allowed(self) -> float:
        if not self._events:
            return 0.0
        now = time.time()
        oldest = self._events[0]
        elapsed = now - oldest
        remaining = self.per_seconds - elapsed
        return max(remaining, 0.01)


# Named singleton limiters per external dependency
limiters: Dict[str, SimpleRateLimiter] = {}


def get_limiter(name: str, max_calls: int, per_seconds: float) -> SimpleRateLimiter:
    key = f"{name}:{max_calls}:{per_seconds}"
    if key not in limiters:
        limiters[key] = SimpleRateLimiter(max_calls=max_calls, per_seconds=per_seconds)
    return limiters[key]


