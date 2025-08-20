from __future__ import annotations

import time
from typing import Any, Dict

import requests


def post_with_retry(
    url: str,
    payload: Dict[str, Any],
    *,
    timeout_sec: float = 10.0,
    retries: int = 3,
    backoff_sec: float = 0.5,
) -> None:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            requests.post(url, json=payload, timeout=timeout_sec)
            return
        except Exception as exc:  # noqa: BLE001 - best-effort fire-and-forget
            last_exc = exc
            time.sleep(backoff_sec * (2**attempt))
    # Final failure is intentionally swallowed to avoid crashing the caller
    # Optionally, add file logging here if needed
    _ = last_exc


