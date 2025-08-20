from __future__ import annotations

from typing import Any, Dict


def build_error_response(status: int, code: str, message: str, request_id: str | None = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "status": status,
        "code": code,
        "message": message,
    }
    if request_id:
        body["requestId"] = request_id
    return body


