from datetime import datetime, timezone
from typing import Any

from app.core.request_context import get_request_id


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def success_response(data: Any = None, message: str = "success", code: int = 0) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "data": {} if data is None else data,
        "timestamp": _utc_now_iso(),
        "requestId": get_request_id(),
    }


def error_response(code: int, message: str, data: Any = None) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "data": {} if data is None else data,
        "timestamp": _utc_now_iso(),
        "requestId": get_request_id(),
    }
