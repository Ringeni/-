from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import WebSocket
from jose import JWTError, jwt

from app.core.config import get_settings
from app.core.errors import AppException, ErrorCode

_settings = get_settings()


def create_access_token(subject: str, extra_payload: dict[str, Any] | None = None) -> str:
    payload: dict[str, Any] = {"sub": subject}
    if extra_payload:
        payload.update(extra_payload)
    expire_at = datetime.now(timezone.utc) + timedelta(hours=_settings.jwt_access_expire_hours)
    payload["exp"] = int(expire_at.timestamp())
    payload["type"] = "access"
    return jwt.encode(payload, _settings.jwt_secret_key, algorithm=_settings.jwt_algorithm)


def create_refresh_token(subject: str) -> str:
    expire_at = datetime.now(timezone.utc) + timedelta(days=_settings.jwt_refresh_expire_days)
    payload: dict[str, Any] = {
        "sub": subject,
        "exp": int(expire_at.timestamp()),
        "type": "refresh",
    }
    return jwt.encode(payload, _settings.jwt_secret_key, algorithm=_settings.jwt_algorithm)


def verify_jwt_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, _settings.jwt_secret_key, algorithms=[_settings.jwt_algorithm])
    except JWTError as exc:
        raise AppException(code=ErrorCode.FORBIDDEN, message="invalid token", http_status=401) from exc


async def verify_websocket_token(websocket: WebSocket) -> dict[str, Any]:
    token = websocket.query_params.get("token")
    if not token:
        raise AppException(code=ErrorCode.FORBIDDEN, message="token is required", http_status=401)
    return verify_jwt_token(token)
