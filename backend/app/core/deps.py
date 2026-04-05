from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.core.errors import AppException, ErrorCode
from app.core.security import verify_jwt_token
from app.crud.user import UserCRUD
from app.models.user import User

bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db_session),
) -> User:
    if credentials is None or not credentials.credentials:
        raise AppException(code=ErrorCode.FORBIDDEN, message="missing token", http_status=401)

    payload = verify_jwt_token(credentials.credentials)
    if payload.get("type") != "access":
        raise AppException(code=ErrorCode.FORBIDDEN, message="invalid token type", http_status=401)

    subject = payload.get("sub")
    if not subject:
        raise AppException(code=ErrorCode.FORBIDDEN, message="invalid token subject", http_status=401)

    try:
        user_id = int(subject)
    except ValueError as exc:
        raise AppException(code=ErrorCode.FORBIDDEN, message="invalid token subject", http_status=401) from exc

    user = await UserCRUD.get_by_id(db, user_id)
    if user is None:
        raise AppException(code=ErrorCode.FORBIDDEN, message="user not found", http_status=401)
    return user
