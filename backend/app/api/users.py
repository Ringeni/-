from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.core.deps import get_current_user
from app.core.errors import AppException, ErrorCode
from app.core.response import success_response
from app.crud.user import UserCRUD
from app.models.user import User
from app.schemas.user import UserCreate, UserOut
from app.services.auth_service import AuthService

router = APIRouter(prefix="/users", tags=["users"])


@router.post("")
async def create_user(
    payload: UserCreate,
    _current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    existing = await UserCRUD.get_by_username(db, payload.username)
    if existing is not None:
        raise AppException(code=ErrorCode.STATUS_CONFLICT, message="username already exists", http_status=409)

    password_hash = AuthService.hash_password(payload.password)
    user = await UserCRUD.create(db, payload, password_hash)
    return success_response(data=UserOut.model_validate(user).model_dump(mode="json"))


@router.get("")
async def list_users(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=200),
    _current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    users = await UserCRUD.list(db, offset=offset, limit=limit)
    return success_response(data={"items": [UserOut.model_validate(item).model_dump(mode='json') for item in users]})


@router.get("/{user_id}")
async def get_user(
    user_id: int,
    _current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    user = await UserCRUD.get_by_id(db, user_id)
    if user is None:
        raise AppException(code=ErrorCode.NOT_FOUND, message="user not found", http_status=404)
    return success_response(data=UserOut.model_validate(user).model_dump(mode="json"))
