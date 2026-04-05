from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.core.deps import get_current_user
from app.core.response import success_response
from app.models.user import User
from app.schemas.auth import CurrentUser, LoginRequest
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login")
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db_session)) -> dict:
    user = await AuthService.authenticate_user(db, payload.username, payload.password)
    tokens = AuthService.build_tokens(user.user_id)
    user_info = CurrentUser(user_id=user.user_id, username=user.username, role=user.role)
    return success_response(data={"token": tokens.model_dump(), "userInfo": user_info.model_dump()})


@router.get("/me")
async def me(current_user: User = Depends(get_current_user)) -> dict:
    user_info = CurrentUser(
        user_id=current_user.user_id,
        username=current_user.username,
        role=current_user.role,
    )
    return success_response(data=user_info.model_dump())
