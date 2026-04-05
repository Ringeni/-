from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.core.deps import get_current_user
from app.core.errors import AppException, ErrorCode
from app.core.response import success_response
from app.crud.target import TargetCRUD
from app.models.user import User
from app.schemas.target import TargetCreate, TargetOut

router = APIRouter(prefix="/targets", tags=["targets"])


@router.post("")
async def create_target(
    payload: TargetCreate,
    _current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    target = await TargetCRUD.create(db, payload)
    return success_response(data=TargetOut.model_validate(target).model_dump(mode="json"))


@router.get("")
async def list_targets(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=200),
    _current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    targets = await TargetCRUD.list(db, offset=offset, limit=limit)
    return success_response(data={"items": [TargetOut.model_validate(item).model_dump(mode='json') for item in targets]})


@router.get("/{target_id}")
async def get_target(
    target_id: int,
    _current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    target = await TargetCRUD.get_by_id(db, target_id)
    if target is None:
        raise AppException(code=ErrorCode.NOT_FOUND, message="target not found", http_status=404)
    return success_response(data=TargetOut.model_validate(target).model_dump(mode="json"))
