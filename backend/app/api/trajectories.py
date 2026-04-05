from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.core.deps import get_current_user
from app.core.errors import AppException, ErrorCode
from app.core.response import success_response
from app.crud.trajectory import TrajectoryCRUD
from app.models.user import User
from app.schemas.trajectory import TrajectoryCreate, TrajectoryOut

router = APIRouter(prefix="/trajectories", tags=["trajectories"])


@router.post("")
async def create_trajectory(
    payload: TrajectoryCreate,
    _current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    trajectory = await TrajectoryCRUD.create(db, payload)
    return success_response(data=TrajectoryOut.model_validate(trajectory).model_dump(mode="json"))


@router.get("")
async def list_trajectories(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=200),
    _current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    trajectories = await TrajectoryCRUD.list(db, offset=offset, limit=limit)
    return success_response(
        data={"items": [TrajectoryOut.model_validate(item).model_dump(mode='json') for item in trajectories]}
    )


@router.get("/{trajectory_id}")
async def get_trajectory(
    trajectory_id: int,
    _current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    trajectory = await TrajectoryCRUD.get_by_id(db, trajectory_id)
    if trajectory is None:
        raise AppException(code=ErrorCode.NOT_FOUND, message="trajectory not found", http_status=404)
    return success_response(data=TrajectoryOut.model_validate(trajectory).model_dump(mode="json"))
