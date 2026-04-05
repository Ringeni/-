from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.core.deps import get_current_user
from app.core.response import success_response
from app.models.user import User
from app.services.track_service import TrackService

router = APIRouter(prefix="/tracks", tags=["tracks"])


@router.get("")
async def list_tracks(
    target_id: int = Query(gt=0, alias="targetId"),
    start_time: datetime = Query(alias="startTime"),
    end_time: datetime = Query(alias="endTime"),
    camera_ids: list[int] | None = Query(default=None, alias="cameraIds"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=200),
    _current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    result = await TrackService.query_trajectory(
        db,
        target_id=target_id,
        start_time=start_time,
        end_time=end_time,
        camera_ids=camera_ids,
        offset=offset,
        limit=limit,
    )
    return success_response(data={"trajectorySegments": [item.model_dump(mode="json") for item in result]})
