from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.core.deps import get_current_user
from app.core.response import success_response
from app.models.user import User
from app.services.event_service import EventService

router = APIRouter(prefix="/events", tags=["events"])


@router.get("")
async def list_events(
    target_id: int | None = Query(default=None, alias="targetId"),
    event_type: int | None = Query(default=None, alias="eventType"),
    event_level: int | None = Query(default=None, ge=0, le=2, alias="eventLevel"),
    start_time: datetime | None = Query(default=None, alias="startTime"),
    end_time: datetime | None = Query(default=None, alias="endTime"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=200),
    _current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    items = await EventService.list_events(
        db,
        target_id=target_id,
        event_type=event_type,
        event_level=event_level,
        start_time=start_time,
        end_time=end_time,
        offset=offset,
        limit=limit,
    )
    return success_response(data={"items": [item.model_dump(by_alias=True) for item in items]})