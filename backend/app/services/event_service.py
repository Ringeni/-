from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.crud.event import EventCRUD
from app.schemas.event import EventCreate, EventOut


class EventService:
    @staticmethod
    async def list_events(
        db: AsyncSession,
        *,
        target_id: int | None,
        event_type: int | None,
        event_level: int | None,
        start_time: datetime | None,
        end_time: datetime | None,
        offset: int,
        limit: int,
    ) -> list[EventOut]:
        events = await EventCRUD.list(
            db,
            target_id=target_id,
            event_type=event_type,
            event_level=event_level,
            start_time=start_time,
            end_time=end_time,
            offset=offset,
            limit=limit,
        )
        return [EventOut.model_validate(item) for item in events]

    @staticmethod
    async def create_event(
        db: AsyncSession,
        *,
        target_id: int,
        stream_id: str,
        event_type: int,
        event_level: int,
        content: str,
        related_warning_id: int | None = None,
    ) -> EventOut:
        event = await EventCRUD.create(
            db,
            EventCreate(
                target_id=target_id,
                stream_id=stream_id,
                event_type=event_type,
                event_level=event_level,
                event_time=datetime.now(),
                content=content,
                related_warning_id=related_warning_id,
            ),
        )
        return EventOut.model_validate(event)