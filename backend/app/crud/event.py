from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.event import Event
from app.schemas.event import EventCreate


class EventCRUD:
    @staticmethod
    async def create(db: AsyncSession, payload: EventCreate) -> Event:
        event = Event(**payload.model_dump())
        db.add(event)
        await db.commit()
        await db.refresh(event)
        return event

    @staticmethod
    async def list(
        db: AsyncSession,
        *,
        target_id: int | None,
        event_type: int | None,
        event_level: int | None,
        start_time: datetime | None,
        end_time: datetime | None,
        offset: int,
        limit: int,
    ) -> list[Event]:
        stmt = select(Event)
        if target_id is not None:
            stmt = stmt.where(Event.target_id == target_id)
        if event_type is not None:
            stmt = stmt.where(Event.event_type == event_type)
        if event_level is not None:
            stmt = stmt.where(Event.event_level == event_level)
        if start_time is not None:
            stmt = stmt.where(Event.event_time >= start_time)
        if end_time is not None:
            stmt = stmt.where(Event.event_time <= end_time)

        stmt = stmt.order_by(Event.event_time.desc()).offset(offset).limit(limit)
        result = await db.execute(stmt)
        return list(result.scalars().all())