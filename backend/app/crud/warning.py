from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.warning import Warning
from app.schemas.warning import WarningCreate


class WarningCRUD:
    @staticmethod
    async def create(db: AsyncSession, payload: WarningCreate) -> Warning:
        warning = Warning(**payload.model_dump())
        db.add(warning)
        await db.commit()
        await db.refresh(warning)
        return warning

    @staticmethod
    async def list(
        db: AsyncSession,
        *,
        status: int | None,
        start_time: datetime | None,
        end_time: datetime | None,
        offset: int,
        limit: int,
    ) -> list[Warning]:
        stmt = select(Warning)
        if status is not None:
            stmt = stmt.where(Warning.status == status)
        if start_time is not None:
            stmt = stmt.where(Warning.trigger_time >= start_time)
        if end_time is not None:
            stmt = stmt.where(Warning.trigger_time <= end_time)

        stmt = stmt.order_by(Warning.trigger_time.desc()).offset(offset).limit(limit)
        result = await db.execute(stmt)
        return list(result.scalars().all())
