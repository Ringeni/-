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
    async def get_by_id(db: AsyncSession, warning_id: int) -> Warning | None:
        stmt = select(Warning).where(Warning.warning_id == warning_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def confirm(
        db: AsyncSession,
        warning_id: int,
        handler_id: int,
        handler_name: str,
    ) -> Warning | None:
        warning = await WarningCRUD.get_by_id(db, warning_id)
        if not warning:
            return None
        warning.status = 1
        warning.handler_id = handler_id
        warning.handler_name = handler_name
        warning.handled_at = datetime.now()
        await db.commit()
        await db.refresh(warning)
        return warning

    @staticmethod
    async def close(
        db: AsyncSession,
        warning_id: int,
        closed_by: int,
        closed_by_name: str,
    ) -> Warning | None:
        warning = await WarningCRUD.get_by_id(db, warning_id)
        if not warning:
            return None
        warning.status = 2
        warning.closed_by = closed_by
        warning.closed_by_name = closed_by_name
        warning.closed_at = datetime.now()
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
        # Use composite index ix_warning_status_trigger_time when filtering by status
        stmt = select(Warning)
        if status is not None:
            stmt = stmt.where(Warning.status == status)
        if start_time is not None:
            stmt = stmt.where(Warning.trigger_time >= start_time)
        if end_time is not None:
            stmt = stmt.where(Warning.trigger_time <= end_time)

        # Force use of composite index when both status and time are present
        if status is not None and (start_time is not None or end_time is not None):
            stmt = stmt.with_hint(Warning, f"FORCE INDEX (ix_warning_status_trigger_time)")

        stmt = stmt.order_by(Warning.trigger_time.desc()).offset(offset).limit(limit)
        result = await db.execute(stmt)
        return list(result.scalars().all())
