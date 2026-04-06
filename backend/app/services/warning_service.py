from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.crud.warning import WarningCRUD
from app.schemas.warning import WarningCreate, WarningOut


class WarningService:
    @staticmethod
    async def list_warnings(
        db: AsyncSession,
        *,
        status: int | None,
        start_time: datetime | None,
        end_time: datetime | None,
        offset: int,
        limit: int,
    ) -> list[WarningOut]:
        warnings = await WarningCRUD.list(
            db,
            status=status,
            start_time=start_time,
            end_time=end_time,
            offset=offset,
            limit=limit,
        )
        return [WarningOut.model_validate(item) for item in warnings]

    @staticmethod
    async def create_warning_for_detection(
        db: AsyncSession,
        *,
        target_id: int,
        warning_level: int,
        content: str,
    ) -> WarningOut:
        warning = await WarningCRUD.create(
            db,
            WarningCreate(
                target_id=target_id,
                warning_level=warning_level,
                trigger_time=datetime.now(timezone.utc),
                content=content,
                status=0,
            ),
        )
        return WarningOut.model_validate(warning)
