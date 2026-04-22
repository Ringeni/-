from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.crud.warning import WarningCRUD
from app.models.warning import Warning
from app.schemas.warning import WarningCreate, WarningOut
from app.services.websocket_manager import ws_manager


class WarningService:
    @staticmethod
    async def get_warning_by_id(db: AsyncSession, warning_id: int) -> Warning | None:
        return await WarningCRUD.get_by_id(db, warning_id)

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
        session_id: str | None = None,
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
        result = WarningOut.model_validate(warning)

        # Push WebSocket message if session_id provided
        if session_id:
            await ws_manager.broadcast_warning_created(
                session_id,
                warning_id=result.warning_id,
                level=result.warning_level,
                message=result.content,
                trigger_time=result.trigger_time,
            )

        return result

    @staticmethod
    async def confirm_warning(
        db: AsyncSession,
        warning_id: int,
        handler_id: int,
        handler_name: str,
    ) -> WarningOut | None:
        warning = await WarningCRUD.confirm(db, warning_id, handler_id, handler_name)
        if not warning:
            return None
        return WarningOut.model_validate(warning)

    @staticmethod
    async def close_warning(
        db: AsyncSession,
        warning_id: int,
        closed_by: int,
        closed_by_name: str,
    ) -> WarningOut | None:
        warning = await WarningCRUD.close(db, warning_id, closed_by, closed_by_name)
        if not warning:
            return None
        return WarningOut.model_validate(warning)
