from datetime import datetime

from fastapi import APIRouter, Depends, Query, Path
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.core.deps import get_current_user
from app.core.errors import AppException, ErrorCode
from app.core.response import success_response
from app.models.user import User
from app.services.warning_service import WarningService

router = APIRouter(prefix="/warnings", tags=["warnings"])

# Status constants
STATUS_NEW = 0
STATUS_CONFIRMED = 1
STATUS_CLOSED = 2


@router.get("")
async def list_warnings(
    status: int | None = Query(default=None, ge=0, le=2),
    start_time: datetime | None = Query(default=None, alias="startTime"),
    end_time: datetime | None = Query(default=None, alias="endTime"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=200),
    _current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    items = await WarningService.list_warnings(
        db,
        status=status,
        start_time=start_time,
        end_time=end_time,
        offset=offset,
        limit=limit,
    )
    return success_response(data={"items": [item.model_dump(by_alias=True) for item in items]})


@router.post("/{warning_id}/confirm")
async def confirm_warning(
    warning_id: int = Path(..., description="告警ID"),
    _current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """确认告警：状态 0 (new) -> 1 (confirmed)"""
    warning = await WarningService.get_warning_by_id(db, warning_id)
    if not warning:
        raise AppException(code=ErrorCode.NOT_FOUND, message="warning not found", http_status=404)

    if warning.status != STATUS_NEW:
        raise AppException(
            code=ErrorCode.STATUS_CONFLICT,
            message=f"cannot confirm warning in status {warning.status}",
            http_status=409,
        )

    result = await WarningService.confirm_warning(
        db,
        warning_id=warning_id,
        handler_id=_current_user.user_id,
        handler_name=_current_user.username,
    )
    return success_response(data=result.model_dump(by_alias=True) if result else None)


@router.post("/{warning_id}/close")
async def close_warning(
    warning_id: int = Path(..., description="告警ID"),
    _current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """关闭告警：状态 1 (confirmed) -> 2 (closed)"""
    warning = await WarningService.get_warning_by_id(db, warning_id)
    if not warning:
        raise AppException(code=ErrorCode.NOT_FOUND, message="warning not found", http_status=404)

    if warning.status != STATUS_CONFIRMED:
        raise AppException(
            code=ErrorCode.STATUS_CONFLICT,
            message=f"cannot close warning in status {warning.status}",
            http_status=409,
        )

    result = await WarningService.close_warning(
        db,
        warning_id=warning_id,
        closed_by=_current_user.user_id,
        closed_by_name=_current_user.username,
    )
    return success_response(data=result.model_dump(by_alias=True) if result else None)
