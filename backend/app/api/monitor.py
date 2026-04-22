from fastapi import APIRouter, Depends
from fastapi.security import HTTPAuthorizationCredentials
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.core.deps import bearer_scheme, get_current_user
from app.core.errors import AppException, ErrorCode
from app.core.redis_client import get_redis_client
from app.core.response import success_response
from app.models.user import User
from app.schemas.monitor import MonitorSessionCreate, SelectTargetRequest
from app.services.monitor_session_service import MonitorSessionService
from app.services.realtime_pipeline_service import RealtimePipelineService
from app.services.tracking_engine_adapter_mock import TrackingEngineAdapterMock

router = APIRouter(prefix="/monitor", tags=["monitor"])


@router.post("/sessions")
async def create_monitor_session(
    payload: MonitorSessionCreate,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    if credentials is None or not credentials.credentials:
        raise AppException(code=ErrorCode.FORBIDDEN, message="missing token", http_status=401)

    session = await MonitorSessionService.create_session(
        db,
        get_redis_client(),
        user_id=current_user.user_id,
        camera_ids=payload.camera_ids,
        monitor_mode=payload.monitor_mode,
        token=credentials.credentials,
    )
    return success_response(
        data={
            "sessionId": session["sessionId"],
            "wsChannel": session["wsChannel"],
            "cameraStates": session["cameraStates"],
        }
    )


@router.post("/select-target")
async def select_target(
    payload: SelectTargetRequest,
    _current_user: User = Depends(get_current_user),
) -> dict:
    redis_client = get_redis_client()
    session = await MonitorSessionService.get_session(redis_client, payload.session_id)
    if session is None:
        raise AppException(code=ErrorCode.NOT_FOUND, message="session not found", http_status=404)

    selected = await RealtimePipelineService.select_target_by_point(
        redis_client,
        session_id=payload.session_id,
        camera_id=payload.camera_id,
        x=payload.x,
        y=payload.y,
    )
    await MonitorSessionService.touch_session(redis_client, payload.session_id)
    return success_response(data=selected.model_dump(by_alias=True))


@router.post("/mock-analyze")
async def mock_analyze(
    sessionId: str,
    _current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    redis_client: Redis = get_redis_client()
    session = await MonitorSessionService.get_session(redis_client, sessionId)
    if session is None:
        raise AppException(code=ErrorCode.NOT_FOUND, message="session not found", http_status=404)

    camera_ids = session.get("cameraIds", [])
    analyze_result = TrackingEngineAdapterMock.analyze_frames(camera_ids)
    summary = await RealtimePipelineService.ingest_mock_result(
        db,
        redis_client,
        session_id=sessionId,
        analyze_result=analyze_result,
    )
    await MonitorSessionService.touch_session(redis_client, sessionId)
    return success_response(data=summary)


@router.post("/sessions/{session_id}/heartbeat")
async def session_heartbeat(
    session_id: str,
    _current_user: User = Depends(get_current_user),
) -> dict:
    """Session heartbeat: refresh session TTL and get remaining time"""
    redis_client = get_redis_client()
    session = await MonitorSessionService.touch_session(redis_client, session_id)
    if not session:
        raise AppException(code=ErrorCode.NOT_FOUND, message="session not found or expired", http_status=404)
    return success_response(
        data={
            "sessionId": session["sessionId"],
            "lastActiveTime": session["lastActiveTime"],
            "expireAt": session["expireAt"],
            "remainingSeconds": session.get("remainingSeconds", 0),
        }
    )
