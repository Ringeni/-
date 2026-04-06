from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.security import verify_websocket_token
from app.services.monitor_session_service import MonitorSessionService
from app.services.websocket_manager import ws_manager
from app.core.redis_client import get_redis_client

router = APIRouter(tags=["ws"])


@router.websocket("/ws/monitor")
async def ws_monitor(websocket: WebSocket) -> None:
    await verify_websocket_token(websocket)
    session_id = websocket.query_params.get("sessionId", "")
    if not session_id:
        await websocket.close(code=1008, reason="sessionId is required")
        return

    session = await MonitorSessionService.get_session(get_redis_client(), session_id)
    if session is None:
        await websocket.close(code=1008, reason="session not found")
        return

    await ws_manager.connect(session_id, websocket)
    await ws_manager.publish(session_id, {"type": "ws.connected", "sessionId": session_id})

    try:
        while True:
            _ = await websocket.receive_text()
            await MonitorSessionService.touch_session(get_redis_client(), session_id)
    except WebSocketDisconnect:
        ws_manager.disconnect(session_id, websocket)
