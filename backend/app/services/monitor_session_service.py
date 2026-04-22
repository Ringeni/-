from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.crud.camera import CameraCRUD


class MonitorSessionService:
    SESSION_TTL_SECONDS = 30 * 60  # 30 minutes default
    SESSION_EXPIRE_CHECK_THRESHOLD = 60  # seconds

    @staticmethod
    async def create_session(
        db: AsyncSession,
        redis_client: Redis,
        *,
        user_id: int,
        camera_ids: list[int],
        monitor_mode: str,
        token: str,
    ) -> dict:
        session_id = str(uuid4())
        now = datetime.now(timezone.utc)
        expire_at = now + timedelta(seconds=MonitorSessionService.SESSION_TTL_SECONDS)
        ws_channel = f"/ws/monitor?sessionId={session_id}&token={token}"

        camera_states: list[dict] = []
        for cid in camera_ids:
            camera = await CameraCRUD.get_by_id(db, cid)
            camera_states.append(
                {
                    "cameraId": cid,
                    "status": camera.status if camera else 0,
                }
            )

        session_payload = {
            "sessionId": session_id,
            "userId": user_id,
            "cameraIds": camera_ids,
            "monitorMode": monitor_mode,
            "wsChannel": ws_channel,
            "cameraStates": camera_states,
            "createTime": now.isoformat(),
            "lastActiveTime": now.isoformat(),
            "expireAt": expire_at.isoformat(),
        }

        key = f"monitor:session:{session_id}"
        await redis_client.set(key, json.dumps(session_payload, ensure_ascii=False), ex=MonitorSessionService.SESSION_TTL_SECONDS)
        return session_payload

    @staticmethod
    async def get_session(redis_client: Redis, session_id: str) -> dict | None:
        data = await redis_client.get(f"monitor:session:{session_id}")
        if not data:
            return None
        session = json.loads(data)
        # Lazy expiration check
        expire_at = datetime.fromisoformat(session["expireAt"])
        if datetime.now(timezone.utc) > expire_at:
            await redis_client.delete(f"monitor:session:{session_id}")
            return None
        return session

    @staticmethod
    async def touch_session(redis_client: Redis, session_id: str) -> dict | None:
        """Heartbeat: refresh session TTL and lastActiveTime"""
        session = await MonitorSessionService.get_session(redis_client, session_id)
        if not session:
            return None

        now = datetime.now(timezone.utc)
        expire_at = now + timedelta(seconds=MonitorSessionService.SESSION_TTL_SECONDS)

        session["lastActiveTime"] = now.isoformat()
        session["expireAt"] = expire_at.isoformat()

        key = f"monitor:session:{session_id}"
        await redis_client.set(key, json.dumps(session, ensure_ascii=False), ex=MonitorSessionService.SESSION_TTL_SECONDS)

        # Return remaining seconds
        remaining = int((expire_at - now).total_seconds())
        session["remainingSeconds"] = remaining

        return session

    @staticmethod
    async def delete_session(redis_client: Redis, session_id: str) -> bool:
        """Delete a session"""
        result = await redis_client.delete(f"monitor:session:{session_id}")
        return result > 0
