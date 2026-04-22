from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from typing import Any

from fastapi import WebSocket


class WebSocketConnectionManager:
    def __init__(self) -> None:
        self._session_connections: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._session_connections[session_id].add(websocket)

    def disconnect(self, session_id: str, websocket: WebSocket) -> None:
        conns = self._session_connections.get(session_id)
        if not conns:
            return
        conns.discard(websocket)
        if not conns:
            self._session_connections.pop(session_id, None)

    async def publish(self, session_id: str, payload: dict[str, Any]) -> None:
        for ws in list(self._session_connections.get(session_id, set())):
            await ws.send_json(payload)

    async def broadcast_to_session(self, session_id: str, message: dict[str, Any]) -> None:
        """Broadcast message to all connections in a session"""
        await self.publish(session_id, message)

    async def broadcast_warning_created(
        self,
        session_id: str,
        *,
        warning_id: int,
        level: int,
        message: str,
        trigger_time: datetime,
    ) -> None:
        """Broadcast warning.created message"""
        payload = {
            "type": "warning.created",
            "warningId": warning_id,
            "level": level,
            "message": message,
            "triggerTime": trigger_time.isoformat() if hasattr(trigger_time, "isoformat") else trigger_time,
        }
        await self.publish(session_id, payload)

    async def broadcast_target_selected(
        self,
        session_id: str,
        *,
        target_id: int,
        source_camera_id: str,
        related_views: list[dict],
    ) -> None:
        """Broadcast target.selected message"""
        payload = {
            "type": "target.selected",
            "targetId": target_id,
            "sourceCameraId": source_camera_id,
            "relatedViews": related_views,
        }
        await self.publish(session_id, payload)

    async def broadcast_monitor_update(
        self,
        session_id: str,
        *,
        camera_id: str,
        timestamp: datetime,
        targets: list[dict],
    ) -> None:
        """Broadcast monitor.update message"""
        payload = {
            "type": "monitor.update",
            "sessionId": session_id,
            "cameraId": camera_id,
            "timestamp": timestamp.isoformat() if hasattr(timestamp, "isoformat") else timestamp,
            "targets": targets,
        }
        await self.publish(session_id, payload)


ws_manager = WebSocketConnectionManager()
