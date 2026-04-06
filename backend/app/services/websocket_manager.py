from __future__ import annotations

from collections import defaultdict
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


ws_manager = WebSocketConnectionManager()
