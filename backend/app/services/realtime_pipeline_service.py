from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.mock import AnalyzeResult
from app.schemas.monitor import SelectTargetOut
from app.services.warning_service import WarningService
from app.services.websocket_manager import ws_manager


class RealtimePipelineService:
    @staticmethod
    async def ingest_mock_result(
        db: AsyncSession,
        redis_client: Redis,
        *,
        session_id: str,
        analyze_result: AnalyzeResult,
    ) -> dict[str, Any]:
        warning_count = 0
        for track_result in analyze_result.results:
            camera_key = f"camera:activeTargets:{track_result.camera_id}"
            for detection in track_result.detections:
                target_key = f"target:lastViews:{detection.target_id}"
                view_payload = {
                    "cameraId": track_result.camera_id,
                    "box": detection.box,
                    "score": detection.score,
                    "timestamp": track_result.timestamp.isoformat(),
                }
                await redis_client.hset(target_key, str(track_result.camera_id), json.dumps(view_payload, ensure_ascii=False))
                await redis_client.expire(target_key, 2 * 60 * 60)

                await redis_client.hset(
                    camera_key,
                    str(detection.target_id),
                    json.dumps(
                        {
                            "targetId": detection.target_id,
                            "box": detection.box,
                            "score": detection.score,
                            "timestamp": track_result.timestamp.isoformat(),
                        },
                        ensure_ascii=False,
                    ),
                )
                await redis_client.expire(camera_key, 2 * 60 * 60)

                if detection.score < 0.93:
                    warning = await WarningService.create_warning_for_detection(
                        db,
                        target_id=detection.target_id,
                        warning_level=2,
                        content=f"mock warning for target {detection.target_id}",
                    )
                    warning_count += 1
                    await ws_manager.publish(
                        session_id,
                        {
                            "type": "warning.created",
                            "warningId": warning.warning_id,
                            "level": warning.warning_level,
                            "message": warning.content,
                            "triggerTime": warning.trigger_time.isoformat(),
                        },
                    )

        return {
            "cameraCount": len(analyze_result.results),
            "warningCount": warning_count,
            "modelTimeMs": analyze_result.model_time_ms,
        }

    @staticmethod
    async def select_target_by_point(
        redis_client: Redis,
        *,
        session_id: str,
        camera_id: int,
        x: int,
        y: int,
    ) -> SelectTargetOut:
        active_targets = await redis_client.hgetall(f"camera:activeTargets:{camera_id}")

        target_id = None
        current_box: list[float] = []
        for k, raw in active_targets.items():
            payload = json.loads(raw)
            box = payload.get("box", [])
            if len(box) == 4 and box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                target_id = int(payload.get("targetId", k))
                current_box = [float(v) for v in box]
                break

        if target_id is None and active_targets:
            first = json.loads(next(iter(active_targets.values())))
            target_id = int(first.get("targetId"))
            current_box = [float(v) for v in first.get("box", [])]

        if target_id is None:
            # 缓存未命中兜底
            target_id = 0
            current_box = []

        related_views_raw = await redis_client.hgetall(f"target:lastViews:{target_id}") if target_id else {}
        related_views = []
        for _cid, raw in related_views_raw.items():
            payload = json.loads(raw)
            related_views.append({"cameraId": payload.get("cameraId"), "box": payload.get("box", [])})

        selected = SelectTargetOut(
            targetId=target_id,
            currentBox=current_box,
            relatedViews=related_views,
        )

        await ws_manager.publish(
            session_id,
            {
                "type": "target.selected",
                "targetId": selected.target_id,
                "sourceCameraId": camera_id,
                "relatedViews": [{"cameraId": rv.camera_id, "box": rv.box} for rv in selected.related_views],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        return selected
