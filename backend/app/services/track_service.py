from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.crud.trajectory import TrajectoryCRUD
from app.schemas.trajectory import TrajectoryOut


class TrackService:
    @staticmethod
    def merge_segments(segments: list[TrajectoryOut]) -> list[TrajectoryOut]:
        if not segments:
            return []

        sorted_segments = sorted(segments, key=lambda item: item.start_time)
        merged: list[TrajectoryOut] = [sorted_segments[0]]

        for current in sorted_segments[1:]:
            prev = merged[-1]
            if current.stream_id == prev.stream_id and current.start_time <= prev.end_time:
                merged[-1] = TrajectoryOut(
                    trajectory_id=prev.trajectory_id,
                    target_id=prev.target_id,
                    stream_id=prev.stream_id,
                    start_time=prev.start_time,
                    end_time=max(prev.end_time, current.end_time),
                )
            else:
                merged.append(current)
        return merged

    @staticmethod
    async def query_trajectory(
        db: AsyncSession,
        *,
        target_id: int,
        start_time: datetime,
        end_time: datetime,
        camera_ids: list[int] | None,
        offset: int,
        limit: int,
    ) -> list[TrajectoryOut]:
        trajectories = await TrajectoryCRUD.query_by_target_time(
            db,
            target_id=target_id,
            start_time=start_time,
            end_time=end_time,
            stream_ids=camera_ids,
            offset=offset,
            limit=limit,
        )
        segment_out = [TrajectoryOut.model_validate(item) for item in trajectories]
        return TrackService.merge_segments(segment_out)
