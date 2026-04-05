from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.trajectory import Trajectory
from app.schemas.trajectory import TrajectoryCreate, TrajectoryUpdate


class TrajectoryCRUD:
    @staticmethod
    async def create(db: AsyncSession, payload: TrajectoryCreate) -> Trajectory:
        trajectory = Trajectory(**payload.model_dump())
        db.add(trajectory)
        await db.commit()
        await db.refresh(trajectory)
        return trajectory

    @staticmethod
    async def get_by_id(db: AsyncSession, trajectory_id: int) -> Trajectory | None:
        result = await db.execute(select(Trajectory).where(Trajectory.trajectory_id == trajectory_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def list(db: AsyncSession, offset: int, limit: int) -> list[Trajectory]:
        result = await db.execute(select(Trajectory).offset(offset).limit(limit).order_by(Trajectory.trajectory_id.desc()))
        return list(result.scalars().all())

    @staticmethod
    async def update(db: AsyncSession, trajectory: Trajectory, payload: TrajectoryUpdate) -> Trajectory:
        for key, value in payload.model_dump(exclude_none=True).items():
            setattr(trajectory, key, value)
        await db.commit()
        await db.refresh(trajectory)
        return trajectory

    @staticmethod
    async def delete(db: AsyncSession, trajectory: Trajectory) -> None:
        await db.delete(trajectory)
        await db.commit()

    @staticmethod
    async def query_by_target_time(
        db: AsyncSession,
        *,
        target_id: int,
        start_time: datetime,
        end_time: datetime,
        stream_ids: list[int] | None,
        offset: int,
        limit: int,
    ) -> list[Trajectory]:
        stmt = (
            select(Trajectory)
            .where(Trajectory.target_id == target_id)
            .where(Trajectory.start_time <= end_time)
            .where(Trajectory.end_time >= start_time)
        )
        if stream_ids:
            stmt = stmt.where(Trajectory.stream_id.in_(stream_ids))

        stmt = stmt.order_by(Trajectory.start_time.asc()).offset(offset).limit(limit)
        result = await db.execute(stmt)
        return list(result.scalars().all())
