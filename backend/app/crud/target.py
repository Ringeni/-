from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.target import Target
from app.schemas.target import TargetCreate, TargetUpdate


class TargetCRUD:
    @staticmethod
    async def create(db: AsyncSession, payload: TargetCreate) -> Target:
        target = Target(**payload.model_dump())
        db.add(target)
        await db.commit()
        await db.refresh(target)
        return target

    @staticmethod
    async def get_by_id(db: AsyncSession, target_id: int) -> Target | None:
        result = await db.execute(select(Target).where(Target.target_id == target_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def list(db: AsyncSession, offset: int, limit: int) -> list[Target]:
        result = await db.execute(select(Target).offset(offset).limit(limit).order_by(Target.target_id.desc()))
        return list(result.scalars().all())

    @staticmethod
    async def update(db: AsyncSession, target: Target, payload: TargetUpdate) -> Target:
        for key, value in payload.model_dump(exclude_none=True).items():
            setattr(target, key, value)
        await db.commit()
        await db.refresh(target)
        return target

    @staticmethod
    async def delete(db: AsyncSession, target: Target) -> None:
        await db.delete(target)
        await db.commit()
