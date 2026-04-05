from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.camera import Camera
from app.schemas.camera import CameraCreate, CameraUpdate


class CameraCRUD:
    @staticmethod
    async def create(db: AsyncSession, payload: CameraCreate) -> Camera:
        camera = Camera(**payload.model_dump())
        db.add(camera)
        await db.commit()
        await db.refresh(camera)
        return camera

    @staticmethod
    async def get_by_id(db: AsyncSession, camera_id: int) -> Camera | None:
        result = await db.execute(select(Camera).where(Camera.camera_id == camera_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def list(db: AsyncSession, offset: int, limit: int) -> list[Camera]:
        result = await db.execute(select(Camera).offset(offset).limit(limit).order_by(Camera.camera_id.desc()))
        return list(result.scalars().all())

    @staticmethod
    async def update(db: AsyncSession, camera: Camera, payload: CameraUpdate) -> Camera:
        for key, value in payload.model_dump(exclude_none=True).items():
            setattr(camera, key, value)
        await db.commit()
        await db.refresh(camera)
        return camera

    @staticmethod
    async def delete(db: AsyncSession, camera: Camera) -> None:
        await db.delete(camera)
        await db.commit()
