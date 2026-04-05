from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate


class UserCRUD:
    @staticmethod
    async def create(db: AsyncSession, payload: UserCreate, password_hash: str) -> User:
        user = User(
            username=payload.username,
            password_hash=password_hash,
            role=payload.role,
            email=payload.email,
            phone=payload.phone,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        return user

    @staticmethod
    async def get_by_id(db: AsyncSession, user_id: int) -> User | None:
        result = await db.execute(select(User).where(User.user_id == user_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_username(db: AsyncSession, username: str) -> User | None:
        result = await db.execute(select(User).where(User.username == username))
        return result.scalar_one_or_none()

    @staticmethod
    async def list(db: AsyncSession, offset: int, limit: int) -> list[User]:
        result = await db.execute(select(User).offset(offset).limit(limit).order_by(User.user_id.desc()))
        return list(result.scalars().all())

    @staticmethod
    async def update(db: AsyncSession, user: User, payload: UserUpdate) -> User:
        for key, value in payload.model_dump(exclude_none=True).items():
            setattr(user, key, value)
        await db.commit()
        await db.refresh(user)
        return user

    @staticmethod
    async def update_last_login(db: AsyncSession, user: User) -> None:
        user.last_login_time = datetime.now(timezone.utc)
        await db.commit()

    @staticmethod
    async def delete(db: AsyncSession, user: User) -> None:
        await db.delete(user)
        await db.commit()
