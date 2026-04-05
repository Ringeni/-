from fastapi import APIRouter, Depends
from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.core.errors import AppException, ErrorCode
from app.core.redis_client import get_redis_client
from app.core.response import success_response

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def live() -> dict:
    return success_response(data={"status": "alive"})


@router.get("/ready")
async def ready(db: AsyncSession = Depends(get_db_session)) -> dict:
    redis_client: Redis = get_redis_client()
    try:
        await db.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        raise AppException(code=ErrorCode.DB_UNAVAILABLE, message="database unavailable", http_status=503) from exc

    try:
        await redis_client.ping()
    except RedisError as exc:
        raise AppException(code=ErrorCode.CACHE_UNAVAILABLE, message="cache unavailable", http_status=503) from exc

    return success_response(data={"status": "ready", "db": "ok", "redis": "ok"})
