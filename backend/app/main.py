import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.router import api_router
from app.api.ws import router as ws_router
from app.core.config import get_settings
from app.core.database import close_engine
from app.core.errors import AppException, ErrorCode
from app.core.logging_config import configure_logging
from app.core.redis_client import close_redis_client
from app.core.request_context import set_request_id
from app.core.response import error_response


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        set_request_id(request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging()
    logging.getLogger(__name__).info("application startup")
    yield
    await close_redis_client()
    await close_engine()
    logging.getLogger(__name__).info("application shutdown")


settings = get_settings()
app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(RequestContextMiddleware)
app.include_router(api_router, prefix="/api/v1")
app.include_router(ws_router)


@app.exception_handler(AppException)
async def app_exception_handler(_request: Request, exc: AppException) -> JSONResponse:
    return JSONResponse(status_code=exc.http_status, content=error_response(code=exc.code, message=exc.message))


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, _exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content=error_response(code=ErrorCode.INTERNAL_ERROR, message="internal server error"),
    )
