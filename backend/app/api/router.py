from fastapi import APIRouter

from app.api.auth import router as auth_router
from app.api.cameras import router as cameras_router
from app.api.health import router as health_router
from app.api.targets import router as targets_router
from app.api.tracks import router as tracks_router
from app.api.trajectories import router as trajectories_router
from app.api.users import router as users_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(auth_router)
api_router.include_router(tracks_router)
api_router.include_router(cameras_router)
api_router.include_router(users_router)
api_router.include_router(targets_router)
api_router.include_router(trajectories_router)
