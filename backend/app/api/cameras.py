from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.core.deps import get_current_user
from app.core.errors import AppException, ErrorCode
from app.core.response import success_response
from app.crud.camera import CameraCRUD
from app.models.user import User
from app.schemas.camera import CameraCreate, CameraOut

router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.post("")
async def create_camera(
    payload: CameraCreate,
    _current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    camera = await CameraCRUD.create(db, payload)
    return success_response(data=CameraOut.model_validate(camera).model_dump(mode="json"))


@router.get("")
async def list_cameras(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=200),
    _current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    cameras = await CameraCRUD.list(db, offset=offset, limit=limit)
    return success_response(data={"items": [CameraOut.model_validate(item).model_dump(mode='json') for item in cameras]})


@router.get("/{camera_id}")
async def get_camera(
    camera_id: int,
    _current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    camera = await CameraCRUD.get_by_id(db, camera_id)
    if camera is None:
        raise AppException(code=ErrorCode.NOT_FOUND, message="camera not found", http_status=404)
    return success_response(data=CameraOut.model_validate(camera).model_dump(mode="json"))
