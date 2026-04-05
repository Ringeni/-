from datetime import datetime

from pydantic import BaseModel, Field


class CameraCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    location: str | None = Field(default=None, max_length=256)
    stream_url: str = Field(min_length=1, max_length=512)
    status: int = Field(default=1, ge=0, le=2)


class CameraUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=128)
    location: str | None = Field(default=None, max_length=256)
    stream_url: str | None = Field(default=None, min_length=1, max_length=512)
    status: int | None = Field(default=None, ge=0, le=2)


class CameraOut(BaseModel):
    camera_id: int
    name: str
    location: str | None
    stream_url: str
    status: int
    create_time: datetime
    update_time: datetime

    model_config = {"from_attributes": True}
