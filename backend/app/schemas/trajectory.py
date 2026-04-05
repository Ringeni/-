from datetime import datetime

from pydantic import BaseModel, Field


class TrajectoryCreate(BaseModel):
    target_id: int = Field(gt=0)
    stream_id: int = Field(gt=0)
    start_time: datetime
    end_time: datetime


class TrajectoryUpdate(BaseModel):
    start_time: datetime | None = None
    end_time: datetime | None = None


class TrajectoryOut(BaseModel):
    trajectory_id: int
    target_id: int
    stream_id: int
    start_time: datetime
    end_time: datetime

    model_config = {"from_attributes": True}


class TrajectoryQuery(BaseModel):
    target_id: int = Field(gt=0)
    start_time: datetime
    end_time: datetime
    camera_ids: list[int] | None = None
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=200)
