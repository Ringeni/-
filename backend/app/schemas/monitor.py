from datetime import datetime

from pydantic import BaseModel, Field


class MonitorSessionCreate(BaseModel):
    camera_ids: list[int] = Field(alias="cameraIds", min_length=1)
    monitor_mode: str = Field(default="realtime", alias="monitorMode")


class MonitorSessionOut(BaseModel):
    session_id: str = Field(alias="sessionId")
    ws_channel: str = Field(alias="wsChannel")
    camera_states: list[dict] = Field(alias="cameraStates")


class SelectTargetRequest(BaseModel):
    session_id: str = Field(alias="sessionId")
    camera_id: int = Field(alias="cameraId", gt=0)
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    timestamp: datetime | None = None


class RelatedView(BaseModel):
    camera_id: int = Field(alias="cameraId")
    box: list[float]


class SelectTargetOut(BaseModel):
    target_id: int = Field(alias="targetId")
    current_box: list[float] = Field(alias="currentBox")
    related_views: list[RelatedView] = Field(alias="relatedViews")
