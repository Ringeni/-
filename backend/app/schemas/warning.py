from datetime import datetime

from pydantic import BaseModel, Field


class WarningOut(BaseModel):
    warning_id: int = Field(alias="warningId")
    event_id: int | None = Field(default=None, alias="eventId")
    target_id: int = Field(alias="targetId")
    warning_level: int = Field(alias="warningLevel")
    trigger_time: datetime = Field(alias="triggerTime")
    content: str
    status: int
    handler_id: int | None = Field(default=None, alias="handlerId")
    handler_name: str | None = Field(default=None, alias="handlerName")
    handled_at: datetime | None = Field(default=None, alias="handledAt")
    closed_at: datetime | None = Field(default=None, alias="closedAt")
    closed_by: int | None = Field(default=None, alias="closedBy")
    closed_by_name: str | None = Field(default=None, alias="closedByName")

    model_config = {"from_attributes": True, "populate_by_name": True}


class WarningCreate(BaseModel):
    event_id: int | None = None
    target_id: int
    warning_level: int = 1
    trigger_time: datetime
    content: str
    status: int = 0
    handler_id: int | None = None


class WarningQuery(BaseModel):
    status: int | None = Field(default=None, ge=0, le=2)
    start_time: datetime | None = Field(default=None, alias="startTime")
    end_time: datetime | None = Field(default=None, alias="endTime")
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=200)
