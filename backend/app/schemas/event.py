from datetime import datetime

from pydantic import BaseModel, Field


class EventOut(BaseModel):
    event_id: int = Field(alias="eventId")
    target_id: int = Field(alias="targetId")
    stream_id: str = Field(alias="streamId")
    event_type: int = Field(alias="eventType")
    event_level: int = Field(alias="eventLevel")
    event_time: datetime = Field(alias="eventTime")
    content: str
    related_warning_id: int | None = Field(default=None, alias="relatedWarningId")

    model_config = {"from_attributes": True, "populate_by_name": True}


class EventCreate(BaseModel):
    target_id: int
    stream_id: str
    event_type: int
    event_level: int = 1
    event_time: datetime
    content: str
    related_warning_id: int | None = None


class EventQuery(BaseModel):
    target_id: int | None = Field(default=None, alias="targetId")
    event_type: int | None = Field(default=None, alias="eventType")
    event_level: int | None = Field(default=None, ge=0, le=2, alias="eventLevel")
    start_time: datetime | None = Field(default=None, alias="startTime")
    end_time: datetime | None = Field(default=None, alias="endTime")
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=200)