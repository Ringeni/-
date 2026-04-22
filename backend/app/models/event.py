from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Event(Base):
    __tablename__ = "event"
    __table_args__ = (
        Index("ix_event_target_time", "target_id", "event_time"),
        Index("ix_event_stream_time_type", "stream_id", "event_time", "event_type"),
    )

    event_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    target_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    stream_id: Mapped[str] = mapped_column(String(64), nullable=False)
    event_type: Mapped[int] = mapped_column(Integer, nullable=False)
    event_level: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    event_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    related_warning_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)