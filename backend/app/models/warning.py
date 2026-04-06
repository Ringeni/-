from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Index, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Warning(Base):
    __tablename__ = "warning"
    __table_args__ = (
        Index("ix_warning_status_trigger_time", "status", "trigger_time"),
        Index("ix_warning_target_handler", "target_id", "handler_id"),
    )

    warning_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    target_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    warning_level: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    trigger_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    handler_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
