from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, BigInteger
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Trajectory(Base):
    __tablename__ = "trajectory"
    __table_args__ = (
        Index("ix_trajectory_target_start_time", "target_id", "start_time"),
        Index("ix_trajectory_stream_start_time", "stream_id", "start_time"),
    )

    trajectory_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    target_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    stream_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    start_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    end_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
