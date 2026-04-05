from sqlalchemy import Integer, JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Target(Base):
    __tablename__ = "target"

    target_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    appearance_features: Mapped[dict | None] = mapped_column(JSON, nullable=True)
