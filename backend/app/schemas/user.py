from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=6, max_length=128)
    role: str = Field(default="viewer", max_length=32)
    email: EmailStr | None = None
    phone: str | None = Field(default=None, max_length=32)


class UserUpdate(BaseModel):
    role: str | None = Field(default=None, max_length=32)
    email: EmailStr | None = None
    phone: str | None = Field(default=None, max_length=32)


class UserOut(BaseModel):
    user_id: int
    username: str
    role: str
    email: str | None
    phone: str | None
    last_login_time: datetime | None
    create_time: datetime

    model_config = {"from_attributes": True}
