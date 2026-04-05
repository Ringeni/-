from pydantic import BaseModel


class TargetCreate(BaseModel):
    appearance_features: dict | None = None


class TargetUpdate(BaseModel):
    appearance_features: dict | None = None


class TargetOut(BaseModel):
    target_id: int
    appearance_features: dict | None

    model_config = {"from_attributes": True}
