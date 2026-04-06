from datetime import datetime

from pydantic import BaseModel, Field


class Detection(BaseModel):
    box: list[float]
    class_name: str = Field(alias="className")
    score: float
    target_id: int = Field(alias="targetId")
    assoc_score: float | None = Field(default=None, alias="assocScore")
    feature_digest: str | None = Field(default=None, alias="featureDigest")


class TrackResult(BaseModel):
    camera_id: int = Field(alias="cameraId")
    timestamp: datetime
    detections: list[Detection]


class AnalyzeResult(BaseModel):
    results: list[TrackResult]
    model_time_ms: int = Field(alias="modelTimeMs")
    debug_summary: dict | None = Field(default=None, alias="debugSummary")
