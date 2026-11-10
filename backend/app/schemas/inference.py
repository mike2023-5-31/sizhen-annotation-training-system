"""推理相关的请求模型（6.15 / 6.16 / 6.18）。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    model_path: str = Field(min_length=1)
    dataset_id: int = Field(gt=0)


class PreannotationRequest(PredictRequest):
    confidence_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
