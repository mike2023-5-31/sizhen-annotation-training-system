"""ONNX 转换/对比请求模型（6.19 / 6.20）。"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ConvertRequest(BaseModel):
    model_path: str = Field(min_length=1)


class CompareRequest(BaseModel):
    model_path: str = Field(min_length=1)
    onnx_name: str = Field(min_length=1)
    dataset_id: Optional[int] = Field(default=None, gt=0)
