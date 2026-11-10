"""数据集相关的请求/响应模型（6.1 / 6.2）。"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)


class DatasetRename(BaseModel):
    name: str = Field(min_length=1, max_length=255)


class DatasetOut(BaseModel):
    id: int
    name: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DatasetListItem(DatasetOut):
    image_count: Optional[int] = None
    annotated_count: Optional[int] = None


class PullRequest(BaseModel):
    target_dataset_id: int = Field(gt=0)
    start_time: datetime
    end_time: datetime
    quality: str = "合格"


class PullResult(BaseModel):
    pulled: int


class ImportSummary(BaseModel):
    imported: int
    skipped: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
