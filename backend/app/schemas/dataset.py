"""数据集 CRUD 的 Pydantic schema（T6，需求 6.1）。

- ``DatasetCreate`` / ``DatasetRename``：接收名称和可选的四诊类型；
- ``DatasetOut``：数据集详情。历史数据的四诊类型可以未知。
  时间字段为 naive UTC（见 orm._utcnow），序列化后形如 ``2026-09-04T03:30:00``。
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DatasetNameIn(BaseModel):
    """新建/重命名共用的 name 校验：去首尾空白后至少 1 字符，最长 255（列宽）。"""

    name: str = Field(min_length=1, max_length=255, description="数据集名称（唯一）")
    modality: Literal["舌", "面", "脉", "声音", "混合"] | None = None

    @field_validator("name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("数据集名称不能为空")
        return v


class DatasetCreate(DatasetNameIn):
    """POST /api/datasets 请求体。"""


class DatasetRename(DatasetNameIn):
    """PATCH /api/datasets/{id} 请求体。"""


class DatasetOut(BaseModel):
    """数据集响应；历史数据可保留空的四诊类型。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    modality: str | None = None
    created_at: datetime
    updated_at: datetime


class DatasetListItem(DatasetOut):
    """列表可选返回标注进度所需的统计，默认保持原有字段。"""

    image_count: int | None = None
    annotated_count: int | None = None


class ImportBatchOut(BaseModel):
    id: int
    actor_id: str | None
    created_at: datetime
    source: Literal["zip", "pull"]
    filename: str | None
    start_time: datetime | None
    end_time: datetime | None
    quality: str | None
    imported: int
    skipped: int
    errors: int


class ImportHistoryOut(BaseModel):
    items: list[ImportBatchOut]
    has_more: bool


class ImportBatchListItem(ImportBatchOut):
    """全局导入记录：带上数据集归属，供导入记录中心使用。"""

    dataset_id: int
    dataset_name: str


class ImportBatchListOut(BaseModel):
    items: list[ImportBatchListItem]
    has_more: bool
