"""数据集 CRUD 的 Pydantic schema（T6，需求 6.1）。

- ``DatasetCreate`` / ``DatasetRename``：入参，均仅含 ``name``；
- ``DatasetOut``：响应模型，对应 ``Dataset`` 表的四个展示字段。
  时间字段为 naive UTC（见 orm._utcnow），序列化后形如 ``2026-09-04T03:30:00``。
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DatasetNameIn(BaseModel):
    """新建/重命名共用的 name 校验：去首尾空白后至少 1 字符，最长 255（列宽）。"""

    name: str = Field(min_length=1, max_length=255, description="数据集名称（唯一）")

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
    """数据集响应：表字段全量回显（orm.Dataset 的四个列）。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    created_at: datetime
    updated_at: datetime
