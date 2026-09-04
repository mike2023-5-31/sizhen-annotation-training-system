"""数据集 CRUD 路由（T6，需求 6.1）。

前缀 ``/datasets``，挂载于 ``main.api_router``（整体 ``/api`` 前缀）后对外为
``/api/datasets``。T7/T8 的 ZIP 导入 / 内置库拉取将追加到本 router（plan 约定）。

错误映射：
- ``ValueError``（目标不存在）→ 404；
- ``IntegrityError``（name 唯一约束冲突）→ 409。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlalchemy.exc import IntegrityError

from app.models.orm import Dataset
from app.schemas.dataset import DatasetCreate, DatasetOut, DatasetRename
from app.services import dataset_service

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("", response_model=DatasetOut, status_code=201)
def create_dataset(payload: DatasetCreate) -> Dataset:
    """新建数据集。"""
    try:
        return dataset_service.create_dataset(payload.name)
    except IntegrityError:
        raise HTTPException(status_code=409, detail=f"数据集名称「{payload.name}」已存在")


@router.get("", response_model=list[DatasetOut])
def list_datasets() -> list[Dataset]:
    """全部数据集（id 升序）。"""
    return dataset_service.list_datasets()


@router.patch("/{dataset_id}", response_model=DatasetOut)
def rename_dataset(dataset_id: int, payload: DatasetRename) -> Dataset:
    """重命名数据集。"""
    try:
        return dataset_service.rename_dataset(dataset_id, payload.name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except IntegrityError:
        raise HTTPException(status_code=409, detail=f"数据集名称「{payload.name}」已存在")


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: int) -> dict[str, bool]:
    """删除数据集（仅 DB 行；磁盘图片文件保留）。"""
    try:
        dataset_service.delete_dataset(dataset_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"ok": True}
