"""数据集 CRUD 路由（T6，需求 6.1）+ 内置库拉取（T8，需求 6.2b）+
ZIP 导出（T11，需求 6.5）。

前缀 ``/datasets``，挂载于 ``main.api_router``（整体 ``/api`` 前缀）后对外为
``/api/datasets``。T7 的 ZIP 导入追加在本 router（plan 约定）；T11 的导出
同样追加于此。

错误映射：
- ``ValueError``（目标不存在）→ 404；
- ``IntegrityError``（name 唯一约束冲突）→ 409；
- quality 非法 → 400。
"""
from __future__ import annotations

import io
from datetime import datetime
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError

from app.models.orm import Dataset
from app.schemas.dataset import DatasetCreate, DatasetOut, DatasetRename
from app.services import dataset_service, export_service

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


class PullRecordsIn(BaseModel):
    """POST /api/datasets/pull 请求体（T8，需求 6.2b）。"""

    target_dataset_id: int
    start_time: datetime
    end_time: datetime
    quality: str = "全部"  # "合格" | "全部"；非法值 → 400


class PullRecordsOut(BaseModel):
    """拉取结果：实际复制进目标数据集（并落 Record）的条数。"""

    pulled: int


@router.post("/pull", response_model=PullRecordsOut)
def pull_records(payload: PullRecordsIn) -> dict[str, int]:
    """内置库按时间范围拉取 + 质量过滤（需求 6.2b）。

    源为目标数据集之外的全部 Record 行（内置库，与普通数据集共用 records 表），
    复制图片+同名 XML 进目标数据集目录并镜像建 Record；源文件只读不动。
    目标不存在 → 404，quality 非法 → 400。
    """
    if payload.quality not in ("合格", "全部"):
        raise HTTPException(status_code=400, detail="quality 仅支持「合格」或「全部」")
    try:
        return dataset_service.pull_records(
            target_dataset_id=payload.target_dataset_id,
            start_time=payload.start_time,
            end_time=payload.end_time,
            quality=payload.quality,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/{dataset_id}/export")
def export_dataset(dataset_id: int) -> StreamingResponse:
    """数据集整体导出为 ZIP（T11，需求 6.5）：images/ 图片 + annotations/ 同名 XML。

    流式返回 ``application/zip``，Content-Disposition 携带数据集名（RFC 5987
    编码）；数据集不存在 → 404；空数据集 → 合法空 ZIP（REQ-6.5-4）。
    """
    try:
        dataset_name = export_service.get_dataset_name(dataset_id)
        zip_bytes = export_service.export_dataset(dataset_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                f"attachment; filename=dataset-{dataset_id}.zip; "
                f"filename*=UTF-8''{quote(dataset_name)}.zip"
            )
        },
    )
