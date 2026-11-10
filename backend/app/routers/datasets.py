"""数据集、导入、图片浏览、标注、导出的路由（6.1–6.5）。"""
from __future__ import annotations

import io
from urllib.parse import quote

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from starlette.concurrency import run_in_threadpool

from app import config
from app.labels_config import CATEGORIES
from app.schemas.dataset import (
    DatasetCreate,
    DatasetListItem,
    DatasetOut,
    DatasetRename,
    PullRequest,
    PullResult,
)
from app.services import annotations, datasets, exporter, images, importer

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("", response_model=DatasetOut, status_code=201)
def create_dataset(payload: DatasetCreate) -> DatasetOut:
    try:
        return datasets.create_dataset(payload.name)
    except IntegrityError:
        raise HTTPException(status_code=409, detail=f"数据集名称「{payload.name}」已存在")


@router.get("", response_model=list[DatasetListItem])
def list_datasets(include_stats: bool = False) -> list[DatasetListItem]:
    items = datasets.list_datasets()
    if not include_stats:
        return [DatasetListItem(**DatasetOut.model_validate(d).model_dump()) for d in items]
    stats = datasets.count_datasets()
    result = []
    for d in items:
        entry = stats.get(d.id, {"image_count": 0, "annotated_count": 0})
        result.append(
            DatasetListItem(**DatasetOut.model_validate(d).model_dump(), **entry)
        )
    return result


@router.patch("/{dataset_id}", response_model=DatasetOut)
def rename_dataset(dataset_id: int, payload: DatasetRename) -> DatasetOut:
    try:
        return datasets.rename_dataset(dataset_id, payload.name)
    except IntegrityError:
        raise HTTPException(status_code=409, detail=f"数据集名称「{payload.name}」已存在")
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: int) -> dict[str, bool]:
    try:
        datasets.delete_dataset(dataset_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"ok": True}


@router.post("/pull", response_model=PullResult)
def pull_records(payload: PullRequest) -> PullResult:
    if payload.quality not in ("合格", "全部"):
        raise HTTPException(status_code=400, detail="quality 仅支持「合格」或「全部」")
    try:
        return datasets.pull_records(
            payload.target_dataset_id,
            payload.start_time,
            payload.end_time,
            payload.quality,
        )
    except datasets.InvalidPullRange as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/{dataset_id}/import")
async def import_zip(dataset_id: int, file: UploadFile = File(...)) -> dict:
    content = await file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(content) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="ZIP 超过上传大小限制")
    try:
        return await run_in_threadpool(importer.import_zip, dataset_id, content)
    except importer.ImportValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/{dataset_id}/images")
def list_images(dataset_id: int) -> dict[str, list[str]]:
    try:
        return {"images": images.list_images(dataset_id)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/{dataset_id}/images/search")
def search_images(dataset_id: int, q: str = "") -> dict[str, list[str]]:
    try:
        return {"images": images.search_images(dataset_id, q)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/{dataset_id}/images/{filename}")
def get_image(dataset_id: int, filename: str) -> FileResponse:
    try:
        path = images.get_image_path(dataset_id, filename)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"图片文件缺失：{filename}")
    return FileResponse(path)


@router.get("/{dataset_id}/image/{filename}/labels")
def read_labels(dataset_id: int, filename: str) -> dict:
    try:
        return images.read_labels(dataset_id, filename)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/{dataset_id}/export")
def export_dataset(dataset_id: int) -> StreamingResponse:
    try:
        name = exporter.dataset_name(dataset_id)
        payload = exporter.export_dataset(dataset_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return StreamingResponse(
        io.BytesIO(payload),
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}.zip"
        },
    )


# ---------------------------------------------------------------------------
# 标注路由（标签树 / 保存 / 删除，6.4）
# ---------------------------------------------------------------------------

labels_router = APIRouter(tags=["annotations"])

_ALLOWED_QUALITY = ("合格", "不合格", "")


class _LabelItem(BaseModel):
    category: str = Field(min_length=1)
    name: str = Field(min_length=1)
    confidence: float | None = None


class _SaveLabels(BaseModel):
    labels: list[_LabelItem] = Field(default_factory=list)
    quality: str = ""


@labels_router.get("/labels")
def get_labels() -> dict[str, list[dict]]:
    return {"categories": CATEGORIES}


@labels_router.put("/datasets/{dataset_id}/image/{filename}/labels")
def save_labels(dataset_id: int, filename: str, payload: _SaveLabels) -> dict:
    if payload.quality not in _ALLOWED_QUALITY:
        raise HTTPException(status_code=400, detail="质量标记仅支持「合格」「不合格」或空字符串")
    labels = [{"category": l.category, "name": l.name} for l in payload.labels]
    try:
        return annotations.save_labels(dataset_id, filename, labels, payload.quality)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@labels_router.delete("/datasets/{dataset_id}/image/{filename}")
def delete_image(dataset_id: int, filename: str) -> dict[str, bool]:
    try:
        annotations.delete_record(dataset_id, filename)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"ok": True}
