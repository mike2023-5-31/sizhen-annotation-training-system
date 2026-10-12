"""标注路由（T10，需求 6.3/6.4）。

挂载于 ``main.api_router``（整体 ``/api`` 前缀）后对外为：

- ``GET /api/labels`` → ``{"categories": [{"name": "舌", "labels": [...]}, ...]}``
  （标注面板标签树数据源，来自 ``app.labels_config.CATEGORIES``）；
- ``PUT /api/datasets/{dataset_id}/image/{filename}/labels``
  请求体 ``{"labels": [{"category","name","confidence"}], "quality": "合格"}`` →
  原子写回 XML + quality 镜像，返回写盘后的 dict；
- ``DELETE /api/datasets/{dataset_id}/image/{filename}`` → ``{"ok": true}``
  （硬删除记录 + 图片 + 同名 XML）。

错误映射：
- 图片记录不存在（``ValueError``）→ 404；
- quality 非法 / 标签不在标签体系 / 每 category 多于一个 / labels 超 4 个 → 400
  （NF-7.3-3：非法标签不写入脏数据）。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel, Field

from app.labels_config import CATEGORIES
from app.services import annotation_service

router = APIRouter(tags=["annotations"])

_LABELS_BY_CATEGORY: dict[str, set[str]] = {
    category["name"]: set(category["labels"]) for category in CATEGORIES
}

# 「不合格」仅经导入 XML 带入，标注页不直接设置（REQ-6.4-4）
_ALLOWED_QUALITY = ("合格", "不合格", "")


@router.get("/labels")
def get_labels() -> dict[str, list[dict]]:
    """标签体系：四类并列 + 每类子标签（标注面板标签树数据源）。"""
    return {"categories": CATEGORIES}


class LabelIn(BaseModel):
    """单条标签请求项；confidence 由服务端恒写 1.0（人工标注），此处仅容错接收。"""

    category: str = Field(min_length=1)
    name: str = Field(min_length=1)
    confidence: float | None = None


class SaveLabelsIn(BaseModel):
    """PUT 写回请求体：0–4 个 label（每 category 至多一个）+ quality。"""

    labels: list[LabelIn] = Field(default_factory=list)
    quality: str = ""
    expected_revision: str | None = None


def _validate_payload(payload: SaveLabelsIn) -> list[dict[str, str]]:
    """标签/质量校验：非法 → HTTPException(400)，不写入脏数据（NF-7.3-3）。"""
    if payload.quality not in _ALLOWED_QUALITY:
        raise HTTPException(
            status_code=400,
            detail="quality 仅支持「合格」「不合格」或空字符串（待质检）",
        )
    if len(payload.labels) > 4:
        raise HTTPException(status_code=400, detail="labels 至多 4 个（每 category 一个）")
    seen: set[str] = set()
    for label in payload.labels:
        if label.category not in _LABELS_BY_CATEGORY:
            raise HTTPException(status_code=400, detail=f"未知 category「{label.category}」")
        if label.name not in _LABELS_BY_CATEGORY[label.category]:
            raise HTTPException(
                status_code=400,
                detail=f"category「{label.category}」不存在子标签「{label.name}」",
            )
        if label.category in seen:
            raise HTTPException(
                status_code=400,
                detail=f"category「{label.category}」重复（每 category 至多一个 label）",
            )
        seen.add(label.category)
    return [{"category": l.category, "name": l.name} for l in payload.labels]


@router.put("/datasets/{dataset_id}/image/{filename}/labels")
def save_labels(dataset_id: int, filename: str, payload: SaveLabelsIn, request: Request) -> dict:
    """原子写回当前图片标注（labels + quality），返回写盘后的完整 dict。"""
    labels = _validate_payload(payload)
    if payload.expected_revision is None:
        raise HTTPException(status_code=428, detail="请先读取当前标注版本后保存")
    try:
        return annotation_service.save_labels(dataset_id, filename, labels, payload.quality,
                                              expected_revision=payload.expected_revision,
                                              actor_id=getattr(request.state, "user", {}).get("id"))
    except annotation_service.AnnotationConflict as exc:
        raise HTTPException(status_code=409, detail={"message": str(exc), "current": exc.current})
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/datasets/{dataset_id}/image/{filename}/history")
def history(dataset_id: int, filename: str, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=100)):
    try:
        return annotation_service.list_edits(dataset_id, filename, offset, limit)
    except ValueError as exc:
        raise HTTPException(404, str(exc))


@router.delete("/datasets/{dataset_id}/image/{filename}")
def delete_image(dataset_id: int, filename: str) -> dict[str, bool]:
    """硬删除当前图片（记录行 + 图片文件 + 同名 XML）。"""
    try:
        annotation_service.delete_record(dataset_id, filename)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"ok": True}
