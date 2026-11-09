"""图片浏览/检索服务（T9，需求 6.3）。

- ``list_images``：数据集全部图片 filename（``Record`` 表，filename 升序）；
- ``search_images``：文件名大小写不敏感子串匹配（DB 端 ``lower().contains``，
  字面量匹配，`%`/`_` 不作通配符）；
- ``get_image_path``：图片绝对路径（来自 ``Record.image_path``，存储约定
  ``data/datasets/<dataset_id>/<filename>``）；
- ``read_labels``：返回图片元数据与同名 XML 的内容；无 XML 时返回可用于首次保存的空标注。

与数据集服务同约定：每个函数自管 Session（``SessionLocal`` 上下文）；
目标不存在 → 抛 ``ValueError``（router 转 404）。

安全说明：所有按 filename 的定位都经 DB 精确匹配 ``Record.filename``，
URL 中的 filename 只是查询键，天然杜绝路径穿越。
"""
from __future__ import annotations

from pathlib import Path
from stat import S_ISREG
from typing import Any

from sqlalchemy import func, select

from app.db import SessionLocal
from app.models.orm import Dataset, Record
from app.services import xml_io


def _ensure_dataset_exists(db, dataset_id: int) -> None:
    if db.get(Dataset, dataset_id) is None:
        raise ValueError(f"数据集 {dataset_id} 不存在")


def list_images(dataset_id: int) -> list[str]:
    """数据集全部图片 filename（升序）。数据集不存在 → ValueError。"""
    with SessionLocal() as db:
        _ensure_dataset_exists(db, dataset_id)
        stmt = (
            select(Record.filename)
            .where(Record.dataset_id == dataset_id)
            .order_by(Record.filename)
        )
        return list(db.scalars(stmt).all())


def search_images(dataset_id: int, q: str) -> list[str]:
    """文件名大小写不敏感子串匹配（q 为空 → 全部）。数据集不存在 → ValueError。"""
    with SessionLocal() as db:
        _ensure_dataset_exists(db, dataset_id)
        stmt = (
            select(Record.filename)
            .where(
                Record.dataset_id == dataset_id,
                func.lower(Record.filename).contains(q.lower(), autoescape=True),
            )
            .order_by(Record.filename)
        )
        return list(db.scalars(stmt).all())


def _get_record(dataset_id: int, filename: str) -> Record:
    with SessionLocal() as db:
        rec = db.scalars(
            select(Record).where(
                Record.dataset_id == dataset_id,
                Record.filename == filename,
            )
        ).first()
        if rec is None:
            raise ValueError(f"图片 {filename} 不存在于数据集 {dataset_id}")
        return rec


def get_image_path(dataset_id: int, filename: str) -> Path:
    """图片绝对路径（来自 ``Record.image_path``）。记录不存在 → ValueError。"""
    return Path(_get_record(dataset_id, filename).image_path)


def read_labels(dataset_id: int, filename: str) -> dict[str, Any]:
    """读取同名 XML 标注与图片记录的文件大小和记录时间。"""
    record = _get_record(dataset_id, filename)
    image_path = Path(record.image_path)
    try:
        image_stat = image_path.stat()
        file_size = image_stat.st_size if S_ISREG(image_stat.st_mode) else None
    except (FileNotFoundError, NotADirectoryError):
        file_size = None
    xml_path = image_path.with_suffix(".xml")
    try:
        labels = xml_io.read_versioned_xml(xml_path)
    except (FileNotFoundError, NotADirectoryError) as exc:
        if file_size is None:
            raise ValueError(f"图片和标注文件缺失: {filename}") from exc
        labels = {"filename": filename, "labels": [], "quality": record.quality or "",
                  "revision": "missing", "annotation_state": "未标注"}
    labels["file_size_bytes"] = file_size
    labels["record_timestamp"] = record.timestamp.isoformat() + "Z" if record.timestamp else None
    labels.setdefault("collector_alias", None)
    return labels
