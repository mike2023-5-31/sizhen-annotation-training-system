"""图片浏览 / 检索 / 标签读取（6.3）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from app.db import SessionLocal
from app.models.orm import Dataset, Record
from app.services import xml_io


def _record(db, dataset_id: int, filename: str) -> Record:
    rec = db.scalars(
        select(Record).where(
            Record.dataset_id == dataset_id,
            Record.filename == filename,
        )
    ).first()
    if rec is None:
        raise ValueError(f"图片 {filename} 不存在于数据集 {dataset_id}")
    return rec


def _ensure_dataset(db, dataset_id: int) -> None:
    if db.get(Dataset, dataset_id) is None:
        raise ValueError(f"数据集 {dataset_id} 不存在")


def list_images(dataset_id: int) -> list[str]:
    with SessionLocal() as db:
        _ensure_dataset(db, dataset_id)
        return list(
            db.scalars(
                select(Record.filename)
                .where(Record.dataset_id == dataset_id)
                .order_by(Record.filename)
            )
        )


def search_images(dataset_id: int, query: str) -> list[str]:
    with SessionLocal() as db:
        _ensure_dataset(db, dataset_id)
        stmt = select(Record.filename).where(
            Record.dataset_id == dataset_id,
            func.lower(Record.filename).contains(query.lower()),
        )
        return list(db.scalars(stmt.order_by(Record.filename)))


def get_image_path(dataset_id: int, filename: str) -> Path:
    with SessionLocal() as db:
        rec = _record(db, dataset_id, filename)
        return Path(rec.image_path)


def read_labels(dataset_id: int, filename: str) -> dict[str, Any]:
    """读取图片同名 XML 标注；XML 缺失时返回空标注结构。"""
    with SessionLocal() as db:
        rec = _record(db, dataset_id, filename)
        xml_path = Path(rec.image_path).with_suffix(".xml")
        if not xml_path.is_file():
            return {
                "folder": "",
                "filename": filename,
                "quality": "",
                "labels": [],
            }
        return xml_io.read_xml(xml_path)
