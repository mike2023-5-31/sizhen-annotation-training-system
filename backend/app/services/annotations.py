"""标注保存与删除（6.4）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.db import SessionLocal
from app.labels_config import CATEGORIES
from app.models.orm import Record
from app.services import xml_io

_ALLOWED_QUALITY = {"合格", "不合格", ""}


def _taxonomy() -> dict[str, set[str]]:
    return {item["name"]: set(item["labels"]) for item in CATEGORIES}


def validate_labels(labels: list[dict[str, Any]]) -> None:
    allowed = _taxonomy()
    seen: set[str] = set()
    for label in labels:
        category, name = label["category"], label["name"]
        if category not in allowed or name not in allowed[category]:
            raise ValueError(f"未知标签: {category}/{name}")
        if category in seen:
            raise ValueError(f"类别 {category} 出现多个标签")
        seen.add(category)


def save_labels(
    dataset_id: int,
    filename: str,
    labels: list[dict[str, Any]],
    quality: str,
) -> dict[str, Any]:
    """把标签与质量写入同名 XML（原子写），并镜像 quality 到数据库。"""
    if quality not in _ALLOWED_QUALITY:
        raise ValueError("质量标记仅支持「合格」「不合格」或空字符串")
    validate_labels(labels)

    with SessionLocal() as db:
        rec = db.scalars(
            select(Record).where(
                Record.dataset_id == dataset_id,
                Record.filename == filename,
            )
        ).first()
        if rec is None:
            raise ValueError(f"图片 {filename} 不存在于数据集 {dataset_id}")

        xml_path = Path(rec.image_path).with_suffix(".xml")
        current = xml_io.read_xml(xml_path) if xml_path.is_file() else {}
        current.update(
            {
                "folder": current.get("folder", ""),
                "filename": filename,
                "path": str(Path(rec.image_path)),
                "quality": quality,
                "labels": [
                    {
                        "category": label["category"],
                        "name": label["name"],
                        "confidence": 1.0,
                    }
                    for label in labels
                ],
            }
        )
        xml_io.write_xml(xml_path, current)

        rec.quality = quality
        db.commit()
        return current


def delete_record(dataset_id: int, filename: str) -> None:
    """硬删除记录行、图片文件与同名 XML；内置库源记录不受影响。"""
    with SessionLocal() as db:
        rec = db.scalars(
            select(Record).where(
                Record.dataset_id == dataset_id,
                Record.filename == filename,
            )
        ).first()
        if rec is None:
            raise ValueError(f"图片 {filename} 不存在于数据集 {dataset_id}")
        image_path = Path(rec.image_path)
        db.delete(rec)
        db.commit()

    image_path.unlink(missing_ok=True)
    image_path.with_suffix(".xml").unlink(missing_ok=True)
