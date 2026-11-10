"""数据集整体导出为 ZIP（6.5）。"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

from sqlalchemy import select

from app.db import SessionLocal
from app.models.orm import Dataset, Record


def dataset_name(dataset_id: int) -> str:
    with SessionLocal() as db:
        dataset = db.get(Dataset, dataset_id)
        if dataset is None:
            raise ValueError(f"数据集 {dataset_id} 不存在")
        return dataset.name


def export_dataset(dataset_id: int) -> bytes:
    """把数据集全部图片与同名 XML 打包为 ZIP（images/ + annotations/）。"""
    with SessionLocal() as db:
        dataset = db.get(Dataset, dataset_id)
        if dataset is None:
            raise ValueError(f"数据集 {dataset_id} 不存在")
        records = list(
            db.scalars(
                select(Record).where(Record.dataset_id == dataset_id).order_by(Record.id)
            )
        )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for rec in records:
            image_path = Path(rec.image_path)
            if not image_path.is_file():
                continue
            zf.write(image_path, arcname=f"images/{rec.filename}")
            xml_path = image_path.with_suffix(".xml")
            if xml_path.is_file():
                zf.write(xml_path, arcname=f"annotations/{xml_path.name}")
    return buffer.getvalue()
