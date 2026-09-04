"""图片浏览/检索服务（T9，需求 6.3）。

- ``list_images``：数据集全部图片 filename（``Record`` 表，filename 升序）；
- ``search_images``：文件名大小写不敏感子串匹配（DB 端 ``lower().contains``，
  字面量匹配，`%`/`_` 不作通配符）；
- ``get_image_path``：图片绝对路径（来自 ``Record.image_path``，存储约定
  ``data/datasets/<dataset_id>/<filename>``）；
- ``read_labels``：图片同名 ``.xml`` → dict（``xml_io.read_xml``），XML 缺失抛
  ``ValueError``（router 转 404）。

与数据集服务同约定：每个函数自管 Session（``SessionLocal`` 上下文）；
目标不存在 → 抛 ``ValueError``（router 转 404）。

安全说明：所有按 filename 的定位都经 DB 精确匹配 ``Record.filename``，
URL 中的 filename 只是查询键，天然杜绝路径穿越。
"""
from __future__ import annotations

from pathlib import Path
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
                func.lower(Record.filename).contains(q.lower()),
            )
            .order_by(Record.filename)
        )
        return list(db.scalars(stmt).all())


def get_image_path(dataset_id: int, filename: str) -> Path:
    """图片绝对路径（来自 ``Record.image_path``）。记录不存在 → ValueError。"""
    with SessionLocal() as db:
        rec = db.scalars(
            select(Record).where(
                Record.dataset_id == dataset_id,
                Record.filename == filename,
            )
        ).first()
        if rec is None:
            raise ValueError(f"图片 {filename} 不存在于数据集 {dataset_id}")
        return Path(rec.image_path)


def read_labels(dataset_id: int, filename: str) -> dict[str, Any]:
    """读取图片同名 XML 标注 → dict（folder/filename/path/size/quality/labels）。

    图片记录不存在或同名 XML 缺失 → ValueError（router 转 404）。
    """
    image_path = get_image_path(dataset_id, filename)
    xml_path = image_path.with_suffix(".xml")
    if not xml_path.is_file():
        raise ValueError(f"图片 {filename} 无标注 XML")
    return xml_io.read_xml(xml_path)
