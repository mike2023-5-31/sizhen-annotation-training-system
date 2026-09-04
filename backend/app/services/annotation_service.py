"""标注写回/硬删除服务（T10，需求 6.4）。

- ``save_labels``：读既有 XML 保留 folder/path/size，合并新 labels + quality 后
  ``xml_io.write_xml`` 原子写（写 XML 即真相源），并镜像 ``Record.quality``；
- ``delete_record``：硬删除 Record 行 + 图片文件 + 同名 XML。仅作用于
  ``dataset_id`` 内的记录，内置库等其他数据集的同图副本不受影响
  （锁定决策 7 / C-5）。

规则（需求 6.4 + 锁定决策）：
- labels 0–4 个，每 category 至多一个；``confidence`` 恒 1.0（人工标注）；
- ``quality`` 仅接受「合格」/空字符串（「不合格」只经导入 XML 带入，本接口不设置）；
- 图片记录不存在 → ``ValueError``（router 转 404）；
- 非法标签/质量由 router 侧校验（400），本服务信任输入、只做归一化。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.db import SessionLocal
from app.models.orm import Record
from app.services import xml_io


def _get_record(db, dataset_id: int, filename: str) -> Record:
    """按 (dataset_id, filename) 精确定位记录；不存在 → ValueError。"""
    rec = db.scalars(
        select(Record).where(
            Record.dataset_id == dataset_id,
            Record.filename == filename,
        )
    ).first()
    if rec is None:
        raise ValueError(f"图片 {filename} 不存在于数据集 {dataset_id}")
    return rec


def _normalize_labels(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """归一化：confidence 恒 1.0；每 category 至多一个（重复保留首个）。"""
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for label in labels or []:
        category = str(label.get("category", ""))
        if category in seen:
            continue
        seen.add(category)
        normalized.append(
            {
                "category": category,
                "name": str(label.get("name", "")),
                "confidence": 1.0,  # 人工标注
            }
        )
    return normalized


def save_labels(
    dataset_id: int,
    filename: str,
    labels: list[dict[str, Any]],
    quality: str,
) -> dict[str, Any]:
    """写回标注：读既有 XML 保留 folder/path/size → 合并 labels+quality →
    原子写 XML → 镜像 ``Record.quality``。返回写盘后的完整 dict。

    XML 缺失时以数据集名/文件名兜底 folder/filename/path（正常导入路径下
    XML 恒存在，此处仅防御）。
    """
    with SessionLocal() as db:
        rec = _get_record(db, dataset_id, filename)
        image_path = Path(rec.image_path)
        xml_path = image_path.with_suffix(".xml")

        existing: dict[str, Any]
        if xml_path.is_file():
            existing = xml_io.read_xml(xml_path)
        else:
            existing = {}

        data = {
            "folder": existing.get("folder") or rec.dataset.name,
            "filename": existing.get("filename") or filename,
            "path": existing.get("path") or f"{dataset_id}/{filename}",
            "size": existing.get("size"),
            "quality": quality,
            "labels": _normalize_labels(labels),
        }
        xml_io.write_xml(xml_path, data)

        # quality 镜像（锁定决策 11：DB 不存标签，仅镜像 quality）
        rec.quality = quality
        db.commit()
        return data


def delete_record(dataset_id: int, filename: str) -> None:
    """硬删除：Record 行 + 图片文件 + 同名 XML（missing_ok，重复删除安全）。

    仅影响 ``dataset_id`` 内记录；内置库等其他数据集的记录与其文件不动。
    """
    with SessionLocal() as db:
        rec = _get_record(db, dataset_id, filename)
        image_path = Path(rec.image_path)
        xml_path = image_path.with_suffix(".xml")

        db.delete(rec)
        db.commit()

        image_path.unlink(missing_ok=True)
        xml_path.unlink(missing_ok=True)
