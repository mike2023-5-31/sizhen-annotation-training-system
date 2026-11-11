"""数据集增删改查 + 内置库按时段拉取（6.1 / 6.2b）。"""
from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from sqlalchemy import select

from app import config
from app.db import SessionLocal
from app.models.orm import Dataset, Record
from app.services import xml_io


def _to_utc_naive(value: datetime) -> datetime:
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def create_dataset(name: str) -> Dataset:
    with SessionLocal() as db:
        ds = Dataset(name=name)
        db.add(ds)
        db.commit()
        db.refresh(ds)
        return ds


def list_datasets() -> list[Dataset]:
    with SessionLocal() as db:
        return list(db.scalars(select(Dataset).order_by(Dataset.id)).all())


def rename_dataset(dataset_id: int, name: str) -> Dataset:
    with SessionLocal() as db:
        ds = db.get(Dataset, dataset_id)
        if ds is None:
            raise ValueError(f"数据集 {dataset_id} 不存在")
        ds.name = name
        db.commit()
        db.refresh(ds)
        return ds


def delete_dataset(dataset_id: int) -> None:
    with SessionLocal() as db:
        ds = db.get(Dataset, dataset_id)
        if ds is None:
            raise ValueError(f"数据集 {dataset_id} 不存在")
        db.delete(ds)
        db.commit()


def count_datasets() -> dict[int, dict[str, int]]:
    """每个数据集的图片数与已标注数（6.1 列表统计）。"""
    with SessionLocal() as db:
        rows = db.execute(
            select(Record.dataset_id, Record.quality)
        ).all()
    stats: dict[int, dict[str, int]] = {}
    for dataset_id, quality in rows:
        entry = stats.setdefault(dataset_id, {"image_count": 0, "annotated_count": 0})
        entry["image_count"] += 1
        if quality == "合格":
            entry["annotated_count"] += 1
    return stats


class InvalidPullRange(ValueError):
    """拉取时间区间非法（起始晚于结束）。"""


def pull_records(
    target_dataset_id: int,
    start_time: datetime,
    end_time: datetime,
    quality: str,
) -> dict[str, int]:
    """从内置库（其它数据集的记录并集）按时间范围 + 质量过滤拉取到目标数据集。

    命中记录会把图片与同名 XML 复制到目标数据集目录，并在目标数据集下新建
    Record；源记录与源文件只读不动。
    """
    start_time = _to_utc_naive(start_time)
    end_time = _to_utc_naive(end_time)
    if start_time > end_time:
        raise InvalidPullRange("起始时间不能晚于结束时间")

    with SessionLocal() as db:
        target = db.get(Dataset, target_dataset_id)
        if target is None:
            raise ValueError(f"数据集 {target_dataset_id} 不存在")

        stmt = select(Record).where(
            Record.dataset_id != target_dataset_id,
            Record.timestamp >= start_time,
            Record.timestamp <= end_time,
        )
        if quality == "合格":
            stmt = stmt.where(Record.quality == "合格")
        sources = list(db.scalars(stmt.order_by(Record.id)).all())

        target_dir = Path(config.DATASETS_DIR) / str(target_dataset_id)
        target_dir.mkdir(parents=True, exist_ok=True)

        pulled = 0
        for source in sources:
            src_image = Path(source.image_path)
            if not src_image.is_file():
                continue
            # 目标目录已有同名文件时重命名，避免覆盖已有数据
            dest_name = source.filename
            if (target_dir / dest_name).exists():
                stem = PurePosixPath(dest_name).stem
                suffix = PurePosixPath(dest_name).suffix
                digest = hashlib.sha256(
                    f"{source.id}:{dest_name}".encode()
                ).hexdigest()[:8]
                dest_name = f"{stem}-{digest}{suffix}"

            dest_image = target_dir / dest_name
            shutil.copyfile(src_image, dest_image)
            dest_xml = dest_image.with_suffix(".xml")
            src_xml = src_image.with_suffix(".xml")
            if src_xml.is_file():
                shutil.copyfile(src_xml, dest_xml)
                # 重写 XML 的 folder/filename/path，避免指向源数据集造成漂移
                data = xml_io.read_xml(dest_xml)
                data.update(
                    {
                        "folder": target.name,
                        "filename": dest_name,
                        "path": str(dest_image),
                    }
                )
                xml_io.write_xml(dest_xml, data)
            db.add(
                Record(
                    dataset_id=target_dataset_id,
                    filename=dest_name,
                    image_path=str(dest_image),
                    timestamp=source.timestamp,
                    quality=source.quality,
                )
            )
            pulled += 1

        db.commit()
        return {"pulled": pulled}
