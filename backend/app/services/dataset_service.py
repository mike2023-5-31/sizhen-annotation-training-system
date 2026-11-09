"""数据集 CRUD 业务封装（T6，需求 6.1）。

每个函数自管 Session（``SessionLocal`` 上下文），不接收 db 参数：
- 目标不存在 → 抛 ``ValueError``（router 转 404）；
- 重名（唯一约束）→ 让 ``IntegrityError`` 上抛（router 转 409）——
  失败 commit 后的 session 随 ``with`` 退出自动 close/rollback；
- 删除仅清 DB 行（records 经 ORM cascade + FK ON DELETE CASCADE），
  **不触碰磁盘图片文件**（T6 锁定约束，见 PLAN T6 Must NOT）。
"""
from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from xml.etree.ElementTree import ParseError

from sqlalchemy import select

from app import config
from app.db import SessionLocal
from app.models.orm import Dataset, ImportBatch, Record
from app.services import xml_io


def create_dataset(name: str, modality: str | None = None) -> Dataset:
    """新建数据集。重名时 IntegrityError 上抛（调用方转 409）。"""
    with SessionLocal() as db:
        ds = Dataset(name=name, modality=modality)
        db.add(ds)
        db.commit()
        db.refresh(ds)
        return ds


def list_datasets() -> list[Dataset]:
    """按 id 升序返回全部数据集。"""
    with SessionLocal() as db:
        stmt = select(Dataset).order_by(Dataset.id)
        return list(db.scalars(stmt).all())


def get_dataset_counts() -> dict[int, dict[str, int]]:
    """统计图片记录与含标签的有效 XML，质量标记不计为标注。"""
    with SessionLocal() as db:
        rows = db.execute(select(Record.dataset_id, Record.image_path)).all()

    counts: dict[int, dict[str, int]] = {}
    for dataset_id, image_path in rows:
        current = counts.setdefault(dataset_id, {"image_count": 0, "annotated_count": 0})
        current["image_count"] += 1
        try:
            annotation = xml_io.read_xml(Path(image_path).with_suffix(".xml"))
        except (OSError, ParseError, ValueError):
            continue
        if annotation["labels"]:
            current["annotated_count"] += 1
    return counts


def rename_dataset(
    dataset_id: int, name: str, *, modality: str | None = None, update_modality: bool = False,
) -> Dataset:
    """重命名数据集。不存在抛 ValueError；重名时 IntegrityError 上抛。"""
    with SessionLocal() as db:
        ds = db.get(Dataset, dataset_id)
        if ds is None:
            raise ValueError(f"数据集 {dataset_id} 不存在")
        ds.name = name
        if update_modality:
            ds.modality = modality
        db.commit()
        db.refresh(ds)
        return ds


def delete_dataset(dataset_id: int) -> None:
    """删除数据集（DB 行，含级联 records）。不存在抛 ValueError。

    不删除 ``data/datasets/<name>/`` 下的图片/XML 磁盘文件。
    """
    with SessionLocal() as db:
        ds = db.get(Dataset, dataset_id)
        if ds is None:
            raise ValueError(f"数据集 {dataset_id} 不存在")
        db.delete(ds)
        db.commit()


def _to_naive_utc(dt: datetime) -> datetime:
    """带时区的入参统一转 naive UTC（与 orm._utcnow 落盘格式一致，便于比较）。"""
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


class InvalidPullRange(ValueError):
    """结束时间早于开始时间。"""


def list_import_batches(dataset_id: int, offset: int, limit: int) -> dict:
    with SessionLocal() as db:
        if db.get(Dataset, dataset_id) is None:
            raise ValueError(f"数据集 {dataset_id} 不存在")
        rows = db.scalars(select(ImportBatch).where(ImportBatch.dataset_id == dataset_id)
                          .order_by(ImportBatch.id.desc()).offset(offset).limit(limit + 1)).all()
        return {
            "items": [{
                "id": row.id, "actor_id": row.actor_id,
                "created_at": row.created_at.replace(tzinfo=timezone.utc),
                "source": row.source, "filename": row.filename, "quality": row.quality,
                "start_time": row.start_time.replace(tzinfo=timezone.utc) if row.start_time else None,
                "end_time": row.end_time.replace(tzinfo=timezone.utc) if row.end_time else None,
                "imported": row.imported, "skipped": row.skipped, "errors": row.errors,
            } for row in rows[:limit]],
            "has_more": len(rows) > limit,
        }


def list_all_import_batches(offset: int, limit: int, dataset_id: int | None = None) -> dict:
    """跨数据集的导入记录（导入记录中心）；可按数据集过滤并分页。"""
    with SessionLocal() as db:
        if dataset_id is not None and db.get(Dataset, dataset_id) is None:
            raise ValueError(f"数据集 {dataset_id} 不存在")
        query = select(ImportBatch, Dataset.name).join(Dataset, Dataset.id == ImportBatch.dataset_id)
        if dataset_id is not None:
            query = query.where(ImportBatch.dataset_id == dataset_id)
        rows = db.execute(
            query.order_by(ImportBatch.id.desc()).offset(offset).limit(limit + 1)
        ).all()
        items = [
            {
                "id": row.id,
                "dataset_id": row.dataset_id,
                "dataset_name": dataset_name,
                "actor_id": row.actor_id,
                "created_at": row.created_at.replace(tzinfo=timezone.utc),
                "source": row.source,
                "filename": row.filename,
                "quality": row.quality,
                "start_time": row.start_time.replace(tzinfo=timezone.utc) if row.start_time else None,
                "end_time": row.end_time.replace(tzinfo=timezone.utc) if row.end_time else None,
                "imported": row.imported,
                "skipped": row.skipped,
                "errors": row.errors,
            }
            for row, dataset_name in rows[:limit]
        ]
        return {"items": items, "has_more": len(rows) > limit}


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pull_records(
    target_dataset_id: int,
    start_time: datetime,
    end_time: datetime,
    quality: str,
    actor_id: str | None = None,
) -> dict[str, int]:
    """内置库按时间范围拉取 + 质量过滤（T8，需求 6.2b）。

    - 源范围：**目标数据集之外的全部** Record 行（内置库与普通数据集共用
      records 表，见概要设计 9.1），``timestamp`` 落在闭区间
      ``[start_time, end_time]``（naive UTC 比较；NULL 时间戳自然排除）。
      排除目标自身记录；相同图片内容和采集时间视为同一份采集记录；
    - ``quality="合格"`` 仅命中 ``quality == "合格"`` 的行，``"全部"`` 不过滤
      （值合法性由 router 校验）；
    - 每个未重复的命中行：复制图片与同名 XML（若存在），有同名时生成新名称；
    - 目标数据集不存在 → ValueError（router 转 404）；
    - 源图片文件缺失的记录跳过（不制造悬空路径记录）；
    - 事务失败时清理本次新建文件。
    """
    start_time = _to_naive_utc(start_time)
    end_time = _to_naive_utc(end_time)
    if end_time < start_time:
        raise InvalidPullRange("结束时间不能早于开始时间")

    with SessionLocal() as db:
        if db.get_bind().dialect.name == "sqlite":
            db.connection().exec_driver_sql("BEGIN IMMEDIATE")
        target = db.get(Dataset, target_dataset_id)
        if target is None:
            raise ValueError(f"数据集 {target_dataset_id} 不存在")
        batch = ImportBatch(dataset_id=target_dataset_id, source="pull", actor_id=actor_id,
                            start_time=start_time, end_time=end_time, quality=quality,
                            imported=0, skipped=0, errors=0)
        db.add(batch)

        stmt = (
            select(Record)
            .where(
                Record.dataset_id != target_dataset_id,
                Record.timestamp >= start_time,
                Record.timestamp <= end_time,
            )
            .order_by(Record.id)
        )
        if quality == "合格":
            stmt = stmt.where(Record.quality == "合格")
        sources = list(db.scalars(stmt).all())
        if not sources:
            db.commit()
            return {"pulled": 0}

        target_dir = Path(config.DATASETS_DIR) / str(target_dataset_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_rows = db.execute(select(Record.filename, Record.image_path, Record.timestamp)
                                 .where(Record.dataset_id == target_dataset_id)).all()
        occupied = {path.stem.casefold() for path in target_dir.iterdir()}
        existing = set()
        for filename, image_path, timestamp in target_rows:
            occupied.add(Path(filename).stem.casefold())
            path = Path(image_path)
            if timestamp is not None and start_time <= timestamp <= end_time and path.is_file():
                existing.add((timestamp, _file_digest(path)))

        pulled = 0
        created: list[Path] = []
        try:
            for src in sources:
                src_img = Path(src.image_path)
                filename = src.filename
                if (not src_img.is_file() or filename in {"", ".", ".."}
                        or any(char in filename for char in "/\\:")):
                    batch.errors += 1
                    continue
                digest = _file_digest(src_img)
                identity = (src.timestamp, digest)
                if identity in existing:
                    batch.skipped += 1
                    continue

                dest_img = target_dir / filename
                if dest_img.stem.casefold() in occupied:
                    base = f"{dest_img.stem}-{digest[:10]}"
                    ext = dest_img.suffix
                    dest_img = target_dir / f"{base}{ext}"
                    index = 1
                    while dest_img.stem.casefold() in occupied:
                        dest_img = target_dir / f"{base}-{index}{ext}"
                        index += 1

                files = [(src_img, dest_img)]
                src_xml = src_img.with_suffix(".xml")
                if src_xml.is_file():
                    files.append((src_xml, dest_img.with_suffix(".xml")))
                for source, destination in files:
                    with source.open("rb") as input_stream, destination.open("xb") as output:
                        created.append(destination)
                        shutil.copyfileobj(input_stream, output)

                db.add(Record(
                    dataset_id=target_dataset_id,
                    filename=dest_img.name,
                    image_path=str(dest_img),
                    timestamp=src.timestamp,
                    quality=src.quality,
                ))
                occupied.add(dest_img.stem.casefold())
                existing.add(identity)
                pulled += 1

            batch.imported = pulled
            db.commit()
        except BaseException:
            db.rollback()
            for path in reversed(created):
                path.unlink(missing_ok=True)
            raise
        return {"pulled": pulled}
