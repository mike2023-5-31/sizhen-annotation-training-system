"""数据集 CRUD 业务封装（T6，需求 6.1）。

每个函数自管 Session（``SessionLocal`` 上下文），不接收 db 参数：
- 目标不存在 → 抛 ``ValueError``（router 转 404）；
- 重名（唯一约束）→ 让 ``IntegrityError`` 上抛（router 转 409）——
  失败 commit 后的 session 随 ``with`` 退出自动 close/rollback；
- 删除仅清 DB 行（records 经 ORM cascade + FK ON DELETE CASCADE），
  **不触碰磁盘图片文件**（T6 锁定约束，见 PLAN T6 Must NOT）。
"""
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from app import config
from app.db import SessionLocal
from app.models.orm import Dataset, Record


def create_dataset(name: str) -> Dataset:
    """新建数据集。重名时 IntegrityError 上抛（调用方转 409）。"""
    with SessionLocal() as db:
        ds = Dataset(name=name)
        db.add(ds)
        db.commit()
        db.refresh(ds)
        return ds


def list_datasets() -> list[Dataset]:
    """按 id 升序返回全部数据集。"""
    with SessionLocal() as db:
        stmt = select(Dataset).order_by(Dataset.id)
        return list(db.scalars(stmt).all())


def rename_dataset(dataset_id: int, name: str) -> Dataset:
    """重命名数据集。不存在抛 ValueError；重名时 IntegrityError 上抛。"""
    with SessionLocal() as db:
        ds = db.get(Dataset, dataset_id)
        if ds is None:
            raise ValueError(f"数据集 {dataset_id} 不存在")
        ds.name = name
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


def pull_records(
    target_dataset_id: int,
    start_time: datetime,
    end_time: datetime,
    quality: str,
) -> dict[str, int]:
    """内置库按时间范围拉取 + 质量过滤（T8，需求 6.2b）。

    - 源范围：**目标数据集之外的全部** Record 行（内置库与普通数据集共用
      records 表，见概要设计 9.1），``timestamp`` 落在闭区间
      ``[start_time, end_time]``（naive UTC 比较；NULL 时间戳自然排除）。
      排除目标自身记录，避免把已拉入该数据集的记录重复拉取；
    - ``quality="合格"`` 仅命中 ``quality == "合格"`` 的行，``"全部"`` 不过滤
      （值合法性由 router 校验）；
    - 每个命中行：将图片与同名 XML（若存在）**复制**到
      ``data/datasets/<target_dataset_id>/``（T7 同款存储约定，源文件只读不动），
      并在目标数据集下新建 Record（filename/timestamp/quality 镜像源行）；
    - 目标数据集不存在 → ValueError（router 转 404）；
    - 源图片文件缺失的记录跳过（不制造悬空路径记录）；
    - 复制用 ``shutil.copyfile`` 而非 ``copy2``：本机 Windows 环境下 copy2
      （CopyFile2）覆盖目标已有文件时稳定触发 WinError 32 共享冲突。
    """
    start_time = _to_naive_utc(start_time)
    end_time = _to_naive_utc(end_time)

    with SessionLocal() as db:
        target = db.get(Dataset, target_dataset_id)
        if target is None:
            raise ValueError(f"数据集 {target_dataset_id} 不存在")

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

        target_dir = Path(config.DATASETS_DIR) / str(target_dataset_id)
        target_dir.mkdir(parents=True, exist_ok=True)

        pulled = 0
        for src in sources:
            src_img = Path(src.image_path)
            if not src_img.is_file():
                continue  # 源文件缺失：跳过，不落悬空记录
            dest_img = target_dir / src.filename
            shutil.copyfile(src_img, dest_img)
            src_xml = src_img.with_suffix(".xml")
            if src_xml.is_file():
                shutil.copyfile(src_xml, dest_img.with_suffix(".xml"))
            db.add(
                Record(
                    dataset_id=target_dataset_id,
                    filename=src.filename,
                    image_path=str(dest_img),
                    timestamp=src.timestamp,
                    quality=src.quality,
                )
            )
            pulled += 1

        db.commit()
        return {"pulled": pulled}
