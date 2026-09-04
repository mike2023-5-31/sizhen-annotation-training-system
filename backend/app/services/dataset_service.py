"""数据集 CRUD 业务封装（T6，需求 6.1）。

每个函数自管 Session（``SessionLocal`` 上下文），不接收 db 参数：
- 目标不存在 → 抛 ``ValueError``（router 转 404）；
- 重名（唯一约束）→ 让 ``IntegrityError`` 上抛（router 转 409）——
  失败 commit 后的 session 随 ``with`` 退出自动 close/rollback；
- 删除仅清 DB 行（records 经 ORM cascade + FK ON DELETE CASCADE），
  **不触碰磁盘图片文件**（T6 锁定约束，见 PLAN T6 Must NOT）。
"""
from __future__ import annotations

from sqlalchemy import select

from app.db import SessionLocal
from app.models.orm import Dataset


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
