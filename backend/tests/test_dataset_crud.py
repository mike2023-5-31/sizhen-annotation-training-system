"""数据集 CRUD 测试（T6，需求 6.1）：创建/列表/重命名/删除与约束语义。

覆盖：
- create/list/rename/delete 正常路径（id 升序列表、返回对象字段正确）；
- 重名（唯一约束）→ IntegrityError，且失败后 session 可继续使用；
- 目标不存在 → ValueError（router 转 404）；
- 删除级联清 records 行，但**保留磁盘文件**（T6 锁定约束）。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models.orm import Dataset, Record
from app.services import dataset_service


@pytest.fixture
def crud_env(tmp_path: Path, monkeypatch):
    """临时 SQLite，替换 dataset_service.SessionLocal（不触碰真实 data/db.sqlite）。"""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test_dataset_crud.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    monkeypatch.setattr(dataset_service, "SessionLocal", factory)
    return factory


# ---------------------------------------------------------------------------
# create / list
# ---------------------------------------------------------------------------


def test_create_and_list_ordered_by_id(crud_env) -> None:
    """创建多个数据集 → list_datasets 按 id 升序返回，字段完整。"""
    b = dataset_service.create_dataset("B")
    a = dataset_service.create_dataset("A")
    c = dataset_service.create_dataset("C")

    assert b.id < a.id < c.id  # autoincrement 递增
    ds_list = dataset_service.list_datasets()
    assert [ds.name for ds in ds_list] == ["B", "A", "C"]  # 按 id 升序，非字母序
    assert [ds.id for ds in ds_list] == [b.id, a.id, c.id]
    for ds in ds_list:
        assert isinstance(ds, Dataset)
        assert ds.created_at is not None


def test_create_duplicate_name_raises_integrity_error(crud_env) -> None:
    """重名 → IntegrityError；失败后 session 恢复，后续操作正常。"""
    dataset_service.create_dataset("dup")
    with pytest.raises(IntegrityError):
        dataset_service.create_dataset("dup")

    # 上一条失败不应污染后续会话
    ok = dataset_service.create_dataset("other")
    assert [ds.name for ds in dataset_service.list_datasets()] == ["dup", "other"]
    assert ok.name == "other"


# ---------------------------------------------------------------------------
# rename
# ---------------------------------------------------------------------------


def test_rename_dataset(crud_env) -> None:
    """重命名成功：返回对象同 id、新 name；列表随之更新。"""
    ds = dataset_service.create_dataset("old")
    renamed = dataset_service.rename_dataset(ds.id, "new")

    assert renamed.id == ds.id
    assert renamed.name == "new"
    assert [d.name for d in dataset_service.list_datasets()] == ["new"]


def test_rename_missing_dataset_raises(crud_env) -> None:
    with pytest.raises(ValueError, match="不存在"):
        dataset_service.rename_dataset(9999, "whatever")


def test_rename_duplicate_name_raises_integrity_error(crud_env) -> None:
    """重命名为已有名称 → IntegrityError，且原名保持不变。"""
    dataset_service.create_dataset("taken")
    ds = dataset_service.create_dataset("mine")
    with pytest.raises(IntegrityError):
        dataset_service.rename_dataset(ds.id, "taken")
    assert dataset_service.list_datasets()[1].name == "mine"


# ---------------------------------------------------------------------------
# delete：级联清记录、保留磁盘文件（T6 锁定约束）
# ---------------------------------------------------------------------------


def test_delete_cascades_records_keeps_disk_files(tmp_path: Path, crud_env) -> None:
    """删除数据集：DB 行与 records 级联清空，磁盘图片文件原样保留。"""
    ds = dataset_service.create_dataset("victim")
    disk_file = tmp_path / "keep.png"
    disk_file.write_bytes(b"fake-image-bytes")
    with crud_env() as db:
        db.add(
            Record(
                dataset_id=ds.id,
                filename="keep.png",
                image_path=str(disk_file.resolve()),
                quality="合格",
            )
        )
        db.commit()

    dataset_service.delete_dataset(ds.id)

    assert dataset_service.list_datasets() == []
    with crud_env() as db:
        assert db.scalars(select(Record)).all() == []  # records 级联删除
    assert disk_file.is_file()  # 磁盘文件不被触碰（T6 锁定约束）
    assert disk_file.read_bytes() == b"fake-image-bytes"


def test_delete_missing_dataset_raises(crud_env) -> None:
    with pytest.raises(ValueError, match="不存在"):
        dataset_service.delete_dataset(9999)
    # 删除失败不影响其它数据集
    dataset_service.create_dataset("still-here")
    assert [d.name for d in dataset_service.list_datasets()] == ["still-here"]
