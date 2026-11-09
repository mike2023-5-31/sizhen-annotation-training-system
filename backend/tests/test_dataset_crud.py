"""数据集 CRUD 测试（T6，需求 6.1）：创建/列表/重命名/删除与约束语义。

覆盖：
- create/list/rename/delete 正常路径（id 升序列表、返回对象字段正确）；
- 重名（唯一约束）→ IntegrityError，且失败后 session 可继续使用；
- 目标不存在 → ValueError（router 转 404）；
- 删除级联清 records 行，但**保留磁盘文件**（T6 锁定约束）。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app import config
from app.db import Base
from app.main import app
from app.models.orm import Dataset, ImportBatch, Record
from app.services import annotation_service, dataset_service, image_service


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
    monkeypatch.setattr(image_service, "SessionLocal", factory)
    monkeypatch.setattr(annotation_service, "SessionLocal", factory)
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path / "datasets")
    yield factory
    engine.dispose()


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


def test_global_import_history_includes_dataset(crud_env, login_admin) -> None:
    """导入记录中心：跨数据集列表返回数据集归属与分页字段。"""
    ds = dataset_service.create_dataset("导入中心")
    with crud_env() as db:
        db.add(ImportBatch(dataset_id=ds.id, source="zip", filename="batch.zip",
                           imported=2, skipped=1, errors=0))
        db.commit()

    client = login_admin(TestClient(app))
    body = client.get("/api/datasets/imports", params={"limit": 10}).json()
    assert body["has_more"] is False
    assert body["items"][0]["dataset_id"] == ds.id
    assert body["items"][0]["dataset_name"] == "导入中心"
    assert body["items"][0]["imported"] == 2


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


def test_dataset_modality_can_be_declared_changed_or_left_unknown(crud_env, login_admin) -> None:
    with TestClient(app) as client:
        login_admin(client)
        invalid = client.post("/api/datasets", json={"name": "invalid", "modality": "自动推断"})
        assert invalid.status_code == 422
        dataset = client.post("/api/datasets", json={"name": "source", "modality": "舌"}).json()
        assert dataset["modality"] == "舌"
        url = f"/api/datasets/{dataset['id']}"
        assert client.patch(url, json={"name": "renamed"}).json()["modality"] == "舌"
        assert client.patch(url, json={"name": "renamed", "modality": None}).json()["modality"] is None
        assert client.get("/api/datasets", params={"include_stats": True}).json()[0]["modality"] is None


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


def _add_record(db, dataset_id: int, filename: str, image: bytes, xml: bytes | None = None,
                quality: str = "合格") -> Path:
    path = config.DATASETS_DIR / str(dataset_id) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(image)
    if xml is not None:
        path.with_suffix(".xml").write_bytes(xml)
    db.add(Record(dataset_id=dataset_id, filename=filename, image_path=str(path),
                  timestamp=datetime(2026, 9, 7, 4), quality=quality))
    return path


def test_pull_preserves_existing_files_and_is_safe_to_retry(crud_env) -> None:
    source = dataset_service.create_dataset("source")
    target = dataset_service.create_dataset("target")
    source_xml = b"<annotation><quality>\xe5\x90\x88\xe6\xa0\xbc</quality></annotation>"
    target_xml = b"<annotation><quality/></annotation>"
    with crud_env() as db:
        original = _add_record(db, target.id, "same.png", b"edited image", target_xml)
        source_path = _add_record(db, source.id, "same.png", b"source image", source_xml)
        _add_record(db, source.id, "excluded.png", b"excluded", quality="")
        db.commit()

    timestamp = datetime(2026, 9, 7, 4, tzinfo=timezone.utc)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(
            lambda _: dataset_service.pull_records(target.id, timestamp, timestamp, "合格"), range(2)
        ))

    assert sorted(result["pulled"] for result in results) == [0, 1]
    assert original.read_bytes() == b"edited image"
    assert original.with_suffix(".xml").read_bytes() == target_xml
    assert source_path.read_bytes() == b"source image"
    assert source_path.with_suffix(".xml").read_bytes() == source_xml
    with crud_env() as db:
        copied = db.scalars(select(Record).where(Record.dataset_id == target.id)).all()
    assert len(copied) == 2
    imported = next(record for record in copied if record.filename != "same.png")
    assert Path(imported.image_path).read_bytes() == b"source image"
    assert Path(imported.image_path).with_suffix(".xml").read_bytes() == source_xml


def test_pull_rolls_back_new_files_on_failed_commit(crud_env, monkeypatch) -> None:
    source = dataset_service.create_dataset("source")
    target = dataset_service.create_dataset("target")
    with crud_env() as db:
        _add_record(db, source.id, "source.png", b"source", b"<annotation/>")
        db.commit()

    def fail_commit(_session):
        raise RuntimeError("commit failed")

    monkeypatch.setattr(crud_env.class_, "commit", fail_commit)
    timestamp = datetime(2026, 9, 7, 4)
    with pytest.raises(RuntimeError, match="commit failed"):
        dataset_service.pull_records(target.id, timestamp, timestamp, "全部")
    assert not list((config.DATASETS_DIR / str(target.id)).glob("*"))
    with crud_env() as db:
        assert db.scalars(select(Record).where(Record.dataset_id == target.id)).all() == []


def test_filename_search_treats_wildcards_as_literal(crud_env) -> None:
    dataset = dataset_service.create_dataset("search")
    with crud_env() as db:
        _add_record(db, dataset.id, "tongue_100%.png", b"matching")
        _add_record(db, dataset.id, "tongueA100B.png", b"unrelated")
        db.commit()

    assert image_service.search_images(dataset.id, "TONGUE_100%") == ["tongue_100%.png"]


def test_image_details_report_file_metadata_and_allow_first_legacy_edit(crud_env, login_admin) -> None:
    dataset = dataset_service.create_dataset("legacy")
    with crud_env() as db:
        image_path = _add_record(db, dataset.id, "untagged.png", b"image", quality="合格")
        db.commit()

    client = login_admin(TestClient(app))
    url = f"/api/datasets/{dataset.id}/image/untagged.png/labels"
    response = client.get(url)
    assert response.status_code == 200, response.text
    details = response.json()
    assert details["file_size_bytes"] == 5
    assert details["record_timestamp"] == "2026-09-07T04:00:00Z"
    assert details["revision"] == "missing"
    assert details["annotation_state"] == "未标注"
    assert details["quality"] == "合格"

    saved = client.put(url, json={"expected_revision": details["revision"], "quality": details["quality"],
                                 "labels": [{"category": "舌", "name": "舌苔黄"}]})
    assert saved.status_code == 200, saved.text
    assert client.get(url).json()["revision"] == saved.json()["revision"]

    image_path.unlink()
    archived = client.get(url)
    assert archived.status_code == 200, archived.text
    assert archived.json()["revision"] == saved.json()["revision"]
    assert archived.json()["file_size_bytes"] is None

    image_path.with_suffix(".xml").unlink()
    assert client.get(url).status_code == 404


def test_dataset_list_opt_in_stats_use_real_xml_labels(crud_env, login_admin) -> None:
    dataset = dataset_service.create_dataset("with records")
    empty = dataset_service.create_dataset("empty")
    with crud_env() as db:
        _add_record(db, dataset.id, "labeled.png", b"first", b"<annotation><labels><label>"
                    b"<category>tongue</category><name>label</name></label></labels></annotation>")
        _add_record(db, dataset.id, "unlabeled.png", b"second", b"<annotation/>")
        _add_record(db, dataset.id, "invalid.png", b"third", b"incomplete")
        _add_record(db, dataset.id, "missing.png", b"fourth")
        db.commit()

    client = login_admin(TestClient(app))
    original = client.get("/api/datasets")
    assert original.status_code == 200, original.text
    assert set(original.json()[0]) == {"id", "name", "created_at", "updated_at"}
    detailed = client.get("/api/datasets?include_stats=true")
    assert detailed.status_code == 200, detailed.text
    assert [(item["id"], item["image_count"], item["annotated_count"]) for item in detailed.json()] == [
        (dataset.id, 4, 1), (empty.id, 0, 0),
    ]


def test_pull_validates_range_and_defaults_to_qualified(crud_env, login_admin) -> None:
    source = dataset_service.create_dataset("source")
    target = dataset_service.create_dataset("target")
    with crud_env() as db:
        _add_record(db, source.id, "qualified.png", b"qualified")
        _add_record(db, source.id, "unqualified.png", b"unqualified", quality="不合格")
        db.commit()

    client = login_admin(TestClient(app))
    payload = {"target_dataset_id": target.id, "start_time": "2026-09-07T12:00:00+08:00",
               "end_time": "2026-09-07T03:59:59Z"}
    assert client.post("/api/datasets/pull", json=payload).status_code == 400
    payload["end_time"] = "2026-09-07T04:00:00Z"
    response = client.post("/api/datasets/pull", json=payload)
    assert response.status_code == 200, response.text
    assert response.json() == {"pulled": 1}
    assert image_service.list_images(target.id) == ["qualified.png"]
