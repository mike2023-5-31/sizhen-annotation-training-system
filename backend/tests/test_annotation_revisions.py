"""四人标注的数据基础：冲突、版本快照与XML/数据库回滚。"""
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models.orm import AnnotationEdit, Dataset, Record
from app.routers import annotations, images
from app.services import annotation_service as service, image_service, xml_io


@pytest.fixture
def env(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'annotations.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(service, "SessionLocal", factory)
    monkeypatch.setattr(image_service, "SessionLocal", factory)
    path = tmp_path / "a.png"
    Image.new("RGB", (12, 12)).save(path)
    # 非canonical的旧XML，回滚必须恢复确切原字节。
    path.with_suffix(".xml").write_bytes(b"<annotation>\n<quality/><labels/>\n<sample_id>s1</sample_id></annotation>")
    with factory() as db:
        dataset = Dataset(name="a")
        db.add(dataset)
        db.flush()
        db.add(Record(dataset_id=dataset.id, filename="a.png", image_path=str(path), quality=""))
        db.commit()
        dataset_id = dataset.id
    app = FastAPI()
    app.include_router(annotations.router)
    app.include_router(images.router)
    with TestClient(app) as client:
        yield factory, dataset_id, path.with_suffix(".xml"), client
    engine.dispose()


def test_http_precondition_conflict_and_quality(env):
    factory, dataset_id, path, client = env
    url = f"/datasets/{dataset_id}/image/a.png/labels"
    revision = client.get(url).json()["revision"]
    payload = {"labels": [{"category": "舌", "name": "舌苔黄"}], "quality": "不合格"}
    assert client.put(url, json=payload).status_code == 428
    saved = client.put(url, json={**payload, "expected_revision": revision})
    assert saved.status_code == 200
    assert saved.json()["annotation_state"] == "已标注"
    assert saved.json()["sample_id"] == "s1"
    assert saved.json()["annotation_source"] == "human"
    conflict = client.put(url, json={**payload, "quality": "合格", "expected_revision": revision})
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["current"]["revision"] == saved.json()["revision"]
    assert xml_io.read_xml(path)["quality"] == "不合格"
    with factory() as db:
        audit = list(db.scalars(select(AnnotationEdit)).all())
        assert len(audit) == 1 and audit[0].previous_revision == revision
        assert audit[0].actor_id is None  # 账号系统接入前不伪造操作者


def test_concurrent_edits_only_one_revision_wins(env):
    factory, dataset_id, path, client = env
    revision = xml_io.read_versioned_xml(path)["revision"]
    def edit(name):
        try:
            return service.save_labels(dataset_id, "a.png", [{"category": "舌", "name": name}], "合格",
                                       expected_revision=revision)
        except service.AnnotationConflict:
            return "conflict"
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(edit, ["舌苔黄", "舌质红"]))
    assert outcomes.count("conflict") == 1
    with factory() as db:
        assert len(list(db.scalars(select(AnnotationEdit)))) == 1


def test_database_failure_restores_original_xml(env, monkeypatch):
    factory, dataset_id, path, client = env
    original = path.read_bytes()
    def fail(_):
        raise RuntimeError("DB unavailable")
    monkeypatch.setattr(factory.class_, "commit", fail)
    with pytest.raises(RuntimeError, match="DB unavailable"):
        service.save_labels(dataset_id, "a.png", [], "合格")
    assert path.read_bytes() == original
    with factory() as db:
        assert len(list(db.scalars(select(AnnotationEdit)))) == 0
        assert db.scalar(select(Record)).quality == ""
