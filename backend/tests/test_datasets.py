"""pull_records 回归测试：复制后 XML 元数据（folder/filename/path）必须指向目标数据集。"""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.orm import Base, Dataset, Record
from app.services import datasets as ds
from app.services import xml_io


@pytest.fixture()
def db_env(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    Base.metadata.create_all(engine)
    monkeypatch.setattr(ds, "SessionLocal", Session)
    monkeypatch.setattr(ds.config, "DATASETS_DIR", tmp_path / "datasets")
    return tmp_path


def test_pull_rewrites_xml_metadata(db_env):
    tmp_path = db_env

    with ds.SessionLocal() as db:
        src = Dataset(name="源集")
        tgt = Dataset(name="目标集")
        db.add_all([src, tgt])
        db.commit()
        db.refresh(src)
        db.refresh(tgt)
        src_id, tgt_id = src.id, tgt.id

    src_dir = tmp_path / "datasets" / str(src_id)
    src_dir.mkdir(parents=True)
    img = src_dir / "a.jpg"
    img.write_bytes(b"fake-image-bytes")
    xml_io.write_xml(
        src_dir / "a.xml",
        {
            "folder": "源集",
            "filename": "a.jpg",
            "path": str(img),
            "size": {"width": 64, "height": 64, "depth": 3},
            "quality": "合格",
            "labels": [{"category": "舌", "name": "舌苔黄", "confidence": 1.0}],
        },
    )

    ts = datetime(2024, 1, 1, 12, 0, 0)
    with ds.SessionLocal() as db:
        db.add(
            Record(
                dataset_id=src_id,
                filename="a.jpg",
                image_path=str(img),
                timestamp=ts,
                quality="合格",
            )
        )
        db.commit()

    result = ds.pull_records(
        tgt_id,
        start_time=datetime(2024, 1, 1, 0, 0, 0),
        end_time=datetime(2024, 12, 31, 23, 59, 59),
        quality="全部",
    )
    assert result["pulled"] == 1

    tgt_dir = tmp_path / "datasets" / str(tgt_id)
    data = xml_io.read_xml(tgt_dir / "a.xml")
    assert data["folder"] == "目标集"
    assert data["filename"] == "a.jpg"
    assert data["path"] == str(tgt_dir / "a.jpg")
    # 标签内容应保留
    assert data["labels"] == [{"category": "舌", "name": "舌苔黄", "confidence": 1.0}]
