"""v2 ZIP契约：使用真实图片验证安全、配对、未知标签和数据保护。"""
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app import config
from app.db import Base
from app.models.orm import Dataset, Record
from app.services import import_service as importer, xml_io


@pytest.fixture
def env(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(importer, "SessionLocal", factory)
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")
    with factory() as db:
        dataset = Dataset(name="test")
        db.add(dataset)
        db.commit()
        dataset_id = dataset.id
    yield SimpleNamespace(factory=factory, dataset_id=dataset_id, root=config.DATASETS_DIR / str(dataset_id))
    engine.dispose()


def picture(fmt="PNG", color=(90, 50, 30)):
    output = io.BytesIO()
    Image.new("RGB", (16, 12), color).save(output, format=fmt)
    return output.getvalue()


def archive(entries):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for name, value in entries.items():
            z.writestr(name, value)
    return output.getvalue()


def annotation(quality="合格", name="舌苔黄"):
    return (f"<annotation><quality>{quality}</quality><labels><label>"
            f"<category>舌</category><name>{name}</name><confidence>1.0</confidence>"
            "</label></labels></annotation>").encode()


def records(env):
    with env.factory() as db:
        return list(db.scalars(select(Record).where(Record.dataset_id == env.dataset_id)).all())


def test_all_image_formats_and_missing_xml(env):
    entries = {f"image{i}.{ext}": picture(fmt, (i * 25, 50, 30))
               for i, (ext, fmt) in enumerate([("jpg", "JPEG"), ("jpeg", "JPEG"), ("PNG", "PNG"), ("bmp", "BMP"), ("gif", "GIF")])}
    entries["notes.txt"] = b"ignored"
    result = importer.import_zip(env.dataset_id, archive(entries))
    assert result["imported"] == result["initialized"] == 5
    assert not result["errors"] and not result["skipped"]
    for record in records(env):
        data = xml_io.read_xml(Path(record.image_path).with_suffix(".xml"))
        assert data["labels"] == [] and data["quality"] == ""
        assert data["size"]["width"] == 16
        assert data["sample_id"]
        assert data["timestamp_source"] == "archive_mtime"


def test_corrupt_image_and_xml_are_isolated(env):
    result = importer.import_zip(env.dataset_id, archive({
        "ok.jpg": picture("JPEG"), "ok.xml": annotation(),
        "bad.png": b"fake image", "broken.png": picture(color=(11, 12, 13)),
        "broken.xml": b"<<<invalid>>>",
    }))
    assert result["imported"] == 1
    assert len(result["errors"]) == 2
    assert [r.filename for r in records(env)] == ["ok.jpg"]
    assert (env.root / "ok.xml").read_bytes() == annotation()


def test_relative_pairing_same_basename_and_platform_export_layout(env):
    result = importer.import_zip(env.dataset_id, archive({
        "a/image.png": picture(), "a/image.xml": annotation(),
        "b/image.png": picture(color=(2, 3, 4)), "b/image.xml": annotation("不合格"),
        "images/third.png": picture(color=(5, 6, 7)), "labels/third.xml": annotation(),
    }))
    assert result["imported"] == 3 and not result["errors"]
    assert len({r.filename for r in records(env)}) == 3
    assert sorted(r.quality for r in records(env)) == ["不合格", "合格", "合格"]
    assert len(list(env.root.glob("*.xml"))) == 3


def test_shared_xml_is_not_guessed(env):
    result = importer.import_zip(env.dataset_id, archive({
        "a.jpg": picture("JPEG"), "a.png": picture(), "a.xml": annotation(),
    }))
    assert result["imported"] == 0 and len(result["errors"]) == 2
    assert records(env) == []


@pytest.mark.parametrize("path", ["../evil.png", "/evil.png", "C:/evil.png", r"..\evil.png", "NUL.png"])
def test_unsafe_path_rejected_without_artifacts(env, path):
    with pytest.raises(importer.ImportValidationError):
        importer.import_zip(env.dataset_id, archive({path: picture()}))
    assert records(env) == []
    assert not list(env.root.iterdir())
    assert not list(config.UPLOAD_DIR.iterdir())


def test_duplicate_content_and_conflicting_annotations_never_overwrite(env):
    first = archive({"a.png": picture(), "a.xml": annotation()})
    assert importer.import_zip(env.dataset_id, first)["imported"] == 1
    before = (env.root / "a.xml").read_bytes()
    assert importer.import_zip(env.dataset_id, first)["duplicates"] == 1
    result = importer.import_zip(env.dataset_id, archive({"a.png": picture(), "a.xml": annotation("不合格")}))
    assert result["imported"] == 0 and "标注冲突" in result["errors"][0]
    assert (env.root / "a.xml").read_bytes() == before
    assert len(records(env)) == 1


def test_different_content_same_name_does_not_overwrite(env):
    original = picture()
    importer.import_zip(env.dataset_id, archive({"a.png": original}))
    result = importer.import_zip(env.dataset_id, archive({"a.png": picture(color=(5, 5, 5))}))
    assert result["imported"] == 1
    assert (env.root / "a.png").read_bytes() == original
    assert len(records(env)) == 2


def test_concurrent_reimports_are_idempotent(env):
    data = archive({"a.png": picture()})
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: importer.import_zip(env.dataset_id, data), range(2)))
    assert sum(r["imported"] for r in results) == 1
    assert sum(r["duplicates"] for r in results) == 1
    assert len(records(env)) == 1


def test_rollback_cleans_only_new_files(env, monkeypatch):
    importer.import_zip(env.dataset_id, archive({"old.png": picture()}))
    before = (env.root / "old.png").read_bytes()
    def fail(_):
        raise RuntimeError("disk/DB failure")
    monkeypatch.setattr(env.factory.class_, "commit", fail)
    with pytest.raises(RuntimeError):
        importer.import_zip(env.dataset_id, archive({"new.png": picture(color=(1, 4, 9))}))
    assert (env.root / "old.png").read_bytes() == before
    assert not (env.root / "new.png").exists()
    assert not (env.root / "new.xml").exists()
    assert len(records(env)) == 1


@pytest.mark.parametrize("limit,value,pattern", [
    ("MAX_UPLOAD_BYTES", 1, "上传"), ("MAX_ARCHIVE_BYTES", 1, "解压"),
    ("MAX_ARCHIVE_ENTRIES", 0, "条目"), ("MAX_ARCHIVE_RATIO", 0.1, "压缩比"),
])
def test_archive_limits(env, monkeypatch, limit, value, pattern):
    monkeypatch.setattr(config, limit, value)
    with pytest.raises(importer.ImportValidationError, match=pattern):
        importer.import_zip(env.dataset_id, archive({"a.png": picture()}))
    assert records(env) == []


def test_xml_entities_unknown_labels_and_pixel_limit(env, monkeypatch):
    result = importer.import_zip(env.dataset_id, archive({
        "a.png": picture(),
        "a.xml": b'<!DOCTYPE annotation [<!ENTITY x "bad">]><annotation>&x;</annotation>',
        "b.png": picture(color=(1, 1, 1)), "b.xml": annotation(name="unknown"),
    }))
    assert result["imported"] == 0 and len(result["errors"]) == 2
    monkeypatch.setattr(config, "MAX_IMAGE_PIXELS", 1)
    assert importer.import_zip(env.dataset_id, archive({"c.png": picture()}))["imported"] == 0


def test_timestamp_and_metadata_roundtrip(env, tmp_path):
    xml_path = tmp_path / "stamp.xml"
    xml_io.write_xml(xml_path, {"labels": [], "quality": "",
        "timestamp": "2026-09-07T12:00:00+08:00", "sample_id": "sample-1"})
    result = importer.import_zip(env.dataset_id, archive({"a.png": picture(), "a.xml": xml_path.read_bytes()}))
    assert result["imported"] == 1
    assert records(env)[0].timestamp == datetime(2026, 9, 7, 4, 0)
    assert xml_io.read_xml(env.root / "a.xml")["sample_id"] == "sample-1"


def test_invalid_zip_and_empty_archive(env):
    with pytest.raises(zipfile.BadZipFile):
        importer.import_zip(env.dataset_id, b"not ZIP")
    with pytest.raises(importer.ImportValidationError, match="不含"):
        importer.import_zip(env.dataset_id, archive({"readme.txt": b"text"}))
