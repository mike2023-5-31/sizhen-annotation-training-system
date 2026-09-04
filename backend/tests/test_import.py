"""ZIP 导入服务测试（T7，需求 6.2a）：白名单过滤 / 配对落盘 / 异常与安全。

覆盖：
- 白名单扩展名（.jpg .jpeg .png .bmp .gif，大小写不敏感）过滤；
- 非白名单文件完全忽略（不提取、不建记录）；
- 图片无同名 XML → 计入 skipped，不落盘、不建记录；
- 损坏 ZIP → BadZipFile 上抛；XML 解析失败 → 计入 errors；
- Zip Slip：仅取 basename，杜绝目录穿越；
- 记录 quality 取自 XML、timestamp 优先 XML 否则回退文件 mtime。
"""

from __future__ import annotations

import io
import zipfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zipfile import BadZipFile

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app import config
from app.db import Base
from app.models.orm import Dataset, Record
from app.services import import_service, xml_io


# ---------------------------------------------------------------------------
# 工具 / fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def import_env(tmp_path: Path, monkeypatch):
    """临时 SQLite + 临时 DATASETS_DIR/UPLOAD_DIR，替换 import_service 依赖。"""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test_import.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    monkeypatch.setattr(import_service, "SessionLocal", factory)

    datasets_dir = tmp_path / "datasets"
    upload_dir = tmp_path / "uploads"
    datasets_dir.mkdir()
    upload_dir.mkdir()
    # import_service 经 `config.DATASETS_DIR / config.UPLOAD_DIR` 运行时取值 → 打补丁生效
    monkeypatch.setattr(config, "DATASETS_DIR", datasets_dir)
    monkeypatch.setattr(config, "UPLOAD_DIR", upload_dir)
    return SimpleNamespace(
        factory=factory,
        datasets_dir=datasets_dir,
        upload_dir=upload_dir,
        tmp_path=tmp_path,
    )


def _create_dataset(factory, name: str) -> int:
    with factory() as db:
        ds = Dataset(name=name)
        db.add(ds)
        db.commit()
        db.refresh(ds)
        return ds.id


def _xml_bytes(tmp_path: Path, filename: str, quality: str = "合格") -> bytes:
    """用公共 xml_io.write_xml 生成合法标注 XML，返回字节（供 zip 打包）。"""
    build_dir = tmp_path / "build"
    build_dir.mkdir(exist_ok=True)
    xml_path = build_dir / f"{filename}.xml"
    xml_io.write_xml(
        xml_path,
        {
            "folder": "import-test",
            "filename": filename,
            "path": filename,
            "size": {"width": 64, "height": 64, "depth": 3},
            "quality": quality,
            "labels": [{"category": "舌", "name": "舌苔黄", "confidence": 1.0}],
        },
    )
    return xml_path.read_bytes()


def _make_zip(entries: dict[str, tuple[bytes, tuple[int, int, int, int, int, int]]]) -> bytes:
    """entries: {arcname: (data, date_time)} → ZIP 字节。"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, (data, date_time) in entries.items():
            info = zipfile.ZipInfo(arcname, date_time=date_time)
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)
    return buf.getvalue()


def _records(factory, dataset_id: int) -> list[Record]:
    with factory() as db:
        return list(db.scalars(select(Record).where(Record.dataset_id == dataset_id)).all())


def _filenames(factory, dataset_id: int) -> list[str]:
    return [r.filename for r in _records(factory, dataset_id)]


_DT = (2024, 5, 6, 7, 8, 10)


# ---------------------------------------------------------------------------
# 白名单过滤 + 无 XML 跳过（需求 6.2a）
# ---------------------------------------------------------------------------


def test_whitelist_filtering_and_skips(import_env) -> None:
    """白名单（含大小写）图片与同名 XML 配对导入；无 XML 图片跳过；其余文件忽略。"""
    ds_id = _create_dataset(import_env.factory, "wl")
    zip_bytes = _make_zip(
        {
            "a.jpg": (b"\xff\xd8fake-jpg", _DT),
            "a.xml": (_xml_bytes(import_env.tmp_path, "a.jpg"), _DT),
            "B.PNG": (b"\x89PNGfake", _DT),
            "b.xml": (_xml_bytes(import_env.tmp_path, "B.PNG"), _DT),
            "c.jpeg": (b"fake-jpeg", _DT),
            "c.xml": (_xml_bytes(import_env.tmp_path, "c.jpeg"), _DT),
            "d.gif": (b"GIF89afake", _DT),        # 图片但无 XML → skipped
            "e.bmp": (b"BMfake", _DT),            # 图片但无 XML → skipped
            "notes.txt": (b"hello", _DT),         # 非图片 → 忽略
            "sub/doc.docx": (b"docx", _DT),       # 子目录非图片 → 忽略
            "readme.md": (b"# md", _DT),          # 非图片 → 忽略
            "noext": (b"raw", _DT),               # 无扩展名 → 忽略
        }
    )

    result = import_service.import_zip(ds_id, zip_bytes)

    assert result["imported"] == 3
    assert result["skipped"] == ["d.gif", "e.bmp"]
    assert result["errors"] == []
    assert sorted(_filenames(import_env.factory, ds_id)) == ["B.PNG", "a.jpg", "c.jpeg"]

    # 数据集目录只落盘配对成功的图片 + 同名 XML，非白名单文件一律不落盘
    ds_dir = import_env.datasets_dir / str(ds_id)
    assert sorted(p.name for p in ds_dir.iterdir()) == [
        "B.PNG", "a.jpg", "a.xml", "b.xml", "c.jpeg", "c.xml",
    ]
    assert not (ds_dir / "notes.txt").exists()
    assert not (ds_dir / "d.gif").exists()


def test_image_without_xml_skipped_no_record(import_env) -> None:
    """仅图片无 XML → imported=0、计入 skipped，不落盘、不建记录（需求 6.2a）。"""
    ds_id = _create_dataset(import_env.factory, "no-xml")
    zip_bytes = _make_zip({"only.jpg": (b"fake", _DT)})

    result = import_service.import_zip(ds_id, zip_bytes)

    assert result["imported"] == 0
    assert result["skipped"] == ["only.jpg"]
    assert result["errors"] == []
    assert _records(import_env.factory, ds_id) == []
    assert not (import_env.datasets_dir / str(ds_id) / "only.jpg").exists()


# ---------------------------------------------------------------------------
# 异常路径：损坏 ZIP / XML 解析失败
# ---------------------------------------------------------------------------


def test_corrupted_zip_raises_badzipfile(import_env) -> None:
    """非 ZIP 字节 → BadZipFile 上抛，且不产生任何落盘/记录。"""
    ds_id = _create_dataset(import_env.factory, "corrupt")
    with pytest.raises(BadZipFile):
        import_service.import_zip(ds_id, b"definitely not a zip archive at all")
    assert _records(import_env.factory, ds_id) == []
    assert list((import_env.datasets_dir / str(ds_id)).iterdir()) == []


def test_xml_parse_failure_reported(import_env) -> None:
    """同名 XML 解析失败 → 计入 errors、不建记录；同包内正常配对不受影响。"""
    ds_id = _create_dataset(import_env.factory, "badxml")
    zip_bytes = _make_zip(
        {
            "ok.jpg": (b"fake", _DT),
            "ok.xml": (_xml_bytes(import_env.tmp_path, "ok.jpg"), _DT),
            "bad.png": (b"fake", _DT),
            "bad.xml": (b"<<<not xml>>>", _DT),
        }
    )

    result = import_service.import_zip(ds_id, zip_bytes)

    assert result["imported"] == 1
    assert result["skipped"] == []
    assert len(result["errors"]) == 1
    assert "XML 解析失败" in result["errors"][0]
    assert sorted(_filenames(import_env.factory, ds_id)) == ["ok.jpg"]
    ds_dir = import_env.datasets_dir / str(ds_id)
    assert not (ds_dir / "bad.png").exists()


# ---------------------------------------------------------------------------
# Zip Slip：仅取 basename（安全约束）
# ---------------------------------------------------------------------------


def test_zip_slip_basename_only(import_env) -> None:
    """`../evil.jpg` 与 `deep/nested/pic.jpg` 均按 basename 落盘，无目录穿越。"""
    ds_id = _create_dataset(import_env.factory, "slip")
    zip_bytes = _make_zip(
        {
            "../evil.jpg": (b"evil", _DT),
            "../evil.xml": (_xml_bytes(import_env.tmp_path, "evil.jpg"), _DT),
            "deep/nested/pic.jpg": (b"pic", _DT),
            "deep/nested/pic.xml": (_xml_bytes(import_env.tmp_path, "pic.jpg"), _DT),
        }
    )

    result = import_service.import_zip(ds_id, zip_bytes)

    assert result["imported"] == 2
    assert result["errors"] == []
    assert sorted(_filenames(import_env.factory, ds_id)) == ["evil.jpg", "pic.jpg"]

    ds_dir = import_env.datasets_dir / str(ds_id)
    # 全部落在数据集目录内（basename），绝无逃逸到上层
    assert (ds_dir / "evil.jpg").is_file()
    assert (ds_dir / "pic.jpg").is_file()
    assert not (import_env.tmp_path / "evil.jpg").exists()
    assert not (import_env.datasets_dir / "evil.jpg").exists()
    # 解压临时目录已整体删除：UPLOAD_DIR 无孤儿文件
    assert list(import_env.upload_dir.iterdir()) == []
    # 记录 image_path 均在数据集目录内
    for rec in _records(import_env.factory, ds_id):
        assert Path(rec.image_path).parent == ds_dir.resolve()


# ---------------------------------------------------------------------------
# quality / timestamp 落库
# ---------------------------------------------------------------------------


def test_records_quality_and_timestamp_fallback(import_env) -> None:
    """quality 取自 XML；XML 无 timestamp → 回退解压文件 mtime（zip 条目时间）。"""
    ds_id = _create_dataset(import_env.factory, "meta")
    dt = (2023, 1, 2, 3, 4, 6)
    zip_bytes = _make_zip(
        {
            "q_ok.jpg": (b"ok", dt),
            "q_ok.xml": (_xml_bytes(import_env.tmp_path, "q_ok.jpg", quality="合格"), dt),
            "q_bad.jpg": (b"bad", dt),
            "q_bad.xml": (
                _xml_bytes(import_env.tmp_path, "q_bad.jpg", quality="不合格"), dt,
            ),
        }
    )

    result = import_service.import_zip(ds_id, zip_bytes)
    assert result["imported"] == 2

    by_name = {r.filename: r for r in _records(import_env.factory, ds_id)}
    assert by_name["q_ok.jpg"].quality == "合格"
    assert by_name["q_bad.jpg"].quality == "不合格"
    # 回退路径：timestamp = 落盘文件 mtime（zip 条目 date_time 经 copy2 保留）
    for rec in by_name.values():
        assert rec.timestamp is not None
        assert rec.timestamp == datetime.fromtimestamp(Path(rec.image_path).stat().st_mtime)
        assert rec.timestamp == datetime(2023, 1, 2, 3, 4, 6)
        assert Path(rec.image_path).is_file()


def test_xml_timestamp_takes_priority(import_env, monkeypatch) -> None:
    """XML 提供 timestamp → 优先采用，不回退文件 mtime（前向兼容分支）。"""
    ds_id = _create_dataset(import_env.factory, "ts-priority")
    real_read_xml = xml_io.read_xml

    def _fake_read_xml(path):
        data = real_read_xml(path)
        data["timestamp"] = "2022-03-04T05:06:07"
        return data

    monkeypatch.setattr(import_service.xml_io, "read_xml", _fake_read_xml)
    zip_bytes = _make_zip(
        {
            "t.jpg": (b"fake", (2020, 1, 1, 0, 0, 0)),  # 故意与 XML 时间不同
            "t.xml": (_xml_bytes(import_env.tmp_path, "t.jpg"), (2020, 1, 1, 0, 0, 0)),
        }
    )

    result = import_service.import_zip(ds_id, zip_bytes)
    assert result["imported"] == 1
    rec = _records(import_env.factory, ds_id)[0]
    assert rec.timestamp == datetime(2022, 3, 4, 5, 6, 7)
