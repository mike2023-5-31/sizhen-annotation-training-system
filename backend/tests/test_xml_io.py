"""xml_io 单元测试：往返逐字节一致 / 中文 / 缺失 size / 空 labels / quality。"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from app.services.xml_io import read_xml, write_xml


def _full_data() -> dict:
    return {
        "folder": "数据集A",
        "filename": "image_001.jpg",
        "path": "subdir/image_001.jpg",
        "size": {"width": 640, "height": 480, "depth": 3},
        "quality": "合格",
        "labels": [
            {"category": "舌", "name": "舌苔黄", "confidence": 1.0},
            {"category": "面", "name": "面色红润", "confidence": 1.0},
            {"category": "脉", "name": "浮", "confidence": 1.0},
            {"category": "声音", "name": "清晰", "confidence": 1.0},
        ],
    }


def test_roundtrip_byte_identical(tmp_path: Path) -> None:
    """write → read → write 产物逐字节一致（确定性序列化）。"""
    p1 = tmp_path / "a.xml"
    p2 = tmp_path / "b.xml"
    data = _full_data()
    write_xml(p1, data)
    write_xml(p2, read_xml(p1))
    assert p1.read_bytes() == p2.read_bytes()


def test_roundtrip_field_equality(tmp_path: Path) -> None:
    """read 返回 dict 与原数据字段全等。"""
    p = tmp_path / "a.xml"
    data = _full_data()
    write_xml(p, data)
    assert read_xml(p) == data


def test_chinese_labels_preserved(tmp_path: Path) -> None:
    """中文 category/name/quality 经 UTF-8 往返不失真。"""
    p = tmp_path / "zh.xml"
    data = {
        "folder": "舌诊集",
        "filename": "舌像.jpg",
        "path": "舌像/舌像.jpg",
        "size": {"width": 640, "height": 480, "depth": 3},
        "quality": "不合格",
        "labels": [{"category": "舌", "name": "舌苔黄腻", "confidence": 1.0}],
    }
    write_xml(p, data)
    assert read_xml(p) == data
    raw = p.read_text(encoding="utf-8")
    assert "舌苔黄腻" in raw and "不合格" in raw


def test_missing_size(tmp_path: Path) -> None:
    """缺失 <size> 容错：read 返回 size=None，重写后不产生 <size> 元素。"""
    p = tmp_path / "nosize.xml"
    p.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<annotation><folder>d</folder><filename>a.jpg</filename>"
        "<path>a.jpg</path><quality>合格</quality><labels /></annotation>",
        encoding="utf-8",
    )
    data = read_xml(p)
    assert data["size"] is None

    p2 = tmp_path / "nosize_rewritten.xml"
    write_xml(p2, data)
    tree = ET.parse(str(p2))
    assert tree.getroot().find("size") is None


def test_empty_labels(tmp_path: Path) -> None:
    """<labels /> → labels == []，往返后仍为空且字节一致。"""
    p = tmp_path / "empty.xml"
    write_xml(p, {**{k: _full_data()[k] for k in ("folder", "filename", "path")},
                   "size": None, "quality": "", "labels": []})
    assert read_xml(p)["labels"] == []
    p2 = tmp_path / "empty2.xml"
    write_xml(p2, read_xml(p))
    assert p.read_bytes() == p2.read_bytes()


def test_quality_field_present(tmp_path: Path) -> None:
    """<quality> 元素恒存在：合格 / '' 两种取值均可往返。"""
    for quality in ("合格", ""):
        p = tmp_path / f"q_{quality or 'blank'}.xml"
        data = _full_data()
        data["quality"] = quality
        write_xml(p, data)
        root = ET.parse(str(p)).getroot()
        assert root.find("quality") is not None
        assert read_xml(p)["quality"] == quality


def test_preannotation_confidence(tmp_path: Path) -> None:
    """预标注 confidence ∈ (0,1) 往返保持数值不变。"""
    p = tmp_path / "pre.xml"
    data = {
        "folder": "d",
        "filename": "a.jpg",
        "path": "a.jpg",
        "size": {"width": 224, "height": 224, "depth": 3},
        "quality": "",
        "labels": [{"category": "舌", "name": "舌苔黄", "confidence": 0.87}],
    }
    write_xml(p, data)
    assert read_xml(p)["labels"][0]["confidence"] == 0.87


def test_atomic_write_no_temp_leftover(tmp_path: Path) -> None:
    """原子写后目录内无残留临时文件，且目标可被正常读回。"""
    target = tmp_path / "sub" / "a.xml"  # 父目录不存在，验证 mkdir
    write_xml(target, _full_data())
    leftovers = [f.name for f in tmp_path.rglob("*") if f.is_file() and f.name != "a.xml"]
    assert leftovers == []
    assert read_xml(target)["filename"] == "image_001.jpg"
