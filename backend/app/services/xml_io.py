"""四诊标注 XML 读写（T3）——全系统标注 XML 的唯一真相源。

标注写入（T10）与预标注生成（T17）共用本模块，保证 XML 结构一致。

XML schema（锁定决策 2）:
    <annotation>
      <folder>dataset_name</folder>
      <filename>image.jpg</filename>
      <path>relative/path/image.jpg</path>
      <size><width>640</width><height>480</height><depth>3</depth></size>
      <quality>合格</quality>
      <labels>
        <label><category>舌</category><name>舌苔黄</name><confidence>1.0</confidence></label>
      </labels>
    </annotation>

规则:
    - `labels` 容纳 0–4 个 `<label>`（每 category 至多一个）。
    - `confidence`：人工标注 = 1.0；预标注 = [0, 1] 概率。
    - `quality` ∈ {合格, 不合格, ''}。
    - `write_xml` 原子写：临时文件 + `os.replace`，并发/崩溃不会留下半截文件。
    - 序列化确定（canonical）：`write_xml(path, read_xml(path))` 产物逐字节一致。
"""

from __future__ import annotations

import io
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# 序列化（canonical、确定性）
# ---------------------------------------------------------------------------

_FIELD_ORDER: List[str] = ["folder", "filename", "path"]
_SIZE_KEYS: List[str] = ["width", "height", "depth"]


def _fmt_confidence(value: Any) -> str:
    """confidence 规范化：float → 最短可往返十进制表示（str==repr for float）。"""
    return repr(float(value))


def _set_text(elem: ET.Element, value: str) -> None:
    """空字符串文本 → None（序列化为自闭合标签），保证 canonical 唯一形式。"""
    elem.text = value if value else None


def _build_tree(data: Dict[str, Any]) -> ET.Element:
    root = ET.Element("annotation")

    for key in _FIELD_ORDER:
        _set_text(ET.SubElement(root, key), str(data.get(key, "") or ""))

    size: Optional[Dict[str, Any]] = data.get("size")
    if size:  # None / 空 dict → 省略 <size>
        size_elem = ET.SubElement(root, "size")
        for key in _SIZE_KEYS:
            value = size.get(key)
            if value is not None:
                _set_text(ET.SubElement(size_elem, key), str(int(value)))

    _set_text(ET.SubElement(root, "quality"), str(data.get("quality", "") or ""))

    labels_elem = ET.SubElement(root, "labels")
    for label in data.get("labels") or []:
        label_elem = ET.SubElement(labels_elem, "label")
        _set_text(ET.SubElement(label_elem, "category"), str(label.get("category", "") or ""))
        _set_text(ET.SubElement(label_elem, "name"), str(label.get("name", "") or ""))
        ET.SubElement(label_elem, "confidence").text = _fmt_confidence(
            label.get("confidence", 1.0)
        )

    return root


def _serialize(data: Dict[str, Any]) -> bytes:
    tree = ET.ElementTree(_build_tree(data))
    buf = io.BytesIO()
    tree.write(buf, encoding="utf-8", xml_declaration=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------


def _parse_size(size_elem: Optional[ET.Element]) -> Optional[Dict[str, Optional[int]]]:
    """<size> 缺失 → None；子字段缺失 → None。"""
    if size_elem is None:
        return None
    size: Dict[str, Optional[int]] = {}
    for key in _SIZE_KEYS:
        child = size_elem.find(key)
        size[key] = int(child.text) if (child is not None and child.text) else None
    return size


def _parse_confidence(label_elem: ET.Element) -> float:
    """confidence 缺失 → 默认 1.0（人工标注语义）。"""
    conf_elem = label_elem.find("confidence")
    if conf_elem is None or conf_elem.text is None:
        return 1.0
    return float(conf_elem.text)


def _parse_labels(labels_elem: Optional[ET.Element]) -> List[Dict[str, Any]]:
    if labels_elem is None:
        return []
    labels: List[Dict[str, Any]] = []
    for label_elem in labels_elem.findall("label"):
        labels.append(
            {
                "category": label_elem.findtext("category") or "",
                "name": label_elem.findtext("name") or "",
                "confidence": _parse_confidence(label_elem),
            }
        )
    return labels


def read_xml(path: Any) -> Dict[str, Any]:
    """读取标注 XML → dict。容忍缺失 <size>/<labels>/<quality>（各自按缺省处理）。"""
    tree = ET.parse(str(path))
    root = tree.getroot()

    if root.tag != "annotation":
        raise ValueError(
            f"不是四诊标注 XML（根元素为 <{root.tag}>，应为 <annotation>）: {path}"
        )

    return {
        "folder": root.findtext("folder") or "",
        "filename": root.findtext("filename") or "",
        "path": root.findtext("path") or "",
        "size": _parse_size(root.find("size")),
        "quality": root.findtext("quality") or "",
        "labels": _parse_labels(root.find("labels")),
    }


# ---------------------------------------------------------------------------
# 写入（原子）
# ---------------------------------------------------------------------------


def write_xml(path: Any, data: Dict[str, Any]) -> None:
    """原子写标注 XML：同目录临时文件 + os.replace，崩溃/并发不留半截文件。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = _serialize(data)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target)  # 原子替换
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
