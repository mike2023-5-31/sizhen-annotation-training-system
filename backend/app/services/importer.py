"""ZIP 数据导入（6.2a）：解压识别图片 + 同名 XML 配对落库。"""
from __future__ import annotations

import hashlib
import io
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image
from sqlalchemy import select

from app import config
from app.db import SessionLocal
from app.labels_config import CATEGORIES
from app.models.orm import Dataset, Record
from app.services import xml_io

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


class ImportValidationError(ValueError):
    """ZIP 损坏或不符合导入契约（路由层转 400）。"""


def _is_image(name: str) -> bool:
    return PurePosixPath(name).suffix.lower() in IMAGE_SUFFIXES


def _safe_basename(name: str) -> str:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ImportValidationError(f"非法归档路径: {name}")
    return path.parts[-1]


def _read_image_size(path: Path) -> dict[str, int]:
    with Image.open(path) as image:
        if image.width * image.height > config.MAX_IMAGE_PIXELS:
            raise ValueError("图片像素数超过限制")
        return {
            "width": image.width,
            "height": image.height,
            "depth": len(image.getbands()),
        }


def _validate_labels(data: dict[str, Any]) -> None:
    if data.get("quality", "") not in {"", "合格", "不合格"}:
        raise ValueError("非法质量标记")
    allowed = {item["name"]: set(item["labels"]) for item in CATEGORIES}
    seen: set[str] = set()
    for label in data.get("labels", []):
        category, name = label["category"], label["name"]
        if category not in allowed or name not in allowed[category]:
            raise ValueError(f"未知标签: {category}/{name}")
        if category in seen:
            raise ValueError(f"类别 {category} 出现多个标签")
        seen.add(category)


def _mtime_to_datetime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).replace(tzinfo=None)


def import_zip(dataset_id: int, payload: bytes) -> dict[str, Any]:
    """解压 ZIP，把白名单图片与同名 XML 配对导入目标数据集。

    返回 ``{"imported", "skipped", "errors"}``；损坏 ZIP / 无有效图片抛
    ImportValidationError；数据集不存在抛 ValueError。写入前先在临时目录解压
    校验，全部成功后统一落盘并建记录，失败时清理已写文件，不残留孤儿文件。
    """
    if len(payload) > config.MAX_UPLOAD_BYTES:
        raise ImportValidationError("ZIP 超过上传大小限制")

    with SessionLocal() as db:
        dataset = db.get(Dataset, dataset_id)
        if dataset is None:
            raise ValueError(f"数据集 {dataset_id} 不存在")
        dataset_dir = Path(config.DATASETS_DIR) / str(dataset_id)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        existing_hashes = {
            hashlib.sha256(Path(rec.image_path).read_bytes()).hexdigest()
            for rec in db.scalars(select(Record).where(Record.dataset_id == dataset_id))
            if Path(rec.image_path).is_file()
        }
        occupied_stems = {p.stem.casefold() for p in dataset_dir.iterdir()}

    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise ImportValidationError("不是有效的 ZIP 文件") from exc

    with archive, tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        xml_entries: dict[str, str] = {}
        image_entries: list[tuple[str, str]] = []

        for info in archive.infolist():
            if info.is_dir():
                continue
            basename = _safe_basename(info.filename)
            if not basename:
                continue
            lower = basename.casefold()
            if lower.endswith(".xml"):
                xml_entries[lower] = info.filename
            elif _is_image(basename):
                image_entries.append((basename, info.filename))

        if not image_entries:
            raise ImportValidationError("ZIP 不含任何有效图片")

        imported = 0
        skipped: list[str] = []
        errors: list[str] = []
        created_files: list[Path] = []
        new_records: list[Record] = []

        try:
            for basename, arcname in image_entries:
                stem = PurePosixPath(basename).stem.casefold()
                xml_arcname = xml_entries.get(stem + ".xml") or xml_entries.get(
                    basename.casefold() + ".xml"
                )

                try:
                    image_bytes = archive.read(arcname)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{basename}: 读取失败: {exc}")
                    continue

                digest = hashlib.sha256(image_bytes).hexdigest()
                if digest in existing_hashes:
                    skipped.append(basename)
                    continue

                tmp_image = tmp_dir / basename
                tmp_image.write_bytes(image_bytes)
                try:
                    size = _read_image_size(tmp_image)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{basename}: 图片解码失败: {exc}")
                    continue

                if xml_arcname is None:
                    errors.append(f"{basename}: 缺少同名 XML 标注")
                    continue

                try:
                    xml_bytes = archive.read(xml_arcname)
                    tmp_xml = tmp_dir / f"{stem}.xml"
                    tmp_xml.write_bytes(xml_bytes)
                    data = xml_io.read_xml(tmp_xml)
                    _validate_labels(data)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{basename}: XML 解析失败或标注无效: {exc}")
                    continue

                dest_name = basename
                if stem in occupied_stems:
                    suffix = hashlib.sha256(basename.encode()).hexdigest()[:8]
                    dest_name = f"{PurePosixPath(basename).stem}-{suffix}{PurePosixPath(basename).suffix}"

                dest_image = dataset_dir / dest_name
                dest_xml = dest_image.with_suffix(".xml")
                dest_image.write_bytes(image_bytes)
                data.update(
                    {
                        "folder": dataset.name,
                        "filename": dest_name,
                        "path": str(dest_image),
                        "size": size,
                    }
                )
                xml_io.write_xml(dest_xml, data)

                new_records.append(
                    Record(
                        dataset_id=dataset_id,
                        filename=dest_name,
                        image_path=str(dest_image),
                        timestamp=_mtime_to_datetime(dest_image),
                        quality=data.get("quality", ""),
                    )
                )
                created_files.extend([dest_image, dest_xml])
                occupied_stems.add(PurePosixPath(dest_name).stem.casefold())
                existing_hashes.add(digest)
                imported += 1

            with SessionLocal() as db:
                db.add_all(new_records)
                db.commit()
        except BaseException:
            for path in reversed(created_files):
                path.unlink(missing_ok=True)
            raise

    return {"imported": imported, "skipped": skipped, "errors": errors}
