"""受限ZIP导入：相对路径配对、真实解码、无XML初始化，绝不覆盖现有资产。"""
from __future__ import annotations

import hashlib
import io
import os
import shutil
import stat
import tempfile
import time
import uuid
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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
XML_EXTENSION = ".xml"


class ImportValidationError(ValueError):
    """归档不符合导入契约（HTTP 400），不是数据集不存在。"""


def _is_image(name: str) -> bool:
    return PurePosixPath(name).suffix.lower() in IMAGE_EXTENSIONS


def _safe_name(info: zipfile.ZipInfo) -> str:
    name = info.filename.replace("\\", "/")
    path = PurePosixPath(name)
    mode = info.external_attr >> 16
    if path.is_absolute() or ".." in path.parts or ":" in name or stat.S_ISLNK(mode):
        raise ImportValidationError(f"不安全的归档路径: {info.filename}")
    if not path.name or any(part.rstrip(" .") != part for part in path.parts):
        raise ImportValidationError(f"非法归档文件名: {info.filename}")
    reserved = {"CON", "PRN", "AUX", "NUL"} | {f"{prefix}{i}" for prefix in ("COM", "LPT") for i in range(1, 10)}
    if path.stem.upper() in reserved or any(c in name for c in '<>"|?*'):
        raise ImportValidationError(f"非法归档文件名: {info.filename}")
    return path.as_posix()


def _resolve_timestamp(value: Any, image_on_disk: Path) -> datetime:
    if value:
        try:
            parsed = datetime.fromisoformat(str(value))
            return parsed.astimezone(timezone.utc).replace(tzinfo=None) if parsed.tzinfo else parsed
        except ValueError:
            pass
    return datetime.fromtimestamp(image_on_disk.stat().st_mtime, timezone.utc).replace(tzinfo=None)


def _image_size(path: Path) -> dict[str, int]:
    with Image.open(path) as image:
        if image.width * image.height > config.MAX_IMAGE_PIXELS:
            raise ValueError("图片像素数超过限制")
        image.verify()
    with Image.open(path) as image:
        image.seek(0)
        image.load()
        return {"width": image.width, "height": image.height, "depth": len(image.getbands())}


def _validate_annotation(data: dict) -> None:
    if data.get("quality", "") not in {"", "合格", "不合格"}:
        raise ValueError("非法质量标记")
    allowed = {item["name"]: set(item["labels"]) for item in CATEGORIES}
    seen = set()
    for label in data.get("labels", []):
        category, name = label["category"], label["name"]
        if category not in allowed or name not in allowed[category]:
            raise ValueError(f"未知标签: {category}/{name}")
        if category in seen:
            raise ValueError(f"当前分类模板每属性只能有一个值: {category}")
        if not 0 <= label.get("confidence", 1) <= 1:
            raise ValueError("标签置信度必须在[0,1]范围内")
        seen.add(category)


def _signature(data: dict) -> tuple:
    return (data.get("quality", ""), tuple(sorted(
        (label["category"], label["name"], label.get("confidence", 1.0))
        for label in data.get("labels", [])
    )))


def import_zip(dataset_id: int, zip_bytes: bytes) -> dict[str, Any]:
    if len(zip_bytes) > config.MAX_UPLOAD_BYTES:
        raise ImportValidationError("ZIP 超过上传大小限制")
    imported, initialized, duplicates = 0, 0, 0
    skipped, errors, mappings = [], [], []
    created: list[Path] = []
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with SessionLocal() as db, tempfile.TemporaryDirectory(dir=config.UPLOAD_DIR, prefix="import-") as temporary:
        # SQLite事务在读取已有记录前获取写锁，避免多worker同时选择相同文件名。
        if db.get_bind().dialect.name == "sqlite":
            db.connection().exec_driver_sql("BEGIN IMMEDIATE")
        dataset = db.get(Dataset, dataset_id)
        if dataset is None:
            raise ValueError(f"数据集 {dataset_id} 不存在")
        dataset_dir = config.DATASETS_DIR / str(dataset_id)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        extracted: dict[str, tuple[str, Path]] = {}
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
                infos = archive.infolist()
                if len(infos) > config.MAX_ARCHIVE_ENTRIES:
                    raise ImportValidationError("ZIP 条目数超过限制")
                if sum(info.file_size for info in infos) > config.MAX_ARCHIVE_BYTES:
                    raise ImportValidationError("ZIP 解压总量超过限制")
                for index, info in enumerate(infos):
                    if info.is_dir():
                        continue
                    name = _safe_name(info)
                    if not (_is_image(name) or name.lower().endswith(".xml")):
                        continue
                    if info.file_size / max(info.compress_size, 1) > config.MAX_ARCHIVE_RATIO:
                        raise ImportValidationError(f"ZIP 压缩比超过限制: {name}")
                    key = name.casefold()
                    if key in extracted:
                        raise ImportValidationError(f"ZIP 重复路径（大小写不敏感）: {name}")
                    target = Path(temporary) / f"{index}{PurePosixPath(name).suffix}"
                    with archive.open(info) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
                    mtime = time.mktime((*info.date_time, 0, 0, -1))
                    os.utime(target, (mtime, mtime))
                    extracted[key] = (name, target)

            records = list(db.scalars(select(Record).where(Record.dataset_id == dataset_id)).all())
            by_hash: dict[str, list[Path]] = {}
            for record in records:
                existing = Path(record.image_path)
                if existing.is_file():
                    digest = hashlib.sha256(existing.read_bytes()).hexdigest()
                    by_hash.setdefault(digest, []).append(existing.with_suffix(".xml"))
            claimed_xml: dict[str, list[str]] = {}
            candidates = []
            for key, (name, source) in sorted(extracted.items()):
                if not _is_image(key):
                    continue
                # 同目录及平台历史 images/... + labels/... 导出结构。
                xml_keys = [key + ".xml", str(PurePosixPath(key).with_suffix(".xml"))]
                if key.startswith("images/"):
                    xml_keys += [prefix + item[len("images/"):] for prefix in ("labels/", "annotations/")
                                 for item in xml_keys.copy()]
                found = [item for item in xml_keys if item in extracted]
                if len(found) > 1:
                    errors.append(f"{name}: 存在多个候选 XML，无法确定配对")
                    continue
                xml_key = found[0] if found else None
                if xml_key:
                    claimed_xml.setdefault(xml_key, []).append(name)
                candidates.append((name, source, xml_key))
            for name, source, xml_key in candidates:
                if xml_key and len(claimed_xml[xml_key]) > 1:
                    errors.append(f"{name}: 多张图片共用 XML，配对存在歧义")
                    continue
                try:
                    size = _image_size(source)
                except Exception as exc:
                    errors.append(f"{name}: 图片解码失败: {exc}")
                    continue
                xml_path = extracted[xml_key][1] if xml_key else None
                try:
                    data = xml_io.read_xml(xml_path) if xml_path else {"labels": [], "quality": ""}
                    _validate_annotation(data)
                except Exception as exc:
                    errors.append(f"{name}: XML 解析失败或标注无效: {exc}")
                    continue
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                existing_xml = by_hash.get(digest, [])
                if existing_xml:
                    matching = any(path.is_file() and _signature(xml_io.read_xml(path)) == _signature(data)
                                   for path in existing_xml)
                    if matching or xml_path is None:
                        skipped.append(name)
                        duplicates += 1
                    else:
                        errors.append(f"{name}: 图片内容重复但标注冲突，保留现有标注")
                    continue
                filename = PurePosixPath(name).name
                destination = dataset_dir / filename
                occupied = {path.stem.casefold() for path in dataset_dir.iterdir()}
                if destination.stem.casefold() in occupied:
                    stem = destination.stem
                    suffix = hashlib.sha256(name.encode()).hexdigest()[:10]
                    destination = dataset_dir / f"{stem}-{suffix}{destination.suffix}"
                    counter = 1
                    while destination.stem.casefold() in occupied:
                        destination = dataset_dir / f"{stem}-{suffix}-{counter}{destination.suffix}"
                        counter += 1
                annotation_path = destination.with_suffix(".xml")
                with destination.open("xb") as output:
                    created.append(destination)
                    with source.open("rb") as input_stream:
                        shutil.copyfileobj(input_stream, output)
                shutil.copystat(source, destination)
                with annotation_path.open("xb"):
                    created.append(annotation_path)
                if xml_path and destination.name == filename:
                    shutil.copyfile(xml_path, annotation_path)
                else:
                    data.update({"filename": destination.name, "folder": dataset.name,
                                 "path": f"{dataset_id}/{destination.name}", "size": size,
                                 "source_path": name, "image_sha256": digest,
                                 "schema_version": "2", "sample_id": str(uuid.uuid4()),
                                 "timestamp_source": "archive_mtime" if not data.get("timestamp") else "annotation"})
                    xml_io.write_xml(annotation_path, data)
                db.add(Record(dataset_id=dataset_id, filename=destination.name,
                              image_path=str(destination.resolve()), quality=data.get("quality", ""),
                              timestamp=_resolve_timestamp(data.get("timestamp"), destination)))
                by_hash.setdefault(digest, []).append(annotation_path)
                mappings.append({"source": name, "filename": destination.name, "sha256": digest})
                imported += 1
                initialized += int(xml_path is None)
            if not candidates and not errors:
                raise ImportValidationError("ZIP 不含有效的候选图片")
            db.commit()
        except BaseException:
            db.rollback()
            for path in reversed(created):
                path.unlink(missing_ok=True)
            raise
    return {"imported": imported, "initialized": initialized, "duplicates": duplicates,
            "skipped": skipped, "errors": errors, "files": mappings}
