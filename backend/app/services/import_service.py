"""ZIP 数据集导入服务（T7，需求 6.2a）。

落盘约定（与 T8/T9/T10/T11 共用，锁定决策 11）：
- 图片与同名 XML 并列存放于 ``data/datasets/<dataset_id>/<filename>``；
- ``Record.image_path`` = 图片绝对路径；``Record.filename`` = basename；
- 白名单扩展名仅 ``.jpg .jpeg .png .bmp .gif``（大小写不敏感），其余文件一律忽略；
- 图片无同名 XML → 跳过（计入 ``skipped``），不落盘、不建记录；
- 同名 XML 解析失败 → 计入 ``errors``，不落盘、不建记录；
- 解压临时目录位于 ``config.UPLOAD_DIR`` 下，结束后整体删除（无孤儿文件）。

时间戳：``read_xml`` 结果若含 ``timestamp`` 键则优先采用（当前 XML schema 无该
元素，属前向兼容）；否则取解压后文件 mtime（zipfile 解压时保留条目原始时间）。

错误分类：``BadZipFile`` 由本函数上抛（router 转 400）；数据集不存在抛
``ValueError``（router 转 404）；单文件级失败只记入返回值的 ``errors``。
"""
from __future__ import annotations

import io
import os
import shutil
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from app import config
from app.db import SessionLocal
from app.models.orm import Dataset, Record
from app.services import xml_io

# 白名单图片扩展名（小写、含点号）——需求 6.2a 锁定
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
XML_EXTENSION = ".xml"


def _is_image(name: str) -> bool:
    """扩展名（大小写不敏感）是否在白名单内。"""
    return Path(name).suffix.lower() in IMAGE_EXTENSIONS


def _zip_mtime(info: zipfile.ZipInfo) -> float:
    """ZIP 条目记录的原始 mtime（epoch 秒，本地时区解释）。"""
    return time.mktime((*info.date_time, 0, 0, -1))


def _resolve_timestamp(value: Any, image_on_disk: Path) -> datetime | None:
    """XML 提供的采集时间优先；缺失/不可解析时回退文件 mtime。"""
    if value is not None:
        try:
            return datetime.fromisoformat(str(value))
        except ValueError:
            pass
    return datetime.fromtimestamp(image_on_disk.stat().st_mtime)


def import_zip(dataset_id: int, zip_bytes: bytes) -> dict[str, Any]:
    """解压 ZIP，识别白名单图片并与同名 XML 配对后导入数据集。

    返回 ``{"imported": N, "skipped": [filenames], "errors": [messages]}``。
    """
    with SessionLocal() as db:
        dataset = db.get(Dataset, dataset_id)
        if dataset is None:
            raise ValueError(f"数据集 {dataset_id} 不存在")

        dataset_dir = config.DATASETS_DIR / str(dataset_id)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        imported = 0
        skipped: list[str] = []
        errors: list[str] = []

        # 解压到 UPLOAD_DIR 下的临时目录（finally 清理，无孤儿文件）
        tmp_dir = Path(
            tempfile.mkdtemp(prefix=f"import-{dataset_id}-", dir=str(config.UPLOAD_DIR))
        )
        try:
            # BadZipFile 由此处上抛 → router 转 400；此时未产生任何落盘
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                extracted: dict[str, Path] = {}  # 小写 basename -> 临时文件路径
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    # 仅取 basename：杜绝 Zip Slip 目录穿越
                    name = PurePosixPath(info.filename).name
                    if not name:
                        continue
                    key = name.lower()
                    if key in extracted:
                        errors.append(f"重复文件名（忽略）: {info.filename}")
                        continue
                    # 非白名单图片、非 XML → 完全忽略（不提取）
                    if not (_is_image(name) or key.endswith(XML_EXTENSION)):
                        continue
                    target = tmp_dir / name
                    with zf.open(info) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    mtime = _zip_mtime(info)
                    os.utime(target, (mtime, mtime))
                    extracted[key] = target

            for img_path in sorted(p for k, p in extracted.items() if _is_image(k)):
                xml_path = extracted.get(PurePosixPath(img_path.name).stem.lower() + XML_EXTENSION)
                if xml_path is None:
                    # 图片无同名 XML → 跳过，不建记录（需求 6.2a）
                    skipped.append(img_path.name)
                    continue

                try:
                    data = xml_io.read_xml(xml_path)
                except Exception as exc:  # ParseError / ValueError（根元素不符）
                    errors.append(f"{img_path.name}: XML 解析失败（{exc}）")
                    continue

                dst_img = dataset_dir / img_path.name
                dst_xml = dataset_dir / xml_path.name
                try:
                    shutil.copy2(img_path, dst_img)
                    shutil.copy2(xml_path, dst_xml)
                except OSError as exc:
                    errors.append(f"{img_path.name}: 落盘失败（{exc}）")
                    dst_img.unlink(missing_ok=True)
                    dst_xml.unlink(missing_ok=True)
                    continue

                db.add(
                    Record(
                        dataset_id=dataset_id,
                        filename=img_path.name,
                        image_path=str(dst_img.resolve()),
                        timestamp=_resolve_timestamp(data.get("timestamp"), dst_img),
                        quality=data.get("quality") or "",
                    )
                )
                imported += 1

            db.commit()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return {"imported": imported, "skipped": skipped, "errors": errors}
