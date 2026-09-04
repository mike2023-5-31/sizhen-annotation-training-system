"""数据集 ZIP 导出服务（T11，需求 6.5，锁定决策 12）。

导出结构（与 T7 导入约定互逆，保证往返保真）：
    images/<filename>            每记录的图片字节（读自 ``Record.image_path``）
    annotations/<filename>.xml   同名标注 XML 字节（按磁盘原始字节打包，不重序列化）

规则（需求 6.5 + 锁定决策 12）：
- 遍历数据集全部 ``Record`` 行，图片与 XML 均按原字节入包——
  ``import_zip``（T7）解压后逐字节落盘，标签与 quality 与导出前逐字节一致；
- 同名 XML 缺失 → 仍导出图片，**不生成占位空 XML**（「未标注」≠「无 XML」：
  labels 为空的记录同样带 XML，满足 REQ-6.5-2；缺失 XML 的记录再导入时会被
  T7 计入 skipped，属防御路径）；
- 图片文件缺失 → 跳过该记录（防御悬空路径，与 pull_records 同款约定）；
- 空数据集 → 返回合法空 ZIP（不报错，满足 REQ-6.5-4）；
- 数据集不存在 → 抛 ``ValueError``（router 转 404）。
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

from sqlalchemy import select

from app.db import SessionLocal
from app.models.orm import Dataset, Record

XML_EXTENSION = ".xml"


def get_dataset_name(dataset_id: int) -> str:
    """数据集名称（供导出响应的 Content-Disposition 文件名）。不存在 → ValueError。"""
    with SessionLocal() as db:
        dataset = db.get(Dataset, dataset_id)
        if dataset is None:
            raise ValueError(f"数据集 {dataset_id} 不存在")
        return dataset.name


def export_dataset(dataset_id: int) -> bytes:
    """将数据集全部记录（图片 + 同名 XML）打包为内存 ZIP，返回字节流。

    使用 ``io.BytesIO`` + ``zipfile.ZipFile(..., "w", ZIP_DEFLATED)``；
    数据集不存在 → ValueError。
    """
    with SessionLocal() as db:
        dataset = db.get(Dataset, dataset_id)
        if dataset is None:
            raise ValueError(f"数据集 {dataset_id} 不存在")

        records = list(
            db.scalars(
                select(Record)
                .where(Record.dataset_id == dataset_id)
                .order_by(Record.id)
            ).all()
        )

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for rec in records:
                image_path = Path(rec.image_path)
                if not image_path.is_file():
                    continue  # 磁盘图片缺失：跳过悬空记录，不制造空条目
                zf.write(image_path, arcname=f"images/{rec.filename}")
                xml_path = image_path.with_suffix(XML_EXTENSION)
                if xml_path.is_file():
                    zf.write(xml_path, arcname=f"annotations/{xml_path.name}")
        return buf.getvalue()
