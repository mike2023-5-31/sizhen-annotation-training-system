"""推理服务（T17，需求 6.15/6.16/6.17/6.18）。

- ``load_model_from_dir(path)``：读取 T13 训练的 checkpoint（dict：``model_state``
  /``architecture``/``num_classes``/``category_names``/...），按其中的结构与类别数
  经 ``factory.build_model`` 重建网络并载权，返回 ``(model, meta)``，模型置于
  eval 模式。无法识别结构的第三方 .pth → 明确 ``ValueError``（REQ-6.16-3）。
- ``predict_dataset(dataset_id, model)``：整集批量前向（``torch.no_grad`` +
  ``model.eval()``），无数据增强，仅 ``Resize((224, 224)) + ToTensor``；每个
  分类头独立 softmax，返回每图 ``{"filename", "predictions":
  [{"category","label","confidence"} ×4]}``（每类 argmax 子标签 + 概率，
  软输出保留概率，REQ-6.16-2）。
- ``generate_preannotation_xml(dataset_id, model)``：仅对**尚无人工标注**
  （同名 XML 的 ``labels`` 为空）的图片，用 ``xml_io.write_xml`` 生成与人工
  标注完全同 schema 的预标注 XML（confidence = 预测概率，REQ-6.18-2），
  写到图片旁（同一位置，标注界面 T10 直接读取），不覆盖已有人工标注
  （REQ-6.18-4）。返回生成数量。

约束（锁定决策 8 / 决策 6 精神 / 需求 6.7）：
- 结构读取仅保证 resnet18/resnet50；``architecture == "custom"`` 时按 checkpoint
  内 ``custom_model_path`` 经 ``factory.load_custom_model`` 重建并做输出契约
  包装，缺少 ``custom_model_path`` 或文件缺失 → 明确 ``ValueError``；
- 其余 ``architecture`` 一律按「无法识别」处理，显式报错，不猜测结构；
- 推理前校验模型类别数与当前 ``labels.yaml`` 一致（4 头类别数逐一相等），
  taxonomy 不一致 → 硬失败，绝不静默按错误下标映射子标签名。
"""
from __future__ import annotations

from collections.abc import Callable
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from typing import Any
from uuid import uuid4

import torch
import torch.nn as nn
from PIL import Image
from sqlalchemy import select
from torchvision import transforms

from app import config
from app.db import SessionLocal
from app.labels_config import CATEGORIES, num_classes_per_head
from app.models import factory
from app.models.orm import Dataset, Record
from app.services import checkpoint_io, xml_io

# 系统可自动识别结构的 backbone（锁定决策 8：仅保证 ResNet18/50）
_SUPPORTED_ARCHITECTURES = {"resnet18", "resnet50"}

# 推理预处理：无数据增强，仅强制 224×224（各头共享同一主干，要求固定输入尺寸）
_INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

# 批量推理的 batch size（CPU 友好）
_BATCH_SIZE = 16

# 预测概率取整位数（软输出数值稳定，往返保真）
_CONFIDENCE_ROUND = 6

_UNRECOGNIZED_MSG = "无法识别模型结构，请使用系统训练产物"
_RECENT_BATCH_JOBS = 4
logger = logging.getLogger(__name__)


class DatasetNotFound(ValueError):
    """目标数据集不存在。"""


def _category_labels() -> list[list[str]]:
    """当前标签体系：CATEGORIES 顺序的每类子标签名列表。"""
    return checkpoint_io.current_class_labels()


def _category_names() -> list[str]:
    """当前标签体系：4 个 category 名（舌/面/脉/声音）。"""
    return [c["name"] for c in CATEGORIES]


def _resolve_custom_model_path(rel_path: str) -> Path:
    """把 checkpoint 内的 custom_model_path 解析到 MODEL_DIR 内（需求 6.7，
    NF-7.4 防路径穿越），返回绝对路径。

    Raises:
        ValueError: 绝对路径 / 非 .py / ``..`` 穿越出模型目录 / 文件不存在。
    """
    requested = Path(rel_path)
    if requested.is_absolute():
        raise ValueError("custom_model_path 仅支持模型目录内的相对路径")
    if requested.suffix.lower() != ".py":
        raise ValueError("custom_model_path 必须为 .py 模型文件")
    model_dir = config.MODEL_DIR.resolve()
    resolved = (model_dir / requested).resolve()
    if resolved != model_dir and model_dir not in resolved.parents:
        raise ValueError("custom_model_path 越出模型目录（非法路径）")
    if not resolved.is_file():
        raise ValueError(f"自定义模型文件不存在：{requested}")
    return resolved


def load_model_from_dir(
    path: str, *, include_digest: bool = False, allow_custom: bool = True
) -> tuple[nn.Module, dict[str, Any]]:
    """从 checkpoint 文件重建网络并载权（REQ-6.16-3 / REQ-6.18-3）。

    checkpoint 契约（T13 产出）：dict 含 ``model_state``（state_dict）、
    ``architecture``（'resnet18'/'resnet50'/'custom'）、``num_classes``（长度 4 的
    类别数列表）、``category_names``（4 个 category 名）等元数据；
    ``architecture == "custom"`` 时另需 ``custom_model_path``（模型目录内相对
    .py 路径）以重建结构。

    Args:
        path: checkpoint 文件路径（.pth，位于模型目录 MODEL_DIR 内）。

    Returns:
        ``(model, meta)``：model 为 eval 模式、已载权的四头分类器；
        meta 为 checkpoint 元数据 dict（architecture/custom_model_path/
        num_classes/category_names/dropout 等原样保留）。

    Raises:
        ValueError: 文件不存在 / 非 checkpoint dict / 缺 ``architecture`` 或
            ``num_classes``（第三方权重）/ 结构不受支持 / custom 存档缺
            ``custom_model_path`` 或模型文件缺失 / 缺 ``model_state`` /
            权重与结构不匹配。
    """
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise ValueError(f"模型文件不存在：{ckpt_path}")

    checkpoint, source_sha256 = checkpoint_io.read_checkpoint(ckpt_path, include_digest=include_digest)

    architecture = checkpoint.get("architecture")
    num_classes = checkpoint.get("num_classes")
    model_state = checkpoint.get("model_state")

    if not isinstance(architecture, str) or not isinstance(num_classes, list):
        raise ValueError(_UNRECOGNIZED_MSG)
    is_custom = architecture == "custom"
    if architecture not in _SUPPORTED_ARCHITECTURES and not (is_custom and allow_custom):
        raise ValueError(_UNRECOGNIZED_MSG)
    if not isinstance(model_state, dict):
        raise ValueError(f"checkpoint 缺少 model_state 权重（文件损坏）：{ckpt_path}")
    checkpoint_io.validate_taxonomy(checkpoint)
    try:
        dropout = float(checkpoint.get("dropout", 0.0))
    except (ValueError, TypeError) as exc:
        raise ValueError("存档 dropout 无效") from exc

    if is_custom:
        custom_model_path = checkpoint.get("custom_model_path")
        if not isinstance(custom_model_path, str) or not custom_model_path.strip():
            raise ValueError(
                "custom 模型存档缺少 custom_model_path，无法重建模型结构"
            )
        model = factory.load_custom_model(
            str(_resolve_custom_model_path(custom_model_path)), list(num_classes)
        )
        model = factory.wrap_custom_model(model, list(num_classes))
    else:
        model = factory.build_model(
            architecture,
            list(num_classes),
            dropout=dropout,
        )
    try:
        model.load_state_dict(model_state)
    except Exception as exc:
        raise ValueError(
            f"权重与模型结构不匹配（checkpoint 损坏或非本系统产物）：{exc}"
        ) from exc

    model.eval()
    meta = {
        "architecture": architecture,
        "custom_model_path": checkpoint.get("custom_model_path"),
        "num_classes": list(num_classes),
        "category_names": list(checkpoint.get("category_names") or []),
        "dropout": dropout,
        "source_sha256": source_sha256,
    }
    return model, meta


def _validate_model_taxonomy(model: nn.Module) -> None:
    """校验模型 4 头类别数/类别名与当前 labels.yaml 一致，否则硬失败。

    直接传入模型时额外校验头部维度。存档中的标签顺序在加载时校验。
    """
    model_nc = getattr(model, "num_classes", None)
    current_nc = num_classes_per_head()
    if model_nc is not None and list(model_nc) != current_nc:
        raise ValueError(
            f"模型类别数 {list(model_nc)} 与当前标签体系 {current_nc} 不一致，"
            "请确认 labels.yaml 未变更或重新训练模型"
        )


def _predict_batch(model: nn.Module, batch: torch.Tensor) -> list[dict[str, Any]]:
    """单 batch 前向 → 每张图 4 个分类头各自的 softmax 概率（B, n_cat）。"""
    with torch.no_grad():
        logits = model(batch)
    probs = {name: torch.softmax(lg, dim=1).tolist() for name, lg in logits.items()}
    # 按 4 个 category 顺序重排为 (B, 4, n_cat) 视图
    head_order = _category_names()
    return [
        [probs[name][i] for name in head_order] for i in range(batch.size(0))
    ]


def _prediction_entry(filename: str, head_probs: list[list[float]]) -> dict[str, Any]:
    """一张图 → {"filename", "predictions": [{category, label, confidence} ×4]}。"""
    labels_by_category = _category_labels()
    predictions: list[dict[str, Any]] = []
    for category, labels, probs in zip(_category_names(), labels_by_category, head_probs):
        idx = int(max(range(len(probs)), key=probs.__getitem__))
        predictions.append(
            {
                "category": category,
                "label": labels[idx],
                "confidence": round(float(probs[idx]), _CONFIDENCE_ROUND),
            }
        )
    return {"filename": filename, "predictions": predictions}


def _iter_dataset_records(dataset_id: int, limit: int | None = None) -> list[Record]:
    """数据集存在性校验 + 全部记录（filename 升序）。不存在 → ValueError。"""
    with SessionLocal() as db:
        if db.get(Dataset, dataset_id) is None:
            raise DatasetNotFound(f"数据集 {dataset_id} 不存在")
        query = select(Record).where(Record.dataset_id == dataset_id).order_by(Record.filename)
        return list(db.scalars(query.limit(limit)).all())


def _load_image_tensor(record: Record) -> torch.Tensor:
    try:
        with Image.open(record.image_path) as image:
            if image.width * image.height > config.MAX_IMAGE_PIXELS:
                raise ValueError("图片像素超过限制")
            return _INFERENCE_TRANSFORM(image.convert("RGB"))
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError(f"图片读取失败：{record.filename}（{exc}）") from exc


def comparison_images(dataset_id: int, limit: int) -> list[tuple[str, torch.Tensor]]:
    records = _iter_dataset_records(dataset_id, limit)
    if not records:
        raise ValueError("所选数据集没有可对比的图片")
    return [(record.filename, _load_image_tensor(record).unsqueeze(0)) for record in records]


def predict_dataset(dataset_id: int, model: nn.Module) -> list[dict[str, Any]]:
    """整集批量推理：每图每类预测子标签 + 概率（REQ-6.16）。

    预处理无数据增强，仅 ``Resize((224, 224)) + ToTensor``；``torch.no_grad`` +
    ``model.eval()``。返回结果条数 == 数据集图片数（filename 升序），每条含
    4 个 ``{"category", "label", "confidence"}``（label = 该头 argmax 子标签，
    confidence = 该子标签 softmax 概率）。

    Raises:
        ValueError: 数据集不存在 / 模型 taxonomy 与当前标签体系不一致 /
            图片文件缺失或损坏（含文件名）。
    """
    return _predict_records(_iter_dataset_records(dataset_id), model)


def _predict_records(
    records: list[Record],
    model: nn.Module,
    on_batch: Callable[[int], None] | None = None,
) -> list[dict[str, Any]]:
    _validate_model_taxonomy(model)
    model.eval()
    results: list[dict[str, Any]] = []
    for start in range(0, len(records), _BATCH_SIZE):
        chunk = records[start : start + _BATCH_SIZE]
        batch = torch.stack([_load_image_tensor(rec) for rec in chunk])
        head_probs = _predict_batch(model, batch)
        for rec, probs in zip(chunk, head_probs):
            results.append(_prediction_entry(rec.filename, probs))
        if on_batch is not None:
            on_batch(len(results))
    return results


class BatchInferenceBusy(Exception):
    """The workspace has a batch inference job in progress."""


class BatchInferenceJobs:
    """Keep one active CPU batch and a bounded set of recent results per worker."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def start(self, dataset_id: int, path: Path) -> str:
        records = _iter_dataset_records(dataset_id)
        with self._lock:
            if any(job["status"] == "running" for job in self._jobs.values()):
                raise BatchInferenceBusy
            if len(self._jobs) >= _RECENT_BATCH_JOBS:
                del self._jobs[next(iter(self._jobs))]
            job_id = uuid4().hex
            self._jobs[job_id] = {
                "job_id": job_id, "status": "running", "processed": 0,
                "total": len(records), "predictions": [], "error": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            worker = Thread(target=self._run, args=(job_id, path, records), daemon=True)
            try:
                worker.start()
            except RuntimeError:
                del self._jobs[job_id]
                raise
            return job_id

    def status(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job is not None else None

    def list(self) -> list[dict[str, Any]]:
        """作业中心：最近任务摘要（不返回整批预测，只给数量）。"""
        with self._lock:
            jobs = sorted(
                self._jobs.values(),
                key=lambda job: job.get("created_at", ""),
                reverse=True,
            )
            return [
                {
                    **{key: value for key, value in job.items() if key != "predictions"},
                    "prediction_count": len(job.get("predictions") or []),
                }
                for job in jobs
            ]

    def _update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            self._jobs[job_id].update(fields)

    def _run(self, job_id: str, path: Path, records: list[Record]) -> None:
        try:
            model, _ = load_model_from_dir(str(path))
            predictions = _predict_records(
                records, model, lambda processed: self._update(job_id, processed=processed),
            )
        except ValueError as exc:
            self._update(job_id, status="failed", error=str(exc))
        except Exception:
            logger.exception("Batch inference job %s failed", job_id)
            self._update(job_id, status="failed", error="批量推理失败，请联系管理员查看日志")
        else:
            self._update(job_id, status="completed", predictions=predictions)


batch_jobs = BatchInferenceJobs()


def _read_existing_xml(xml_path: Path) -> dict[str, Any] | None:
    """同名 XML 存在则读回，否则 None（缺 XML 视同未标注）。"""
    if xml_path.is_file():
        return xml_io.read_xml(xml_path)
    return None


def generate_preannotation_xml(
    dataset_id: int, model: nn.Module, *, confidence_threshold: float = 0.0
) -> int:
    """基于预测结果生成预标注 XML（REQ-6.18），返回生成数量。

    规则：
    - 跳过已有标签或明确为人工编辑的 XML（包括人工清空标签）；
    - 预标注与人工标注共用 ``xml_io.write_xml``（同 schema，REQ-6.18-1），
      写到图片旁（同一位置，标注界面 T10 直接读取）；
    - 保存达到阈值的预测标签，confidence = 预测概率 [0, 1]；
    - 保留既有 XML 的 folder/filename/path/size/quality（缺失时以数据集名/
      文件名兜底，quality 空字符串）。
    """
    predictions = predict_dataset(dataset_id, model)

    written = 0
    for entry in predictions:
        filename = entry["filename"]
        labels = [{"category": p["category"], "name": p["label"], "confidence": p["confidence"]}
                  for p in entry["predictions"] if p["confidence"] >= confidence_threshold]
        if not labels:
            continue
        with SessionLocal() as db:
            # 与人工保存使用同一 SQLite 写锁；推理完成后再加锁读最新 XML。
            if db.get_bind().dialect.name == "sqlite":
                db.connection().exec_driver_sql("BEGIN IMMEDIATE")
            rec = db.scalar(select(Record).where(Record.dataset_id == dataset_id, Record.filename == filename))
            if rec is None:
                continue
            xml_path = Path(rec.image_path).with_suffix(".xml")
            existing = _read_existing_xml(xml_path) or {}
            if existing.get("labels") or existing.get("annotation_source") == "human":
                continue
            data = {
                **existing,
                "annotation_source": "model",
                "folder": existing.get("folder") or rec.dataset.name,
                "filename": existing.get("filename") or filename,
                "path": existing.get("path") or f"{dataset_id}/{filename}",
                "size": existing.get("size"),
                "quality": existing.get("quality", rec.quality or ""),
                "labels": labels,
            }
            xml_io.write_xml(xml_path, data)
        written += 1
    return written
