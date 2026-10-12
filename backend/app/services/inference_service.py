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

from pathlib import Path
from typing import Any

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
from app.services import xml_io

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


def _category_labels() -> list[list[str]]:
    """当前标签体系：CATEGORIES 顺序的每类子标签名列表。"""
    return [list(c["labels"]) for c in CATEGORIES]


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


def load_model_from_dir(path: str) -> tuple[nn.Module, dict[str, Any]]:
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

    try:
        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except Exception:
        # 兼容含非常规对象的系统产物（fallback 放宽；文件来自管理员可控的模型目录）
        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, dict):
        raise ValueError(_UNRECOGNIZED_MSG)

    architecture = checkpoint.get("architecture")
    num_classes = checkpoint.get("num_classes")
    model_state = checkpoint.get("model_state")

    if architecture is None or num_classes is None:
        raise ValueError(_UNRECOGNIZED_MSG)
    is_custom = architecture == "custom"
    if not is_custom and architecture not in _SUPPORTED_ARCHITECTURES:
        raise ValueError(_UNRECOGNIZED_MSG)
    if model_state is None:
        raise ValueError(f"checkpoint 缺少 model_state 权重（文件损坏）：{ckpt_path}")

    checkpoint_categories = checkpoint.get("category_names") or []
    if checkpoint_categories and list(checkpoint_categories) != _category_names():
        raise ValueError(
            f"checkpoint 分类头 {list(checkpoint_categories)} 与当前标签体系 "
            f"{_category_names()} 不一致，请确认 labels.yaml 未变更或重新训练模型"
        )

    # 结构与当前标签体系的一致性由 predict_dataset 校验（load 只负责重建）
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
            dropout=float(checkpoint.get("dropout", 0.0)),
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
        "dropout": float(checkpoint.get("dropout", 0.0)),
    }
    return model, meta


def _validate_model_taxonomy(model: nn.Module) -> None:
    """校验模型 4 头类别数/类别名与当前 labels.yaml 一致，否则硬失败。

    子标签名不存于 checkpoint（标签唯一真相源是 labels.yaml），下标映射依赖
    类别数逐头一致；不一致时静默映射会给出错误标签名，必须显式报错
    （与锁定决策 6 的 taxonomy 一致性硬失败同精神）。
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


def _iter_dataset_records(dataset_id: int) -> list[Record]:
    """数据集存在性校验 + 全部记录（filename 升序）。不存在 → ValueError。"""
    with SessionLocal() as db:
        if db.get(Dataset, dataset_id) is None:
            raise ValueError(f"数据集 {dataset_id} 不存在")
        return list(
            db.scalars(
                select(Record)
                .where(Record.dataset_id == dataset_id)
                .order_by(Record.filename)
            ).all()
        )


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
    _validate_model_taxonomy(model)
    records = _iter_dataset_records(dataset_id)
    if not records:
        return []

    model.eval()
    results: list[dict[str, Any]] = []
    for start in range(0, len(records), _BATCH_SIZE):
        chunk = records[start : start + _BATCH_SIZE]
        tensors: list[torch.Tensor] = []
        for rec in chunk:
            image_path = Path(rec.image_path)
            if not image_path.is_file():
                raise ValueError(f"图片文件缺失：{rec.filename}")
            try:
                img = Image.open(image_path).convert("RGB")
            except Exception as exc:
                raise ValueError(f"图片读取失败：{rec.filename}（{exc}）") from exc
            tensors.append(_INFERENCE_TRANSFORM(img))
        batch = torch.stack(tensors)
        head_probs = _predict_batch(model, batch)
        for rec, probs in zip(chunk, head_probs):
            results.append(_prediction_entry(rec.filename, probs))
    return results


def _read_existing_xml(xml_path: Path) -> dict[str, Any] | None:
    """同名 XML 存在则读回，否则 None（缺 XML 视同未标注）。"""
    if xml_path.is_file():
        return xml_io.read_xml(xml_path)
    return None


def _record_for_filename(dataset_id: int, filename: str) -> Record:
    """按 (dataset_id, filename) 定位记录；不存在 → ValueError。"""
    with SessionLocal() as db:
        rec = db.scalars(
            select(Record).where(
                Record.dataset_id == dataset_id,
                Record.filename == filename,
            )
        ).first()
        if rec is None:
            raise ValueError(f"图片 {filename} 不存在于数据集 {dataset_id}")
        return rec


def generate_preannotation_xml(dataset_id: int, model: nn.Module) -> int:
    """基于预测结果生成预标注 XML（REQ-6.18），返回生成数量。

    规则：
    - 仅处理同名 XML **labels 为空**（含 XML 缺失）的图片——已有手动标注的
      图片跳过，绝不覆盖（REQ-6.18-4）；
    - 预标注与人工标注共用 ``xml_io.write_xml``（同 schema，REQ-6.18-1），
      写到图片旁（同一位置，标注界面 T10 直接读取）；
    - 每张写 4 个 label（每 category 一个），confidence = 预测概率 [0, 1]
      （REQ-6.18-2）；
    - 保留既有 XML 的 folder/filename/path/size/quality（缺失时以数据集名/
      文件名兜底，quality 空字符串）。
    """
    predictions = predict_dataset(dataset_id, model)

    with SessionLocal() as db:
        dataset = db.get(Dataset, dataset_id)
        if dataset is None:
            raise ValueError(f"数据集 {dataset_id} 不存在")
        dataset_name = dataset.name

    written = 0
    for entry in predictions:
        filename = entry["filename"]
        rec = _record_for_filename(dataset_id, filename)
        xml_path = Path(rec.image_path).with_suffix(".xml")

        existing = _read_existing_xml(xml_path)
        if existing is not None and existing.get("labels"):
            continue  # 已有人工标注：不覆盖

        data: dict[str, Any] = {
            **(existing or {}),
            "annotation_source": "model",
            "folder": (existing or {}).get("folder") or dataset_name,
            "filename": (existing or {}).get("filename") or filename,
            "path": (existing or {}).get("path") or f"{dataset_id}/{filename}",
            "size": (existing or {}).get("size"),
            "quality": (existing or {}).get("quality", ""),
            "labels": [
                {"category": p["category"], "name": p["label"], "confidence": p["confidence"]}
                for p in entry["predictions"]
            ],
        }
        xml_io.write_xml(xml_path, data)
        written += 1
    return written
