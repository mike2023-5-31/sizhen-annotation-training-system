"""推理：模型加载、批量预测、预标注 XML 生成（6.15–6.18）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms
from sqlalchemy import select

from app import config
from app.db import SessionLocal
from app.labels_config import CATEGORIES
from app.models import factory
from app.models.orm import Dataset, Record
from app.services import checkpoint, xml_io

_SUPPORTED_ARCHITECTURES = {"resnet18", "resnet50"}

_NORMALIZE = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


class DatasetNotFound(ValueError):
    pass


def _preprocess(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = transforms.Resize((224, 224))(image)
        tensor = transforms.ToTensor()(image)
    return _NORMALIZE(tensor)


def _category_label_maps() -> tuple[list[str], list[list[str]]]:
    names = [item["name"] for item in CATEGORIES]
    labels = [item["labels"] for item in CATEGORIES]
    return names, labels


def _dataset_categories(dataset_id: int) -> list[str]:
    """扫描数据集所有 XML，返回出现过的 category（按标签体系顺序）。"""
    with SessionLocal() as db:
        records = list(
            db.scalars(select(Record).where(Record.dataset_id == dataset_id))
        )
    seen: set[str] = set()
    for rec in records:
        xml_path = Path(rec.image_path).with_suffix(".xml")
        if not xml_path.is_file():
            continue
        try:
            data = xml_io.read_xml(xml_path)
        except Exception:  # noqa: BLE001
            continue
        for label in data.get("labels", []):
            if label.get("category") and float(label.get("confidence", 1.0)) >= 1.0:
                seen.add(label["category"])
    order = [item["name"] for item in CATEGORIES]
    return [name for name in order if name in seen]


def _resolve_active_heads(dataset_id: int, active_heads: list[int] | None) -> list[int]:
    """确定要输出的分类头索引：优先数据集实际维度，回退模型训练维度，最后回退全部。"""
    order = [item["name"] for item in CATEGORIES]
    dataset_cats = _dataset_categories(dataset_id)
    if dataset_cats:
        return [order.index(name) for name in dataset_cats]
    if active_heads:
        return [h for h in active_heads if 0 <= h < len(order)]
    return list(range(len(order)))


def load_model(path: str | Path, allow_custom: bool = True) -> tuple[torch.nn.Module, dict[str, Any]]:
    """从 checkpoint 重建网络并载入权重，返回 (model, meta)。"""
    ckpt = checkpoint.load_checkpoint(path)
    checkpoint.validate_taxonomy(ckpt)

    architecture = ckpt["architecture"]
    num_classes = [int(v) for v in ckpt["num_classes"]]
    dropout = float(ckpt.get("config", {}).get("dropout", 0.0))
    if architecture not in _SUPPORTED_ARCHITECTURES:
        if architecture == "custom" and allow_custom:
            pass
        else:
            raise ValueError(f"无法识别的模型结构：{architecture}")

    custom_path = None
    if architecture == "custom":
        rel = ckpt.get("custom_model_path")
        if not rel:
            raise ValueError("自定义模型存档缺少模型文件路径")
        custom_path = str(checkpoint.resolve_model_path(Path(config.MODEL_DIR), rel))

    model = factory.instantiate(architecture, num_classes, dropout=dropout, custom_path=custom_path)
    try:
        model.load_state_dict(ckpt["model_state"])
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"权重与模型结构不匹配：{exc}") from exc
    model.eval()
    return model, {
        "architecture": architecture,
        "num_classes": num_classes,
        "active_heads": ckpt.get("active_heads"),
    }


def _predict_one(model: torch.nn.Module, tensor: torch.Tensor, device: torch.device, heads: list[int]) -> list[dict[str, Any]]:
    names, labels = _category_label_maps()
    with torch.no_grad():
        outputs = model(tensor.unsqueeze(0).to(device))
    predictions = []
    for head in heads:
        probs = torch.softmax(outputs[head], dim=1)[0].cpu()
        best = int(torch.argmax(probs))
        predictions.append(
            {
                "category": names[head],
                "label": labels[head][best],
                "confidence": float(probs[best]),
            }
        )
    return predictions


def predict_dataset(
    dataset_id: int,
    model: torch.nn.Module,
    active_heads: list[int] | None = None,
) -> dict[str, Any]:
    """对数据集全部图片做批量预测，仅输出该数据集实际包含的维度。"""
    with SessionLocal() as db:
        if db.get(Dataset, dataset_id) is None:
            raise DatasetNotFound(f"数据集 {dataset_id} 不存在")
        records = list(
            db.scalars(select(Record).where(Record.dataset_id == dataset_id).order_by(Record.filename))
        )

    heads = _resolve_active_heads(dataset_id, active_heads)
    order = [item["name"] for item in CATEGORIES]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    results = []
    for rec in records:
        image_path = Path(rec.image_path)
        if not image_path.is_file():
            raise ValueError(f"图片文件缺失：{rec.filename}")
        try:
            tensor = _preprocess(image_path)
            predictions = _predict_one(model, tensor, device, heads)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"图片 {rec.filename} 推理失败：{exc}") from exc
        results.append({"filename": rec.filename, "predictions": predictions})
    return {"categories": [order[h] for h in heads], "predictions": results}


def generate_preannotation_xml(
    dataset_id: int,
    model: torch.nn.Module,
    confidence_threshold: float = 0.0,
    active_heads: list[int] | None = None,
) -> int:
    """基于预测为未人工标注的图片生成预标注 XML，仅针对数据集实际维度，返回生成数量。"""
    with SessionLocal() as db:
        if db.get(Dataset, dataset_id) is None:
            raise DatasetNotFound(f"数据集 {dataset_id} 不存在")
        records = list(
            db.scalars(select(Record).where(Record.dataset_id == dataset_id).order_by(Record.filename))
        )

    heads = _resolve_active_heads(dataset_id, active_heads)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    generated = 0
    for rec in records:
        image_path = Path(rec.image_path)
        if not image_path.is_file():
            continue
        xml_path = image_path.with_suffix(".xml")
        if xml_path.is_file():
            existing = xml_io.read_xml(xml_path)
            if any(float(l.get("confidence", 1.0)) >= 1.0 for l in existing.get("labels", [])):
                continue  # 已有人工标注，不覆盖

        try:
            tensor = _preprocess(image_path)
            predictions = _predict_one(model, tensor, device, heads)
        except Exception:  # noqa: BLE001
            continue

        labels = [
            {"category": p["category"], "name": p["label"], "confidence": p["confidence"]}
            for p in predictions
            if p["confidence"] >= confidence_threshold
        ]
        data = {
            "folder": "",
            "filename": rec.filename,
            "path": str(image_path),
            "quality": "",
            "labels": labels,
        }
        xml_io.write_xml(xml_path, data)
        generated += 1
    return generated


def comparison_images(dataset_id: int, limit: int) -> list[tuple[str, torch.Tensor]]:
    """取数据集前 limit 张图片的预处理张量，供 ONNX 精度对比抽样。"""
    with SessionLocal() as db:
        if db.get(Dataset, dataset_id) is None:
            raise DatasetNotFound(f"数据集 {dataset_id} 不存在")
        records = list(
            db.scalars(select(Record).where(Record.dataset_id == dataset_id).order_by(Record.filename).limit(limit))
        )
    result = []
    for rec in records:
        image_path = Path(rec.image_path)
        if not image_path.is_file():
            continue
        result.append((rec.filename, _preprocess(image_path)))
    return result
