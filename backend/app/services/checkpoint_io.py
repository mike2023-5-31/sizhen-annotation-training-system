"""受管模型路径与 PyTorch 权重存档读取。"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from app import config
from app.labels_config import CATEGORIES, num_classes_per_head


CHECKPOINT_SUFFIXES = {".pth", ".pt"}


def resolve_managed_path(value: str, *, allow_absolute: bool = False) -> Path:
    root = config.MODEL_DIR.resolve()
    path = Path(value)
    if (path.is_absolute() and not allow_absolute) or path.suffix.lower() not in CHECKPOINT_SUFFIXES:
        raise ValueError("请选择模型目录内的 .pth 或 .pt 存档")
    try:
        resolved = (root / path).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("无法解析模型目录内的存档路径") from exc
    if not resolved.is_relative_to(root):
        raise ValueError("存档路径越出模型目录")
    return resolved


def read_checkpoint(path: Path, *, include_digest: bool = False) -> tuple[dict[str, Any], str | None]:
    """仅反序列化权重和基本类型，可对同一打开的文件计算源文件摘要。"""
    source_sha256 = None
    try:
        with path.open("rb") as stream:
            if include_digest:
                digest = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                source_sha256 = digest.hexdigest()
                stream.seek(0)
            checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("无法读取存档，请检查文件是否损坏或包含不受支持的对象") from exc
    if not isinstance(checkpoint, dict):
        raise ValueError("无法识别模型结构，存档应为系统权重字典")
    return checkpoint, source_sha256


def current_class_labels() -> list[list[str]]:
    return [list(category["labels"]) for category in CATEGORIES]


def validate_taxonomy(checkpoint: dict[str, Any]) -> None:
    if checkpoint.get("num_classes") != num_classes_per_head():
        raise ValueError("存档类别数与当前标签体系不一致，请确认 labels.yaml 或重新训练")
    category_names = checkpoint.get("category_names")
    if category_names and category_names != [category["name"] for category in CATEGORIES]:
        raise ValueError("存档分类头与当前标签体系不一致，请确认 labels.yaml 或重新训练")
    if "class_labels" in checkpoint and checkpoint["class_labels"] != current_class_labels():
        raise ValueError("存档子标签或顺序与当前标签体系不一致，请确认 labels.yaml 或重新训练")
