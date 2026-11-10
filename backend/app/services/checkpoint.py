"""训练存档（checkpoint）读写与校验。

checkpoint 是训练（保存）、推理/ONNX（加载）之间的契约，字段约定：
    model_state       模型权重 state_dict
    optimizer_state   优化器状态（断点续训用，可缺省）
    scheduler_state   学习率调度器状态（可缺省）
    epoch             已完成的训练轮数
    best_metric       历史最佳宏平均验证准确率
    config            训练超参快照
    architecture      resnet18 / resnet50 / custom
    num_classes       4 个分类头的类别数列表
    category_names    4 个类别的名称列表
    custom_model_path 自定义模型 .py 相对路径（architecture=custom 时必需）
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from app import config
from app.labels_config import CATEGORIES, num_classes_per_head

CHECKPOINT_SUFFIXES = {".pth", ".pt"}


def current_class_labels() -> list[list[str]]:
    """标签体系当前各类别的子标签列表（顺序与分类头一致）。"""
    return [item["labels"] for item in CATEGORIES]


def current_category_names() -> list[str]:
    return [item["name"] for item in CATEGORIES]


def resolve_model_path(model_dir: Path, rel_path: str) -> Path:
    """把用户传入的相对路径解析到模型目录内，拒绝绝对路径与 ``..`` 穿越。"""
    candidate = Path(rel_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("模型路径必须位于模型目录内")
    resolved = (model_dir / candidate).resolve()
    root = model_dir.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("模型路径越出模型目录")
    return resolved


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # 同目录临时文件 + 原子替换：失败时保留上一份可用存档。
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """读取并校验 checkpoint 结构；损坏或不完整抛 ValueError。"""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"存档文件不存在：{source}")
    try:
        data = torch.load(source, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"存档无法读取（损坏或非本系统产物）：{exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("存档格式损坏（不是 checkpoint 字典）")

    required = ("model_state", "architecture", "num_classes")
    for key in required:
        if key not in data:
            raise ValueError(f"存档缺少字段：{key}")
    return data


def validate_taxonomy(checkpoint: dict[str, Any]) -> None:
    """校验存档的类别数与当前标签体系一致，不一致拒绝加载。"""
    stored = checkpoint.get("num_classes")
    current = num_classes_per_head()
    if not isinstance(stored, list) or [int(v) for v in stored] != current:
        raise ValueError(
            f"存档类别数 {stored} 与当前标签体系 {current} 不一致"
        )

    if checkpoint.get("category_names", current_category_names()) != current_category_names():
        raise ValueError("存档分类头顺序与当前标签体系不一致")
    if checkpoint.get("class_labels", current_class_labels()) != current_class_labels():
        raise ValueError("存档标签名称或顺序与当前标签体系不一致")


def list_checkpoints(model_dir: Path) -> list[str]:
    if not model_dir.is_dir():
        return []
    files = [
        str(p.relative_to(model_dir)).replace("\\", "/")
        for p in sorted(model_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in CHECKPOINT_SUFFIXES
    ]
    return files
