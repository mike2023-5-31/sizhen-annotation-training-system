"""训练服务（T13，需求 6.6/6.8/6.9/6.10/6.11/6.14；T14 扩展：监控推送 + 评估）。

- ``TrainManager``：线程安全的训练任务管理器（指标缓冲 + 任务状态），
  ``start_training(config)`` / ``resume_training(checkpoint_path, config)``
  后台线程执行、立即返回 job id（REQ-6.12-3 后台训练不阻塞）。
- T14 监控（REQ-6.12）：每轮指标追加时置位 ``threading.Event``
  （``metrics_event``）唤醒 WebSocket 推送；训练线程只 ``set()`` 从不阻塞发送；
  ``get_metrics(job_id=None)`` 返回指标缓冲快照（轮询兜底 / 刷新重建曲线）。
- T14 评估（REQ-6.13）：训练完成后对**验证集**做一次前向，保留 softmax 概率，
  逐类别（舌/面/脉/声音）计算子标签混淆矩阵 + one-vs-rest micro 平均 ROC/AUC
  （``compute_eval(job_id)``，缓存于 job；无验证集 → 空 dict 不报错）。
- 数据管线：torch Dataset 读取每张 ``Record`` 图片 + 同名 XML（``xml_io.read_xml``），
  标签映射为各分类头类别下标（未标注 category → 下标 0 =「正常/无」，锁定决策 1）；
  训练集应用所选增强（RandomHorizontalFlip / RandomRotation / ColorJitter /
  RandomCrop → Resize(224)），验证集不增强（REQ-6.10-2）。
- 设备：0 卡 → CPU FP32；1 卡 → 单卡；>1 卡 → DataParallel；GPU 上按配置启用
  ``torch.cuda.amp`` FP16 混合精度（GradScaler），无 GPU 自动降级 FP32
  （REQ-6.8，锁定决策 9）。
- 损失：CrossEntropy / FocalLoss 作用于**全部 4 个分类头**并对各头损失求和
  （REQ-6.9-6）；优化器 SGD/Adam/AdamW（REQ-6.9-4）；Dropout 传入 ``build_model``
  （REQ-6.9-5）。
- Checkpoint（T13 产出、T17 消费）：dict ``{"model_state", "optimizer_state",
  "scheduler_state", "epoch", "best_metric", "config", "architecture",
  "num_classes", "category_names", "dropout"}``；每轮保存 latest、指标提升时保存
  best 到 ``config.MODEL_DIR``（REQ-6.14-3）。
- ``best_metric`` = 4 个分类头 macro val_acc 的算术均值（锁定决策 3），
  best 存档在提升时覆盖保存（REQ-6.14-2）。

约束（锁定决策 6 / 6.11-3）：训练启动前同步校验全部所选数据集每一张标注 XML 的
(category, label) 均在当前 labels.yaml 内，未知标签 → **硬失败** ValueError，
绝不静默映射错误下标；断点存档同样校验结构/类别数与当前标签体系一致。
"""
from __future__ import annotations

import copy
import random
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sqlalchemy import select
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from app import config
from app.config import MODEL_DIR
from app.db import SessionLocal
from app.labels_config import CATEGORIES, num_classes_per_head
from app.models import factory
from app.models.orm import Dataset, Record
from app.services import xml_io

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_TARGET_SIZE = 224          # 各头共享主干要求的固定输入尺寸
_CROP_SIZE = 196            # RandomCrop 后统一 Resize(224)（crop → resize 语义）
_DEFAULT_LR = 1e-3

# 训练引擎支持的结构（锁定决策 8：结构读取仅保证 ResNet18/50）
_SUPPORTED_BACKBONES = {"resnet18", "resnet50"}
_SUPPORTED_OPTIMIZERS = {"sgd", "adam", "adamw"}
_SUPPORTED_LOSSES = {"crossentropy", "focalloss"}

# 增强方式别名表（canonical 键 → 可接受的输入写法集合）
_AUGMENTATION_ALIASES: dict[str, set[str]] = {
    "randomhorizontalflip": {"randomhorizontalflip", "randomflip", "hflip", "flip", "horizontalflip"},
    "randomrotation": {"randomrotation", "rotation", "rotate"},
    "colorjitter": {"colorjitter", "jitter", "color"},
    "randomcrop": {"randomcrop", "crop"},
}

# 标签体系快速索引（模块导入时快照，与 factory/labels_config 同源）
_CATEGORY_ORDER: dict[str, int] = {c["name"]: i for i, c in enumerate(CATEGORIES)}
_CATEGORY_LABEL_INDEX: dict[str, dict[str, int]] = {
    c["name"]: {label: j for j, label in enumerate(c["labels"])} for c in CATEGORIES
}
_CATEGORY_NAMES: list[str] = [c["name"] for c in CATEGORIES]


def _canonical_key(value: str) -> str:
    """小写化并去除非字母数字字符（别名归一化）。"""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


# ---------------------------------------------------------------------------
# 损失函数（REQ-6.9-6）
# ---------------------------------------------------------------------------


class _FocalLoss(nn.Module):
    """多分类 Focal Loss：CE 乘以 (1 - p_t)^γ 调制项，缓解类别不平衡。

    作用于**单个**分类头；训练引擎对 4 个头分别计算后求和。
    """

    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def _build_loss(kind: str) -> nn.Module:
    """按配置构建损失（作用于单头，训练引擎逐头求和）。"""
    if kind == "focalloss":
        return _FocalLoss(gamma=2.0)
    return nn.CrossEntropyLoss()


# ---------------------------------------------------------------------------
# 配置归一化 / 校验（REQ-6.9 参数可调）
# ---------------------------------------------------------------------------


def normalize_config(raw: Optional[dict[str, Any]]) -> dict[str, Any]:
    """归一化并校验训练配置，产出 canonical dict。

    键：``backbone``（resnet18/resnet50/custom）、``datasets``（非空 id 列表，首个为主
    训练集、其余叠加，REQ-6.6）、``lr``、``batch_size``、``epochs``、``optimizer``
    （sgd/adam/adamw）、``dropout``、``loss``（crossentropy/focalloss）、
    ``augmentation``（列表）、``fp16``（bool）、``val_split``、``seed``、
    ``resume_from``（可选路径）、``model_dir``（可选，默认 ``config.MODEL_DIR``）。
    ``backbone="custom"`` 时必须提供 ``custom_model_path``（需求 6.7：MODEL_DIR 内
    相对 .py 路径，防路径穿越），归一化后保留原值。

    Raises:
        ValueError: 任一键取值非法 / backbone、optimizer、loss、增强方式不受支持 /
            custom 缺 custom_model_path 或路径非法 / 自定义模型文件不存在。
    """
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("训练配置必须是 dict")

    cfg: dict[str, Any] = {}

    backbone = str(raw.get("backbone", "resnet18")).lower()
    if backbone not in _SUPPORTED_BACKBONES and backbone != "custom":
        raise ValueError(
            f"不支持的 backbone：{backbone!r}（仅支持 resnet18 / resnet50 / custom）"
        )
    cfg["backbone"] = backbone

    if backbone == "custom":
        custom_model_path = raw.get("custom_model_path")
        _resolve_custom_model_path(custom_model_path)  # .py / 防穿越 / 文件存在
        cfg["custom_model_path"] = str(custom_model_path).strip()

    cfg["datasets"] = _normalize_datasets(raw.get("datasets"))

    try:
        lr = float(raw.get("lr", _DEFAULT_LR))
    except (TypeError, ValueError):
        lr = 0.0
    if not lr > 0:
        raise ValueError(f"lr 必须为正数，得到 {raw.get('lr')!r}")
    cfg["lr"] = lr

    try:
        batch_size = int(raw.get("batch_size", 16))
        epochs = int(raw.get("epochs", 10))
    except (TypeError, ValueError):
        batch_size = epochs = 0
    if batch_size < 1:
        raise ValueError(f"batch_size 必须 ≥ 1，得到 {raw.get('batch_size')!r}")
    if epochs < 1:
        raise ValueError(f"epochs 必须 ≥ 1，得到 {raw.get('epochs')!r}")
    cfg["batch_size"] = batch_size
    cfg["epochs"] = epochs

    optimizer = str(raw.get("optimizer", "adam")).lower()
    if optimizer not in _SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"不支持的优化器：{optimizer!r}（仅支持 sgd / adam / adamw）"
        )
    cfg["optimizer"] = optimizer

    try:
        dropout = float(raw.get("dropout", 0.0))
    except (TypeError, ValueError):
        dropout = -1.0
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout 必须在 [0, 1) 范围内，得到 {raw.get('dropout')!r}")
    cfg["dropout"] = dropout

    loss = str(raw.get("loss", "crossentropy")).lower()
    if loss not in _SUPPORTED_LOSSES:
        raise ValueError(
            f"不支持的损失函数：{loss!r}（仅支持 crossentropy / focalloss）"
        )
    cfg["loss"] = loss

    cfg["augmentation"] = _normalize_augmentation(raw.get("augmentation"))

    fp16 = raw.get("fp16", False)
    cfg["fp16"] = (
        str(fp16).strip().lower() in {"1", "true", "yes", "on"}
        if isinstance(fp16, str)
        else bool(fp16)
    )

    try:
        val_split = float(raw.get("val_split", 0.2))
    except (TypeError, ValueError):
        val_split = -1.0
    if not 0.0 <= val_split < 1.0:
        raise ValueError(f"val_split 必须在 [0, 1) 范围内，得到 {raw.get('val_split')!r}")
    cfg["val_split"] = val_split

    try:
        seed = int(raw.get("seed", 42))
    except (TypeError, ValueError):
        seed = 42
    cfg["seed"] = seed

    resume_from = raw.get("resume_from")
    cfg["resume_from"] = str(resume_from) if resume_from else None

    model_dir = raw.get("model_dir")
    cfg["model_dir"] = str(Path(model_dir)) if model_dir else str(MODEL_DIR)

    return cfg


def _normalize_datasets(raw: Any) -> list[int]:
    """datasets → 去重保序的 int id 列表；空/非法 → ValueError（REQ-6.6-1）。"""
    if isinstance(raw, (str, int)):
        raw = [raw]
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("config.datasets 必须为非空的数据集 id 列表（至少一个主训练集）")
    ids: list[int] = []
    for item in raw:
        try:
            ds_id = int(item)
        except (TypeError, ValueError):
            raise ValueError(f"数据集 id 非法：{item!r}") from None
        if ds_id not in ids:
            ids.append(ds_id)
    return ids


def _normalize_augmentation(raw: Any) -> list[str]:
    """augmentation 输入 → canonical 键列表（保序去重）；未知项 → ValueError。"""
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        raise ValueError("augmentation 必须是字符串列表")

    alias_map: dict[str, str] = {
        alias: canon for canon, aliases in _AUGMENTATION_ALIASES.items() for alias in aliases
    }
    canonical: list[str] = []
    for item in raw:
        canon = alias_map.get(_canonical_key(item))
        if canon is None:
            raise ValueError(
                f"不支持的增强方式：{item!r}"
                "（可选：randomhorizontalflip / randomrotation / colorjitter / randomcrop）"
            )
        if canon not in canonical:
            canonical.append(canon)
    return canonical


def _resolve_custom_model_path(rel_path: Any) -> Path:
    """校验 custom_model_path 为 MODEL_DIR 内相对 .py 路径（需求 6.7，NF-7.4
    防路径穿越），返回绝对路径。

    Raises:
        ValueError: 非字符串/空 / 绝对路径 / 非 .py / ``..`` 穿越出模型目录 /
            文件不存在。
    """
    if not isinstance(rel_path, str) or not rel_path.strip():
        raise ValueError("backbone=custom 时必须提供 custom_model_path（模型目录内的 .py 相对路径）")
    requested = Path(rel_path.strip())
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


# ---------------------------------------------------------------------------
# 数据增强（REQ-6.10）
# ---------------------------------------------------------------------------

_AUG_OPS: dict[str, Any] = {
    "randomhorizontalflip": transforms.RandomHorizontalFlip(p=0.5),
    "randomrotation": transforms.RandomRotation(degrees=15),
    "colorjitter": transforms.ColorJitter(
        brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
    ),
    # crop 后统一 Resize(224)；pad_if_needed 保证小图不报错
    "randomcrop": transforms.RandomCrop(
        size=_CROP_SIZE, pad_if_needed=True, fill=0
    ),
}


def _build_train_transform(augmentation: list[str]) -> Any:
    """训练集 transform：所选增强（按勾选顺序）+ Resize(224) + ToTensor。"""
    ops = [_AUG_OPS[name] for name in augmentation]
    ops += [transforms.Resize((_TARGET_SIZE, _TARGET_SIZE)), transforms.ToTensor()]
    return transforms.Compose(ops)


def _build_val_transform() -> Any:
    """验证/推理 transform：无增强，仅 Resize(224) + ToTensor（REQ-6.10-2）。"""
    return transforms.Compose(
        [transforms.Resize((_TARGET_SIZE, _TARGET_SIZE)), transforms.ToTensor()]
    )


# ---------------------------------------------------------------------------
# 设备 / 混合精度（REQ-6.8）
# ---------------------------------------------------------------------------


def resolve_device() -> tuple[torch.device, int]:
    """自动检测：0 卡 → CPU；1 卡 → cuda:0；>1 卡 → cuda:0（外层包 DataParallel）。"""
    n_gpus = torch.cuda.device_count()
    if n_gpus <= 0:
        return torch.device("cpu"), 0
    return torch.device("cuda:0"), n_gpus


# ---------------------------------------------------------------------------
# 数据集与标签体系校验（锁定决策 6）
# ---------------------------------------------------------------------------


class _AnnotationDataset(Dataset):
    """图片 + 4 头标签向量数据集。

    ``items``：``(image_path, target_vector)``，vector 长度 4（与 CATEGORIES 顺序
    一致），未标注 category 取 0（=「正常/无」）。
    """

    def __init__(self, items: list[tuple[str, list[int]]], transform: Any) -> None:
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, vector = self.items[idx]
        img = Image.open(image_path).convert("RGB")
        return self.transform(img), torch.tensor(vector, dtype=torch.long)


def _target_vector(image_path: str, filename: str) -> list[int]:
    """读同名 XML → 4 头标签下标向量；XML 缺失/未标注 → 全 0（「正常/无」）。

    未知 (category, label) → ValueError 硬失败（锁定决策 6）。
    """
    xml_path = Path(image_path).with_suffix(".xml")
    if not xml_path.is_file():
        return [0] * len(_CATEGORY_ORDER)
    try:
        data = xml_io.read_xml(xml_path)
    except Exception as exc:
        raise ValueError(f"标注 XML 解析失败：{filename}（{exc}）") from exc

    vector = [0] * len(_CATEGORY_ORDER)
    for label in data.get("labels") or []:
        category = str(label.get("category") or "")
        name = str(label.get("name") or "")
        if category not in _CATEGORY_ORDER:
            raise ValueError(
                f"未知标注类别「{category}」：{filename}"
                "（与 labels.yaml 不一致，请修正标注或标签体系后再训练）"
            )
        label_index = _CATEGORY_LABEL_INDEX[category]
        if name not in label_index:
            raise ValueError(
                f"类别「{category}」下未知标签「{name}」：{filename}"
                "（taxonomy 校验失败，锁定决策 6：未知标签硬失败）"
            )
        vector[_CATEGORY_ORDER[category]] = label_index[name]
    return vector


def _prepare_data(dataset_ids: list[int]) -> list[tuple[str, list[int]]]:
    """校验数据集存在、收集全部 Record 并逐张做 taxonomy 校验（同步、硬失败）。

    Returns:
        ``[(image_path, target_vector), ...]``，按 (dataset_id, filename) 升序。
    """
    with SessionLocal() as db:
        for ds_id in dataset_ids:
            if db.get(Dataset, ds_id) is None:
                raise ValueError(f"数据集 {ds_id} 不存在")
        records = list(
            db.scalars(
                select(Record)
                .where(Record.dataset_id.in_(dataset_ids))
                .order_by(Record.dataset_id, Record.filename)
            ).all()
        )
    if not records:
        raise ValueError("所选数据集没有任何图片记录，无法训练")

    items: list[tuple[str, list[int]]] = []
    for rec in records:
        image_path = str(rec.image_path)
        if not Path(image_path).is_file():
            raise ValueError(f"图片文件缺失：{rec.filename}")
        items.append((image_path, _target_vector(image_path, rec.filename)))
    return items


def _split_train_val(
    items: list[tuple[str, list[int]]], val_split: float, seed: int
) -> tuple[list[int], set[int]]:
    """固定 seed 划分 train/val 下标（首训与续训一致，best/评估判据可比）。"""
    rng = random.Random(seed)
    indices = list(range(len(items)))
    rng.shuffle(indices)
    n_val = int(len(items) * val_split) if val_split else 0
    val_set = set(indices[:n_val])
    train_set = [i for i in indices if i not in val_set]
    if len(items) >= 2 and not train_set:
        train_set = [indices[-1]]
        val_set = set(indices[:-1])
    return train_set, val_set


# ---------------------------------------------------------------------------
# 优化器 / 调度器（REQ-6.9-4）
# ---------------------------------------------------------------------------


def _build_base_model(
    architecture: str,
    cfg: dict[str, Any],
    payload: Optional[dict[str, Any]] = None,
) -> nn.Module:
    """按架构重建基础模型（需求 6.7）。

    - resnet18/resnet50 → ``factory.build_model``（原路径不变）；
    - custom → ``factory.load_custom_model`` + ``factory.wrap_custom_model``
      （输出契约归一化为 4 分类头 logits dict）。``custom_model_path`` 优先取
      存档（payload），其次取配置；缺失/文件不存在 → ValueError。
    """
    num_classes = num_classes_per_head()
    if architecture != "custom":
        return factory.build_model(
            architecture, list(num_classes), dropout=cfg["dropout"]
        )
    rel_path = (payload.get("custom_model_path") if payload else None) or cfg.get(
        "custom_model_path"
    )
    if not rel_path:
        raise ValueError(
            "custom 模型缺少 custom_model_path，无法重建（请重新上传自定义模型 .py 后重试）"
        )
    custom_model = factory.load_custom_model(
        str(_resolve_custom_model_path(rel_path)), list(num_classes)
    )
    return factory.wrap_custom_model(custom_model, list(num_classes))


def _build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    kind = cfg["optimizer"]
    params = model.parameters()
    if kind == "sgd":
        return torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, weight_decay=1e-4)
    if kind == "adam":
        return torch.optim.Adam(params, lr=cfg["lr"], weight_decay=1e-4)
    return torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=1e-4)


def _build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: dict[str, Any]
) -> torch.optim.lr_scheduler.LRScheduler:
    """StepLR：step_size = max(1, epochs // 3)，每 epoch 末 step 一次。"""
    return torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, cfg["epochs"] // 3), gamma=0.1
    )


# ---------------------------------------------------------------------------
# 指标（锁定决策 3：best_metric = 4 头 macro val_acc 的算术均值）
# ---------------------------------------------------------------------------


def _macro_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, list[float]]:
    """单头 macro accuracy：各类别正确率（仅统计出现过的类别）的算术均值。"""
    preds = logits.argmax(dim=1)
    per_class: list[float] = []
    for cls in range(logits.size(1)):
        mask = targets == cls
        count = int(mask.sum())
        if count > 0:
            per_class.append(float((preds[mask] == cls).float().mean()))
    mean = float(sum(per_class) / len(per_class)) if per_class else 0.0
    return mean, per_class


# ---------------------------------------------------------------------------
# Checkpoint（T13 产出 / T17 消费；REQ-6.11 / 6.14）
# ---------------------------------------------------------------------------

_CHECKPOINT_KEYS = (
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "epoch",
    "best_metric",
    "config",
    "architecture",
    "num_classes",
    "category_names",
)


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_metric: Optional[float],
    cfg: dict[str, Any],
    num_classes: list[int],
    category_names: list[str],
) -> None:
    """保存系统训练存档（dict 契约，与 T17 load_model_from_dir 一致）。"""
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "best_metric": float(best_metric) if best_metric is not None else None,
        "config": cfg,
        "architecture": cfg["backbone"],
        "custom_model_path": cfg.get("custom_model_path"),
        "num_classes": list(num_classes),
        "category_names": list(category_names),
        "dropout": cfg["dropout"],
    }
    torch.save(checkpoint, path)


def _load_resume_payload(
    path: str, config: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """加载并校验断点存档（REQ-6.11-3），返回续训负载。

    校验项：文件存在、dict 结构、architecture 受支持、num_classes/category_names
    与当前 labels.yaml 一致；若提供 config 则 backbone 必须与存档一致。
    """
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise ValueError(f"存档文件不存在：{ckpt_path}")
    try:
        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except Exception:
        # 兼容含 config dict（Path 等非常规对象）的系统产物（管理员可控的模型目录）
        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, dict):
        raise ValueError("存档格式损坏：不是系统训练存档（dict）")

    architecture = checkpoint.get("architecture")
    num_classes = checkpoint.get("num_classes")
    model_state = checkpoint.get("model_state")
    if (
        (architecture not in _SUPPORTED_BACKBONES and architecture != "custom")
        or not isinstance(num_classes, list)
        or model_state is None
    ):
        raise ValueError(
            "存档格式损坏或非系统训练产物（缺少 architecture/num_classes/model_state）"
        )
    if list(num_classes) != num_classes_per_head():
        raise ValueError(
            f"存档类别数 {list(num_classes)} 与当前标签体系 {num_classes_per_head()} "
            "不一致，请确认 labels.yaml 未变更或重新训练（REQ-6.11-3）"
        )
    ckpt_categories = list(checkpoint.get("category_names") or [])
    if ckpt_categories and ckpt_categories != _CATEGORY_NAMES:
        raise ValueError(
            f"存档分类头 {ckpt_categories} 与当前标签体系 {_CATEGORY_NAMES} 不一致"
        )
    if config is not None:
        cfg_backbone = str(config.get("backbone", architecture)).lower()
        if cfg_backbone != architecture:
            raise ValueError(
                f"训练配置 backbone「{cfg_backbone}」与存档 architecture"
                f"「{architecture}」不一致"
            )

    return {
        "model_state": model_state,
        "optimizer_state": checkpoint.get("optimizer_state"),
        "scheduler_state": checkpoint.get("scheduler_state"),
        "epoch": int(checkpoint.get("epoch") or 0),
        "best_metric": checkpoint.get("best_metric"),
        "architecture": architecture,
        "custom_model_path": checkpoint.get("custom_model_path"),
        "config": dict(checkpoint.get("config") or {}),
    }


# ---------------------------------------------------------------------------
# 训练任务管理器（线程安全）
# ---------------------------------------------------------------------------


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


class TrainManager:
    """后台训练任务管理器：非阻塞启动 + 线程安全指标缓冲 + 任务状态查询。

    任务状态：``created → running → completed | stopped | failed``。
    每个 job dict 公开字段：job_id / status / config / metrics（每轮一条）/
    start_epoch / total_epochs / current_epoch / device / use_amp /
    checkpoint_best / checkpoint_latest / resume_from / error /
    created_at / finished_at。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        # T14：指标/状态变化推送信号。训练线程只 set()（非阻塞），
        # WS 连接在各自事件循环侧 wait() + clear()（见 routers/training.py）。
        self._metrics_event = threading.Event()

    @property
    def metrics_event(self) -> threading.Event:
        """指标追加/状态变化的推送信号（T14，供 WS 连接等待唤醒）。"""
        return self._metrics_event

    # -- 启动入口 ----------------------------------------------------------

    def start(self, config: dict[str, Any]) -> str:
        """启动新训练任务（非阻塞，返回 job id）。

        同步完成（可抛 ValueError）：配置归一化校验、数据集存在性、逐张 taxonomy
        校验（锁定决策 6）、resume_from 存档校验。模型构建与训练循环在后台线程。
        """
        cfg = normalize_config(config)
        payload = _load_resume_payload(cfg["resume_from"], cfg) if cfg["resume_from"] else None
        items = _prepare_data(cfg["datasets"])
        return self._launch(cfg, items, payload)

    def resume(self, checkpoint_path: str, config: Optional[dict[str, Any]] = None) -> str:
        """从存档续训（REQ-6.11）：恢复模型/优化器/调度器/轮次/best-so-far。

        ``config`` 缺省时沿用存档内训练配置；提供时覆盖（epochs 语义 = 续训
        **追加**轮数）。存档结构/类别数/分类头与当前标签体系不一致 → ValueError。
        """
        payload = _load_resume_payload(checkpoint_path, config)
        merged = {**payload["config"], **(config or {})}
        cfg = normalize_config(merged)
        cfg["resume_from"] = str(checkpoint_path)
        items = _prepare_data(cfg["datasets"])
        return self._launch(cfg, items, payload)

    def _launch(
        self,
        cfg: dict[str, Any],
        items: list[tuple[str, list[int]]],
        payload: Optional[dict[str, Any]],
    ) -> str:
        job_id = uuid.uuid4().hex[:12]
        job: dict[str, Any] = {
            "job_id": job_id,
            "status": "created",
            "config": cfg,
            "metrics": [],
            "start_epoch": payload["epoch"] if payload else 0,
            "total_epochs": (payload["epoch"] if payload else 0) + cfg["epochs"],
            "current_epoch": payload["epoch"] if payload else 0,
            "device": "",
            "use_amp": False,
            "checkpoint_best": "",
            "checkpoint_latest": "",
            "resume_from": cfg["resume_from"],
            "error": None,
            "created_at": _utcnow_iso(),
            "finished_at": None,
            "stop_event": threading.Event(),
            "thread": None,
            "_items": items,
            "_payload": payload,
        }
        thread = threading.Thread(
            target=self._run, args=(job,), daemon=True, name=f"train-{job_id}"
        )
        with self._lock:
            self._jobs[job_id] = job
            job["thread"] = thread
            job["status"] = "running"
        thread.start()
        return job_id

    # -- 查询 / 控制 --------------------------------------------------------

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        """任务公开视图（深拷贝，线程安全）；不存在 → None。"""
        with self._lock:
            job = self._jobs.get(job_id)
            return self._public_view(job) if job else None

    def list_jobs(self) -> list[dict[str, Any]]:
        """全部任务公开视图，按创建顺序。"""
        with self._lock:
            return [self._public_view(j) for j in self._jobs.values()]

    def get_metrics(self, job_id: Optional[str] = None) -> dict[str, Any]:
        """指标缓冲快照（T14）：单个任务或全部任务的 status + metrics。

        供 ``GET /training/status`` 轮询兜底与 WS 连接初始快照共用
        （REQ-6.12-4 刷新后按已记录指标重建曲线）。
        """
        if job_id is not None:
            job = self.get_job(job_id)
            return {"jobs": [job] if job else []}
        return {"jobs": self.list_jobs()}

    def stop(self, job_id: str) -> bool:
        """请求停止：置 stop 事件，训练线程在下一 batch/epoch 边界退出并存档。"""
        with self._lock:
            job = self._jobs.get(job_id)
        if not job or job["status"] not in {"created", "running"}:
            return False
        job["stop_event"].set()
        return True

    def wait(self, job_id: str, timeout: Optional[float] = None) -> dict[str, Any]:
        """等待任务结束（测试/同步场景），返回最终公开视图。"""
        with self._lock:
            job = self._jobs.get(job_id)
            thread = job["thread"] if job else None
        if thread is not None:
            thread.join(timeout)
        view = self.get_job(job_id)
        return view if view is not None else {}

    def compute_eval(self, job_id: str) -> Optional[dict[str, Any]]:
        """评估结果（T14，REQ-6.13）：验证集混淆矩阵 + one-vs-rest ROC/AUC。

        训练完成时已在后台线程算好并缓存于 job；缓存缺失（如任务被停止/跨进程
        查询）且 latest 存档可用时，从存档重建模型对验证集重新计算一次。

        Returns:
            ``{"confusion": {类别: [[..]]}, "roc": {类别: {"fpr", "tpr", "auc"}}}``；
            无验证集 → 两个空 dict；任务不存在或仍在运行/评估不可用 → None。
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            cached = job.get("eval")
            if cached is not None:
                return copy.deepcopy(cached)
            status = job["status"]
            cfg = copy.deepcopy(job["config"])
            items = job.get("_items")
            checkpoint_path = job.get("checkpoint_latest")
        if status in {"created", "running"}:
            return None
        if not checkpoint_path or not items:
            return {"confusion": {}, "roc": {}}
        try:
            result = _eval_from_checkpoint(cfg, items, str(checkpoint_path))
        except Exception:
            return {"confusion": {}, "roc": {}}
        self._update(job, eval=result)
        return copy.deepcopy(result)

    def _public_view(self, job: dict[str, Any]) -> dict[str, Any]:
        view = {
            key: copy.deepcopy(job[key])
            for key in (
                "job_id",
                "status",
                "config",
                "metrics",
                "start_epoch",
                "total_epochs",
                "current_epoch",
                "device",
                "use_amp",
                "checkpoint_best",
                "checkpoint_latest",
                "resume_from",
                "error",
                "created_at",
                "finished_at",
            )
        }
        return view

    # -- 内部：状态更新（均在锁内）------------------------------------------

    def _update(self, job: dict[str, Any], **fields: Any) -> None:
        with self._lock:
            job.update(fields)
        if "status" in fields or "eval" in fields:
            self._metrics_event.set()

    def _append_metric(self, job: dict[str, Any], metric: dict[str, Any]) -> None:
        with self._lock:
            job["metrics"].append(copy.deepcopy(metric))
            job["current_epoch"] = metric["epoch"]
        self._metrics_event.set()

    # -- 训练主循环 ---------------------------------------------------------

    def _run(self, job: dict[str, Any]) -> None:
        cfg: dict[str, Any] = job["config"]
        items: list[tuple[str, list[int]]] = job["_items"]
        payload: Optional[dict[str, Any]] = job["_payload"]
        stop_event: threading.Event = job["stop_event"]
        try:
            self._train(job, cfg, items, payload)
        except Exception as exc:  # 后台线程兜底：任何异常 → failed + error
            self._update(
                job,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                finished_at=_utcnow_iso(),
            )
        finally:
            if job["status"] == "running":
                self._update(job, status="stopped", finished_at=_utcnow_iso())
            stop_event.clear()

    def _train(
        self,
        job: dict[str, Any],
        cfg: dict[str, Any],
        items: list[tuple[str, list[int]]],
        payload: Optional[dict[str, Any]],
    ) -> None:
        stop_event = job["stop_event"]

        torch.manual_seed(cfg["seed"])
        device, n_gpus = resolve_device()
        use_amp = n_gpus > 0 and cfg["fp16"]  # 无 GPU → FP32（REQ-6.8-3）
        self._update(job, device=str(device), use_amp=use_amp)

        # 模型（T12 build_model；custom 走 load_custom_model + 输出契约包装；
        # 续训时 architecture 以存档为准）
        architecture = payload["architecture"] if payload else cfg["backbone"]
        num_classes = num_classes_per_head()
        base_model = _build_base_model(architecture, cfg, payload)
        if payload:
            base_model.load_state_dict(payload["model_state"])
        model = nn.DataParallel(base_model) if n_gpus > 1 else base_model
        model = model.to(device)
        category_names = list(base_model.category_names)

        loss_fn = _build_loss(cfg["loss"])
        optimizer = _build_optimizer(base_model, cfg)
        if payload and payload.get("optimizer_state"):
            optimizer.load_state_dict(payload["optimizer_state"])
            for group in optimizer.param_groups:  # 配置 lr 优先于存档恢复值
                group["lr"] = cfg["lr"]
        scheduler = _build_scheduler(optimizer, cfg)
        if payload and payload.get("scheduler_state"):
            scheduler.load_state_dict(payload["scheduler_state"])
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        start_epoch = payload["epoch"] if payload else 0
        best_metric: Optional[float] = (
            float(payload["best_metric"]) if payload and payload.get("best_metric") is not None else None
        )

        # 划分 train/val（seed 固定 → 续训与首训划分一致，best 判据可比）
        train_set, val_set = _split_train_val(items, cfg["val_split"], cfg["seed"])

        train_loader = DataLoader(
            _AnnotationDataset(
                [items[i] for i in train_set],
                _build_train_transform(cfg["augmentation"]),
            ),
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=0,
        )
        val_loader = None
        if val_set:
            val_loader = DataLoader(
                _AnnotationDataset(
                    [items[i] for i in sorted(val_set)], _build_val_transform()
                ),
                batch_size=cfg["batch_size"],
                shuffle=False,
                num_workers=0,
            )

        model_dir = Path(cfg["model_dir"])
        model_dir.mkdir(parents=True, exist_ok=True)
        latest_path = model_dir / f"{job['job_id']}_latest.pth"
        best_path = model_dir / f"{job['job_id']}_best.pth"
        self._update(
            job,
            checkpoint_latest=str(latest_path),
            checkpoint_best=str(best_path),
        )

        for epoch in range(start_epoch + 1, start_epoch + cfg["epochs"] + 1):
            if stop_event.is_set():
                break
            train_loss, train_acc, train_per_head = self._train_epoch(
                model, train_loader, loss_fn, optimizer, scaler, device, use_amp,
                category_names, stop_event,
            )
            if stop_event.is_set():
                break

            if val_loader is not None:
                val_loss, val_acc, val_per_head = self._eval_epoch(
                    model, val_loader, loss_fn, device, use_amp, category_names
                )
            else:  # 无验证集 → 以训练指标兜底（REQ-6.13-4 精神：不报错）
                val_loss, val_acc, val_per_head = float("nan"), train_acc, train_per_head

            scheduler.step()
            metric = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_acc_per_head": train_per_head,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_acc_per_head": val_per_head,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            self._append_metric(job, metric)

            # 每轮存 latest（REQ-6.14-3）
            _save_checkpoint(
                latest_path, base_model, optimizer, scheduler, epoch,
                best_metric, cfg, list(num_classes), category_names,
            )
            # best = 4 头 macro val_acc 均值最大时覆盖保存（REQ-6.14-2，锁定决策 3）
            if best_metric is None or val_acc > best_metric:
                best_metric = val_acc
                _save_checkpoint(
                    best_path, base_model, optimizer, scheduler, epoch,
                    best_metric, cfg, list(num_classes), category_names,
                )

        if stop_event.is_set():
            # 优雅停止：落一份 latest 供断点续训
            try:
                _save_checkpoint(
                    latest_path, base_model, optimizer, scheduler,
                    max(epoch, start_epoch), best_metric, cfg,
                    list(num_classes), category_names,
                )
            except Exception:
                pass
            self._update(job, status="stopped", finished_at=_utcnow_iso())
        else:
            # T14（REQ-6.13）：完成后对验证集计算混淆矩阵 + one-vs-rest ROC，
            # 与状态更新同锁原子落盘；无验证集 → 空 dict 不报错（REQ-6.13-4）。
            eval_result: dict[str, Any] = {"confusion": {}, "roc": {}}
            if val_loader is not None:
                try:
                    collected = _collect_val_probs(
                        model, val_loader, device, use_amp, category_names
                    )
                    eval_result = _compute_eval_from_probs(
                        collected, category_names, list(num_classes)
                    )
                except Exception:
                    eval_result = {"confusion": {}, "roc": {}}
            self._update(
                job,
                eval=eval_result,
                status="completed",
                finished_at=_utcnow_iso(),
            )

    # -- epoch 级训练/评估 --------------------------------------------------

    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        device: torch.device,
        use_amp: bool,
        category_names: list[str],
        stop_event: threading.Event,
    ) -> tuple[float, float, list[float]]:
        model.train()
        total_loss = 0.0
        count = 0
        logits_acc: dict[str, list[torch.Tensor]] = {n: [] for n in category_names}
        targets_acc: dict[str, list[torch.Tensor]] = {n: [] for n in category_names}

        for images, targets in loader:
            if stop_event.is_set():
                break
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outs = model(images)
                loss = sum(
                    loss_fn(outs[name], targets[:, i])
                    for i, name in enumerate(category_names)
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.detach()) * images.size(0)
            count += images.size(0)
            with torch.no_grad():
                for i, name in enumerate(category_names):
                    logits_acc[name].append(outs[name].detach().cpu())
                    targets_acc[name].append(targets[:, i].cpu())

        mean_loss = total_loss / count if count else float("nan")
        return mean_loss, *_mean_macro_acc(logits_acc, targets_acc, category_names)

    def _eval_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
        use_amp: bool,
        category_names: list[str],
    ) -> tuple[float, float, list[float]]:
        model.eval()
        total_loss = 0.0
        count = 0
        logits_acc: dict[str, list[torch.Tensor]] = {n: [] for n in category_names}
        targets_acc: dict[str, list[torch.Tensor]] = {n: [] for n in category_names}

        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device)
                targets = targets.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outs = model(images)
                    loss = sum(
                        loss_fn(outs[name], targets[:, i])
                        for i, name in enumerate(category_names)
                    )
                total_loss += float(loss.detach()) * images.size(0)
                count += images.size(0)
                for i, name in enumerate(category_names):
                    logits_acc[name].append(outs[name].detach().cpu())
                    targets_acc[name].append(targets[:, i].cpu())

        mean_loss = total_loss / count if count else float("nan")
        return mean_loss, *_mean_macro_acc(logits_acc, targets_acc, category_names)


def _mean_macro_acc(
    logits_acc: dict[str, list[torch.Tensor]],
    targets_acc: dict[str, list[torch.Tensor]],
    category_names: list[str],
) -> tuple[float, list[float]]:
    """4 头 macro acc 列表 + 算术均值（= best_metric 判据，锁定决策 3）。"""
    per_head: list[float] = []
    for name in category_names:
        if logits_acc[name]:
            logits = torch.cat(logits_acc[name])
            targets = torch.cat(targets_acc[name])
            head_acc, _ = _macro_accuracy(logits, targets)
            per_head.append(head_acc)
        else:
            per_head.append(0.0)
    mean = float(sum(per_head) / len(per_head)) if per_head else 0.0
    return mean, per_head


# ---------------------------------------------------------------------------
# 评估计算（T14，需求 6.13：验证集混淆矩阵 + one-vs-rest ROC）
# ---------------------------------------------------------------------------


def _collect_val_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    category_names: list[str],
) -> dict[str, tuple[Any, Any]]:
    """验证集前向一遍，收集各头 softmax 概率 + 真值下标（REQ-6.13-3 保留概率）。

    Returns:
        ``{类别: (probs ndarray (N, n_cls), targets ndarray (N,))}``。
    """
    model.eval()
    prob_acc: dict[str, list[torch.Tensor]] = {n: [] for n in category_names}
    targ_acc: dict[str, list[torch.Tensor]] = {n: [] for n in category_names}
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outs = model(images)
            for i, name in enumerate(category_names):
                prob_acc[name].append(
                    F.softmax(outs[name], dim=1).detach().float().cpu()
                )
                targ_acc[name].append(targets[:, i].cpu())
    return {
        name: (
            torch.cat(prob_acc[name]).numpy() if prob_acc[name] else None,
            torch.cat(targ_acc[name]).numpy() if targ_acc[name] else None,
        )
        for name in category_names
    }


def _one_vs_rest_roc(
    y_true: Any, probs: Any, n_classes: int
) -> tuple[list[float], list[float], float]:
    """单类别多分类头的 one-vs-rest micro 平均 ROC（REQ-6.13-2）。

    子标签二值化（label_binarize）后与 softmax 概率按列展平，得到该类别一条
    聚合 ROC 曲线（= 各子标签 one-vs-rest 的 micro 平均）与 AUC 标量。
    """
    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import label_binarize

    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    if y_bin.ndim == 1:
        y_bin = y_bin.reshape(-1, 1)
    fpr, tpr, _ = roc_curve(y_bin.ravel(), probs.reshape(-1).ravel())
    try:
        auc_val = float(auc(fpr, tpr))
    except Exception:
        auc_val = 0.0
    return [float(x) for x in fpr], [float(x) for x in tpr], auc_val


def _compute_eval_from_probs(
    collected: dict[str, tuple[Any, Any]],
    category_names: list[str],
    num_classes: list[int],
) -> dict[str, Any]:
    """逐类别计算子标签混淆矩阵 + ROC/AUC（REQ-6.13-1/2）。

    Returns:
        ``{"confusion": {类别: [[int,...]...]}, "roc": {类别: {"fpr", "tpr", "auc"}}}``；
        某类别无验证样本时该类别缺席（不伪造数据）。
    """
    from sklearn.metrics import confusion_matrix

    confusion: dict[str, list[list[int]]] = {}
    roc: dict[str, dict[str, Any]] = {}
    for i, name in enumerate(category_names):
        probs, targets = collected.get(name, (None, None))
        if probs is None or targets is None or targets.size == 0:
            continue
        n_cls = int(num_classes[i])
        matrix = confusion_matrix(targets, probs.argmax(axis=1), labels=list(range(n_cls)))
        confusion[name] = [[int(v) for v in row] for row in matrix.tolist()]
        fpr, tpr, auc_val = _one_vs_rest_roc(targets, probs, n_cls)
        roc[name] = {"fpr": fpr, "tpr": tpr, "auc": auc_val}
    return {"confusion": confusion, "roc": roc}


def _eval_from_checkpoint(
    cfg: dict[str, Any], items: list[tuple[str, list[int]]], checkpoint_path: str
) -> dict[str, Any]:
    """从 latest 存档重建模型，对验证集计算评估（缓存缺失时的兜底路径）。

    与训练线程使用同一 ``_split_train_val`` 划分，保证评估范围一致。
    """
    payload = _load_resume_payload(checkpoint_path, cfg)
    device, _n_gpus = resolve_device()
    model = _build_base_model(payload["architecture"], cfg, payload)
    model.load_state_dict(payload["model_state"])
    model = model.to(device)

    _, val_set = _split_train_val(items, cfg["val_split"], cfg["seed"])
    if not val_set:
        return {"confusion": {}, "roc": {}}
    loader = DataLoader(
        _AnnotationDataset([items[i] for i in sorted(val_set)], _build_val_transform()),
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    collected = _collect_val_probs(model, loader, device, False, _CATEGORY_NAMES)
    return _compute_eval_from_probs(collected, _CATEGORY_NAMES, list(num_classes))


# ---------------------------------------------------------------------------
# 模块级入口（供 T14 训练路由使用）
# ---------------------------------------------------------------------------

# 全局单例管理器
train_manager = TrainManager()


def start_training(config: dict[str, Any]) -> str:
    """后台启动训练（REQ-6.6 至 6.11），返回 job id（非阻塞）。

    Raises:
        ValueError: 配置非法 / 数据集不存在 / taxonomy 不一致（未知标签）/
            断点存档校验失败。
    """
    return train_manager.start(config)


def resume_training(checkpoint_path: str, config: Optional[dict[str, Any]] = None) -> str:
    """从存档续训（REQ-6.11），返回新 job id；config 缺省沿用存档配置。"""
    return train_manager.resume(checkpoint_path, config)
