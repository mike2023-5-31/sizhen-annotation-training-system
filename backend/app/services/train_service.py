"""训练服务（T13，需求 6.6/6.8/6.9/6.10/6.11/6.14）。

- ``TrainManager``：线程安全的训练任务管理器（指标缓冲 + 任务状态），
  ``start_training(config)`` / ``resume_training(checkpoint_path, config)``
  后台线程执行、立即返回 job id（REQ-6.12-3 后台训练不阻塞）。
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

    键：``backbone``（resnet18/resnet50）、``datasets``（非空 id 列表，首个为主
    训练集、其余叠加，REQ-6.6）、``lr``、``batch_size``、``epochs``、``optimizer``
    （sgd/adam/adamw）、``dropout``、``loss``（crossentropy/focalloss）、
    ``augmentation``（列表）、``fp16``（bool）、``val_split``、``seed``、
    ``resume_from``（可选路径）、``model_dir``（可选，默认 ``config.MODEL_DIR``）。

    Raises:
        ValueError: 任一键取值非法 / backbone、optimizer、loss、增强方式不受支持。
    """
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("训练配置必须是 dict")

    cfg: dict[str, Any] = {}

    backbone = str(raw.get("backbone", "resnet18")).lower()
    if backbone not in _SUPPORTED_BACKBONES:
        raise ValueError(
            f"不支持的 backbone：{backbone!r}（仅支持 resnet18 / resnet50）"
        )
    cfg["backbone"] = backbone

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


# ---------------------------------------------------------------------------
# 优化器 / 调度器（REQ-6.9-4）
# ---------------------------------------------------------------------------


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
        architecture not in _SUPPORTED_BACKBONES
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

    def _append_metric(self, job: dict[str, Any], metric: dict[str, Any]) -> None:
        with self._lock:
            job["metrics"].append(copy.deepcopy(metric))
            job["current_epoch"] = metric["epoch"]

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

        # 模型（T12 build_model；续训时 architecture 以存档为准）
        architecture = payload["architecture"] if payload else cfg["backbone"]
        num_classes = num_classes_per_head()
        base_model = factory.build_model(
            architecture, list(num_classes), dropout=cfg["dropout"]
        )
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
        rng = random.Random(cfg["seed"])
        indices = list(range(len(items)))
        rng.shuffle(indices)
        n_val = int(len(items) * cfg["val_split"]) if cfg["val_split"] else 0
        val_set = set(indices[:n_val])
        train_set = [i for i in indices if i not in val_set]
        if len(items) >= 2 and not train_set:
            train_set = [indices[-1]]
            val_set = set(indices[:-1])

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
            self._update(job, status="completed", finished_at=_utcnow_iso())

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
