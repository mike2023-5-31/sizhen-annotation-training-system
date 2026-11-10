"""训练引擎：混合数据集训练、GPU/FP16、增强、断点续训、监控与评估（6.6–6.14）。"""
from __future__ import annotations

import copy
import random
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from app import config
from app.db import SessionLocal
from app.labels_config import CATEGORIES, num_classes_per_head
from app.models import factory
from app.models.orm import Dataset as DatasetRecord, Record
from app.services import checkpoint, xml_io
from sqlalchemy import select

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SUPPORTED_BACKBONES = {"resnet18", "resnet50", "custom"}
SUPPORTED_OPTIMIZERS = {"sgd", "adam", "adamw"}
SUPPORTED_LOSSES = {"crossentropy", "focalloss"}
SUPPORTED_AUGMENTATIONS = {"flip", "rotate", "colorjitter", "crop"}


class TrainingBusyError(RuntimeError):
    """已有训练任务在运行，设备被占用。"""


# ---------------------------------------------------------------------------
# 配置归一化与校验（6.9 / 6.10）
# ---------------------------------------------------------------------------


def normalize_config(raw: dict[str, Any]) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "backbone": str(raw.get("backbone", "resnet18")).lower(),
        "lr": float(raw.get("lr", 1e-3)),
        "batch_size": int(raw.get("batch_size", 16)),
        "epochs": int(raw.get("epochs", 10)),
        "optimizer": str(raw.get("optimizer", "adam")).lower(),
        "dropout": float(raw.get("dropout", 0.0)),
        "loss": str(raw.get("loss", "crossentropy")).lower(),
        "augmentation": [str(a).lower() for a in raw.get("augmentation", [])],
        "fp16": bool(raw.get("fp16", False)),
        "val_split": float(raw.get("val_split", 0.2)),
        "custom_model_path": raw.get("custom_model_path"),
        "datasets": list(raw.get("datasets") or []),
    }

    if cfg["backbone"] not in SUPPORTED_BACKBONES:
        raise ValueError(f"不支持的骨干网络：{cfg['backbone']}")
    if not 0 < cfg["lr"] <= 1:
        raise ValueError("学习率必须在 (0, 1] 范围内")
    if not 1 <= cfg["batch_size"] <= 512:
        raise ValueError("batch size 必须在 1 到 512 之间")
    if not 1 <= cfg["epochs"] <= 500:
        raise ValueError("训练轮数必须在 1 到 500 之间")
    if cfg["optimizer"] not in SUPPORTED_OPTIMIZERS:
        raise ValueError(f"不支持的优化器：{cfg['optimizer']}")
    if not 0 <= cfg["dropout"] < 1:
        raise ValueError("dropout 必须在 [0, 1) 范围内")
    if cfg["loss"] not in SUPPORTED_LOSSES:
        raise ValueError(f"不支持的损失函数：{cfg['loss']}")
    for aug in cfg["augmentation"]:
        if aug not in SUPPORTED_AUGMENTATIONS:
            raise ValueError(f"不支持的数据增强：{aug}")
    if not 0 < cfg["val_split"] < 1:
        raise ValueError("验证集比例必须在 (0, 1) 范围内")
    if not cfg["datasets"]:
        raise ValueError("必须至少选择一个训练数据集")
    return cfg


# ---------------------------------------------------------------------------
# 数据准备
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    image_path: Path
    targets: list[int]  # 长度 4，-1 表示该类别无标注


def _parse_targets(data: dict[str, Any]) -> list[int]:
    """把 XML 的 labels 转成 4 个头的类别索引，缺失为 -1。"""
    head_index = {item["name"]: i for i, item in enumerate(CATEGORIES)}
    label_index = {
        item["name"]: {name: j for j, name in enumerate(item["labels"])}
        for item in CATEGORIES
    }
    targets = [-1] * 4
    for label in data.get("labels", []):
        category = label["category"]
        name = label["name"]
        if category not in head_index or name not in label_index[category]:
            raise ValueError(f"标注标签不属于当前标签体系：{category}/{name}")
        targets[head_index[category]] = label_index[category][name]
    return targets


def collect_samples(dataset_ids: list[int]) -> list[Sample]:
    """从多个数据集收集带有效标注的样本（合并训练）。"""
    samples: list[Sample] = []
    with SessionLocal() as db:
        for dataset_id in dict.fromkeys(dataset_ids):
            if db.get(DatasetRecord, dataset_id) is None:
                raise ValueError(f"训练数据集不存在：{dataset_id}")
            records = list(
                db.scalars(
                    select(Record).where(Record.dataset_id == dataset_id)
                )
            )
            for rec in records:
                xml_path = Path(rec.image_path).with_suffix(".xml")
                if not xml_path.is_file():
                    continue
                try:
                    data = xml_io.read_xml(xml_path)
                except Exception as exc:  # noqa: BLE001
                    raise ValueError(f"标注文件无法读取：{xml_path}") from exc
                targets = _parse_targets(data)
                if all(t < 0 for t in targets):
                    continue  # 完全没有标注，无法训练
                samples.append(Sample(image_path=Path(rec.image_path), targets=targets))
    return samples


def _train_transform(augmentations: list[str]) -> Callable:
    pipeline: list[Any] = []
    if "crop" in augmentations:
        pipeline.append(transforms.RandomResizedCrop(224, scale=(0.7, 1.0)))
    else:
        pipeline.append(transforms.Resize((224, 224)))
    if "flip" in augmentations:
        pipeline.append(transforms.RandomHorizontalFlip(p=0.5))
    if "rotate" in augmentations:
        pipeline.append(transforms.RandomRotation(15))
    pipeline.append(transforms.ToTensor())
    if "colorjitter" in augmentations:
        pipeline.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))
    pipeline.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(pipeline)


def _eval_transform() -> Callable:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class _ImageDataset(Dataset):
    def __init__(self, samples: list[Sample], transform: Callable):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        try:
            with Image.open(sample.image_path) as image:
                image = image.convert("RGB")
                tensor = self.transform(image)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"训练图片无法读取：{sample.image_path}") from exc
        return tensor, torch.tensor(sample.targets, dtype=torch.long)


# ---------------------------------------------------------------------------
# 模型与损失
# ---------------------------------------------------------------------------


def build_model(cfg: dict[str, Any]) -> nn.Module:
    head_sizes = num_classes_per_head()
    custom_path = None
    if cfg["backbone"] == "custom":
        if not cfg["custom_model_path"]:
            raise ValueError("自定义模型缺少文件路径")
        custom_path = str(
            checkpoint.resolve_model_path(
                Path(config.MODEL_DIR), cfg["custom_model_path"]
            )
        )
    return factory.instantiate(
        cfg["backbone"], head_sizes, dropout=cfg["dropout"], custom_path=custom_path
    )


def build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    params = model.parameters()
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9)
    if cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(params, lr=cfg["lr"])
    return torch.optim.Adam(params, lr=cfg["lr"])


class _FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        gathered = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal = -((1 - gathered) ** self.gamma) * log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        return focal.mean()


def build_loss(kind: str) -> nn.Module:
    if kind == "focalloss":
        return _FocalLoss()
    return nn.CrossEntropyLoss()


def _masked_head_loss(loss_fn, logits, targets, valid_mask):
    """只在有标注的样本上计算某头的损失；无有效样本返回 None。"""
    if valid_mask.sum() == 0:
        return None
    return loss_fn(logits[valid_mask], targets[valid_mask])


# ---------------------------------------------------------------------------
# 设备与能力（6.8）
# ---------------------------------------------------------------------------


def detect_device(cfg: dict[str, Any]) -> tuple[torch.device, int, bool]:
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        return torch.device("cpu"), 0, False
    use_amp = cfg["fp16"]
    return torch.device("cuda:0"), n_gpus, use_amp


def _gpu_name(index: int) -> str | None:
    try:
        return torch.cuda.get_device_name(index)
    except Exception:  # noqa: BLE001
        return None


def training_capabilities() -> dict[str, Any]:
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        return {"device": "cpu", "gpu_count": 0, "parallel_mode": "cpu", "devices": []}
    devices = []
    for i in range(n_gpus):
        try:
            free, total = torch.cuda.mem_get_info(i)
        except Exception:  # noqa: BLE001
            free, total = None, None
        devices.append(
            {
                "index": i,
                "name": _gpu_name(i),
                "total_bytes": total,
                "status": "detected",
            }
        )
    mode = "data_parallel" if n_gpus > 1 else "single_gpu"
    return {"device": "cuda:0", "gpu_count": n_gpus, "parallel_mode": mode, "devices": devices}


# ---------------------------------------------------------------------------
# 评估（6.13）
# ---------------------------------------------------------------------------


def _compute_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    """对验证集计算每头宏平均准确率，并收集概率/真值供混淆矩阵与 ROC。"""
    head_sizes = num_classes_per_head()
    model.eval()
    # 每头累积：真值列表、预测概率矩阵（行=样本，列=类别）
    trues: list[list[int]] = [[] for _ in range(4)]
    probs: list[list[np.ndarray]] = [None for _ in range(4)]

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images)
            for h in range(4):
                logits = outputs[h].cpu()
                prob = torch.softmax(logits, dim=1).numpy()
                t = targets[:, h].numpy()
                valid = t >= 0
                for idx in np.where(valid)[0]:
                    trues[h].append(int(t[idx]))
                    if probs[h] is None:
                        probs[h] = []
                    probs[h].append(prob[idx])

    macro_accs = []
    for h in range(4):
        if not trues[h]:
            macro_accs.append(0.0)
            continue
        preds = np.argmax(np.stack(probs[h]), axis=1)
        truth = np.array(trues[h])
        accs = []
        for cls in range(head_sizes[h]):
            mask = truth == cls
            if mask.sum() == 0:
                continue
            accs.append(float((preds[mask] == cls).mean()))
        macro_accs.append(float(np.mean(accs)) if accs else 0.0)

    return {
        "macro_accs": macro_accs,
        "trues": trues,
        "probs": probs,
    }


def _build_eval_report(result: dict[str, Any]) -> dict[str, Any]:
    """由累积的真值/概率生成混淆矩阵与 ROC/AUC（one-vs-rest）。"""
    from sklearn.metrics import auc, confusion_matrix, roc_curve

    head_sizes = num_classes_per_head()
    category_names = [item["name"] for item in CATEGORIES]
    confusion: dict[str, list] = {}
    roc: dict[str, dict[str, Any]] = {}

    for h in range(4):
        name = category_names[h]
        if not result["trues"][h]:
            continue
        truth = np.array(result["trues"][h])
        prob_matrix = np.stack(result["probs"][h])
        preds = np.argmax(prob_matrix, axis=1)
        labels = list(range(head_sizes[h]))
        cm = confusion_matrix(truth, preds, labels=labels)
        confusion[name] = cm.tolist()

        roc[name] = {}
        for cls in labels:
            y_true = (truth == cls).astype(int)
            if np.unique(y_true).size < 2:
                roc[name][str(cls)] = {"fpr": [], "tpr": [], "auc": None}
                continue
            try:
                fpr, tpr, _ = roc_curve(y_true, prob_matrix[:, cls])
                roc[name][str(cls)] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "auc": float(auc(fpr, tpr)),
                }
            except ValueError:
                roc[name][str(cls)] = {"fpr": [], "tpr": [], "auc": None}

    return {"confusion": confusion, "roc": roc, "macro_accs": result["macro_accs"]}


# ---------------------------------------------------------------------------
# 训练任务管理器
# ---------------------------------------------------------------------------


class TrainManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self.metrics_event = threading.Event()

    def _update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            self._jobs[job_id].update(fields)
        self.metrics_event.set()

    def _append_metric(self, job_id: str, metric: dict[str, Any]) -> None:
        with self._lock:
            self._jobs[job_id]["metrics"].append(metric)
        self.metrics_event.set()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._jobs.values())

    def get_metrics(self, job_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            if job_id is not None:
                jobs = [self._jobs[job_id]] if job_id in self._jobs else []
            else:
                jobs = list(self._jobs.values())
        return {"jobs": [self._public_view(job) for job in jobs]}

    @staticmethod
    def _public_view(job: dict[str, Any]) -> dict[str, Any]:
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "error": job["error"],
            "finished_at": job["finished_at"],
            "device": job["device"],
            "gpu_count": job["gpu_count"],
            "use_amp": job["use_amp"],
            "metrics": job["metrics"],
        }

    def start(self, cfg: dict[str, Any]) -> str:
        job_id = uuid.uuid4().hex[:12]
        device, n_gpus, use_amp = detect_device(cfg)
        with self._lock:
            if any(j["status"] == "running" for j in self._jobs.values()):
                raise TrainingBusyError("已有训练任务在运行，请先停止")
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "running",
                "error": None,
                "finished_at": None,
                "device": str(device),
                "gpu_count": n_gpus,
                "use_amp": use_amp,
                "metrics": [],
                "eval_cache": None,
                "config": cfg,
            }
        thread = threading.Thread(target=self._run, args=(job_id, cfg), daemon=True)
        thread.start()
        return job_id

    def stop(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job["_stop_requested"] = True
        self.metrics_event.set()

    def compute_eval(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.get("eval_cache") is not None:
                return job["eval_cache"]
            if job["status"] != "completed":
                return None
            val_loader = job.get("_val_loader")
            model_state = job.get("_best_state")
        if val_loader is None or model_state is None:
            return {}
        model = build_model(job["config"])
        model.load_state_dict(model_state)
        device = torch.device(job["device"])
        model.to(device)
        result = _compute_metrics(model, val_loader, device)
        report = _build_eval_report(result)
        with self._lock:
            self._jobs[job_id]["eval_cache"] = report
        return report

    # -- 训练主循环 --
    def _run(self, job_id: str, cfg: dict[str, Any]) -> None:
        try:
            samples = collect_samples(cfg["datasets"])
            if not samples:
                raise ValueError("所选数据集没有可用于训练的标注样本")

            head_sizes = num_classes_per_head()
            model = build_model(cfg)
            device, n_gpus, use_amp = detect_device(cfg)

            if n_gpus > 1:
                model = nn.DataParallel(model)
            model.to(device)

            random.Random(42).shuffle(samples)
            split = int(len(samples) * (1 - cfg["val_split"]))
            split = max(1, min(split, len(samples) - 1)) if len(samples) > 1 else len(samples)
            train_samples = samples[:split]
            val_samples = samples[split:]
            train_loader = DataLoader(
                _ImageDataset(train_samples, _train_transform(cfg["augmentation"])),
                batch_size=cfg["batch_size"],
                shuffle=True,
            )
            val_loader = DataLoader(
                _ImageDataset(val_samples, _eval_transform()),
                batch_size=cfg["batch_size"],
                shuffle=False,
            )

            optimizer = build_optimizer(model, cfg)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg["epochs"] // 3), gamma=0.1)
            loss_fn = build_loss(cfg["loss"])

            # 断点续训（6.11）
            start_epoch = 0
            best_metric = -1.0
            if cfg.get("resume_from"):
                ckpt = checkpoint.load_checkpoint(cfg["resume_from"])
                checkpoint.validate_taxonomy(ckpt)
                (model.module if n_gpus > 1 else model).load_state_dict(ckpt["model_state"])
                if ckpt.get("optimizer_state") and cfg["optimizer"] == ckpt.get("config", {}).get("optimizer", cfg["optimizer"]):
                    optimizer.load_state_dict(ckpt["optimizer_state"])
                if ckpt.get("scheduler_state"):
                    scheduler.load_state_dict(ckpt["scheduler_state"])
                start_epoch = int(ckpt.get("epoch", 0))
                best_metric = float(ckpt.get("best_metric", -1.0))

            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

            best_state = copy.deepcopy(ckpt.get("best_model_state", ckpt["model_state"])) if cfg.get("resume_from") else None
            model_dir = Path(config.MODEL_DIR) / job_id
            model_dir.mkdir(parents=True, exist_ok=True)

            for epoch in range(start_epoch, cfg["epochs"]):
                with self._lock:
                    if self._jobs[job_id].get("_stop_requested"):
                        break

                # 训练阶段
                model.train()
                running_loss = 0.0
                train_correct = [0] * 4
                train_valid = [0] * 4
                for images, targets in train_loader:
                    with self._lock:
                        if self._jobs[job_id].get("_stop_requested"):
                            break
                    images = images.to(device)
                    targets = targets.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        outputs = model(images)
                        total_loss = None
                        for h in range(4):
                            mask = targets[:, h] >= 0
                            head_loss = _masked_head_loss(loss_fn, outputs[h], targets[:, h], mask)
                            if head_loss is not None:
                                total_loss = head_loss if total_loss is None else total_loss + head_loss
                    if total_loss is None:
                        continue
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    running_loss += total_loss.item()

                    with torch.no_grad():
                        for h in range(4):
                            mask = targets[:, h] >= 0
                            if mask.sum() == 0:
                                continue
                            pred = outputs[h][mask].argmax(dim=1)
                            train_correct[h] += (pred == targets[:, h][mask]).sum().item()
                            train_valid[h] += mask.sum().item()

                with self._lock:
                    if self._jobs[job_id].get("_stop_requested"):
                        break  # 不把未完成的轮次写为可续训存档
                scheduler.step()

                # 验证阶段
                val_accs, val_loss = self._validate(model, val_loader, loss_fn, device, use_amp)

                train_acc = sum(train_correct) / max(sum(train_valid), 1)
                mean_val_acc = float(np.mean(val_accs)) if val_accs else 0.0
                self._append_metric(
                    job_id,
                    {
                        "epoch": epoch + 1,
                        "loss": running_loss / max(len(train_loader), 1),
                        "train_acc": train_acc,
                        "val_acc": mean_val_acc,
                    },
                )

                # 独立复制最佳权重，避免 state_dict 的共享张量被下一轮覆盖。
                state = {k: v.detach().cpu().clone() for k, v in
                         (model.module if n_gpus > 1 else model).state_dict().items()}
                improved = mean_val_acc > best_metric
                if improved:
                    best_metric = mean_val_acc
                    best_state = state
                payload = {
                    "model_state": state,
                    "best_model_state": best_state,
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "best_metric": best_metric,
                    "config": cfg,
                    "architecture": cfg["backbone"],
                    "num_classes": head_sizes,
                    "category_names": [item["name"] for item in CATEGORIES],
                    "class_labels": checkpoint.current_class_labels(),
                    "custom_model_path": cfg.get("custom_model_path"),
                }
                checkpoint.save_checkpoint(model_dir / "latest.pth", payload)
                if improved:
                    checkpoint.save_checkpoint(model_dir / "best.pth", {**payload, "model_state": best_state})

            with self._lock:
                job = self._jobs[job_id]
                stopped = job.get("_stop_requested", False)
                job["_val_loader"] = val_loader
                job["_best_state"] = best_state if best_state is not None else (model.module if n_gpus > 1 else model).state_dict()
            self._update(job_id, status="stopped" if stopped else "completed", finished_at=_now_iso())

        except Exception as exc:  # noqa: BLE001
            self._update(job_id, status="failed", error=str(exc), finished_at=_now_iso())

    def _validate(self, model, loader, loss_fn, device, use_amp):
        model.eval()
        sizes = num_classes_per_head()
        correct = [torch.zeros(n, dtype=torch.long) for n in sizes]
        valid = [torch.zeros(n, dtype=torch.long) for n in sizes]
        total_loss = 0.0
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device)
                targets = targets.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(images)
                for h in range(4):
                    mask = targets[:, h] >= 0
                    if mask.sum() == 0:
                        continue
                    pred = outputs[h][mask].argmax(dim=1)
                    truth = targets[:, h][mask]
                    valid[h] += torch.bincount(truth.cpu(), minlength=sizes[h])
                    correct[h] += torch.bincount(truth[pred == truth].cpu(), minlength=sizes[h])
                    head_loss = loss_fn(outputs[h][mask], targets[:, h][mask])
                    total_loss += head_loss.item()
        accs = [float((correct[h][valid[h] > 0] / valid[h][valid[h] > 0]).mean())
                if (valid[h] > 0).any() else 0.0 for h in range(4)]
        return accs, total_loss / max(len(loader), 1)


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


train_manager = TrainManager()


def start_training(options: dict[str, Any]) -> str:
    cfg = normalize_config(options)
    if cfg.get("resume_from"):
        checkpoint.load_checkpoint(cfg["resume_from"])
    return train_manager.start(cfg)


def resume_training(checkpoint_path: str, overrides: dict[str, Any]) -> str:
    ckpt = checkpoint.load_checkpoint(checkpoint_path)
    checkpoint.validate_taxonomy(ckpt)
    cfg = normalize_config({**ckpt.get("config", {}), **overrides})
    if cfg["epochs"] <= int(ckpt.get("epoch", 0)):
        raise ValueError("目标总轮数必须大于存档已完成轮数")
    cfg["resume_from"] = checkpoint_path
    return train_manager.start(cfg)
