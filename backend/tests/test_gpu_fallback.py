"""GPU 降级测试（T13，REQ-6.8，锁定决策 9）：无 CUDA 环境 → CPU FP32，不崩溃。

仅测试**模拟的 CPU 降级路径**（monkeypatch 关闭 CUDA），不触碰任何真实 GPU 路径：
- resolve_device 在无 GPU 时返回 cpu + 0 卡；
- fp16=True 显式请求 → 无 GPU 自动降级 FP32（use_amp=False），训练正常完成；
- GradScaler 在禁用态（FP32）下工作正常，无 CUDA 导入/运行错误。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torchvision.models as tvm
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models.orm import Dataset, Record
from app.services import train_service, xml_io


# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def cpu_only(monkeypatch):
    """模拟无 CUDA 环境：is_available=False 且 device_count=0。"""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


@pytest.fixture
def db(tmp_path: Path, monkeypatch):
    """临时 SQLite，替换 train_service.SessionLocal。"""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test_gpu_fallback.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    monkeypatch.setattr(train_service, "SessionLocal", factory)
    return factory


@pytest.fixture
def no_download(monkeypatch):
    """训练线程内 build_model 不下载预训练权重（weights=None）。"""
    original = tvm.resnet18
    monkeypatch.setattr(tvm, "resnet18", lambda weights=None: original(weights=None))


# ---------------------------------------------------------------------------
# 设备解析降级
# ---------------------------------------------------------------------------


def test_resolve_device_degrades_to_cpu(cpu_only) -> None:
    """CUDA 不可用 → resolve_device 返回 cpu、0 卡（REQ-6.8-3）。"""
    device, n_gpus = train_service.resolve_device()
    assert device.type == "cpu"
    assert n_gpus == 0


def test_amp_scaler_inert_without_cuda(cpu_only) -> None:
    """训练引擎的 FP32 GradScaler 构造（enabled=False）在无 CUDA 下可正常使用。"""
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # train_service._train 同款构造
    t = torch.tensor(1.0, requires_grad=True)
    scaler.scale(t).backward()
    assert t.grad is not None
    assert torch.isfinite(t.grad)
    assert float(t.grad) == 1.0  # disabled 态 scale 恒等 → 梯度不经缩放


# ---------------------------------------------------------------------------
# fp16=True 显式请求 → CPU FP32 降级冒烟（不崩溃）
# ---------------------------------------------------------------------------


def test_fp16_requested_degrades_to_fp32_cpu(
    tmp_path: Path, cpu_only, db, no_download
) -> None:
    """无 GPU 时 fp16=True 被降级为 FP32：device=cpu、use_amp=False，训练正常完成。"""
    imgs_dir = tmp_path / "imgs"
    imgs_dir.mkdir()
    with db() as session:
        ds = Dataset(name="gpu-fallback")
        session.add(ds)
        session.flush()
        for i in range(2):
            fn = f"img{i}.png"
            path = imgs_dir / fn
            Image.new("RGB", (64, 64), (10, 20, 30)).save(path)
            session.add(
                Record(
                    dataset_id=ds.id,
                    filename=fn,
                    image_path=str(path.resolve()),
                    quality="合格",
                )
            )
        session.commit()
        ds_id = ds.id
    for i in range(2):
        xml_io.write_xml(
            imgs_dir / f"img{i}.xml",
            {
                "folder": "gpu-fallback",
                "filename": f"img{i}.png",
                "path": f"img{i}.png",
                "size": {"width": 64, "height": 64, "depth": 3},
                "quality": "合格",
                "labels": [{"category": "舌", "name": "舌苔黄", "confidence": 1.0}],
            },
        )

    cfg = {
        "datasets": [ds_id],
        "backbone": "resnet18",
        "epochs": 1,
        "batch_size": 2,
        "lr": 1e-3,
        "optimizer": "adam",
        "loss": "crossentropy",
        "dropout": 0.0,
        "augmentation": [],
        "fp16": True,  # 显式请求混合精度 → 无 GPU 必须降级 FP32
        "val_split": 0.5,
        "seed": 0,
        "model_dir": str(tmp_path / "models"),
    }

    job_id = train_service.start_training(cfg)
    job = train_service.train_manager.wait(job_id, timeout=900)
    assert job["status"] == "completed", job.get("error")
    assert job["device"] == "cpu"       # 无 GPU → CPU
    assert job["use_amp"] is False      # FP16 请求降级为 FP32（REQ-6.8-3，锁定决策 9）
    assert len(job["metrics"]) == 1
