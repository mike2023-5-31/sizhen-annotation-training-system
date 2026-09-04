"""训练服务单元测试（T13）：CPU 冒烟 / 断点续训 / taxonomy 硬失败 / 配置与工具函数。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torchvision.models as tvm
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.labels_config import CATEGORIES
from app.models.orm import Dataset, Record
from app.services import train_service, xml_io

CATEGORY_NAMES = [c["name"] for c in CATEGORIES]
NUM_CLASSES_4 = [len(c["labels"]) for c in CATEGORIES]


# ---------------------------------------------------------------------------
# 工具 / fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path: Path, monkeypatch):
    """临时 SQLite（建表 + Session 工厂），替换 train_service.SessionLocal。"""
    db_path = tmp_path / "test_train.db"
    engine = create_engine(
        f"sqlite:///{db_path.as_posix()}", connect_args={"check_same_thread": False}
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


def _add_dataset(
    session_factory,
    tmp_path: Path,
    name: str,
    images: list[str],
    labels_by_file: dict[str, list[tuple[str, str]]] | None = None,
) -> int:
    """建 Dataset + 每图真实图片文件 + 可选同名标注 XML，返回 dataset_id。"""
    imgs_dir = tmp_path / "imgs"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    with session_factory() as db:
        dataset = Dataset(name=name)
        db.add(dataset)
        db.flush()
        ds_id = dataset.id
        for fn in images:
            path = imgs_dir / fn
            Image.new("RGB", (64, 64), (110, 70, 40)).save(path)
            db.add(
                Record(
                    dataset_id=ds_id,
                    filename=fn,
                    image_path=str(path.resolve()),
                    quality="合格",
                )
            )
        db.commit()
    for fn, labels in (labels_by_file or {}).items():
        xml_path = imgs_dir / f"{Path(fn).stem}.xml"
        xml_io.write_xml(
            xml_path,
            {
                "folder": name,
                "filename": fn,
                "path": fn,
                "size": {"width": 64, "height": 64, "depth": 3},
                "quality": "合格",
                "labels": [
                    {"category": c, "name": n, "confidence": 1.0} for c, n in labels
                ],
            },
        )
    return ds_id


def _base_config(ds_ids: list[int], model_dir: Path) -> dict:
    return {
        "datasets": ds_ids,
        "backbone": "resnet18",
        "epochs": 2,
        "batch_size": 4,
        "lr": 1e-3,
        "optimizer": "adam",
        "loss": "crossentropy",
        "dropout": 0.0,
        "augmentation": ["randomhorizontalflip"],
        "fp16": False,
        "val_split": 0.5,
        "seed": 0,
        "model_dir": str(model_dir),
    }


# ---------------------------------------------------------------------------
# 端到端：CPU 2-epoch 冒烟 + 断点续训
# ---------------------------------------------------------------------------


def test_train_cpu_smoke_and_resume(tmp_path: Path, db, no_download) -> None:
    """2 epoch CPU 训练：checkpoint 齐全、metrics 长度 == epochs；续训从存档继续。"""
    images = [f"img{i}.png" for i in range(8)]
    labels_by_file = {
        "img0.png": [("舌", "舌苔黄"), ("脉", "浮")],
        "img1.png": [("面", "面色苍白"), ("声音", "嘶哑")],
        "img2.png": [("舌", "舌质红"), ("面", "面色红润")],
        "img3.png": [("脉", "数"), ("声音", "低微")],
        "img4.png": [("舌", "舌苔薄白")],
        "img5.png": [("声音", "清晰")],
        # img6/img7 无 XML → 全部「正常/无」下标 0
    }
    ds_id = _add_dataset(db, tmp_path, "smoke", images, labels_by_file)
    model_dir = tmp_path / "models"

    # --- 首次训练 2 epochs ---
    job_id = train_service.start_training(_base_config([ds_id], model_dir))
    job = train_service.train_manager.wait(job_id, timeout=900)
    assert job["status"] == "completed", job.get("error")
    assert job["device"] == "cpu"  # 本地无 GPU → 自动降级 CPU FP32（REQ-6.8-3）
    assert len(job["metrics"]) == 2
    assert [m["epoch"] for m in job["metrics"]] == [1, 2]

    latest = Path(job["checkpoint_latest"])
    best = Path(job["checkpoint_best"])
    assert latest.is_file() and best.is_file()
    ckpt = torch.load(latest, map_location="cpu", weights_only=False)
    for key in (
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "epoch",
        "best_metric",
        "config",
        "architecture",
        "num_classes",
        "category_names",
    ):
        assert key in ckpt, f"checkpoint 缺少键 {key}"
    assert ckpt["epoch"] == 2
    assert isinstance(ckpt["best_metric"], float)
    assert 0.0 <= ckpt["best_metric"] <= 1.0
    assert ckpt["architecture"] == "resnet18"
    assert list(ckpt["num_classes"]) == NUM_CLASSES_4
    assert list(ckpt["category_names"]) == CATEGORY_NAMES

    # --- 续训 2 epochs（epochs 语义 = 追加轮数，REQ-6.11-2）---
    job_id2 = train_service.resume_training(str(latest), {"epochs": 2})
    job2 = train_service.train_manager.wait(job_id2, timeout=900)
    assert job2["status"] == "completed", job2.get("error")
    assert job2["start_epoch"] == 2
    assert len(job2["metrics"]) == 2
    assert [m["epoch"] for m in job2["metrics"]] == [3, 4]
    ckpt2 = torch.load(
        Path(job2["checkpoint_latest"]), map_location="cpu", weights_only=False
    )
    assert ckpt2["epoch"] == 4
    assert Path(job2["checkpoint_best"]).is_file()


# ---------------------------------------------------------------------------
# taxonomy 硬失败（锁定决策 6）
# ---------------------------------------------------------------------------


def test_unknown_label_hard_error(tmp_path: Path, db, no_download) -> None:
    """XML 中出现 labels.yaml 之外标签 → start_training 同步 ValueError，不创建任务。"""
    ds_id = _add_dataset(db, tmp_path, "bad", ["a.png"], {"a.png": [("舌", "不存在的标签")]})
    before = len(train_service.train_manager.list_jobs())
    with pytest.raises(ValueError, match="未知标签|taxonomy"):
        train_service.start_training(_base_config([ds_id], tmp_path / "models"))
    assert len(train_service.train_manager.list_jobs()) == before


def test_unknown_category_hard_error(tmp_path: Path, db, no_download) -> None:
    """XML 中出现未知 category（如「手」）→ 同步 ValueError。"""
    ds_id = _add_dataset(db, tmp_path, "badcat", ["b.png"], {"b.png": [("手", "正常/无")]})
    with pytest.raises(ValueError, match="未知标注类别"):
        train_service.start_training(_base_config([ds_id], tmp_path / "models"))


# ---------------------------------------------------------------------------
# 配置校验 / 工具函数
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad,pattern",
    [
        ({"datasets": [1], "optimizer": "rmsprop"}, "优化器"),
        ({"datasets": [1], "loss": "mse"}, "损失函数"),
        ({"datasets": [1], "augmentation": ["nope"]}, "增强"),
        ({"datasets": [1], "backbone": "resnet101"}, "backbone"),
        ({"datasets": []}, "datasets"),
        ({"datasets": [1], "epochs": 0}, "epochs"),
        ({"datasets": [1], "batch_size": 0}, "batch_size"),
        ({"datasets": [1], "dropout": 1.5}, "dropout"),
        ({"datasets": [1], "lr": -1}, "lr"),
        ({"datasets": [1], "val_split": 1.0}, "val_split"),
    ],
)
def test_normalize_config_invalid_raises(bad: dict, pattern: str) -> None:
    with pytest.raises(ValueError, match=pattern):
        train_service.normalize_config(bad)


def test_normalize_config_defaults_and_aliases() -> None:
    cfg = train_service.normalize_config(
        {"datasets": [3, "5", 3], "augmentation": ["Flip", "rotate", "flip"]}
    )
    assert cfg["datasets"] == [3, 5]  # 去重保序
    assert cfg["backbone"] == "resnet18"
    assert cfg["epochs"] == 10
    assert cfg["loss"] == "crossentropy"
    assert cfg["augmentation"] == ["randomhorizontalflip", "randomrotation"]  # 别名归一化


def test_augmentation_transforms_compose() -> None:
    img = Image.new("RGB", (300, 200), (1, 2, 3))
    train_tf = train_service._build_train_transform(
        ["randomhorizontalflip", "randomrotation", "colorjitter", "randomcrop"]
    )
    out = train_tf(img)
    assert tuple(out.shape) == (3, 224, 224)
    val_tf = train_service._build_val_transform()
    out_val = val_tf(img)
    assert tuple(out_val.shape) == (3, 224, 224)


def test_resolve_device_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    device, n_gpus = train_service.resolve_device()
    assert n_gpus == 0
    assert device.type == "cpu"


def test_focal_loss_runs() -> None:
    loss_fn = train_service._build_loss("focalloss")
    out = loss_fn(torch.randn(4, 6), torch.randint(0, 6, (4,)))
    assert out.ndim == 0
    assert torch.isfinite(out)


def test_target_vector_maps_labels_and_defaults(tmp_path: Path) -> None:
    img = tmp_path / "x.png"
    Image.new("RGB", (32, 32)).save(img)
    xml_io.write_xml(
        img.with_suffix(".xml"),
        {
            "folder": "t",
            "filename": "x.png",
            "path": "x.png",
            "size": {"width": 32, "height": 32, "depth": 3},
            "quality": "合格",
            "labels": [{"category": "脉", "name": "浮", "confidence": 1.0}],
        },
    )
    pulse_labels = [c["labels"] for c in CATEGORIES if c["name"] == "脉"][0]
    assert train_service._target_vector(str(img), "x.png") == [
        0,
        0,
        pulse_labels.index("浮"),
        0,
    ]
    # 无 XML → 全 0（「正常/无」）
    assert train_service._target_vector(str(tmp_path / "no_xml.png"), "no_xml.png") == [
        0,
        0,
        0,
        0,
    ]
