"""自定义模型端到端测试（需求 6.7）：上传 / 配置校验 / 训练 / checkpoint / 推理重建。

覆盖链路：
- ``POST /api/models/custom`` 上传 .py → ``{"model_path": "custom/<name>.py"}``；
  非 .py / 空文件 → 400；
- ``normalize_config``：backbone=custom 强制 custom_model_path（缺省/绝对路径/
  ``..`` 穿越/非 .py → ValueError）；
- 自定义模型（4 头 dict 输出）CPU 训练 1 epoch → checkpoint 含
  ``architecture == "custom"`` 与 ``custom_model_path``；
- ``load_model_from_dir`` 按 custom_model_path 重建 → predict 输出 4 分类；
- 输出契约非法（非 4 头）→ ``wrap_custom_model`` ValueError；
- custom 存档缺 custom_model_path → 明确 ValueError。
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import config
from app.db import Base
from app.labels_config import CATEGORIES
from app.main import app
from app.models import factory
from app.models.orm import Dataset, Record
from app.services import inference_service, train_service

CATEGORY_NAMES = [c["name"] for c in CATEGORIES]
NUM_CLASSES_4 = [len(c["labels"]) for c in CATEGORIES]

_TINY_4HEAD_SRC = textwrap.dedent(
    """
    import torch.nn as nn

    CATEGORIES = ["舌", "面", "脉", "声音"]


    class _TinyModel(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.heads = nn.ModuleDict(
                {name: nn.Linear(3, n_cat) for name, n_cat in zip(CATEGORIES, num_classes)}
            )

        def forward(self, x):
            feats = self.pool(x).flatten(1)
            return {name: head(feats) for name, head in self.heads.items()}


    def build_model(num_classes):
        return _TinyModel(num_classes)
    """
)

_TINY_TUPLE_SRC = textwrap.dedent(
    """
    import torch.nn as nn


    class _TinyTuple(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.num_classes = list(num_classes)
            self.fc = nn.Linear(3, sum(num_classes))

        def forward(self, x):
            return tuple(self.fc(self.pool(x).flatten(1)).split(self.num_classes, dim=1))


    def build_model(num_classes):
        return _TinyTuple(num_classes)
    """
)

_TINY_BAD_SRC = textwrap.dedent(
    """
    import torch.nn as nn


    class _TinyBad(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(3, sum(num_classes))

        def forward(self, x):
            return self.fc(self.pool(x).flatten(1))  # 单张量：非 4 头 logits


    def build_model(num_classes):
        return _TinyBad(num_classes)
    """
)


@pytest.fixture
def model_dir(tmp_path: Path, monkeypatch) -> Path:
    """隔离模型目录：patch config.MODEL_DIR（训练/推理服务运行时读取）。"""
    mdir = tmp_path / "models"
    mdir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "MODEL_DIR", mdir)
    return mdir


@pytest.fixture
def db(tmp_path: Path, monkeypatch):
    """临时 SQLite + 替换 train_service/inference_service 的 SessionLocal。"""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test_custom.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    monkeypatch.setattr(train_service, "SessionLocal", session)
    monkeypatch.setattr(inference_service, "SessionLocal", session)
    return session


@pytest.fixture
def client(db, model_dir, login_admin):
    with TestClient(app) as test_client:
        login_admin(test_client)
        yield test_client


def _add_dataset(
    session_factory,
    tmp_path: Path,
    name: str,
    images: list[str],
) -> int:
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
            from app.services import xml_io
            xml_io.write_xml(path.with_suffix(".xml"), {
                "quality": "合格", "labels": [{"category": "舌", "name": "正常/无"}],
            })
            db.add(
                Record(
                    dataset_id=ds_id,
                    filename=fn,
                    image_path=str(path.resolve()),
                    quality="合格",
                )
            )
        db.commit()
    return ds_id


def _custom_config(ds_ids: list[int], model_dir: Path, model_path: str) -> dict:
    return {
        "datasets": ds_ids,
        "backbone": "custom",
        "custom_model_path": model_path,
        "epochs": 1,
        "batch_size": 4,
        "lr": 1e-3,
        "optimizer": "adam",
        "loss": "crossentropy",
        "dropout": 0.0,
        "augmentation": [],
        "fp16": False,
        "val_split": 0.0,
        "seed": 0,
        "model_dir": str(model_dir),
    }


# ---------------------------------------------------------------------------
# 上传端点（需求 6.7）
# ---------------------------------------------------------------------------


def test_upload_custom_model_endpoint(client: TestClient, model_dir: Path) -> None:
    """上传 .py → 保存到 MODEL_DIR/custom/ 并返回相对路径。"""
    resp = client.post(
        "/api/models/custom",
        files={"file": ("tiny_model.py", _TINY_4HEAD_SRC.encode("utf-8"), "text/x-python")},
    )
    assert resp.status_code == 200
    assert resp.json() == {"model_path": "custom/tiny_model.py"}
    saved = model_dir / "custom" / "tiny_model.py"
    assert saved.is_file()
    assert saved.read_text(encoding="utf-8") == _TINY_4HEAD_SRC


def test_upload_custom_model_rejects_non_py(client: TestClient) -> None:
    """非 .py 文件 → 400。"""
    resp = client.post(
        "/api/models/custom",
        files={"file": ("model.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 400
    assert ".py" in resp.json()["detail"]


def test_upload_custom_model_rejects_empty(client: TestClient) -> None:
    """空 .py 文件 → 400。"""
    resp = client.post(
        "/api/models/custom",
        files={"file": ("empty.py", b"   \n", "text/x-python")},
    )
    assert resp.status_code == 400


def test_upload_custom_model_basename_only(client: TestClient, model_dir: Path) -> None:
    """文件名含路径成分时仅取 basename（防路径注入）。"""
    resp = client.post(
        "/api/models/custom",
        files={"file": ("../../evil.py", _TINY_4HEAD_SRC.encode("utf-8"), "text/x-python")},
    )
    assert resp.status_code == 200
    assert resp.json() == {"model_path": "custom/evil.py"}
    assert (model_dir / "custom" / "evil.py").is_file()


# ---------------------------------------------------------------------------
# 配置校验（normalize_config，需求 6.7）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "custom_model_path,pattern",
    [
        (None, "custom_model_path"),
        ("", "custom_model_path"),
        ("<ABSOLUTE>", "相对路径"),
        ("../evil.py", "越出模型目录"),
        ("model.txt", ".py"),
        ("custom/nope.py", "不存在"),
    ],
)
def test_normalize_config_custom_invalid_raises(
    custom_model_path, pattern: str, model_dir: Path
) -> None:
    raw = {"datasets": [1], "backbone": "custom"}
    if custom_model_path == "<ABSOLUTE>":
        raw["custom_model_path"] = str(Path.cwd() / "model.py")  # 任意绝对路径
    elif custom_model_path is not None:
        raw["custom_model_path"] = custom_model_path
    with pytest.raises(ValueError, match=pattern):
        train_service.normalize_config(raw)


def test_normalize_config_custom_ok(model_dir: Path) -> None:
    (model_dir / "custom").mkdir(exist_ok=True)
    (model_dir / "custom" / "ok.py").write_text(_TINY_4HEAD_SRC, encoding="utf-8")
    cfg = train_service.normalize_config(
        {"datasets": [1], "backbone": "custom", "custom_model_path": "custom/ok.py"}
    )
    assert cfg["backbone"] == "custom"
    assert cfg["custom_model_path"] == "custom/ok.py"


# ---------------------------------------------------------------------------
# 输出契约（wrap_custom_model）
# ---------------------------------------------------------------------------


def test_wrap_custom_model_dict_output(tmp_path: Path) -> None:
    p = tmp_path / "d.py"
    p.write_text(_TINY_4HEAD_SRC, encoding="utf-8")
    wrapped = factory.wrap_custom_model(
        factory.load_custom_model(str(p), NUM_CLASSES_4), NUM_CLASSES_4
    )
    outs = wrapped(torch.randn(2, 3, 224, 224))
    assert list(outs.keys()) == CATEGORY_NAMES
    for name, n_cat in zip(CATEGORY_NAMES, NUM_CLASSES_4):
        assert tuple(outs[name].shape) == (2, n_cat)


def test_wrap_custom_model_tuple_output(tmp_path: Path) -> None:
    p = tmp_path / "t.py"
    p.write_text(_TINY_TUPLE_SRC, encoding="utf-8")
    wrapped = factory.wrap_custom_model(
        factory.load_custom_model(str(p), NUM_CLASSES_4), NUM_CLASSES_4
    )
    outs = wrapped(torch.randn(2, 3, 224, 224))
    assert list(outs.keys()) == CATEGORY_NAMES
    for name, n_cat in zip(CATEGORY_NAMES, NUM_CLASSES_4):
        assert tuple(outs[name].shape) == (2, n_cat)


def test_wrap_custom_model_bad_output_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.py"
    p.write_text(_TINY_BAD_SRC, encoding="utf-8")
    with pytest.raises(ValueError, match="自定义模型输出需为 4 个分类头 logits"):
        factory.wrap_custom_model(
            factory.load_custom_model(str(p), NUM_CLASSES_4), NUM_CLASSES_4
        )


# ---------------------------------------------------------------------------
# 训练 + checkpoint + 推理重建（需求 6.7 端到端）
# ---------------------------------------------------------------------------


def test_custom_model_train_checkpoint_reload_predict(
    tmp_path: Path, db, model_dir: Path
) -> None:
    """1 epoch CPU 训练自定义模型 → 存档含 custom 元数据 → 推理侧重建并预测。"""
    custom_dir = model_dir / "custom"
    custom_dir.mkdir(parents=True, exist_ok=True)
    py_path = custom_dir / "tiny_model.py"
    py_path.write_text(_TINY_4HEAD_SRC, encoding="utf-8")

    ds_id = _add_dataset(db, tmp_path, "custom_ds", [f"img{i}.png" for i in range(4)])

    job_id = train_service.start_training(
        _custom_config([ds_id], model_dir, "custom/tiny_model.py")
    )
    job = train_service.train_manager.wait(job_id, timeout=900)
    assert job["status"] == "completed", job.get("error")
    assert len(job["metrics"]) == 1

    latest = Path(job["checkpoint_latest"])
    assert latest.is_file()
    ckpt = torch.load(latest, map_location="cpu", weights_only=False)
    assert ckpt["architecture"] == "custom"
    assert ckpt["custom_model_path"] == "custom/tiny_model.py"
    assert list(ckpt["num_classes"]) == NUM_CLASSES_4

    loaded, meta = inference_service.load_model_from_dir(str(latest))
    assert loaded.training is False
    assert meta["architecture"] == "custom"
    assert meta["custom_model_path"] == "custom/tiny_model.py"

    predictions = inference_service.predict_dataset(ds_id, loaded)
    assert len(predictions) == 4
    for entry in predictions:
        assert len(entry["predictions"]) == 4
        assert [p["category"] for p in entry["predictions"]] == CATEGORY_NAMES


def test_load_model_from_dir_custom_missing_path_raises(tmp_path: Path, model_dir: Path) -> None:
    """custom 存档缺 custom_model_path → 明确 ValueError（无法重建）。"""
    tiny = torch.nn.Linear(3, 3)  # 占位权重（不真正加载）
    ckpt_path = tmp_path / "custom_ckpt.pth"
    torch.save(
        {
            "model_state": tiny.state_dict(),
            "architecture": "custom",
            "num_classes": NUM_CLASSES_4,
            "category_names": CATEGORY_NAMES,
            "dropout": 0.0,
        },
        ckpt_path,
    )
    with pytest.raises(ValueError, match="custom_model_path"):
        inference_service.load_model_from_dir(str(ckpt_path))
