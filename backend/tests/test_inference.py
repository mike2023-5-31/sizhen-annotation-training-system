"""推理服务单元测试（T17）：checkpoint 加载 / 批量预测形状 / 预标注往返 / 第三方权重报错。"""

from __future__ import annotations

import collections
from pathlib import Path

import pytest
import torch
import torchvision.models as tvm
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.labels_config import CATEGORIES
from app.models.factory import build_model
from app.models.orm import Dataset, Record
from app.services import inference_service, xml_io

CATEGORY_NAMES = [c["name"] for c in CATEGORIES]
LABELS_BY_CATEGORY = {c["name"]: c["labels"] for c in CATEGORIES}
NUM_CLASSES_4 = [len(c["labels"]) for c in CATEGORIES]


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def _make_image(path: Path, size: tuple[int, int] = (64, 48), color: tuple = (120, 30, 90)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _build_resnet18(monkeypatch) -> torch.nn.Module:
    """构建 resnet18 四头模型（weights=None，避免测试期下载大权重）。"""
    original = tvm.resnet18
    monkeypatch.setattr(tvm, "resnet18", lambda weights=None: original(weights=None))
    return build_model("resnet18", NUM_CLASSES_4, dropout=0.0)


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    architecture: str = "resnet18",
    num_classes: list[int] | None = None,
    category_names: list[str] | None = None,
) -> Path:
    checkpoint = {
        "model_state": model.state_dict(),
        "architecture": architecture,
        "num_classes": num_classes if num_classes is not None else NUM_CLASSES_4,
        "category_names": category_names if category_names is not None else CATEGORY_NAMES,
        "dropout": 0.0,
    }
    torch.save(checkpoint, path)
    return path


@pytest.fixture
def db_session_factory(tmp_path: Path, monkeypatch):
    """临时 SQLite DB（建表 + Session 工厂），替换 inference_service.SessionLocal。"""
    db_path = tmp_path / "test_inference.db"
    engine = create_engine(
        f"sqlite:///{db_path.as_posix()}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    monkeypatch.setattr(inference_service, "SessionLocal", factory)
    return tmp_path


def _add_dataset(
    tmp_path: Path,
    name: str,
    images: list[str],
    dataset_dir: Path | None = None,
) -> tuple[int, Path]:
    """建 Dataset + 每图一条 Record 与真实图片文件，返回 (dataset_id, 图片目录)。"""
    with inference_service.SessionLocal() as db:
        dataset = Dataset(name=name)
        db.add(dataset)
        db.flush()
        dataset_id = dataset.id
        for filename in images:
            if dataset_dir is None:
                path = tmp_path / "imgs" / filename
            else:
                path = dataset_dir / filename
            _make_image(path)
            db.add(
                Record(
                    dataset_id=dataset_id,
                    filename=filename,
                    image_path=str(path.resolve()),
                    quality="",
                )
            )
        db.commit()
        return dataset_id, (dataset_dir if dataset_dir is not None else tmp_path / "imgs")


# ---------------------------------------------------------------------------
# load_model_from_dir
# ---------------------------------------------------------------------------


def test_load_model_roundtrip(tmp_path: Path, monkeypatch) -> None:
    """系统训练产物：load 后结构与权重一致，模型处于 eval 模式，meta 完整。"""
    model = _build_resnet18(monkeypatch)
    ckpt_path = _save_checkpoint(tmp_path / "best.pth", model)

    loaded, meta = inference_service.load_model_from_dir(str(ckpt_path))

    assert loaded.training is False
    assert meta["architecture"] == "resnet18"
    assert meta["num_classes"] == NUM_CLASSES_4
    assert meta["category_names"] == CATEGORY_NAMES
    loaded_sd = loaded.state_dict()
    original_sd = model.state_dict()
    assert set(loaded_sd.keys()) == set(original_sd.keys())
    for key in original_sd:
        assert torch.equal(loaded_sd[key], original_sd[key])


def test_load_model_resnet50(tmp_path: Path, monkeypatch) -> None:
    """resnet50 checkpoint 同样可重建载权（锁定决策 8：两结构均保证）。"""
    original = tvm.resnet50
    monkeypatch.setattr(tvm, "resnet50", lambda weights=None: original(weights=None))
    model = build_model("resnet50", NUM_CLASSES_4)
    ckpt_path = _save_checkpoint(tmp_path / "r50.pth", model, architecture="resnet50")

    loaded, meta = inference_service.load_model_from_dir(str(ckpt_path))

    assert loaded.training is False
    assert meta["architecture"] == "resnet50"
    outs = loaded(torch.randn(1, 3, 224, 224))
    assert set(outs.keys()) == set(CATEGORY_NAMES)


def test_load_foreign_pth_raises(tmp_path: Path) -> None:
    """第三方权重（无 architecture/num_classes 键）→ ValueError「无法识别模型结构」。"""
    foreign = tmp_path / "foreign.pth"
    torch.save({"state_dict": collections.OrderedDict(), "classes": 1000}, foreign)
    with pytest.raises(ValueError, match="无法识别模型结构"):
        inference_service.load_model_from_dir(str(foreign))


def test_load_unsupported_architecture_raises(tmp_path: Path, monkeypatch) -> None:
    """architecture 非 resnet18/50（如自定义/未知）→ ValueError 显式报错。"""
    model = _build_resnet18(monkeypatch)
    ckpt_path = _save_checkpoint(tmp_path / "custom.pth", model, architecture="efficientnet")
    with pytest.raises(ValueError, match="无法识别模型结构"):
        inference_service.load_model_from_dir(str(ckpt_path))


def test_load_non_dict_raises(tmp_path: Path) -> None:
    """非 dict 的 .pth（如仅权重张量）→ ValueError。"""
    p = tmp_path / "raw.pth"
    torch.save(torch.zeros(1), p)
    with pytest.raises(ValueError, match="无法识别模型结构"):
        inference_service.load_model_from_dir(str(p))


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="不存在"):
        inference_service.load_model_from_dir(str(tmp_path / "nope.pth"))


# ---------------------------------------------------------------------------
# predict_dataset
# ---------------------------------------------------------------------------


def test_predict_dataset_shape_and_probabilities(
    db_session_factory, monkeypatch
) -> None:
    """每图 4 个预测，category 顺序 = 标签体系；label 属于该类别；confidence ∈ [0,1]。"""
    tmp_path = db_session_factory
    dataset_id, _ = _add_dataset(tmp_path, "推理集", ["a.jpg", "b.jpg"])

    model = _build_resnet18(monkeypatch)
    results = inference_service.predict_dataset(dataset_id, model)

    assert len(results) == 2
    assert [r["filename"] for r in results] == ["a.jpg", "b.jpg"]
    for entry in results:
        assert len(entry["predictions"]) == 4
        assert [p["category"] for p in entry["predictions"]] == CATEGORY_NAMES
        for pred in entry["predictions"]:
            assert pred["label"] in LABELS_BY_CATEGORY[pred["category"]]
            assert 0.0 <= pred["confidence"] <= 1.0


def test_predict_dataset_deterministic(db_session_factory, monkeypatch) -> None:
    """eval 模式 + no_grad：同一模型两次预测结果完全一致。"""
    tmp_path = db_session_factory
    dataset_id, _ = _add_dataset(tmp_path, "确定性集", ["a.jpg"])

    model = _build_resnet18(monkeypatch)
    first = inference_service.predict_dataset(dataset_id, model)
    second = inference_service.predict_dataset(dataset_id, model)
    assert first == second


def test_predict_dataset_unknown_dataset_raises(db_session_factory, monkeypatch) -> None:
    model = _build_resnet18(monkeypatch)
    with pytest.raises(ValueError, match="数据集"):
        inference_service.predict_dataset(999, model)


def test_predict_dataset_empty_dataset(db_session_factory, monkeypatch) -> None:
    tmp_path = db_session_factory
    with inference_service.SessionLocal() as db:
        dataset = Dataset(name="空集")
        db.add(dataset)
        db.commit()
        dataset_id = dataset.id

    model = _build_resnet18(monkeypatch)
    assert inference_service.predict_dataset(dataset_id, model) == []


# ---------------------------------------------------------------------------
# generate_preannotation_xml
# ---------------------------------------------------------------------------


def test_preannotation_roundtrip(db_session_factory, monkeypatch) -> None:
    """仅空标注图片生成预标注；已有手动标注不覆盖；XML 读回与预测一致。"""
    tmp_path = db_session_factory
    dataset_id, imgs_dir = _add_dataset(
        tmp_path, "预标注集", ["manual.jpg", "empty.jpg", "no_xml.jpg"]
    )

    with inference_service.SessionLocal() as db:
        dataset_name = db.get(Dataset, dataset_id).name

    manual_xml = imgs_dir / "manual.xml"
    xml_io.write_xml(
        manual_xml,
        {
            "folder": dataset_name,
            "filename": "manual.jpg",
            "path": f"{dataset_id}/manual.jpg",
            "size": {"width": 64, "height": 48, "depth": 3},
            "quality": "合格",
            "labels": [{"category": "舌", "name": "舌苔黄", "confidence": 1.0}],
        },
    )
    empty_xml = imgs_dir / "empty.xml"
    xml_io.write_xml(
        empty_xml,
        {
            "folder": dataset_name,
            "filename": "empty.jpg",
            "path": f"{dataset_id}/empty.jpg",
            "size": {"width": 64, "height": 48, "depth": 3},
            "quality": "",
            "labels": [],
        },
    )

    model = _build_resnet18(monkeypatch)
    written = inference_service.generate_preannotation_xml(dataset_id, model)
    assert written == 2  # empty.jpg + no_xml.jpg；manual.jpg 跳过

    predictions = {
        entry["filename"]: entry["predictions"]
        for entry in inference_service.predict_dataset(dataset_id, model)
    }

    manual_back = xml_io.read_xml(manual_xml)
    assert [l["name"] for l in manual_back["labels"]] == ["舌苔黄"]
    assert manual_back["quality"] == "合格"

    for filename in ("empty.jpg", "no_xml.jpg"):
        data = xml_io.read_xml(imgs_dir / f"{Path(filename).stem}.xml")
        assert data["filename"] == filename
        assert data["folder"] == dataset_name
        assert data["path"] == f"{dataset_id}/{filename}"
        assert len(data["labels"]) == 4
        for label, pred in zip(data["labels"], predictions[filename]):
            assert label["category"] == pred["category"]
            assert label["name"] == pred["label"]
            assert label["confidence"] == pred["confidence"]
        for label in data["labels"]:
            assert 0.0 <= label["confidence"] <= 1.0


def test_preannotation_preserves_existing_fields(db_session_factory, monkeypatch) -> None:
    """既有空标注 XML 的 size/quality 在预标注后原样保留。"""
    tmp_path = db_session_factory
    dataset_id, imgs_dir = _add_dataset(tmp_path, "保留字段集", ["keep.jpg"])
    xml_io.write_xml(
        imgs_dir / "keep.xml",
        {
            "folder": "旧文件夹",
            "filename": "keep.jpg",
            "path": "old/path.jpg",
            "size": {"width": 640, "height": 480, "depth": 3},
            "quality": "不合格",
            "labels": [],
        },
    )

    model = _build_resnet18(monkeypatch)
    assert inference_service.generate_preannotation_xml(dataset_id, model) == 1

    data = xml_io.read_xml(imgs_dir / "keep.xml")
    assert data["size"] == {"width": 640, "height": 480, "depth": 3}
    assert data["quality"] == "不合格"
    assert data["folder"] == "旧文件夹"
    assert data["path"] == "old/path.jpg"
    assert len(data["labels"]) == 4


def test_preannotation_skips_all_manual(db_session_factory, monkeypatch) -> None:
    """全部图片已有人工标注 → 生成数为 0，文件不变。"""
    tmp_path = db_session_factory
    dataset_id, imgs_dir = _add_dataset(tmp_path, "全人工集", ["m.jpg"])
    xml_io.write_xml(
        imgs_dir / "m.xml",
        {
            "folder": "d",
            "filename": "m.jpg",
            "path": "m.jpg",
            "size": None,
            "quality": "合格",
            "labels": [{"category": "脉", "name": "浮", "confidence": 1.0}],
        },
    )

    model = _build_resnet18(monkeypatch)
    assert inference_service.generate_preannotation_xml(dataset_id, model) == 0
    assert len(xml_io.read_xml(imgs_dir / "m.xml")["labels"]) == 1


def test_preannotation_unknown_dataset_raises(db_session_factory, monkeypatch) -> None:
    model = _build_resnet18(monkeypatch)
    with pytest.raises(ValueError, match="数据集"):
        inference_service.generate_preannotation_xml(999, model)
