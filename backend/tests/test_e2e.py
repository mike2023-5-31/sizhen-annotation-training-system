"""端到端集成测试（T25，锁定决策 9）：services 全链路 + 隔离 DATA_DIR/DB_URL。

单条用例按真实业务顺序走完 8 个阶段（全程合成小图、无网络、无 GPU 依赖）：
  1) 建数据集 → ``import_zip`` 导入小 ZIP（图片 + 同名 XML）→ Record 落库；
  2) ``save_labels`` 写回一张图 → XML 自动写盘（auto-save 语义）；
  3) ``export_dataset`` → ZIP 字节 → 再 ``import_zip`` → 标签 XML 与 quality
     逐字节一致（往返保真，锁定决策 12）；
  4) ``start_training``（resnet18 / CPU / epochs=2）→ 等待完成 →
     MODEL_DIR 有 latest/best 存档且 metrics 长度 == 2；
  5) 训练评估（混淆矩阵 + ROC）→ 4 个 category 全部覆盖；
  6) ``load_model_from_dir`` + ``predict_dataset`` → 每图 4 个预测；
  7) ``generate_preannotation_xml`` → 仅空标签图片生成、数量 > 0、XML 可读、
     人工标注不被覆盖；
  8) ``onnx_service.convert`` + ``compare`` → onnx 文件存在、4 头全部 PASS
     （max_abs_diff < 1e-3，锁定决策 5）、download 字节与原产物一致。

隔离策略（env-driven 精神，锁定决策 10）：monkeypatch ``app.config`` 的
DATA_DIR / 派生目录 / DB_URL 指向 tmp_path，并把各服务模块的 ``SessionLocal``
统一绑定到临时 SQLite 引擎（与 test_train/test_inference 同款约定）；
训练用 ``weights=None`` 禁下载预训练权重。断言不含任何 GPU 结论——
GPU 真实路径（FP16 / DataParallel）由 ``docs/GPU_SMOKE.md`` 在服务器验证。
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
import torchvision.models as tvm
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import config
from app import db as app_db
from app.db import Base
from app.labels_config import CATEGORIES
from app.models.orm import Dataset, Record
from app.routers import onnx as onnx_router
from app.services import (
    annotation_service,
    dataset_service,
    export_service,
    import_service,
    inference_service,
    onnx_service,
    train_service,
    xml_io,
)

CATEGORY_NAMES = [c["name"] for c in CATEGORIES]

# 合成数据：6 张小图，标签分布刻意覆盖「人工标注 / 空标签 / 不合格」三类，
# 供 save_labels / 往返 / 预标注「仅空标签生成、不覆盖人工标注」各阶段复用。
_IMAGES: list[str] = [f"img{i}.png" for i in range(6)]
_LABELS_BY_FILE: dict[str, list[tuple[str, str]]] = {
    "img0.png": [],  # 阶段 2 由 save_labels 写入
    "img1.png": [("舌", "舌苔黄")],
    "img2.png": [("面", "面色苍白"), ("脉", "浮")],
    "img3.png": [],  # 空标签 + 不合格（仅导入路径可带入）
    "img4.png": [],
    "img5.png": [("声音", "清晰")],
}
_QUALITY_BY_FILE: dict[str, str] = {
    "img0.png": "",
    "img1.png": "合格",
    "img2.png": "合格",
    "img3.png": "不合格",
    "img4.png": "",
    "img5.png": "合格",
}
# 阶段 7 预标注应只作用于空标签图片：img3 / img4（其余 4 张均有人工标签）
_EXPECTED_PREANNOTATION_COUNT = 2


# ---------------------------------------------------------------------------
# fixture：隔离的 DATA_DIR / DB_URL + 各服务 SessionLocal 指向临时 SQLite
# ---------------------------------------------------------------------------


@pytest.fixture
def e2e_env(tmp_path: Path, monkeypatch):
    """monkeypatch app.config 全路径 + DB_URL 到 tmp_path，重建临时引擎。

    各服务模块的 ``SessionLocal``（from app.db import 绑定）逐个替换为临时
    引擎的会话工厂；同时禁 resnet18 预训练权重下载（训练线程内建模型）。
    """
    data_dir = tmp_path / "data"
    datasets_dir = data_dir / "datasets"
    model_dir = data_dir / "models"
    export_dir = data_dir / "exports"
    upload_dir = data_dir / "uploads"
    for d in (datasets_dir, model_dir, export_dir, upload_dir):
        d.mkdir(parents=True)

    # app.config 全量隔离（锁定决策 10 精神：路径/env 唯一来源）
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "DATASETS_DIR", datasets_dir)
    monkeypatch.setattr(config, "MODEL_DIR", model_dir)
    monkeypatch.setattr(config, "EXPORT_DIR", export_dir)
    monkeypatch.setattr(config, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(config, "STATIC_DIR", datasets_dir)
    monkeypatch.setattr(
        config, "DB_URL", f"sqlite:///{(data_dir / 'e2e.db').as_posix()}"
    )

    db_path = data_dir / "e2e.db"
    engine = create_engine(
        f"sqlite:///{db_path.as_posix()}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    _service_modules = (
        app_db,
        import_service,
        annotation_service,
        export_service,
        dataset_service,
        inference_service,
        train_service,
    )
    for mod in _service_modules:
        monkeypatch.setattr(mod, "SessionLocal", session_factory)

    # 禁预训练权重下载（无网络确定性；结构不变，与 test_train 同款约定）
    original = tvm.resnet18
    monkeypatch.setattr(tvm, "resnet18", lambda weights=None: original(weights=None))

    return {
        "session_factory": session_factory,
        "model_dir": model_dir,
        "export_dir": export_dir,
    }


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def _build_source_zip(staging: Path) -> bytes:
    """生成导入用小 ZIP：根级平铺的 6 张 PNG + 同名 XML（import 仅取 basename）。"""
    colors = [(110 + i * 24, 70, 40 + i * 13) for i in range(len(_IMAGES))]
    for fn, color in zip(_IMAGES, colors):
        Image.new("RGB", (48, 48), color).save(staging / fn)
        xml_io.write_xml(
            staging / f"{Path(fn).stem}.xml",
            {
                "folder": "e2e-ds1",
                "filename": fn,
                "path": f"e2e-ds1/{fn}",
                "size": {"width": 48, "height": 48, "depth": 3},
                "quality": _QUALITY_BY_FILE[fn],
                "labels": [
                    {"category": c, "name": n, "confidence": 1.0}
                    for c, n in _LABELS_BY_FILE[fn]
                ],
            },
        )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(staging.iterdir()):
            zf.write(f, f.name)
    return buf.getvalue()


def _records_by_filename(session_factory, dataset_id: int) -> dict[str, Record]:
    with session_factory() as db:
        recs = db.query(Record).filter(Record.dataset_id == dataset_id).all()
        return {r.filename: r for r in recs}


# ---------------------------------------------------------------------------
# 全链路用例（8 阶段顺序执行，任一阶段失败即阻断后续——e2e 语义）
# ---------------------------------------------------------------------------


def test_full_pipeline_e2e(tmp_path: Path, e2e_env) -> None:
    session_factory = e2e_env["session_factory"]
    model_dir: Path = e2e_env["model_dir"]
    export_dir: Path = e2e_env["export_dir"]

    # -- 阶段 1：建数据集 + import_zip（图片 + XML → Record 落库）-----------
    staging = tmp_path / "source-zip"
    staging.mkdir()
    zip_bytes = _build_source_zip(staging)

    ds1 = dataset_service.create_dataset("e2e-ds1")
    result1 = import_service.import_zip(ds1.id, zip_bytes)
    assert result1["imported"] == len(_IMAGES)
    assert not result1["errors"] and not result1["skipped"]
    recs1 = _records_by_filename(session_factory, ds1.id)
    assert sorted(recs1) == sorted(_IMAGES)
    for fn in _IMAGES:  # 图片/XML 均落盘于 DATASETS_DIR/<id>/
        assert (config.DATASETS_DIR / str(ds1.id) / fn).is_file()
        assert (config.DATASETS_DIR / str(ds1.id) / f"{Path(fn).stem}.xml").is_file()

    # -- 阶段 2：save_labels 写回一张图 → XML 自动写盘（auto-save）----------
    saved = annotation_service.save_labels(
        ds1.id,
        "img0.png",
        [
            {"category": "舌", "name": "舌苔黄"},
            {"category": "面", "name": "面色苍白"},
            {"category": "脉", "name": "浮"},
            {"category": "声音", "name": "清晰"},
        ],
        "合格",
    )
    img0_xml = config.DATASETS_DIR / str(ds1.id) / "img0.xml"
    assert img0_xml.is_file()
    on_disk = xml_io.read_xml(img0_xml)
    assert len(on_disk["labels"]) == 4
    assert {l["category"] for l in on_disk["labels"]} == set(CATEGORY_NAMES)
    assert all(l["confidence"] == 1.0 for l in on_disk["labels"])  # 人工标注
    assert on_disk["quality"] == "合格" and saved["quality"] == "合格"
    with session_factory() as db:  # quality 镜像入库（锁定决策 11）
        assert db.get(Record, recs1["img0.png"].id).quality == "合格"

    # -- 阶段 3：export_dataset → ZIP 字节 → 再 import_zip（往返保真）--------
    exported = export_service.export_dataset(ds1.id)
    assert isinstance(exported, bytes) and exported

    ds2 = dataset_service.create_dataset("e2e-ds2")
    result2 = import_service.import_zip(ds2.id, exported)
    assert result2["imported"] == len(_IMAGES)
    assert not result2["errors"] and not result2["skipped"]
    recs2 = _records_by_filename(session_factory, ds2.id)
    assert sorted(recs2) == sorted(_IMAGES)
    # 重新取 ds1 记录（阶段 2 save_labels 已镜像更新 quality，旧快照已过期）
    recs1 = _records_by_filename(session_factory, ds1.id)

    for fn in _IMAGES:  # 图片 + 标签 XML 逐字节一致 + quality 一致
        img1 = Path(recs1[fn].image_path)
        img2 = Path(recs2[fn].image_path)
        assert img2.is_file()
        assert img1.read_bytes() == img2.read_bytes()
        assert (
            img1.with_suffix(".xml").read_bytes()
            == img2.with_suffix(".xml").read_bytes()
        )
        assert recs1[fn].quality == recs2[fn].quality

    # -- 阶段 4：start_training（resnet18 / CPU / epochs=2）→ 等待完成 ------
    job_id = train_service.start_training(
        {
            "datasets": [ds2.id],  # 用往返产物训练，验证整链可训练
            "backbone": "resnet18",
            "epochs": 2,
            "batch_size": 2,
            "lr": 1e-3,
            "optimizer": "adam",
            "loss": "crossentropy",
            "dropout": 0.0,
            "augmentation": [],
            "fp16": False,
            "val_split": 0.5,
            "seed": 0,
            "model_dir": str(model_dir),
        }
    )
    view = train_service.train_manager.wait(job_id, timeout=600)
    assert view["status"] == "completed", f"训练失败: {view.get('error')}"
    assert view["device"] == "cpu" and view["use_amp"] is False  # 本地无 GPU 降级
    assert len(view["metrics"]) == 2  # metrics 长度 == epochs
    latest_path = Path(view["checkpoint_latest"])
    best_path = Path(view["checkpoint_best"])
    assert latest_path.is_file() and latest_path.parent == model_dir
    assert best_path.is_file() and best_path.parent == model_dir

    # -- 阶段 5：训练评估（混淆矩阵 + ROC）→ 4 个 category 全覆盖 ----------
    eval_report = train_service.train_manager.compute_eval(job_id)
    # seed=0下4个有效样本的验证集为img2、img0；未知属性不补正常。
    assert set(eval_report["confusion"]) == set(CATEGORY_NAMES)
    assert set(eval_report["roc"]) == set(CATEGORY_NAMES)
    expected_counts = {"舌": 1, "面": 2, "脉": 2, "声音": 1}
    for name in CATEGORY_NAMES:
        matrix = eval_report["confusion"][name]
        assert sum(sum(row) for row in matrix) == expected_counts[name]
        assert matrix and all(len(row) == len(matrix) for row in matrix)
        roc = eval_report["roc"][name]
        assert set(roc) >= {"fpr", "tpr", "auc"}
        assert roc["auc"] is None  # 每个头只有一种已观察标签，不伪造ROC

    # -- 阶段 6：load_model_from_dir + predict_dataset → 每图 4 个预测 ------
    model, meta = inference_service.load_model_from_dir(str(latest_path))
    assert meta["architecture"] == "resnet18"
    predictions = inference_service.predict_dataset(ds2.id, model)
    assert len(predictions) == len(_IMAGES)
    for entry in predictions:
        assert [p["category"] for p in entry["predictions"]] == CATEGORY_NAMES
        assert all(0.0 <= p["confidence"] <= 1.0 for p in entry["predictions"])

    # -- 阶段 7：generate_preannotation_xml → 仅空标签图生成、不覆盖人工 -----
    written = inference_service.generate_preannotation_xml(ds2.id, model)
    assert written == _EXPECTED_PREANNOTATION_COUNT  # img3 / img4
    pre_xml = config.DATASETS_DIR / str(ds2.id) / "img3.xml"
    pre = xml_io.read_xml(pre_xml)
    assert len(pre["labels"]) == 4
    assert all(0.0 <= l["confidence"] <= 1.0 for l in pre["labels"])  # 预标注概率
    # 人工标注（img0）不被预标注覆盖（REQ-6.18-4）
    img0_after = xml_io.read_xml(config.DATASETS_DIR / str(ds2.id) / "img0.xml")
    assert all(l["confidence"] == 1.0 for l in img0_after["labels"])

    # -- 阶段 8：onnx convert + compare（4 头 < 1e-3）+ download 字节一致 ---
    onnx_path = export_dir / "e2e.onnx"
    onnx_service.convert(model, str(onnx_path))
    assert onnx_path.is_file() and onnx_path.stat().st_size > 0

    report = onnx_service.compare(model, str(onnx_path))
    assert [r["category"] for r in report["results"]] == CATEGORY_NAMES
    assert all(r["pass"] and r["max_abs_diff"] < 1e-3 for r in report["results"])
    assert report["overall_pass"] is True

    # download（router 直调，走 EXPORT_DIR 路径守卫）与原产物逐字节一致
    response = onnx_router.download("e2e.onnx")
    assert Path(response.path).read_bytes() == onnx_path.read_bytes()
