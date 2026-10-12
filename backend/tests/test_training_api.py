"""训练监控 API 测试（T14）：状态轮询 / WebSocket 推送 / 混淆矩阵与 ROC 评估。

覆盖 REQ-6.12（实时指标、动态曲线推送、刷新重建）与 REQ-6.13
（4 类别混淆矩阵 + one-vs-rest ROC/AUC、概率保留、空验证集不报错）。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
import torchvision.models as tvm
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.labels_config import CATEGORIES
from app.main import app
from app.models.orm import Dataset, Record
from app.services import train_service, xml_io

CATEGORY_NAMES = [c["name"] for c in CATEGORIES]
NUM_CLASSES = [len(c["labels"]) for c in CATEGORIES]


# ---------------------------------------------------------------------------
# 工具 / fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path: Path, monkeypatch):
    """临时 SQLite + 替换 train_service.SessionLocal（与路由共享同一模块）。"""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test_training_api.db'}",
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


@pytest.fixture
def client(db, no_download, login_admin):
    with TestClient(app) as test_client:
        login_admin(test_client)
        yield test_client


def _add_dataset(
    session_factory,
    tmp_path: Path,
    name: str,
    images: list[str],
    labels_by_file: dict[str, list[tuple[str, str]]] | None = None,
) -> int:
    """建 Dataset + 每图真实图片文件 + 可选同名标注 XML，返回 dataset_id。"""
    imgs_dir = tmp_path / f"imgs_{name}"
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


def _assert_eval_structure(body: dict) -> None:
    """断言评估 JSON：4 类别混淆矩阵形状正确 + 4 ROC 曲线含 AUC。"""
    expected_counts = {"舌": 2, "面": 2, "声音": 1}
    assert set(body["confusion"]) == set(expected_counts)
    assert set(body["roc"]) == set(expected_counts)
    assert body["details"]["脉"]["valid_count"] == 0
    for i, name in enumerate(CATEGORY_NAMES):
        if name not in expected_counts:
            continue
        matrix = body["confusion"][name]
        assert sum(sum(row) for row in matrix) == expected_counts[name]
        assert len(matrix) == NUM_CLASSES[i], f"{name} 混淆矩阵行数"
        assert all(len(row) == NUM_CLASSES[i] for row in matrix), f"{name} 混淆矩阵列数"
        assert all(isinstance(v, int) for row in matrix for v in row)
        roc = body["roc"][name]
        assert set(roc) >= {"fpr", "tpr", "auc"}, f"{name} ROC 键"
        if name == "声音":
            assert roc["auc"] is None and roc["reason"]
        else:
            assert len(roc["fpr"]) == len(roc["tpr"]) >= 2
            assert 0.0 <= roc["auc"] <= 1.0


# ---------------------------------------------------------------------------
# 端到端：状态轮询 + WebSocket 推送 + 评估结果
# ---------------------------------------------------------------------------


def test_training_monitor_status_ws_and_eval(
    tmp_path: Path, db, client: TestClient
) -> None:
    """CPU 小任务：轮询 status 至完成；WS 收到快照 + 每 epoch 指标帧 + completed；
    评估端点返回 4 混淆矩阵 + 4 ROC（含 AUC）。"""
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
    ds_id = _add_dataset(db, tmp_path, "monitor", images, labels_by_file)
    job_id = train_service.start_training(_base_config([ds_id], tmp_path / "models"))

    frames: list[dict] = []
    receive_errors: list[Exception] = []

    with client.websocket_connect("/api/training/ws/training") as ws:

        def _receive_forever() -> None:
            try:
                while True:
                    frames.append(ws.receive_json())
            except Exception as exc:  # 连接关闭后 receive 抛错 → 记录并退出
                receive_errors.append(exc)

        receiver = threading.Thread(target=_receive_forever, daemon=True)
        receiver.start()

        # 等待 WS 收到 completed 状态帧（训练线程并行推进）
        deadline = time.time() + 600
        while time.time() < deadline:
            if any(
                f.get("type") == "status" and f.get("status") == "completed"
                for f in frames
            ):
                break
            time.sleep(0.2)
        assert any(
            f.get("type") == "status" and f.get("status") == "completed"
            for f in frames
        ), f"WS 未收到 completed 状态帧，frames={frames!r}"

    # 快照帧：连接即发，包含任务公开视图（REQ-6.12-4 刷新重建）
    snapshots = [f for f in frames if f.get("type") == "snapshot"]
    assert len(snapshots) == 1
    assert any(j["job_id"] == job_id for j in snapshots[0]["jobs"])

    # 指标帧：每 epoch 推送一条增量（REQ-6.12-2 动态曲线无需手动刷新）
    pushed = [
        m for f in frames if f.get("type") == "metrics" for m in f["metrics"]
    ]
    assert [m["epoch"] for m in pushed] == [1, 2]

    # 轮询兜底（REQ-6.12-4）：status 含完整指标缓冲
    r = client.get("/api/training/status", params={"job_id": job_id})
    assert r.status_code == 200
    jobs = r.json()["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["status"] == "completed"
    assert [m["epoch"] for m in jobs[0]["metrics"]] == [1, 2]

    # 评估结果（REQ-6.13）：4 混淆矩阵 + 4 ROC
    r_eval = client.get(f"/api/training/eval/{job_id}")
    assert r_eval.status_code == 200
    _assert_eval_structure(r_eval.json())


def test_eval_empty_val_no_crash(tmp_path: Path, db, client: TestClient) -> None:
    """val_split=0（无验证集）→ 评估返回空 dict，不报错（REQ-6.13-4）。"""
    images = [f"img{i}.png" for i in range(4)]
    labels_by_file = {
        "img0.png": [("舌", "舌苔黄")],
        "img1.png": [("脉", "浮")],
    }
    ds_id = _add_dataset(db, tmp_path, "noval", images, labels_by_file)
    cfg = _base_config([ds_id], tmp_path / "models")
    cfg["val_split"] = 0.0
    job_id = train_service.start_training(cfg)
    job = train_service.train_manager.wait(job_id, timeout=900)
    assert job["status"] == "completed", job.get("error")

    assert all(metric["val_acc"] is None and metric["val_loss"] is None for metric in job["metrics"])
    assert not Path(job["checkpoint_best"]).is_file()
    r = client.get(f"/api/training/eval/{job_id}")
    assert r.status_code == 200
    assert r.json() == {"confusion": {}, "roc": {}}


def test_status_unknown_job_returns_empty(client: TestClient) -> None:
    r = client.get("/api/training/status", params={"job_id": "nonexistent"})
    assert r.status_code == 200
    assert r.json() == {"jobs": []}


def test_eval_unknown_job_404(client: TestClient) -> None:
    assert client.get("/api/training/eval/nonexistent").status_code == 404


def test_eval_running_job_409(tmp_path: Path, db, client: TestClient) -> None:
    """训练未完成且无缓存评估 → 409（评估暂不可用）。"""
    images = [f"img{i}.png" for i in range(4)]
    ds_id = _add_dataset(db, tmp_path, "running", images, {"img0.png": [("舌", "舌苔黄")]})
    cfg = _base_config([ds_id], tmp_path / "models")
    cfg["epochs"] = 50
    job_id = train_service.start_training(cfg)
    try:
        assert client.get(f"/api/training/eval/{job_id}").status_code == 409
    finally:
        train_service.train_manager.stop(job_id)
