"""ONNX 转换/精度对比/下载测试（T19，需求 6.19/6.20/6.21）。

覆盖：ResNet18 导出产物存在且可被 ONNX Runtime 加载（REQ-6.19-3）；
4 头逐头 max abs diff < 1e-3 全部 PASS（REQ-6.20-2，锁定决策 5）；
第三方 .pth 转换 → 400「该模型不支持一键转换」（REQ-6.19-2，锁定决策 8）；
下载内容与原产物一致、越界文件名被拒（REQ-6.21-1，NF-7.4）。
"""
from __future__ import annotations

import collections
from pathlib import Path

import onnx
import onnxruntime as ort
import pytest
import torch
import torchvision.models as tvm
from fastapi.testclient import TestClient

from app import config
from app.labels_config import CATEGORIES
from app.main import app
from app.models.factory import build_model
from app.services import onnx_service

CATEGORY_NAMES = [c["name"] for c in CATEGORIES]
NUM_CLASSES_4 = [len(c["labels"]) for c in CATEGORIES]


# ---------------------------------------------------------------------------
# 工具 / fixture
# ---------------------------------------------------------------------------


def _build_resnet18(monkeypatch) -> torch.nn.Module:
    """构建 resnet18 四头模型（weights=None，避免测试期下载大权重）。"""
    original = tvm.resnet18
    monkeypatch.setattr(tvm, "resnet18", lambda weights=None: original(weights=None))
    return build_model("resnet18", NUM_CLASSES_4, dropout=0.0)


def _save_checkpoint(path: Path, model: torch.nn.Module) -> Path:
    """按 T13 checkpoint 契约落盘系统训练产物。"""
    torch.save(
        {
            "model_state": model.state_dict(),
            "architecture": "resnet18",
            "num_classes": NUM_CLASSES_4,
            "category_names": CATEGORY_NAMES,
            "dropout": 0.0,
        },
        path,
    )
    return path


@pytest.fixture
def dirs(tmp_path: Path, monkeypatch):
    """把 MODEL_DIR/EXPORT_DIR 指到临时目录（路由与 load_model_from_dir 共用）。"""
    model_dir = tmp_path / "models"
    export_dir = tmp_path / "exports"
    model_dir.mkdir()
    export_dir.mkdir()
    monkeypatch.setattr(config, "MODEL_DIR", model_dir)
    monkeypatch.setattr(config, "EXPORT_DIR", export_dir)
    return model_dir, export_dir


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------


def test_convert_creates_loadable_onnx(tmp_path: Path, monkeypatch) -> None:
    """factory 构建的 ResNet18 导出成功：文件存在、ONNX 校验通过、ORT 可加载。"""
    model = _build_resnet18(monkeypatch)
    out = tmp_path / "model.onnx"

    onnx_service.convert(model, str(out))

    assert out.is_file() and out.stat().st_size > 0
    onnx.checker.check_model(onnx.load(str(out)))
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].name == "input"
    assert [o.name for o in session.get_outputs()] == [f"head_{i}" for i in range(4)]


def test_convert_unsupported_model_raises(tmp_path: Path, monkeypatch) -> None:
    """非本系统四头模型（无 category_names）→ ValueError「该模型不支持一键转换」。"""
    original = tvm.resnet18
    monkeypatch.setattr(tvm, "resnet18", lambda weights=None: original(weights=None))
    plain = tvm.resnet18(weights=None)
    with pytest.raises(ValueError, match="该模型不支持一键转换"):
        onnx_service.convert(plain, str(tmp_path / "x.onnx"))


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def test_compare_all_heads_pass(tmp_path: Path, monkeypatch) -> None:
    """同一输入下 PyTorch vs ONNX：4 头全部 PASS（diff < 1e-3），overall_pass=True。"""
    model = _build_resnet18(monkeypatch)
    out = tmp_path / "m.onnx"
    onnx_service.convert(model, str(out))

    report = onnx_service.compare(model, str(out))

    assert [r["category"] for r in report["results"]] == CATEGORY_NAMES
    for r in report["results"]:
        assert r["max_abs_diff"] < 1e-3
        assert r["pass"] is True
    assert report["overall_pass"] is True


def test_compare_missing_onnx_raises(tmp_path: Path, monkeypatch) -> None:
    model = _build_resnet18(monkeypatch)
    with pytest.raises(ValueError, match="ONNX 文件不存在"):
        onnx_service.compare(model, str(tmp_path / "nope.onnx"))


# ---------------------------------------------------------------------------
# API：第三方 .pth → 400；下载往返
# ---------------------------------------------------------------------------


def test_convert_foreign_pth_rejected(dirs, monkeypatch) -> None:
    """第三方权重（无 architecture/num_classes 键）POST /convert → 400。"""
    model_dir, _ = dirs
    torch.save(
        {"state_dict": collections.OrderedDict(), "classes": 1000},
        model_dir / "foreign.pth",
    )
    with TestClient(app) as client:
        resp = client.post("/api/onnx/convert", json={"model_path": "foreign.pth"})
    assert resp.status_code == 400
    assert "该模型不支持一键转换" in resp.json()["detail"]


def test_download_roundtrip_and_traversal_guard(dirs, monkeypatch) -> None:
    """转换 → 下载内容与产物逐字节一致；越界文件名 → 400（REQ-6.21-1 / NF-7.4）。"""
    model_dir, export_dir = dirs
    model = _build_resnet18(monkeypatch)
    _save_checkpoint(model_dir / "best.pth", model)

    with TestClient(app) as client:
        conv = client.post("/api/onnx/convert", json={"model_path": "best.pth"})
        assert conv.status_code == 200
        onnx_name = conv.json()["onnx_name"]
        assert onnx_name == "best.onnx"
        artifact = export_dir / onnx_name
        assert artifact.is_file()

        dl = client.get(f"/api/onnx/{onnx_name}/download")
        assert dl.status_code == 200
        assert dl.content == artifact.read_bytes()

        # 到达 handler 的非法名（backslash 穿越、非 .onnx 后缀）→ 400 守卫
        for bad in ("..%5Csecret.onnx", "not_onnx.pth"):
            resp = client.get(f"/api/onnx/{bad}/download")
            assert resp.status_code == 400, bad

        # 含斜杠的路径在路由匹配阶段即被拒（{onnx_name} 不含 /），请求不达 handler → 404
        for unreachable in ("a/b.onnx", "..%2Fsecret.onnx"):
            resp = client.get(f"/api/onnx/{unreachable}/download")
            assert resp.status_code == 404, unreachable

        missing = client.get("/api/onnx/missing.onnx/download")
        assert missing.status_code == 404
