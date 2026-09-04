"""模型工厂单元测试（T12）：ResNet18/50 四头结构 + 自定义模型动态加载/校验。"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torchvision.models as tvm

from app.labels_config import CATEGORIES
from app.models.factory import NUM_CLASSES, build_model, load_custom_model

CATEGORY_NAMES = [c["name"] for c in CATEGORIES]
NUM_CLASSES_4 = [6, 4, 5, 4]


def test_num_classes_reexport_matches_labels_config() -> None:
    """NUM_CLASSES 与 labels.yaml 每类子标签数一致。"""
    assert NUM_CLASSES == [len(c["labels"]) for c in CATEGORIES]


def test_build_model_resnet18_forward_dict(monkeypatch) -> None:
    """(a) resnet18 forward 返回 4 键 dict（键序 = CATEGORIES），各头 (B, n_cat)。"""
    # 避免测试期大体积权重下载：直接以 weights=None 构造（下载路径由冒烟验证覆盖）。
    original = tvm.resnet18
    monkeypatch.setattr(tvm, "resnet18", lambda weights=None: original(weights=None))

    model = build_model("resnet18", NUM_CLASSES_4, dropout=0.5)
    assert isinstance(model, nn.Module)
    assert model.category_names == CATEGORY_NAMES
    assert model.num_classes == NUM_CLASSES_4
    assert set(model.heads.keys()) == set(CATEGORY_NAMES)
    assert isinstance(model.backbone.fc, nn.Identity)

    outs = model(torch.randn(2, 3, 224, 224))
    assert isinstance(outs, dict)
    assert list(outs.keys()) == CATEGORY_NAMES
    for name, n_cat in zip(CATEGORY_NAMES, NUM_CLASSES_4):
        assert tuple(outs[name].shape) == (2, n_cat)


def test_build_model_resnet50_weights_none(monkeypatch) -> None:
    """(b) resnet50 同样可构建；用 weights=None 避免第二次大体积下载。"""
    original = tvm.resnet50
    monkeypatch.setattr(tvm, "resnet50", lambda weights=None: original(weights=None))

    model = build_model("resnet50", NUM_CLASSES_4)
    assert model.category_names == CATEGORY_NAMES
    assert isinstance(model.backbone.fc, nn.Identity)
    outs = model(torch.randn(1, 3, 224, 224))
    assert set(outs.keys()) == set(CATEGORY_NAMES)
    assert tuple(outs["舌"].shape) == (1, 6)


def test_build_model_invalid_backbone_raises() -> None:
    with pytest.raises(ValueError, match="backbone"):
        build_model("resnet101", NUM_CLASSES_4)


def test_build_model_invalid_num_classes_raises() -> None:
    with pytest.raises(ValueError, match="num_classes"):
        build_model("resnet18", [6, 4, 5])  # 长度不足 4


_TINY_BUILD_MODEL_SRC = textwrap.dedent(
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


def test_load_custom_model_build_model(tmp_path: Path) -> None:
    """(c) 临时 .py 暴露 build_model(num_classes) → 实例化成功且 forward 可用。"""
    p = tmp_path / "tiny_model.py"
    p.write_text(_TINY_BUILD_MODEL_SRC, encoding="utf-8")
    model = load_custom_model(str(p), NUM_CLASSES_4)
    assert isinstance(model, nn.Module)
    outs = model(torch.randn(2, 3, 224, 224))
    assert list(outs.keys()) == CATEGORY_NAMES
    for name, n_cat in zip(CATEGORY_NAMES, NUM_CLASSES_4):
        assert tuple(outs[name].shape) == (2, n_cat)


def test_load_custom_model_model_class(tmp_path: Path) -> None:
    """临时 .py 暴露 Model(num_classes) 类 → 同样可加载。"""
    p = tmp_path / "model_class.py"
    p.write_text(
        textwrap.dedent(
            """
            import torch.nn as nn


            class Model(nn.Module):
                def __init__(self, num_classes):
                    super().__init__()
                    self.pool = nn.AdaptiveAvgPool2d(1)
                    self.fc = nn.Linear(3, sum(num_classes))

                def forward(self, x):
                    return self.fc(self.pool(x).flatten(1))
            """
        ),
        encoding="utf-8",
    )
    model = load_custom_model(str(p), NUM_CLASSES_4)
    assert isinstance(model, nn.Module)
    out = model(torch.randn(1, 3, 224, 224))
    assert tuple(out.shape) == (1, sum(NUM_CLASSES_4))


def test_load_custom_model_garbage_raises(tmp_path: Path) -> None:
    """(d) 既无 build_model 也无 Model 的 .py → ValueError。"""
    p = tmp_path / "garbage.py"
    p.write_text("x = 42  # 既无 build_model 函数也无 Model 类\n", encoding="utf-8")
    with pytest.raises(ValueError, match="build_model|Model"):
        load_custom_model(str(p), NUM_CLASSES_4)


def test_load_custom_model_not_py_raises(tmp_path: Path) -> None:
    p = tmp_path / "model.txt"
    p.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match=r"\.py"):
        load_custom_model(str(p), NUM_CLASSES_4)


def test_load_custom_model_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="不存在"):
        load_custom_model(str(tmp_path / "nope.py"), NUM_CLASSES_4)


def test_load_custom_model_non_module_raises(tmp_path: Path) -> None:
    """build_model 返回非 nn.Module → ValueError。"""
    p = tmp_path / "not_a_module.py"
    p.write_text(
        "def build_model(num_classes):\n    return 'not a module'\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="nn.Module"):
        load_custom_model(str(p), NUM_CLASSES_4)


def test_load_custom_model_bad_forward_raises(tmp_path: Path) -> None:
    """forward 不接受 4D 张量（无 forward 实现）→ ValueError。"""
    p = tmp_path / "no_forward.py"
    p.write_text(
        "import torch.nn as nn\n\n"
        "def build_model(num_classes):\n"
        "    return nn.Module()\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="forward"):
        load_custom_model(str(p), NUM_CLASSES_4)
