"""模型构建：ResNet18/50 骨干 + 四个并行分类头，以及自定义模型加载。

四诊分类模型在骨干网络之后接 4 个相互独立的线性头，分别对应
舌 / 面 / 脉 / 声音 四个类别，每个头的输出维度等于该类别子标签数。
骨干部分使用 torchvision 的预训练权重做初始化。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, List, Sequence

import torch
from torch import nn
from torchvision import models as tv_models

# 内置骨干对应的特征维度（去掉最后的 fc 层之后）
_BACKBONE_FEATURES = {
    "resnet18": 512,
    "resnet50": 2048,
}


def _strip_fc(backbone: nn.Module) -> nn.Module:
    """移除 torchvision resnet 的最后一层全连接，保留到全局池化为止。"""
    return nn.Sequential(*list(backbone.children())[:-1])


class FourHeadClassifier(nn.Module):
    """共享骨干 + 4 个独立分类头的多任务分类器。"""

    def __init__(
        self,
        backbone_name: str,
        head_sizes: Sequence[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if backbone_name == "resnet18" \
            else tv_models.ResNet50_Weights.IMAGENET1K_V1
        try:
            base = (
                tv_models.resnet18(weights=weights)
                if backbone_name == "resnet18"
                else tv_models.resnet50(weights=weights)
            )
        except Exception:  # noqa: BLE001 - 预训练权重下载失败时退化为随机初始化
            base = (
                tv_models.resnet18(weights=None)
                if backbone_name == "resnet18"
                else tv_models.resnet50(weights=None)
            )
        self.backbone = _strip_fc(base)
        self.feature_dim = _BACKBONE_FEATURES[backbone_name]
        self.backbone_name = backbone_name

        self.heads = nn.ModuleList()
        for size in head_sizes:
            head = nn.Sequential()
            if dropout > 0:
                head.append(nn.Dropout(p=dropout))
            head.append(nn.Linear(self.feature_dim, size))
            self.heads.append(head)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = self.backbone(x).flatten(1)
        return [head(features) for head in self.heads]


def build_backbone(
    architecture: str,
    head_sizes: Sequence[int],
    dropout: float = 0.0,
) -> nn.Module:
    """按架构名构建四头分类器，仅支持 resnet18 / resnet50。"""
    if architecture not in _BACKBONE_FEATURES:
        raise ValueError(f"不支持的骨干网络：{architecture}（仅支持 resnet18 / resnet50）")
    return FourHeadClassifier(architecture, head_sizes, dropout)


def load_custom_model(path: str | Path) -> nn.Module:
    """从单个 .py 文件动态加载用户自定义模型。

    约定：该 .py 必须暴露一个名为 ``build_model`` 的函数，签名
    ``build_model(num_classes: list[int]) -> nn.Module``，返回的模块
    forward 输出 4 个 logit 张量（顺序与标签体系一致）。
    """
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"自定义模型文件不存在：{source}")
    if source.suffix != ".py":
        raise ValueError("自定义模型必须是 .py 文件")

    module_name = f"_custom_model_{abs(hash(source.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法加载自定义模型文件：{source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - 用户代码异常需透传
        raise ValueError(f"自定义模型执行失败：{exc}") from exc

    builder = getattr(module, "build_model", None)
    if not callable(builder):
        raise ValueError("自定义模型必须提供 build_model(num_classes) 函数")

    def _wrap(sizes: Sequence[int]) -> nn.Module:
        model = builder(list(sizes))
        if not isinstance(model, nn.Module):
            raise ValueError("build_model 必须返回 torch.nn.Module 实例")
        return model

    return _wrap


def instantiate(
    architecture: str,
    head_sizes: Sequence[int],
    dropout: float = 0.0,
    custom_path: str | None = None,
) -> nn.Module:
    """统一入口：内置骨干走 build_backbone，custom 走动态加载。"""
    if architecture == "custom":
        if not custom_path:
            raise ValueError("自定义模型缺少文件路径")
        builder = load_custom_model(custom_path)
        return builder(head_sizes)
    return build_backbone(architecture, head_sizes, dropout)
