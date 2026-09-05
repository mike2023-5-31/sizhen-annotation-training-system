"""模型工厂（T12）：ResNet18/50 四头分类器 + 自定义模型动态加载。

- `build_model(backbone, num_classes, dropout)`：torchvision ResNet18/ResNet50
  （ImageNet 预训练），把 fc 替换为 `nn.Identity()`，再挂 4 个并行分类头
  `nn.ModuleDict`（键 = CATEGORIES 顺序的类别名 舌/面/脉/声音），每头 =
  `Sequential(Dropout(dropout), Linear(in_features -> n_cat))`。
- `load_custom_model(py_path, num_classes)`：importlib 动态加载用户 .py，
  要求暴露 `build_model(num_classes)` 或 `Model(num_classes)`，做签名/类型/
  forward 校验。
- `NUM_CLASSES`：由 `app.labels_config.num_classes_per_head()` 派生的各头类别数。

⚠️ 安全声明（锁定决策 8 相关）：`load_custom_model` 会**直接执行**用户提供的
.py 文件（无沙箱、无隔离）。加载恶意文件等同于在本机运行任意代码。仅应在
可信来源下使用（T13/T17 只加载管理员放入模型目录的脚本）；不可信代码必须
在进程级隔离环境（容器/子进程）中执行后再接入。
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torchvision.models as tvm
from torch import Tensor

from app.labels_config import CATEGORIES, num_classes_per_head

# 类别名（按 labels.yaml CATEGORIES 顺序），即 4 个分类头的键名。
CATEGORY_NAMES: List[str] = [c["name"] for c in CATEGORIES]

# 每个分类头的类别数（= 对应 category 的子标签数）——训练/推理/checkpoint 元数据共用。
NUM_CLASSES: List[int] = num_classes_per_head()

# 各 backbone 主干输出的特征维数（ResNet 原始 fc 的 in_features）。
_BACKBONE_IN_FEATURES: Dict[str, int] = {"resnet18": 512, "resnet50": 2048}

_MODULE_COUNTER = itertools.count()


class _MultiHeadResNet(nn.Module):
    """ResNet 主干（fc=Identity）+ 4 个并行分类头的包装模块。

    属性（供 T13/T17/T19 及 checkpoint 元数据消费）：
    - backbone: torchvision ResNet，fc 已替换为 nn.Identity()
    - heads: nn.ModuleDict，键 = category_names，值 = Sequential(Dropout, Linear)
    - category_names: List[str]，分类头键名顺序（与 CATEGORIES 一致）
    - num_classes: List[int]，各头输出类别数
    """

    def __init__(
        self,
        backbone: nn.Module,
        in_features: int,
        num_classes: List[int],
        dropout: float,
        category_names: List[str],
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.category_names = list(category_names)
        self.num_classes = list(num_classes)
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, n_cat))
                for name, n_cat in zip(category_names, num_classes)
            }
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """返回 dict：类别名 → logits `(B, n_cat)`，键顺序与 category_names 一致。"""
        feats = self.backbone(x)
        return {name: self.heads[name](feats) for name in self.category_names}


def _validate_num_classes(num_classes: Any) -> List[int]:
    """校验 num_classes 为与 CATEGORIES 等长的正整数列表，否则抛 ValueError。"""
    try:
        nc = [int(n) for n in num_classes]
    except (TypeError, ValueError):
        nc = []
    if len(nc) != len(CATEGORY_NAMES) or any(n <= 0 for n in nc):
        raise ValueError(
            f"num_classes 必须为 {len(CATEGORY_NAMES)} 个正整数"
            f"（对应舌/面/脉/声音四类），得到 {num_classes!r}"
        ) from None
    return nc


def build_model(
    backbone: str,
    num_classes: List[int],
    dropout: float = 0.0,
) -> nn.Module:
    """构建 ResNet18/ResNet50 四头分类器（T12）。

    Args:
        backbone: ``'resnet18'`` 或 ``'resnet50'``。
        num_classes: 长度 4 的类别数列表，顺序与 CATEGORIES（舌/面/脉/声音）一致。
        dropout: 每个分类头 Linear 前的 Dropout 概率（锁定决策 6.9），范围 [0, 1)。

    Returns:
        nn.Module：属性 ``.backbone``（fc 已替换为 Identity）/ ``.heads``（ModuleDict）/
        ``.category_names`` / ``.num_classes``；``forward(x)`` 返回
        ``dict[str, Tensor]``（类别名 → logits ``(B, n_cat)``）。

    Notes:
        预训练权重用 ``IMAGENET1K_V1``；若权重下载失败（无网络）则退回
        ``weights=None`` 随机初始化，模型结构不受影响（冒烟测试仍可跑通）。
    """
    if backbone not in _BACKBONE_IN_FEATURES:
        raise ValueError(f"不支持的 backbone：{backbone!r}（仅支持 resnet18 / resnet50）")
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout 必须在 [0, 1) 范围内，得到 {dropout!r}")
    nc = _validate_num_classes(num_classes)

    if backbone == "resnet18":
        weights_cls = tvm.ResNet18_Weights
    else:
        weights_cls = tvm.ResNet50_Weights
    builder = getattr(tvm, backbone)
    try:
        base = builder(weights=weights_cls.IMAGENET1K_V1)
    except Exception:  # 无网络/权重下载失败 → 随机初始化（结构不变）
        base = builder(weights=None)

    in_features = base.fc.in_features
    base.fc = nn.Identity()
    return _MultiHeadResNet(base, in_features, nc, dropout, CATEGORY_NAMES)


def load_custom_model(py_path: str, num_classes: List[int]) -> nn.Module:
    """动态加载用户自定义模型 .py（T12）。

    ⚠️ 安全声明：本函数会**直接执行** `py_path` 指定的 Python 文件（无沙箱）。
    加载恶意文件等同于在本机运行任意代码。仅应加载管理员明确放入模型目录的
    可信脚本；不可信来源必须在进程级隔离环境（容器/子进程）中执行。

    约定：目标 .py 必须暴露以下二者之一：
    - 函数 ``build_model(num_classes) -> nn.Module``；或
    - 类 ``Model``，构造签名 ``Model(num_classes)``。
    两者并存时优先使用 ``build_model``。加载后校验：实例为 ``nn.Module``、
    且 ``forward`` 能接受 4D 张量 ``(B, C, H, W)``（用随机输入做一次无梯度冒烟）。

    Args:
        py_path: 自定义模型 .py 文件的路径。
        num_classes: 长度 4 的类别数列表（顺序与 CATEGORIES 一致），原样传给
            用户入口。

    Returns:
        校验通过的 nn.Module 实例。

    Raises:
        ValueError: 文件不存在 / 非 .py / 缺少 build_model 与 Model /
            实例化失败 / 结果不是 nn.Module / forward 不接受 4D 张量。
    """
    path = Path(py_path)
    if path.suffix.lower() != ".py":
        raise ValueError(f"自定义模型文件必须是 .py 文件：{path}")
    if not path.is_file():
        raise ValueError(f"自定义模型文件不存在：{path}")

    nc = _validate_num_classes(num_classes)

    module_name = f"_sizhen_custom_model_{path.stem}_{next(_MODULE_COUNTER)}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法解析自定义模型文件：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise ValueError(f"执行自定义模型文件失败：{path}（{exc}）") from exc

        builder = getattr(module, "build_model", None)
        model_cls = getattr(module, "Model", None)
        if callable(builder):
            try:
                model = builder(nc)
            except Exception as exc:
                raise ValueError(f"调用 build_model({nc}) 失败：{exc}") from exc
        elif isinstance(model_cls, type):
            try:
                model = model_cls(nc)
            except Exception as exc:
                raise ValueError(f"实例化 Model({nc}) 失败：{exc}") from exc
        else:
            raise ValueError(
                f"自定义模型文件必须暴露 build_model(num_classes) 函数或 "
                f"Model(num_classes) 类：{path}"
            )

        if not isinstance(model, nn.Module):
            raise ValueError(
                f"自定义模型必须是 nn.Module 实例，得到 {type(model).__name__}：{path}"
            )

        # 冒烟校验：forward 必须接受 4D 张量 (B, C, H, W)
        try:
            with torch.no_grad():
                model(torch.randn(1, 3, 224, 224))
        except Exception as exc:
            raise ValueError(
                f"自定义模型 forward 无法处理 4D 输入 (B, C, H, W)：{exc}"
            ) from exc
        return model
    finally:
        sys.modules.pop(module_name, None)


class _CustomOutputWrapper(nn.Module):
    """自定义模型输出契约包装（需求 6.7）：统一为 4 个分类头 logits 的 dict。

    构造时用 ``(1, 3, 224, 224)`` 随机输入做一次无梯度前向校验输出契约：
    - ``dict``：必须包含全部 4 个 category 键（多出键忽略，按 CATEGORIES
      顺序输出）；
    - ``tuple``/``list``：必须恰好 4 个张量，按 CATEGORIES 顺序映射为
      4 个分类头；
    - 其它类型 → ``ValueError``「自定义模型输出需为 4 个分类头 logits」。

    属性：``category_names`` / ``num_classes``（供训练循环、checkpoint 元数据
    与推理 taxonomy 校验消费）。``forward(x)`` 每次前向同样做归一化。
    """

    def __init__(
        self, model: nn.Module, category_names: List[str], num_classes: List[int]
    ) -> None:
        super().__init__()
        self.model = model
        self.category_names = list(category_names)
        self.num_classes = list(num_classes)
        # 构造期契约校验（无梯度冒烟前向；输出非法 → 立即 ValueError）
        with torch.no_grad():
            self._normalize(self.model(torch.randn(1, 3, 224, 224)))

    def _normalize(self, outs: Any) -> Dict[str, Tensor]:
        names = self.category_names
        if isinstance(outs, dict):
            if any(name not in outs for name in names):
                raise ValueError("自定义模型输出需为 4 个分类头 logits")
            return {name: outs[name] for name in names}
        if isinstance(outs, (tuple, list)):
            if len(outs) != len(names):
                raise ValueError("自定义模型输出需为 4 个分类头 logits")
            return {name: out for name, out in zip(names, outs)}
        raise ValueError("自定义模型输出需为 4 个分类头 logits")

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        return self._normalize(self.model(x))


def wrap_custom_model(model: nn.Module, num_classes: List[int]) -> nn.Module:
    """校验并包装自定义模型输出为统一的 4 分类头 logits dict（需求 6.7）。

    供训练引擎（T13）与推理服务（T17）加载自定义模型后调用：包装器构造期
    用无梯度冒烟前向校验输出契约，之后每次 ``forward`` 都归一化为
    ``dict[str, Tensor]``（键序 = CATEGORIES）。

    Args:
        model: ``load_custom_model`` 返回的 nn.Module。
        num_classes: 长度 4 的类别数列表（顺序与 CATEGORIES 一致）。

    Returns:
        ``_CustomOutputWrapper`` 实例（带 category_names/num_classes 属性）。

    Raises:
        ValueError: 输出既不是含 4 分类头的 dict，也不是恰好 4 个张量的
            tuple/list。
    """
    nc = _validate_num_classes(num_classes)
    return _CustomOutputWrapper(model, CATEGORY_NAMES, nc)
