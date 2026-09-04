"""ONNX 转换与精度对比服务（T19，需求 6.19/6.20）。

- ``convert(model, onnx_path)``：把四头分类器导出为 .onnx。四头模型的
  ``forward`` 返回 ``dict[str, Tensor]``（ONNX 无法导出 dict 输出），故用一个
  薄包装 ``nn.Module`` 把 4 个 logits 按 ``category_names`` 顺序改成 ``tuple``
  后再导出（REQ-6.19-3 产物可被 ONNX Runtime 加载）。
- ``compare(model, onnx_path) -> dict``：同一 dummy 输入 ``(1, 3, 224, 224)``
  分别过 PyTorch（eval + ``no_grad``）与 ``onnxruntime.InferenceSession``，
  逐头计算 ``max_abs_diff``，< 1e-3 判 PASS（锁定决策 5），返回
  ``{"results": [{"category","max_abs_diff","pass"} ×4], "overall_pass": bool}``。

范围保证（锁定决策 8）：仅支持本系统四头 ResNet18/50 产物（``forward`` 返回
dict、且带 ``category_names`` 属性）；自定义/第三方模型 → ``ValueError``
「该模型不支持一键转换」，不猜测结构、不静默降级。

精度说明：训练采用 FP16 混合精度但**保留 FP32 master weights**，checkpoint
载入后为 FP32 权重，ONNX 导出亦为 FP32，因此本对比是 FP32-vs-FP32，阈值
1e-3 用于容纳导出图优化（算子融合/重排）引入的浮点舍入差异。

实现说明：``torch.onnx.export`` 显式 ``dynamo=False`` 使用 legacy exporter
（新 dynamo 路径依赖 ``onnxscript``，本环境不随 torch 附带且不允许新增依赖）。
"""
from __future__ import annotations

import warnings
from pathlib import Path

import onnxruntime as ort
import torch
import torch.nn as nn
from torch import Tensor

# 每头最大绝对误差判定阈值（锁定决策 5）
_MAX_ABS_DIFF_TOLERANCE = 1e-3

# 导出输入形状（四头共享同一 ResNet 主干，固定 224×224）
_INPUT_SHAPE = (1, 3, 224, 224)

# ONNX opset：≥ 11（BatchNorm15 等 ResNet 算子稳定导出），固定 batch=1
_OPSET_VERSION = 13

# 自定义/第三方模型统一报错文案（REQ-6.19-2，锁定决策 8）
_UNSUPPORTED_MSG = "该模型不支持一键转换"


class _TupleOutputWrapper(nn.Module):
    """把 dict 输出的四头模型包装为 tuple 输出（按 category_names 顺序）。

    ONNX 只支持 list/tuple/tensor 输出，dict 输出无法直接导出；本包装
    ``forward(x)`` 先调用原模型的 dict 输出，再按 ``category_names`` 顺序
    重排为 ``tuple[Tensor ×4]``，键名顺序与 ``factory._MultiHeadResNet`` 一致。
    """

    def __init__(self, model: nn.Module, category_names: list[str]) -> None:
        super().__init__()
        self.model = model
        self.category_names = list(category_names)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        outs = self.model(x)
        return tuple(outs[name] for name in self.category_names)


def _validate_multihead_model(model: nn.Module) -> list[str]:
    """校验模型为本系统四头分类器，返回 category_names；否则 ValueError。

    判定标准：带 ``category_names`` 属性（长度 4）且 ``forward`` 返回
    ``dict[str, Tensor]``（键集合 == category_names）。自定义模型（无该属性
    或非 dict 输出）按锁定决策 8 显式报「该模型不支持一键转换」。
    """
    names = getattr(model, "category_names", None)
    if not isinstance(names, (list, tuple)) or len(names) != 4 or not all(
        isinstance(n, str) for n in names
    ):
        raise ValueError(_UNSUPPORTED_MSG)
    try:
        with torch.no_grad():
            outs = model(torch.randn(*_INPUT_SHAPE))
    except Exception as exc:
        raise ValueError(f"{_UNSUPPORTED_MSG}：forward 执行失败（{exc}）") from exc
    if not isinstance(outs, dict) or set(outs.keys()) != set(names):
        raise ValueError(_UNSUPPORTED_MSG)
    return list(names)


def convert(model: nn.Module, onnx_path: str) -> str:
    """导出四头分类器为 ONNX（REQ-6.19-1/6.19-3）。

    Args:
        model: 本系统四头分类器（``inference_service.load_model_from_dir``
            或 ``factory.build_model`` 产物），eval 模式。
        onnx_path: 输出 .onnx 文件路径。

    Returns:
        onnx_path（导出成功）。

    Raises:
        ValueError: 模型不支持一键转换（非本系统四头结构）。
    """
    names = _validate_multihead_model(model)

    model.eval()
    wrapper = _TupleOutputWrapper(model, names)
    wrapper.eval()

    dummy_input = torch.randn(*_INPUT_SHAPE)
    out_path = Path(onnx_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # legacy exporter（dynamo=False）：dynamo 路径依赖未安装的 onnxscript
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(out_path),
            dynamo=False,
            opset_version=_OPSET_VERSION,
            input_names=["input"],
            output_names=[f"head_{i}" for i in range(4)],
        )
    return str(out_path)


def compare(model: nn.Module, onnx_path: str) -> dict:
    """PyTorch 与 ONNX 在同一输入下逐头精度对比（REQ-6.20）。

    同一 dummy 输入 ``(1, 3, 224, 224)`` 分别过 PyTorch（eval + no_grad）与
    onnxruntime；每头计算 ``max_abs_diff = max|pt - onnx|``，< 1e-3 判 PASS
    （锁定决策 5）。FP32-vs-FP32 对比（FP16 训练保留 FP32 master weights）。

    Args:
        model: 本系统四头分类器（eval 模式）。
        onnx_path: 已导出的 .onnx 文件路径。

    Returns:
        ``{"results": [{"category", "max_abs_diff", "pass"} ×4],
        "overall_pass": bool}``。

    Raises:
        ValueError: ONNX 文件不存在 / 模型不支持一键转换 / ONNX 输出头数不为 4。
    """
    if not Path(onnx_path).is_file():
        raise ValueError(f"ONNX 文件不存在：{onnx_path}")
    names = _validate_multihead_model(model)

    model.eval()
    dummy_input = torch.randn(*_INPUT_SHAPE)

    # PyTorch 输出（dict → category_names 顺序）
    with torch.no_grad():
        pt_outs = model(dummy_input)
    pt_logits = [pt_outs[name] for name in names]

    # ONNX Runtime 输出（tuple，顺序 = 导出时 head_0..head_3 == category_names）
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_outs = session.run(None, {session.get_inputs()[0].name: dummy_input.numpy()})
    if len(ort_outs) != 4:
        raise ValueError(
            f"ONNX 输出头数为 {len(ort_outs)}（应为 4），该文件可能非本系统导出产物"
        )

    results = []
    for name, pt_logit, ort_logit in zip(names, pt_logits, ort_outs):
        max_abs_diff = float(
            (pt_logit - torch.as_tensor(ort_logit)).abs().max().item()
        )
        results.append(
            {
                "category": name,
                "max_abs_diff": max_abs_diff,
                "pass": max_abs_diff < _MAX_ABS_DIFF_TOLERANCE,
            }
        )
    return {"results": results, "overall_pass": all(r["pass"] for r in results)}
