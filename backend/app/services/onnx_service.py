"""ONNX 转换与精度对比服务（T19，需求 6.19/6.20）。

- ``convert(model, onnx_path)``：把四头分类器导出为 .onnx。四头模型的
  ``forward`` 返回 ``dict[str, Tensor]``（ONNX 无法导出 dict 输出），故用一个
  薄包装 ``nn.Module`` 把 4 个 logits 按 ``category_names`` 顺序改成 ``tuple``
  后再导出（REQ-6.19-3 产物可被 ONNX Runtime 加载）。
- ``compare(model, onnx_path) -> dict``：同一真实样本或固定合成输入 ``(1, 3, 224, 224)``
  分别过 PyTorch（eval + ``no_grad``）与 ``onnxruntime.InferenceSession``，
  逐头计算 ``max_abs_diff``，< 1e-3 判 PASS（锁定决策 5），返回
  逐头报告最大/平均绝对差、Top-1 一致率与样本明细。

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

import json
import math
import os
import tempfile
import warnings
from pathlib import Path

import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from torch import Tensor

from app.labels_config import CATEGORIES

# 每头最大绝对误差判定阈值（锁定决策 5）
_MAX_ABS_DIFF_TOLERANCE = 1e-3

# 导出输入形状（四头共享同一 ResNet 主干，固定 224×224）
INPUT_SHAPE = (1, 3, 224, 224)

# ONNX opset：≥ 11（BatchNorm15 等 ResNet 算子稳定导出），固定 batch=1
OPSET_VERSION = 13
_OUTPUT_NAMES = [f"head_{i}" for i in range(4)]
_SOURCE_SHA256_KEY = "sizhen.source_sha256"
_INFERENCE_CONTRACT_KEY = "sizhen.inference_contract"

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
    model.eval()
    try:
        with torch.no_grad():
            outs = model(_probe_input())
    except Exception as exc:
        raise ValueError(f"{_UNSUPPORTED_MSG}：forward 执行失败（{exc}）") from exc
    if not isinstance(outs, dict) or set(outs.keys()) != set(names):
        raise ValueError(_UNSUPPORTED_MSG)
    return list(names)


def _probe_input() -> Tensor:
    return torch.rand(INPUT_SHAPE, generator=torch.Generator().manual_seed(0))


def convert(model: nn.Module, onnx_path: str, *, source_sha256: str | None = None) -> str:
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

    wrapper = _TupleOutputWrapper(model, names)
    wrapper.eval()

    dummy_input = _probe_input()
    out_path = Path(onnx_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 同目录暂存，结构和运行校验均通过后再原子发布。
    with tempfile.TemporaryDirectory(dir=out_path.parent, prefix=".onnx-") as temporary:
        candidate = Path(temporary) / "candidate.onnx"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.onnx.export(
                    wrapper,
                    dummy_input,
                    str(candidate),
                    dynamo=False,
                    opset_version=OPSET_VERSION,
                    input_names=["input"],
                    output_names=_OUTPUT_NAMES,
                )
            artifact = onnx.load(str(candidate))
            contract = {
                "version": 1,
                "input": {"name": "input", "shape": INPUT_SHAPE, "layout": "NCHW",
                          "dtype": "float32", "color": "RGB", "divide_pixels_by": 255,
                          "resize": {"width": INPUT_SHAPE[3], "height": INPUT_SHAPE[2],
                                     "interpolation": "bilinear"}},
                "outputs": [{"name": output, "category": category["name"], "labels": category["labels"]}
                            for output, category in zip(_OUTPUT_NAMES, CATEGORIES)],
                "output_type": "logits",
                "postprocess": "softmax_per_head_then_argmax",
            }
            properties = {_INFERENCE_CONTRACT_KEY: json.dumps(contract, ensure_ascii=False)}
            if source_sha256 is not None:
                properties[_SOURCE_SHA256_KEY] = source_sha256
            onnx.helper.set_model_props(artifact, properties)
            onnx.save(artifact, str(candidate))
            onnx.checker.check_model(str(candidate))
            report = compare(model, str(candidate), source_sha256=source_sha256)
            if not report["overall_pass"]:
                raise ValueError("导出前后输出差异超过允许阈值")
            os.replace(candidate, out_path)
        except Exception as exc:
            raise ValueError(f"ONNX 转换或产物校验失败：{exc}") from exc
    return str(out_path)


def compare(
    model: nn.Module, onnx_path: str, *, samples: list[tuple[str, Tensor]] | None = None,
    source_sha256: str | None = None,
) -> dict:
    """使用相同输入比较 FP32 输出。没有真实图像时运行固定的合成冒烟输入。"""
    if not Path(onnx_path).is_file():
        raise ValueError(f"ONNX 文件不存在：{onnx_path}")
    names = _validate_multihead_model(model)
    sample_type = "dataset" if samples is not None else "synthetic"
    inputs = samples if samples is not None else [("synthetic", _probe_input())]
    if not inputs:
        raise ValueError("没有可对比的输入")
    try:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise ValueError(f"无法加载 ONNX 文件：{exc}") from exc
    metadata = session.get_modelmeta().custom_metadata_map
    artifact_source = metadata.get(_SOURCE_SHA256_KEY)
    if source_sha256 and artifact_source and source_sha256 != artifact_source:
        raise ValueError("ONNX 源存档与所选 PyTorch 存档不一致，请重新转换")
    onnx_inputs = session.get_inputs()
    if (len(onnx_inputs) != 1 or onnx_inputs[0].shape != list(INPUT_SHAPE)
            or onnx_inputs[0].type != "tensor(float)"
            or [item.name for item in session.get_outputs()] != _OUTPUT_NAMES):
        raise ValueError("ONNX 输入形状、类型或输出头不符合系统导出契约")

    image_results = []
    for filename, image in inputs:
        with torch.inference_mode():
            pt_outs = model(image)
        try:
            ort_outs = session.run(_OUTPUT_NAMES, {onnx_inputs[0].name: image.numpy()})
        except Exception as exc:
            raise ValueError(f"ONNX 推理失败：{filename}（{exc}）") from exc
        heads = []
        for name, ort_output in zip(names, ort_outs):
            pt_logits = pt_outs[name]
            ort_logits = torch.as_tensor(ort_output)
            if ort_logits.shape != pt_logits.shape:
                raise ValueError(f"ONNX 分类头「{name}」输出形状与源模型不一致")
            difference = (pt_logits - ort_logits).abs()
            max_difference = difference.max().item()
            if not math.isfinite(max_difference):
                raise ValueError(f"分类头「{name}」输出包含非有限数值")
            heads.append({"category": name, "max_abs_diff": max_difference,
                          "mean_abs_diff": difference.mean().item(),
                          "pytorch_top1_index": pt_logits.argmax(dim=1).item(),
                          "onnx_top1_index": ort_logits.argmax(dim=1).item()})
        image_results.append({"filename": filename, "heads": heads})

    results = []
    for index, name in enumerate(names):
        heads = [result["heads"][index] for result in image_results]
        max_diff = max(head["max_abs_diff"] for head in heads)
        results.append({"category": name, "max_abs_diff": max_diff,
                        "mean_abs_diff": sum(head["mean_abs_diff"] for head in heads) / len(heads),
                        "top1_agreement": sum(head["pytorch_top1_index"] == head["onnx_top1_index"] for head in heads) / len(heads),
                        "pass": max_diff < _MAX_ABS_DIFF_TOLERANCE})
    return {"results": results, "overall_pass": all(r["pass"] for r in results),
            "sample_type": sample_type, "sample_count": len(inputs),
            "sample_filenames": [filename for filename, _ in inputs] if samples is not None else [],
            "image_results": image_results, "source_sha256": artifact_source}
