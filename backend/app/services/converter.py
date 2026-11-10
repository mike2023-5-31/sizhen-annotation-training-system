"""PyTorch → ONNX 转换与精度对比（6.19 / 6.20）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

INPUT_SHAPE = (1, 3, 224, 224)
OPSET_VERSION = 13
_OUTPUT_NAMES = ["head_tongue", "head_face", "head_pulse", "head_voice"]
_DIFF_THRESHOLD = 1e-3


class _ExportWrapper(nn.Module):
    """把模型的 list 输出固定为 4 个具名输出，供 ONNX 导出。"""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        outputs = self.model(x)
        return outputs[0], outputs[1], outputs[2], outputs[3]


def convert(model: nn.Module, out_path: str | Path) -> None:
    """把四头分类模型导出为 ONNX 文件。"""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    wrapper = _ExportWrapper(model).eval()
    dummy = torch.randn(*INPUT_SHAPE)
    torch.onnx.export(
        wrapper,
        dummy,
        str(target),
        opset_version=OPSET_VERSION,
        input_names=["image"],
        output_names=list(_OUTPUT_NAMES),
        dynamic_axes={"image": {0: "batch"}},
    )


def _pytorch_outputs(model: nn.Module, tensor: torch.Tensor) -> list[np.ndarray]:
    model.eval()
    with torch.no_grad():
        outputs = model(tensor.unsqueeze(0))
    return [o.cpu().numpy() for o in outputs]


def _onnx_outputs(onnx_path: str | Path, tensor: torch.Tensor) -> list[np.ndarray]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    feed = {input_name: tensor.unsqueeze(0).numpy().astype(np.float32)}
    return session.run(None, feed)


def compare(
    model: nn.Module,
    onnx_path: str | Path,
    samples: list[tuple[str, torch.Tensor]] | None = None,
) -> dict[str, Any]:
    """逐头对比 PyTorch 与 ONNX 输出，返回各头最大绝对误差与通过状态。"""
    tensors = [t for _, t in samples] if samples else [torch.randn(3, 224, 224)]
    filenames = [name for name, _ in samples] if samples else [f"random_{i}" for i in range(len(tensors))]

    heads: list[dict[str, Any]] = []
    overall_pass = True
    for head in range(4):
        diffs: list[float] = []
        for tensor in tensors:
            pt = _pytorch_outputs(model, tensor)[head]
            onnx = _onnx_outputs(onnx_path, tensor)[head]
            diffs.append(float(np.max(np.abs(pt - onnx))))
        max_diff = max(diffs)
        passed = max_diff < _DIFF_THRESHOLD
        overall_pass = overall_pass and passed
        heads.append(
            {
                "head": _OUTPUT_NAMES[head],
                "max_abs_diff": max_diff,
                "passed": passed,
            }
        )

    return {
        "passed": overall_pass,
        "threshold": _DIFF_THRESHOLD,
        "sample_filenames": filenames,
        "heads": heads,
    }
