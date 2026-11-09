"""ONNX 转换路由（T19，需求 6.19/6.20/6.21）。

挂载于 ``main.api_router``（整体 ``/api`` 前缀）后对外为：

- ``POST /api/onnx/convert`` 请求体 ``{"model_path"}`` → 加载 checkpoint
  （仅 ResNet18/50；第三方/自定义 → 400「该模型不支持一键转换」），导出到
  ``EXPORT_DIR/<stem>-<sha256>.onnx``，返回文件名、来源摘要及输入契约（REQ-6.19）；
- ``POST /api/onnx/compare`` 请求体 ``{"model_path", "onnx_name", "dataset_id"?}`` →
  逐头输出对比及样本明细，省略数据集时做合成输入冒烟（REQ-6.20）；
- ``GET /api/onnx/{onnx_name}/download`` → ``FileResponse`` 下载 .onnx
  （REQ-6.21）。

错误映射：
- ``model_path`` 越出 ``MODEL_DIR``（绝对路径/``..`` 穿越）→ 400；模型文件
  不存在 → 404；第三方/自定义 checkpoint（服务层 ``ValueError``）→ 400；
- ``onnx_name`` 越出 ``EXPORT_DIR`` / 非 .onnx 后缀 → 400（NF-7.4 路径限定）；
  ONNX 文件不存在 → 404。
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import config
from app.services import checkpoint_io, inference_service, onnx_service

router = APIRouter(prefix="/onnx", tags=["onnx"])

# 第三方/自定义 checkpoint 的统一报错前缀（REQ-6.19-2，锁定决策 8）
_UNSUPPORTED_PREFIX = "该模型不支持一键转换"
_COMPARISON_IMAGE_LIMIT = 8


class _ConvertIn(BaseModel):
    """转换请求体：模型（MODEL_DIR 内相对路径）。"""

    model_path: str = Field(min_length=1)


class _CompareIn(BaseModel):
    """对比请求体：模型（MODEL_DIR 内相对路径）+ 已导出的 ONNX 文件名。"""

    model_path: str = Field(min_length=1)
    onnx_name: str = Field(min_length=1)
    dataset_id: int | None = Field(default=None, gt=0)


def _resolve_model_path(model_path: str) -> Path:
    """把请求中的 model_path 解析到 MODEL_DIR 内，越界 → 400（NF-7.4）。"""
    try:
        return checkpoint_io.resolve_managed_path(model_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _resolve_onnx_path(onnx_name: str) -> Path:
    """把请求中的 onnx_name 解析到 EXPORT_DIR 内，越界/非 .onnx → 400（NF-7.4）。

    convert 的产物全部平铺写入 EXPORT_DIR 根（无子目录），故此处只接受**裸
    文件名**：绝对路径、含目录分隔（``/`` 或 ``\\`` 的 ``..`` 穿越）与非
    .onnx 后缀一律 400。
    """
    export_dir = config.EXPORT_DIR.resolve()
    requested = Path(onnx_name)
    if requested.is_absolute() or requested.parent != Path(".") or "/" in onnx_name or "\\" in onnx_name:
        raise HTTPException(status_code=400, detail="onnx_name 仅支持导出目录内的文件名")
    if requested.suffix.lower() != ".onnx":
        raise HTTPException(status_code=400, detail="onnx_name 必须以 .onnx 结尾")
    resolved = (export_dir / requested).resolve()
    if resolved != export_dir and export_dir not in resolved.parents:
        raise HTTPException(status_code=400, detail="onnx_name 越出导出目录（非法路径）")
    return resolved


def _load_or_400(model_path: str) -> tuple[object, dict]:
    """解析 + 加载 checkpoint：文件缺失 → 404；结构无法识别 → 400。"""
    resolved = _resolve_model_path(model_path)
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"模型文件不存在：{model_path}")
    try:
        return inference_service.load_model_from_dir(str(resolved), include_digest=True, allow_custom=False)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"{_UNSUPPORTED_PREFIX}：{exc}"
        ) from exc


@router.get("")
def list_exports() -> dict[str, list[dict[str, object]]]:
    """产物仓库：列出 EXPORT_DIR 内的 ONNX 文件（含源 checkpoint 摘要）。"""
    root = config.EXPORT_DIR.resolve()
    items: list[dict[str, object]] = []
    if root.is_dir():
        for path in root.glob("*.onnx"):
            stat = path.stat()
            match = re.search(r"-([0-9a-f]{64})\.onnx$", path.name)
            items.append({
                "onnx_name": path.name,
                "source_sha256": match.group(1) if match else None,
                "size_bytes": stat.st_size,
                "exported_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            })
    items.sort(key=lambda item: str(item["exported_at"]), reverse=True)
    return {"items": items[:200]}


@router.post("/convert")
def convert(payload: _ConvertIn) -> dict:
    """导出 checkpoint 为 ONNX 到 EXPORT_DIR，返回产物文件名（REQ-6.19）。"""
    model, meta = _load_or_400(payload.model_path)
    source_sha256 = meta["source_sha256"]
    onnx_name = f"{Path(payload.model_path).stem[:64]}-{source_sha256}.onnx"
    out_path = _resolve_onnx_path(onnx_name)
    try:
        onnx_service.convert(model, str(out_path), source_sha256=source_sha256)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    stat = out_path.stat()
    return {"onnx_name": onnx_name, "source_sha256": source_sha256,
            "input_shape": onnx_service.INPUT_SHAPE, "opset": onnx_service.OPSET_VERSION,
            "size_bytes": stat.st_size,
            "exported_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()}


@router.post("/compare")
def compare(payload: _CompareIn) -> dict:
    """逐头对比 PyTorch 与 ONNX 输出，返回精度报告（REQ-6.20）。"""
    model, meta = _load_or_400(payload.model_path)
    onnx_path = _resolve_onnx_path(payload.onnx_name)
    if not onnx_path.is_file():
        raise HTTPException(status_code=404, detail=f"ONNX 文件不存在：{payload.onnx_name}")
    try:
        samples = None if payload.dataset_id is None else inference_service.comparison_images(
            payload.dataset_id, _COMPARISON_IMAGE_LIMIT
        )
        return onnx_service.compare(model, str(onnx_path), samples=samples, source_sha256=meta["source_sha256"])
    except inference_service.DatasetNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{onnx_name}/download")
def download(onnx_name: str) -> FileResponse:
    """下载 .onnx 文件（REQ-6.21，路径限定在 EXPORT_DIR 内）。"""
    onnx_path = _resolve_onnx_path(onnx_name)
    if not onnx_path.is_file():
        raise HTTPException(status_code=404, detail=f"ONNX 文件不存在：{onnx_name}")
    return FileResponse(
        onnx_path,
        media_type="application/octet-stream",
        filename=onnx_path.name,
    )
