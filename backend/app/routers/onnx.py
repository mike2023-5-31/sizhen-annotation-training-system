"""ONNX 转换 / 精度对比 / 下载路由（6.19 / 6.20 / 6.21）。"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app import config
from app.schemas.onnx import CompareRequest, ConvertRequest
from app.services import checkpoint, converter, predictor

router = APIRouter(prefix="/onnx", tags=["onnx"])


def _resolve_onnx_name(onnx_name: str) -> Path:
    candidate = Path(onnx_name)
    if candidate.is_absolute() or candidate.parent != Path(".") or "/" in onnx_name or "\\" in onnx_name:
        raise HTTPException(status_code=400, detail="onnx_name 仅支持导出目录内的文件名")
    if candidate.suffix.lower() != ".onnx":
        raise HTTPException(status_code=400, detail="onnx_name 必须以 .onnx 结尾")
    export_dir = config.EXPORT_DIR.resolve()
    resolved = (export_dir / candidate).resolve()
    if export_dir not in resolved.parents and resolved != export_dir:
        raise HTTPException(status_code=400, detail="onnx_name 越出导出目录")
    return resolved


def _load_supported_model(model_path: str):
    try:
        resolved = checkpoint.resolve_model_path(config.MODEL_DIR, model_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"模型文件不存在：{model_path}")
    try:
        model, meta = predictor.load_model(resolved, allow_custom=False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"该模型不支持一键转换：{exc}")
    if meta["architecture"] not in ("resnet18", "resnet50"):
        raise HTTPException(status_code=400, detail="该模型不支持一键转换（仅 ResNet18/50）")
    return model, resolved


@router.get("")
def list_exports() -> dict[str, list[dict[str, Any]]]:
    root = config.EXPORT_DIR.resolve()
    items: list[dict[str, Any]] = []
    if root.is_dir():
        for path in root.glob("*.onnx"):
            stat = path.stat()
            match = re.search(r"-([0-9a-f]{16})\.onnx$", path.name)
            items.append(
                {
                    "onnx_name": path.name,
                    "source_tag": match.group(1) if match else None,
                    "size_bytes": stat.st_size,
                    "exported_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                }
            )
    items.sort(key=lambda x: str(x["exported_at"]), reverse=True)
    return {"items": items[:200]}


@router.post("/convert")
def convert(body: ConvertRequest) -> dict[str, Any]:
    model, source_path = _load_supported_model(body.model_path)
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()[:16]
    onnx_name = f"{Path(body.model_path).stem[:48]}-{digest}.onnx"
    out_path = _resolve_onnx_name(onnx_name)
    try:
        converter.convert(model, out_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"ONNX 转换失败：{exc}")
    stat = out_path.stat()
    return {
        "onnx_name": onnx_name,
        "source_tag": digest,
        "input_shape": list(converter.INPUT_SHAPE),
        "opset": converter.OPSET_VERSION,
        "size_bytes": stat.st_size,
    }


@router.post("/compare")
def compare(body: CompareRequest) -> dict[str, Any]:
    model, _ = _load_supported_model(body.model_path)
    onnx_path = _resolve_onnx_name(body.onnx_name)
    if not onnx_path.is_file():
        raise HTTPException(status_code=404, detail=f"ONNX 文件不存在：{body.onnx_name}")
    samples = None
    if body.dataset_id is not None:
        try:
            samples = predictor.comparison_images(body.dataset_id, 8)
        except predictor.DatasetNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc))
    try:
        return converter.compare(model, onnx_path, samples=samples)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"精度对比失败：{exc}")


@router.get("/{onnx_name}/download")
def download(onnx_name: str) -> FileResponse:
    onnx_path = _resolve_onnx_name(onnx_name)
    if not onnx_path.is_file():
        raise HTTPException(status_code=404, detail=f"ONNX 文件不存在：{onnx_name}")
    return FileResponse(onnx_path, media_type="application/octet-stream", filename=onnx_path.name)
