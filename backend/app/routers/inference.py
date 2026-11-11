"""推理路由（6.15 / 6.16 / 6.17 / 6.18）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from app import config
from app.schemas.inference import PredictRequest, PreannotationRequest
from app.services import checkpoint, predictor

router = APIRouter(prefix="/inference", tags=["inference"])


def _load_or_404(model_path: str):
    try:
        resolved = checkpoint.resolve_model_path(config.MODEL_DIR, model_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"模型文件不存在：{model_path}")
    try:
        return predictor.load_model(resolved)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/models")
def list_models() -> dict[str, list[str]]:
    return {"models": checkpoint.list_checkpoints(config.MODEL_DIR)}


@router.post("/predict")
def predict(body: PredictRequest) -> dict[str, Any]:
    model, meta = _load_or_404(body.model_path)
    try:
        return predictor.predict_dataset(
            body.dataset_id, model, active_heads=meta.get("active_heads")
        )
    except predictor.DatasetNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/preannotation")
def preannotation(body: PreannotationRequest) -> dict[str, int]:
    model, meta = _load_or_404(body.model_path)
    try:
        generated = predictor.generate_preannotation_xml(
            body.dataset_id,
            model,
            confidence_threshold=body.confidence_threshold,
            active_heads=meta.get("active_heads"),
        )
    except predictor.DatasetNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"generated": generated}
