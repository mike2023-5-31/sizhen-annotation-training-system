"""推理路由（T17，需求 6.15/6.16/6.17/6.18）。

挂载于 ``main.api_router``（整体 ``/api`` 前缀）后对外为：

- ``GET /api/inference/models`` → ``{"models": [checkpoint 相对路径]}``
  （``MODEL_DIR`` 下全部 .pth/.pt 文件，供前端模型下拉选择，REQ-6.15-2）；
- ``POST /api/inference/predict``  请求体 ``{"model_path", "dataset_id"}`` →
  ``{"predictions": [{"filename", "predictions": [{"category","label",
  "confidence"} ×4]}]}``（REQ-6.16/6.17）；
- ``POST /api/inference/jobs`` 启动后台批量任务（202），
  ``GET /api/inference/jobs/{job_id}`` 返回进度与最终结果；
- ``POST /api/inference/preannotation`` 请求体 ``{"model_path", "dataset_id"}``
  → ``{"generated": N}``（REQ-6.18）。

错误映射：
- ``model_path`` 越出 ``MODEL_DIR``（绝对路径/``..`` 穿越）→ 400（NF-7.4 路径
  写入限定系统目录）；模型文件不存在 → 404；第三方/损坏 checkpoint（服务层
  ``ValueError``）→ 400；
- 数据集不存在 → 404；图片缺失/损坏 → 400（整批失败即失败，不静默跳过）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

from app import config
from app.services import checkpoint_io, inference_service

router = APIRouter(prefix="/inference", tags=["inference"])

# 训练 checkpoint 扩展名（T13 产物）；.onnx 等其它产物不在本列表（T19 职责）
_CHECKPOINT_SUFFIXES = checkpoint_io.CHECKPOINT_SUFFIXES


@router.get("/models")
def list_models() -> dict[str, list[str]]:
    """MODEL_DIR 下全部 checkpoint 文件（相对路径，供预测/预标注选择）。"""
    model_dir = config.MODEL_DIR
    files = [
        str(p.relative_to(model_dir)).replace("\\", "/")
        for p in sorted(model_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in _CHECKPOINT_SUFFIXES
    ]
    return {"models": files}


class _InferenceIn(BaseModel):
    """预测/预标注请求体：模型（MODEL_DIR 内相对路径）+ 目标数据集。"""

    model_path: str = Field(min_length=1)
    dataset_id: int = Field(gt=0)


class _PreannotationIn(_InferenceIn):
    confidence_threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class _BatchStatus(BaseModel):
    job_id: str
    status: Literal["running", "completed", "failed"]
    processed: int
    total: int
    predictions: list[dict[str, Any]]
    error: str | None


def _resolve_model_path(model_path: str) -> Path:
    """把请求中的 model_path 解析到 MODEL_DIR 内，越界 → 400（NF-7.4）。

    仅接受 MODEL_DIR 内的文件名/相对路径；绝对路径与 ``..`` 穿越一律拒绝。
    """
    try:
        return checkpoint_io.resolve_managed_path(model_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _existing_model_path(model_path: str) -> Path:
    resolved = _resolve_model_path(model_path)
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"模型文件不存在：{model_path}")
    return resolved


def _load_or_404(model_path: str) -> tuple[object, dict]:
    """解析 + 加载 checkpoint：文件缺失 → 404；结构无法识别 → 400。"""
    resolved = _existing_model_path(model_path)
    try:
        return inference_service.load_model_from_dir(str(resolved))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/predict")
def predict(payload: _InferenceIn) -> dict[str, list[dict]]:
    """整集批量推理：每图每类预测子标签 + 概率（REQ-6.16）。"""
    model, _meta = _load_or_404(payload.model_path)
    try:
        predictions = inference_service.predict_dataset(payload.dataset_id, model)
    except inference_service.DatasetNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"predictions": predictions}


@router.post(
    "/jobs",
    status_code=202,
    responses={409: {"description": "已有批量推理任务在运行"}},
)
def start_batch(payload: _InferenceIn, response: Response) -> dict[str, str]:
    """接受批量推理任务，通过 Location 查询处理进度和结果。"""
    path = _existing_model_path(payload.model_path)
    try:
        job_id = inference_service.batch_jobs.start(payload.dataset_id, path)
    except inference_service.DatasetNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except inference_service.BatchInferenceBusy as exc:
        raise HTTPException(status_code=409, detail="已有批量推理任务在运行") from exc
    response.headers["Location"] = f"/api/inference/jobs/{job_id}"
    return {"job_id": job_id}


@router.get("/jobs")
def list_batches() -> dict[str, list[dict[str, Any]]]:
    """作业中心：最近批量推理任务摘要（不含整批预测）。"""
    return {"items": inference_service.batch_jobs.list()}


@router.get("/jobs/{job_id}", response_model=_BatchStatus)
def get_batch(job_id: str) -> dict[str, Any]:
    """查询当前批次。完成后返回与同步预测接口相同的预测项。"""
    status = inference_service.batch_jobs.status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="批量推理任务不存在或结果已过期")
    return status


@router.post("/preannotation")
def preannotation(payload: _PreannotationIn) -> dict[str, int]:
    """基于预测生成预标注 XML（仅未人工标注图片），返回生成数量（REQ-6.18）。"""
    model, _meta = _load_or_404(payload.model_path)
    try:
        generated = inference_service.generate_preannotation_xml(
            payload.dataset_id, model, confidence_threshold=payload.confidence_threshold
        )
    except inference_service.DatasetNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"generated": generated}
