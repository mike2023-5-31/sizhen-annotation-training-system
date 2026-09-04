"""训练路由（T14 监控 + 训练控制入口，需求 6.6/6.11/6.12/6.13）。

挂载于 ``main.api_router``（整体 ``/api`` 前缀）后对外为：

- ``POST /api/training/start`` → body 训练配置（Pydantic 校验 + 服务层
  ``normalize_config``）→ ``{"job_id": ...}``，非阻塞后台启动（REQ-6.12-3）；
- ``POST /api/training/stop/{job_id}`` → ``{"ok": true}``，请求停止训练；
- ``POST /api/training/resume`` → body ``{"checkpoint_path": str, ...可选覆盖}``
  → ``{"job_id": ...}``，从断点续训（REQ-6.11，覆盖缺省时沿用存档内配置）；
- ``GET /api/training/status`` → ``{"jobs": [job 公开视图（含 metrics 缓冲）]}``，
  可选 ``?job_id=`` 过滤单个任务（轮询兜底，REQ-6.12-4 刷新重建曲线）；
- ``GET /api/training/eval/{job_id}`` → ``{"confusion": {类别: [[..]]},
  "roc": {类别: {"fpr": [...], "tpr": [...], "auc": float}}}``（4 类别，
  REQ-6.13-1/2/3）；无验证集 → 空 dict（REQ-6.13-4 不报错）；
- ``WS /api/training/ws/training`` → 连接即发全量快照，随后每个 epoch 指标
  追加/状态变化时实时推送（REQ-6.12-1/2）。

推送实现：训练线程追加指标后只 ``Event.set()``（非阻塞）；本端在事件循环侧
经线程池 ``wait(timeout)`` 唤醒，按各自游标对 metrics 缓冲做 diff 增量发送，
每个连接互不影响。

错误映射：任务不存在 → 404；训练未完成/评估暂不可用 → 409；配置非法 /
taxonomy 校验失败 / 存档非法（同步 ValueError）→ 400。
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from app.services import train_service

router = APIRouter(prefix="/training", tags=["training"])

# WS 每次等待推送信号的超时（秒）：既是唤醒等待，也是断线/兜底轮询节拍
_WS_WAIT_TIMEOUT = 0.5


# ---------------------------------------------------------------------------
# 训练配置请求模型（键与 train_service.normalize_config 对齐，REQ-6.9）
# ---------------------------------------------------------------------------


class TrainingConfigBody(BaseModel):
    """训练超参（可选键默认值 = normalize_config 同源默认值）。

    取值域校验（lr>0、epochs≥1、backbone/optimizer/loss/增强受支持等）刻意留在
    服务层 ``normalize_config``：语义错误 → ValueError → 端点映射 400，文案统一。
    """

    backbone: str = "resnet18"
    lr: float = 1e-3
    batch_size: int = 16
    epochs: int = 10
    optimizer: str = "adam"
    dropout: float = 0.0
    loss: str = "crossentropy"
    augmentation: list[str] = Field(default_factory=list)
    fp16: bool = False
    val_split: float = 0.2
    datasets: Optional[list[int]] = None


class StartTrainingRequest(TrainingConfigBody):
    """POST /training/start 请求体：datasets 必填（首个为主训练集，REQ-6.6）。"""

    datasets: list[int]


class ResumeTrainingRequest(TrainingConfigBody):
    """POST /training/resume 请求体：checkpoint_path 必填 + 可选覆盖超参。

    覆盖键缺省时沿用存档内训练配置（``config=None``，REQ-6.11-1）；仅显式传入
    的键会覆盖存档配置（``exclude_unset``）。
    """

    checkpoint_path: str


def _value_error_to_400(exc: ValueError) -> HTTPException:
    """同步校验失败（配置非法/taxonomy/存档）→ 400，携带服务层中文文案。"""
    return HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------------------
# 训练控制入口（start / stop / resume）
# ---------------------------------------------------------------------------


@router.post("/start")
def start_job(body: StartTrainingRequest) -> dict[str, str]:
    """启动新训练任务（后台非阻塞，REQ-6.6 至 6.11）→ ``{"job_id"}``。

    同步完成配置归一化校验、数据集存在性与逐张 taxonomy 校验（锁定决策 6），
    失败 → ValueError → 400，不创建任务。
    """
    try:
        job_id = train_service.start_training(body.model_dump())
    except ValueError as exc:
        raise _value_error_to_400(exc) from exc
    return {"job_id": job_id}


@router.post("/stop/{job_id}")
def stop_job(job_id: str) -> dict[str, bool]:
    """请求停止训练任务（下一 batch/epoch 边界优雅退出并存档）→ ``{"ok": true}``。

    任务不存在 → 404；任务已结束的重复停止请求幂等返回 ``ok: true``。
    """
    manager = train_service.train_manager
    if manager.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail=f"训练任务不存在：{job_id}")
    manager.stop(job_id)
    return {"ok": True}


@router.post("/resume")
def resume_job(body: ResumeTrainingRequest) -> dict[str, str]:
    """从断点存档续训（REQ-6.11）→ ``{"job_id"}``。

    存档不存在/结构损坏/类别数与标签体系不一致 → ValueError → 400。
    """
    overrides = body.model_dump(exclude_unset=True, exclude={"checkpoint_path"})
    try:
        job_id = train_service.resume_training(
            body.checkpoint_path, overrides or None
        )
    except ValueError as exc:
        raise _value_error_to_400(exc) from exc
    return {"job_id": job_id}


@router.get("/status")
def get_status(job_id: Optional[str] = None) -> dict[str, list[dict[str, Any]]]:
    """当前训练任务状态 + 指标缓冲（轮询兜底，REQ-6.12-4）。"""
    return train_service.train_manager.get_metrics(job_id)


@router.get("/eval/{job_id}")
def get_eval(job_id: str) -> dict[str, Any]:
    """训练评估结果：4 类别混淆矩阵 + one-vs-rest ROC/AUC（REQ-6.13）。

    任务不存在 → 404；仍在运行且无缓存评估 → 409；无验证集 → 空 dict。
    """
    manager = train_service.train_manager
    if manager.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail=f"训练任务不存在：{job_id}")
    result = manager.compute_eval(job_id)
    if result is None:
        raise HTTPException(status_code=409, detail=f"训练未完成，评估暂不可用：{job_id}")
    return result


@router.websocket("/ws/training")
async def ws_training(websocket: WebSocket) -> None:
    """训练指标推送（REQ-6.12-1/2）：连接即发快照，随后增量推送指标/状态帧。

    帧协议：
    - ``{"type": "snapshot", "jobs": [...]}`` 连接时全量（含 metrics 缓冲）；
    - ``{"type": "metrics", "job_id", "metrics": [新增轮次...]}`` 每 epoch 一条；
    - ``{"type": "status", "job_id", "status", "error", "finished_at"}`` 状态变化。
    """
    await websocket.accept()
    manager = train_service.train_manager
    cursor: dict[str, int] = {}
    last_status: dict[str, str] = {}
    try:
        snapshot = manager.get_metrics()
        for job in snapshot["jobs"]:
            cursor[job["job_id"]] = len(job["metrics"])
            last_status[job["job_id"]] = job["status"]
        await websocket.send_json({"type": "snapshot", "jobs": snapshot["jobs"]})

        while True:
            await asyncio.to_thread(manager.metrics_event.wait, _WS_WAIT_TIMEOUT)
            manager.metrics_event.clear()
            for job in manager.list_jobs():
                job_id = job["job_id"]
                metrics = job["metrics"]
                seen = cursor.get(job_id, 0)
                if len(metrics) > seen:
                    cursor[job_id] = len(metrics)
                    await websocket.send_json(
                        {"type": "metrics", "job_id": job_id, "metrics": metrics[seen:]}
                    )
                if job["status"] != last_status.get(job_id):
                    last_status[job_id] = job["status"]
                    await websocket.send_json(
                        {
                            "type": "status",
                            "job_id": job_id,
                            "status": job["status"],
                            "error": job.get("error"),
                            "finished_at": job.get("finished_at"),
                        }
                    )
    except WebSocketDisconnect:
        return
