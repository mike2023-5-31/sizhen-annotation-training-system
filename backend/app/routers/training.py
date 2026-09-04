"""训练监控路由（T14，需求 6.12/6.13）。

挂载于 ``main.api_router``（整体 ``/api`` 前缀）后对外为：

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

错误映射：任务不存在 → 404；训练未完成/评估暂不可用 → 409。
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from app.services import train_service

router = APIRouter(prefix="/training", tags=["training"])

# WS 每次等待推送信号的超时（秒）：既是唤醒等待，也是断线/兜底轮询节拍
_WS_WAIT_TIMEOUT = 0.5


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
