"""训练控制与监控路由（6.6 / 6.8 / 6.11 / 6.12 / 6.13）。"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from app import config
from app.schemas.training import (
    ResumeTrainingRequest,
    StartTrainingRequest,
    TrainingCapabilities,
)
from app.services import checkpoint, trainer

router = APIRouter(prefix="/training", tags=["training"])

_WS_WAIT_SECONDS = 0.5
_WS_HEARTBEAT_SECONDS = 25
_STATUS_FIELDS = ("status", "error", "finished_at", "device", "gpu_count", "use_amp")


@router.post("/start")
def start_training(body: StartTrainingRequest) -> dict[str, str]:
    options = body.model_dump()
    try:
        job_id = trainer.start_training(options)
    except trainer.TrainingBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"job_id": job_id}


@router.post("/resume")
def resume_training(body: ResumeTrainingRequest) -> dict[str, str]:
    try:
        path = checkpoint.resolve_model_path(config.MODEL_DIR, body.checkpoint_path)
        overrides = body.model_dump(exclude_unset=True, exclude={"checkpoint_path"})
        job_id = trainer.resume_training(str(path), overrides)
    except trainer.TrainingBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"job_id": job_id}


@router.post("/stop/{job_id}")
def stop_training(job_id: str) -> dict[str, bool]:
    manager = trainer.train_manager
    if manager.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail=f"训练任务不存在：{job_id}")
    manager.stop(job_id)
    return {"ok": True}


@router.get("/status")
def training_status(job_id: Optional[str] = None) -> dict[str, Any]:
    return trainer.train_manager.get_metrics(job_id)


@router.get("/capabilities", response_model=TrainingCapabilities)
def capabilities() -> dict[str, Any]:
    return trainer.training_capabilities()


@router.get("/eval/{job_id}")
def evaluation(job_id: str) -> dict[str, Any]:
    manager = trainer.train_manager
    if manager.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail=f"训练任务不存在：{job_id}")
    report = manager.compute_eval(job_id)
    if report is None:
        raise HTTPException(status_code=409, detail=f"训练未完成，评估暂不可用：{job_id}")
    return report


@router.websocket("/ws/training")
async def ws_training(websocket: WebSocket) -> None:
    """训练指标实时推送：连接即发快照，随后增量推送轮次与状态变化。"""
    await websocket.accept()
    manager = trainer.train_manager
    cursors: dict[str, int] = {}
    last_status: dict[str, dict[str, Any]] = {}

    async def wait_disconnect() -> None:
        while (await websocket.receive())["type"] != "websocket.disconnect":
            pass

    disconnected = asyncio.create_task(wait_disconnect())
    try:
        snapshot = manager.get_metrics()
        for job in snapshot["jobs"]:
            cursors[job["job_id"]] = len(job["metrics"])
            last_status[job["job_id"]] = {f: job.get(f) for f in _STATUS_FIELDS}
        await websocket.send_json({"type": "snapshot", "jobs": snapshot["jobs"]})
        last_sent = time.monotonic()

        while not disconnected.done():
            await asyncio.to_thread(manager.metrics_event.wait, _WS_WAIT_SECONDS)
            if disconnected.done():
                break
            manager.metrics_event.clear()
            sent = False
            for job in manager.list_jobs():
                job_id = job["job_id"]
                metrics = job["metrics"]
                seen = cursors.get(job_id, 0)
                if len(metrics) > seen:
                    cursors[job_id] = len(metrics)
                    await websocket.send_json(
                        {"type": "metrics", "job_id": job_id, "metrics": metrics[seen:]}
                    )
                    sent = True
                status = {f: job.get(f) for f in _STATUS_FIELDS}
                if status != last_status.get(job_id):
                    last_status[job_id] = status
                    await websocket.send_json({"type": "status", "job_id": job_id, **status})
                    sent = True
            now = time.monotonic()
            if sent:
                last_sent = now
            elif now - last_sent >= _WS_HEARTBEAT_SECONDS:
                await websocket.send_json({"type": "heartbeat"})
                last_sent = now
    except WebSocketDisconnect:
        return
    finally:
        disconnected.cancel()
        try:
            await disconnected
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
