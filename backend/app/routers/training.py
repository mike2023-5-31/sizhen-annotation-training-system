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

错误映射：任务不存在 → 404；设备已占用/评估暂不可用 → 409；配置非法 /
taxonomy 校验失败 / 存档非法（同步 ValueError）→ 400。
"""
from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from app import config
from app.services import checkpoint_io, train_service

router = APIRouter(prefix="/training", tags=["training"])

# WS 每次等待推送信号的超时（秒）：既是唤醒等待，也是断线/兜底轮询节拍
_WS_WAIT_TIMEOUT = 0.5
# 空闲保活间隔（秒）：低于反向代理常见的 60-90s 读超时，避免 epoch 间隔过长被断开
_WS_HEARTBEAT_SECONDS = 25
_STATUS_FIELDS = ("status", "error", "finished_at", "device", "gpu_count", "use_amp")


# ---------------------------------------------------------------------------
# 训练配置请求模型（键与 train_service.normalize_config 对齐，REQ-6.9）
# ---------------------------------------------------------------------------


class TrainingConfigBody(BaseModel):
    """训练超参（可选键默认值 = normalize_config 同源默认值）。

    取值域校验（lr>0、epochs≥1、backbone/optimizer/loss/增强受支持等）刻意留在
    服务层 ``normalize_config``：语义错误 → ValueError → 端点映射 400，文案统一。
    """

    backbone: str = "resnet18"
    custom_model_path: str | None = None
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
    resume_from: str | None = None


class ResumeTrainingRequest(TrainingConfigBody):
    """POST /training/resume 请求体：checkpoint_path 必填 + 可选覆盖超参。

    覆盖键缺省时沿用存档内训练配置（``config=None``，REQ-6.11-1）；仅显式传入
    的键会覆盖存档配置（``exclude_unset``）。
    """

    checkpoint_path: str


class TrainingGPU(BaseModel):
    index: int
    name: str | None
    total_bytes: int | None
    status: Literal["detected", "unavailable"]
    temperature_c: int | None
    utilization_percent: int | None


class TrainingCapabilities(BaseModel):
    device: str
    gpu_count: int
    parallel_mode: Literal["cpu", "single_gpu", "data_parallel"]
    devices: list[TrainingGPU]


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
        options = body.model_dump()
        if body.resume_from:
            options["resume_from"] = str(checkpoint_io.resolve_managed_path(body.resume_from, allow_absolute=True))
        job_id = train_service.start_training(options)
    except train_service.TrainingBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
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
        path = checkpoint_io.resolve_managed_path(body.checkpoint_path, allow_absolute=True)
        overrides["model_dir"] = str(config.MODEL_DIR)
        job_id = train_service.resume_training(str(path), overrides)
    except train_service.TrainingBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise _value_error_to_400(exc) from exc
    return {"job_id": job_id}


@router.get("/status")
def get_status(job_id: Optional[str] = None) -> dict[str, list[dict[str, Any]]]:
    """当前训练任务状态 + 指标缓冲（轮询兜底，REQ-6.12-4）。"""
    return train_service.train_manager.get_metrics(job_id)


@router.get("/capabilities", response_model=TrainingCapabilities)
def get_capabilities() -> dict[str, Any]:
    """报告训练进程能检测到的设备和当前遥测。"""
    return train_service.training_capabilities()


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
    - ``{"type": "status", "job_id", "status", "error", "finished_at", "device", "gpu_count", "use_amp"}``
      状态或设备信息变化；
    - ``{"type": "heartbeat"}`` 空闲保活帧（客户端忽略未知帧）。
    """
    await websocket.accept()
    manager = train_service.train_manager
    cursor: dict[str, int] = {}
    last_status: dict[str, dict[str, Any]] = {}

    async def wait_for_disconnect() -> None:
        while (await websocket.receive())["type"] != "websocket.disconnect":
            pass

    disconnected = asyncio.create_task(wait_for_disconnect())
    try:
        snapshot = manager.get_metrics()
        for job in snapshot["jobs"]:
            cursor[job["job_id"]] = len(job["metrics"])
            last_status[job["job_id"]] = {field: job.get(field) for field in _STATUS_FIELDS}
        await websocket.send_json({"type": "snapshot", "jobs": snapshot["jobs"]})
        last_sent = time.monotonic()

        while not disconnected.done():
            await asyncio.to_thread(manager.metrics_event.wait, _WS_WAIT_TIMEOUT)
            if disconnected.done():
                break
            manager.metrics_event.clear()
            sent = False
            for job in manager.list_jobs():
                job_id = job["job_id"]
                metrics = job["metrics"]
                seen = cursor.get(job_id, 0)
                if len(metrics) > seen:
                    cursor[job_id] = len(metrics)
                    await websocket.send_json(
                        {"type": "metrics", "job_id": job_id, "metrics": metrics[seen:]}
                    )
                    sent = True
                status = {field: job.get(field) for field in _STATUS_FIELDS}
                if status != last_status.get(job_id):
                    last_status[job_id] = status
                    await websocket.send_json(
                        {"type": "status", "job_id": job_id, **status}
                    )
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
        with suppress(asyncio.CancelledError):
            await disconnected
