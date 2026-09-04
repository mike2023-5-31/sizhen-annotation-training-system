"""FastAPI 应用骨架（任务 T5）。

- ``/api`` 前缀路由聚合器 ``api_router``：T5 仅含 ``GET /api/health``；
  T6+ 各业务 router（datasets/images/training）由后续任务 include 进来。
- CORS：开发期允许任意来源（``*``，同时覆盖 http://localhost:5173），
  生产可用 ``CORS_ORIGINS`` 环境变量收紧。
- ``/static``：静态图片服务，根目录 = ``<DATA_DIR>/datasets``。
- 全局异常处理器：统一返回 JSON ``{"detail": ...}``。

依赖边界：本文件不直接 import ``app.db`` / ``app.models``（T4 职责）；
业务依赖经 routers → services 间接引用（T6 起）。
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.routers import datasets

logger = logging.getLogger("uvicorn.error")

# StaticFiles 挂载要求目录已存在；先确保数据目录树（幂等）
config.ensure_dirs()

app = FastAPI(
    title="四诊数据标注与AI模型训练系统",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,  # 开发期 ["*"]；生产经 CORS_ORIGINS 收紧
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# /api 路由（后续任务在此 include_router 追加业务路由）
# --------------------------------------------------------------------------
api_router = APIRouter(prefix="/api")


@api_router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# 业务路由（数据集 CRUD，T6）
api_router.include_router(datasets.router)


app.include_router(api_router)

# --------------------------------------------------------------------------
# 全局异常处理：统一 JSON {"detail": ...}
# --------------------------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": f"Internal server error: {exc}"})


# --------------------------------------------------------------------------
# 静态图片服务：GET /static/<dataset>/<filename>
# --------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
