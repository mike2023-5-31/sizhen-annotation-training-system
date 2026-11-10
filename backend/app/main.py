"""FastAPI 应用入口：聚合所有业务路由、CORS 与静态图片服务。"""
from __future__ import annotations

import logging

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.db import Base, engine
from app.routers import datasets, inference, models, onnx, training

logger = logging.getLogger("uvicorn.error")

config.ensure_dirs()
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="四诊数据标注与AI模型训练系统",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_router = APIRouter(prefix="/api")


@api_router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


api_router.include_router(datasets.router)
api_router.include_router(datasets.labels_router)
api_router.include_router(inference.router)
api_router.include_router(models.router)
api_router.include_router(training.router)
api_router.include_router(onnx.router)

app.include_router(api_router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "服务内部错误，请稍后再试"})


app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
