"""FastAPI 应用骨架（任务 T5）。

- ``/api`` 前缀路由聚合器 ``api_router``：T5 仅含 ``GET /api/health``；
  T6+ 各业务 router（datasets/images/training）由后续任务 include 进来。
- CORS：开发期允许任意来源（``*``，同时覆盖 http://localhost:5173），
  生产可用 ``CORS_ORIGINS`` 环境变量收紧。
- ``/static``：受登录保护的静态图片服务，根目录 = ``<DATA_DIR>/datasets``。
- 登录会话统一覆盖 HTTP 与 WebSocket；账号初始化见 app.accounts。
- 全局异常处理器：统一返回 JSON ``{"detail": ...}``。

依赖边界：本文件不直接 import ``app.db`` / ``app.models``（T4 职责）；
业务依赖经 routers → services 间接引用（T6 起）。
"""
from __future__ import annotations

import importlib
import logging

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.auth_middleware import AuthenticationMiddleware
from app.routers import auth
from app.routers import annotations, datasets, images, inference, models, onnx, training
from app.services.auth_service import COOKIE_NAME

logger = logging.getLogger("uvicorn.error")

# StaticFiles 挂载要求目录已存在；先确保数据目录树（幂等）
config.ensure_dirs()

app = FastAPI(
    title="四诊数据标注与AI模型训练系统",
    version="0.1.0",
)

app.add_middleware(AuthenticationMiddleware)

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
api_router.include_router(auth.router)


@api_router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# 业务路由（数据集 CRUD，T6）
api_router.include_router(datasets.router)

# 业务路由（图片浏览/检索，T9）
api_router.include_router(images.router)

# 业务路由（标注标签树/写回/硬删除，T10）
api_router.include_router(annotations.router)

# 业务路由（推理：模型列表/批量预测/预标注 XML，T17）
api_router.include_router(inference.router)

# 业务路由（自定义模型上传，需求 6.7）
api_router.include_router(models.router)

# 业务路由（训练监控：状态轮询/评估结果/指标 WebSocket，T14）
api_router.include_router(training.router)

# 业务路由（ONNX 转换/精度对比/下载，T19）
api_router.include_router(onnx.router)

# 业务路由（ZIP 导入，T7）：模块文件名 import.py 与 Python 关键字同名，
# 无法写 `from app.routers import import`，改经 importlib 加载。
api_router.include_router(importlib.import_module("app.routers.import").router)


app.include_router(api_router)


def openapi_with_session() -> dict:
    """Describe the HTTP requirements enforced by AuthenticationMiddleware."""
    if app.openapi_schema is not None:
        return app.openapi_schema
    schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
    schemes = schema.setdefault("components", {}).setdefault("securitySchemes", {})
    schemes.update({
        "sessionCookie": {
            "type": "apiKey", "in": "cookie", "name": COOKIE_NAME,
            "description": "通过同源登录接口签发的 HttpOnly 会话 Cookie。",
        },
        "csrfHeader": {
            "type": "apiKey", "in": "header", "name": "X-CSRF-Token",
            "description": "写操作必需，登录或 /api/auth/me 返回会话对应的令牌。",
        },
    })
    for path, methods in schema["paths"].items():
        if not path.startswith("/api/"):
            continue
        for method, operation in methods.items():
            if (path, method) == ("/api/health", "get"):
                operation["security"] = []
                continue
            if (path, method) == ("/api/auth/login", "post"):
                operation["security"] = []
                operation.setdefault("parameters", []).append({
                    "in": "header", "name": "X-Requested-With", "required": True,
                    "schema": {"type": "string", "enum": ["XMLHttpRequest"]},
                })
                continue
            security: dict[str, list] = {"sessionCookie": []}
            responses = operation.setdefault("responses", {})
            responses.setdefault("401", {"description": "需要有效登录会话"})
            if method not in {"get", "head", "options"}:
                security["csrfHeader"] = []
                responses.setdefault("403", {"description": "请求来源、CSRF 或操作权限校验失败"})
            operation["security"] = [security]
    app.openapi_schema = schema
    return schema


app.openapi = openapi_with_session

# --------------------------------------------------------------------------
# 全局异常处理：统一 JSON {"detail": ...}
# --------------------------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "服务内部错误，请联系管理员查看日志"})


# --------------------------------------------------------------------------
# 静态图片服务：GET /static/<dataset>/<filename>
# --------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
