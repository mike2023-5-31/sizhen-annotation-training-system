"""全部业务 HTTP、静态文件和 WebSocket 的统一登录边界。"""
import hmac

from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app import config
from app.services import auth_service as service


class AuthenticationMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in {"http", "websocket"}:
            return await self.app(scope, receive, send)
        connection = HTTPConnection(scope)
        path = scope["path"].rstrip("/")
        method = scope.get("method", "GET")
        origin = connection.headers.get("origin")
        host_origin = f"{connection.url.scheme.replace('ws', 'http')}://{connection.headers.get('host', '')}"
        origin_ok = not origin or origin.rstrip("/") in [host_origin, *config.AUTH_ALLOWED_ORIGINS]

        async def reject(status, detail):
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 4401 if status == 401 else 4403})
            else:
                await JSONResponse({"detail": detail}, status_code=status,
                                   headers={"Cache-Control": "no-store"})(scope, receive, send)

        if path == "/api/health" and method == "GET":
            return await self.app(scope, receive, send)
        if path == "/api/auth/login" and method == "POST":
            if not origin_ok or connection.headers.get("x-requested-with") != "XMLHttpRequest":
                return await reject(403, "登录请求来源无效")
            return await self.app(scope, receive, send)
        token = connection.cookies.get(service.COOKIE_NAME)
        user = await run_in_threadpool(service.authenticate, token)
        if not user:
            return await reject(401, "请登录后访问")
        owner = connection.headers.get("x-annotation-owner")
        if owner and owner != user["id"]:
            return await reject(409, "浏览器账号已改变，请重新登录原账号恢复草稿")
        if scope["type"] == "websocket" and not origin_ok:
            return await reject(403, "连接来源无效")
        if method not in {"GET", "HEAD", "OPTIONS"}:
            csrf = connection.headers.get("x-csrf-token", "")
            if not origin_ok or not hmac.compare_digest(csrf.encode(), service.csrf_token(token).encode()):
                return await reject(403, "会话校验失败，请重新加载页面")
        if path == "/api/models/custom" and user["role"] != "admin":
            return await reject(403, "自定义可执行模型仅允许管理员上传")
        scope.setdefault("state", {})["user"] = user

        async def guarded_send(message):
            if message["type"] == "websocket.send":
                if not await run_in_threadpool(service.authenticate, token):
                    from starlette.websockets import WebSocketDisconnect
                    await send({"type": "websocket.close", "code": 4401})
                    raise WebSocketDisconnect(4401)
            if message["type"] == "http.response.start":
                message["headers"] = [(k, v) for k, v in message.get("headers", []) if k.lower() != b"cache-control"]
                message["headers"].append((b"cache-control", b"no-store"))
            await send(message)
        return await self.app(scope, receive, guarded_send)
