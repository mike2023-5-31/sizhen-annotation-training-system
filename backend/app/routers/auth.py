from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from app import config
from app.services import auth_service as service

router = APIRouter(prefix="/auth", tags=["accounts"])


class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class UserIn(LoginIn):
    role: str = "member"


class UpdateIn(BaseModel):
    active: bool | None = None
    password: str | None = Field(default=None, min_length=12, max_length=128)


def admin(request):
    if getattr(request.state, "user", {}).get("role") != "admin":
        raise HTTPException(403, "仅管理员可以管理账号")


@router.post("/login")
def login(payload: LoginIn, request: Request, response: Response):
    try:
        user, token = service.login(payload.username, payload.password)
    except ValueError as exc:
        raise HTTPException(401, str(exc))
    previous = request.cookies.get(service.COOKIE_NAME)
    if previous:
        service.logout(previous)
    response.set_cookie(service.COOKIE_NAME, token, max_age=config.SESSION_SECONDS,
                        httponly=True, secure=config.COOKIE_SECURE, samesite="strict", path="/")
    response.headers["Cache-Control"] = "no-store"
    return {"user": user, "csrf_token": service.csrf_token(token)}


@router.get("/me")
def me(request: Request):
    return {"user": request.state.user, "csrf_token": service.csrf_token(request.cookies[service.COOKIE_NAME])}


@router.post("/logout")
def logout(request: Request, response: Response):
    service.logout(request.cookies[service.COOKIE_NAME])
    response.delete_cookie(service.COOKIE_NAME, path="/", secure=config.COOKIE_SECURE, httponly=True, samesite="strict")
    return {"ok": True}


@router.get("/users")
def users(request: Request):
    admin(request)
    return service.list_users()


@router.post("/users", status_code=201)
def create(payload: UserIn, request: Request):
    admin(request)
    try:
        return service.create_user(payload.username, payload.password, payload.role)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.patch("/users/{user_id}")
def update(user_id: str, payload: UpdateIn, request: Request):
    admin(request)
    try:
        return service.update_user(user_id, **payload.model_dump(exclude_unset=True))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
