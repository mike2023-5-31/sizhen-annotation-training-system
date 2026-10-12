"""独立账号与可撤销会话。密码和会话凭证仅存加盐派生值/摘要。"""
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import secrets
import uuid
import re

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from app import config
from app.db import SessionLocal
from app.models.orm import User, LoginSession

COOKIE_NAME = "sizhen_session"
ITERATIONS = 600_000


def now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def hash_password(password: str) -> str:
    if not 12 <= len(password) <= 128:
        raise ValueError("密码长度须为 12–128 个字符")
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), ITERATIONS)
    return f"pbkdf2_sha256${ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    if len(password) > 128:
        return False
    try:
        algorithm, rounds, salt, digest = stored.split("$")
        if algorithm != "pbkdf2_sha256" or int(rounds) != ITERATIONS:
            return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(rounds)).hex()
        return hmac.compare_digest(actual, digest)
    except (ValueError, TypeError):
        return False


def public_user(user):
    return {"id": user.id, "username": user.username, "role": user.role, "active": user.active}


def create_user(username: str, password: str, role: str = "member") -> dict:
    username = username.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{2,63}", username):
        raise ValueError("账号须为 3–64 位小写字母、数字或 _.-，以字母或数字开头")
    if role not in {"admin", "member"}:
        raise ValueError("角色仅支持 admin/member")
    password_hash = hash_password(password)
    with SessionLocal() as db:
        user = User(id=str(uuid.uuid4()), username=username, password_hash=password_hash, role=role)
        db.add(user)
        try:
            db.commit()
        except IntegrityError as exc:
            raise ValueError("账号已存在") from exc
        return public_user(user)


def token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


def csrf_token(token):
    return hashlib.sha256(("sizhen.csrf:" + token).encode()).hexdigest()


def login(username: str, password: str):
    with SessionLocal() as db:
        # 串行更新失败计数，防止并发尝试绕过锁定。
        db.connection().exec_driver_sql("BEGIN IMMEDIATE")
        user = db.scalar(select(User).where(User.username == username.strip().lower()))
        if user is None:
            # 与存在账号相同的密码派生工作量，避免快速枚举。
            hashlib.pbkdf2_hmac("sha256", password.encode(), b"unknown-account", ITERATIONS)
            raise ValueError("账号或密码错误，或账号暂不可用")
        if not user.active or (user.locked_until and user.locked_until > now()):
            raise ValueError("账号或密码错误，或账号暂不可用")
        if not verify_password(password, user.password_hash):
            user.failed_logins += 1
            if user.failed_logins >= 5:
                user.locked_until = now() + timedelta(minutes=15)
                user.failed_logins = 0
            db.commit()
            raise ValueError("账号或密码错误，或账号暂不可用")
        user.failed_logins = 0
        user.locked_until = None
        token = secrets.token_urlsafe(32)
        db.execute(delete(LoginSession).where(LoginSession.expires_at <= now()))
        db.add(LoginSession(token_hash=token_hash(token), user_id=user.id,
                            expires_at=now() + timedelta(seconds=config.SESSION_SECONDS)))
        db.commit()
        return public_user(user), token


def authenticate(token: str | None):
    if not token or len(token) > 128:
        return None
    with SessionLocal() as db:
        session = db.get(LoginSession, token_hash(token))
        if not session or session.expires_at <= now():
            return None
        user = db.get(User, session.user_id)
        return public_user(user) if user and user.active else None


def logout(token):
    with SessionLocal() as db:
        db.execute(delete(LoginSession).where(LoginSession.token_hash == token_hash(token)))
        db.commit()


def list_users():
    with SessionLocal() as db:
        return [public_user(user) for user in db.scalars(select(User).order_by(User.username))]


def update_user(user_id: str, *, active=None, password=None):
    password_hash = hash_password(password) if password is not None else None
    with SessionLocal() as db:
        db.connection().exec_driver_sql("BEGIN IMMEDIATE")
        user = db.get(User, user_id)
        if not user:
            raise ValueError("账号不存在")
        if active is False and user.role == "admin":
            admins = list(db.scalars(select(User).where(User.role == "admin", User.active.is_(True))))
            if user.active and len(admins) <= 1:
                raise ValueError("不能停用最后一个管理员")
        if active is not None:
            user.active = active
        if password_hash:
            user.password_hash = password_hash
            user.failed_logins = 0
            user.locked_until = None
        db.execute(delete(LoginSession).where(LoginSession.user_id == user_id))
        db.commit()
        return public_user(user)
