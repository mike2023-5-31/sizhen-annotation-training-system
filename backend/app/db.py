"""SQLAlchemy engine / Session / 声明式基类（T4）。

数据库地址唯一来源：``app.config.DB_URL``（默认 ``sqlite:///<root>/data/db.sqlite``，
env-driven，见 config.py 与锁定决策 10）——本模块不重新定义。

建表职责不在本模块：表结构定义于 ``app.models.orm``，由调用方显式执行
``Base.metadata.create_all(engine)`` 或后续迁移工具创建。
"""
from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import DB_URL

# SQLite + FastAPI（多线程）需允许跨线程复用同一连接；
# 每个请求经 get_db() 取独立 Session，连接归还连接池。
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})

if engine.dialect.name == "sqlite":

    @event.listens_for(engine, "connect")
    def _enable_sqlite_fk(dbapi_conn, _record):  # pragma: no cover - trivial
        """SQLite 默认不强制外键，逐连接开启（FK 违约 → IntegrityError）。"""
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """ORM 声明式基类：所有模型（Dataset/Record）继承本类注册元数据。"""


def get_db() -> Generator[Session, None, None]:
    """FastAPI 依赖生成器：每请求一个 Session，请求结束即关闭。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
