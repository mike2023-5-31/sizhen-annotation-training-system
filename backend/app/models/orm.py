"""ORM 模型：``Dataset`` / ``Record``（T4，锁定决策 11）。

范围边界（Must NOT）：
- **标签不入库**——标注的唯一真相源是同名 XML（xml_io），DB 仅镜像
  ``quality`` 供需求 6.2「按质量过滤」快速查询，避免双源漂移；
- **不做归一化 label 表**；删除语义 = 硬删除（无 ``deleted_at``，见锁定决策 7）。
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, String, JSON, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def _utcnow() -> datetime:
    """naive UTC 当前时间（SQLite 落盘干净、可跨行比较）。"""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Dataset(Base):
    """数据集：一次导入/创建的图片集合（一个数据目录 + 一份 XML 标注集）。"""

    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=_utcnow, onupdate=_utcnow
    )

    records: Mapped[list["Record"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )


class Record(Base):
    """单张图片记录：路径 + quality/timestamp 镜像（标签存于同目录同名 XML）。"""

    __tablename__ = "records"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    image_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    # 文件时间戳/拍摄时间与质量镜像：未知时允许 NULL（import 时缺省 '' 亦合法）
    timestamp: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    quality: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=_utcnow
    )

    dataset: Mapped[Dataset] = relationship(back_populates="records")


class AnnotationEdit(Base):
    """标注修改历史，业务删除后仍保留审计；不把未知操作者伪造为用户。"""

    __tablename__ = "annotation_edits"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    record_id: Mapped[int] = mapped_column(index=True)
    dataset_id: Mapped[int] = mapped_column(index=True)
    actor_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    previous_revision: Mapped[str] = mapped_column(String(64))
    revision: Mapped[str] = mapped_column(String(64))
    before: Mapped[dict] = mapped_column(JSON)
    after: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    role: Mapped[str] = mapped_column(String(16), default="member")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    failed_logins: Mapped[int] = mapped_column(default=0)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class LoginSession(Base):
    __tablename__ = "login_sessions"
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime)
