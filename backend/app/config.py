"""应用配置：全部 env-driven，无硬编码绝对路径（锁定决策 10）。

本地 Windows 与远程服务器共用同一份代码：所有路径默认值由项目根锚点
(``backend/app/config.py`` -> app -> backend -> <project root>) 推导，
并可通过同名环境变量覆盖。环境变量为相对路径时按项目根解析。

env 变量一览：
    DATA_DIR    数据根目录（下含 datasets/ models/ exports/ uploads/），默认 <root>/data
    MODEL_DIR   模型保存目录，默认 <DATA_DIR>/models
    EXPORT_DIR  导出目录，默认 <DATA_DIR>/exports
    UPLOAD_DIR  ZIP 导入暂存目录，默认 <DATA_DIR>/uploads
    DB_URL      SQLAlchemy URL，默认 sqlite:///<root>/data/db.sqlite
    LABELS_PATH 标签体系 YAML 路径，默认 <root>/backend/config/labels.yaml
    CORS_ORIGINS 逗号分隔的允许来源，默认 "*"（开发期任意来源）
"""
from __future__ import annotations

import os
from pathlib import Path

# 项目根锚点：本文件位于 <root>/backend/app/config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_env_path(name: str, default: Path) -> Path:
    """读取环境变量作为路径；未设置/为空用默认值，相对路径按项目根解析。"""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    p = Path(raw)
    return p if p.is_absolute() else PROJECT_ROOT / p


# --- 目录（默认全部位于 <root>/data 之下）---
DATA_DIR = _resolve_env_path("DATA_DIR", PROJECT_ROOT / "data")
DATASETS_DIR = DATA_DIR / "datasets"
MODEL_DIR = _resolve_env_path("MODEL_DIR", DATA_DIR / "models")
EXPORT_DIR = _resolve_env_path("EXPORT_DIR", DATA_DIR / "exports")
UPLOAD_DIR = _resolve_env_path("UPLOAD_DIR", DATA_DIR / "uploads")

# --- 数据库 ---
# 默认 SQLite 落盘于 <root>/data/db.sqlite；Windows 绝对路径需正斜杠
_DEFAULT_DB_PATH = "sqlite:///" + (DATA_DIR / "db.sqlite").as_posix()
DB_URL = os.getenv("DB_URL", _DEFAULT_DB_PATH)

# --- 标签体系 ---
# 由 T3 在 <root>/backend/config/labels.yaml 落地
LABELS_PATH = _resolve_env_path(
    "LABELS_PATH", PROJECT_ROOT / "backend" / "config" / "labels.yaml"
)

# --- 静态文件（数据集图片服务根 = DATA_DIR/datasets，每数据集一个子目录）---
STATIC_DIR = DATASETS_DIR

# --- CORS ---
# 开发期默认 "*"（同时覆盖 http://localhost:5173）；生产用 CORS_ORIGINS 收紧
CORS_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()
]


def ensure_dirs() -> None:
    """确保数据目录树存在（StaticFiles 挂载要求目录已存在）。"""
    for d in (DATA_DIR, DATASETS_DIR, MODEL_DIR, EXPORT_DIR, UPLOAD_DIR):
        d.mkdir(parents=True, exist_ok=True)
