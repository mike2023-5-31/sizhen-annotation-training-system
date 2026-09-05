"""自定义模型上传路由（需求 6.7）：``POST /api/models/custom``。

multipart 字段 ``file``（必须为 .py），保存到 ``MODEL_DIR/custom/<basename>.py``
并返回相对路径 ``{"model_path": "custom/<basename>.py"}``（相对 MODEL_DIR），
供训练配置的 ``custom_model_path`` 使用。

错误映射：非 .py 文件 / 空文件 → 400；文件名取 basename（防路径注入），
目标目录幂等创建。
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app import config

router = APIRouter(prefix="/models", tags=["models"])


@router.post("/custom")
async def upload_custom_model(file: UploadFile = File(...)) -> dict[str, str]:
    """上传自定义模型 .py → MODEL_DIR/custom/ 下，返回模型目录内相对路径。"""
    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".py"):
        raise HTTPException(status_code=400, detail="仅支持上传 .py 自定义模型文件")
    content = await file.read()
    if not content.strip():
        raise HTTPException(status_code=400, detail="上传的模型文件为空")

    custom_dir = config.MODEL_DIR / "custom"
    custom_dir.mkdir(parents=True, exist_ok=True)
    dest = custom_dir / filename
    dest.write_bytes(content)
    return {"model_path": f"custom/{filename}"}
