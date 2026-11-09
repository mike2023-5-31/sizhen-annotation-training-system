"""自定义模型上传路由（需求 6.7）：``POST /api/models/custom``。

multipart 字段 ``file``（必须为 .py），按文件内容摘要保存至 ``MODEL_DIR/custom``
并返回相对路径 ``{"model_path": "custom/<sha256>/<basename>.py"}``，
供训练配置的 ``custom_model_path`` 使用。

错误映射：非 .py 文件 / 空文件 → 400；文件名取 basename（防路径注入），
目标目录幂等创建。
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app import config
from app.services.xml_io import write_bytes_atomic

router = APIRouter(prefix="/models", tags=["models"])
_MAX_CUSTOM_MODEL_BYTES = 512 * 1024


@router.get("")
def list_checkpoints() -> dict[str, list[dict[str, object]]]:
    """模型仓库：列出 MODEL_DIR 内的 .pth/.pt 存档（相对路径/大小/修改时间）。"""
    root = config.MODEL_DIR.resolve()
    items: list[dict[str, object]] = []
    if root.is_dir():
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".pth", ".pt"}:
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                continue
            stat = resolved.stat()
            items.append({
                "model_path": resolved.relative_to(root).as_posix(),
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            })
    items.sort(key=lambda item: str(item["modified_at"]), reverse=True)
    return {"items": items[:200]}


@router.post("/custom")
async def upload_custom_model(file: UploadFile = File(...)) -> dict[str, str]:
    """上传自定义模型 .py → MODEL_DIR/custom/ 下，返回模型目录内相对路径。"""
    filename = Path((file.filename or "").replace("\\", "/")).name
    if Path(filename).suffix.lower() != ".py":
        raise HTTPException(status_code=400, detail="仅支持上传 .py 自定义模型文件")
    content = await file.read(_MAX_CUSTOM_MODEL_BYTES + 1)
    if len(content) > _MAX_CUSTOM_MODEL_BYTES:
        raise HTTPException(status_code=413, detail="自定义模型源码不得超过 512 KiB")
    if not content.strip():
        raise HTTPException(status_code=400, detail="上传的模型文件为空")

    root = config.MODEL_DIR.resolve()
    relative = Path("custom") / hashlib.sha256(content).hexdigest() / filename
    dest = root / relative
    if not dest.resolve().is_relative_to(root):
        raise HTTPException(status_code=400, detail="模型路径越出模型目录")
    await run_in_threadpool(write_bytes_atomic, dest, content)
    return {"model_path": relative.as_posix()}
