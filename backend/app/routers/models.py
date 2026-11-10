"""模型仓库：checkpoint 列表 + 自定义模型上传（6.7 / 6.14）。"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app import config
from app.services import checkpoint
from app.services.xml_io import write_bytes_atomic

router = APIRouter(prefix="/models", tags=["models"])

_MAX_CUSTOM_BYTES = 512 * 1024


@router.get("")
def list_checkpoints() -> dict[str, list[dict[str, Any]]]:
    root = config.MODEL_DIR.resolve()
    items: list[dict[str, Any]] = []
    if root.is_dir():
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".pth", ".pt"}:
                continue
            stat = path.stat()
            items.append(
                {
                    "model_path": path.relative_to(root).as_posix(),
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                }
            )
    items.sort(key=lambda x: str(x["modified_at"]), reverse=True)
    return {"items": items[:200]}


@router.post("/custom")
async def upload_custom_model(file: UploadFile = File(...)) -> dict[str, str]:
    filename = Path((file.filename or "").replace("\\", "/")).name
    if Path(filename).suffix.lower() != ".py":
        raise HTTPException(status_code=400, detail="仅支持上传 .py 自定义模型文件")
    content = await file.read(_MAX_CUSTOM_BYTES + 1)
    if len(content) > _MAX_CUSTOM_BYTES:
        raise HTTPException(status_code=413, detail="自定义模型源码不得超过 512 KiB")
    if not content.strip():
        raise HTTPException(status_code=400, detail="上传的模型文件为空")

    digest = hashlib.sha256(content).hexdigest()
    root = config.MODEL_DIR.resolve()
    relative = Path("custom") / digest / filename
    destination = root / relative
    if destination.resolve().parent != (root / "custom" / digest).resolve() and \
            not destination.resolve().is_relative_to(root):
        raise HTTPException(status_code=400, detail="模型路径越出模型目录")
    await run_in_threadpool(write_bytes_atomic, destination, content)
    return {"model_path": relative.as_posix()}
