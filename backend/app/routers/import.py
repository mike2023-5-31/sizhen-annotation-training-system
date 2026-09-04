"""ZIP 导入路由（T7，需求 6.2a）。

``POST /api/datasets/{dataset_id}/import``（multipart，文件字段 ``file``）：

- 成功 → ``{"imported": N, "skipped": [filenames], "errors": [messages]}``；
- 非 ZIP / 损坏 ZIP（``BadZipFile``）→ 400；
- 数据集不存在（``ValueError``）→ 404。

注意：本模块文件名 ``import.py`` 与 Python 关键字同名，注册时由
``main.py`` 经 ``importlib.import_module`` 加载（见 main.py 注释）。
"""
from __future__ import annotations

from zipfile import BadZipFile

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services import import_service

router = APIRouter(prefix="/datasets", tags=["import"])


@router.post("/{dataset_id}/import")
async def import_zip(dataset_id: int, file: UploadFile = File(...)) -> dict:
    """上传 ZIP 导入图片/XML 对。汇总返回导入数、跳过文件与错误信息。"""
    zip_bytes = await file.read()
    try:
        return import_service.import_zip(dataset_id, zip_bytes)
    except BadZipFile:
        raise HTTPException(status_code=400, detail="不是有效的 ZIP 文件")
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
