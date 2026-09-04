"""图片浏览/检索路由（T9，需求 6.3）。

前缀 ``/datasets``，挂载于 ``main.api_router``（整体 ``/api`` 前缀）后对外为：

- ``GET /api/datasets/{dataset_id}/images`` → ``{"images": [filenames]}``（filename 升序）；
- ``GET /api/datasets/{dataset_id}/images/search?q=`` → 文件名大小写不敏感子串匹配；
- ``GET /api/datasets/{dataset_id}/images/{filename}`` → 图片二进制（``FileResponse``）；
- ``GET /api/datasets/{dataset_id}/image/{filename}/labels`` → 同名 XML 标注 dict。

错误映射：``ValueError``（数据集/图片/XML 不存在）→ 404。

路由顺序注意：``/images/search`` 必须先于 ``/images/{filename}`` 声明，
否则 "search" 会被当作 filename 吞掉。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.services import image_service

router = APIRouter(prefix="/datasets", tags=["images"])


@router.get("/{dataset_id}/images")
def list_images(dataset_id: int) -> dict[str, list[str]]:
    """数据集全部图片 filename（升序）。"""
    try:
        return {"images": image_service.list_images(dataset_id)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/{dataset_id}/images/search")
def search_images(dataset_id: int, q: str = "") -> dict[str, list[str]]:
    """文件名模糊搜索（大小写不敏感子串；q 为空返回全部）。"""
    try:
        return {"images": image_service.search_images(dataset_id, q)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/{dataset_id}/images/{filename}")
def get_image(dataset_id: int, filename: str) -> FileResponse:
    """图片二进制（路径来自 ``Record.image_path``）。"""
    try:
        path = image_service.get_image_path(dataset_id, filename)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"图片文件缺失: {filename}")
    return FileResponse(path)


@router.get("/{dataset_id}/image/{filename}/labels")
def get_labels(dataset_id: int, filename: str) -> dict:
    """读取图片同名 XML 标注。图片/XML 不存在 → 404。"""
    try:
        return image_service.read_labels(dataset_id, filename)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
