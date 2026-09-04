"""四诊标签体系加载器（T3）。

标签唯一真相源：`backend/config/labels.yaml`。
- `CATEGORIES`：list[{"name": str, "labels": list[str]}]，顺序即分类头顺序。
- `num_classes_per_head()`：4 个分类头的类别数（每类子标签数），供模型工厂（T12）构建 4 头分类器。

⚠️ 不 import `app.config`（config.py 由 T5 创建）；路径由环境变量
`LABELS_PATH` 覆盖，默认取本文件同级 `../config/labels.yaml`。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

_DEFAULT_LABELS_PATH = Path(__file__).resolve().parent.parent / "config" / "labels.yaml"


def _resolve_labels_path() -> Path:
    """LABELS_PATH 环境变量优先，缺省用仓库内 config/labels.yaml。"""
    return Path(os.getenv("LABELS_PATH", str(_DEFAULT_LABELS_PATH)))


def _load_categories(path: Path) -> List[Dict[str, Any]]:
    """读取并校验 YAML：4 类，每类首项必须为「正常/无」。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict) or "categories" not in raw:
        raise ValueError(f"标签配置缺少 categories 字段: {path}")

    categories: List[Dict[str, Any]] = []
    for item in raw["categories"]:
        name = str(item["name"])
        labels = [str(label) for label in item["labels"]]
        if not labels:
            raise ValueError(f"category「{name}」子标签列表为空: {path}")
        if labels[0] != "正常/无":
            raise ValueError(
                f"category「{name}」第一个标签必须为「正常/无」（当前为「{labels[0]}」）: {path}"
            )
        if len(set(labels)) != len(labels):
            raise ValueError(f"category「{name}」子标签存在重复: {labels}")
        categories.append({"name": name, "labels": labels})

    if len(categories) != 4:
        raise ValueError(f"标签配置必须恰好 4 个 category（舌/面/脉/声音），当前 {len(categories)} 个: {path}")
    return categories


# 模块导入时加载一次；标签表变更需重启后端生效。
CATEGORIES: List[Dict[str, Any]] = _load_categories(_resolve_labels_path())


def num_classes_per_head() -> List[int]:
    """每个分类头的类别数 = 对应 category 的子标签数，顺序与 CATEGORIES 一致。"""
    return [len(category["labels"]) for category in CATEGORIES]
