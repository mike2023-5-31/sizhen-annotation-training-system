"""标签体系配置测试（T3）：CATEGORIES 结构、num_classes_per_head、坏 YAML 校验。

覆盖锁定决策 1 的全部约束：
- 恰好 4 个 category（舌/面/脉/声音），对应 4 个分类头；
- 每个 category 第一个标签固定为「正常/无」；
- 同 category 内子标签不得重复；
- 违反任一约束 → 模块加载（或 _load_categories）抛 ValueError。
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

import app.labels_config as labels_config
from app.labels_config import CATEGORIES, _load_categories, num_classes_per_head


def _yaml_text(categories: list[dict]) -> str:
    return yaml.safe_dump({"categories": categories}, allow_unicode=True, sort_keys=False)


# ---------------------------------------------------------------------------
# 真实配置（backend/config/labels.yaml）结构断言
# ---------------------------------------------------------------------------


def test_categories_structure() -> None:
    """CATEGORIES 恰好 4 类（舌/面/脉/声音），每类首项「正常/无」，无重复子标签。"""
    assert len(CATEGORIES) == 4
    assert [c["name"] for c in CATEGORIES] == ["舌", "面", "脉", "声音"]
    for category in CATEGORIES:
        assert category["labels"], f"category「{category['name']}」子标签为空"
        assert category["labels"][0] == "正常/无"
        assert len(set(category["labels"])) == len(category["labels"])


def test_num_classes_per_head_lengths() -> None:
    """num_classes_per_head() == 各 category 子标签数，且与占位体系 [6, 4, 5, 4] 一致。"""
    assert num_classes_per_head() == [len(c["labels"]) for c in CATEGORIES]
    assert num_classes_per_head() == [6, 4, 5, 4]
    assert len(num_classes_per_head()) == 4  # 恰好 4 个分类头


# ---------------------------------------------------------------------------
# 坏 YAML → ValueError（直接调用加载函数，逐条覆盖锁定决策 1）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_yaml,pattern",
    [
        # 根不是 dict / 缺少 categories 字段
        (yaml.safe_dump({"foo": "bar"}, allow_unicode=True), "categories"),
        (yaml.safe_dump([1, 2], allow_unicode=True), "categories"),
        # 第一个标签不是「正常/无」
        (_yaml_text([{"name": "舌", "labels": ["舌苔黄"]}]), "第一个标签"),
        # 子标签重复
        (
            _yaml_text([{"name": "舌", "labels": ["正常/无", "舌苔黄", "舌苔黄"]}]),
            "重复",
        ),
        # 子标签列表为空
        (_yaml_text([{"name": "舌", "labels": []}]), "为空"),
        # 只有 3 个 category
        (
            _yaml_text(
                [
                    {"name": "舌", "labels": ["正常/无"]},
                    {"name": "面", "labels": ["正常/无"]},
                    {"name": "脉", "labels": ["正常/无"]},
                ]
            ),
            "恰好 4 个",
        ),
        # 5 个 category
        (
            _yaml_text(
                [
                    {"name": "舌", "labels": ["正常/无"]},
                    {"name": "面", "labels": ["正常/无"]},
                    {"name": "脉", "labels": ["正常/无"]},
                    {"name": "声音", "labels": ["正常/无"]},
                    {"name": "耳", "labels": ["正常/无"]},
                ]
            ),
            "恰好 4 个",
        ),
    ],
)
def test_load_categories_bad_yaml_raises(
    tmp_path: Path, bad_yaml: str, pattern: str
) -> None:
    """各类非法标签 YAML → _load_categories 抛 ValueError（锁定决策 1）。"""
    p = tmp_path / "labels.yaml"
    p.write_text(bad_yaml, encoding="utf-8")
    with pytest.raises(ValueError, match=pattern):
        _load_categories(p)


# ---------------------------------------------------------------------------
# 模块加载路径（LABELS_PATH 覆盖 + importlib.reload）
# ---------------------------------------------------------------------------


def test_module_load_bad_yaml_raises_and_state_intact(tmp_path: Path, monkeypatch) -> None:
    """模块级加载坏 YAML → ValueError，且失败不破坏已加载的 CATEGORIES。"""
    bad = tmp_path / "bad.yaml"
    bad.write_text("not: valid\n", encoding="utf-8")
    monkeypatch.setenv("LABELS_PATH", str(bad))
    with pytest.raises(ValueError, match="categories"):
        importlib.reload(labels_config)
    # 失败的 reload 不替换既有属性 → 默认 4 类体系仍可用
    assert len(CATEGORIES) == 4

    # 恢复默认配置路径（防跨测试污染）
    monkeypatch.setenv("LABELS_PATH", str(labels_config._DEFAULT_LABELS_PATH))
    importlib.reload(labels_config)
    assert [c["name"] for c in CATEGORIES] == ["舌", "面", "脉", "声音"]


def test_module_reload_custom_valid_yaml(tmp_path: Path, monkeypatch) -> None:
    """LABELS_PATH 指向合法自定义 YAML → reload 后 CATEGORIES 与头数随之更新。"""
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        _yaml_text(
            [
                {"name": "舌", "labels": ["正常/无", "A", "B"]},
                {"name": "面", "labels": ["正常/无", "C"]},
                {"name": "脉", "labels": ["正常/无"]},
                {"name": "声音", "labels": ["正常/无", "D", "E"]},
            ]
        ),
        encoding="utf-8",
    )
    try:
        monkeypatch.setenv("LABELS_PATH", str(custom))
        importlib.reload(labels_config)
        assert [c["name"] for c in CATEGORIES] == ["舌", "面", "脉", "声音"]
        assert num_classes_per_head() == [3, 2, 1, 3]
    finally:
        # 无论断言成败，恢复默认配置（后续测试依赖真实 labels.yaml）
        monkeypatch.setenv("LABELS_PATH", str(labels_config._DEFAULT_LABELS_PATH))
        importlib.reload(labels_config)
    assert num_classes_per_head() == [6, 4, 5, 4]
