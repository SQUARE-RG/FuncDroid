"""规则库加载与检测点过滤。"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_RULES_PATH = Path(__file__).resolve().parent / "rules.json"
LEGACY_RULES_PATH = Path(__file__).resolve().parent / "legacy_rules_v1.json"

# 检测点组的调度优先级：冷启动流程最先（越早清数据代价越小，且登录态尚未建立），
# 其次权限拒绝路径（会破坏状态，需要尽早），再是表单/入口检索类（依赖已建立 FFG）。
CHECKPOINT_GROUP_ORDER = (
    "first_launch_flow",
    "permission_denial_flow",
    "consent_observation",
    "registration_form",
    "settings_search",
    "rights_entry_search",
    "policy_text_review",
)

GROUP_LABELS = {
    "first_launch_flow": "首次启动与同意流程",
    "permission_denial_flow": "权限拒绝路径",
    "consent_observation": "敏感信息单独同意",
    "registration_form": "注册表单最小必要",
    "settings_search": "设置公示与开关",
    "rights_entry_search": "用户权利入口",
    "policy_text_review": "隐私政策文本审查",
}

GUI_DETECTABILITY = {"gui_direct", "gui_collect_llm"}


def _normalize_rules(raw: Any) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        rule_id = str(item.get("rule_id", "")).strip()
        clause = str(item.get("clause", "")).strip()
        if not rule_id or not clause:
            continue
        rules.append(item)
    return rules


@lru_cache(maxsize=4)
def _load_rules_cached(path: str) -> tuple[dict[str, Any], ...]:
    return tuple(_load_rules_uncached(path))


def _load_rules_uncached(path: str) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"合规规则库不存在: {target}")
    with target.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return _normalize_rules(data.get("rules"))


def load_rules(rules_path: str | os.PathLike[str] | None = None) -> list[dict[str, Any]]:
    """加载规则库；同一进程内重复加载走缓存。"""
    path = str(rules_path) if rules_path else str(DEFAULT_RULES_PATH)
    return list(_load_rules_cached(path))


def rule_by_id(rule_id: str, rules_path: str | os.PathLike[str] | None = None) -> dict[str, Any] | None:
    rule_id = str(rule_id or "").strip()
    for rule in load_rules(rules_path):
        if rule["rule_id"] == rule_id:
            return rule
    # 离线复核仍可读取历史 R 编号；新运行的 gui_rules 只调度当前 P01-P20。
    if rules_path is None and rule_id.startswith("R"):
        for rule in load_rules(LEGACY_RULES_PATH):
            if rule["rule_id"] == rule_id:
                return rule
    return None


def gui_rules(rules_path: str | os.PathLike[str] | None = None) -> list[dict[str, Any]]:
    """可由 GUI 动态测试覆盖的规则（gui_direct + gui_collect_llm）。"""
    return [
        rule
        for rule in load_rules(rules_path)
        if rule.get("detectability") in GUI_DETECTABILITY
    ]


def pending_gui_checkpoints(
    checklist_status: dict[str, str] | None,
    rules_path: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """按组优先级返回尚未执行过的 GUI 检测点。

    checklist_status: {rule_id: status}；status 见 compliance.checkpoints。
    跳过已终态（done）、执行中（executing）与等待新证据（deferred）的检测点；
    deferred 只能由闭环调度器在 FFG 证据变化后显式重开。
    """
    from compliance.checkpoints import (
        CHECK_STATUS_DEFERRED,
        CHECK_STATUS_DONE,
        CHECK_STATUS_EXECUTING,
    )

    status_map = checklist_status if isinstance(checklist_status, dict) else {}
    pending: list[dict[str, Any]] = []
    for rule in gui_rules(rules_path):
        status = str(status_map.get(rule["rule_id"], "") or "").strip()
        if status in {CHECK_STATUS_DONE, CHECK_STATUS_EXECUTING, CHECK_STATUS_DEFERRED}:
            continue
        pending.append(rule)
    group_rank = {group: rank for rank, group in enumerate(CHECKPOINT_GROUP_ORDER)}
    pending.sort(
        key=lambda rule: (
            group_rank.get(str(rule.get("checkpoint", {}).get("group", "")), 99),
            rule["rule_id"],
        )
    )
    return pending
