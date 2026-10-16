"""探索边界规则引擎（通用）。

允许为不同目标应用配置"探索边界限制"：当运行中进入目标应用内的
第三方服务内容（如小程序、内嵌网页、游戏中心等）时，自动退出并结束
当前场景，使探索聚焦在目标应用自身的核心功能上。

设计约束：
- 本模块不包含任何应用专属逻辑，规则全部来自外部 JSON 配置文件；
- 未配置规则的应用行为与未引入本模块时完全一致（默认零影响）；
- 规则文件路径由环境变量 ``BOUNDARY_RULES_FILE`` 指定，默认项目根目录
  下的 ``boundary_rules.json``；文件不存在时优雅降级为"全部关闭"。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

_RULES_CACHE: tuple[str | None, dict[str, Any]] | None = None
_EMPTY_RULES: dict[str, Any] = {}


@dataclass(frozen=True)
class BoundaryRule:
    """单个应用的边界规则。"""

    package_name: str
    enabled: bool = False
    label: str = "第三方服务内容"
    activity_markers: tuple[str, ...] = ()
    process_markers: tuple[str, ...] = ()
    exit_action: str = "press_back"

    def to_prompt_hint(self) -> str:
        """生成可注入规划提示词的通用约束文案。"""
        if not self.enabled:
            return ""
        return (
            f"本应用边界限制：不得规划进入{self.label}"
            "（如小程序、内嵌网页等第三方服务内容）；"
            "应用内入口可打开此类内容，但进入即视为离开目标应用边界。"
        )


@dataclass(frozen=True)
class BoundaryHit:
    """一次边界命中结果。"""

    rule: BoundaryRule
    via: str          # "activity" | "process"
    matched: str      # 命中的特征串
    evidence: str = ""  # 原始观测值（Activity 名或进程快照片段）


def _rules_file() -> str:
    return os.getenv(
        "BOUNDARY_RULES_FILE",
        os.path.join(os.getcwd(), "boundary_rules.json"),
    )


def _load_rules() -> tuple[str | None, dict[str, Any]]:
    """加载规则文件；带模块级缓存，文件缺失/损坏时降级为空规则。"""
    global _RULES_CACHE
    path = _rules_file()
    if _RULES_CACHE is not None and _RULES_CACHE[0] == path:
        return _RULES_CACHE
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        rules = raw if isinstance(raw, dict) else _EMPTY_RULES
    except FileNotFoundError:
        rules = _EMPTY_RULES
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[探索边界] 规则文件读取失败，已降级为不启用边界限制: {exc}")
        rules = _EMPTY_RULES
    _RULES_CACHE = (path, rules)
    return _RULES_CACHE


def _lower_markers(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(m).strip().lower() for m in value if str(m).strip())


def rule_for(package_name: str) -> BoundaryRule:
    """返回指定应用的边界规则；未配置时返回 disabled 规则。"""
    package_name = (package_name or "").strip()
    if not package_name:
        return BoundaryRule(package_name=package_name)
    _, rules = _load_rules()
    entry = rules.get(package_name)
    if not isinstance(entry, dict) or not entry:
        return BoundaryRule(package_name=package_name)
    enabled = bool(entry.get("enabled", False))
    if not enabled:
        return BoundaryRule(package_name=package_name)
    return BoundaryRule(
        package_name=package_name,
        enabled=True,
        label=str(entry.get("label") or "第三方服务内容").strip(),
        activity_markers=_lower_markers(entry.get("activity_markers")),
        process_markers=_lower_markers(entry.get("process_markers")),
        exit_action=str(entry.get("exit_action") or "press_back").strip(),
    )


def check_boundary(
    package_name: str,
    activity: str | None = None,
    process_snapshot: str | None = None,
) -> BoundaryHit | None:
    """检测当前是否越过目标应用探索边界。

    Args:
        package_name: 目标应用包名。
        activity: 当前前台 Activity 名（手段 1：Activity 名特征匹配）。
        process_snapshot: 可选进程快照（如 ``ps -A`` 输出），仅当
            activity 为空/不可读时才用于兜底判定。边界进程（如微信
            ``:appbrand``）退出后常驻，“进程存在”不代表“正在边界内”；
            读到 Activity 时以 Activity 为准，进入与退出判据保持一致。
            调用方还应确认前台包名是目标应用后再提供快照。

    Returns:
        命中时返回 :class:`BoundaryHit`，否则返回 None。
    """
    rule = rule_for(package_name)
    if not rule.enabled:
        return None

    activity = (activity or "").strip()
    if activity and rule.activity_markers:
        lowered = activity.lower()
        for marker in rule.activity_markers:
            if marker and marker in lowered:
                return BoundaryHit(rule=rule, via="activity", matched=marker, evidence=activity)

    if activity:
        return None
    snapshot = (process_snapshot or "").strip()
    if snapshot and rule.process_markers:
        lowered = snapshot.lower()
        for marker in rule.process_markers:
            if marker and marker in lowered:
                return BoundaryHit(rule=rule, via="process", matched=marker, evidence=snapshot[:300])
    return None


def enabled_packages() -> list[str]:
    """返回已启用边界规则的所有包名（供测试/诊断使用）。"""
    _, rules = _load_rules()
    return sorted(
        name for name, entry in rules.items()
        if isinstance(entry, dict) and entry.get("enabled")
    )


def reset_cache() -> None:
    """清空规则缓存（测试用）。"""
    global _RULES_CACHE
    _RULES_CACHE = None
