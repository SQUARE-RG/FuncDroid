"""官方问题到 20 项检测规则的相关问题对应关系。"""

from __future__ import annotations

import re


def normalize_issue(issue: str) -> str:
    return re.sub(r"\s+", "", str(issue)).removeprefix("APP")


OFFICIAL_PROBLEM_RULES = {
    "信息窗口无法关闭": ("P14",),
    "信息窗口点击乱跳转": ("P15",),
    "强制用户使用定向推送功能": ("P08",),
    "违规收集个人信息": ("P01", "P02", "P03", "P04", "P05", "P06", "P07", "P09"),
    "违规使用个人信息": ("P01", "P02", "P03", "P05", "P08", "P09", "P11"),
    "强制、频繁、过度索取权限": ("P06", "P07"),
    "频繁索要非必要权限": ("P06", "P07"),
    "强制或频繁索要非必要权限": ("P06", "P07"),
    "频繁自启动和关联启动": ("P19",),
    "违规提供自动续费服务": ("P16",),
    "欺骗误导用户下载APP": ("P15",),
    "超范围收集个人信息": ("P04",),
    "SDK信息公示不到位": ("P09",),
    "未公开个人信息收集使用规则": ("P01",),
    "未逐一列出收集使用个人信息的SDK": ("P09",),
    "未完整准确列明收集使用个人信息情况": ("P02", "P09"),
    "未提供有效账号注销功能": ("P10",),
}


def official_problem_rule_ids(issue: str) -> list[str]:
    """返回同类或相关问题编号；命中不等于还原官方的具体触发路径。"""
    return list(OFFICIAL_PROBLEM_RULES.get(normalize_issue(issue), ()))
