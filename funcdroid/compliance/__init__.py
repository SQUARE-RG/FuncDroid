"""App 合规性动态检测（GUI 探索 + 轨迹合规判定）。

规则来源：检测规则罗列表，经 GUI 动态可观测性筛选后写入
compliance/rules.json。检测点分两类：
- gui_direct：GUI 动态测试可直接判定（本工具核心能力）；
- gui_collect_llm：GUI 采集截图后由 LLM 做条款文本审查；
静态分析、流量监控或服务端审计专属条款不进入本规则库和报告。
"""

from compliance.rule_library import (
    CHECKPOINT_GROUP_ORDER,
    load_rules,
    pending_gui_checkpoints,
    rule_by_id,
)
from compliance.checkpoints import (
    CHECK_STATUS_PENDING,
    CHECK_STATUS_EXECUTING,
    CHECK_STATUS_DEFERRED,
    CHECK_STATUS_DONE,
    ComplianceChecklist,
    build_compliance_scenarios,
)
from compliance.report import build_compliance_report, save_compliance_report

__all__ = [
    "CHECKPOINT_GROUP_ORDER",
    "load_rules",
    "pending_gui_checkpoints",
    "rule_by_id",
    "CHECK_STATUS_PENDING",
    "CHECK_STATUS_EXECUTING",
    "CHECK_STATUS_DEFERRED",
    "CHECK_STATUS_DONE",
    "ComplianceChecklist",
    "build_compliance_scenarios",
    "build_compliance_report",
    "save_compliance_report",
]
