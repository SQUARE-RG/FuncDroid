"""合规检测点状态管理与场景生成。

检测点生命周期：
    pending → executing（规划器派发合规场景）
      → done（得到 compliant / violation / not_applicable 终态 verdict）
      → deferred（证据不足，等待 FFG 出现新的相关探索证据后重开）

重试上限只用于阻止同一批证据上的机械重复，不会再把 undetermined 伪装成
done。超过直接重试次数后，检测点仍保持 deferred；普通探索发现新的相关
页面、控件或路径后可以再次重开。

已得到 compliant / not_applicable 的 done 条款在相关 FFG 出现新范围时可以
复核，但复核次数单独计数（reverify_counts，上限 COMPLIANCE_MAX_REVERIFY）；
复核只得到 undetermined 时保留原终态结论，只把本次结果记入 history。

同一组（group）的检测点共享一次场景执行——例如 11 条 first_launch_flow 规则
在同一次冷启动观察中一起取证、一起判定；单条规则不单独发起设备操作。
"""

from __future__ import annotations

import os
from typing import Any

from compliance.rule_library import (
    CHECKPOINT_GROUP_ORDER,
    GROUP_LABELS,
    gui_rules,
    rule_by_id,
)

CHECK_STATUS_PENDING = "pending"
CHECK_STATUS_EXECUTING = "executing"
CHECK_STATUS_DEFERRED = "deferred"
CHECK_STATUS_DONE = "done"

MAX_COMPLIANCE_ATTEMPTS = max(
    1,
    int(os.getenv("COMPLIANCE_MAX_ATTEMPTS", "2") or 2),
)
# done 条款（compliant/not_applicable）因 FFG 新范围被重开复核的次数上限；
# 与 attempts 分开计数，避免 FFG 签名持续变化时无限重开。0 表示不复核。
MAX_COMPLIANCE_REVERIFY = max(
    0,
    int(os.getenv("COMPLIANCE_MAX_REVERIFY", "1") or 0),
)

VERDICT_COMPLIANT = "compliant"
VERDICT_VIOLATION = "violation"
VERDICT_UNDETERMINED = "undetermined"
VERDICT_NOT_APPLICABLE = "not_applicable"

VALID_VERDICTS = {
    VERDICT_COMPLIANT,
    VERDICT_VIOLATION,
    VERDICT_UNDETERMINED,
    VERDICT_NOT_APPLICABLE,
}
TERMINAL_VERDICTS = {VERDICT_COMPLIANT, VERDICT_VIOLATION, VERDICT_NOT_APPLICABLE}

GROUP_EXECUTION_REQUIREMENTS = {
    "first_launch_flow": (
        "【本组必须执行的分支】先查看可访问的协议/政策内容，然后必须实际点击一次"
        "“不同意/拒绝”并观察拒绝后的界面与应用可用性；仅看到拒绝按钮不算完成。"
        "若拒绝导致退出或回到启动页，可重新打开应用，在拒绝结果已经截图后再点击同意，"
        "以便继续观察后续首启界面。"
    ),
    "permission_denial_flow": (
        "【本组必须执行的分支】必须在真实系统权限弹窗中实际点击一次“拒绝/不允许”，"
        "随后再次尝试原功能并观察是否被限制或短时间内重复索权；还需实际执行一次不依赖被拒权限的基本功能并截图记录结果，回到首页或入口可见不代表功能可用；仅看到权限弹窗不算完成。"
    ),
}


def _copy_value(value: Any) -> Any:
    if isinstance(value, list):
        return [dict(item) if isinstance(item, dict) else item for item in value]
    if isinstance(value, dict):
        return dict(value)
    return value


def _copy_finding(finding: dict[str, Any]) -> dict[str, Any]:
    """复制一条结论（含 evidence/history 等嵌套结构），快照与 worker 不共享可变对象。"""
    copied = {key: _copy_value(value) for key, value in finding.items() if key != "history"}
    if "history" in finding:
        history = finding.get("history")
        copied["history"] = [
            {key: _copy_value(value) for key, value in entry.items()} if isinstance(entry, dict) else entry
            for entry in (history if isinstance(history, list) else [])
        ]
    return copied


def _int_map(values: dict[str, Any] | None) -> dict[str, int]:
    result: dict[str, int] = {}
    for rule_id, count in (values or {}).items():
        try:
            result[str(rule_id)] = max(0, int(count or 0))
        except (TypeError, ValueError):
            continue
    return result


class ComplianceChecklist:
    """合规检测点清单的内存表示与状态迁移。"""

    def __init__(
        self,
        status: dict[str, str] | None = None,
        findings: list[dict[str, Any]] | None = None,
        attempts: dict[str, int] | None = None,
        evidence_signatures: dict[str, str] | None = None,
        discovery_baselines: dict[str, str] | None = None,
        reverify_counts: dict[str, int] | None = None,
    ):
        self.status: dict[str, str] = dict(status or {})
        self.findings: list[dict[str, Any]] = [
            _copy_finding(finding) for finding in (findings or []) if isinstance(finding, dict)
        ]
        self.attempts: dict[str, int] = _int_map(attempts)
        # done 条款被 FFG 新范围重开复核的次数（与 attempts 分开计数）。
        self.reverify_counts: dict[str, int] = _int_map(reverify_counts)
        self.evidence_signatures: dict[str, str] = {
            str(rule_id): str(signature or "")
            for rule_id, signature in (evidence_signatures or {}).items()
        }
        # 达到直接重试上限后，先记录一次合规场景结束后的相关 FFG 基线；
        # 只有后续普通探索让该基线再次变化，才允许重开，避免原地循环。
        self.discovery_baselines: dict[str, str] = {
            str(rule_id): str(signature or "")
            for rule_id, signature in (discovery_baselines or {}).items()
        }

    # ------------------------------------------------------------------
    # 序列化（state / 报告）
    # ------------------------------------------------------------------
    def to_state(self) -> dict[str, Any]:
        # 逐条复制 findings（含 history 列表）：快照会被主线程 json.dumps，
        # 而 worker 线程随后可能继续更新清单。
        return {
            "status": dict(self.status),
            "findings": [_copy_finding(finding) for finding in self.findings],
            "attempts": dict(self.attempts),
            "evidence_signatures": dict(self.evidence_signatures),
            "discovery_baselines": dict(self.discovery_baselines),
            "reverify_counts": dict(self.reverify_counts),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any] | None) -> "ComplianceChecklist":
        state = state if isinstance(state, dict) else {}
        status = state.get("status") if isinstance(state.get("status"), dict) else {}
        findings = state.get("findings") if isinstance(state.get("findings"), list) else []
        attempts = state.get("attempts") if isinstance(state.get("attempts"), dict) else {}
        evidence_signatures = (
            state.get("evidence_signatures")
            if isinstance(state.get("evidence_signatures"), dict)
            else {}
        )
        discovery_baselines = (
            state.get("discovery_baselines")
            if isinstance(state.get("discovery_baselines"), dict)
            else {}
        )
        # 旧持久化格式没有 reverify_counts，按 0 次复核加载。
        reverify_counts = (
            state.get("reverify_counts")
            if isinstance(state.get("reverify_counts"), dict)
            else {}
        )
        checklist = cls(
            status=status,
            findings=findings,
            attempts=attempts,
            evidence_signatures=evidence_signatures,
            discovery_baselines=discovery_baselines,
            reverify_counts=reverify_counts,
        )
        # 兼容旧持久化格式：过去达到重试上限的 undetermined 也会被写成
        # done。恢复时统一迁移为 deferred，避免把证据不足当成已完成。
        for finding in checklist.findings:
            rule_id = str(finding.get("check_id", "") or "").strip()
            if (
                rule_id
                and str(finding.get("verdict", "") or "").strip() == VERDICT_UNDETERMINED
                and checklist.status.get(rule_id) == CHECK_STATUS_DONE
            ):
                checklist.attempts[rule_id] = max(
                    1,
                    int(checklist.attempts.get(rule_id, 0) or 0),
                )
                checklist.status[rule_id] = CHECK_STATUS_DEFERRED
                if checklist.attempts[rule_id] >= MAX_COMPLIANCE_ATTEMPTS:
                    checklist.discovery_baselines.setdefault(rule_id, "")
        return checklist

    # ------------------------------------------------------------------
    # 状态迁移
    # ------------------------------------------------------------------
    @staticmethod
    def _signature_for(
        rule_id: str,
        evidence_signature: str | dict[str, str],
    ) -> str:
        if isinstance(evidence_signature, dict):
            return str(evidence_signature.get(rule_id, "") or "")
        return str(evidence_signature or "")

    def mark_executing(
        self,
        rule_ids: list[str],
        evidence_signature: str | dict[str, str] = "",
    ) -> None:
        for rule_id in rule_ids:
            if str(self.status.get(rule_id, "")).strip() != CHECK_STATUS_DONE:
                self.attempts[rule_id] = int(self.attempts.get(rule_id, 0) or 0) + 1
                self.evidence_signatures[rule_id] = self._signature_for(
                    rule_id,
                    evidence_signature,
                )
                self.discovery_baselines.pop(rule_id, None)
                self.status[rule_id] = CHECK_STATUS_EXECUTING

    def apply_results(self, results: list[dict[str, Any]]) -> None:
        """合并检测 worker 的判定结果（按 rule_id 幂等，后写覆盖前写）。"""
        for item in results if isinstance(results, list) else []:
            if not isinstance(item, dict):
                continue
            rule_id = str(item.get("check_id", "") or item.get("rule_id", "")).strip()
            verdict = str(item.get("verdict", "")).strip().lower()
            if not rule_id or verdict not in VALID_VERDICTS:
                continue
            attempt = max(1, int(self.attempts.get(rule_id, 0) or 0))
            self.attempts[rule_id] = attempt
            previous = next((finding for finding in self.findings if str(finding.get("check_id", "")) == rule_id), None)
            previous_verdict = str(previous.get("verdict", "") or "") if previous else ""
            if previous_verdict in TERMINAL_VERDICTS and (
                verdict == VERDICT_UNDETERMINED
                or (previous_verdict == VERDICT_VIOLATION and verdict != VERDICT_VIOLATION)
            ):
                # 复核只得到 undetermined 不能推翻已有终态结论；通过的路径也不能
                # 抹掉此前观察到的违规。保留原结论与 done 状态，只追加历史。
                # 用新字典替换而不是原地改 previous：快照可能仍引用旧对象。
                entry = {
                    key: _copy_value(value)
                    for key, value in item.items()
                    if key != "history"
                }
                entry.update({"check_id": rule_id, "verdict": verdict, "attempt": attempt})
                kept = _copy_finding(previous)
                kept["history"] = list(kept.get("history", [])) + [entry]
                self.findings = [kept if finding is previous else finding for finding in self.findings]
                self.status[rule_id] = CHECK_STATUS_DONE
                self.discovery_baselines.pop(rule_id, None)
                continue
            if verdict == VERDICT_UNDETERMINED:
                self.status[rule_id] = CHECK_STATUS_DEFERRED
                if attempt >= MAX_COMPLIANCE_ATTEMPTS:
                    # 空串表示规划器尚未记录本次合规场景完成后的相关 FFG
                    # 基线。记录基线本身不触发第三次机械复判。
                    self.discovery_baselines[rule_id] = ""
            else:
                self.status[rule_id] = CHECK_STATUS_DONE
                self.discovery_baselines.pop(rule_id, None)
            history = list(_copy_finding(previous).get("history", [])) if previous else []
            if previous:
                history.append({key: _copy_value(value) for key, value in previous.items() if key != "history"})
            self.findings = [
                finding
                for finding in self.findings
                if str(finding.get("check_id", "")).strip() != rule_id
            ]
            self.findings.append(
                {
                    "check_id": rule_id,
                    "verdict": verdict,
                    "description": str(item.get("description", "")).strip(),
                    "evidence": _copy_value(item.get("evidence")) if isinstance(item.get("evidence"), list) else [],
                    "attempt": attempt,
                    "history": history,
                    "scope": "observed_paths_only",
                    **({"applicability": dict(item["applicability"])}
                       if isinstance(item.get("applicability"), dict) else {}),
                    **({"trace_path": item["trace_path"]} if item.get("trace_path") else {}),
                    **({"functionality_evidence": dict(item["functionality_evidence"])}
                       if isinstance(item.get("functionality_evidence"), dict) else {}),
                }
            )

    def reopen_deferred(
        self,
        evidence_signature: str | dict[str, str],
        *,
        force: bool = False,
        rules_path: Any = None,
    ) -> list[str]:
        """新 FFG 证据出现后，把可重试的 undetermined 条款恢复为 pending。"""
        reopened: list[str] = []
        valid_ids = {rule["rule_id"] for rule in gui_rules(rules_path)}
        for rule_id, status in list(self.status.items()):
            if rule_id not in valid_ids:
                continue
            if status == CHECK_STATUS_DONE:
                finding = next((item for item in self.findings if item.get("check_id") == rule_id), {})
                if finding.get("verdict") not in {VERDICT_COMPLIANT, VERDICT_NOT_APPLICABLE}:
                    continue
                # 复核次数有上限：FFG 签名在长跑中几乎每轮都会变化，不设上限会让
                # 已完成条款被无限重开。
                if int(self.reverify_counts.get(rule_id, 0) or 0) >= MAX_COMPLIANCE_REVERIFY:
                    continue
                signature = self._signature_for(rule_id, evidence_signature)
                if not signature:
                    continue
                baseline = self.discovery_baselines.get(rule_id)
                if not baseline:
                    self.discovery_baselines[rule_id] = signature
                    continue
                if baseline != signature:
                    self.status[rule_id] = CHECK_STATUS_PENDING
                    self.discovery_baselines.pop(rule_id, None)
                    self.reverify_counts[rule_id] = int(self.reverify_counts.get(rule_id, 0) or 0) + 1
                    reopened.append(rule_id)
                continue
            if status != CHECK_STATUS_DEFERRED:
                continue
            current_signature = self._signature_for(rule_id, evidence_signature)
            attempt = int(self.attempts.get(rule_id, 0) or 0)
            discovery_changed = False
            if attempt >= MAX_COMPLIANCE_ATTEMPTS:
                baseline = self.discovery_baselines.get(rule_id)
                if baseline is None or baseline == "":
                    # 第一次看到合规场景执行后的图谱时只建立基线，不重开。
                    self.discovery_baselines[rule_id] = current_signature
                    continue
                if baseline == current_signature:
                    continue
                # 只有达到上限后的相关 FFG 又出现新变化，才说明普通探索
                # 找到了新入口/控件/路径，允许继续闭环。
                self.discovery_baselines.pop(rule_id, None)
                discovery_changed = True
            else:
                previous_signature = str(self.evidence_signatures.get(rule_id, "") or "")
                if not force and previous_signature == current_signature:
                    continue
            if force and attempt >= MAX_COMPLIANCE_ATTEMPTS and not discovery_changed:
                # force 只提供达到直接重试预算前的最后机会，不能绕过发现门槛。
                continue
            self.status[rule_id] = CHECK_STATUS_PENDING
            reopened.append(rule_id)
        return reopened

    def recover_interrupted(self) -> list[str]:
        """进程恢复时把没有后台任务承接的 executing 条款退回 pending。"""
        recovered: list[str] = []
        for rule_id, status in list(self.status.items()):
            if status != CHECK_STATUS_EXECUTING:
                continue
            self.status[rule_id] = CHECK_STATUS_PENDING
            self.attempts[rule_id] = max(
                0,
                int(self.attempts.get(rule_id, 0) or 0) - 1,
            )
            self.evidence_signatures.pop(rule_id, None)
            self.discovery_baselines.pop(rule_id, None)
            recovered.append(rule_id)
        return recovered

    def release_scheduled(self, rule_ids: list[str]) -> list[str]:
        """把因基础设施原因（队列拒收 / 场景执行异常）未能真正执行判定的
        executing 检测点退回 pending，并归还 mark_executing 消耗的尝试预算。

        M3：这条路径此前走 apply_results(undetermined)，与"模型正常判定但
        证据不足"共用同一套预算消耗逻辑——语义不对称：recover_interrupted
        （进程重启）明确不消耗预算，defer_checks（排队/执行失败）却消耗。
        一次基础设施故障就烧掉 MAX_COMPLIANCE_ATTEMPTS 的一半，第二次失败
        即永久停在 undetermined。这里与 recover_interrupted 对齐：没有产生
        真实判定，就不该计入重试预算。
        """
        released: list[str] = []
        for rule_id in rule_ids:
            if self.status.get(rule_id) != CHECK_STATUS_EXECUTING:
                continue
            self.status[rule_id] = CHECK_STATUS_PENDING
            self.attempts[rule_id] = max(
                0,
                int(self.attempts.get(rule_id, 0) or 0) - 1,
            )
            self.evidence_signatures.pop(rule_id, None)
            self.discovery_baselines.pop(rule_id, None)
            released.append(rule_id)
        return released

    def deferred_rules(self, rules_path: Any = None) -> list[dict[str, Any]]:
        return [
            rule
            for rule in gui_rules(rules_path)
            if self.status.get(rule["rule_id"]) == CHECK_STATUS_DEFERRED
        ]

    def pending_rules(self, rules_path: Any = None) -> list[dict[str, Any]]:
        from compliance.rule_library import pending_gui_checkpoints

        return pending_gui_checkpoints(self.status, rules_path)

    def next_group_rules(self, rules_path: Any = None) -> list[dict[str, Any]]:
        """返回下一个待执行检测组内的全部规则（同组一次取证）。

        调度采用“尝试次数少者优先，固定组顺序次之”。因此某组反复证据
        不足并在发现新线索后重开时，会排在尚未测试或重试次数更少的组后面，
        避免早期组长期占用执行权。同组规则仍合并取证，不为单条规则重复操作。
        """
        pending = self.pending_rules(rules_path)
        if not pending:
            return []
        grouped: dict[str, list[dict[str, Any]]] = {}
        for rule in pending:
            group = str(rule.get("checkpoint", {}).get("group", "") or "")
            # 没有 group 的自定义规则不能被意外合并，按 rule_id 单独调度。
            key = group or f"__rule__:{rule['rule_id']}"
            grouped.setdefault(key, []).append(rule)

        group_rank = {
            group: rank for rank, group in enumerate(CHECKPOINT_GROUP_ORDER)
        }

        def scheduling_key(item: tuple[str, list[dict[str, Any]]]) -> tuple[int, int, str]:
            key, rules = item
            # 场景能同时覆盖组内全部规则，只要组内还有低尝试规则就值得先跑；
            # 当整组都已多次失败时，最小值也会升高，从而自然降权。
            retry_level = min(
                int(self.attempts.get(str(rule.get("rule_id", "")), 0) or 0)
                for rule in rules
            )
            group = str(rules[0].get("checkpoint", {}).get("group", "") or "")
            return retry_level, group_rank.get(group, 99), key

        _selected_key, selected_rules = min(grouped.items(), key=scheduling_key)
        return selected_rules

    def summary(self, rules_path: Any = None) -> dict[str, int]:
        total = resolved = processed = violation = compliant = undetermined = 0
        not_applicable = deferred = executing = 0
        findings_by_id = {
            str(item.get("check_id", "") or "").strip(): item
            for item in self.findings
            if isinstance(item, dict)
        }
        for rule in gui_rules(rules_path):
            total += 1
            rule_id = rule["rule_id"]
            status = str(self.status.get(rule_id, "")).strip()
            if status == CHECK_STATUS_DEFERRED:
                deferred += 1
            elif status == CHECK_STATUS_EXECUTING:
                executing += 1
            if status == CHECK_STATUS_DONE:
                resolved += 1
            finding = findings_by_id.get(rule_id)
            verdict = str(finding.get("verdict", "") or "").strip() if finding else ""
            if verdict not in VALID_VERDICTS:
                continue
            processed += 1
            if verdict == VERDICT_VIOLATION:
                violation += 1
            elif verdict == VERDICT_COMPLIANT:
                compliant += 1
            elif verdict == VERDICT_NOT_APPLICABLE:
                not_applicable += 1
            else:
                undetermined += 1
        return {
            "gui_total": total,
            # executed 保留为 JSON 兼容别名；界面和日志统一使用 processed。
            "executed": processed,
            "processed": processed,
            "resolved": resolved,
            "compliant": compliant,
            "violation": violation,
            "undetermined": undetermined,
            "not_applicable": not_applicable,
            "deferred": deferred,
            "unresolved": undetermined,
            "executing": executing,
            "untested": max(0, total - processed),
            "pending": max(0, total - processed),
        }


# ----------------------------------------------------------------------
# 场景生成（确定性，不经过 LLM——LLM 只负责执行与判定）
# ----------------------------------------------------------------------
def build_compliance_scenarios(rules: list[dict[str, Any]]) -> tuple[str, str, dict[str, Any]]:
    """为一组同组检测点生成 (navigation_scenario, test_scenario, plan_extra)。

    返回的 plan_extra 会并入 current_plan：
    - compliance_check_ids: 本场景覆盖的规则编号；
    - compliance_cold_start: 是否需要运行器先清数据冷启动。
    """
    valid_rules = [rule for rule in rules if rule.get("checkpoint")]
    if not valid_rules:
        raise ValueError("build_compliance_scenarios 需要至少一个带检测点的规则")
    checkpoints = [rule["checkpoint"] for rule in valid_rules]
    group = str(checkpoints[0].get("group", "") or "compliance")
    label = GROUP_LABELS.get(group, group)
    rule_ids = [rule["rule_id"] for rule in valid_rules]

    clause_lines = []
    for rule in valid_rules:
        hints = "；".join(rule["checkpoint"].get("verdict_hints") or [])
        clause_lines.append(f"- {rule['rule_id']}（{rule['regulation']}）：{rule['clause']}" + (f"｜关注：{hints}" if hints else ""))
    clause_text = "\n".join(clause_lines)

    cold_start = all(bool(cp.get("cold_start")) for cp in checkpoints)

    # 合规场景从当前页面直接开始。cold_start 只表示在 test 阶段前重启
    # 进程；应用数据只允许由主程序在整次新运行开始时清除一次。
    # 不需要独立导航阶段；navigation_scenario 留空让执行器跳过导航。
    navigation_scenario = ""
    objectives = list(
        dict.fromkeys(
            str(checkpoint.get("objective", "") or "").strip()
            for checkpoint in checkpoints
            if str(checkpoint.get("objective", "") or "").strip()
        )
    )
    objective_text = "\n".join(f"- {objective}" for objective in objectives)
    group_requirement = GROUP_EXECUTION_REQUIREMENTS.get(group, "")
    test_scenario = (
        f"【合规检测｜{label}】取证目标：\n{objective_text}\n"
        f"需要对照的检测条款：\n{clause_text}\n"
        + (f"{group_requirement}\n" if group_requirement else "")
        +
        "执行要求：按上述目标逐项取证，每一步操作前先让截图完整呈现当前界面；"
        "涉及系统权限弹窗或应用内同意弹窗时，先截图记录弹窗原文与按钮布局，"
        "再按目标要求点击（拒绝/仅必要/同意）。只测试当前 App 实际存在且可安全到达的业务；"
        "不要为了覆盖不适用的行业或功能条款虚构入口，也不要执行注销、删除、支付等不可逆操作；"
        "严禁点击“退出登录/登出/切换账号”，也不得仅为查看登录或注册表单而先退出登录——"
        "运行器没有账号凭据，退出后无法恢复，会使本次测试后续所有依赖登录态的取证全部失效；"
        "登录/注册表单确实不可达时，直接按“目标不可达”结束本场景。"
        "已完成可达目标或确认目标不适用后结束。"
    )
    plan_extra = {
        "compliance_check_ids": rule_ids,
        "compliance_group": group,
        "compliance_cold_start": bool(cold_start),
    }
    return navigation_scenario, test_scenario, plan_extra


def check_rules_by_ids(rule_ids: list[str], rules_path: Any = None) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for rule_id in rule_ids or []:
        rule = rule_by_id(str(rule_id), rules_path)
        if rule is not None:
            rules.append(rule)
    return rules
