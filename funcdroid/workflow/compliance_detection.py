"""执行轨迹合规判定。

与 trace_defect_detection 平行的管线：输入一次执行轨迹 + 关联的合规检测点，
输出每个检测点的 verdict（compliant/violation/undetermined）。
截图直接复用采集阶段的统一高清编码，避免二次缩放和有损压缩：合规证据要读取
弹窗小字，且 COORDINATE_CONTEXT 声明的执行器图片尺寸必须与复核图片一致。
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from compliance.checkpoints import (
    VALID_VERDICTS,
    VERDICT_COMPLIANT,
    VERDICT_NOT_APPLICABLE,
    VERDICT_UNDETERMINED,
    VERDICT_VIOLATION,
    check_rules_by_ids,
)
from workflow.llm import invoke_llm_json_with_retry, llm_json_result_failed
from workflow.nodes._shared import JSON_BLOCK_PATTERN
from workflow.nodes.prompt import COMPLIANCE_DETECT_SYSTEM_PROMPT
from workflow.trace_messages import (
    build_step_text,
    coerce_step,
    image_block,
    image_ref_text,
    normalize_evidence_items,
    raw_step_index,
    step_screenshot,
)
from tools.actions import _screenshot_health_error
from tools.utils import decode_screenshot


class ComplianceReviewError(RuntimeError):
    """判定模型调用/解析在重试耗尽后仍未成功，与"模型正常返回但漏答某条款"区分开。"""


TERMINAL_EVIDENCE_VERDICTS = {VERDICT_COMPLIANT, VERDICT_VIOLATION, VERDICT_NOT_APPLICABLE}


def _collect_steps(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """合规取证保留全部步骤（含弹窗/恢复步骤），只要求带页面观测。"""
    steps: list[dict[str, Any]] = []
    for raw_index, step in enumerate(trace, start=1):
        if not isinstance(step, dict):
            continue
        if not step_screenshot(step):
            continue
        copied = dict(step)
        copied["raw_step_index"] = raw_index
        steps.append(copied)
    return steps


def _compact_screenshot(screenshot: str) -> str:
    """复用执行时的 JPEG（只剥离 data URL 前缀）；不再继承 FFG 参数或重复压缩证据图片。

    这里刻意不走 workflow.image_compact：缩放会让复核图片与执行器图片尺寸不一致。
    """
    screenshot = str(screenshot or "").strip()
    if screenshot.startswith("data:image") and "," in screenshot:
        screenshot = screenshot.split(",", 1)[1]
    return screenshot


def _downgrade_without_evidence(normalized: dict[str, Any], reason: str) -> dict[str, Any]:
    """终态结论没有有效截图证据时降级为 undetermined（原地更新并返回）。"""
    if normalized["verdict"] in TERMINAL_EVIDENCE_VERDICTS and not normalized["evidence"]:
        normalized["verdict"] = VERDICT_UNDETERMINED
        normalized["description"] = reason + normalized["description"]
    return normalized


def _normalize_result(item: dict[str, Any]) -> dict[str, Any] | None:
    check_id = str(item.get("check_id", "") or item.get("rule_id", "")).strip()
    verdict = str(item.get("verdict", "")).strip().lower()
    if not check_id or verdict not in VALID_VERDICTS:
        return None
    evidence = normalize_evidence_items(item.get("evidence"))
    description = str(item.get("description", "")).strip()
    applicability = item.get("applicability")
    applicability = applicability if isinstance(applicability, dict) else {}
    if verdict == VERDICT_NOT_APPLICABLE and (
        applicability.get("basis") != "explicit_scope_exclusion"
        or not str(applicability.get("explanation", "") or "").strip()
    ):
        verdict = VERDICT_UNDETERMINED
        description = "缺少排除条款适用性的正向证据，暂未观察到不等于不适用。" + description
    return _downgrade_without_evidence(
        {
            "check_id": check_id,
            "verdict": verdict,
            "description": description,
            "evidence": evidence,
            **({"applicability": applicability} if applicability else {}),
        },
        "判定模型未提供有效截图证据。",
    )


def _validate_functionality_evidence(item, steps):
    """Require ordered, executed actions and a later observed result."""
    chain = item.get("functionality_evidence")
    if not isinstance(chain, dict):
        return False
    # 与 evidence 归一化同一套步号规则：接受 5 与 "5"，拒绝 bool/0/负数。
    indices = [
        coerce_step(chain.get(key))
        for key in ("denial_step", "functional_action_step", "result_observation_step")
    ]
    if any(value is None for value in indices):
        return False
    denial, action, result = indices
    by_index = {step["raw_step_index"]: step for step in steps}
    if not (denial < action < result) or any(value not in by_index for value in indices):
        return False
    for value in indices:
        step = by_index[value]
        screenshot = step_screenshot(step)
        if step.get("screenshot_health_error") or _screenshot_health_error(decode_screenshot(screenshot)):
            return False
    for value in (denial, action):
        step = by_index[value]
        if step.get("action_failed") or step.get("action_name") not in {"click", "long_press", "type", "swipe", "scroll"}:
            return False
    return bool(str(chain.get("observed_result", "")).strip())


def _extract_raw_check_results(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Accept the small set of equivalent JSON shapes used by review models."""
    if not isinstance(result, dict):
        return []
    payload: Any = None
    for key in ("check_results", "results", "checks"):
        candidate = result.get(key)
        if isinstance(candidate, (list, dict)):
            payload = candidate
            break
    if payload is None and any(re.fullmatch(r"(?:R\d{3}|P\d{2})", str(key).strip()) for key in result):
        payload = result
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    extracted: list[dict[str, Any]] = []
    for check_id, value in payload.items():
        if not isinstance(value, dict):
            continue
        item = dict(value)
        item.setdefault("check_id", str(check_id))
        extracted.append(item)
    return extracted


def _build_messages(
    state: dict[str, Any],
    steps: list[dict[str, Any]],
    rules: list[dict[str, Any]],
) -> list:
    check_lines = []
    for rule in rules:
        checkpoint = rule.get("checkpoint", {}) or {}
        hints = "；".join(checkpoint.get("verdict_hints") or [])
        check_lines.append(
            f"- {rule['rule_id']}（{rule['regulation']}）：{rule['clause']}"
            + (f"｜判定要点：{hints}" if hints else "")
        )
    header_text = (
        "【合规检测轨迹判定】\n"
        "以下是本轮合规取证场景的完整动作序列，每一步都附带该动作执行前的截图。\n\n"
        "【本轮检测条款】\n" + "\n".join(check_lines) + "\n\n"
        "【场景目标】\n"
        f"{str(state.get('current_test_scenario', ''))[:800]}\n\n"
        "- ACTION_DESCRIPTION、动作参数与截图共同说明实际执行了哪个动作；例如“点击禁止按钮”"
        "不能被概括成未执行拒绝。\n"
        "- STEP N 的截图也是 STEP N-1 动作后的结果观测；只有最后一步没有后续结果截图。\n"
        "- 步骤里的系统权限弹窗、应用内同意弹窗、设置开关截图是核心证据，重点读取其中的文案与按钮。\n"
        "- 对要求拒绝后基本功能可用的条款，compliant 必须附 functionality_evidence："
        "{denial_step:实际拒绝步骤, functional_action_step:拒绝后实际验证基本功能的步骤, "
        "result_observation_step:后续结果截图步骤, observed_result:实际成功结果描述}。"
        "入口可见、回到首页、代理的 thought 均不能证明功能可用；未实际验证则 undetermined。"
        "- 每个条款都要给出结论；轨迹没有覆盖到的条款给 undetermined，不要臆断。"
    )
    message_content: list[dict[str, Any]] = [{"type": "text", "text": header_text}]
    previous_raw_index: int | None = None
    for index, step in enumerate(steps, start=1):
        raw_index = raw_step_index(step, index)
        compact = _compact_screenshot(step_screenshot(step))
        text = build_step_text(
            step,
            raw_index,
            coordinate_note="请区分图片坐标与屏幕坐标，不能仅凭执行器的文字判断点击是否命中目标。",
        )
        # 只有上一条保留步骤正好是原始轨迹的前一步时，当前截图才是它的动作结果；
        # 中间有无截图而被跳过的步骤时，这张图反映的是被跳过步骤之后的界面。
        if compact and previous_raw_index is not None and previous_raw_index == raw_index - 1:
            text += (
                f"\n[OBSERVES_PREVIOUS_RESULT] 当前图片是 STEP {previous_raw_index} "
                "动作执行后的界面。"
            )
        text += image_ref_text(raw_index, has_image=bool(compact), image_note="动作执行前的界面。")
        message_content.append({"type": "text", "text": text})
        if compact:
            message_content.append(image_block(compact))
        previous_raw_index = raw_index
    return [
        SystemMessage(content=COMPLIANCE_DETECT_SYSTEM_PROMPT),
        HumanMessage(content=message_content),
    ]


def analyze_execution_for_compliance(
    llm,
    state: dict[str, Any],
) -> dict:
    """对一次合规取证轨迹做条款判定，返回 {check_results}。"""
    check_ids = [
        str(item).strip()
        for item in (state.get("current_plan", {}) or {}).get("compliance_check_ids", []) or []
        if str(item).strip()
    ]
    rules_path = str(state.get("compliance_rules_path", "") or "").strip() or None
    rules = check_rules_by_ids(check_ids, rules_path)
    if not rules:
        return {"check_results": []}

    navigation_trace = state.get("navigation_execution_trace", [])
    scenario_trace = state.get("scenario_execution_trace", [])
    navigation_trace = list(navigation_trace) if isinstance(navigation_trace, list) else []
    scenario_trace = list(scenario_trace) if isinstance(scenario_trace, list) else []
    steps = _collect_steps(navigation_trace + scenario_trace)

    if not steps:
        # 没有任何带截图的步骤：全部条款记 undetermined，仍要销项避免死循环。
        return {
            "check_results": [
                {
                    "check_id": rule["rule_id"],
                    "verdict": VERDICT_UNDETERMINED,
                    "description": "本轮轨迹没有任何带截图的步骤，无法取证。",
                    "evidence": [],
                }
                for rule in rules
            ]
        }

    print(
        f"[合规检测] 输入状态: check_ids={len(rules)}, "
        f"recordable_steps={len(steps)}"
    )
    messages = _build_messages(state, steps, rules)
    result = invoke_llm_json_with_retry(
        llm,
        messages,
        log_prefix="[合规检测]",
        json_block_pattern=JSON_BLOCK_PATTERN,
    )
    if llm_json_result_failed(result):
        # H4: 调用/解析重试耗尽是基础设施失败，不是"模型正常返回但漏答该条款"。
        # 之前这里直接往下走，把两种情况都写成同一句 undetermined 兜底文案
        # （"判定模型未返回该条款结论。"），main.py 的产物校验也不检查这类失败，
        # 导致合规判定整体失效时产物仍可能标记为 success/complete。抛出异常，
        # 交给 compliance_worker 记入 failures 并按基础设施失败处理。
        raise ComplianceReviewError(getattr(result, "failure_reason", "模型判定失败"))
    raw_results = _extract_raw_check_results(result)
    print(
        "[合规检测] 模型返回结构: "
        f"top_keys={list(result)[:8] if isinstance(result, dict) else []}, "
        f"raw_results={len(raw_results)}, "
        f"ids={[str(item.get('check_id', '') or item.get('rule_id', '')) for item in raw_results[:12]]}"
    )
    normalized_by_id: dict[str, dict[str, Any]] = {}
    observed_indices = {step["raw_step_index"] for step in steps}
    rules_by_id = {rule["rule_id"]: rule for rule in rules}
    for item in raw_results:
        normalized = _normalize_result(item if isinstance(item, dict) else {})
        if normalized is not None:
            normalized["evidence"] = [
                evidence for evidence in normalized["evidence"]
                if evidence["global_step"] in observed_indices
            ]
            _downgrade_without_evidence(normalized, "判定引用的步骤没有有效截图证据。")
            rule = rules_by_id.get(normalized["check_id"], {})
            requires_functionality = rule.get("checkpoint", {}).get("requires_functionality_evidence", False)
            if normalized["verdict"] == VERDICT_COMPLIANT and requires_functionality:
                if not _validate_functionality_evidence(item, steps):
                    normalized["verdict"] = VERDICT_UNDETERMINED
                    normalized["description"] = "缺少拒绝后实际验证基本功能及结果截图的完整证据链。" + normalized["description"]
                else:
                    normalized["functionality_evidence"] = dict(item["functionality_evidence"])
            normalized_by_id[normalized["check_id"]] = normalized
    # LLM 漏掉的条款补 undetermined，保证每个 check_id 都有结论。
    check_results = [
        normalized_by_id.get(
            rule["rule_id"],
            {
                "check_id": rule["rule_id"],
                "verdict": VERDICT_UNDETERMINED,
                "description": "判定模型未返回该条款结论。",
                "evidence": [],
            },
        )
        for rule in rules
    ]
    verdict_counts: dict[str, int] = {}
    for item in check_results:
        verdict_counts[item["verdict"]] = verdict_counts.get(item["verdict"], 0) + 1
    print(f"[合规检测] 本轮判定: {json.dumps(verdict_counts, ensure_ascii=False)}")
    return {"check_results": check_results}
