"""执行轨迹缺陷分析。"""

import base64
import json
import os
import re
from typing import Any

import cv2
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage

from model.flow_graph import Page
from tools.utils import resize_image
from workflow.bug_dedupe import bug_identity_key, dedupe_bugs, is_similar_bug_key
from workflow.llm import invoke_llm_json_with_retry
from workflow.nodes.prompt import BUG_DETECT_SYSTEM_PROMPT
from workflow.state import AgentState

JSON_BLOCK_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$")
FUNCTIONAL_BUG_TYPES = {"ui_error", "logic_error"}
DIRECT_EVIDENCE_CATEGORIES = {
    "rendering_corruption",
    "persisted_result_error",
    "simultaneous_state_contradiction",
    "single_action_resource_mutation",
}
SEMANTIC_ANALYSIS_STATUSES = {
    "completed",
    "stalled",
    "failed_goal",
    "external_transition_blocked",
}
ENVIRONMENT_BUG_TERMS = (
    "failed to connect",
    "connection timed out",
    "connection refused",
    "network error",
    "api.",
    "timeout",
    "timed out",
    "unable to load",
    "连接错误",
    "连接失败",
    "网络连接错误",
    "网络异常",
    "网络超时",
    "请求超时",
    "无法连接",
)
PERSISTENCE_ACTION_PATTERNS = (
    r"保存",
    r"提交",
    r"确认",
    r"完成图标",
    r"对勾",
    r"\bsave\b",
    r"\bsubmit\b",
    r"\bconfirm\b",
    r"\bapply\b",
    r"\bdone\b",
    r"\bcheckmark\b",
)
SPECULATIVE_NAVIGATION_SCENARIO_PATTERN = re.compile(
    r"观察[^\n。]{0,80}是否(?:跳转|出现|打开|进入|切换)"
)
SPECULATIVE_NAVIGATION_BUG_PATTERN = re.compile(
    r"未[^\n。]{0,40}(?:预期|跳转)|反而|业务语义|应跳转|应进入"
)
SPATIAL_ASSOCIATION_PATTERN = re.compile(
    r"旁边|旁的|对应位置|关联到|错误关联|而非目标|无关"
)
BOUNDARY_INPUT_SCENARIO_PATTERN = re.compile(
    r"不输入任何|空标题|空值|特殊字符|超长|边界|truncation|!@#",
    re.IGNORECASE,
)
UNSUPPORTED_INPUT_RULE_PATTERN = re.compile(
    r"必填|不能为空|应当|不应|校验|合法|业务.*逻辑"
)
INPUT_WAS_NOT_APPLIED_PATTERN = re.compile(
    r"输入框(?:仍|尚未|为)?空|未输入任何|没有输入|输入未生效"
)
RESOURCE_MUTATION_PATTERN = re.compile(
    r"重复|丢失|丢失|删除|新增|新建|创建|生成|关联|写入|覆盖|错位"
)
PLACEHOLDER_OR_EMPTY_ITEM_PATTERN = re.compile(
    r"placeholder|new item|空白项|空项|占位|未输入",
    re.IGNORECASE,
)
BUG_DETECT_IMAGE_MAX_SIDE = max(
    160,
    int(os.getenv("BUG_DETECT_IMAGE_MAX_SIDE", os.getenv("FFG_UPDATE_IMAGE_MAX_SIDE", "960"))),
)
BUG_DETECT_IMAGE_QUALITY = max(
    30,
    min(
        int(os.getenv("BUG_DETECT_IMAGE_QUALITY", os.getenv("FFG_UPDATE_IMAGE_QUALITY", "100"))),
        100,
    ),
)
BUG_CONTEXT_FIELDS = (
    "scenario_type",
    "target_function_id",
    "src_function_id",
    "dst_function_id",
)
NON_CRASH_LOGCAT_PATTERNS = (
    "BufferLayer",
    "Timestamp",
    "seems implausible",
    "expectedPresent",
)
ENVIRONMENT_CRASH_LOGCAT_PATTERNS = (
    "EGL_BAD_ALLOC",
    "EGL_NOT_INITIALIZED",
    "Failed to create EGLSurface",
    "Failed to choose config",
    "DeadSystemException",
)
def _extract_page(step: dict[str, Any]) -> Page | dict[str, Any] | None:
    page = step.get("current_page")
    if isinstance(page, (Page, dict)):
        return page
    return None


def _page_screenshot(page: Page | dict[str, Any] | None) -> str:
    if isinstance(page, Page):
        return page.representative_screenshot
    if isinstance(page, dict):
        return str(page.get("screenshot", "") or page.get("screenshot_base64", ""))
    return ""


def _is_recordable_step(step: dict[str, Any]) -> bool:
    return (
        not bool(step.get("is_interruption"))
        and not bool(step.get("action_failed"))
        and not bool(step.get("external_context"))
    )


def _filter_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for raw_index, step in enumerate(trace, start=1):
        if not _is_recordable_step(step):
            continue
        copied = dict(step)
        copied["raw_step_index"] = raw_index
        filtered.append(copied)
    return filtered


def _compact_screenshot_for_llm(screenshot: str) -> str:
    screenshot = str(screenshot or "").strip()
    if not screenshot:
        return ""
    if screenshot.startswith("data:image") and "," in screenshot:
        screenshot = screenshot.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(screenshot, validate=False)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            return screenshot
        image = resize_image(image, max_side=BUG_DETECT_IMAGE_MAX_SIDE)
        success, buffer = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), BUG_DETECT_IMAGE_QUALITY],
        )
        if not success:
            return screenshot
        return base64.b64encode(buffer.tobytes()).decode("utf-8")
    except Exception as exc:
        print(f"[缺陷检测] 压缩截图失败，使用原图: {exc}")
        return screenshot


def _bug_context_from_plan(
    current_plan: dict[str, Any],
    execution_result: dict[str, Any] | None = None,
    state: AgentState | None = None,
) -> dict[str, str]:
    execution_result = execution_result or {}
    state = state or {}
    scenario_type = str(
        current_plan.get("scenario_type")
        or execution_result.get("scenario_type")
        or ""
    ).strip()
    target_function_id = str(current_plan.get("target_function_id", "")).strip()
    if scenario_type == "intra_test" and not target_function_id:
        target_function_id = str(state.get("current_testing_function_id", "")).strip()
    return {
        "scenario_type": scenario_type,
        "target_function_id": target_function_id,
        "src_function_id": str(current_plan.get("src_function_id", "")).strip(),
        "dst_function_id": str(current_plan.get("dst_function_id", "")).strip(),
    }


def _apply_bug_context(bug: dict[str, Any], bug_context: dict[str, str]) -> dict[str, Any]:
    for field in BUG_CONTEXT_FIELDS:
        bug[field] = str(bug_context.get(field, "")).strip()
    return bug


def _logcat_has_target_process_crash(crash_logcat: str, package_name: str) -> bool:
    text = str(crash_logcat or "")
    package_name = str(package_name or "").strip()
    if not text or not package_name:
        return False
    non_empty_lines = [line for line in text.splitlines() if line.strip()]
    if non_empty_lines and all(
        all(pattern in line for pattern in NON_CRASH_LOGCAT_PATTERNS)
        for line in non_empty_lines
    ):
        return False

    escaped_package = re.escape(package_name)
    if re.search(rf"\bANR in\s+{escaped_package}\b", text):
        return True
    if re.search(rf"\bProcess:\s*{escaped_package}\b", text) and "FATAL EXCEPTION" in text:
        return True
    if "Fatal signal" in text and (
        f">>> {package_name} <<<" in text
        or re.search(rf"\bCmdline:\s*{escaped_package}\b", text)
    ):
        return True
    strong_crash_signal = any(
        signal in text
        for signal in ("FATAL EXCEPTION", "AndroidRuntime", "Fatal signal", "App crash", "ANR in")
    )
    if strong_crash_signal and re.search(rf"\bForce finishing activity\s+{escaped_package}/", text):
        return True
    if strong_crash_signal and re.search(rf"\bProcess\s+{escaped_package}\s+\(pid\s+\d+\)\s+has died\b", text):
        return True
    return False


def _normalize_bug(
    bug: dict[str, Any],
    scenario: str,
    bug_context: dict[str, str] | None = None,
) -> dict[str, Any]:
    evidence = []
    for item in bug.get("evidence", []) if isinstance(bug.get("evidence"), list) else []:
        if not isinstance(item, dict):
            continue
        raw_step = item.get("global_step")
        if isinstance(raw_step, str) and raw_step.strip().isdigit():
            raw_step = int(raw_step.strip())
        if not isinstance(raw_step, int) or isinstance(raw_step, bool) or raw_step <= 0:
            continue
        role = str(item.get("role", "") or "").strip().lower()
        if role not in {"before", "action", "after"}:
            continue
        evidence.append({"global_step": raw_step, "role": role})
    normalized = {
        "type": str(bug.get("type", "")).strip().lower(),
        "evidence_category": str(bug.get("evidence_category", "")).strip().lower(),
        "severity": str(bug.get("severity", "")).strip(),
        "confidence": str(bug.get("confidence", "")).strip().lower(),
        "description": str(bug.get("description", "")).strip(),
        "expected": str(bug.get("expected", "")).strip(),
        "actual": str(bug.get("actual", "")).strip(),
        "evidence": evidence,
        "scenario": scenario,
        "source": "trace_analyzer",
    }
    return _apply_bug_context(normalized, bug_context or {})


def _runtime_crash_bug(
    execution_result: dict[str, Any],
    execution_trace: list[dict[str, Any]],
    scenario: str,
    package_name: str,
    bug_context: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    navigation_status = str(execution_result.get("navigation_status", "")).strip()
    scenario_status = str(execution_result.get("scenario_status", "")).strip()
    crashed = (
        navigation_status == "target_app_crashed"
        or scenario_status == "target_app_crashed"
        or bool(execution_result.get("crash_detected"))
    )
    if not execution_trace and not crashed:
        return None
    if not crashed:
        return None

    crash_logcat = str(execution_result.get("crash_logcat", "")).strip()
    escaped_package = re.escape(str(package_name or "").strip())
    has_explicit_app_runtime_crash = bool(
        escaped_package
        and (
            re.search(rf"\bANR in\s+{escaped_package}\b", crash_logcat)
            or (
                "FATAL EXCEPTION" in crash_logcat
                and re.search(rf"\bProcess:\s*{escaped_package}\b", crash_logcat)
            )
        )
    )
    if (
        not has_explicit_app_runtime_crash
        and any(pattern in crash_logcat for pattern in ENVIRONMENT_CRASH_LOGCAT_PATTERNS)
    ):
        print("[缺陷检测] 崩溃由模拟器图形栈/系统服务异常触发，不记录为目标 App 缺陷。")
        return None
    if not _logcat_has_target_process_crash(crash_logcat, package_name):
        if crash_logcat:
            print("[缺陷检测] 未确认目标进程崩溃堆栈，不记录 crash 缺陷。")
        return None

    crash_phase = "导航" if navigation_status == "target_app_crashed" else "测试"
    error_key = "navigation_error" if crash_phase == "导航" else "scenario_error"
    actual = (
        str(execution_result.get(error_key, "")).strip()
        or f"{crash_phase}场景执行过程中目标应用崩溃。"
    )
    if crash_logcat:
        actual = f"{actual}\n\nlogcat 摘要:\n{crash_logcat[:4000]}"
    recovered = bool(execution_result.get("recovered_to_target_app"))
    recovery_text = "运行器已重启目标应用" if recovered else "运行器未能恢复目标应用"
    bug = {
        "type": "crash",
        "severity": "high",
        "confidence": "high",
        "description": f"目标应用在{crash_phase}场景执行过程中发生崩溃，{recovery_text}并结束当前场景。",
        "expected": f"目标应用应保持稳定并完成当前{crash_phase}场景。",
        "actual": actual,
        "scenario": scenario,
        "source": "runtime_monitor",
    }
    return _apply_bug_context(bug, bug_context or {})


def _is_environment_noise_bug(bug: dict[str, Any]) -> bool:
    if bug.get("type") != "logic_error":
        return False
    text = " ".join(
        str(bug.get(key, ""))
        for key in ("description", "expected", "actual")
    ).lower()
    return any(term in text for term in ENVIRONMENT_BUG_TERMS)


def _dedupe_bugs(bugs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for bug in bugs:
        if not isinstance(bug, dict):
            continue
        if str(bug.get("confidence", "")).strip().lower() != "high":
            print(
                "[缺陷检测] 跳过非 high 置信度缺陷: "
                f"{bug.get('description', '')}"
            )
            continue
        if _is_environment_noise_bug(bug):
            print(f"[缺陷检测] 跳过外部环境/网络类疑似噪声缺陷: {bug.get('description', '')}")
            continue
        filtered.append(bug)
    deduped = dedupe_bugs(filtered)
    if len(deduped) != len(filtered):
        print(f"[缺陷检测] 已去重重复缺陷: {len(filtered) - len(deduped)} 个")
    return deduped


def _trace_has_persistence_action(trace: list[dict[str, Any]]) -> bool:
    for step in trace:
        if not isinstance(step, dict) or not _is_recordable_step(step):
            continue
        if str(step.get("action_name", "")).strip() == "finished":
            continue
        action_text = " ".join(
            str(step.get(field, "")).strip()
            for field in ("action_description", "thought")
        )
        if any(re.search(pattern, action_text, re.IGNORECASE) for pattern in PERSISTENCE_ACTION_PATTERNS):
            return True
    return False


def _bug_has_required_trace_evidence(
    bug: dict[str, Any],
    trace: list[dict[str, Any]],
) -> bool:
    evidence = bug.get("evidence", [])
    if not isinstance(evidence, list) or not evidence:
        return False
    step_by_index = {
        int(step.get("raw_step_index", index) or index): step
        for index, step in enumerate(trace, start=1)
        if isinstance(step, dict) and _is_recordable_step(step)
    }
    referenced: list[tuple[int, str, dict[str, Any]]] = []
    for item in evidence:
        if not isinstance(item, dict):
            return False
        index = item.get("global_step")
        role = str(item.get("role", "") or "").strip().lower()
        if not isinstance(index, int) or index not in step_by_index or role not in {"before", "action", "after"}:
            return False
        referenced.append((index, role, step_by_index[index]))
    if not any(str(step.get("trace_phase", "")).strip() == "test" for _, _, step in referenced):
        return False

    category = bug.get("evidence_category")
    if category in {"persisted_result_error", "single_action_resource_mutation"}:
        action_refs = [item for item in referenced if item[1] == "action"]
        after_refs = [item for item in referenced if item[1] == "after"]
        if len(action_refs) != 1 or not after_refs:
            return False
        action_index, _, action_step = action_refs[0]
        if str(action_step.get("trace_phase", "")).strip() != "test":
            return False
        if not any(index > action_index for index, _, _ in after_refs):
            return False
    if category == "persisted_result_error" and not _trace_has_persistence_action(trace):
        return False

    bug_text = " ".join(str(bug.get(key, "")) for key in ("description", "expected", "actual"))
    if PLACEHOLDER_OR_EMPTY_ITEM_PATTERN.search(bug_text):
        action_indices = [index for index, role, _ in referenced if role == "action"]
        cutoff = min(action_indices) if action_indices else 10**9
        has_nonempty_input = any(
            int(step.get("raw_step_index", index) or index) < cutoff
            and str(step.get("trace_phase", "")).strip() == "test"
            and str(step.get("action_name", "")).strip() == "type"
            and bool(str(step.get("content", "")).strip())
            for index, step in enumerate(trace, start=1)
            if isinstance(step, dict)
        )
        if not has_nonempty_input:
            return False
    return True


def _bug_uses_speculative_or_automation_expectation(
    bug: dict[str, Any],
    scenario: str,
) -> bool:
    bug_text = " ".join(
        str(bug.get(key, "")).strip()
        for key in ("description", "expected", "actual")
    )
    scenario = str(scenario or "").strip()
    if (
        bug.get("evidence_category") == "single_action_resource_mutation"
        and not RESOURCE_MUTATION_PATTERN.search(bug_text)
    ):
        return True
    if (
        SPECULATIVE_NAVIGATION_SCENARIO_PATTERN.search(scenario)
        and SPECULATIVE_NAVIGATION_BUG_PATTERN.search(bug_text)
    ):
        return True
    if "旁边" in scenario and SPATIAL_ASSOCIATION_PATTERN.search(bug_text):
        return True
    if (
        BOUNDARY_INPUT_SCENARIO_PATTERN.search(scenario)
        and UNSUPPORTED_INPUT_RULE_PATTERN.search(bug_text)
    ):
        return True
    if (
        re.search(r"输入(?:文本)?\s*[\"'“‘][^\"'”’]+[\"'”’]", scenario)
        and INPUT_WAS_NOT_APPLIED_PATTERN.search(bug_text)
    ):
        return True
    return False


def _ensure_bug_ids(bugs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    used_ids = {
        str(bug.get("bug_id", "")).strip()
        for bug in bugs
        if isinstance(bug, dict) and str(bug.get("bug_id", "")).strip()
    }
    next_index = 1
    for bug in bugs:
        if not isinstance(bug, dict):
            continue
        for field in BUG_CONTEXT_FIELDS:
            bug.setdefault(field, "")
        if str(bug.get("bug_id", "")).strip():
            continue
        while f"BUG-{next_index:03d}" in used_ids:
            next_index += 1
        bug["bug_id"] = f"BUG-{next_index:03d}"
        used_ids.add(bug["bug_id"])
        next_index += 1
    return bugs


def _bug_trace_quality(trace: list[dict[str, Any]]) -> tuple[int, int, int]:
    if not isinstance(trace, list):
        return (0, 0, 0)
    step_count = 0
    screenshot_count = 0
    non_empty_page_count = 0
    for step in trace:
        if not isinstance(step, dict):
            continue
        step_count += 1
        page = step.get("current_page")
        if isinstance(page, Page):
            if page.representative_screenshot:
                screenshot_count += 1
            if page.activity or page.description or page.operations or page.representative_screenshot:
                non_empty_page_count += 1
        elif isinstance(page, dict):
            screenshot = str(page.get("screenshot") or page.get("screenshot_base64") or "").strip()
            if screenshot:
                screenshot_count += 1
            if screenshot or str(page.get("activity", "")).strip() or str(page.get("description", "")).strip():
                non_empty_page_count += 1
    return (step_count, screenshot_count, non_empty_page_count)


def _is_better_bug_trace(candidate: list[dict[str, Any]], existing: list[dict[str, Any]] | None) -> bool:
    if not isinstance(existing, list) or not existing:
        return bool(candidate)
    return _bug_trace_quality(candidate) > _bug_trace_quality(existing)


def _current_bug_trace(state: AgentState) -> list[dict[str, Any]]:
    execution_result = state.get("execution_result", {}) or {}
    navigation_trace = state.get("navigation_execution_trace", [])
    scenario_trace = state.get("scenario_execution_trace", [])
    navigation_trace = list(navigation_trace) if isinstance(navigation_trace, list) else []
    scenario_trace = list(scenario_trace) if isinstance(scenario_trace, list) else []
    if str(execution_result.get("navigation_status", "")).strip() == "target_app_crashed":
        return navigation_trace
    return navigation_trace + scenario_trace


def _update_bug_trace_records(
    state: AgentState,
    detected_bugs: list[dict[str, Any]],
    current_bugs: list[dict[str, Any]],
    trace: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    existing_records = state.get("bug_trace_records", {})
    records = dict(existing_records) if isinstance(existing_records, dict) else {}
    if not current_bugs or not trace:
        return records
    current_bug_ids = {
        str(bug.get("bug_id", "")).strip()
        for bug in current_bugs
        if isinstance(bug, dict) and str(bug.get("bug_id", "")).strip()
    }
    current_bug_keys = [
        key
        for bug in current_bugs
        if isinstance(bug, dict)
        for key in [bug_identity_key(bug)]
        if key
    ]
    for bug in detected_bugs:
        if not isinstance(bug, dict):
            continue
        bug_id = str(bug.get("bug_id", "")).strip()
        if not bug_id:
            continue
        bug_key = bug_identity_key(bug)
        if (
            bug_id not in current_bug_ids
            and not any(is_similar_bug_key(bug_key, current_key) for current_key in current_bug_keys)
        ):
            continue
        if _is_better_bug_trace(trace, records.get(bug_id)):
            records[bug_id] = list(trace)
    return records


def _build_full_trace_messages(
    state: AgentState,
    trace: list[dict[str, Any]],
) -> list:
    """Build detection messages from the complete recordable trace.

    Every recordable step's screenshot is sent independently (no reuse of the
    action-agent conversation, no sampling): this function is the sole
    message-building path so that async/background detection only needs a
    serializable state snapshot, not a live LLM conversation object.
    """
    execution_result = dict(state.get("execution_result", {}) or {})
    execution_result.pop("crash_logcat", None)
    current_plan = dict(state.get("current_plan", {}) or {})
    existing_bug_summaries = [
        {
            field: str(bug.get(field, "")).strip()[:limit]
            for field, limit in (
                ("bug_id", 80),
                ("type", 40),
                ("evidence_category", 80),
                ("target_function_id", 80),
                ("description", 320),
                ("expected", 240),
                ("actual", 320),
            )
            if str(bug.get(field, "")).strip()
        }
        for bug in state.get("detected_bugs", []) or []
        if isinstance(bug, dict)
        and any(
            str(bug.get(field, "")).strip()
            for field in ("description", "expected", "actual")
        )
    ]
    review_context = {
        "navigation_scenario": str(state.get("current_navigation_scenario", "")).strip(),
        "test_scenario": str(state.get("current_test_scenario", "")).strip(),
        "current_plan": current_plan,
        "execution_result": execution_result,
        "existing_bug_summaries": existing_bug_summaries,
    }
    header_text = (
        "【完整执行轨迹缺陷复核】\n"
        "以下是本轮 navigation/test 的完整动作序列，每一步都附带该动作执行前的截图；"
        "请只依据这些截图和步骤信息判断，不要假设未展示的截图内容。\n\n"
        "【本轮权威上下文】\n"
        f"{json.dumps(review_context, ensure_ascii=False, default=str)}\n\n"
        "- is_interruption/action_failed/external_context/foreground_recovery 步骤默认是自动化或外部环境事件，不得直接当作 App 缺陷。\n"
        "- 测试场景文本是测试意图，不是产品承诺；expected 仍必须有截图、明确保存/提交结果或同屏矛盾支撑。\n"
        "- existing_bug_summaries 是此前已确认缺陷的摘要，仅用于按根因和用户可见后果去重，"
        "不能当作本轮证据；与其中任一缺陷重复时不要再次输出。"
    )
    message_content: list[dict[str, Any]] = [{"type": "text", "text": header_text}]
    for index, step in enumerate(trace, start=1):
        page = _extract_page(step)
        action_name = str(step.get("action_name", "")).strip()
        content = str(step.get("content", "")).strip()
        direction = str(step.get("direction", "")).strip()
        raw_index = int(step.get("raw_step_index", index) or index)
        trace_phase = str(step.get("trace_phase", "")).strip() or "unknown"
        text = (
            f"[STEP] index={raw_index}, recordable_index={index}, "
            f"phase={trace_phase}, action={action_name}"
        )
        if content:
            text += f", content={content}"
        if direction:
            text += f", direction={direction}"
        text += (
            f"\n[IMAGE_REF] BEFORE_STEP_{index}: 紧随其后的图片属于 recordable_index={index}，"
            "表示该动作执行前的页面状态；不要把这张图片误当作该动作执行后的结果。"
        )
        if index < len(trace):
            text += f" STEP {index} 的动作结果通常体现在 BEFORE_STEP_{index + 1}。"
        message_content.append({"type": "text", "text": text})
        screenshot = _compact_screenshot_for_llm(_page_screenshot(page))
        if screenshot:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{screenshot}"},
            })
    return [
        SystemMessage(content=BUG_DETECT_SYSTEM_PROMPT),
        HumanMessage(content=message_content),
    ]


def analyze_execution_for_bugs(
    llm,
    state: AgentState,
) -> dict:
    """Analyze a completed navigation/test execution and update bug evidence."""
    navigation_trace = state.get("navigation_execution_trace", [])
    scenario_trace = state.get("scenario_execution_trace", [])
    navigation_trace = list(navigation_trace) if isinstance(navigation_trace, list) else []
    scenario_trace = list(scenario_trace) if isinstance(scenario_trace, list) else []
    full_trace = navigation_trace + scenario_trace
    trace = _filter_trace(full_trace)
    execution_result = state.get("execution_result", {}) or {}
    current_plan = state.get("current_plan", {}) or {}
    test_scenario = str(state.get("current_test_scenario", "")).strip()
    navigation_scenario = str(state.get("current_navigation_scenario", "")).strip()
    scenario_for_bug = (
        test_scenario
        or navigation_scenario
    )
    package_name = str(state.get("package_name", "")).strip()
    existing_bugs = list(state.get("detected_bugs", []) or [])
    bug_context = _bug_context_from_plan(current_plan, execution_result, state)

    runtime_bugs = [
        bug
        for bug in [
            _runtime_crash_bug(
                execution_result,
                full_trace,
                scenario_for_bug,
                package_name,
                bug_context,
            )
        ]
        if bug is not None
    ]

    print(
        "[缺陷检测] 输入状态: "
        f"scenario_status={execution_result.get('scenario_status', '')}, "
        f"navigation_steps={len(navigation_trace)}, test_steps={len(scenario_trace)}, "
        f"recordable_steps={len(trace)}, "
        f"existing_bugs={len(existing_bugs)}"
    )

    scenario_status = str(execution_result.get("scenario_status", "")).strip()
    skip_reason = ""
    if not full_trace:
        skip_reason = "执行 trace 为空"
    elif not trace:
        skip_reason = "过滤无效步骤后执行 trace 为空"
    elif scenario_status not in SEMANTIC_ANALYSIS_STATUSES:
        skip_reason = f"执行状态 {scenario_status or '未知'} 缺少可靠的最终观测"
    elif current_plan.get("audit_exploration") is True:
        skip_reason = "入口探索场景只建立页面/功能证据，跳过无业务结果的 LLM 缺陷复核"

    if skip_reason:
        print(f"[缺陷检测] {skip_reason}，跳过 LLM 缺陷检测。")
        detected_bugs = _ensure_bug_ids(_dedupe_bugs(existing_bugs + runtime_bugs))
        bug_trace_records = _update_bug_trace_records(
            state,
            detected_bugs,
            runtime_bugs,
            _current_bug_trace(state),
        )
        return {
            "detected_bugs": detected_bugs,
            "bug_trace_records": bug_trace_records,
        }

    print(
        "[缺陷检测] 使用完整轨迹独立复核: "
        f"recordable_steps={len(trace)}（每步均附带执行前截图，不复用执行模型对话上下文）"
    )
    detection_messages = _build_full_trace_messages(state, trace)

    result = invoke_llm_json_with_retry(
        llm,
        detection_messages,
        log_prefix="[缺陷检测]",
        json_block_pattern=JSON_BLOCK_PATTERN,
    )

    bugs_patch = result.get("bugs", []) if isinstance(result.get("bugs"), list) else []
    new_bugs = [
        normalized
        for bug in bugs_patch
        for normalized in [_normalize_bug(bug, scenario_for_bug, bug_context)]
        if normalized["description"] and normalized["type"] in FUNCTIONAL_BUG_TYPES
        and normalized["evidence_category"] in DIRECT_EVIDENCE_CATEGORIES
        and normalized["confidence"] == "high"
        and _bug_has_required_trace_evidence(normalized, trace)
        and not _bug_uses_speculative_or_automation_expectation(
            normalized,
            scenario_for_bug,
        )
    ]
    if len(new_bugs) != len(bugs_patch):
        print(
            "[缺陷检测] 已过滤非法缺陷候选: "
            f"accepted={len(new_bugs)}, candidates={len(bugs_patch)}"
        )

    current_bugs = runtime_bugs + new_bugs
    detected_bugs = _ensure_bug_ids(_dedupe_bugs(existing_bugs + current_bugs))
    bug_trace_records = _update_bug_trace_records(
        state,
        detected_bugs,
        current_bugs,
        _current_bug_trace(state),
    )
    print(
        "[缺陷检测] 本轮结果: "
        f"runtime={len(runtime_bugs)}, llm={len(new_bugs)}, total={len(detected_bugs)}"
    )
    return {
        "detected_bugs": detected_bugs,
        "bug_trace_records": bug_trace_records,
    }
