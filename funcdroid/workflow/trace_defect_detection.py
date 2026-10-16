"""执行轨迹缺陷分析。"""

import json
import os
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from model.flow_graph import Page
from tools.utils import screenshot_dimensions
from workflow.bug_dedupe import bug_identity_key, dedupe_bugs, is_similar_bug_key
from workflow.image_compact import compact_screenshot
from workflow.llm import invoke_llm_json_with_retry, llm_json_result_failed
from workflow.logcat_crash import NON_CRASH_LOGCAT_PATTERNS, _logcat_has_target_process_crash  # noqa: F401 - 兼容既有导入
from workflow.nodes._shared import JSON_BLOCK_PATTERN
from workflow.nodes.prompt import BUG_DETECT_SYSTEM_PROMPT
from workflow.state import AgentState
from workflow.trace_messages import (
    build_step_text,
    image_block,
    image_ref_text,
    normalize_evidence_items,
    page_screenshot,
    raw_step_index,
)


class DefectReviewError(RuntimeError):
    """The review failed to produce a valid bugs array; [] is a valid verdict."""


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
    "left_app",
}
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
    r"重复|丢失|删除|新增|新建|创建|生成|关联|写入|覆盖|错位"
)
# 一次性任务（点“开始/一键/立即…”后运行一次，展示本次执行进度与结果）：
# 结果页只是本次执行的反馈，再次进入时重新运行、回到初始态是常见设计，
# 不是“保存后结果丢失”。按启动句式和“任务完成”句式识别，不依赖某类 app
# 的业务动词；描述里出现设置、输入、保存等持久化对象时仍按缺陷处理。
ONE_SHOT_TASK_PATTERN = re.compile(
    r"(?:一键|立即|开始)[\u4e00-\u9fa5]{1,4}"
    r"|(?:任务|执行|运行|处理|检测|扫描)(?:成功|完成|结束)"
)
REENTRY_RESET_PATTERN = re.compile(
    r"(?:再次|重新|二次)进入|恢复(?:为|到|成)?(?:初始|默认|原来|执行前|操作前)"
    r"|重置|又回到|回到[^\n。]{0,8}(?:初始|之前|前的)|重新出现[^\n。]{0,10}(?:开始|一键|立即)"
)
PERSISTENT_STATE_PATTERN = re.compile(
    r"开关|设置|模式|选项|勾选|输入|编辑|保存|提交|收藏|添加|删除|修改|账号|登录"
)
# 任务进行中的过渡画面：进度动画按自己的节奏循环，截图抓到的图形帧与同屏
# 进度数值不同步是常态，不是“同图矛盾”。三者同时出现才豁免：进行中标志、
# 动画/图形、进度数值；两个文字状态互斥或静止结果页矛盾仍报告。
IN_PROGRESS_STATE_PATTERN = re.compile(
    r"正在[^\n。，,]{0,8}"
    r"|[\u4e00-\u9fa5]{2}中\s*(?:…|\.{2,}|⋯)"
    r"|(?:加载|处理|下载|上传|同步|安装|更新|检测|扫描|执行|运行)中"
)
ANIMATION_GRAPHIC_PATTERN = re.compile(
    r"动画|图形|图示|图案|填充|进度条|进度环|圆环|波纹|指针"
)
PROGRESS_VALUE_PATTERN = re.compile(r"\d{1,3}\s*[%％]|进度")
PLACEHOLDER_OR_EMPTY_ITEM_PATTERN = re.compile(
    r"placeholder|new item|空白项|空项|占位|未输入",
    re.IGNORECASE,
)
# 默认参数与 FFG 更新一致（max_side=960, quality=80, 上限 95），这样同一张轨迹截图
# 在 workflow.image_compact 的共享缓存里只压一次；q100 重新编码反而会放大体积。
BUG_DETECT_IMAGE_MAX_SIDE = max(
    160,
    int(os.getenv("BUG_DETECT_IMAGE_MAX_SIDE", os.getenv("FFG_UPDATE_IMAGE_MAX_SIDE", "960"))),
)
BUG_DETECT_IMAGE_QUALITY = max(
    30,
    min(
        int(os.getenv("BUG_DETECT_IMAGE_QUALITY", os.getenv("FFG_UPDATE_IMAGE_QUALITY", "80"))),
        95,
    ),
)
BUG_CONTEXT_FIELDS = (
    "scenario_type",
    "target_function_id",
    "src_function_id",
    "dst_function_id",
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
    return page_screenshot(page)


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


def _compact_screenshot_for_llm(
    screenshot: str,
    *,
    max_side: int = BUG_DETECT_IMAGE_MAX_SIDE,
    quality: int = BUG_DETECT_IMAGE_QUALITY,
) -> str:
    """压缩缺陷判定输入截图；实现在进程级共享缓存上，同一原图只压一次。"""
    return compact_screenshot(
        screenshot,
        max_side=max_side,
        quality=quality,
        source_label="缺陷检测",
    )


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


def _normalize_bug(
    bug: dict[str, Any],
    scenario: str,
    bug_context: dict[str, str] | None = None,
) -> dict[str, Any]:
    evidence = normalize_evidence_items(bug.get("evidence"))
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
        navigation_status == "app_crashed"
        or scenario_status == "app_crashed"
        or bool(execution_result.get("crash_detected"))
    )
    # execution_trace 仅为接口兼容保留：是否记录崩溃只取决于运行器的崩溃判定与 logcat。
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

    crash_phase = "导航" if navigation_status == "app_crashed" else "测试"
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
        filtered.append(bug)
    deduped = dedupe_bugs(filtered)
    if len(deduped) != len(filtered):
        print(f"[缺陷检测] 已去重重复缺陷: {len(filtered) - len(deduped)} 个")
    return deduped


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
        if screenshot_dimensions(_page_screenshot(step_by_index[index].get("current_page"))) is None:
            return False
        # 截图健康检查失败的步骤（如防截屏页面返回的黑图）不能作为缺陷证据，
        # 否则会把“防截屏”误判成“应用黑屏”之类的自动化误报。
        if str(step_by_index[index].get("screenshot_health_error", "") or "").strip():
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
        # Verify that the referenced action actually executed, not how it was named.
        if action_step.get("action_name") not in {"click", "long_press", "type", "swipe", "scroll", "press_back"}:
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
        bug.get("evidence_category") == "persisted_result_error"
        and ONE_SHOT_TASK_PATTERN.search(bug_text)
        and REENTRY_RESET_PATTERN.search(bug_text)
        and not PERSISTENT_STATE_PATTERN.search(bug_text)
    ):
        return True
    if (
        bug.get("evidence_category") == "simultaneous_state_contradiction"
        and IN_PROGRESS_STATE_PATTERN.search(bug_text)
        and ANIMATION_GRAPHIC_PATTERN.search(bug_text)
        and PROGRESS_VALUE_PATTERN.search(bug_text)
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


def copy_bug_record(bug: dict[str, Any]) -> dict[str, Any]:
    """复制单条缺陷记录（含一层嵌套 list/dict），供跨线程快照使用。"""
    return {
        key: (
            [dict(item) if isinstance(item, dict) else item for item in value]
            if isinstance(value, list)
            else dict(value) if isinstance(value, dict) else value
        )
        for key, value in bug.items()
    }


def _ensure_bug_ids(bugs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """补齐上下文字段与 bug_id；返回新字典，不原地修改输入。

    输入里的缺陷字典可能同时被主线程快照引用（json.dumps 期间），原地
    setdefault 会触发“dictionary changed size during iteration”。
    """
    used_ids = {
        str(bug.get("bug_id", "")).strip()
        for bug in bugs
        if isinstance(bug, dict) and str(bug.get("bug_id", "")).strip()
    }
    next_index = 1
    result: list[dict[str, Any]] = []
    for bug in bugs:
        if not isinstance(bug, dict):
            result.append(bug)
            continue
        bug = {**{field: "" for field in BUG_CONTEXT_FIELDS}, **bug}
        if not str(bug.get("bug_id", "")).strip():
            while f"BUG-{next_index:03d}" in used_ids:
                next_index += 1
            bug["bug_id"] = f"BUG-{next_index:03d}"
            used_ids.add(bug["bug_id"])
            next_index += 1
        result.append(bug)
    return result


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
    if str(execution_result.get("navigation_status", "")).strip() == "app_crashed":
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
    for index, bug in enumerate(detected_bugs):
        if not isinstance(bug, dict):
            continue
        bug_id = str(bug.get("bug_id", "")).strip()
        if not bug_id:
            continue
        bug_key = bug_identity_key(bug)
        candidate = next((current for current in current_bugs if isinstance(current, dict) and (
            str(current.get("bug_id", "")).strip() == bug_id
            or is_similar_bug_key(bug_key, bug_identity_key(current))
        )), None)
        if candidate is not None and _is_better_bug_trace(trace, records.get(bug_id)):
            # Replace verdict, evidence and trace together; never mutate the worker's old snapshot.
            detected_bugs[index] = {**candidate, "bug_id": bug_id}
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
    # 本轮是“铺入口”还是“深度业务场景”，会影响该怎么看这条轨迹。
    # 只给阶段，不注入测试预期文本：测试意图不是产品承诺，注入会诱导误报。
    function_phase = str(current_plan.get("function_phase", "") or "").strip()
    if function_phase:
        review_context["function_phase"] = function_phase
    header_text = (
        "【完整执行轨迹缺陷复核】\n"
        "以下是本轮 navigation/test 的完整动作序列，每一步都附带该动作执行前的截图；"
        "请只依据这些截图和步骤信息判断，不要假设未展示的截图内容。\n\n"
        "【本轮权威上下文】\n"
        f"{json.dumps(review_context, ensure_ascii=False, default=str)}\n\n"
        "- [STEP_FLAGS] 标出的 is_interruption/action_failed/external_context/foreground_recovery 步骤"
        "默认是自动化或外部环境事件，不得直接当作 App 缺陷。\n"
        "- [AGENT_OBSERVATION] 是执行器的自述观察，只用于定位可疑界面；它不是证据，"
        "结论仍必须以截图为准。\n"
        "- function_phase=depth 表示本轮沿功能业务逻辑走了一条完整操作链，"
        "可重点关注操作前后、离开再返回后的结果对照；仍不得把测试意图当作产品承诺。\n"
        "- 测试场景文本是测试意图，不是产品承诺；expected 仍必须有截图、明确保存/提交结果或同屏矛盾支撑。\n"
        "- existing_bug_summaries 是此前已确认缺陷的摘要，仅用于按根因和用户可见后果去重，"
        "不能当作本轮证据；与其中任一缺陷重复时不要再次输出。"
    )
    message_content: list[dict[str, Any]] = [{"type": "text", "text": header_text}]
    # 先确定每步实际附带的图片：[IMAGE_REF] 与“结果体现在下一张图”的提示只能指向
    # 真实发出的图片，否则模型会把后一张图错认成当前步骤的页面状态。
    screenshots = [
        _compact_screenshot_for_llm(_page_screenshot(_extract_page(step)))
        for step in trace
    ]
    for index, step in enumerate(trace, start=1):
        raw_index = raw_step_index(step, index)
        screenshot = screenshots[index - 1]
        text = build_step_text(
            step,
            raw_index,
            coordinate_note="当前复核图片可能另有缩放，不要把屏幕像素直接当作图中位置。",
            include_flags=True,
        )
        # 图片编号必须与 [STEP] index 同源。提示词要求模型用“STEP index”填
        # global_step，而证据校验按 raw_step_index 建索引表；这里若改用连续序号，
        # 只要轨迹里有被过滤的步骤（中断/动作失败/外部应用），两套编号就会分叉，
        # 模型引用到的编号不在索引表里，整条缺陷会被静默丢弃。
        text += image_ref_text(
            raw_index,
            has_image=bool(screenshot),
            image_note="动作执行前的页面状态；不要把这张图片误当作该动作执行后的结果。",
        )
        if index < len(trace) and screenshots[index]:
            next_raw_index = raw_step_index(trace[index], index + 1)
            text += f" STEP {raw_index} 的动作结果通常体现在 BEFORE_STEP_{next_raw_index}。"
        message_content.append({"type": "text", "text": text})
        if screenshot:
            message_content.append(image_block(screenshot))
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
    last_step = full_trace[-1] if full_trace else {}
    observed_at_step_limit = (
        scenario_status == "incomplete"
        and bool(last_step.get("terminal_observation"))
        and bool(last_step.get("observation_only"))
        and not last_step.get("screenshot_health_error")
        and _is_recordable_step(last_step)
        and bool(_page_screenshot(_extract_page(last_step)))
    )
    skip_reason = ""
    if not full_trace:
        skip_reason = "执行 trace 为空"
    elif not trace:
        skip_reason = "过滤无效步骤后执行 trace 为空"
    elif scenario_status not in SEMANTIC_ANALYSIS_STATUSES and not observed_at_step_limit:
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

    if llm_json_result_failed(result):
        raise DefectReviewError(getattr(result, "failure_reason", "模型判定失败"))
    bugs_patch = result.get("bugs")
    if not isinstance(bugs_patch, list) or any(not isinstance(bug, dict) for bug in bugs_patch):
        raise DefectReviewError("缺陷判定响应缺少有效 bugs 数组")
    new_bugs = []
    for candidate in bugs_patch:
        normalized = _normalize_bug(candidate, scenario_for_bug, bug_context)
        reason = ""
        if not normalized["description"] or normalized["type"] not in FUNCTIONAL_BUG_TYPES:
            reason = "invalid_description_or_type"
        elif normalized["evidence_category"] not in DIRECT_EVIDENCE_CATEGORIES:
            reason = "unsupported_evidence_category"
        elif normalized["confidence"] != "high":
            reason = "insufficient_confidence"
        elif not _bug_has_required_trace_evidence(normalized, trace):
            reason = "incomplete_or_unordered_evidence"
        elif _bug_uses_speculative_or_automation_expectation(normalized, scenario_for_bug):
            reason = "unsupported_expectation"
        if reason:
            print("[缺陷候选过滤] " + json.dumps(
                {"reason": reason, "candidate": normalized}, ensure_ascii=False))
        else:
            new_bugs.append(normalized)
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
