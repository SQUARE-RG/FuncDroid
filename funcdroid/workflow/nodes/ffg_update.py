"""Lightweight FFG update node."""

import base64
import json
import os
import re
import time
from difflib import SequenceMatcher
from typing import Any

import cv2
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage

from model.flow_graph import (
    TERMINAL_FUNCTION_STATUSES,
    FlowGraph,
    FunctionDependency,
    FunctionNode,
    Page,
    PageTransition,
    control_family_key,
    operation_family_key,
)
from tools.utils import resize_image
from workflow.llm import invoke_llm_json_with_retry, llm_json_result_failed
from workflow.nodes._shared import (
    JSON_BLOCK_PATTERN,
    screenshot_similarity,
    screenshot_thumbnail,
)
from workflow.nodes.prompt import FFG_UPDATE_SYSTEM_PROMPT, PAGE_AUDIT_SYSTEM_PROMPT
from workflow.page_audit_worker import DEFAULT_BATCH_SIZE, page_audit_worker
from workflow.state import AgentState

TEXT_NORMALIZE_PATTERN = re.compile(r"[\"'“”‘’（）()【】\[\]{}<>《》、，,。.：:；;!！?？\-_/\s]+")
FUNCTION_FAILURE_BLOCK_THRESHOLD = max(1, int(os.getenv("FUNCTION_FAILURE_BLOCK_THRESHOLD", "3")))
CROSS_FAILURE_BLOCK_THRESHOLD = max(1, int(os.getenv("CROSS_FAILURE_BLOCK_THRESHOLD", "1")))
# 是否把“执行成功但零结构产出”计入功能无进展次数。关掉可退回旧行为。
ZERO_YIELD_COUNTS_AS_NO_PROGRESS = os.getenv(
    "FFG_ZERO_YIELD_COUNTS_AS_NO_PROGRESS", "1"
).strip().lower() in {"1", "true", "yes", "on"}
# 覆盖率过半后才把零产出计入无进展：探索初期本就需要几轮铺垫。
ZERO_YIELD_MIN_COVERAGE = max(
    0,
    min(int(os.getenv("FFG_ZERO_YIELD_MIN_COVERAGE", "50")), 100),
)
# 同 activity 且控件族集合重合度达到该比例时，判定为同一逻辑页面并复用，
# 避免同一页被拆成多个 page_N 后无法合并、覆盖率分母持续膨胀。
PAGE_SIGNATURE_MATCH_RATIO = max(
    0.5,
    min(float(os.getenv("FFG_UPDATE_PAGE_SIGNATURE_MATCH_RATIO", "0.85")), 1.0),
)
PAGE_SIGNATURE_MIN_CONTROLS = max(
    2,
    int(os.getenv("FFG_UPDATE_PAGE_SIGNATURE_MIN_CONTROLS", "3")),
)
FFG_UPDATE_IMAGE_MAX_SIDE = max(160, int(os.getenv("FFG_UPDATE_IMAGE_MAX_SIDE", "960")))
FFG_UPDATE_IMAGE_QUALITY = max(30, min(int(os.getenv("FFG_UPDATE_IMAGE_QUALITY", "80")), 95))
FFG_UPDATE_MATCHED_IMAGE_THRESHOLD = max(
    0.0,
    min(float(os.getenv("FFG_UPDATE_MATCHED_IMAGE_THRESHOLD", "0.98")), 1.0),
)
FFG_UPDATE_FAST_PATH_IMAGE_THRESHOLD = max(
    FFG_UPDATE_MATCHED_IMAGE_THRESHOLD,
    min(float(os.getenv("FFG_UPDATE_FAST_PATH_IMAGE_THRESHOLD", "0.995")), 1.0),
)
FFG_UPDATE_OMIT_MATCHED_TRACE_IMAGES = os.getenv(
    "FFG_UPDATE_OMIT_MATCHED_TRACE_IMAGES", "0"
).strip().lower() in {"1", "true", "yes", "on"}
FFG_UPDATE_FAST_PATH_ENABLED = os.getenv(
    "FFG_UPDATE_FAST_PATH_ENABLED", "0"
).strip().lower() in {"1", "true", "yes", "on"}
FFG_UPDATE_MAX_REFERENCE_IMAGES = max(
    0,
    int(os.getenv("FFG_UPDATE_MAX_REFERENCE_IMAGES", "12")),
)
# 只限制附加的已存 Page 参考图；本轮尚未匹配的轨迹截图属于一手证据，
# 即使超过该值也必须完整发送，不能因为请求预算隐藏潜在新页面。
FFG_UPDATE_MAX_TOTAL_IMAGES = max(
    1,
    int(os.getenv("FFG_UPDATE_MAX_TOTAL_IMAGES", "6")),
)
FFG_UPDATE_LLM_TIMEOUT_SECONDS = max(
    1.0,
    float(os.getenv("FFG_UPDATE_LLM_TIMEOUT", "300") or 300),
)
FFG_UPDATE_TIME_SAFETY_SECONDS = max(
    0.0,
    float(os.getenv("FFG_UPDATE_TIME_SAFETY_SECONDS", "5") or 5),
)
FFG_UPDATE_MAX_REFINEMENT_IMAGES = max(
    1,
    int(os.getenv("FFG_UPDATE_MAX_REFINEMENT_IMAGES", "3")),
)
FFG_UPDATE_CANDIDATES_PER_OBSERVATION = max(
    1,
    int(os.getenv("FFG_UPDATE_CANDIDATES_PER_OBSERVATION", "3")),
)
FFG_UPDATE_MAX_BOUNDARY_PAGES = max(
    4,
    int(os.getenv("FFG_UPDATE_MAX_BOUNDARY_PAGES", "24")),
)
FFG_UPDATE_MAX_GRANULARITY_REVIEWS_PER_FUNCTION = max(
    1, int(os.getenv("FFG_UPDATE_MAX_GRANULARITY_REVIEWS_PER_FUNCTION", "2"))
)
FFG_UPDATE_GRANULARITY_MIN_PAGES = max(
    4, int(os.getenv("FFG_UPDATE_GRANULARITY_MIN_PAGES", "6"))
)
FFG_UPDATE_GRANULARITY_MIN_ACTIVITIES = max(
    2, int(os.getenv("FFG_UPDATE_GRANULARITY_MIN_ACTIVITIES", "4"))
)
FFG_UPDATE_MAX_GRANULARITY_SCENARIOS = max(
    8, int(os.getenv("FFG_UPDATE_MAX_GRANULARITY_SCENARIOS", "30"))
)
FFG_UPDATE_MAX_FOCUSED_FUNCTIONS = max(
    4, int(os.getenv("FFG_UPDATE_MAX_FOCUSED_FUNCTIONS", "12"))
)
FFG_UPDATE_MAX_FOCUSED_PAGES = max(
    8, int(os.getenv("FFG_UPDATE_MAX_FOCUSED_PAGES", "24"))
)
FFG_UPDATE_MAX_GLOBAL_FUNCTIONS = max(
    20, int(os.getenv("FFG_UPDATE_MAX_GLOBAL_FUNCTIONS", "160"))
)
FFG_UPDATE_SUMMARY_MAX_CHARS = max(
    8000, int(os.getenv("FFG_UPDATE_SUMMARY_MAX_CHARS", "32000"))
)
FFG_CONTEXT_FIELD_MAX_CHARS = max(
    120, int(os.getenv("FFG_CONTEXT_FIELD_MAX_CHARS", "500"))
)
EXECUTION_FAILURE_STATUSES = {
    "error",
    "stalled",
    "action_failed",
    "max_steps_reached",
    "target_app_crashed",
    "target_app_restarted",
    "target_app_recovery_failed",
    "external_transition_blocked",
    "screenshot_unavailable",
    "failed_goal",
    "blocked_by_safety_policy",
    "blocked_by_verification",
    "skipped_due_navigation_failure",
    "skipped_due_screenshot_unavailable",
    "skipped_due_target_app_error",
    "skipped_due_target_app_crashed",
    "skipped_due_target_app_restarted",
    "skipped_due_target_app_recovery_failed",
}
CONFIRMED_STALL_BLOCK_PATTERN = re.compile(
    r"相同点击意图.{0,20}页面(?:持续|始终)无明显变化|"
    r"(?:连续|重复)(?:点击|操作|尝试).{0,24}页面(?:持续|始终)无明显变化|"
    r"页面(?:持续|始终)无变化.{0,20}(?:卡死|无法继续)"
)


def _trace_has_device_action(trace: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(step, dict)
        and not bool(step.get("observation_only"))
        and str(step.get("action_name", "")).strip()
        not in {"", "finished", "capture_screenshot", "wait"}
        for step in trace
    )
BLACK_SCREEN_PAGE_NAME = "黑屏或页面不可用"
BLACK_SCREEN_PAGE_DESCRIPTION = (
    "执行动作后无法获取有效界面；表示黑屏、空白屏或连续截图不可用的异常终点。"
    "规划时应避开通向该页面的已记录操作和场景。"
)


def _extract_page(step: dict[str, Any]) -> Page | dict[str, Any] | None:
    page = step.get("current_page")
    if isinstance(page, (Page, dict)):
        return page
    return None


def _page_activity(page: Page | dict[str, Any] | None) -> str:
    if isinstance(page, Page):
        return page.activity
    if isinstance(page, dict):
        return str(page.get("activity", ""))
    return ""


def _page_screenshot(page: Page | dict[str, Any] | None) -> str:
    if isinstance(page, Page):
        return page.representative_screenshot
    if isinstance(page, dict):
        representative = str(page.get("representative_screenshot", "") or "")
        if representative:
            return representative
        screenshots = page.get("screenshots")
        if isinstance(screenshots, list) and screenshots:
            return str(screenshots[0] or "")
        return str(page.get("screenshot", ""))
    return ""


def _unrepresented_trace_activities(
    graph: FlowGraph,
    trace: list[dict[str, Any]],
    package_name: str,
    scenario: str,
    trace_page_map: dict[int, str] | None = None,
) -> list[str]:
    """Find observed Activities that neither the graph nor the model mapped to a Page.

    A logical Page may be rendered by a transient/proxy Activity. When the FFG model
    explicitly reuses an existing page_ref for that trace step, its semantic merge is
    authoritative and must not be undone by an Activity-string-only retry loop.
    """
    trace_page_map = trace_page_map or {}
    represented = {
        str(page.activity or "").strip()
        for page in graph.pages.values()
        if str(page.activity or "").strip()
    }
    missing: list[str] = []
    for index, step in enumerate(trace, start=1):
        raw_step = int(step.get("raw_step_index", index) or index)
        mapped_page_id = str(trace_page_map.get(raw_step, "") or "").strip()
        if mapped_page_id and graph.get_page(mapped_page_id) is not None:
            continue
        # 外部应用页面（桌面/其他 App）与自动恢复步骤不参与目标应用 Page 建模，
        # 其 Activity 不应要求 LLM 新增 Page，否则与 ignore 对齐规则互相矛盾。
        if _is_ignorable_observation_step(step):
            continue
        activity = str(_page_activity(_extract_page(step)) or "").strip()
        if activity and activity not in represented and activity not in missing:
            missing.append(activity)
    return missing


class _ScreenshotThumbnailCache:
    """缓存截图的灰度缩略图，避免对同一张已持久化页面截图重复解码。

    FFG 更新中 `_retrieve_page_candidates` 与 `_reference_pages_for_update` 会对
    每个 trace step 与每张已存页面截图两两计算相似度；若不缓存，同一张 stored
    截图会在 N 个 step 的循环里被反复 base64 解码 + cv2.imdecode，随页面数增长
    接近 O(steps × pages × 每页截图数) 次完整解码。这里只缓存 64×64 缩略图，
    把解码次数降到每张截图一次。
    """

    def __init__(self, size: int = 64):
        self._size = size
        self._thumbnails: dict[str, np.ndarray | None] = {}

    def clear(self) -> None:
        self._thumbnails.clear()

    def preload_graph(self, graph: FlowGraph) -> None:
        for page in graph.active_pages():
            for screenshot in [
                page.representative_screenshot,
                *page.evidence_screenshots,
            ]:
                self.thumbnail(screenshot)

    def thumbnail(self, screenshot: str | None) -> np.ndarray | None:
        if not screenshot:
            return None
        key = str(screenshot)
        if key not in self._thumbnails:
            self._thumbnails[key] = screenshot_thumbnail(key, self._size)
        return self._thumbnails[key]

    def similarity(self, left: str | None, right: str | None) -> float | None:
        left_thumb = self.thumbnail(left)
        right_thumb = self.thumbnail(right)
        if left_thumb is None or right_thumb is None:
            return None
        diff = np.mean(
            np.abs(left_thumb.astype(np.float32) - right_thumb.astype(np.float32))
        ) / 255.0
        return 1.0 - float(diff)


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
        image = resize_image(image, max_side=FFG_UPDATE_IMAGE_MAX_SIDE)
        success, buffer = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), FFG_UPDATE_IMAGE_QUALITY],
        )
        if not success:
            return screenshot
        return base64.b64encode(buffer.tobytes()).decode("utf-8")
    except Exception as exc:
        print(f"[FFG 更新] 压缩截图失败，使用原图: {exc}")
        return screenshot


def _natural_language_controls(page: Page | dict[str, Any] | None) -> list[str]:
    """读取模型或持久化页面已经给出的自然语言控件描述。"""
    if isinstance(page, Page) and page.controls:
        existing = list(page.controls)
    elif isinstance(page, dict) and isinstance(page.get("controls"), list):
        existing = [str(item).strip() for item in page["controls"] if str(item).strip()]
    else:
        existing = []
    return existing


# ---------------------------------------------------------------------------
# 页面控件后台审计：合并结果 + 投喂新任务
# ---------------------------------------------------------------------------
PAGE_AUDIT_ENABLED = os.getenv("PAGE_AUDIT_ENABLED", "1").strip().lower() in {
    "1", "true", "yes", "on",
}
# 全局导航类控件族（返回/关闭可由系统返回键完成），审计不补充：
# 与执行器 press_back 不记入页面操作的口径一致。
PAGE_AUDIT_EXCLUDED_FAMILIES = {"返回", "关闭", "返回键"}
# 单页单次审计最多删除的操作数：防误删清空清单；transitions 保留。
PAGE_AUDIT_MAX_REMOVALS = 5
PAGE_AUDIT_MAX_CONTROL_MERGES = 3


def _page_audit_fingerprint(page: Page) -> str:
    """页面截图与操作清单的组合指纹。

    注意：不含 operation_bindings——绑定变化不产生新的审计价值，纳入
    会让升级后存量页面全部判定为“未审计”触发审计风暴。
    """
    import hashlib

    parts = [str(page.representative_screenshot or "")]
    parts.extend(str(item or "") for item in page.evidence_screenshots)
    parts.append("\x00operations")
    parts.extend(str(item or "") for item in page.operations)
    digest = hashlib.sha1("|".join(parts).encode("utf-8", errors="replace")).hexdigest()
    return digest[:16]


def _page_audit_semantically_duplicate(candidate: str, existing: list[str]) -> bool:
    """新增项与已有控件语义重复时丢弃。

    复用项目统一的 control_family_key 控件族归一化（去除位置限定词与
    控件后缀），同族即视为重复，语义口径与 FFG 覆盖统计保持一致。
    """
    normalized = TEXT_NORMALIZE_PATTERN.sub("", str(candidate or ""))
    if not normalized:
        return True
    family = control_family_key(candidate)
    for item in existing:
        if not family or not str(item or "").strip():
            continue
        if control_family_key(item) == family:
            return True
    return False


def _merge_page_audit_results(graph: FlowGraph) -> int:
    """主线程合并后台审计结果：校验通过才 add_control。

    校验防线：与既有控件语义重复的丢弃（同功能即重复）；全局导航类
    （返回/关闭）丢弃；每页单次新增上限由 worker 解析侧限制。合并后
    回报实际新增数，驱动 worker 的饱和退出判定。

    注意：不再要求控件名带位置限定词——实测该过滤把“从相册中选择”
    “拍一张”这类真实控件挡在清单外，导致后续操作绑定无处可挂、
    覆盖率无法增长；防“为增加而增加”由 prompt 的“宁缺毋滥”规则
    和语义去重承担。
    """
    if not PAGE_AUDIT_ENABLED:
        return 0
    total_changes = 0
    results = page_audit_worker.collect_results()
    for result in results:
        page_id = str(result.get("page_id", "") or "")
        page = graph.get_page(page_id) if page_id else None
        if page is None:
            # 页面已被合并/拆分移除：零新增回报，避免状态残留。
            page_audit_worker.record_result(page_id, 0)
            continue
        existing = list(page.controls)
        added: list[str] = []
        for control in result.get("missing_controls") or []:
            text = str(control or "").strip()
            if not text:
                continue
            if control_family_key(text) in PAGE_AUDIT_EXCLUDED_FAMILIES:
                print(f"[页面审计] 丢弃全局导航类候选控件（返回键可完成）: {text!r}")
                continue
            if _page_audit_semantically_duplicate(text, existing + added):
                print(f"[页面审计] 丢弃与既有控件语义重复的候选: {text!r}")
                continue
            added.append(text)
        for control in added:
            page.add_control(control)
        merged_controls = _apply_page_audit_control_merges(
            page,
            result.get("control_merges") or [],
        )
        removed = _apply_page_audit_operation_removals(
            page,
            result.get("invalid_operations") or [],
        )
        page_audit_worker.record_result(
            page_id,
            len(added) + len(merged_controls) + len(removed),
        )
        total_changes += len(added) + len(merged_controls) + len(removed)
        if added:
            print(
                f"[页面审计] 补充页面控件: page={page_id} "
                f"({page.name or '未命名'}), 新增 {len(added)} 项: "
                + "; ".join(added[:5])
            )
        if removed:
            print(
                f"[页面审计] 清理页面问题操作: page={page_id} "
                f"({page.name or '未命名'}), 删除 {len(removed)} 条: "
                + "; ".join(removed[:5])
            )
        if merged_controls:
            print(
                f"[页面审计] 合并重复资源控件: page={page_id} "
                f"({page.name or '未命名'}), 合并 {len(merged_controls)} 组: "
                + "; ".join(merged_controls[:3])
            )
    return total_changes


def _apply_page_audit_control_merges(
    page: Page,
    merge_items: list[dict[str, Any]],
) -> list[str]:
    """Apply exact, high-confidence type-level merges without losing bindings."""
    import os.path

    merged: list[str] = []
    for item in merge_items[:PAGE_AUDIT_MAX_CONTROL_MERGES]:
        if not isinstance(item, dict):
            continue
        sources = list(dict.fromkeys(
            str(value or "").strip()
            for value in item.get("sources", [])
            if str(value or "").strip()
        ))
        target = str(item.get("target", "") or "").strip()
        if len(sources) < 3 or not target or any(source not in page.controls for source in sources):
            print(f"[页面审计] 丢弃来源不完整的控件合并: sources={sources!r}, target={target!r}")
            continue
        normalized_sources = [TEXT_NORMALIZE_PATTERN.sub("", source) for source in sources]
        common_prefix = os.path.commonprefix(normalized_sources)
        normalized_target = TEXT_NORMALIZE_PATTERN.sub("", target)
        if len(common_prefix) < 2 or common_prefix not in normalized_target:
            print(
                "[页面审计] 丢弃缺少共同业务前缀的控件合并: "
                f"prefix={common_prefix!r}, target={target!r}"
            )
            continue
        if page.merge_controls(sources, target):
            merged.append(f"{','.join(sources)} -> {target}")
    return merged


def _apply_page_audit_operation_removals(
    page: Page,
    invalid_operations: list[str],
) -> list[str]:
    """点名式删除：与 page.operations 原文完全一致才删，其余丢弃。

    LLM 只能复制清单原文，不能改写或新增，结构上杜绝幻觉清空；
    每页单次删除上限 PAGE_AUDIT_MAX_REMOVALS；page_transitions 中的
    操作记录不受影响（沿用 press_back 先例：清单删、边保留）。

    已声明操作-控件绑定（operation_bindings）的操作一律不删：绑定
    意味着该操作已被验证为作用于真实控件的有效业务动作，删除会连带
    丢失绑定并直接损失该控件的覆盖率（实测 page_8 的“点击语音输入
    按钮”被误删导致 func_7 覆盖率受损）。若确属无效，应先解除绑定。
    """
    bound_operations = page.bound_operation_texts()
    removed: list[str] = []
    seen: set[str] = set()
    for item in invalid_operations:
        text = str(item or "").strip()
        if text.endswith("（未绑定控件）"):
            text = text[: -len("（未绑定控件）")].rstrip()
        if not text or text in seen:
            continue
        seen.add(text)
        if len(removed) >= PAGE_AUDIT_MAX_REMOVALS:
            break
        if text in bound_operations:
            print(
                f"[页面审计] 拒绝删除已绑定控件的操作（有效业务动作）: {text!r}"
            )
            continue
        if page.remove_operation(text):
            removed.append(text)
        else:
            print(f"[页面审计] 丢弃与操作清单原文不一致的待删项: {text!r}")
    return removed


def _submit_page_audit_batch(
    graph: FlowGraph,
    llm=None,
    *,
    exclude_fingerprints: set[tuple[str, str]] | None = None,
) -> int:
    """按优先级投喂一批页面审计任务到后台线程。

    优先级：已完成功能的页面（主循环不再踏入、自愈机会最少）
    > 控件数量少的页面。截图指纹未变或已饱和的页面由 worker 过滤。
    """
    if not PAGE_AUDIT_ENABLED or (llm is None and not page_audit_worker.is_configured()):
        return 0
    completed_page_ids: set[str] = set()
    for node in graph.functions.values():
        if str(node.test_status or "") in TERMINAL_FUNCTION_STATUSES:
            completed_page_ids.update(str(pid) for pid in node.coverage_page_ids)
    candidates: list[dict[str, Any]] = []
    for page in graph.pages.values():
        if not str(page.representative_screenshot or "").strip():
            continue
        candidate = {
            "page_id": page.id,
            "fingerprint": _page_audit_fingerprint(page),
            "completed_owner": page.id in completed_page_ids,
            "control_count": len(page.controls),
        }
        if exclude_fingerprints is not None and (
            candidate["page_id"], candidate["fingerprint"]
        ) in exclude_fingerprints:
            continue
        candidates.append(candidate)
    if not candidates:
        return 0
    candidates.sort(
        key=lambda item: (
            0 if item["completed_owner"] else 1,
            item["control_count"],
        )
    )
    allowed = page_audit_worker.filter_candidates(candidates)
    submitted = 0
    for candidate in allowed:
        if submitted >= DEFAULT_BATCH_SIZE:
            break
        page = graph.get_page(candidate["page_id"])
        if page is None:
            continue
        bound_operations = page.bound_operation_texts()
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"页面名称：{page.name or '未命名'}\n"
                    f"页面描述：{page.description or '无'}\n"
                    "已记录控件清单：\n"
                    + ("\n".join(f"- {c}" for c in page.controls) if page.controls else "（空）")
                    + "\n已记录操作清单（带【未绑定控件】标注的操作尚未声明作用控件，优先复核）：\n"
                    + (
                        "\n".join(
                            f"- {o}" if o in bound_operations else f"- {o}（未绑定控件）"
                            for o in page.operations
                        )
                        if page.operations
                        else "（空）"
                    )
                    + "\n请审计截图：补充清单遗漏的真实可交互控件，并清理无锚定或重复变体的问题操作。"
                ),
            },
        ]
        # 一个 Page 可以包含稳定态、键盘、弹层、展开面板等多个可见状态。
        # 审计必须看完该 Page 已持久化的全部截图；在这里截断会使后出现
        # 的面板控件永久无法进入控件全集。Page 自身已对截图去重并限定
        # 持久化规模，因此这里不再设第二道 2/3 张的隐式上限。
        screenshots = list(dict.fromkeys([
            page.representative_screenshot,
            *page.evidence_screenshots,
        ]))
        used = 0
        total = sum(1 for screenshot in screenshots if str(screenshot or "").strip())
        for screenshot in screenshots:
            payload = _compact_screenshot_for_llm(str(screenshot or ""))
            if not payload:
                continue
            content.append({
                "type": "text",
                "text": f"[PAGE_STATE_IMAGE] {used + 1}/{total}",
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{payload}"},
            })
            used += 1
        if used == 0:
            continue
        ok = page_audit_worker.submit({
            "page_id": page.id,
            "fingerprint": candidate["fingerprint"],
            "messages": [
                SystemMessage(content=PAGE_AUDIT_SYSTEM_PROMPT),
                HumanMessage(content=content),
            ],
        })
        if ok:
            submitted += 1
            if exclude_fingerprints is not None:
                exclude_fingerprints.add((page.id, candidate["fingerprint"]))
            print(
                f"[页面审计] 已提交页面清单复核任务: page={page.id} "
                f"({page.name or '未命名'}), 已记录控件={len(page.controls)}, "
                f"已记录操作={len(page.operations)}, 携带截图={used}"
            )
    return submitted


def _clone_page(page: Page | dict[str, Any] | None, page_id: str) -> Page:
    return Page(
        id=page_id,
        name=page.name if isinstance(page, Page) else "",
        description=page.description if isinstance(page, Page) else "",
        representative_screenshot=_page_screenshot(page),
        controls=_natural_language_controls(page),
        activity=_page_activity(page),
    )


def _is_recordable_step(step: dict[str, Any]) -> bool:
    return True


def _filter_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """保留所有带页面观测的执行轨迹，由 FFG 更新节点统一判断如何落图。"""
    filtered: list[dict[str, Any]] = []
    for raw_index, step in enumerate(trace, start=1):
        if not _is_recordable_step(step):
            continue
        copied = dict(step)
        copied["raw_step_index"] = raw_index
        filtered.append(copied)
    return filtered


def _needs_trace_screenshot(graph: FlowGraph, step: dict[str, Any]) -> bool:
    """Keep screenshots until strict Activity + visual matching can replace them."""
    return bool(_page_screenshot(_extract_page(step)))


def _find_matched_existing_page(
    graph: FlowGraph,
    step: dict[str, Any],
    raw_step: int,
    step_similarity_cache: dict[int, dict[str, float]],
    threshold: float | None = None,
) -> str:
    """Find a persisted Page that visually matches this step's screenshot.

    双条件判定：image_similarity 达到极高阈值 且 activity 完全一致。
    命中说明该 step 的页面与已存页面视觉一致，可用其文本摘要替代截图，
    从而避免把 test 阶段重复访问已知页面的截图全部上传。
    返回匹配 page_id；无匹配返回空串。
    """
    resolved_threshold = (
        FFG_UPDATE_MATCHED_IMAGE_THRESHOLD
        if threshold is None
        else max(0.0, min(float(threshold), 1.0))
    )
    scores = step_similarity_cache.get(raw_step, {})
    if not scores:
        return ""
    step_page = _extract_page(step)
    step_activity = str(_page_activity(step_page) or "").strip()
    if not step_activity:
        return ""
    best_id = ""
    best_score = resolved_threshold
    for page_id, image_similarity in scores.items():
        if image_similarity < best_score:
            continue
        page = graph.get_page(page_id)
        if page is None:
            continue
        page_activity = str(page.activity or "").strip()
        if not page_activity or step_activity != page_activity:
            continue
        best_id = page_id
        best_score = image_similarity
    return best_id


def _ffg_no_change_fast_path(
    graph: FlowGraph,
    state: AgentState,
    trace: list[dict[str, Any]],
    current_plan: dict[str, Any],
    step_similarity_cache: dict[int, dict[str, float]],
) -> dict[int, str] | None:
    """Return deterministic Page alignment when every observation is known."""
    if not FFG_UPDATE_FAST_PATH_ENABLED:
        return None
    execution_result = state.get("execution_result", {}) or {}
    scenario_status = str(execution_result.get("scenario_status") or "").strip()
    if (
        int(state.get("ffg_update_step", 1) or 1) > 1
        or str(current_plan.get("scenario_type", "") or "").strip() != "intra_test"
        or scenario_status != "completed"
        or state.get("ffg_update_validation_issues")
    ):
        return None

    target_function_id = _resolve_plan_target_function_id(graph, current_plan)
    if not target_function_id or graph.get_node(target_function_id) is None:
        return None

    trace_page_map: dict[int, str] = {}
    for index, step in enumerate(trace, start=1):
        if not isinstance(step, dict) or _extract_page(step) is None:
            continue
        if any(
            bool(step.get(key))
            for key in (
                "action_failed",
                "is_interruption",
                "foreground_recovery",
                "observation_after_recovery",
                "screenshot_health_error",
            )
        ):
            return None
        raw_step = int(step.get("raw_step_index", index) or index)
        page_id = _find_matched_existing_page(
            graph,
            step,
            raw_step,
            step_similarity_cache,
            threshold=FFG_UPDATE_FAST_PATH_IMAGE_THRESHOLD,
        )
        if not page_id or target_function_id not in graph.get_page_owners(page_id):
            return None
        trace_page_map[raw_step] = page_id

    if not trace_page_map:
        return None
    return trace_page_map


def _combined_execution_trace(state: AgentState) -> list[dict[str, Any]]:
    """导航与测试动作都属于本轮真实执行证据，按执行先后合并建图。"""
    combined: list[dict[str, Any]] = []
    for phase, key in (
        ("navigation", "navigation_execution_trace"),
        ("test", "scenario_execution_trace"),
    ):
        phase_trace = state.get(key, []) or []
        if not isinstance(phase_trace, list):
            continue
        for step in phase_trace:
            if not isinstance(step, dict):
                continue
            copied = dict(step)
            copied.setdefault("trace_phase", phase)
            combined.append(copied)
    return combined


def _screenshot_unavailable_source_step(
    state: AgentState,
    execution_result: dict[str, Any],
) -> tuple[dict[str, Any] | None, int]:
    """Return the last real action before the unavailable screenshot."""
    navigation_trace = list(state.get("navigation_execution_trace", []) or [])
    test_trace = list(state.get("scenario_execution_trace", []) or [])
    combined_trace = [*navigation_trace, *test_trace]
    unhealthy_observation_index = next(
        (
            index
            for index in range(len(combined_trace) - 1, -1, -1)
            if isinstance(combined_trace[index], dict)
            and str(combined_trace[index].get("screenshot_health_error", "")).strip()
            and bool(combined_trace[index].get("observation_only"))
        ),
        -1,
    )
    if unhealthy_observation_index >= 0:
        phase = str(combined_trace[unhealthy_observation_index].get("trace_phase", "")).strip()
        for index in range(unhealthy_observation_index - 1, -1, -1):
            step = combined_trace[index]
            if not isinstance(step, dict):
                continue
            if phase and str(step.get("trace_phase", "")).strip() != phase:
                break
            if (
                _extract_page(step) is not None
                and not str(step.get("screenshot_health_error", "")).strip()
                and not bool(step.get("observation_only"))
                and not bool(step.get("action_failed"))
                and str(step.get("action_name", "")).strip()
                not in {"finished", "press_back", "open_app", "restart_app"}
            ):
                return step, index + 1
        return None, 0

    if str(execution_result.get("navigation_status", "")).strip() == "screenshot_unavailable":
        trace = navigation_trace
        raw_offset = 0
    else:
        trace = test_trace
        raw_offset = len(navigation_trace)

    for index in range(len(trace) - 1, -1, -1):
        step = trace[index]
        if (
            isinstance(step, dict)
            and _is_recordable_step(step)
            and _extract_page(step) is not None
            and not bool(step.get("observation_only"))
            and str(step.get("action_name", "")).strip() not in {"finished", "restart_app"}
        ):
            return step, raw_offset + index + 1
    return None, 0


def _resolve_observed_page_id(
    graph: FlowGraph,
    step: dict[str, Any] | None,
    raw_step: int,
    trace_page_map: dict[int, str] | None = None,
) -> str:
    if not isinstance(step, dict):
        return ""
    trace_page_map = trace_page_map or {}
    mapped_page_id = str(trace_page_map.get(raw_step, "") or "").strip()
    if mapped_page_id and graph.get_page(mapped_page_id) is not None:
        return mapped_page_id

    page = _extract_page(step)
    runtime_page_id = str(
        page.id if isinstance(page, Page) else (page or {}).get("id", "")
    ).strip()
    resolved_page_id = graph.resolve_page_id(runtime_page_id)
    if resolved_page_id:
        return resolved_page_id

    screenshot = _page_screenshot(page)
    activity = str(_page_activity(page) or "").strip()
    best_match = ""
    best_similarity = 0.0
    for candidate in graph.active_pages():
        if candidate.name == BLACK_SCREEN_PAGE_NAME:
            continue
        if activity and candidate.activity and activity != candidate.activity:
            continue
        for stored in [candidate.representative_screenshot, *candidate.evidence_screenshots]:
            similarity = screenshot_similarity(screenshot, stored)
            if similarity is not None and similarity > best_similarity:
                best_similarity = similarity
                best_match = str(candidate.id or "")
    return best_match if best_similarity >= 0.98 else ""


def _black_screen_operation(step: dict[str, Any] | None) -> str:
    if not isinstance(step, dict):
        return "执行场景后出现黑屏或页面不可用"
    action = str(step.get("action_description", "") or "").strip()
    if not action:
        action_name = str(step.get("action_name", "") or "未知动作").strip()
        if step.get("x") is not None and step.get("y") is not None:
            action = f"{action_name}({step.get('x')},{step.get('y')})"
        elif str(step.get("content", "") or "").strip():
            action = f"{action_name}({str(step.get('content')).strip()})"
        elif str(step.get("direction", "") or "").strip():
            action = f"{action_name}({str(step.get('direction')).strip()})"
        else:
            action = action_name
    return f"{action}；结果：黑屏或页面不可用（截图健康检查失败）"


def _record_screenshot_unavailable_page(
    graph: FlowGraph,
    state: AgentState,
    execution_result: dict[str, Any],
    trace_page_map: dict[int, str] | None = None,
) -> list[str]:
    """Persist one reusable error Page without extending the FFG schema."""
    has_unavailable_observation = any(
        str(step.get("screenshot_health_error", "")).strip()
        for step in _combined_execution_trace(state)
    )
    if not (
        _is_screenshot_unavailable_execution(execution_result)
        or has_unavailable_observation
    ):
        return []

    source_step, raw_step = _screenshot_unavailable_source_step(state, execution_result)
    source_page_id = _resolve_observed_page_id(
        graph,
        source_step,
        raw_step,
        trace_page_map,
    )
    current_plan = state.get("current_plan", {}) or {}
    owner_ids: list[str] = []

    def add_owner(function_id: object):
        resolved = graph.resolve_function_id(str(function_id or "").strip())
        if resolved and resolved not in owner_ids:
            owner_ids.append(resolved)

    if source_page_id:
        for function_id in graph.get_page_owners(source_page_id):
            add_owner(function_id)
    for key in ("target_function_id", "src_function_id", "dst_function_id"):
        add_owner(current_plan.get(key))
    if not owner_ids:
        print("[FFG 更新] 黑屏没有可关联的功能节点，仅保留执行结果。")
        return []

    black_page = next(
        (page for page in graph.active_pages() if page.name == BLACK_SCREEN_PAGE_NAME),
        None,
    )
    if black_page is None:
        next_page_index = _next_numeric_id(list(graph.pages), "page_")
        page_id, _ = _allocate_page_id(graph, next_page_index)
        black_page = Page(
            id=page_id,
            name=BLACK_SCREEN_PAGE_NAME,
            description=BLACK_SCREEN_PAGE_DESCRIPTION,
        )
        graph.pages[page_id] = black_page
        print(f"[FFG 更新] 新增统一黑屏异常页面: {page_id}")
    else:
        black_page.description = BLACK_SCREEN_PAGE_DESCRIPTION

    black_page_id = str(black_page.id or "")
    for function_id in owner_ids:
        node = graph.get_node(function_id)
        if node is not None:
            node.add_page_id(black_page_id)

    operation = _black_screen_operation(source_step)
    source_page = graph.get_page(source_page_id)
    if source_page is not None:
        source_page.add_operation(operation)
        graph.add_page_transition(PageTransition(
            src_page_id=source_page_id,
            dst_page_id=black_page_id,
            operation=operation,
        ))
        print(
            "[FFG 更新] 已记录黑屏异常路径: "
            f"{source_page_id} --[{operation}]--> {black_page_id}"
        )
    else:
        print("[FFG 更新] 已记录黑屏异常页面；本轮没有可解析的最后有效源页面。")
    return owner_ids


def _runtime_page_with_ffg_identity(
    runtime_page: Page | dict[str, Any] | None,
    recorded_page: Page,
) -> Page:
    screenshot = _page_screenshot(runtime_page) or recorded_page.representative_screenshot
    activity = _page_activity(runtime_page) or recorded_page.activity
    return Page(
        id=recorded_page.id,
        name=recorded_page.name,
        description=recorded_page.description,
        representative_screenshot=screenshot,
        controls=list(recorded_page.controls),
        operations=list(recorded_page.operations),
        activity=activity,
    )


def _map_current_page_after_update(
    graph: FlowGraph,
    state: AgentState,
    trace_page_map: dict[int, str] | None = None,
) -> Page | None:
    """把设备最后实际界面绑定到本轮更新后的稳定 FFG Page。"""
    trace_page_map = trace_page_map or {}
    raw_trace = _combined_execution_trace(state)
    runtime_page = state.get("current_page")
    if not isinstance(runtime_page, Page):
        runtime_page = _extract_page(raw_trace[-1]) if raw_trace else None

    if raw_trace:
        last_observation = next(
            (
                (index, step, page)
                for index in range(len(raw_trace) - 1, -1, -1)
                for step in [raw_trace[index]]
                if (page := _extract_page(step)) is not None
            ),
            None,
        )
        if last_observation is not None:
            observation_index, last_step, last_page = last_observation
            runtime_page = last_page
        else:
            observation_index = -1
            last_step = raw_trace[-1]
        if str(last_step.get("screenshot_health_error", "")).strip():
            black_page = next(
                (page for page in graph.active_pages() if page.name == BLACK_SCREEN_PAGE_NAME),
                None,
            )
            if black_page is not None:
                return _runtime_page_with_ffg_identity(runtime_page, black_page)

        raw_step = observation_index + 1
        mapped_page_id = str(trace_page_map.get(raw_step, "") or "").strip()
        if mapped_page_id and (mapped_page := graph.get_page(mapped_page_id)) is not None:
            return _runtime_page_with_ffg_identity(runtime_page, mapped_page)

    if isinstance(runtime_page, Page):
        resolved_page_id = graph.resolve_page_id(str(runtime_page.id or ""))
        if resolved_page_id and (recorded_page := graph.get_page(resolved_page_id)) is not None:
            return _runtime_page_with_ffg_identity(runtime_page, recorded_page)
        resolved_page_id = _resolve_observed_page_id(
            graph,
            {"current_page": runtime_page},
            0,
        )
        if resolved_page_id and (recorded_page := graph.get_page(resolved_page_id)) is not None:
            return _runtime_page_with_ffg_identity(runtime_page, recorded_page)
        return runtime_page
    return None


def _endpoint_function_state(
    graph: FlowGraph,
    current_page: Page | None,
    current_plan: dict[str, Any],
    previous_function_id: str,
    touched_function_ids: list[str] | None = None,
) -> tuple[str, str]:
    """以执行终点页面的真实 owner 作为下一轮当前功能。"""
    if not isinstance(current_page, Page):
        status = graph.get_node_test_status(previous_function_id) if previous_function_id else "not_started"
        return previous_function_id, status

    page_id = graph.resolve_page_id(str(current_page.id or ""))
    owner_ids = graph.get_page_coverage_owners(page_id)
    if not owner_ids:
        owner_ids = graph.get_page_owners(page_id)
    if not owner_ids:
        status = graph.get_node_test_status(previous_function_id) if previous_function_id else "not_started"
        return previous_function_id, status

    scenario_type = str(current_plan.get("scenario_type", "") or "").strip()
    preferred_ids: list[str] = []
    if scenario_type == "cross_test":
        preferred_ids.append(str(current_plan.get("dst_function_id", "") or "").strip())
    else:
        planned_target_id = _resolve_plan_target_function_id(graph, current_plan)
        distinct_touched_owners = [
            resolved
            for function_id in reversed(touched_function_ids or [])
            if (resolved := graph.resolve_function_id(str(function_id or "").strip()))
            and resolved in owner_ids
            and resolved not in {previous_function_id, planned_target_id}
        ]
        distinct_touched_owners = list(dict.fromkeys(distinct_touched_owners))
        if len(distinct_touched_owners) == 1:
            preferred_ids.append(distinct_touched_owners[0])
        preferred_ids.extend([
            previous_function_id,
            planned_target_id,
        ])
    preferred_ids.extend(owner_ids)

    for function_id in preferred_ids:
        resolved = graph.resolve_function_id(str(function_id or "").strip())
        if resolved and resolved in owner_ids:
            return resolved, graph.get_node_test_status(resolved)
    selected = owner_ids[0]
    return selected, graph.get_node_test_status(selected)


def _trace_step_page_id(
    graph: FlowGraph,
    step: dict[str, Any],
    raw_step: int,
    trace_page_map: dict[int, str],
) -> str:
    if str(step.get("screenshot_health_error", "")).strip():
        black_page = next(
            (page for page in graph.active_pages() if page.name == BLACK_SCREEN_PAGE_NAME),
            None,
        )
        if black_page is not None:
            return str(black_page.id or "")
    return _resolve_observed_page_id(graph, step, raw_step, trace_page_map)


def _trace_step_operation_text(step: dict[str, Any]) -> tuple[str, bool]:
    """返回 (操作文本, 是否记入清单)；与 trace 落图口径完全一致。

    operation_bindings 的 source_step 绑定依赖同一口径推导操作原文，
    保证“声明绑定的步骤”与“实际记录的操作”是同一条文本。
    """
    action_name = str(step.get("action_name", "")).strip()
    operation = str(step.get("action_description", "") or "").strip() or action_name
    # 系统返回键是全局导航手段，不是页面自身支持的业务操作：
    # 不记入 page.operations，但保留页面跳转边（回退路径是导航图的一部分）。
    record_operation = action_name != "press_back"
    if action_name == "press_back":
        operation = "返回"
    if step.get("action_failed"):
        error = str(step.get("action_error", "") or "").strip() or "动作执行失败"
        operation = f"尝试{operation}失败：{error}"
    elif step.get("foreground_recovery") or step.get("is_interruption"):
        operation = f"自动恢复：{operation}"
    return operation, record_operation


def _apply_trace_operations_and_transitions(
    graph: FlowGraph,
    trace: list[dict[str, Any]],
    trace_page_map: dict[int, str] | None = None,
) -> list[str]:
    """用当前最小 FFG 结构确定性保存已执行操作和直接观测到的页面结果。"""
    trace_page_map = trace_page_map or {}
    touched_function_ids: list[str] = []

    def touch_page(page_id: str):
        for function_id in graph.get_page_coverage_owners(page_id):
            if function_id not in touched_function_ids:
                touched_function_ids.append(function_id)

    for index, step in enumerate(trace):
        if not isinstance(step, dict) or bool(step.get("observation_only")):
            continue
        action_name = str(step.get("action_name", "")).strip()
        if action_name in {"", "finished", "capture_screenshot", "wait"}:
            continue
        raw_step = int(step.get("raw_step_index", index + 1) or index + 1)
        if step.get("observation_after_recovery"):
            dst_page_id = _trace_step_page_id(
                graph,
                step,
                raw_step,
                trace_page_map,
            )
            src_page_id = ""
            for previous_index in range(index - 1, -1, -1):
                previous_step = trace[previous_index]
                if not isinstance(previous_step, dict) or _extract_page(previous_step) is None:
                    continue
                previous_raw_step = int(
                    previous_step.get("raw_step_index", previous_index + 1)
                    or previous_index + 1
                )
                src_page_id = _trace_step_page_id(
                    graph,
                    previous_step,
                    previous_raw_step,
                    trace_page_map,
                )
                break
            src_page = graph.get_page(src_page_id)
            dst_page = graph.get_page(dst_page_id)
            if src_page is not None and dst_page is not None:
                operation = (
                    str(step.get("action_description", "") or "").strip()
                    or "恢复目标 App"
                )
                operation = f"自动恢复：{operation}"
                src_page.add_operation(operation)
                graph.add_page_transition(PageTransition(
                    src_page_id=src_page_id,
                    dst_page_id=dst_page_id,
                    operation=operation,
                ))
                touch_page(src_page_id)
                touch_page(dst_page_id)
            continue
        src_page_id = _trace_step_page_id(
            graph,
            step,
            raw_step,
            trace_page_map,
        )
        src_page = graph.get_page(src_page_id)
        if src_page is None:
            continue

        next_observation: tuple[int, dict[str, Any]] | None = None
        for next_index in range(index + 1, len(trace)):
            next_step = trace[next_index]
            if isinstance(next_step, dict) and _extract_page(next_step) is not None:
                next_observation = (next_index, next_step)
                break
        if (
            not step.get("action_failed")
            and next_observation is not None
            and str(next_observation[1].get("screenshot_health_error", "")).strip()
        ):
            # 导致黑屏的操作和失败结果由专用异常边统一记录。
            continue

        operation, record_operation = _trace_step_operation_text(step)
        if record_operation:
            src_page.add_operation(operation)
        touch_page(src_page_id)

        if step.get("action_failed"):
            continue
        dst_page_id = ""
        if next_observation is not None:
            next_index, next_step = next_observation
            next_raw_step = int(
                next_step.get("raw_step_index", next_index + 1) or next_index + 1
            )
            dst_page_id = _trace_step_page_id(
                graph,
                next_step,
                next_raw_step,
                trace_page_map,
            )
        if not dst_page_id:
            continue
        dst_page = graph.get_page(dst_page_id)
        if dst_page is None:
            continue
        if dst_page.name == BLACK_SCREEN_PAGE_NAME:
            # 黑屏入边需要带明确失败结果，由专用逻辑统一维护。
            continue
        graph.add_page_transition(PageTransition(
            src_page_id=src_page_id,
            dst_page_id=dst_page_id,
            operation=operation,
        ))
        touch_page(dst_page_id)
    return touched_function_ids


def _apply_operation_bindings(
    graph: FlowGraph,
    patch: dict[str, Any],
    trace: list[dict[str, Any]],
    trace_page_map: dict[int, str],
) -> int:
    """应用 patch 声明的“操作→控件”显式绑定；必须在 trace 操作落图之后调用。

    绑定是覆盖率计算的权威来源（见 FlowGraph.function_handled_controls）：
    - source_step 指向 [STEP] 步骤号时，操作原文由程序按 trace 落图口径
      推导，杜绝模型改写文本造成“绑定了不存在的操作”；
    - 也可直接给 operation（必须与页面操作清单原文一致）；
    - control 必须与所在页面控件清单原文一致（逐字或归一化一致）；
    校验失败的绑定丢弃并告警，不阻塞本轮图增量。即使本轮 patch 结构
    校验失败被回滚，绑定指向的页面/控件若仍然存在也可安全应用：它
    描述的是 trace 中的事实，而非结构性变更。
    """
    items = patch.get("operation_bindings")
    if not isinstance(items, list):
        return 0
    applied = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        control = str(item.get("control", "") or "").strip()
        if not control:
            continue
        source_step = str(item.get("source_step", "") or "").strip()
        page_ref = str(item.get("page_ref", "") or "").strip()
        page = graph.get_page(page_ref) if page_ref else None
        operation_text = ""
        if source_step.isdigit():
            raw_step = int(source_step)
            step = next(
                (
                    candidate
                    for candidate in trace
                    if isinstance(candidate, dict)
                    and int(candidate.get("raw_step_index", 0) or 0) == raw_step
                ),
                None,
            )
            if step is not None:
                operation_text, recordable = _trace_step_operation_text(step)
                if not recordable:
                    # press_back 等不记入清单的操作没有可绑定的操作记录。
                    operation_text = ""
                if page is None:
                    src_page_id = _trace_step_page_id(
                        graph,
                        step,
                        raw_step,
                        trace_page_map,
                    )
                    page = graph.get_page(src_page_id) if src_page_id else None
        elif str(item.get("operation", "") or "").strip():
            operation_text = str(item.get("operation", "") or "").strip()
        if page is None or not operation_text:
            print(
                "[FFG 更新] 丢弃无法定位步骤或页面的操作绑定: "
                f"source_step={source_step}, control={control!r}"
            )
            continue
        operation_key = operation_family_key(operation_text)
        if operation_key:
            operation_text = next(
                (
                    existing
                    for existing in page.operations
                    if operation_family_key(existing) == operation_key
                ),
                operation_text,
            )
        if page.add_operation_binding(operation_text, control):
            applied += 1
        elif page.has_operation_binding(operation_text, control):
            # 前一次尝试（或同轮 patch 结构回滚前的应用）已记录同一绑定：
            # refinement 重复声明属正常现象，不是校验失败。
            print(
                f"[FFG 更新] 跳过重复声明的操作绑定（已存在）: "
                f"page={page.id}, control={control!r}"
            )
        else:
            print(
                "[FFG 更新] 丢弃校验失败的操作绑定（操作或控件不在本页清单）: "
                f"page={page.id}, operation={operation_text!r}, control={control!r}"
            )
    return applied


def _auto_bind_unambiguous_trace_operations(
    graph: FlowGraph,
    trace: list[dict[str, Any]],
    trace_page_map: dict[int, str],
) -> int:
    """Bind only when one executed operation has exactly one control-family match.

    This is a conservative weak-model fallback: both the operation and control
    must already exist in the same Page, and ambiguous matches are left for LLM
    refinement instead of guessing or creating controls.
    """
    applied = 0
    for index, step in enumerate(trace, start=1):
        raw_step = int(step.get("raw_step_index", index) or index)
        operation_text, recordable = _trace_step_operation_text(step)
        if not recordable or not operation_text:
            continue
        page_id = _trace_step_page_id(graph, step, raw_step, trace_page_map)
        page = graph.get_page(page_id) if page_id else None
        if page is None:
            continue
        operation_key = operation_family_key(operation_text)
        if not operation_key or ":" not in operation_key:
            continue
        target_family = operation_key.split(":", 1)[1]
        stored_operation = next(
            (
                existing
                for existing in page.operations
                if operation_family_key(existing) == operation_key
            ),
            "",
        )
        if not stored_operation or stored_operation in page.bound_operation_texts():
            continue
        candidates = [
            control
            for control in page.controls
            if (
                (control_key := control_family_key(control)) == target_family
                or (
                    control_key
                    and target_family
                    and (
                        control_key.endswith(target_family)
                        or target_family.endswith(control_key)
                    )
                )
            )
        ]
        if len(candidates) != 1:
            continue
        if page.add_operation_binding(stored_operation, candidates[0]):
            applied += 1
            print(
                "[FFG 更新] 已自动补齐唯一无歧义操作绑定: "
                f"page={page.id}, operation={stored_operation!r}, control={candidates[0]!r}"
            )
    return applied


def _format_abnormal_step_summary(trace: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, step in enumerate(trace, start=1):
        if _is_recordable_step(step):
            continue
        action = str(step.get("action_name", "")).strip() or "未知动作"
        thought = str(step.get("thought", "")).strip()
        error = str(step.get("action_error", "")).strip()
        parts = [f"raw_step={index}", f"action={action}"]
        if step.get("is_interruption"):
            parts.append("is_interruption=true")
        if step.get("action_failed"):
            parts.append("action_failed=true")
        if step.get("foreground_recovery"):
            parts.append("foreground_recovery=true")
        if step.get("external_context"):
            parts.append("external_context=true")
        if step.get("caused_external_transition"):
            parts.append("caused_external_transition=true")
        previous_package = str(step.get("previous_package", "")).strip()
        target_package = str(step.get("target_package", "")).strip()
        recovery_method = str(step.get("recovery_method", "")).strip()
        external_package = str(step.get("external_package", "")).strip()
        if previous_package:
            parts.append(f"previous_package={previous_package}")
        if target_package:
            parts.append(f"target_package={target_package}")
        if external_package:
            parts.append(f"external_package={external_package}")
        if recovery_method:
            parts.append(f"recovery_method={recovery_method}")
        if thought:
            parts.append(f"thought={thought}")
        if error:
            parts.append(f"error={error}")
        lines.append(", ".join(parts))
    return "\n".join(lines)


def _clip_context_text(value: object, limit: int = FFG_CONTEXT_FIELD_MAX_CHARS) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    marker = "…[已截断]"
    return text[: max(1, limit - len(marker))] + marker


def _focused_update_function_ids(
    graph: FlowGraph,
    current_plan: dict[str, Any],
    trace: list[dict[str, Any]],
) -> list[str]:
    focused: list[str] = []

    def add(function_id: object):
        resolved = graph.resolve_function_id(str(function_id or "").strip())
        if resolved and resolved in graph.functions and resolved not in focused:
            focused.append(resolved)

    for key in ("target_function_id", "src_function_id", "dst_function_id"):
        add(current_plan.get(key))
    trace_activities: set[str] = set()
    for step in trace:
        page = _extract_page(step)
        runtime_page_id = str(
            page.id if isinstance(page, Page) else page.get("id", "") if isinstance(page, dict) else ""
        ).strip()
        if runtime_page_id:
            for owner_id in graph.get_page_owners(runtime_page_id):
                add(owner_id)
        activity = _page_activity(page)
        if activity:
            trace_activities.add(activity)
    for page in graph.active_pages():
        if page.activity and page.activity in trace_activities:
            for owner_id in graph.get_page_owners(str(page.id or "")):
                add(owner_id)

    seed_ids = set(focused)
    for edge in graph.function_dependencies:
        if edge.src_function_id in seed_ids or edge.dst_function_id in seed_ids:
            add(edge.src_function_id)
            add(edge.dst_function_id)
    if not focused:
        for node in sorted(
            graph.functions.values(),
            key=lambda item: (
                (item.test_status or "not_started") in TERMINAL_FUNCTION_STATUSES,
                len(item.scenarios),
                item.id,
            ),
        ):
            add(node.id)
            if len(focused) >= 4:
                break
    return focused[:FFG_UPDATE_MAX_FOCUSED_FUNCTIONS]


def _function_granularity_metrics(
    graph: FlowGraph,
    function_id: str,
) -> dict[str, Any]:
    node = graph.get_node(function_id)
    if node is None:
        return {
            "coverage_page_count": 0,
            "activity_count": 0,
        }
    coverage_page_ids = list(node.coverage_page_ids)
    activities = {
        str(page.activity or "").strip()
        for page_id in coverage_page_ids
        if (page := graph.get_page(page_id)) is not None
        and str(page.activity or "").strip()
    }
    return {
        "coverage_page_count": len(coverage_page_ids),
        "activity_count": len(activities),
    }


def _select_granularity_review_function(
    graph: FlowGraph,
    current_plan: dict[str, Any],
    review_attempts: dict[str, int],
) -> tuple[str, dict[str, Any]]:
    """Select one planned Function whose boundary has grown suspiciously broad.

    Reviews are deliberately spaced out. A Function is reconsidered at roughly
    ``min_pages`` and then again after it doubles, instead of paying for a global
    restructuring decision on every known-page update.
    """
    candidate_ids: list[str] = []
    for key in ("target_function_id", "src_function_id", "dst_function_id"):
        function_id = graph.resolve_function_id(
            str(current_plan.get(key, "") or "").strip()
        )
        if function_id and function_id not in candidate_ids:
            candidate_ids.append(function_id)

    for function_id in candidate_ids:
        attempt = max(0, int(review_attempts.get(function_id, 0) or 0))
        if attempt >= FFG_UPDATE_MAX_GRANULARITY_REVIEWS_PER_FUNCTION:
            continue
        metrics = _function_granularity_metrics(graph, function_id)
        scale = 2 ** attempt
        page_threshold = FFG_UPDATE_GRANULARITY_MIN_PAGES * scale
        activity_threshold = FFG_UPDATE_GRANULARITY_MIN_ACTIVITIES + attempt * 2
        page_count = int(metrics["coverage_page_count"])
        activity_count = int(metrics["activity_count"])
        if page_count < page_threshold and not (
            page_count >= max(4, page_threshold // 2)
            and activity_count >= activity_threshold
        ):
            continue
        return function_id, {
            **metrics,
            "review_attempt": attempt + 1,
            "max_review_attempts": FFG_UPDATE_MAX_GRANULARITY_REVIEWS_PER_FUNCTION,
            "page_threshold": page_threshold,
            "activity_threshold": activity_threshold,
        }
    return "", {}


def _page_semantic_text(page: Page | dict[str, Any] | None) -> str:
    if isinstance(page, Page):
        values = [page.name, page.description, *page.controls]
    elif isinstance(page, dict):
        controls = page.get("controls", [])
        values = [
            str(page.get("name", "")),
            str(page.get("description", "")),
            *(
                str(value)
                for value in controls
                if isinstance(controls, list) and isinstance(value, str)
            ),
        ]
    else:
        values = []
    return " ".join(str(value or "").strip() for value in values if str(value or "").strip())


def _normalized_similarity_text(value: object) -> str:
    return TEXT_NORMALIZE_PATTERN.sub("", str(value or "").strip().lower())


def _semantic_text_similarity(left: object, right: object) -> float:
    left_text = _normalized_similarity_text(left)
    right_text = _normalized_similarity_text(right)
    if not left_text or not right_text:
        return 0.0
    if left_text in right_text or right_text in left_text:
        return min(len(left_text), len(right_text)) / max(len(left_text), len(right_text))
    return SequenceMatcher(None, left_text, right_text).ratio()


def _retrieve_page_candidates(
    graph: FlowGraph,
    step: dict[str, Any],
    current_plan: dict[str, Any],
    previous_page_ids: list[str] | None = None,
    limit: int = FFG_UPDATE_CANDIDATES_PER_OBSERVATION,
    thumbnail_cache: _ScreenshotThumbnailCache | None = None,
    similarity_sink: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Rank persisted Pages that may represent one runtime observation."""
    runtime_page = _extract_page(step)
    if runtime_page is None:
        return []
    runtime_page_id = str(
        runtime_page.id
        if isinstance(runtime_page, Page)
        else runtime_page.get("id", "")
    ).strip()
    runtime_activity = str(_page_activity(runtime_page) or "").strip()
    runtime_screenshot = _page_screenshot(runtime_page)
    runtime_text = " ".join((
        _page_semantic_text(runtime_page),
        str(step.get("action_description", "") or "").strip(),
        str(step.get("thought", "") or "").strip(),
    )).strip()
    scope_ids = {
        resolved
        for key in ("target_function_id", "src_function_id", "dst_function_id")
        if (resolved := graph.resolve_function_id(str(current_plan.get(key, "") or "").strip()))
    }
    previous_ids = {
        resolved
        for value in (previous_page_ids or [])
        if (resolved := graph.resolve_page_id(value))
    }
    adjacent_targets = {
        transition.dst_page_id
        for transition in graph.page_transitions
        if transition.src_page_id in previous_ids
    }

    candidates: list[dict[str, Any]] = []
    for page in graph.active_pages():
        page_id = str(page.id or "").strip()
        if not page_id:
            continue
        exact_id = bool(runtime_page_id and runtime_page_id == page_id)
        activity_match = bool(runtime_activity and page.activity and runtime_activity == page.activity)
        activity_compatible = activity_match or not runtime_activity or not page.activity
        image_similarity = 0.0
        if runtime_screenshot:
            image_scores = []
            for stored in [page.representative_screenshot, *page.evidence_screenshots]:
                similarity = (
                    thumbnail_cache.similarity(runtime_screenshot, stored)
                    if thumbnail_cache is not None
                    else screenshot_similarity(runtime_screenshot, stored)
                )
                if similarity is not None:
                    image_scores.append(similarity)
            image_similarity = max(image_scores, default=0.0)
            if similarity_sink is not None:
                similarity_sink[page_id] = image_similarity
        text_similarity = _semantic_text_similarity(runtime_text, _page_semantic_text(page))
        owners = graph.get_page_owners(page_id)
        scope_match = bool(scope_ids.intersection(owners))
        graph_adjacent = page_id in adjacent_targets
        if exact_id:
            score = 1.0
        else:
            score = (
                0.45 * image_similarity
                + 0.20 * (1.0 if activity_match else 0.35 if activity_compatible else 0.0)
                + 0.15 * text_similarity
                + 0.10 * float(scope_match)
                + 0.10 * float(graph_adjacent)
            )
        candidates.append({
            "page_id": page_id,
            "owners": owners,
            "name": _clip_context_text(page.name or page.description, 160),
            "activity": _clip_context_text(page.activity, 180),
            "score": round(score, 4),
            "exact_id": exact_id,
            "image_similarity": round(image_similarity, 4),
            "activity_match": activity_match,
            "activity_compatible": activity_compatible,
            "semantic_similarity": round(text_similarity, 4),
            "scope_match": scope_match,
            "graph_adjacent_from_previous": graph_adjacent,
        })
    return sorted(
        candidates,
        key=lambda item: (
            bool(item["exact_id"]),
            float(item["score"]),
            float(item["image_similarity"]),
            str(item["page_id"]),
        ),
        reverse=True,
    )[: max(1, limit)]


def _build_observation_candidate_map(
    graph: FlowGraph,
    current_plan: dict[str, Any],
    trace: list[dict[str, Any]],
    thumbnail_cache: _ScreenshotThumbnailCache | None = None,
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, dict[str, float]]]:
    candidate_map: dict[int, list[dict[str, Any]]] = {}
    step_similarity_cache: dict[int, dict[str, float]] = {}
    previous_page_ids: list[str] = []
    for index, step in enumerate(trace, start=1):
        raw_step = int(step.get("raw_step_index", index) or index)
        similarity_sink: dict[str, float] = {}
        candidates = _retrieve_page_candidates(
            graph,
            step,
            current_plan,
            previous_page_ids=previous_page_ids,
            thumbnail_cache=thumbnail_cache,
            similarity_sink=similarity_sink,
        )
        if candidates:
            candidate_map[raw_step] = candidates
            previous_page_ids = [str(item.get("page_id", "")) for item in candidates]
        else:
            previous_page_ids = []
        if similarity_sink:
            step_similarity_cache[raw_step] = similarity_sink
    return candidate_map, step_similarity_cache


def _boundary_context(
    graph: FlowGraph,
    selected_page_ids: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = set(selected_page_ids)
    boundary_ids: list[str] = []
    boundary_edges: list[dict[str, Any]] = []
    for transition in graph.page_transitions:
        src_selected = transition.src_page_id in selected
        dst_selected = transition.dst_page_id in selected
        if src_selected == dst_selected:
            continue
        outside_id = transition.dst_page_id if src_selected else transition.src_page_id
        if outside_id not in boundary_ids:
            boundary_ids.append(outside_id)
        boundary_edges.append({
            "src_page_id": transition.src_page_id,
            "dst_page_id": transition.dst_page_id,
            "operation": _clip_context_text(transition.operation, 240),
        })
    boundary_ids = boundary_ids[:FFG_UPDATE_MAX_BOUNDARY_PAGES]
    retained = selected.union(boundary_ids)
    boundary_edges = [
        edge
        for edge in boundary_edges
        if edge["src_page_id"] in retained and edge["dst_page_id"] in retained
    ]
    boundary_pages = [
        {
            "id": page.id,
            "name": _clip_context_text(page.name or page.description, 180),
            "activity": _clip_context_text(page.activity, 180),
            "owners": graph.get_page_owners(page_id),
        }
        for page_id in boundary_ids
        for page in [graph.get_page(page_id)]
        if page is not None
    ]
    return boundary_pages, boundary_edges


def _build_ffg_summary(
    graph: FlowGraph,
    current_plan: dict[str, Any] | None = None,
    trace: list[dict[str, Any]] | None = None,
    candidate_map: dict[int, list[dict[str, Any]]] | None = None,
    granularity_review_function_id: str = "",
) -> str:
    current_plan = current_plan or {}
    trace = trace or []
    focused_function_ids = _focused_update_function_ids(graph, current_plan, trace)
    focused_id_set = set(focused_function_ids)
    selected_page_ids: list[str] = []

    def add_page(page_id: object):
        resolved = graph.resolve_page_id(str(page_id or "").strip())
        if resolved and graph.get_page(resolved) is not None and resolved not in selected_page_ids:
            selected_page_ids.append(resolved)

    for function_id in focused_function_ids:
        node = graph.get_node(function_id)
        if node is None:
            continue
        for page_id in node.page_ids:
            add_page(page_id)
    trace_activities = {
        _page_activity(_extract_page(step))
        for step in trace
        if _page_activity(_extract_page(step))
    }
    for page in graph.active_pages():
        if page.activity and page.activity in trace_activities:
            add_page(page.id)
    selected_page_ids = selected_page_ids[:FFG_UPDATE_MAX_FOCUSED_PAGES]
    selected_page_id_set = set(selected_page_ids)
    boundary_pages, boundary_transitions = _boundary_context(graph, selected_page_ids)
    if not candidate_map:
        candidate_map, _ = _build_observation_candidate_map(
            graph,
            current_plan,
            trace,
        )

    globally_sorted_nodes = sorted(
        graph.functions.values(),
        key=lambda item: (
            (item.test_status or "not_started") in TERMINAL_FUNCTION_STATUSES,
            len(item.scenarios),
            item.id,
        ),
    )
    facts = {
        "granularity_review": {
            "function_id": granularity_review_function_id,
            "required": bool(granularity_review_function_id),
        },
        "context_scope": {
            "focused_function_ids": focused_function_ids,
            "selected_page_ids": selected_page_ids,
        },
        "global_function_index": (
            [
                {
                    "id": node.id,
                    "name": _clip_context_text(node.name or node.description, 160),
                    "pages": len(node.page_ids),
                }
                for node in globally_sorted_nodes[:FFG_UPDATE_MAX_GLOBAL_FUNCTIONS]
            ]
            if len(graph.functions) > len(focused_function_ids)
            else []
        ),
        "functions": [
            {
                "id": node.id,
                "name": _clip_context_text(node.name, 180),
                "description": _clip_context_text(node.description),
                "page_ids": list(node.page_ids),
                "entry_page_ids": list(node.entry_page_ids),
                "resource_variables": [
                    _clip_context_text(value, 240) for value in node.resource_variables[:8]
                ],
                "scenarios": (
                    [
                        {
                            "scenario_ref": f"scenario_{index}",
                            "text": _clip_context_text(scenario, 360),
                        }
                        for index, scenario in enumerate(
                            node.scenarios[:FFG_UPDATE_MAX_GRANULARITY_SCENARIOS],
                            start=1,
                        )
                    ]
                    if node.id == granularity_review_function_id
                    else []
                ),
            }
            for function_id in focused_function_ids
            for node in [graph.get_node(function_id)]
            if node is not None
        ],
        "pages": [
            {
                "id": page.id,
                "name": _clip_context_text(page.name, 180),
                "description": _clip_context_text(page.description),
                "activity": _clip_context_text(page.activity, 180),
                "owners": graph.get_page_owners(page_id),
                "coverage_owners": graph.get_page_coverage_owners(page_id),
                "screenshot_count": int(bool(page.representative_screenshot)) + len(page.evidence_screenshots),
                "controls": [
                    _clip_context_text(value, 240) for value in page.controls[-30:]
                ],
                "operations": [
                    _clip_context_text(value, 240) for value in page.operations[-5:]
                ],
            }
            for page_id in selected_page_ids
            for page in [graph.get_page(page_id)]
            if page is not None
        ],
        "boundary_pages": boundary_pages,
        "boundary_transitions": boundary_transitions,
        "observation_page_candidates": [
            {
                "source_step": raw_step,
                "candidates": [
                    {
                        "page_id": item.get("page_id"),
                        "owners": item.get("owners"),
                        "name": item.get("name"),
                        "activity": item.get("activity"),
                    }
                    for item in candidates
                ],
            }
            for raw_step, candidates in candidate_map.items()
        ],
        "page_transitions": [
            {
                "src_page_id": transition.src_page_id,
                "dst_page_id": transition.dst_page_id,
                "operation": _clip_context_text(transition.operation, 240),
            }
            for transition in graph.page_transitions
            if transition.src_page_id in selected_page_id_set
            and transition.dst_page_id in selected_page_id_set
        ],
        "function_dependencies": [
            {
                "src_function_id": edge.src_function_id,
                "dst_function_id": edge.dst_function_id,
                "resource_variable": _clip_context_text(edge.resource_variable, 240),
            }
            for edge in graph.function_dependencies
            if edge.src_function_id in focused_id_set or edge.dst_function_id in focused_id_set
        ],
    }
    serialized = json.dumps(facts, ensure_ascii=False, separators=(",", ":"))
    # 裁减顺序：优先裁对模型决策帮助最小、噪声最大的字段（观察候选
    # 页、全局索引、跨功能依赖），最后才裁边界信息——页面归属边界
    # 是 Function 拆分/合并判断的核心证据，必须尽量保留。
    shrink_order = (
        "observation_page_candidates",
        "global_function_index",
        "function_dependencies",
        "page_transitions",
        "pages",
        "boundary_transitions",
        "boundary_pages",
        "functions",
    )
    minimum_sizes = {
        "functions": min(3, len(facts["functions"])),
        "pages": min(8, len(facts["pages"])),
        "global_function_index": min(
            20, len(facts.get("global_function_index", []))
        ),
    }
    while len(serialized) > FFG_UPDATE_SUMMARY_MAX_CHARS:
        changed = False
        for key in shrink_order:
            values = facts.get(key, [])
            if not values:
                continue
            minimum = minimum_sizes.get(key, 0)
            if len(values) > minimum:
                values.pop()
                changed = True
                break
        if not changed:
            break
        serialized = json.dumps(facts, ensure_ascii=False, separators=(",", ":"))
    # 裁剪完成后再统计省略数量，保证与模型实际看到的内容一致。
    facts["context_scope"]["omitted_functions"] = max(
        0, len(graph.functions) - len(facts["functions"])
    )
    facts["context_scope"]["omitted_pages"] = max(
        0, len(graph.active_pages()) - len(facts["pages"])
    )
    serialized = json.dumps(facts, ensure_ascii=False, separators=(",", ":"))
    return "【当前 FFG 聚焦事实快照】\n" + serialized


def _function_context(graph: FlowGraph, function_id: object) -> dict[str, Any]:
    normalized_id = str(function_id or "").strip()
    node = graph.get_node(normalized_id)
    return {
        "id": graph.resolve_function_id(normalized_id) if node is not None else normalized_id,
        "name": (node.name or node.description) if node is not None else "",
        "exists_in_ffg": node is not None,
    }


def _build_plan_context(graph: FlowGraph, current_plan: dict[str, Any]) -> str:
    scenario_type = str(current_plan.get("scenario_type", "")).strip()
    facts = {
        "scenario_type": scenario_type,
        "target_function": _function_context(
            graph,
            current_plan.get("target_function_id"),
        ),
        "src_function": _function_context(
            graph,
            current_plan.get("src_function_id"),
        ),
        "dst_function": _function_context(
            graph,
            current_plan.get("dst_function_id"),
        ),
    }
    return (
        "【本轮权威规划作用域】\n"
        + json.dumps(facts, ensure_ascii=False, indent=2)
        + "\n- 上述 func_x 来自已校验的规划器输出，优先级高于测试场景自然语言。"
        + "\n- scenario_type 只限定本轮测试目标，不决定新页面的永久 Function owner。"
        + "\n- intra_test 进入独立业务能力时可以创建新 Function；中间页可归入目标 Function，稳定入口/结果页可被多个 Function 共享。"
        + "\n- cross_test 只在直接证据支持时连接 src_function 与 dst_function。"
        + "\n- 不得因为名称相似而把本轮证据归到其他 func_x。"
    )


def _build_function_boundary_context(
    graph: FlowGraph,
    current_plan: dict[str, Any],
    trace: list[dict[str, Any]],
    granularity_review_function_id: str = "",
    granularity_review_metrics: dict[str, Any] | None = None,
) -> str:
    target = graph.get_node(str(current_plan.get("target_function_id", "")).strip())
    target_pages = (
        [
            page
            for page_id in target.page_ids
            if (page := graph.get_page(page_id)) is not None
        ]
        if target is not None
        else []
    )
    target_activities = {
        str(page.activity or "").strip() for page in target_pages if page.activity
    }
    trace_activities = {
        activity
        for step in trace
        if (activity := str(_page_activity(_extract_page(step)) or "").strip())
    }
    unseen_activities = sorted(trace_activities - target_activities)
    mandatory_review = bool(
        len(target_pages) >= 4
        or len(trace) >= 4
        or unseen_activities
    )
    facts = {
        "target_function_id": target.id if target is not None else "",
        "target_function_page_count": len(target_pages),
        "target_function_activities": sorted(target_activities),
        "trace_activities_not_yet_in_target": unseen_activities,
        "mandatory_boundary_review": mandatory_review,
        "granularity_review_function_id": granularity_review_function_id,
        "granularity_review_metrics": granularity_review_metrics or {},
    }
    return (
        "【Function 边界审查】\n"
        + json.dumps(facts, ensure_ascii=False, indent=2)
        + "\n- 对本轮每个新 Page 判断它是当前能力的中间状态、共享页面，还是具有独立目标和闭环的新 Function。"
        + "\n- mandatory_boundary_review=true 时，禁止不加说明地把全部新 Page 继续归入 target_function。"
        + "\n- granularity_review_function_id 非空时，还要复核该已有 Function 的全部持久化页面；大功能下的子功能都应拆分为单独的功能节点：共享同一入口（如设置、我的、管理页）但各自承载独立配置域或独立业务流程的页面，应使用 function_restructures 拆分（优先 retire_source=false 的部分拆分，先把最明显的独立子域拆出，不必一次性全拆），而不是只处理本轮新页面。若决定不拆分，必须在 update_control.reason 中说明这些页面为何属于同一用户目标；无理由的拒绝视为复核失败。"
        + "\n- Activity 变化只是审查信号，不可单独作为创建 Function 的充分条件。"
    )


def _screenshot_dimensions(screenshot: str) -> tuple[int, int] | None:
    screenshot = str(screenshot or "").strip()
    if not screenshot:
        return None
    if screenshot.startswith("data:image") and "," in screenshot:
        screenshot = screenshot.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(screenshot, validate=False)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            return None
        height, width = image.shape[:2]
        return width, height
    except Exception:
        return None


def _remaining_run_seconds(state: AgentState) -> float | None:
    """Return the workflow's remaining wall-clock budget when it is available."""
    try:
        start_time = float(state.get("start_time", 0) or 0)
        total_time_limit = float(state.get("total_time_limit", 0) or 0)
    except (TypeError, ValueError):
        return None
    if start_time <= 0 or total_time_limit <= 0:
        return None
    return max(0.0, total_time_limit - (time.time() - start_time))


def _format_trace_step(step: dict[str, Any], raw_index: int, recordable_index: int) -> str:
    page = _extract_page(step)
    if isinstance(page, Page):
        runtime_page_id = str(page.id or "")
    elif isinstance(page, dict):
        runtime_page_id = str(page.get("id", ""))
    else:
        runtime_page_id = ""
    fields = [
        f"[STEP] index={raw_index}",
        f"recordable_index={recordable_index}",
        f"phase={str(step.get('trace_phase', '')).strip() or 'unknown'}",
        f"action={str(step.get('action_name', '')).strip() or 'unknown'}",
        f"runtime_page_id={runtime_page_id}",
        f"activity={_page_activity(page) or 'unknown'}",
    ]
    dimensions = _screenshot_dimensions(_page_screenshot(page))
    if dimensions:
        fields.append(
            f"original_screenshot_size={dimensions[0]}x{dimensions[1]}"
        )
    if step.get("x") is not None and step.get("y") is not None:
        fields.append(f"coordinate=({step.get('x')},{step.get('y')})")
    for key in ("content", "direction", "app_name"):
        value = str(step.get(key, "")).strip()
        if value:
            fields.append(f"{key}={value}")
    action_description = str(step.get("action_description", "")).strip()
    thought = str(step.get("thought", "")).strip()
    if action_description:
        fields.append(f"executor_target_hint={action_description}")
    if thought:
        fields.append(f"executor_reasoning_hint={thought}")
    if step.get("caused_external_transition"):
        fields.append(
            "caused_external_transition=true"
            f", external_package={step.get('external_package', '')}"
        )
    if step.get("caused_target_crash"):
        fields.append("caused_target_crash=true")
    if step.get("action_failed"):
        fields.append(
            "action_failed=true"
            f", action_error={str(step.get('action_error', '')).strip() or 'unknown'}"
        )
    if step.get("is_interruption"):
        fields.append("is_interruption=true")
    if step.get("external_context"):
        fields.append(
            "external_context=true"
            f", external_package={str(step.get('external_package', '')).strip() or 'unknown'}"
        )
    if step.get("foreground_recovery"):
        fields.append(
            "foreground_recovery=true"
            f", recovery_method={str(step.get('recovery_method', '')).strip() or 'unknown'}"
        )
    if step.get("observation_only"):
        fields.append("observation_only=true")
    if step.get("terminal_observation"):
        fields.append("terminal_observation=true")
    if str(step.get("screenshot_health_error", "")).strip():
        fields.append(
            "screenshot_health_error="
            + str(step.get("screenshot_health_error", "")).strip()
        )
    return ", ".join(fields)


def _refinement_focus_steps(
    issues: list[object],
    trace: list[dict[str, Any]],
) -> set[int] | None:
    """Return issue steps plus their neighbors; None means a global review."""
    if not issues:
        return None
    referenced: set[int] = set()
    for issue in issues:
        issue_text = str(issue)
        referenced.update(
            int(value)
            for value in re.findall(
                r"source_step\s+(\d+)", issue_text, re.IGNORECASE
            )
        )
        referenced.update(
            int(value)
            for value in re.findall(
                r"(?:reuse|create|ignore)?\s*alignment\s+(\d+)",
                issue_text,
                re.IGNORECASE,
            )
        )
    if not referenced:
        return None
    ordered_steps = [
        int(step.get("raw_step_index", index) or index)
        for index, step in enumerate(trace, start=1)
    ]
    positions = {raw_step: index for index, raw_step in enumerate(ordered_steps)}
    focused: set[int] = set()
    for raw_step in referenced:
        position = positions.get(raw_step)
        if position is None:
            continue
        for neighbor in ordered_steps[max(0, position - 1): position + 2]:
            focused.add(neighbor)
    return focused or None


def _reference_pages_for_update(
    graph: FlowGraph,
    current_plan: dict[str, Any],
    trace: list[dict[str, Any]],
    candidate_map: dict[int, list[dict[str, Any]]] | None = None,
    thumbnail_cache: _ScreenshotThumbnailCache | None = None,
    step_similarity_cache: dict[int, dict[str, float]] | None = None,
) -> list[Page]:
    if FFG_UPDATE_MAX_REFERENCE_IMAGES <= 0:
        return []

    prioritized_ids: list[str] = []

    def add_page_id(page_id: object):
        normalized_id = str(page_id or "").strip()
        if (
            normalized_id
            and normalized_id not in prioritized_ids
            and graph.get_page(normalized_id) is not None
        ):
            prioritized_ids.append(normalized_id)

    for candidates in (candidate_map or {}).values():
        for candidate in candidates:
            add_page_id(candidate.get("page_id"))

    for key in ("target_function_id", "src_function_id", "dst_function_id"):
        node = graph.get_node(str(current_plan.get(key, "")).strip())
        if node is None:
            continue
        for page_id in node.page_ids:
            add_page_id(page_id)

    scored_pages: list[tuple[float, str]] = []
    for index, step in enumerate(trace, start=1):
        trace_page = _extract_page(step)
        trace_screenshot = _page_screenshot(trace_page)
        trace_activity = _page_activity(trace_page)
        if not trace_screenshot:
            continue
        raw_step = int(step.get("raw_step_index", index) or index)
        cached_by_page = (step_similarity_cache or {}).get(raw_step, {})
        for page in graph.active_pages():
            if trace_activity and page.activity and page.activity != trace_activity:
                continue
            page_id = str(page.id or "")
            similarity = cached_by_page.get(page_id)
            if similarity is None:
                similarities = []
                for stored in [page.representative_screenshot, *page.evidence_screenshots]:
                    value = (
                        thumbnail_cache.similarity(trace_screenshot, stored)
                        if thumbnail_cache is not None
                        else screenshot_similarity(trace_screenshot, stored)
                    )
                    if value is not None:
                        similarities.append(value)
                similarity = max(similarities, default=None)
            if similarity is not None and page_id:
                scored_pages.append((float(similarity), page_id))
    for _similarity, page_id in sorted(scored_pages, reverse=True)[:6]:
        add_page_id(page_id)

    trace_activities = {
        _page_activity(_extract_page(step))
        for step in trace
        if _page_activity(_extract_page(step))
    }
    for page_id, page in graph.pages.items():
        if page.activity and page.activity in trace_activities:
            add_page_id(page_id)

    return [
        page
        for page_id in prioritized_ids[:FFG_UPDATE_MAX_REFERENCE_IMAGES]
        if (page := graph.get_page(page_id)) is not None and page.representative_screenshot
    ]


def _snapshot_graph(graph: FlowGraph) -> dict[str, int]:
    return {
        "functions": len(graph.functions),
        "pages": len(graph.active_pages()),
        "operations": sum(len(page.operations) for page in graph.active_pages()),
        "page_transitions": len(graph.page_transitions),
        "function_dependencies": len(graph.function_dependencies),
    }


def _next_numeric_id(existing: list[str], prefix: str) -> int:
    next_index = 1
    for item in existing:
        if item.startswith(prefix) and item[len(prefix):].isdigit():
            next_index = max(next_index, int(item[len(prefix):]) + 1)
    return next_index


def _normalize_ref(value: object) -> str:
    return str(value or "").strip()


def _function_ref_values(item: dict[str, Any]) -> list[str]:
    """Return functions whose behavior is implemented by the Page."""
    values = item.get("function_refs")
    refs = values if isinstance(values, list) else []
    return list(dict.fromkeys(
        ref for value in refs if (ref := _normalize_ref(value))
    ))


def _entry_function_ref_values(item: dict[str, Any]) -> list[str]:
    values = item.get("entry_function_refs")
    refs = values if isinstance(values, list) else []
    return list(dict.fromkeys(
        ref for value in refs if (ref := _normalize_ref(value))
    ))


def _all_function_ref_values(item: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys([
        *_function_ref_values(item),
        *_entry_function_ref_values(item),
    ]))


def _list_patch(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _normalize_observation_alignment(
    patch: dict[str, Any],
    graph: FlowGraph,
    trace: list[dict[str, Any]],
    candidate_map: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Repair only alignments whose intended mapping is deterministic."""
    alignments = [dict(item) for item in _list_patch(patch.get("observation_alignment"))]
    page_patch_by_ref = {
        _normalize_ref(item.get("page_ref")): item
        for item in _list_patch(patch.get("pages"))
        if _normalize_ref(item.get("page_ref"))
    }
    page_patch_by_step = {
        int(str(item.get("source_step")).strip()): item
        for item in _list_patch(patch.get("pages"))
        if str(item.get("source_step", "")).strip().isdigit()
    }
    trace_by_step = {
        int(step.get("raw_step_index", index) or index): step
        for index, step in enumerate(trace, start=1)
        if _extract_page(step) is not None
    }
    for item in alignments:
        source_step = item.get("source_step")
        if not str(source_step).strip().isdigit():
            continue
        raw_step = int(str(source_step).strip())
        decision = str(item.get("page_decision", "") or "").strip().lower()
        page_ref = _normalize_ref(item.get("page_ref"))
        if decision == "reuse" and page_ref in page_patch_by_ref:
            item["page_decision"] = "create"
            continue
        if decision == "create":
            # 模型常把“复用已有页面”写成 create + 已存在的 page_id。此时既没有
            # 对应的 pages patch（无新页面可建），意图只可能是 reuse，属于纯格式
            # 错误：由代码归一化即可，不必再花一次多模态 refinement 调用。
            if page_ref.startswith("new_page_"):
                if page_ref in page_patch_by_ref:
                    continue
                page_patch = page_patch_by_step.get(raw_step)
                patch_ref = _normalize_ref((page_patch or {}).get("page_ref"))
                if patch_ref:
                    item["page_ref"] = patch_ref
                continue
            if page_ref in page_patch_by_ref:
                continue
            if graph.get_page(page_ref) is None:
                continue
            candidate_ids = {
                str(candidate.get("page_id", ""))
                for candidate in candidate_map.get(raw_step, [])
            }
            # 必须由检索候选独立佐证“这一步确实落在该已有页面上”，才敢改判 reuse；
            # 否则可能把一个真正的新页面并进错误的已有页面，反而丢掉页面覆盖。
            if page_ref not in candidate_ids:
                continue
            item["page_decision"] = "reuse"
            continue
        if decision != "ignore":
            continue
        step = trace_by_step.get(raw_step)
        if step is None or str(step.get("screenshot_health_error", "") or "").strip():
            continue
        if bool(step.get("is_interruption")) and bool(step.get("external_context")):
            continue
        page_patch = page_patch_by_step.get(raw_step)
        if page_patch is not None:
            item["page_decision"] = "create"
            item["page_ref"] = _normalize_ref(page_patch.get("page_ref"))
            continue
        candidates = candidate_map.get(raw_step, [])
        if candidates:
            best = candidates[0]
            high_confidence_visual_match = (
                bool(best.get("activity_match"))
                and float(best.get("image_similarity", 0.0) or 0.0)
                >= FFG_UPDATE_MATCHED_IMAGE_THRESHOLD
            )
            if bool(best.get("exact_id")) or high_confidence_visual_match:
                item["page_decision"] = "reuse"
                item["page_ref"] = str(best.get("page_id", ""))
    aligned_steps = {
        int(item.get("source_step"))
        for item in alignments
        if str(item.get("source_step", "")).strip().isdigit()
    }
    for index, step in enumerate(trace, start=1):
        raw_step = int(step.get("raw_step_index", index) or index)
        if raw_step in aligned_steps or _extract_page(step) is None:
            continue
        if str(step.get("screenshot_health_error", "") or "").strip():
            alignments.append({
                "source_step": raw_step,
                "page_decision": "ignore",
                "page_ref": "",
            })
            aligned_steps.add(raw_step)
            continue
        candidates = candidate_map.get(raw_step, [])
        if not candidates:
            continue
        best = candidates[0]
        if bool(best.get("exact_id")) or float(best.get("score", 0.0) or 0.0) >= 0.97:
            alignments.append({
                "source_step": raw_step,
                "page_decision": "reuse",
                "page_ref": str(best.get("page_id", "")),
            })
            aligned_steps.add(raw_step)
    patch["observation_alignment"] = alignments
    return alignments


# 仅描述“patch 与检索候选集偏差”的告警：页面本身已确认存在于图中，图结构
# 依然自洽，因此不值得为它整轮回滚（会连同本轮所有新页面/操作/跳转一起丢弃）。
# 其余对齐问题（缺失/重复/无锚点的 create）仍属硬错误，必须回滚以防幻觉页面。
ALIGNMENT_ADVISORY_MARKERS = (
    "is outside retrieved candidates",
    # 功能拆分时把非法页面引用（如本 patch 新建的 new_page_x）塞进
    # replacement.page_refs：剔除该引用后图结构仍自洽，拆分结果只是
    # 页面归属不完整，不值得为它整轮回滚。
    "cannot move non-coverage page",
    "has no valid coverage pages",
)


def _is_ignorable_observation_step(step: dict[str, Any]) -> bool:
    """程序可判定为"可忽略观测"的轨迹步骤。

    包括：截图健康错误、运行器自动恢复/干扰步骤（is_interruption、
    foreground_recovery、observation_after_recovery）、以及页面观测本身
    位于目标应用之外（external_context，如桌面、其他 App）。这些步骤
    不参与目标应用的 Page 建模，模型标 ignore 是唯一正确答案。

    旧校验只认 (is_interruption 且 external_context) 同时成立，而自动
    恢复步骤只带 is_interruption、受控外部页面只带 external_context，
    导致模型无论标 ignore 还是漏掉都必然报错，形成 refinement 死循环。
    """
    return bool(
        str(step.get("screenshot_health_error", "") or "").strip()
        or step.get("is_interruption")
        or step.get("foreground_recovery")
        or step.get("observation_after_recovery")
        or step.get("external_context")
    )


def _ensure_ignorable_alignments(
    patch: dict[str, Any],
    trace: list[dict[str, Any]],
) -> int:
    """为程序判定"可忽略"的轨迹步骤程序化补齐/矫正 ignore 对齐项。

    小参数模型在长轨迹上容易整段遗漏 observation_alignment，或对外部/
    恢复步骤给出错误 decision；这些步骤的正确处理方式由程序即可判定
    （ignore），无需依赖模型。返回程序补齐或矫正的对齐项数量。
    """
    alignments = patch.get("observation_alignment")
    if alignments is None or not isinstance(alignments, list):
        alignments = []
    patch["observation_alignment"] = alignments
    by_step: dict[int, dict[str, Any]] = {}
    for item in alignments:
        if isinstance(item, dict) and str(item.get("source_step", "")).strip().isdigit():
            by_step[int(str(item["source_step"]).strip())] = item
    fixed = 0
    for index, step in enumerate(trace, start=1):
        if _extract_page(step) is None:
            continue
        raw_step = int(step.get("raw_step_index", index) or index)
        if not _is_ignorable_observation_step(step):
            continue
        existing = by_step.get(raw_step)
        if existing is None:
            alignments.append({"source_step": raw_step, "page_decision": "ignore"})
            fixed += 1
        elif str(existing.get("page_decision", "") or "").strip().lower() != "ignore":
            existing["page_decision"] = "ignore"
            existing.pop("page_ref", None)
            fixed += 1
    return fixed


def partition_alignment_issues(
    issues: list[str],
) -> tuple[list[str], list[str]]:
    """Split alignment issues into (blocking, advisory)."""
    blocking: list[str] = []
    advisory: list[str] = []
    for issue in issues:
        text = str(issue or "")
        if not text:
            continue
        if any(marker in text for marker in ALIGNMENT_ADVISORY_MARKERS):
            advisory.append(text)
        else:
            blocking.append(text)
    return blocking, advisory


def validate_observation_alignment(
    graph: FlowGraph,
    patch: dict[str, Any],
    trace: list[dict[str, Any]],
    candidate_map: dict[int, list[dict[str, Any]]],
) -> list[str]:
    issues: list[str] = []
    alignments = _list_patch(patch.get("observation_alignment"))
    by_step: dict[int, dict[str, Any]] = {}
    for item in alignments:
        source_step = item.get("source_step")
        if not str(source_step).strip().isdigit():
            issues.append("observation_alignment contains invalid source_step")
            continue
        raw_step = int(str(source_step).strip())
        if raw_step in by_step:
            issues.append(f"observation_alignment duplicates source_step {raw_step}")
            continue
        by_step[raw_step] = item

    page_patch_by_step = {
        int(str(item.get("source_step")).strip()): item
        for item in _list_patch(patch.get("pages"))
        if str(item.get("source_step", "")).strip().isdigit()
    }
    for index, step in enumerate(trace, start=1):
        if _extract_page(step) is None:
            continue
        raw_step = int(step.get("raw_step_index", index) or index)
        alignment = by_step.get(raw_step)
        if alignment is None:
            issues.append(f"missing observation_alignment for source_step {raw_step}")
            continue
        decision = str(alignment.get("page_decision", "") or "").strip().lower()
        page_ref = _normalize_ref(alignment.get("page_ref"))
        if decision not in {"reuse", "create", "ignore"}:
            issues.append(f"invalid page_decision for source_step {raw_step}: {decision or 'empty'}")
            continue
        if decision == "reuse":
            if graph.get_page(page_ref) is None:
                issues.append(f"reuse alignment references unknown Page {page_ref or 'empty'}")
                continue
            candidate_ids = {
                str(candidate.get("page_id", ""))
                for candidate in candidate_map.get(raw_step, [])
            }
            if candidate_ids and page_ref not in candidate_ids:
                issues.append(
                    f"reuse alignment {raw_step}->{page_ref} is outside retrieved candidates"
                )
        elif decision == "create":
            page_patch = page_patch_by_step.get(raw_step)
            if not page_ref.startswith("new_page_"):
                issues.append(f"create alignment must use new_page_x at source_step {raw_step}")
            elif page_patch is None or _normalize_ref(page_patch.get("page_ref")) != page_ref:
                issues.append(f"create alignment lacks matching pages patch at source_step {raw_step}")
        elif not _is_ignorable_observation_step(step):
            issues.append(f"ignore alignment lacks an ignorable observation at source_step {raw_step}")
    return list(dict.fromkeys(issues))


def _resolve_function_ref(ref: str, temp_refs: dict[str, str], graph: FlowGraph) -> str:
    ref = _normalize_ref(ref)
    if not ref:
        return ""
    resolved = graph.resolve_function_id(ref)
    if resolved and graph.get_node(resolved) is not None:
        return resolved
    return temp_refs.get(ref, "")


def _resolve_page_ref(ref: str, temp_refs: dict[str, str], graph: FlowGraph) -> str:
    ref = _normalize_ref(ref)
    if not ref:
        return ""
    resolved = graph.resolve_page_id(ref)
    if resolved and graph.get_page(resolved) is not None:
        return resolved
    return temp_refs.get(ref, "")


def _allocate_function_id(graph: FlowGraph, next_index: int) -> tuple[str, int]:
    function_id = f"func_{next_index}"
    while function_id in graph.functions:
        next_index += 1
        function_id = f"func_{next_index}"
    return function_id, next_index + 1


def _allocate_page_id(graph: FlowGraph, next_index: int) -> tuple[str, int]:
    page_id = f"page_{next_index}"
    while page_id in graph.pages:
        next_index += 1
        page_id = f"page_{next_index}"
    return page_id, next_index + 1


def _function_match_key(value: object) -> str:
    return TEXT_NORMALIZE_PATTERN.sub("", str(value or "").strip().lower())


def _entity_generalized_key(value: object) -> str:
    # Keep identity evidence intact. Concrete entity-name stripping used to be
    # driven by a small App-specific vocabulary and could merge unrelated
    # Functions/Pages. Explicit model function_merges remains the safe place for
    # evidence-backed semantic aggregation.
    return _function_match_key(value)


def _find_exact_duplicate_function_id(
    graph: FlowGraph,
    item: dict[str, Any],
    accepted_keys: dict[str, str],
) -> str:
    name = str(item.get("name", "")).strip()
    description = str(item.get("description", "")).strip()
    key = _function_match_key(f"{name} {description}")
    if not key:
        return "__skip__"
    if key in accepted_keys:
        return accepted_keys[key]
    name_key = _function_match_key(name)
    description_key = _function_match_key(description)
    semantic_name_key = _entity_generalized_key(name)
    semantic_lookup_key = f"semantic:{semantic_name_key}"
    if len(semantic_name_key) >= 6 and semantic_lookup_key in accepted_keys:
        return accepted_keys[semantic_lookup_key]
    for node in graph.functions.values():
        existing_key = _function_match_key(f"{node.name} {node.description}")
        if existing_key and key == existing_key:
            return node.id

    for node in graph.functions.values():
        existing_name_key = _function_match_key(node.name)
        existing_description_key = _function_match_key(node.description)
        if name_key and description_key and (
            name_key == existing_name_key
            and description_key == existing_description_key
        ):
            return node.id
        existing_semantic_name_key = _entity_generalized_key(node.name)
        if (
            len(semantic_name_key) >= 6
            and semantic_name_key == existing_semantic_name_key
        ):
            return node.id
    return ""


def _patch_resource_variables(item: dict[str, Any]) -> list[str]:
    resources: list[str] = []
    seen: set[str] = set()
    for value in item.get("resource_variables", []):
        if not isinstance(value, str):
            continue
        resource = value.strip()
        key = _function_match_key(resource)
        if not resource or not key or key in seen:
            continue
        seen.add(key)
        resources.append(resource)
    return resources


def _merge_function_resource_variables(
    graph: FlowGraph,
    function_id: str,
    resources: list[str],
) -> bool:
    node = graph.get_node(function_id)
    if node is None:
        return False
    before = list(node.resource_variables)
    for resource in resources:
        node.add_resource_variable(resource)
    return before != node.resource_variables


def _source_page(trace: list[dict[str, Any]], source_step: object) -> Page | dict[str, Any] | None:
    if isinstance(source_step, int):
        for step in trace:
            if int(step.get("raw_step_index", 0) or 0) == source_step:
                return _extract_page(step)
    if isinstance(source_step, str) and source_step.strip().isdigit():
        return _source_page(trace, int(source_step.strip()))
    return None


def _merge_page_observation(
    page: Page,
    source_page: Page | dict[str, Any],
    controls: list[str] | None = None,
):
    for control in [*_natural_language_controls(source_page), *(controls or [])]:
        page.add_control(str(control))

    screenshot = _page_screenshot(source_page)
    if screenshot:
        # 已有代表图通常是首次确认的稳定页面。后续观测可能是键盘、弹窗、
        # 表情面板或全屏媒体等临时状态，只作为证据保留，避免最后一次观测
        # 覆盖稳定代表图并污染后续页面识别。
        if page.representative_screenshot:
            page.add_evidence_screenshot(screenshot)
        else:
            page.update_representative_screenshot(screenshot)
    page.activity = page.activity or _page_activity(source_page)


def _find_entity_generalized_page(
    graph: FlowGraph,
    item: dict[str, Any],
    source_page: Page | dict[str, Any],
    target_function_ids: list[str] | None = None,
) -> Page | None:
    """Reuse a dynamic instance only inside an already shared Function."""
    semantic_key = _entity_generalized_key(item.get("name"))
    if len(semantic_key) < 4:
        return None
    activity = str(_page_activity(source_page) or "").strip()
    target_owners = set(target_function_ids or [])
    for page in graph.active_pages():
        if activity and page.activity and activity != page.activity:
            continue
        existing_owners = set(graph.get_page_owners(str(page.id or "")))
        if target_owners and existing_owners and target_owners.isdisjoint(existing_owners):
            continue
        if semantic_key == _entity_generalized_key(page.name):
            return page
    return None


def _control_signature(controls: list[str]) -> set[str]:
    """Normalized control-family signature used for deterministic page identity."""
    return {
        key
        for control in controls
        if (key := control_family_key(control) or _function_match_key(control))
    }


def _find_control_signature_page(
    graph: FlowGraph,
    source_page: Page | dict[str, Any],
    declared_controls: list[str],
    declared_name: str = "",
    target_function_ids: list[str] | None = None,
) -> Page | None:
    """Reuse an existing Page whose activity and control set already match.

    Page ids are allocated monotonically and there is no way to merge them back,
    so splitting one logical page into several ``page_N`` permanently inflates the
    coverage denominator and scatters operations. When the same activity exposes
    the same control family set, it is the same logical page regardless of the
    name the model invented this round. Pages already isolated under different
    Functions are not merged merely because a shared Activity looks similar.
    """
    activity = str(_page_activity(source_page) or "").strip()
    if not activity:
        return None
    signature = _control_signature(declared_controls)
    if len(signature) < PAGE_SIGNATURE_MIN_CONTROLS:
        return None
    declared_name_key = _entity_generalized_key(declared_name)
    target_owners = set(target_function_ids or [])
    best_page: Page | None = None
    best_ratio = 0.0
    for page in graph.active_pages():
        if not page.activity or page.activity != activity:
            continue
        existing_owners = set(graph.get_page_owners(str(page.id or "")))
        if target_owners and existing_owners and target_owners.isdisjoint(existing_owners):
            continue
        existing_name_key = _entity_generalized_key(page.name)
        if (
            declared_name_key
            and existing_name_key
            and declared_name_key != existing_name_key
        ):
            # Many Apps reuse one Activity and layout for different business
            # objects; visible identity must outrank Activity/control similarity.
            continue
        existing = _control_signature(list(page.controls))
        if len(existing) < PAGE_SIGNATURE_MIN_CONTROLS:
            continue
        overlap = len(signature & existing)
        union = len(signature | existing)
        ratio = overlap / union if union else 0.0
        if ratio > best_ratio:
            best_ratio, best_page = ratio, page
    if best_page is not None and best_ratio >= PAGE_SIGNATURE_MATCH_RATIO:
        return best_page
    return None


def _restructure_replacements(item: dict[str, Any]) -> list[dict[str, Any]]:
    values = item.get("replacements")
    return _list_patch(values)


def _merge_source_refs(item: dict[str, Any]) -> list[str]:
    values = item.get("source_function_refs")
    values = values if isinstance(values, list) else []
    return list(dict.fromkeys(
        ref for value in values if (ref := _normalize_ref(value))
    ))


def _persisted_restructure_function_refs(
    graph: FlowGraph,
    patch_items: list[dict[str, Any]],
) -> set[str]:
    """Return new refs backed by Pages already persisted under the source.

    A global split intentionally may create several child Functions even though
    the current trace only revisits one of their Pages. Persisted Page facts are
    sufficient evidence here, provided every declared child owns at least one
    existing Page of the source Function.
    """
    backed_refs: set[str] = set()
    for item in patch_items:
        source_id = graph.resolve_function_id(
            _normalize_ref(item.get("source_function_ref"))
        )
        source = graph.get_node(source_id)
        if source is None:
            continue
        source_page_ids = set(source.coverage_page_ids)
        for replacement in _restructure_replacements(item):
            function_ref = _normalize_ref(replacement.get("function_ref"))
            page_refs = replacement.get("page_refs")
            page_refs = page_refs if isinstance(page_refs, list) else []
            resolved_page_ids = {
                graph.resolve_page_id(_normalize_ref(page_ref))
                for page_ref in page_refs
            }
            if (
                function_ref.startswith("new_func_")
                and resolved_page_ids
                and "" not in resolved_page_ids
                and resolved_page_ids.issubset(source_page_ids)
            ):
                backed_refs.add(function_ref)
    return backed_refs


def _persisted_merge_function_refs(
    graph: FlowGraph,
    patch_items: list[dict[str, Any]],
) -> set[str]:
    """Return new merge targets backed by at least two existing Functions."""
    backed_refs: set[str] = set()
    for item in patch_items:
        target_ref = _normalize_ref(item.get("target_function_ref"))
        source_ids = {
            graph.resolve_function_id(source_ref)
            for source_ref in _merge_source_refs(item)
        }
        if (
            target_ref.startswith("new_func_")
            and len(source_ids) >= 2
            and "" not in source_ids
        ):
            backed_refs.add(target_ref)
    return backed_refs


def _apply_functions(
    graph: FlowGraph,
    patch_items: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    function_refs: dict[str, str],
    page_refs: dict[str, str],
    next_function_index: int,
    persisted_evidence_refs: set[str] | None = None,
) -> int:
    accepted_keys: dict[str, str] = {}
    require_direct_evidence = bool(graph.functions)
    available_steps = {
        int(step.get("raw_step_index", 0) or 0)
        for step in trace
        if int(step.get("raw_step_index", 0) or 0) > 0
    }
    for item in patch_items:
        ref = _normalize_ref(item.get("function_ref"))
        if not ref or ref in function_refs:
            continue
        source_step = item.get("source_step")
        if isinstance(source_step, str) and source_step.strip().isdigit():
            source_step = int(source_step.strip())
        has_direct_evidence = (
            isinstance(source_step, int)
            and not isinstance(source_step, bool)
            and source_step in available_steps
        ) or ref in (persisted_evidence_refs or set())
        resources = _patch_resource_variables(item)
        existing_function_id = graph.resolve_function_id(ref)
        if existing_function_id:
            identity_fields = {
                key: item.get(key)
                for key in ("name", "description")
                if str(item.get(key, "") or "").strip()
            }
            if identity_fields and require_direct_evidence and not has_direct_evidence:
                print(
                    "[FFG 更新] 跳过缺少本轮直接证据的功能名称/描述更新: "
                    f"{existing_function_id}"
                )
            elif identity_fields:
                graph.update_function(existing_function_id, identity_fields)
            if resources and require_direct_evidence and not has_direct_evidence:
                print(
                    "[FFG 更新] 跳过缺少本轮直接证据的功能资源变量更新: "
                    f"{existing_function_id} | {resources}"
                )
            elif resources:
                _merge_function_resource_variables(
                    graph,
                    existing_function_id,
                    resources,
                )
            function_refs[ref] = existing_function_id
            continue
        if require_direct_evidence and not has_direct_evidence:
            print(
                "[FFG 更新] 跳过缺少本轮直接证据的新功能: "
                f"{str(item.get('name') or item.get('description') or ref).strip()}"
            )
            continue
        duplicate_function_id = _find_exact_duplicate_function_id(
            graph,
            item,
            accepted_keys,
        )
        if duplicate_function_id:
            if duplicate_function_id != "__skip__":
                function_refs[ref] = duplicate_function_id
                if resources:
                    _merge_function_resource_variables(
                        graph,
                        duplicate_function_id,
                        resources,
                    )
            print(
                "[FFG 更新] 跳过空内容或精确重复功能节点: "
                f"{str(item.get('name') or item.get('description') or ref).strip()}"
                f"{f' -> {duplicate_function_id}' if duplicate_function_id != '__skip__' else ''}"
            )
            continue
        name = str(item.get("name", "")).strip()
        description = str(item.get("description", "")).strip()
        function_id, next_function_index = _allocate_function_id(graph, next_function_index)
        accepted_keys[_function_match_key(f"{name} {description}")] = function_id
        semantic_name_key = _entity_generalized_key(name)
        if len(semantic_name_key) >= 6:
            accepted_keys[f"semantic:{semantic_name_key}"] = function_id
        node = FunctionNode(
            id=function_id,
            name=name or description or function_id,
            description=description or name or function_id,
            resource_variables=resources,
        )
        graph.add_node(node)
        function_refs[ref] = function_id
    return next_function_index


def _apply_pages(
    graph: FlowGraph,
    patch_items: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    function_refs: dict[str, str],
    page_refs: dict[str, str],
    trace_page_map: dict[int, str],
    next_page_index: int,
) -> int:
    for item in patch_items:
        ref = _normalize_ref(item.get("page_ref"))
        if not ref or ref in page_refs:
            continue
        function_ids = list(dict.fromkeys(
            function_id
            for function_ref in _function_ref_values(item)
            if (function_id := _resolve_function_ref(function_ref, function_refs, graph))
        ))
        entry_function_ids = list(dict.fromkeys(
            function_id
            for function_ref in _entry_function_ref_values(item)
            if (function_id := _resolve_function_ref(function_ref, function_refs, graph))
            and function_id not in function_ids
        ))
        if not function_ids and not entry_function_ids:
            continue
        source_step = item.get("source_step")
        source_page = _source_page(trace, source_step)
        if source_page is None:
            available_raw_steps = [
                int(step.get("raw_step_index", 0) or 0) for step in trace
            ]
            print(
                "[FFG 更新] 跳过缺少原始步骤证据的页面: "
                f"page_ref={ref}, source_step={source_step}, "
                f"available_raw_steps={available_raw_steps}"
            )
            continue
        raw_step = int(source_step) if str(source_step).strip().isdigit() else 0
        declared_controls = [
            str(value).strip()
            for value in item.get("controls", [])
            if isinstance(value, str) and str(value).strip()
        ]
        existing_page = graph.get_page(ref)
        if existing_page is not None:
            for function_id in function_ids:
                graph.add_page(existing_page, function_id)
            for function_id in entry_function_ids:
                graph.add_entry_page(existing_page, function_id)
            _merge_page_observation(existing_page, source_page, declared_controls)
            graph.update_page(existing_page.id or ref, item)
            page_refs[ref] = str(existing_page.id or ref)
            if raw_step:
                trace_page_map[raw_step] = str(existing_page.id or ref)
            continue
        generalized_page = _find_entity_generalized_page(
            graph,
            item,
            source_page,
            target_function_ids=function_ids,
        )
        if generalized_page is not None:
            for function_id in function_ids:
                graph.add_page(generalized_page, function_id)
            for function_id in entry_function_ids:
                graph.add_entry_page(generalized_page, function_id)
            _merge_page_observation(generalized_page, source_page, declared_controls)
            page_refs[ref] = str(generalized_page.id or "")
            if raw_step and generalized_page.id:
                trace_page_map[raw_step] = str(generalized_page.id)
            print(
                "[FFG 更新] 复用实体泛化后的同类页面: "
                f"{item.get('name', ref)} -> {generalized_page.id}"
            )
            continue
        signature_page = _find_control_signature_page(
            graph,
            source_page,
            declared_controls,
            declared_name=str(item.get("name", "") or ""),
            target_function_ids=function_ids,
        )
        if signature_page is not None:
            for function_id in function_ids:
                graph.add_page(signature_page, function_id)
            for function_id in entry_function_ids:
                graph.add_entry_page(signature_page, function_id)
            _merge_page_observation(signature_page, source_page, declared_controls)
            page_refs[ref] = str(signature_page.id or "")
            if raw_step and signature_page.id:
                trace_page_map[raw_step] = str(signature_page.id)
            print(
                "[FFG 更新] 同 activity 控件签名一致，复用已有页面而不新建: "
                f"{item.get('name', ref)} -> {signature_page.id}"
            )
            continue
        page_id, next_page_index = _allocate_page_id(graph, next_page_index)
        page = _clone_page(source_page, page_id)
        page.name = str(item.get("name", "")).strip() or page.name or page_id
        page.description = str(item.get("description", "")).strip() or page.description
        for control in declared_controls:
            page.add_control(control)
        for function_id in function_ids:
            graph.add_page(page, function_id)
        for function_id in entry_function_ids:
            graph.add_entry_page(page, function_id)
        page_refs[ref] = page_id
        if raw_step:
            trace_page_map[raw_step] = page_id
    return next_page_index


def _apply_page_ownership(
    graph: FlowGraph,
    patch_items: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    function_refs: dict[str, str],
    page_refs: dict[str, str],
) -> None:
    """Apply evidence-backed owner additions/removals after entities exist."""
    available_steps = {
        int(step.get("raw_step_index", 0) or 0)
        for step in trace
        if int(step.get("raw_step_index", 0) or 0) > 0
    }
    for item in patch_items:
        source_step = item.get("source_step")
        if isinstance(source_step, str) and source_step.strip().isdigit():
            source_step = int(source_step.strip())
        if source_step not in available_steps:
            print(
                "[FFG 更新] 跳过缺少本轮直接证据的页面归属调整: "
                f"page_ref={item.get('page_ref')}, source_step={source_step}"
            )
            continue
        page_id = _resolve_page_ref(
            _normalize_ref(item.get("page_ref")), page_refs, graph
        )
        page = graph.get_page(page_id)
        if page is None:
            continue

        add_refs = item.get("add_function_refs")
        add_entry_refs = item.get("add_entry_function_refs")
        remove_refs = item.get("remove_function_refs")
        add_refs = add_refs if isinstance(add_refs, list) else []
        add_entry_refs = add_entry_refs if isinstance(add_entry_refs, list) else []
        remove_refs = remove_refs if isinstance(remove_refs, list) else []
        for value in add_refs:
            normalized_ref = _normalize_ref(value)
            function_id = _resolve_function_ref(
                normalized_ref, function_refs, graph
            )
            if function_id:
                if normalized_ref.startswith("new_func_") and not remove_refs:
                    graph.add_entry_page(page, function_id)
                else:
                    graph.add_page(page, function_id)

        for value in add_entry_refs:
            function_id = _resolve_function_ref(
                _normalize_ref(value), function_refs, graph
            )
            if function_id:
                graph.add_entry_page(page, function_id)

        for value in remove_refs:
            function_id = _resolve_function_ref(
                _normalize_ref(value), function_refs, graph
            )
            owners = graph.get_page_owners(page_id)
            if function_id and function_id in owners and len(owners) > 1:
                graph.remove_page_owner(page_id, function_id)


def _separate_new_function_pages_from_planned_target(
    graph: FlowGraph,
    function_items: list[dict[str, Any]],
    page_items: list[dict[str, Any]],
    function_refs: dict[str, str],
    page_refs: dict[str, str],
    current_plan: dict[str, Any],
) -> None:
    """Keep a newly identified independent capability out of its old shell owner."""
    target_function_id = graph.resolve_function_id(
        str(current_plan.get("target_function_id", "") or "").strip()
    )
    if not target_function_id:
        return
    new_function_refs = {
        ref
        for item in function_items
        if (ref := _normalize_ref(item.get("function_ref"))).startswith("new_func_")
    }
    for item in page_items:
        item_function_refs = set(_function_ref_values(item))
        for new_ref in new_function_refs.intersection(item_function_refs):
            function_id = _resolve_function_ref(new_ref, function_refs, graph)
            page_id = _resolve_page_ref(
                _normalize_ref(item.get("page_ref")), page_refs, graph
            )
            if (
                function_id
                and function_id != target_function_id
                and page_id
                and target_function_id in graph.get_page_owners(page_id)
                and function_id in graph.get_page_owners(page_id)
            ):
                graph.remove_page_owner(page_id, target_function_id)


def _scenario_ref_index(value: object) -> int:
    match = re.fullmatch(r"scenario_(\d+)", _normalize_ref(value))
    return int(match.group(1)) if match else 0


def _same_resource(left: object, right: object) -> bool:
    left_key = _function_match_key(left)
    right_key = _function_match_key(right)
    return bool(left_key and right_key and left_key == right_key)


def _dependency_has_replacement(
    graph: FlowGraph,
    dependency: FunctionDependency,
    source_function_id: str,
    replacement_ids: set[str],
) -> bool:
    for candidate in graph.function_dependencies:
        if candidate == dependency:
            continue
        if not _same_resource(candidate.resource_variable, dependency.resource_variable):
            continue
        if dependency.src_function_id == source_function_id:
            if (
                candidate.src_function_id in replacement_ids
                and candidate.dst_function_id == dependency.dst_function_id
            ):
                return True
        elif (
            candidate.dst_function_id in replacement_ids
            and candidate.src_function_id == dependency.src_function_id
        ):
            return True
    return False


def _apply_function_restructures(
    graph: FlowGraph,
    patch_items: list[dict[str, Any]],
    function_refs: dict[str, str],
    page_refs: dict[str, str],
) -> tuple[dict[str, list[str]], list[str]]:
    """Atomically redistribute persisted Pages from coarse Functions.

    The caller runs this on a working graph, so any reported issue rolls the
    entire patch back. A full split may retire the coarse source only after all
    coverage Pages, scenarios, resources and dependencies remain represented.
    """
    replacements_by_source: dict[str, list[str]] = {}
    issues: list[str] = []
    for item_index, item in enumerate(patch_items, start=1):
        source_ref = _normalize_ref(item.get("source_function_ref"))
        source_id = _resolve_function_ref(source_ref, function_refs, graph)
        source = graph.get_node(source_id)
        prefix = f"function_restructures[{item_index}]"
        if source is None:
            issues.append(f"{prefix} references unknown source function {source_ref}")
            continue

        replacements = _restructure_replacements(item)
        if not replacements:
            issues.append(f"{prefix} has no replacements")
            continue

        source_scenarios = list(source.scenarios)
        replacement_ids: list[str] = []
        moved_page_ids: set[str] = set()
        assigned_scenario_indexes: set[int] = set()
        local_valid = True

        for replacement_index, replacement in enumerate(replacements, start=1):
            replacement_prefix = f"{prefix}.replacements[{replacement_index}]"
            function_ref = _normalize_ref(replacement.get("function_ref"))
            function_id = _resolve_function_ref(function_ref, function_refs, graph)
            target = graph.get_node(function_id)
            if target is None or function_id == source_id:
                issues.append(
                    f"{replacement_prefix} references invalid target function {function_ref}"
                )
                local_valid = False
                continue
            if function_id not in replacement_ids:
                replacement_ids.append(function_id)

            raw_page_refs = replacement.get("page_refs")
            raw_page_refs = raw_page_refs if isinstance(raw_page_refs, list) else []
            replacement_page_ids: list[str] = []
            for value in raw_page_refs:
                page_id = _resolve_page_ref(_normalize_ref(value), page_refs, graph)
                if not page_id or page_id not in source.coverage_page_ids:
                    issues.append(
                        f"{replacement_prefix} cannot move non-coverage page {value} "
                        f"from {source_id}"
                    )
                    local_valid = False
                    continue
                if page_id not in replacement_page_ids:
                    replacement_page_ids.append(page_id)
            if not replacement_page_ids:
                issues.append(f"{replacement_prefix} has no valid coverage pages")
                local_valid = False
            for page_id in replacement_page_ids:
                page = graph.get_page(page_id)
                if page is not None:
                    graph.add_page(page, function_id)
                    moved_page_ids.add(page_id)

            raw_entry_refs = replacement.get("entry_page_refs")
            raw_entry_refs = raw_entry_refs if isinstance(raw_entry_refs, list) else []
            for value in raw_entry_refs:
                page_id = _resolve_page_ref(_normalize_ref(value), page_refs, graph)
                page = graph.get_page(page_id)
                if page is None:
                    issues.append(
                        f"{replacement_prefix} references unknown entry page {value}"
                    )
                    local_valid = False
                    continue
                graph.add_entry_page(page, function_id)

            raw_scenario_refs = replacement.get("scenario_refs")
            raw_scenario_refs = (
                raw_scenario_refs if isinstance(raw_scenario_refs, list) else []
            )
            for value in raw_scenario_refs:
                scenario_index = _scenario_ref_index(value)
                if scenario_index < 1 or scenario_index > len(source_scenarios):
                    issues.append(
                        f"{replacement_prefix} references unknown scenario {value}"
                    )
                    local_valid = False
                    continue
                target.add_scenario(source_scenarios[scenario_index - 1])
                assigned_scenario_indexes.add(scenario_index)

            if target.test_status == "not_started" and (
                raw_scenario_refs
                or any(
                    (page := graph.get_page(page_id)) is not None and page.operations
                    for page_id in replacement_page_ids
                )
            ):
                target.test_status = "in_progress"

        if not local_valid:
            continue

        for page_id in moved_page_ids:
            graph.remove_page_owner(page_id, source_id)

        retire_source = bool(item.get("retire_source"))
        if not retire_source:
            replacements_by_source[source_id] = replacement_ids
            continue

        if len(set(replacement_ids)) < 2:
            issues.append(f"{prefix} must have at least two replacements to retire {source_id}")
            continue
        if source.coverage_page_ids:
            issues.append(
                f"{prefix} leaves coverage pages on retired source {source_id}: "
                + ",".join(source.coverage_page_ids)
            )
            continue
        missing_scenarios = [
            f"scenario_{index}"
            for index in range(1, len(source_scenarios) + 1)
            if index not in assigned_scenario_indexes
        ]
        if missing_scenarios:
            issues.append(
                f"{prefix} leaves scenarios on retired source {source_id}: "
                + ",".join(missing_scenarios)
            )
            continue
        replacement_resources = [
            resource
            for function_id in replacement_ids
            if (target := graph.get_node(function_id)) is not None
            for resource in target.resource_variables
        ]
        missing_resources = [
            resource
            for resource in source.resource_variables
            if not any(_same_resource(resource, candidate) for candidate in replacement_resources)
        ]
        if missing_resources:
            issues.append(
                f"{prefix} would drop resources from {source_id}: "
                + ",".join(missing_resources)
            )
            continue
        source_dependencies = [
            dependency
            for dependency in graph.function_dependencies
            if source_id in {
                dependency.src_function_id,
                dependency.dst_function_id,
            }
        ]
        replacement_id_set = set(replacement_ids)
        missing_dependencies = [
            dependency
            for dependency in source_dependencies
            if not _dependency_has_replacement(
                graph,
                dependency,
                source_id,
                replacement_id_set,
            )
        ]
        if missing_dependencies:
            issues.append(
                f"{prefix} would drop function dependencies from {source_id}"
            )
            continue

        # Remaining memberships are navigation-only shell Pages. Replacement
        # entry_page_refs preserve whichever child capabilities need them.
        for page_id in list(source.page_ids):
            graph.remove_page_owner(page_id, source_id)
        graph.function_dependencies = [
            dependency
            for dependency in graph.function_dependencies
            if source_id not in {
                dependency.src_function_id,
                dependency.dst_function_id,
            }
        ]
        graph._function_dependency_set = set(graph.function_dependencies)
        graph.functions.pop(source_id, None)
        replacements_by_source[source_id] = replacement_ids

    return replacements_by_source, list(dict.fromkeys(issues))


def _apply_function_merges(
    graph: FlowGraph,
    patch_items: list[dict[str, Any]],
    function_refs: dict[str, str],
) -> tuple[dict[str, list[str]], list[str]]:
    """Merge two or more existing Functions into one target atomically."""
    replacements_by_source: dict[str, list[str]] = {}
    issues: list[str] = []
    consumed_source_ids: set[str] = set()
    for item_index, item in enumerate(patch_items, start=1):
        prefix = f"function_merges[{item_index}]"
        source_ids: list[str] = []
        for source_ref in _merge_source_refs(item):
            source_id = _resolve_function_ref(source_ref, function_refs, graph)
            if source_id and source_id not in source_ids:
                source_ids.append(source_id)
        if len(source_ids) < 2:
            issues.append(f"{prefix} requires at least two existing source functions")
            continue
        if any(source_id in consumed_source_ids for source_id in source_ids):
            issues.append(f"{prefix} reuses a source function already merged in this patch")
            continue

        target_ref = _normalize_ref(item.get("target_function_ref"))
        target_id = _resolve_function_ref(target_ref, function_refs, graph)
        target = graph.get_node(target_id)
        if target is None:
            issues.append(f"{prefix} references unknown target function {target_ref}")
            continue

        source_nodes = [graph.get_node(source_id) for source_id in source_ids]
        if any(node is None for node in source_nodes):
            issues.append(f"{prefix} references an unavailable source function")
            continue

        # A Page remains coverage-bearing if any merged Function actually
        # implements behavior there; it is entry-only only when every source
        # treats it as navigation context.
        page_roles: dict[str, bool] = {}
        for node in [target, *source_nodes]:
            if node is None:
                continue
            entry_ids = set(node.entry_page_ids)
            for page_id in node.page_ids:
                page_roles[page_id] = page_roles.get(page_id, False) or page_id not in entry_ids
        for page_id, is_coverage_page in page_roles.items():
            page = graph.get_page(page_id)
            if page is None:
                issues.append(f"{prefix} references inactive page {page_id}")
                continue
            if is_coverage_page:
                graph.add_page(page, target_id)
            else:
                graph.add_entry_page(page, target_id)

        for node in source_nodes:
            if node is None or node.id == target_id:
                continue
            for scenario in node.scenarios:
                target.add_scenario(scenario)
            for resource in node.resource_variables:
                target.add_resource_variable(resource)

        old_dependencies = list(graph.function_dependencies)
        graph.function_dependencies = []
        graph._function_dependency_set = set()
        merged_source_id_set = set(source_ids)
        for dependency in old_dependencies:
            src_id = (
                target_id
                if dependency.src_function_id in merged_source_id_set
                else dependency.src_function_id
            )
            dst_id = (
                target_id
                if dependency.dst_function_id in merged_source_id_set
                else dependency.dst_function_id
            )
            if src_id == dst_id:
                target.add_resource_variable(dependency.resource_variable)
                continue
            graph.add_function_dependency(FunctionDependency(
                src_function_id=src_id,
                dst_function_id=dst_id,
                resource_variable=dependency.resource_variable,
            ))

        evidence_exists = bool(target.scenarios) or any(
            (page := graph.get_page(page_id)) is not None and page.operations
            for page_id in target.coverage_page_ids
        )
        if evidence_exists:
            # A changed boundary needs a fresh completion decision even when all
            # source Functions happened to be terminal independently.
            target.test_status = "in_progress"
        else:
            target.test_status = "not_started"

        for source_id in source_ids:
            if source_id == target_id:
                continue
            source = graph.get_node(source_id)
            if source is not None:
                for page_id in list(source.page_ids):
                    graph.remove_page_owner(page_id, source_id)
            graph.functions.pop(source_id, None)
            replacements_by_source[source_id] = [target_id]
        consumed_source_ids.update(source_ids)

    return replacements_by_source, list(dict.fromkeys(issues))


def _apply_observation_alignment(
    graph: FlowGraph,
    alignment_items: list[dict[str, Any]],
    page_refs: dict[str, str],
    trace_page_map: dict[int, str],
) -> None:
    for item in alignment_items:
        if str(item.get("page_decision", "") or "").strip().lower() == "ignore":
            continue
        source_step = item.get("source_step")
        if not str(source_step).strip().isdigit():
            continue
        page_id = _resolve_page_ref(
            _normalize_ref(item.get("page_ref")),
            page_refs,
            graph,
        )
        if page_id:
            trace_page_map[int(str(source_step).strip())] = page_id


def _persist_aligned_trace_screenshots(
    graph: FlowGraph,
    trace: list[dict[str, Any]],
    trace_page_map: dict[int, str],
) -> int:
    """Persist every mapped trace image as evidence of its logical Page.

    Page reuse is primarily expressed through ``observation_alignment``.  The LLM
    is not required to repeat an unchanged page in ``pages``; consequently image
    persistence cannot depend on a pages patch being present.  Keep the original
    representative image stable and append later visible states as evidence.
    """
    added = 0
    for step in trace:
        raw_step = int(step.get("raw_step_index", 0) or 0)
        page_id = str(trace_page_map.get(raw_step, "") or "").strip()
        page = graph.get_page(page_id) if page_id else None
        source_page = _extract_page(step)
        screenshot = _page_screenshot(source_page)
        if page is None or not screenshot:
            continue
        before = (
            page.representative_screenshot,
            tuple(page.evidence_screenshots),
        )
        if page.representative_screenshot:
            page.add_evidence_screenshot(screenshot)
        else:
            page.update_representative_screenshot(screenshot)
        after = (
            page.representative_screenshot,
            tuple(page.evidence_screenshots),
        )
        if after != before:
            added += 1
    return added


def validate_ffg_invariants(graph: FlowGraph) -> list[str]:
    """Return structural consistency violations for an active FFG."""
    issues: list[str] = []
    active_ids = {node.id for node in graph.active_nodes()}
    active_page_ids = {str(page.id) for page in graph.active_pages() if page.id}
    for node in graph.active_nodes():
        for page_id in node.page_ids:
            if page_id not in active_page_ids:
                issues.append(f"function {node.id} references inactive page {page_id}")
            elif node.id not in graph.get_page_owners(page_id):
                issues.append(f"function {node.id} references unowned page {page_id}")
        for page_id in node.entry_page_ids:
            if page_id not in node.page_ids:
                issues.append(f"function {node.id} has detached entry page {page_id}")
    for transition in graph.page_transitions:
        if transition.src_page_id not in active_page_ids or transition.dst_page_id not in active_page_ids:
            issues.append("graph has dangling page transition")
    for page_id in active_page_ids:
        owners = [owner for owner in graph.get_page_owners(page_id) if owner in active_ids]
        if not owners:
            issues.append(f"page {page_id} has no active owner")
        for owner_id in owners:
            owner = graph.get_node(owner_id)
            if owner is not None and page_id not in owner.page_ids:
                issues.append(f"page {page_id} missing from owner {owner_id}")
    for edge in graph.function_dependencies:
        if edge.src_function_id not in active_ids or edge.dst_function_id not in active_ids:
            issues.append(f"function dependency references inactive function {edge.src_function_id}->{edge.dst_function_id}")
    return list(dict.fromkeys(issues))


def _apply_function_dependencies(
    graph: FlowGraph,
    patch_items: list[dict[str, Any]],
    function_refs: dict[str, str],
):
    for item in patch_items:
        src_function_id = _resolve_function_ref(_normalize_ref(item.get("src_function_ref")), function_refs, graph)
        dst_function_id = _resolve_function_ref(_normalize_ref(item.get("dst_function_ref")), function_refs, graph)
        resource_variable = str(item.get("resource_variable", "")).strip()
        if not src_function_id or not dst_function_id or not resource_variable:
            continue
        graph.add_function_dependency(FunctionDependency(
            src_function_id=src_function_id,
            dst_function_id=dst_function_id,
            resource_variable=resource_variable,
        ))


def _append_unique(values: list[str], value: str):
    if value and value not in values:
        values.append(value)


def _resolve_plan_target_function_id(
    graph: FlowGraph,
    current_plan: dict[str, Any],
) -> str:
    target_function_id = str(current_plan.get("target_function_id", "")).strip()
    if graph.get_node(target_function_id) is not None:
        return graph.resolve_function_id(target_function_id)
    return ""


def _record_completed_intra_scenario(
    graph: FlowGraph,
    current_plan: dict[str, Any],
    test_scenario: str,
    endpoint_function_id: str,
) -> str:
    """Attach a completed scenario to the function it was planned for.

    场景学分归属规划时的目标功能（target_function_id）：执行终点落在
    共享页面（如聊天详情页）时，终点功能往往是共享该页面的旧功能，
    按“实际到达功能”归属会系统性偷走目标功能的自身场景计数，导致
    覆盖率已达标的功能因 自身场景=0 永远无法收尾。目标功能无法解析
    时才退回终点功能兜底；下一轮“顺着实际所在功能继续”由
    endpoint_function 的独立逻辑负责，与本归属无关。
    """
    target_function_id = _resolve_plan_target_function_id(graph, current_plan)
    owner_id = target_function_id or graph.resolve_function_id(endpoint_function_id)
    if owner_id and graph.get_node(owner_id) is not None and test_scenario:
        graph.add_scenario(owner_id, test_scenario)
        if graph.get_node_test_status(owner_id) == "not_started":
            graph.set_node_test_status(owner_id, "in_progress")
    return owner_id


def _execution_failure_reason(execution_result: dict[str, Any]) -> str:
    navigation_status = str(execution_result.get("navigation_status", "")).strip()
    navigation_error = str(execution_result.get("navigation_error", "")).strip()
    scenario_status = str(execution_result.get("scenario_status", "")).strip()
    scenario_error = str(execution_result.get("scenario_error", "")).strip()

    if scenario_status in EXECUTION_FAILURE_STATUSES:
        return scenario_error or navigation_error or scenario_status
    if navigation_status and navigation_status not in {"completed", "skipped"}:
        return navigation_error or f"导航状态异常: {navigation_status}"
    return ""


def _is_screenshot_unavailable_execution(execution_result: dict[str, Any]) -> bool:
    navigation_status = str(execution_result.get("navigation_status", "")).strip()
    scenario_status = str(execution_result.get("scenario_status", "")).strip()
    return navigation_status == "screenshot_unavailable" or scenario_status in {
        "screenshot_unavailable",
        "skipped_due_screenshot_unavailable",
    }


def _is_target_app_restarted_execution(execution_result: dict[str, Any]) -> bool:
    navigation_status = str(execution_result.get("navigation_status", "")).strip()
    navigation_error = str(execution_result.get("navigation_error", "")).strip()
    scenario_status = str(execution_result.get("scenario_status", "")).strip()
    scenario_error = str(execution_result.get("scenario_error", "")).strip()
    combined_text = " ".join((
        navigation_status,
        navigation_error,
        scenario_status,
        scenario_error,
    ))
    return (
        bool(execution_result.get("crash_detected"))
        or "target_app_crashed" in combined_text
        or "skipped_due_target_app_crashed" in combined_text
        or "target_app_restarted" in combined_text
        or "skipped_due_target_app_restarted" in combined_text
        or "目标应用离开前台且无法通过返回恢复" in combined_text
        or "检测到目标应用崩溃" in combined_text
    )


def _execution_result_for_llm(execution_result: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(execution_result or {})
    sanitized.pop("crash_logcat", None)
    return sanitized


def _cross_failure_key(src_function_id: str, dst_function_id: str) -> str:
    return f"{str(src_function_id or '').strip()}->{str(dst_function_id or '').strip()}"


def _update_cross_test_execution_progress(
    graph: FlowGraph,
    state: AgentState,
    execution_result: dict[str, Any],
) -> dict[str, int]:
    failure_counts = dict(state.get("cross_execution_failure_counts", {}) or {})
    current_plan = state.get("current_plan", {}) or {}
    if str(current_plan.get("scenario_type", "")).strip() != "cross_test":
        return failure_counts

    src_function_id = str(current_plan.get("src_function_id", "")).strip()
    dst_function_id = str(current_plan.get("dst_function_id", "")).strip()
    if not src_function_id or not dst_function_id:
        return failure_counts

    key = _cross_failure_key(src_function_id, dst_function_id)
    if execution_result.get("scenario_status") == "completed" or _is_target_app_restarted_execution(execution_result):
        failure_counts.pop(key, None)
        return failure_counts

    failure_reason = _execution_failure_reason(execution_result)
    if not failure_reason:
        return failure_counts

    failure_count = int(failure_counts.get(key, 0) or 0) + 1
    if failure_count >= CROSS_FAILURE_BLOCK_THRESHOLD:
        test_scenario = (
            str(state.get("current_test_scenario", "")).strip()
            or str(execution_result.get("current_scenario", "")).strip()
            or "跨功能场景执行失败"
        )
        graph.add_cross_scenario(
            src_function_id,
            dst_function_id,
            f"自动化无法完成，已停止重复尝试: {test_scenario} | 失败原因: {failure_reason}",
        )
        failure_counts.pop(key, None)
        print(
            "[FFG 更新] 跨功能组合执行失败，标记为已尝试以避免重复规划: "
            f"{src_function_id} -> {dst_function_id} "
            f"({failure_count}/{CROSS_FAILURE_BLOCK_THRESHOLD})"
        )
    else:
        failure_counts[key] = failure_count
        print(
            "[FFG 更新] 记录跨功能组合执行失败: "
            f"{src_function_id} -> {dst_function_id} "
            f"({failure_count}/{CROSS_FAILURE_BLOCK_THRESHOLD})"
        )
    return failure_counts


def _close_crashed_cross_test_scope(
    graph: FlowGraph,
    state: AgentState,
    execution_result: dict[str, Any],
    failure_counts: dict[str, int],
    current_testing_function_id: str,
    current_testing_function_status: str,
) -> tuple[dict[str, int], str, str]:
    if not _is_target_app_restarted_execution(execution_result):
        return failure_counts, current_testing_function_id, current_testing_function_status

    current_plan = state.get("current_plan", {}) or {}
    if str(current_plan.get("scenario_type", "")).strip() != "cross_test":
        return failure_counts, current_testing_function_id, current_testing_function_status

    src_function_id = str(current_plan.get("src_function_id", "")).strip()
    dst_function_id = str(current_plan.get("dst_function_id", "")).strip()
    test_scenario = (
        str(state.get("current_test_scenario", "")).strip()
        or str(execution_result.get("current_scenario", "")).strip()
        or "跨功能场景因目标应用崩溃结束"
    )
    if src_function_id and dst_function_id:
        graph.add_cross_scenario(src_function_id, dst_function_id, test_scenario)
        print(
            "[FFG 更新] 崩溃后结束当前跨功能组合，后续切换其它功能: "
            f"{src_function_id} -> {dst_function_id}"
        )

    if src_function_id and graph.get_node(src_function_id) is not None:
        graph.set_node_test_status(src_function_id, "in_progress")
        failure_counts[src_function_id] = max(
            1,
            int(failure_counts.get(src_function_id, 0) or 0),
        )
        if current_testing_function_id == src_function_id:
            current_testing_function_id = ""
            current_testing_function_status = "not_started"

    if dst_function_id:
        failure_counts.pop(dst_function_id, None)
    return failure_counts, current_testing_function_id, current_testing_function_status


def _update_function_execution_progress(
    graph: FlowGraph,
    state: AgentState,
    execution_result: dict[str, Any],
    current_testing_function_id: str,
    current_testing_function_status: str,
    reset_function_ids: list[str] | None = None,
) -> tuple[dict[str, int], str, str]:
    current_plan = state.get("current_plan", {}) or {}
    scenario_type = str(current_plan.get("scenario_type", "")).strip()
    failure_counts = dict(state.get("function_execution_failure_counts", {}) or {})
    if scenario_type != "intra_test":
        return failure_counts, current_testing_function_id, current_testing_function_status

    target_function_id = _resolve_plan_target_function_id(graph, current_plan)
    failure_reason = _execution_failure_reason(execution_result)
    navigation_status = str(
        execution_result.get("navigation_status", "") or ""
    ).strip()
    scenario_status = str(execution_result.get("scenario_status", "") or "").strip()
    confirmed_stall = (
        (scenario_status == "stalled" or navigation_status == "stalled")
        and bool(CONFIRMED_STALL_BLOCK_PATTERN.search(failure_reason))
    )
    verification_blocked = (
        scenario_status == "blocked_by_verification"
        or navigation_status == "blocked_by_verification"
    )
    if verification_blocked or confirmed_stall:
        if target_function_id and graph.get_node(target_function_id) is not None:
            current_status = graph.get_node_test_status(target_function_id)
            if current_status in TERMINAL_FUNCTION_STATUSES:
                # 覆盖审计轮中已完成的 Function 不因执行失败翻转终端状态：
                # 状态翻转会改变审计签名，导致同一功能被反复全量重审并烧掉轮次。
                print(
                    "[FFG 更新] 功能已处于终端状态，保留原状态并让出调度: "
                    f"{target_function_id}({current_status}) | "
                    f"{failure_reason or scenario_status}"
                )
            else:
                graph.set_node_test_status(target_function_id, "unavailable")
                print(
                    "[FFG 更新] 当前功能遇到无法继续的验证或确认页面持续无变化，"
                    "标记为 unavailable 并切换其它功能: "
                    f"{target_function_id} | {failure_reason or scenario_status}"
                )
            failure_counts.pop(target_function_id, None)
        return failure_counts, "", "not_started"
    if _is_target_app_restarted_execution(execution_result):
        if target_function_id and graph.get_node(target_function_id) is not None:
            current_status = graph.get_node_test_status(target_function_id)
            if current_status in TERMINAL_FUNCTION_STATUSES:
                print(
                    "[FFG 更新] 功能已处于终端状态，崩溃恢复不改变其持久化状态: "
                    f"{target_function_id}({current_status})"
                )
            else:
                graph.set_node_test_status(target_function_id, "in_progress")
                failure_counts[target_function_id] = max(
                    1,
                    int(failure_counts.get(target_function_id, 0) or 0),
                )
                print(
                    "[FFG 更新] 崩溃不直接关闭整个功能，保留为 in_progress 并切换其它功能: "
                    f"{target_function_id}"
                )
        return failure_counts, "", "not_started"

    if _is_screenshot_unavailable_execution(execution_result):
        if target_function_id and graph.get_node(target_function_id) is not None:
            current_status = graph.get_node_test_status(target_function_id)
            if current_status not in TERMINAL_FUNCTION_STATUSES:
                graph.set_node_test_status(target_function_id, "in_progress")
                current_testing_function_id = target_function_id
                current_testing_function_status = "in_progress"
                print(
                    "[FFG 更新] 当前功能因截图不可用/黑屏保留为 in_progress: "
                    f"{target_function_id}"
                )
            failure_counts.pop(target_function_id, None)
        return failure_counts, current_testing_function_id, current_testing_function_status

    if failure_reason and (scenario_status == "stalled" or navigation_status == "stalled"):
        if target_function_id and graph.get_node(target_function_id) is not None:
            current_status = graph.get_node_test_status(target_function_id)
            if current_status in TERMINAL_FUNCTION_STATUSES:
                print(
                    "[FFG 更新] 功能已处于终端状态，执行停滞不改变其持久化状态: "
                    f"{target_function_id}({current_status}) | {failure_reason}"
                )
            else:
                graph.set_node_test_status(target_function_id, "in_progress")
                failure_counts[target_function_id] = min(
                    int(failure_counts.get(target_function_id, 0) or 0) + 1,
                    FUNCTION_FAILURE_BLOCK_THRESHOLD,
                )
                print(
                    "[FFG 更新] 页面循环或执行停滞未达到 unavailable 条件，"
                    "保留为 in_progress 并让出当前调度: "
                    f"{target_function_id} | {failure_reason}"
                )
        return failure_counts, "", "not_started"

    if failure_reason:
        if target_function_id:
            current_status = graph.get_node_test_status(target_function_id)
            if current_status in TERMINAL_FUNCTION_STATUSES:
                return failure_counts, current_testing_function_id, current_testing_function_status
            failure_count = int(failure_counts.get(target_function_id, 0) or 0) + 1
            failure_counts[target_function_id] = min(
                failure_count,
                FUNCTION_FAILURE_BLOCK_THRESHOLD,
            )
            if failure_count >= FUNCTION_FAILURE_BLOCK_THRESHOLD:
                graph.set_node_test_status(target_function_id, "in_progress")
                current_testing_function_id = ""
                current_testing_function_status = "not_started"
                print(
                    "[FFG 更新] 功能连续执行失败但未确认页面卡死，"
                    "保持 in_progress 并让出当前调度: "
                    f"{target_function_id} ({failure_count}/{FUNCTION_FAILURE_BLOCK_THRESHOLD})"
                )
            else:
                graph.set_node_test_status(target_function_id, "in_progress")
                current_testing_function_id = target_function_id
                current_testing_function_status = "in_progress"
                print(
                    "[FFG 更新] 记录功能执行失败，保留为 in_progress: "
                    f"{target_function_id} ({failure_count}/{FUNCTION_FAILURE_BLOCK_THRESHOLD})"
                )
        return failure_counts, current_testing_function_id, current_testing_function_status

    if execution_result.get("scenario_status") == "completed":
        # 只清掉“本轮确有结构产出”的功能。target_function_id 的清零交给
        # _record_zero_structural_yield：它在本函数之后执行，若在这里无条件
        # pop，零产出计数会被每一轮成功执行清零，永远累积不到释放阈值。
        reset_candidates = [current_testing_function_id, *(reset_function_ids or [])]
        for function_id in reset_candidates:
            if function_id and function_id != target_function_id:
                failure_counts.pop(function_id, None)

    return failure_counts, current_testing_function_id, current_testing_function_status


def _record_zero_structural_yield(
    graph: FlowGraph,
    state: AgentState,
    before: dict[str, int],
    after: dict[str, int],
    failure_counts: dict[str, int],
    current_testing_function_id: str,
    current_testing_function_status: str,
) -> tuple[dict[str, int], str, str]:
    """把“执行成功但没有任何结构产出”计入无进展次数。

    `_update_function_execution_progress` 只统计执行失败，成功一轮就清零计数，
    因此“每轮都成功、但既不产生新页面也不产生新功能”的无底洞页面（典型是
    WebView 长文）永远不会被降优先级：它的未覆盖控件近乎无穷，规划器一直能
    产出新场景，可以连续十几轮把时间耗在同一 Activity 上。

    只看功能数与页面数增量。操作数和页面跳转几乎每轮都会增长，纳入判据会让
    检测永不触发。
    """
    if not ZERO_YIELD_COUNTS_AS_NO_PROGRESS:
        return failure_counts, current_testing_function_id, current_testing_function_status

    current_plan = state.get("current_plan", {}) or {}
    if str(current_plan.get("scenario_type", "")).strip() != "intra_test":
        return failure_counts, current_testing_function_id, current_testing_function_status

    execution_result = state.get("execution_result", {}) or {}
    # 执行失败已由 _update_function_execution_progress 计数，避免同一轮双重累加。
    if _execution_failure_reason(execution_result):
        return failure_counts, current_testing_function_id, current_testing_function_status

    if after["functions"] > before["functions"] or after["pages"] > before["pages"]:
        # 本轮确有结构产出：清零该功能的连续无进展计数，只有“连续”零产出才释放。
        target_function_id = _resolve_plan_target_function_id(graph, current_plan)
        if target_function_id:
            failure_counts.pop(target_function_id, None)
        return failure_counts, current_testing_function_id, current_testing_function_status

    target_function_id = _resolve_plan_target_function_id(graph, current_plan)
    node = graph.get_node(target_function_id) if target_function_id else None
    if node is None or (node.test_status or "not_started") in TERMINAL_FUNCTION_STATUSES:
        return failure_counts, current_testing_function_id, current_testing_function_status

    # 刚开始探索的功能本就需要几轮铺垫才会产出新页面，不能一上来就降优先级。
    coverage = int(graph.function_coverage_details(node.id)["coverage_percent"])
    if coverage < ZERO_YIELD_MIN_COVERAGE:
        return failure_counts, current_testing_function_id, current_testing_function_status

    no_progress_count = int(failure_counts.get(node.id, 0) or 0) + 1
    failure_counts[node.id] = min(no_progress_count, FUNCTION_FAILURE_BLOCK_THRESHOLD)
    if no_progress_count >= FUNCTION_FAILURE_BLOCK_THRESHOLD:
        # 让出当前调度目标。_select_intra_target 会硬优先返回 preferred_id，
        # 只加计数而不清空的话，同一功能仍会被连续选中。
        print(
            "[FFG 更新] 功能连续多轮无新增页面或功能，让出当前调度: "
            f"{node.id} | coverage={coverage}% "
            f"({no_progress_count}/{FUNCTION_FAILURE_BLOCK_THRESHOLD})"
        )
        return failure_counts, "", "not_started"

    print(
        "[FFG 更新] 记录功能本轮无结构产出: "
        f"{node.id} | coverage={coverage}% "
        f"({no_progress_count}/{FUNCTION_FAILURE_BLOCK_THRESHOLD})"
    )
    return failure_counts, current_testing_function_id, current_testing_function_status


def _apply_non_recordable_execution_outcome(
    graph: FlowGraph,
    state: AgentState,
    execution_result: dict[str, Any],
    *,
    reason: str = "no recordable trace",
) -> dict[str, Any]:
    current_testing_function_id = str(state.get("current_testing_function_id", "")).strip()
    current_testing_function_status = str(state.get("current_testing_function_status", "not_started")).strip() or "not_started"
    current_plan = state.get("current_plan", {}) or {}
    test_scenario = str(state.get("current_test_scenario", "") or "").strip()
    scenario_status = str(execution_result.get("scenario_status", "") or "").strip()
    _record_screenshot_unavailable_page(graph, state, execution_result)
    current_page = _map_current_page_after_update(graph, state)
    if (
        test_scenario
        and str(current_plan.get("scenario_type", "")).strip() == "intra_test"
        and scenario_status in EXECUTION_FAILURE_STATUSES
        and not scenario_status.startswith("skipped_due_navigation")
    ):
        target_function_id = _resolve_plan_target_function_id(graph, current_plan)
        if target_function_id and graph.get_node(target_function_id) is not None:
            graph.add_scenario(target_function_id, test_scenario)
            print(
                "[FFG 更新] 已记忆无有效 trace 的失败场景，后续不再重复规划: "
                f"{target_function_id} | {scenario_status}"
            )
    failure_counts, current_testing_function_id, current_testing_function_status = _update_function_execution_progress(
        graph,
        state,
        execution_result,
        current_testing_function_id,
        current_testing_function_status,
    )
    cross_failure_counts = _update_cross_test_execution_progress(
        graph,
        state,
        execution_result,
    )
    failure_counts, current_testing_function_id, current_testing_function_status = _close_crashed_cross_test_scope(
        graph,
        state,
        execution_result,
        failure_counts,
        current_testing_function_id,
        current_testing_function_status,
    )
    endpoint_function_id, endpoint_function_status = _endpoint_function_state(
        graph,
        current_page,
        current_plan,
        current_testing_function_id,
    )
    if endpoint_function_id:
        current_testing_function_id = endpoint_function_id
        current_testing_function_status = endpoint_function_status

    return {
        "ffg": graph,
        "current_page": current_page,
        "function_execution_failure_counts": failure_counts,
        "cross_execution_failure_counts": cross_failure_counts,
        "current_testing_function_id": current_testing_function_id,
        "current_testing_function_status": current_testing_function_status,
        "ffg_update_decision": "finish",
        "ffg_update_reason": reason,
        "ffg_update_validation_issues": [],
    }


def _collect_touched_function_ids(
    graph: FlowGraph,
    current_plan: dict[str, Any],
    function_refs: dict[str, str],
    pages_patch: list[dict[str, Any]],
    page_ownership_patch: list[dict[str, Any]],
    function_restructures_patch: list[dict[str, Any]] | None = None,
    function_merges_patch: list[dict[str, Any]] | None = None,
    function_replacements: dict[str, list[str]] | None = None,
) -> list[str]:
    touched: list[str] = []

    planned_target = str(current_plan.get("target_function_id", "")).strip()
    if graph.get_node(planned_target) is not None:
        _append_unique(touched, planned_target)

    for item in pages_patch:
        for function_ref in _all_function_ref_values(item):
            function_id = _resolve_function_ref(function_ref, function_refs, graph)
            if graph.get_node(function_id) is not None:
                _append_unique(touched, function_id)

    for item in page_ownership_patch:
        for key in (
            "add_function_refs",
            "add_entry_function_refs",
            "remove_function_refs",
        ):
            values = item.get(key)
            if not isinstance(values, list):
                continue
            for value in values:
                function_id = _resolve_function_ref(
                    _normalize_ref(value), function_refs, graph
                )
                if graph.get_node(function_id) is not None:
                    _append_unique(touched, function_id)

    for item in function_restructures_patch or []:
        for replacement in _restructure_replacements(item):
            function_id = _resolve_function_ref(
                _normalize_ref(replacement.get("function_ref")),
                function_refs,
                graph,
            )
            if graph.get_node(function_id) is not None:
                _append_unique(touched, function_id)
    for item in function_merges_patch or []:
        function_id = _resolve_function_ref(
            _normalize_ref(item.get("target_function_ref")),
            function_refs,
            graph,
        )
        if graph.get_node(function_id) is not None:
            _append_unique(touched, function_id)
    for replacement_ids in (function_replacements or {}).values():
        for function_id in replacement_ids:
            if graph.get_node(function_id) is not None:
                _append_unique(touched, function_id)

    return touched


def apply_ffg_patch(
    graph: FlowGraph,
    patch: dict[str, Any],
    trace: list[dict[str, Any]],
    current_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """在工作副本上原子应用 patch；失败时保留输入图。"""
    working_graph = FlowGraph.from_dict(graph.to_dict())
    functions_patch = _list_patch(patch.get("functions"))
    pages_patch = _list_patch(patch.get("pages"))
    dependencies_patch = _list_patch(patch.get("function_dependencies"))
    observation_alignment = _list_patch(patch.get("observation_alignment"))
    page_ownership_patch = _list_patch(patch.get("page_ownership"))
    function_restructures_patch = _list_patch(patch.get("function_restructures"))
    function_merges_patch = _list_patch(patch.get("function_merges"))

    function_refs: dict[str, str] = {}
    page_refs: dict[str, str] = {}
    trace_page_map: dict[int, str] = {}
    next_function_index = _next_numeric_id(list(working_graph.functions), "func_")
    next_page_index = _next_numeric_id(list(working_graph.pages), "page_")
    persisted_evidence_refs = _persisted_restructure_function_refs(
        working_graph,
        function_restructures_patch,
    )
    persisted_evidence_refs.update(_persisted_merge_function_refs(
        working_graph,
        function_merges_patch,
    ))

    next_function_index = _apply_functions(
        working_graph,
        functions_patch,
        trace,
        function_refs,
        page_refs,
        next_function_index,
        persisted_evidence_refs=persisted_evidence_refs,
    )
    next_page_index = _apply_pages(
        working_graph,
        pages_patch,
        trace,
        function_refs,
        page_refs,
        trace_page_map,
        next_page_index,
    )
    _apply_observation_alignment(
        working_graph,
        observation_alignment,
        page_refs,
        trace_page_map,
    )
    persisted_trace_images = _persist_aligned_trace_screenshots(
        working_graph,
        trace,
        trace_page_map,
    )
    _apply_page_ownership(
        working_graph,
        page_ownership_patch,
        trace,
        function_refs,
        page_refs,
    )
    _separate_new_function_pages_from_planned_target(
        working_graph,
        functions_patch,
        pages_patch,
        function_refs,
        page_refs,
        current_plan or {},
    )
    _apply_function_dependencies(working_graph, dependencies_patch, function_refs)
    merge_replacements, merge_issues = _apply_function_merges(
        working_graph,
        function_merges_patch,
        function_refs,
    )
    restructure_replacements, restructure_issues = _apply_function_restructures(
        working_graph,
        function_restructures_patch,
        function_refs,
        page_refs,
    )
    function_replacements = dict(merge_replacements)
    for source_id, replacement_ids in restructure_replacements.items():
        function_replacements[source_id] = replacement_ids
    working_graph.reconcile_page_membership()

    validation_issues = [
        *merge_issues,
        *restructure_issues,
        *validate_ffg_invariants(working_graph),
    ]
    validation_issues = list(dict.fromkeys(validation_issues))
    applied_graph = graph
    if validation_issues:
        print("[FFG 更新] patch 校验失败，已回滚工作副本: " + "; ".join(validation_issues[:8]))
        persisted_trace_images = 0
        trace_page_map = {
            raw_step: page_id
            for raw_step, page_id in trace_page_map.items()
            if graph.get_page(page_id) is not None
        }
    else:
        graph.functions = working_graph.functions
        graph.pages = working_graph.pages
        graph.page_transitions = working_graph.page_transitions
        graph._page_transition_set = working_graph._page_transition_set
        graph.function_dependencies = working_graph.function_dependencies
        graph._function_dependency_set = working_graph._function_dependency_set

    touched_function_ids = _collect_touched_function_ids(
        applied_graph,
        current_plan or {},
        function_refs,
        pages_patch,
        page_ownership_patch,
        function_restructures_patch,
        function_merges_patch,
        function_replacements,
    )

    return {
        "graph": applied_graph,
        "validation_issues": validation_issues,
        "functions_patch": functions_patch,
        "pages_patch": pages_patch,
        "dependencies_patch": dependencies_patch,
        "observation_alignment": observation_alignment,
        "page_ownership_patch": page_ownership_patch,
        "function_restructures_patch": function_restructures_patch,
        "function_merges_patch": function_merges_patch,
        "function_replacements": function_replacements,
        "function_refs": function_refs,
        "page_refs": page_refs,
        "trace_page_map": trace_page_map,
        "persisted_trace_images": persisted_trace_images,
        "touched_function_ids": touched_function_ids,
    }


def create_ffg_update_node(llm, device=None):
    """Create the lightweight FFG update node."""
    thumbnail_cache = _ScreenshotThumbnailCache()
    compact_cache: dict[str, str] = {}

    def ffg_update_node(state: AgentState) -> dict:
        refinement_step = max(1, int(state.get("ffg_update_step", 1) or 1))
        is_refinement = refinement_step > 1
        raw_trace = _combined_execution_trace(state)
        trace = _filter_trace(raw_trace)
        test_trace = [
            step
            for step in list(state.get("scenario_execution_trace", []) or [])
            if isinstance(step, dict)
        ]
        test_has_device_action = _trace_has_device_action(test_trace)
        graph = state.get("ffg")
        if not isinstance(graph, FlowGraph):
            graph = FlowGraph()
        if not is_refinement:
            thumbnail_cache.clear()
            compact_cache.clear()

        # 页面控件后台审计：先合并已完成的结果（在 before 快照前完成，
        # 使本轮 stats 只反映 LLM patch），再投喂新一批待审页面。
        audit_base_graph: FlowGraph | None = None
        if not is_refinement:
            audit_changes = _merge_page_audit_results(graph)
            if audit_changes > 0:
                # Page audit is an independent evidence pass.  Preserve its
                # accepted changes even if the later trace patch is rolled back
                # for an unrelated alignment error.
                audit_base_graph = FlowGraph.from_dict(graph.to_dict())
            _submit_page_audit_batch(graph, llm)

        before = _snapshot_graph(graph)
        execution_result = state.get("execution_result", {}) or {}
        current_plan = state.get("current_plan", {}) or {}
        test_scenario = str(state.get("current_test_scenario", "")).strip()
        if (
            not is_refinement
            and test_scenario
            and test_trace
            and test_has_device_action
            and str(current_plan.get("scenario_type", "")).strip() == "intra_test"
        ):
            target_function_id = _resolve_plan_target_function_id(graph, current_plan)
            if target_function_id:
                # Attempt history is execution evidence, even when every step
                # is later filtered from graph construction as failed/external.
                graph.add_scenario(target_function_id, test_scenario)
        scenario_completed = execution_result.get("scenario_status") == "completed"
        review_attempts = dict(state.get("granularity_review_attempts", {}) or {})
        granularity_review_function_id = str(
            state.get("ffg_granularity_review_function_id", "") or ""
        ).strip() if is_refinement else ""
        granularity_review_metrics: dict[str, Any] = dict(
            state.get("ffg_granularity_review_metrics", {}) or {}
        ) if is_refinement else {}
        if not is_refinement:
            (
                granularity_review_function_id,
                granularity_review_metrics,
            ) = _select_granularity_review_function(
                graph,
                current_plan,
                review_attempts,
            )
            if granularity_review_function_id:
                review_attempts[granularity_review_function_id] = int(
                    granularity_review_metrics.get("review_attempt", 1) or 1
                )
                print(
                    "[FFG 更新] 触发 Function 粒度复核: "
                    f"{granularity_review_function_id} | "
                    f"pages={granularity_review_metrics.get('coverage_page_count', 0)}, "
                    f"activities={granularity_review_metrics.get('activity_count', 0)}, "
                    f"attempt={granularity_review_metrics.get('review_attempt', 1)}/"
                    f"{granularity_review_metrics.get('max_review_attempts', 1)}"
                )

        print(
            "[FFG 更新] 输入状态: "
            f"scenario_status={execution_result.get('scenario_status', '')}, "
            f"navigation_trace_steps={len(state.get('navigation_execution_trace', []) or [])}, "
            f"scenario_trace_steps={len(state.get('scenario_execution_trace', []) or [])}, "
            f"combined_steps={len(raw_trace)}, processed_steps={len(trace)}"
        )
        if not raw_trace:
            print("[FFG 更新] 场景 trace 为空，改为记录本轮执行结果并更新功能状态。")
            return _apply_non_recordable_execution_outcome(
                graph,
                state,
                execution_result,
            )
        if not trace:
            print("[FFG 更新] 没有可处理的轨迹步骤，改为记录本轮执行结果并更新功能状态。")
            return _apply_non_recordable_execution_outcome(
                graph,
                state,
                execution_result,
            )

        thumbnail_cache.preload_graph(graph)
        candidate_map, step_similarity_cache = _build_observation_candidate_map(
            graph,
            current_plan,
            trace,
            thumbnail_cache=thumbnail_cache,
        )
        fast_path_page_map = _ffg_no_change_fast_path(
            graph,
            state,
            trace,
            current_plan,
            step_similarity_cache,
        )
        if granularity_review_function_id:
            # Known-page replay is exactly when an oversized Function otherwise
            # becomes frozen forever. Force the normal structural review pass.
            fast_path_page_map = None
        ffg_summary = _build_ffg_summary(
            graph,
            current_plan,
            trace,
            candidate_map=candidate_map,
            granularity_review_function_id=granularity_review_function_id,
        )
        abnormal_summary = _format_abnormal_step_summary(raw_trace)
        message_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"【目标应用】name={state.get('app_name', '') or '未知'}，"
                    f"package={state.get('package_name', '') or '未知'}"
                ),
            },
            {"type": "text", "text": ffg_summary},
            {"type": "text", "text": _build_plan_context(graph, current_plan)},
            {
                "type": "text",
                "text": _build_function_boundary_context(
                    graph,
                    current_plan,
                    trace,
                    granularity_review_function_id,
                    granularity_review_metrics,
                ),
            },
            {
                "type": "text",
                "text": (
                    f"执行场景: {test_scenario}\n"
                    f"执行结果: {_execution_result_for_llm(execution_result)}\n"
                    "下面的步骤按实际顺序包含完整导航 trace 和测试 trace；"
                    "动作失败、干扰、外部页面、自动恢复和纯观察步骤都必须逐步检查，"
                    "再依据直接证据决定更新 Page、operation 或 PageTransition。\n"
                    "下方 [STEP] 的 index 是本轮合并轨迹步骤号；"
                    "pages.source_step 必须使用这个步骤号。\n"
                    "每张图片都是紧邻 STEP 的动作执行前状态 BEFORE_STEP_index。"
                    "动作 index=i 的结果只能从后续 STEP 的动作前截图确认；"
                    "新结果页应引用实际显示它的后续 source_step，不能引用触发动作所在的 source_step。"
                ),
            },
        ]
        previous_issues = state.get("ffg_update_validation_issues", []) or []
        if is_refinement:
            message_content.append({
                "type": "text",
                "text": (
                    f"【FFG Agent 第 {refinement_step} 步】上一版已应用到工作副本。"
                    "请检查当前聚焦 FFG 并修正尚未完成的结构或细节；"
                    "如果已经完整，返回 update_control.decision=finish。\n"
                    f"上一轮校验问题: {json.dumps(previous_issues, ensure_ascii=False)}\n"
                    "常见校验错误的修复方法：\n"
                    "- ignore alignment 类错误：ignore 只能对齐执行 trace 中明确的"
                    "外部/系统原因观察（验证阻塞、系统弹窗等）；步骤是正常执行结果时"
                    "改用 move/extract/complete，不要继续用 ignore。\n"
                    "- cannot move non-coverage page：非 coverage 页面不能移动，"
                    "改用新增页面或修正目标页面引用。\n"
                    "- 引用不存在的编号：func_x/page_x 只能引用输入上下文中列出的编号。\n"
                ),
            })
        refinement_focus_steps = (
            _refinement_focus_steps(previous_issues, trace)
            if is_refinement
            else None
        )
        compact_trace_screenshots: dict[int, str] = {}
        matched_step_pages: dict[int, str] = {}
        unmatched_trace_screenshots: dict[int, str] = {}
        for index, step in enumerate(trace, start=1):
            raw_step = int(step.get("raw_step_index", index) or index)
            if not _needs_trace_screenshot(graph, step):
                compact_trace_screenshots[raw_step] = ""
                continue
            screenshot = _page_screenshot(_extract_page(step)) or ""
            if not screenshot:
                compact_trace_screenshots[raw_step] = ""
                continue
            matched_page_id = (
                _find_matched_existing_page(
                    graph, step, raw_step, step_similarity_cache
                )
                if FFG_UPDATE_OMIT_MATCHED_TRACE_IMAGES
                else ""
            )
            if matched_page_id:
                matched_step_pages[raw_step] = matched_page_id
                compact_trace_screenshots[raw_step] = ""
                continue
            unmatched_trace_screenshots[raw_step] = screenshot

        # Every unmatched screenshot is first-hand structural evidence and may be
        # a new logical Page even when its Activity matches another step. Never
        # hide it behind an image budget. Refinement has already seen the complete
        # first pass, so it may narrow to validation-related steps and neighbors.
        selected_trace_steps = set(unmatched_trace_screenshots)
        if is_refinement and refinement_focus_steps is not None:
            selected_trace_steps.intersection_update(refinement_focus_steps)
        for raw_step, screenshot in unmatched_trace_screenshots.items():
            if raw_step not in selected_trace_steps:
                compact_trace_screenshots[raw_step] = ""
                continue
            if screenshot not in compact_cache:
                compact_cache[screenshot] = _compact_screenshot_for_llm(screenshot)
            compact_trace_screenshots[raw_step] = compact_cache[screenshot]
        trace_screenshot_payloads = {
            screenshot for screenshot in compact_trace_screenshots.values() if screenshot
        }
        reference_pages = (
            []
            if fast_path_page_map is not None
            else _reference_pages_for_update(
                graph,
                current_plan,
                trace,
                candidate_map=candidate_map,
                thumbnail_cache=thumbnail_cache,
                step_similarity_cache=step_similarity_cache,
            )
        )
        matched_reference_page_ids = set(matched_step_pages.values())
        reference_candidates = [
            page
            for page in reference_pages
            if str(page.id or "") not in matched_reference_page_ids
        ]
        if is_refinement and refinement_focus_steps is not None:
            focused_candidate_ids = {
                str(candidate.get("page_id", ""))
                for raw_step in refinement_focus_steps
                for candidate in candidate_map.get(raw_step, [])[:2]
            }
            reference_candidates = [
                page
                for page in reference_candidates
                if str(page.id or "") in focused_candidate_ids
            ]
        image_budget = (
            FFG_UPDATE_MAX_REFINEMENT_IMAGES
            if is_refinement
            else FFG_UPDATE_MAX_TOTAL_IMAGES
        )
        remaining_reference_budget = max(
            0,
            image_budget - len(trace_screenshot_payloads),
        )
        image_reference_pages = reference_candidates[:remaining_reference_budget]
        if image_reference_pages:
            message_content.append({
                "type": "text",
                "text": (
                    "【已存 FFG 页面视觉参考】以下图片来自 FFG 中已持久化页面，"
                    "请结合它们与现有页面文字自行判断本轮页面是否应复用已有 page_x。"
                    "它们不是实时轨迹观测，"
                    "不能作为 pages.source_step，也不能证明本轮执行结果。"
                ),
            })
            for page in image_reference_pages:
                message_content.append({
                    "type": "text",
                    "text": (
                        f"[FFG_REFERENCE_IMAGE] page_id={page.id}, "
                        f"owners={graph.get_page_owners(str(page.id or ''))}, "
                        f"name={page.name or page.id}, "
                        f"activity={page.activity or 'unknown'}"
                    ),
                })
                reference_screenshot = page.representative_screenshot or ""
                if reference_screenshot and reference_screenshot not in compact_cache:
                    compact_cache[reference_screenshot] = _compact_screenshot_for_llm(reference_screenshot)
                screenshot = compact_cache.get(reference_screenshot, "")
                if screenshot and screenshot not in trace_screenshot_payloads:
                    message_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{screenshot}",
                        },
                    })
        if abnormal_summary:
            message_content.append({
                "type": "text",
                "text": f"被过滤的异常步骤摘要:\n{abnormal_summary}",
            })
        uploaded_trace_screenshots: dict[str, int] = {}
        for index, step in enumerate(trace, start=1):
            page = _extract_page(step)
            action_name = str(step.get("action_name", "")).strip()
            raw_index = int(step.get("raw_step_index", index) or index)
            message_content.append({
                "type": "text",
                "text": (
                    _format_trace_step(step, raw_index, index)
                    + f"\n[OBSERVATION] BEFORE_STEP_{raw_index}: "
                    + (
                        f"下方截图是 action={action_name or 'unknown'} 尚未执行时的页面。"
                        if _page_screenshot(page)
                        else "本步骤没有可用截图。"
                    )
                ),
            })
            screenshot = compact_trace_screenshots.get(raw_index, "")
            reused_step = uploaded_trace_screenshots.get(screenshot) if screenshot else None
            if screenshot and reused_step is None:
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{screenshot}"},
                })
                uploaded_trace_screenshots[screenshot] = raw_index
            elif reused_step is not None:
                message_content.append({
                    "type": "text",
                    "text": (
                        f"[OBSERVATION_IMAGE_REUSE] BEFORE_STEP_{raw_index} 与 "
                        f"BEFORE_STEP_{reused_step} 使用同一截图，不重复上传图片。"
                    ),
                })
            elif raw_index in matched_step_pages:
                matched_page_id = matched_step_pages[raw_index]
                matched_page = graph.get_page(matched_page_id) if matched_page_id else None
                matched_summary = (
                    f"name={matched_page.name or matched_page_id}, "
                    f"activity={matched_page.activity or 'unknown'}, "
                    f"description={_clip_context_text(matched_page.description, 240) or '无'}, "
                    "controls="
                    f"{[_clip_context_text(control, 80) for control in matched_page.controls[:8]]}"
                    if matched_page is not None
                    else matched_page_id
                )
                message_content.append({
                    "type": "text",
                    "text": (
                        f"[OBSERVATION_IMAGE_MATCHED] BEFORE_STEP_{raw_index} 与已存页面 "
                        f"{matched_page_id}（{matched_summary}）Activity 一致且视觉高度匹配，"
                        "使用其文本摘要替代截图，不重复上传图片。"
                    ),
                })
            elif _page_screenshot(page) and not _needs_trace_screenshot(graph, step):
                message_content.append({
                    "type": "text",
                    "text": (
                        f"[OBSERVATION_IMAGE_OMITTED] BEFORE_STEP_{raw_index} 是 FFG 已知导航页面；"
                        "保留页面、Activity 与动作文本，不重复上传其截图。"
                    ),
                })
            elif _page_screenshot(page):
                message_content.append({
                    "type": "text",
                    "text": (
                        f"[OBSERVATION_IMAGE_BUDGET_OMITTED] BEFORE_STEP_{raw_index} "
                        "不属于本轮 refinement 校验问题及相邻步骤；上一轮已提供截图，"
                        "本轮保留完整页面、Activity、动作和时序文本。"
                    ),
                })

        message_content.append({
            "type": "text",
            "text": (
                "【轨迹时序校验】\n"
                "- STEP i 的截图是动作 i 之前的源页面；动作结果应从下一张可用 BEFORE_STEP 截图判断。\n"
                "- 创建新页面时，source_step 必须指向该页面实际可见的截图步骤；"
                "例如 STEP 2 点击后页面首次出现在 BEFORE_STEP_3，则 source_step=3。\n"
                "- finished 只是一张最终状态观测，不是页面操作，也不能作为跳转触发动作。\n"
                "- executor_target_hint/executor_reasoning_hint 只是执行器解释；"
                "坐标和截图是更高优先级证据，冲突时以截图中的实际目标为准。\n"
                "- coordinate 使用 original_screenshot_size 的原始像素坐标；"
                "上传图片可能等比例缩小，定位时按宽高比例映射。\n"
                "- 最后一个有副作用的动作若没有后续观测，不得据此新增结果页、结果操作或跳转。"
            ),
        })

        text_chars = sum(
            len(str(block.get("text", "")))
            for block in message_content
            if block.get("type") == "text"
        )
        image_blocks = [
            block for block in message_content if block.get("type") == "image_url"
        ]
        image_payload_chars = sum(
            len(str(block.get("image_url", {}).get("url", "")))
            for block in image_blocks
        )
        print(
            "[FFG 更新] 上下文规模: "
            f"ffg_chars={len(ffg_summary)}, text_chars={text_chars}, "
            f"images={len(image_blocks)}, image_payload_chars={image_payload_chars}, "
            f"trace_steps={len(trace)}, reference_images={len(image_reference_pages)}, "
            f"image_budget={image_budget}, mandatory_trace_images={len(trace_screenshot_payloads)}, "
            f"image_candidates={len(trace_screenshot_payloads) + len(reference_candidates)}, "
            f"selected_images={len(image_blocks)}, "
            f"omitted_reference_images={len(reference_candidates) - len(image_reference_pages)}, "
            f"refinement_focus_steps={sorted(refinement_focus_steps or [])}"
        )

        if fast_path_page_map is not None:
            print(
                "[FFG 更新] 已知页面快速路径: Activity 与页面视觉高度匹配，"
                "操作和跳转由程序更新，跳过多模态模型。"
            )
            patch = {
                "update_control": {
                    "decision": "no_change",
                    "reason": "all observations match known pages; operations are deterministic",
                },
                "observation_alignment": [
                    {
                        "source_step": raw_step,
                        "page_decision": "reuse",
                        "page_ref": page_id,
                    }
                    for raw_step, page_id in sorted(fast_path_page_map.items())
                ],
                "functions": [],
                "pages": [],
                "page_ownership": [],
                "function_restructures": [],
                "function_merges": [],
                "function_dependencies": [],
            }
        else:
            remaining_run_seconds = _remaining_run_seconds(state)
            available_llm_seconds = (
                None
                if remaining_run_seconds is None
                else max(0.0, remaining_run_seconds - FFG_UPDATE_TIME_SAFETY_SECONDS)
            )
            if available_llm_seconds is not None and available_llm_seconds <= 0:
                print(
                    "[FFG 更新] 总运行时限已到，跳过多模态调用；"
                    "原始轨迹已落盘，本轮按执行结果收尾，不再触发无意义的对齐重试。"
                )
                deadline_result = _apply_non_recordable_execution_outcome(
                    graph,
                    state,
                    execution_result,
                )
                if audit_base_graph is not None:
                    deadline_result["ffg_update_base_ffg"] = audit_base_graph
                return deadline_result
            else:
                configured_attempts = max(
                    1,
                    int(os.getenv("LLM_RETRY_ATTEMPTS", "3") or 3),
                )
                max_attempts = configured_attempts
                deadline_monotonic = None
                if available_llm_seconds is not None:
                    max_attempts = max(
                        1,
                        min(
                            configured_attempts,
                            int(available_llm_seconds // FFG_UPDATE_LLM_TIMEOUT_SECONDS),
                        ),
                    )
                    deadline_monotonic = time.monotonic() + available_llm_seconds
                patch = invoke_llm_json_with_retry(
                    llm,
                    [
                        SystemMessage(content=FFG_UPDATE_SYSTEM_PROMPT),
                        HumanMessage(content=message_content),
                    ],
                    log_prefix="[FFG 更新]",
                    json_block_pattern=JSON_BLOCK_PATTERN,
                    max_attempts=max_attempts,
                    deadline_monotonic=deadline_monotonic,
                )
        if llm_json_result_failed(patch):
            print(
                "[FFG 更新] 底层多模态调用已耗尽重试，跳过对齐校验与 refinement；"
                "完整原始轨迹和截图已落盘，页面审计结果保持独立生效。"
            )
            failed_result = _apply_non_recordable_execution_outcome(
                graph,
                state,
                execution_result,
                reason="model patch unavailable; raw trace preserved",
            )
            if audit_base_graph is not None:
                failed_result["ffg_update_base_ffg"] = audit_base_graph
            return failed_result
        if not patch:
            print(
                "[FFG 更新] LLM 未返回有效 patch，跳过图增量；"
                "仍继续记录本轮执行结果、执行场景和失败进度。"
            )
            patch = {}

        _normalize_observation_alignment(
            patch,
            graph,
            trace,
            candidate_map,
        )
        auto_fixed = _ensure_ignorable_alignments(patch, trace)
        if auto_fixed:
            print(
                "[FFG 更新] 程序自动补齐外部/恢复步骤 ignore 对齐: "
                f"{auto_fixed} 项"
            )
        alignment_issues = validate_observation_alignment(
            graph,
            patch,
            trace,
            candidate_map,
        )

        update_control = patch.get("update_control", {}) if isinstance(patch.get("update_control"), dict) else {}
        update_decision = str(update_control.get("decision", "finish") or "finish").strip().lower()
        if update_decision not in {"continue", "finish", "no_change"}:
            update_decision = "finish"
        update_reason = str(update_control.get("reason", "") or "").strip()
        patch_result = apply_ffg_patch(graph, patch, trace, current_plan=current_plan)
        graph = patch_result["graph"]
        if patch_result.get("persisted_trace_images"):
            print(
                "[FFG 更新] 已将对齐到已有 Page 的轨迹截图持久化为状态证据: "
                f"{patch_result['persisted_trace_images']} 张"
            )
        functions_patch = patch_result["functions_patch"]
        pages_patch = patch_result["pages_patch"]
        dependencies_patch = patch_result["dependencies_patch"]
        page_ownership_patch = patch_result["page_ownership_patch"]
        function_restructures_patch = patch_result["function_restructures_patch"]
        function_merges_patch = patch_result["function_merges_patch"]
        function_replacements = patch_result.get("function_replacements", {})
        touched_function_ids = patch_result["touched_function_ids"]
        if not patch_result.get("validation_issues"):
            for source_function_id, replacement_ids in function_replacements.items():
                review_attempts.pop(source_function_id, None)
                for replacement_id in replacement_ids:
                    review_attempts.pop(replacement_id, None)
        for function_id in _record_screenshot_unavailable_page(
            graph,
            state,
            execution_result,
            patch_result.get("trace_page_map", {}),
        ):
            if function_id not in touched_function_ids:
                touched_function_ids.append(function_id)
        for function_id in _apply_trace_operations_and_transitions(
            graph,
            trace,
            patch_result.get("trace_page_map", {}),
        ):
            if function_id not in touched_function_ids:
                touched_function_ids.append(function_id)
        bound_operations = _apply_operation_bindings(
            graph,
            patch,
            trace,
            patch_result.get("trace_page_map", {}),
        )
        bound_operations += _auto_bind_unambiguous_trace_operations(
            graph,
            trace,
            patch_result.get("trace_page_map", {}),
        )
        if bound_operations:
            print(
                f"[FFG 更新] 已记录操作-控件显式绑定 {bound_operations} 条"
                "（覆盖命中率以绑定为权威来源）"
            )
        graph.recompute_all_coverages()
        print(
            "[FFG 更新] LLM patch: "
            f"functions={len(functions_patch)}, pages={len(pages_patch)}, "
            f"page_ownership={len(page_ownership_patch)}, "
            f"function_restructures={len(function_restructures_patch)}, "
            f"function_merges={len(function_merges_patch)}, "
            f"dependencies={len(dependencies_patch)}"
        )
        print(
            "[FFG 更新] LLM 决策: "
            f"decision={update_decision}, reason={update_reason or '（未提供）'}"
        )

        current_testing_function_id = str(state.get("current_testing_function_id", "")).strip()
        current_testing_function_status = str(state.get("current_testing_function_status", "not_started")).strip() or "not_started"
        if (
            current_testing_function_id
            and current_testing_function_status == "in_progress"
            and graph.get_node_test_status(current_testing_function_id) == "not_started"
        ):
            graph.set_node_test_status(current_testing_function_id, "in_progress")

        pending_intra_scenario = False
        if not is_refinement and test_scenario and scenario_completed and test_has_device_action:
            if current_plan.get("scenario_type") == "cross_test":
                src_function_id = str(current_plan.get("src_function_id", "")).strip()
                dst_function_id = str(current_plan.get("dst_function_id", "")).strip()
                if src_function_id and dst_function_id:
                    graph.add_cross_scenario(src_function_id, dst_function_id, test_scenario)
            else:
                target_function_id = _resolve_plan_target_function_id(graph, current_plan)
                if target_function_id and graph.get_node(target_function_id) is not None:
                    current_testing_function_id = target_function_id
                    pending_intra_scenario = True
                    if graph.get_node_test_status(target_function_id) == "not_started":
                        graph.set_node_test_status(target_function_id, "in_progress")
        if current_testing_function_id and graph.get_node(current_testing_function_id) is not None:
            current_testing_function_status = graph.get_node_test_status(current_testing_function_id)

        if is_refinement:
            function_failure_counts = dict(state.get("function_execution_failure_counts", {}) or {})
            cross_failure_counts = dict(state.get("cross_execution_failure_counts", {}) or {})
        else:
            function_failure_counts, current_testing_function_id, current_testing_function_status = _update_function_execution_progress(
                graph,
                state,
                execution_result,
                current_testing_function_id,
                current_testing_function_status,
                reset_function_ids=touched_function_ids,
            )
            cross_failure_counts = _update_cross_test_execution_progress(
                graph,
                state,
                execution_result,
            )
            function_failure_counts, current_testing_function_id, current_testing_function_status = _close_crashed_cross_test_scope(
                graph,
                state,
                execution_result,
                function_failure_counts,
                current_testing_function_id,
                current_testing_function_status,
            )

        if current_testing_function_id and graph.get_node(current_testing_function_id) is None:
            replacements = graph.resolve_function_ids(current_testing_function_id)
            print(
                "[FFG 更新] 当前功能已被结构调整替代，释放旧测试目标: "
                f"{current_testing_function_id} -> {','.join(replacements) or 'none'}"
            )
            current_testing_function_id = ""
            current_testing_function_status = "not_started"

        mapped_current_page = _map_current_page_after_update(
            graph,
            state,
            patch_result.get("trace_page_map", {}),
        )
        endpoint_function_id, endpoint_function_status = _endpoint_function_state(
            graph,
            mapped_current_page,
            current_plan,
            current_testing_function_id,
            touched_function_ids,
        )
        if endpoint_function_id and endpoint_function_id != current_testing_function_id:
            print(
                "[FFG 更新] 执行终点进入其它功能，下一轮顺着实际所在功能继续: "
                f"{current_testing_function_id or 'none'} -> {endpoint_function_id}"
            )
        if endpoint_function_id:
            current_testing_function_id = endpoint_function_id
            current_testing_function_status = endpoint_function_status
        if pending_intra_scenario:
            scenario_owner_id = _record_completed_intra_scenario(
                graph,
                current_plan,
                test_scenario,
                endpoint_function_id,
            )
            if scenario_owner_id and scenario_owner_id != str(current_plan.get("target_function_id", "")).strip():
                print(
                    "[FFG 更新] 目标功能无法解析，场景归入实际到达功能: "
                    f"{scenario_owner_id}"
                )
            if scenario_owner_id == current_testing_function_id:
                current_testing_function_status = graph.get_node_test_status(
                    scenario_owner_id
                )

        invariant_issues = [
            *alignment_issues,
            *list(patch_result.get("validation_issues", [])),
        ]
        if not invariant_issues:
            invariant_issues = validate_ffg_invariants(graph)
        missing_activities = _unrepresented_trace_activities(
            graph,
            trace,
            str(state.get("package_name", "") or ""),
            test_scenario,
            patch_result.get("trace_page_map", {}),
        )
        if missing_activities:
            invariant_issues.append(
                "执行轨迹出现尚未被任何 Page 表示的新 Activity，请复核并新增或正确复用页面: "
                + ", ".join(missing_activities)
            )
        if invariant_issues:
            update_decision = "continue"
            update_reason = "; ".join(invariant_issues[:8])
            print(f"[FFG 更新] 图一致性校验待修正: {update_reason}")

        after = _snapshot_graph(graph)
        print(
            "[FFG 更新] 增量: "
            f"功能={after['functions'] - before['functions']}, "
            f"页面={after['pages'] - before['pages']}, "
            f"操作={after['operations'] - before['operations']}, "
            f"页面跳转={after['page_transitions'] - before['page_transitions']}, "
            f"功能依赖={after['function_dependencies'] - before['function_dependencies']}"
        )

        if not is_refinement:
            (
                function_failure_counts,
                current_testing_function_id,
                current_testing_function_status,
            ) = _record_zero_structural_yield(
                graph,
                state,
                before,
                after,
                function_failure_counts,
                current_testing_function_id,
                current_testing_function_status,
            )

        result = {
            "ffg": graph,
            "current_page": mapped_current_page,
            "current_testing_function_id": current_testing_function_id,
            "current_testing_function_status": current_testing_function_status,
            "function_execution_failure_counts": function_failure_counts,
            "cross_execution_failure_counts": cross_failure_counts,
            "granularity_review_attempts": review_attempts,
            "ffg_granularity_review_function_id": granularity_review_function_id,
            "ffg_granularity_review_metrics": granularity_review_metrics,
            "ffg_update_decision": update_decision,
            "ffg_update_reason": update_reason,
            "ffg_update_validation_issues": invariant_issues,
        }
        if audit_base_graph is not None:
            result["ffg_update_base_ffg"] = audit_base_graph
        return result

    return ffg_update_node
