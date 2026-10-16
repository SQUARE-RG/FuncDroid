"""Lightweight FFG update node."""

import hashlib
import json
import os
import re
import time
from collections import OrderedDict
from difflib import SequenceMatcher
from typing import Any

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
from tools.utils import screenshot_dimensions
from workflow.image_compact import compact_screenshot
from workflow.llm import invoke_llm_json_with_retry, llm_json_result_failed
from workflow.nodes._shared import (
    JSON_BLOCK_PATTERN,
    screenshot_change_region,
    screenshot_digest,
    screenshot_thumbnail,
)
from workflow.nodes.prompt import FFG_UPDATE_SYSTEM_PROMPT, PAGE_AUDIT_SYSTEM_PROMPT
from workflow.page_audit_worker import DEFAULT_BATCH_SIZE, page_audit_worker
from workflow.state import AgentState

TEXT_NORMALIZE_PATTERN = re.compile(r"[\"'“”‘’（）()【】\[\]{}<>《》、，,。.：:；;!！?？\-_/\s]+")
# 覆盖率过半后，成功执行却零结构产出才计为无进展：探索初期本就需要几轮铺垫。
ZERO_YIELD_MIN_COVERAGE = 50
# 同一功能连续导航失败达到该次数即延后（reason=navigation_failed），不再无限重排。
NAVIGATION_FAILURE_DEFER_LIMIT = max(1, int(os.getenv("NAVIGATION_FAILURE_DEFER_LIMIT", "3")))
# 导航阶段的这些状态说明“没到达目标功能”，与目标功能自身能力无关。导航中
# 离开应用或崩溃同样没到达目标；不计入会让功能永远不被延后、每轮被重复选中。
NAVIGATION_FAILURE_STATUSES = {
    "blocked", "stalled", "incomplete", "failed_goal", "error", "left_app", "app_crashed",
}
TEST_DATA_PATTERN = re.compile(r"FuncDroid_[0-9A-Za-z_]+")
# 与 scenario_exec.EXPLORATION_BOUNDARY_BLOCKED_END 的固定文案保持一致。
EXPLORATION_BOUNDARY_ERROR_MARKER = "（非目标应用核心功能），已自动返回并结束当前场景"
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
FFG_UPDATE_LLM_RETRY_ATTEMPTS = max(1, int(os.getenv("LLM_RETRY_ATTEMPTS", "3") or 3))
FFG_UPDATE_TIME_SAFETY_SECONDS = max(
    0.0,
    float(os.getenv("FFG_UPDATE_TIME_SAFETY_SECONDS", "5") or 5),
)
# 每轮最多模型调用次数（首轮 + 修补轮）。修补轮只在仍有未解决问题（未对齐的
# 新 Activity、列举式控件名、被跳过的拆分/合并等）时触发，且只发送相关步骤；
# 前一轮的合法内容已写入，不回滚。
FFG_UPDATE_MAX_STEPS = max(1, int(os.getenv("FFG_UPDATE_MAX_STEPS", "3")))
FFG_UPDATE_MAX_REFINEMENT_IMAGES = max(
    1,
    int(os.getenv("FFG_UPDATE_MAX_REFINEMENT_IMAGES", "3")),
)
# 页面审计单任务携带的截图上限：优先放尚未审计过的截图，其次代表截图与最新
# 状态。证据历史已受 FFG_MAX_SCREENSHOTS_PER_PAGE 约束，这里约束的是单次审计
# LLM 调用规模，防止长期运行后单页截图数膨胀拖慢审计。
PAGE_AUDIT_MAX_IMAGES = max(2, int(os.getenv("PAGE_AUDIT_MAX_IMAGES", "6")))
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
# 跨轮保留的缩略图上限：已存页面截图在轮间基本不变，没必要每轮全部重新解码。
FFG_THUMBNAIL_CACHE_SIZE = max(64, int(os.getenv("FFG_THUMBNAIL_CACHE_SIZE", "4096")))
# 观测页解析的截图相似度阈值：低于此值不认为是已有页面。
OBSERVED_PAGE_MATCH_THRESHOLD = 0.98
EXECUTION_FAILURE_STATUSES = {
    "failed_goal",
    "blocked",
    "stalled",
    "incomplete",
    "left_app",
    "app_crashed",
    "screenshot_unavailable",
    "error",
}


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
    """跨轮复用的灰度缩略图缓存（有界 LRU，按截图内容摘要作键）。

    FFG 更新中候选检索、参考图挑选与观测页解析都会对 trace step 与已存页面
    截图两两比较；已存截图在轮间基本不变，因此缓存跨轮保留、不再每轮清空，
    超过上限按最久未用淘汰。键用摘要而不是原串，避免常驻大字符串。
    """

    def __init__(self, size: int = 64, max_entries: int = FFG_THUMBNAIL_CACHE_SIZE):
        self._size = size
        self._max_entries = max(1, int(max_entries))
        self._thumbnails: OrderedDict[str, np.ndarray | None] = OrderedDict()

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
        key = screenshot_digest(str(screenshot))
        if key in self._thumbnails:
            self._thumbnails.move_to_end(key)
            return self._thumbnails[key]
        thumbnail = screenshot_thumbnail(str(screenshot), self._size)
        self._thumbnails[key] = thumbnail
        while len(self._thumbnails) > self._max_entries:
            self._thumbnails.popitem(last=False)
        return thumbnail

    def similarity(self, left: str | None, right: str | None) -> float | None:
        left_thumb = self.thumbnail(left)
        right_thumb = self.thumbnail(right)
        if left_thumb is None or right_thumb is None:
            return None
        diff = np.mean(
            np.abs(left_thumb.astype(np.float32) - right_thumb.astype(np.float32))
        ) / 255.0
        return 1.0 - float(diff)



# 节点外部调用（黑屏页解析、当前页映射）没有节点实例时共用的进程级缓存。
_SHARED_THUMBNAIL_CACHE = _ScreenshotThumbnailCache()


def _best_page_similarity(cache, screenshot: str | None, page: Page) -> float | None:
    """运行时截图与某个已存页面全部截图的最高相似度（cache 需提供 similarity）。"""
    scores = [
        value
        for stored in [page.representative_screenshot, *page.evidence_screenshots]
        if (value := cache.similarity(screenshot, stored)) is not None
    ]
    return max(scores, default=None)


def _compact_screenshot_for_llm(screenshot: str) -> str:
    """压缩截图；实现在 workflow/image_compact 的进程级共享缓存上。

    同一张原图在本进程内只压缩一次，FFG 更新、缺陷判定与页面审计共享结果。
    """
    return compact_screenshot(
        screenshot,
        max_side=FFG_UPDATE_IMAGE_MAX_SIDE,
        quality=FFG_UPDATE_IMAGE_QUALITY,
        source_label="FFG 更新",
    )


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
    会让升级后存量页面全部判定为“未审计”触发审计风暴。截图只以内容摘要
    参与计算，避免每轮把全部 base64 原文拼成一个大字符串再哈希。
    """
    parts = [screenshot_digest(page.representative_screenshot or "")]
    parts.extend(screenshot_digest(item or "") for item in page.evidence_screenshots)
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
        # 审计基于提交时的页面快照；期间页面若有新截图或新操作，删除/合并
        # 判断可能已经过期（会误删刚被轨迹重新记录的真实操作），只保留新增控件。
        result_fingerprint = str(result.get("fingerprint", "") or "")
        stale = bool(result_fingerprint) and result_fingerprint != _page_audit_fingerprint(page)
        if stale:
            print(
                f"[页面审计] 页面在审计期间已变化，仅合并新增控件、跳过删除与合并: page={page_id}"
            )
        existing = list(page.controls)
        added: list[str] = []
        for control in result.get("missing_controls") or []:
            text = str(control or "").strip()
            if not text:
                continue
            if control_family_key(text) in PAGE_AUDIT_EXCLUDED_FAMILIES:
                print(f"[页面审计] 丢弃全局导航类候选控件（返回键可完成）: {text!r}")
                continue
            if _control_name_enumerates_items(text):
                print(f"[页面审计] 丢弃把多个控件列举成一条的候选: {text!r}")
                continue
            if _page_audit_semantically_duplicate(text, existing + added):
                print(f"[页面审计] 丢弃与既有控件语义重复的候选: {text!r}")
                continue
            added.append(text)
        for control in added:
            page.add_control(control)
        merged_controls = [] if stale else _apply_page_audit_control_merges(
            page,
            result.get("control_merges") or [],
        )
        removed = [] if stale else _apply_page_audit_operation_removals(
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


# 开关/切换类控件在两种状态下名字不同（“开启X按钮”/“关闭X按钮”），
# 截图位置相同，审计容易误判为同义名；两者是不同的操作意图，不能合并。
OPPOSITE_STATE_WORDS = (
    ("开启", "关闭"),
    ("打开", "关闭"),
    ("启用", "禁用"),
    ("显示", "隐藏"),
    ("展开", "收起"),
    ("播放", "暂停"),
    ("开始", "停止"),
    ("锁定", "解锁"),
    ("静音", "取消静音"),
    ("收藏", "取消收藏"),
    ("关注", "取消关注"),
)


def _sources_name_opposite_states(sources: list[str]) -> bool:
    for first, second in OPPOSITE_STATE_WORDS:
        # “取消收藏”本身含“收藏”，先剥掉反义词再判断正向词。
        has_first = [first in source.replace(second, "") for source in sources]
        has_second = [second in source for source in sources]
        if any(a and not b for a, b in zip(has_first, has_second)) and any(
            b and not a for a, b in zip(has_first, has_second)
        ):
            return True
    return False


def _apply_page_audit_control_merges(
    page: Page,
    merge_items: list[dict[str, Any]],
) -> list[str]:
    """Apply exact, high-confidence merges without losing bindings.

    两类合并：
    - 同类实例归并（如“商品A”“商品B”“商品C”→“商品列表项”）：至少 3 个来源，
      目标名必须包含来源的共同业务前缀；
    - 同一物理控件的同义名合并（如“搜索按钮”“搜索图标”→“搜索按钮”）：至少 2 个
      来源，且目标必须沿用其中一个已记录的原名，不引入新名字。
    """
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
        synonym_merge = len(sources) >= 2 and target in sources
        if (
            not (synonym_merge or len(sources) >= 3)
            or not target
            or any(source not in page.controls for source in sources)
        ):
            print(f"[页面审计] 丢弃来源不完整的控件合并: sources={sources!r}, target={target!r}")
            continue
        if synonym_merge and _sources_name_opposite_states(sources):
            print(f"[页面审计] 丢弃把开关两种状态当同义名的控件合并: sources={sources!r}")
            continue
        if not synonym_merge:
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


def _select_page_audit_screenshots(page_id: str, screenshots: list[str]) -> list[str]:
    """在单次审计上限内挑选截图：未审计过的优先，其次代表图与最新状态。"""
    if len(screenshots) <= PAGE_AUDIT_MAX_IMAGES:
        return list(screenshots)
    audited = page_audit_worker.audited_screenshot_digests(page_id)
    unaudited = [item for item in screenshots if screenshot_digest(item) not in audited]
    selected: list[str] = []
    for item in [*unaudited, screenshots[0], *reversed(screenshots)]:
        if item not in selected:
            selected.append(item)
        if len(selected) >= PAGE_AUDIT_MAX_IMAGES:
            break
    # 保持截图原有顺序，便于模型按状态先后阅读。
    order = {item: index for index, item in enumerate(screenshots)}
    return sorted(selected, key=lambda item: order[item])


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
        # 审计应逐步看完该 Page 已持久化的全部截图；单次调用另有规模上限。
        # 超限时优先放尚未审计过的截图（其次代表截图与最新状态），被省略的
        # 截图留到下一次审计，不会因为固定保留“代表图 + 最新”而永远看不到。
        all_screenshots = [
            screenshot
            for screenshot in dict.fromkeys([
                page.representative_screenshot,
                *page.evidence_screenshots,
            ])
            if str(screenshot or "").strip()
        ]
        screenshots = _select_page_audit_screenshots(page.id or "", all_screenshots)
        omitted = len(all_screenshots) - len(screenshots)
        used = 0
        total = len(screenshots)
        if omitted:
            content.append({
                "type": "text",
                "text": (
                    f"[PAGE_STATE_IMAGES_OMITTED] 因单次调用规模上限，本次省略 "
                    f"{omitted} 张截图（优先保留尚未审计过的状态），其余留待下次审计。"
                ),
            })
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
            "screenshot_digests": [screenshot_digest(item) for item in screenshots],
            "pending_screenshots": omitted,
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


def _filter_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """保留所有带页面观测的执行轨迹，由 FFG 更新节点统一判断如何落图。"""
    filtered: list[dict[str, Any]] = []
    for raw_index, step in enumerate(trace, start=1):
        if not isinstance(step, dict):
            continue
        copied = dict(step)
        copied["raw_step_index"] = raw_index
        filtered.append(copied)
    return filtered


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
            and _extract_page(step) is not None
            and not bool(step.get("observation_only"))
            and str(step.get("action_name", "")).strip() not in {"finished", "restart_app"}
        ):
            return step, raw_offset + index + 1
    return None, 0


def _black_screen_page(graph: FlowGraph) -> Page | None:
    return next(
        (page for page in graph.active_pages() if page.name == BLACK_SCREEN_PAGE_NAME),
        None,
    )


def _resolve_observed_page_id(
    graph: FlowGraph,
    step: dict[str, Any] | None,
    raw_step: int,
    trace_page_map: dict[int, str] | None = None,
    thumbnail_cache: _ScreenshotThumbnailCache | None = None,
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
    if not screenshot:
        return ""
    activity = str(_page_activity(page) or "").strip()
    cache = thumbnail_cache or _SHARED_THUMBNAIL_CACHE
    best_match = ""
    best_similarity = 0.0
    for candidate in graph.active_pages():
        if candidate.name == BLACK_SCREEN_PAGE_NAME:
            continue
        if activity and candidate.activity and activity != candidate.activity:
            continue
        similarity = _best_page_similarity(cache, screenshot, candidate)
        if similarity is not None and similarity > best_similarity:
            best_similarity = similarity
            best_match = str(candidate.id or "")
    return best_match if best_similarity >= OBSERVED_PAGE_MATCH_THRESHOLD else ""


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

    black_page = _black_screen_page(graph)
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
            black_page = _black_screen_page(graph)
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


def _trace_step_page_id(
    graph: FlowGraph,
    step: dict[str, Any],
    raw_step: int,
    trace_page_map: dict[int, str],
) -> str:
    if str(step.get("screenshot_health_error", "")).strip():
        black_page = _black_screen_page(graph)
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
            # 之前轮次已记录同一绑定：重复声明属正常现象，不是校验失败。
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
    must already exist in the same Page, and ambiguous matches are left for later
    LLM updates instead of guessing or creating controls.
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
    cache = thumbnail_cache if thumbnail_cache is not None else _SHARED_THUMBNAIL_CACHE
    page_owners = graph.page_owners

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
            image_similarity = _best_page_similarity(cache, runtime_screenshot, page) or 0.0
            if similarity_sink is not None:
                similarity_sink[page_id] = image_similarity
        owners = page_owners.get(page_id, [])
        scope_match = bool(scope_ids.intersection(owners))
        graph_adjacent = page_id in adjacent_targets
        base_score = (
            0.45 * image_similarity
            + 0.20 * (1.0 if activity_match else 0.35 if activity_compatible else 0.0)
            + 0.10 * float(scope_match)
            + 0.10 * float(graph_adjacent)
        )
        candidates.append({
            "page_id": page_id,
            "owners": owners,
            "name": _clip_context_text(page.name or page.description, 160),
            "activity": _clip_context_text(page.activity, 180),
            "score": 1.0 if exact_id else base_score,
            "exact_id": exact_id,
            "image_similarity": round(image_similarity, 4),
            "activity_match": activity_match,
            "activity_compatible": activity_compatible,
            "semantic_similarity": 0.0,
            "scope_match": scope_match,
            "graph_adjacent_from_previous": graph_adjacent,
            "_page": page,
        })
    # 文本相似度（SequenceMatcher）占权重 0.15 却最贵。最终得分 = 非文本分 +
    # 0.15×文本分，第 limit 名的最终分不低于第 limit 名的非文本分；非文本分加上
    # 文本满分仍低于该下界的页面不可能进入最终候选，可以精确跳过，排序结果不变。
    text_pool = sorted(candidates, key=lambda item: float(item["score"]), reverse=True)
    keep = max(1, limit)
    lower_bound = float(text_pool[keep - 1]["score"]) if len(text_pool) >= keep else 0.0
    for item in text_pool:
        if item["exact_id"]:
            continue
        if float(item["score"]) + 0.15 < lower_bound:
            break
        text_similarity = _semantic_text_similarity(
            runtime_text,
            _page_semantic_text(item["_page"]),
        )
        item["semantic_similarity"] = round(text_similarity, 4)
        item["score"] = float(item["score"]) + 0.15 * text_similarity
    for item in candidates:
        item.pop("_page", None)
        item["score"] = round(float(item["score"]), 4)
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
        + "\n- granularity_review_function_id 非空时，还要复核该已有 Function 的全部持久化页面；大功能下的子功能都应拆分为单独的功能节点：共享同一入口（如设置、我的、管理页）但各自承载独立配置域或独立业务流程的页面，应使用 function_restructures 拆分（优先 retire_source=false 的部分拆分，先把最明显的独立子域拆出，不必一次性全拆），而不是只处理本轮新页面。"
        + "\n- Activity 变化只是审查信号，不可单独作为创建 Function 的充分条件。"
    )


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
    dimensions = screenshot_dimensions(_page_screenshot(page))
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
            for value in re.findall(r"source_step\s+(\d+)", issue_text, re.IGNORECASE)
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

    cache = thumbnail_cache if thumbnail_cache is not None else _SHARED_THUMBNAIL_CACHE
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
                similarity = _best_page_similarity(cache, trace_screenshot, page)
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


# 一个控件名里列举多个名字（“工具项（A、B、C）”“A、B、C入口”）会把几个可以
# 单独点击的入口压成一条：覆盖率只算一个，其余入口不会被当作独立目标去测。
# 这里不能替模型决定该拆开还是只写类型名（同质资源实例本来就应只写类型名），
# 所以不写入这种名字，交给后续轮次或页面审计重新声明；“（如X、Y）”“（代表：X）”
# 这类示例说明不算列举。
CONTROL_ENUMERATION_BRACKET_PATTERN = re.compile(r"[（(]([^（）()]*)[)）]")
CONTROL_EXAMPLE_PREFIX_PATTERN = re.compile(r"^\s*(?:代表|如|例如|比如|e\.g\.|eg|such as)", re.IGNORECASE)


def _control_name_enumerates_items(name: str) -> bool:
    text = str(name or "").strip()
    if not text:
        return False
    for inner in CONTROL_ENUMERATION_BRACKET_PATTERN.findall(text):
        if CONTROL_EXAMPLE_PREFIX_PATTERN.match(inner) or inner.rstrip().endswith("等"):
            continue
        if re.search(r"[、/／;；]", inner) or len(re.findall(r"[,，]", inner)) >= 2:
            return True
    outside = CONTROL_ENUMERATION_BRACKET_PATTERN.sub("", text)
    return outside.count("、") >= 2


def _strip_enumerated_controls(patch: dict[str, Any]) -> list[str]:
    """从 pages patch 中移除列举式控件名，返回提示模型重新声明的校验问题。"""
    issues: list[str] = []
    for item in _list_patch(patch.get("pages")):
        controls = item.get("controls")
        if not isinstance(controls, list):
            continue
        rejected = [
            str(value).strip()
            for value in controls
            if isinstance(value, str) and _control_name_enumerates_items(value)
        ]
        if not rejected:
            continue
        item["controls"] = [
            value
            for value in controls
            if not (isinstance(value, str) and str(value).strip() in rejected)
        ]
        page_ref = _normalize_ref(item.get("page_ref")) or "unknown"
        for name in rejected:
            issues.append(
                f"{CONTROL_ENUMERATION_ISSUE_MARKER}: page_ref {page_ref}, "
                f"source_step {item.get('source_step', '')}, control {name!r}。"
                "若列举的是可分别点击、进入不同功能的入口，逐项写成独立控件；"
                "若只是仅资源值不同的同质实例，只写一个类型级名称，不附列举"
            )
    return issues


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
        if decision == "reuse" and page_ref in page_patch_by_ref and graph.get_page(page_ref) is None:
            # 引用本 patch 新建的页面（new_page_x）才是 create；reuse 已有 page_x 并在
            # pages 中同步更新它（补控件等）是合法写法，保持 reuse。
            item["page_decision"] = "create"
            continue
        if decision == "create":
            # 模型常把“复用已有页面”写成 create + 已存在的 page_id。此时既没有
            # 对应的 pages patch（无新页面可建），意图只可能是 reuse，属于纯格式
            # 错误：由代码归一化即可。
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
        if _is_ignorable_observation_step(step):
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


CONTROL_ENUMERATION_ISSUE_MARKER = "控件名把多个控件列举成一条，未写入"


def _is_ignorable_observation_step(step: dict[str, Any]) -> bool:
    """程序可判定为"可忽略观测"的轨迹步骤。

    包括：截图健康错误、运行器自动恢复/干扰步骤（is_interruption、
    foreground_recovery、observation_after_recovery）、以及页面观测本身
    位于目标应用之外（external_context，如桌面、其他 App）。这些步骤
    不参与目标应用的 Page 建模，模型标 ignore 是唯一正确答案。

    旧校验只认 (is_interruption 且 external_context) 同时成立，而自动
    恢复步骤只带 is_interruption、受控外部页面只带 external_context，
    导致模型无论标 ignore 还是漏掉都必然报错。
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
    ignorable_steps: set[int] = set()
    for index, step in enumerate(trace, start=1):
        if _extract_page(step) is None:
            continue
        raw_step = int(step.get("raw_step_index", index) or index)
        if not _is_ignorable_observation_step(step):
            continue
        ignorable_steps.add(raw_step)
        existing = by_step.get(raw_step)
        if existing is None:
            alignments.append({"source_step": raw_step, "page_decision": "ignore"})
            fixed += 1
        elif str(existing.get("page_decision", "") or "").strip().lower() != "ignore":
            existing["page_decision"] = "ignore"
            existing.pop("page_ref", None)
            fixed += 1

    # Alignment alone is not enough: a model can emit both `ignore` and a
    # `pages` entity sourced from the same external/system frame.  Remove such
    # entities before patch application so permission controllers, launchers,
    # file pickers, and recovery frames can never become target-App pages.
    pages = patch.get("pages")
    if isinstance(pages, list) and ignorable_steps:
        removed_page_refs = {
            _normalize_ref(item.get("page_ref"))
            for item in pages
            if isinstance(item, dict)
            and str(item.get("source_step", "")).strip().isdigit()
            and int(str(item.get("source_step")).strip()) in ignorable_steps
        }
        removed_page_refs.discard("")
        patch["pages"] = [
            item
            for item in pages
            if not (
                isinstance(item, dict)
                and str(item.get("source_step", "")).strip().isdigit()
                and int(str(item.get("source_step")).strip()) in ignorable_steps
            )
        ]
        if removed_page_refs and isinstance(patch.get("page_ownership"), list):
            patch["page_ownership"] = [
                item
                for item in patch["page_ownership"]
                if not isinstance(item, dict)
                or _normalize_ref(item.get("page_ref")) not in removed_page_refs
            ]
        if removed_page_refs and isinstance(patch.get("page_transitions"), list):
            patch["page_transitions"] = [
                item
                for item in patch["page_transitions"]
                if not isinstance(item, dict)
                or (
                    _normalize_ref(item.get("src_page_ref")) not in removed_page_refs
                    and _normalize_ref(item.get("dst_page_ref")) not in removed_page_refs
                )
            ]
    return fixed


def validate_observation_alignment(
    graph: FlowGraph,
    patch: dict[str, Any],
    trace: list[dict[str, Any]],
    candidate_map: dict[int, list[dict[str, Any]]],
) -> list[str]:
    """诊断模型的 observation_alignment 输出质量。

    只用于日志：对齐缺失或不规范时，页面归属由 pages patch 与程序兜底
    （_fill_unmapped_observations）完成，不阻断本轮更新。
    """
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

    page_patch_refs = {
        _normalize_ref(item.get("page_ref"))
        for item in _list_patch(patch.get("pages"))
        if _normalize_ref(item.get("page_ref"))
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
            if page_ref in page_patch_refs and graph.get_page(page_ref) is None:
                # 同一个新页面出现在多个 STEP：后续步骤 reuse 本 patch 的 new_page_x。
                continue
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
            # pages.source_step 只指向新页面首次出现的步骤；后续步骤同样可以
            # create/reuse 同一个 new_page_x，只要它在 pages patch 中声明过。
            if not page_ref.startswith("new_page_"):
                issues.append(f"create alignment must use new_page_x at source_step {raw_step}")
            elif page_ref not in page_patch_refs:
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
    semantic_name_key = _function_match_key(name)
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
        existing_semantic_name_key = _function_match_key(node.name)
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
    semantic_key = _function_match_key(item.get("name"))
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
        if semantic_key == _function_match_key(page.name):
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
    declared_name_key = _function_match_key(declared_name)
    target_owners = set(target_function_ids or [])
    best_page: Page | None = None
    best_ratio = 0.0
    for page in graph.active_pages():
        if not page.activity or page.activity != activity:
            continue
        existing_owners = set(graph.get_page_owners(str(page.id or "")))
        if target_owners and existing_owners and target_owners.isdisjoint(existing_owners):
            continue
        existing_name_key = _function_match_key(page.name)
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
        semantic_name_key = _function_match_key(name)
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
        if not ref:
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
        # 已解析过的临时引用（同一 patch 重复声明，或修补轮沿用上一轮的 new_page_x）
        # 更新已分配的页面，不重复新建。
        existing_page = graph.get_page(page_refs.get(ref, "")) or graph.get_page(ref)
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


def _plan_function_restructure(
    graph: FlowGraph,
    item: dict[str, Any],
    prefix: str,
    function_refs: dict[str, str],
    page_refs: dict[str, str],
) -> tuple[dict[str, Any] | None, list[str]]:
    """只读校验一条拆分；返回可安全应用的计划与诊断问题，不修改图。

    非法的页面/场景引用直接剔除并记入问题；没有任何合法 replacement 时
    整条跳过。要求退役源功能但页面、场景、资源或依赖没有完整承接时，整条
    跳过（否则源功能会被掏空成没有页面的空壳）。
    """
    issues: list[str] = []
    source_ref = _normalize_ref(item.get("source_function_ref"))
    source_id = _resolve_function_ref(source_ref, function_refs, graph)
    source = graph.get_node(source_id)
    if source is None:
        return None, [f"{prefix} references unknown source function {source_ref}"]
    replacements = _restructure_replacements(item)
    if not replacements:
        return None, [f"{prefix} has no replacements"]

    source_scenarios = list(source.scenarios)
    source_coverage = set(source.coverage_page_ids)
    planned: list[dict[str, Any]] = []
    for replacement_index, replacement in enumerate(replacements, start=1):
        replacement_prefix = f"{prefix}.replacements[{replacement_index}]"
        function_ref = _normalize_ref(replacement.get("function_ref"))
        function_id = _resolve_function_ref(function_ref, function_refs, graph)
        if graph.get_node(function_id) is None or function_id == source_id:
            issues.append(f"{replacement_prefix} references invalid target function {function_ref}")
            continue
        page_ids: list[str] = []
        raw_page_refs = replacement.get("page_refs")
        for value in raw_page_refs if isinstance(raw_page_refs, list) else []:
            page_id = _resolve_page_ref(_normalize_ref(value), page_refs, graph)
            if not page_id or page_id not in source_coverage:
                issues.append(
                    f"{replacement_prefix} cannot move non-coverage page {value} from {source_id}"
                )
                continue
            if page_id not in page_ids:
                page_ids.append(page_id)
        if not page_ids:
            issues.append(f"{replacement_prefix} has no valid coverage pages")
            continue
        entry_page_ids: list[str] = []
        raw_entry_refs = replacement.get("entry_page_refs")
        for value in raw_entry_refs if isinstance(raw_entry_refs, list) else []:
            page_id = _resolve_page_ref(_normalize_ref(value), page_refs, graph)
            if graph.get_page(page_id) is None:
                issues.append(f"{replacement_prefix} references unknown entry page {value}")
                continue
            entry_page_ids.append(page_id)
        scenario_indexes: list[int] = []
        raw_scenario_refs = replacement.get("scenario_refs")
        for value in raw_scenario_refs if isinstance(raw_scenario_refs, list) else []:
            scenario_index = _scenario_ref_index(value)
            if scenario_index < 1 or scenario_index > len(source_scenarios):
                issues.append(f"{replacement_prefix} references unknown scenario {value}")
                continue
            scenario_indexes.append(scenario_index)
        planned.append({
            "function_id": function_id,
            "page_ids": page_ids,
            "entry_page_ids": entry_page_ids,
            "scenario_indexes": scenario_indexes,
        })
    if not planned:
        return None, issues

    replacement_ids = list(dict.fromkeys(entry["function_id"] for entry in planned))
    moved_page_ids = {page_id for entry in planned for page_id in entry["page_ids"]}
    assigned_scenarios = {index for entry in planned for index in entry["scenario_indexes"]}
    retire_source = bool(item.get("retire_source"))
    if retire_source:
        retire_issue = ""
        leftover_pages = [page_id for page_id in source.coverage_page_ids if page_id not in moved_page_ids]
        missing_scenarios = [
            f"scenario_{index}"
            for index in range(1, len(source_scenarios) + 1)
            if index not in assigned_scenarios
        ]
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
        replacement_id_set = set(replacement_ids)
        missing_dependencies = [
            dependency
            for dependency in graph.function_dependencies
            if source_id in {dependency.src_function_id, dependency.dst_function_id}
            and not _dependency_has_replacement(graph, dependency, source_id, replacement_id_set)
        ]
        if len(replacement_ids) < 2:
            retire_issue = f"{prefix} must have at least two replacements to retire {source_id}"
        elif leftover_pages:
            retire_issue = (
                f"{prefix} leaves coverage pages on retired source {source_id}: "
                + ",".join(leftover_pages)
            )
        elif missing_scenarios:
            retire_issue = (
                f"{prefix} leaves scenarios on retired source {source_id}: "
                + ",".join(missing_scenarios)
            )
        elif missing_resources:
            retire_issue = (
                f"{prefix} would drop resources from {source_id}: " + ",".join(missing_resources)
            )
        elif missing_dependencies:
            retire_issue = f"{prefix} would drop function dependencies from {source_id}"
        if retire_issue:
            issues.append(retire_issue)
            return None, issues
    return {
        "source_id": source_id,
        "source_scenarios": source_scenarios,
        "replacements": planned,
        "replacement_ids": replacement_ids,
        "moved_page_ids": moved_page_ids,
        "retire_source": retire_source,
    }, issues


def _apply_function_restructures(
    graph: FlowGraph,
    patch_items: list[dict[str, Any]],
    function_refs: dict[str, str],
    page_refs: dict[str, str],
) -> tuple[dict[str, list[str]], list[str]]:
    """把粗粒度功能的已存页面重新分配给更细的功能。

    每条拆分先只读校验（_plan_function_restructure），再整体应用：校验失败的
    条目或引用被剔除，不会留下“改了一半”的图。只有页面、场景、资源与依赖
    都有承接时才退役源功能，否则只迁移页面。
    """
    replacements_by_source: dict[str, list[str]] = {}
    issues: list[str] = []
    for item_index, item in enumerate(patch_items, start=1):
        plan, item_issues = _plan_function_restructure(
            graph, item, f"function_restructures[{item_index}]", function_refs, page_refs,
        )
        issues.extend(item_issues)
        if plan is None or graph.get_node(plan["source_id"]) is None:
            continue
        source_id = plan["source_id"]
        for entry in plan["replacements"]:
            function_id = entry["function_id"]
            target = graph.get_node(function_id)
            if target is None:
                continue
            for page_id in entry["page_ids"]:
                if (page := graph.get_page(page_id)) is not None:
                    graph.add_page(page, function_id)
            for page_id in entry["entry_page_ids"]:
                if (page := graph.get_page(page_id)) is not None:
                    graph.add_entry_page(page, function_id)
            for scenario_index in entry["scenario_indexes"]:
                target.add_scenario(plan["source_scenarios"][scenario_index - 1])
            if target.test_status == "not_started" and (
                entry["scenario_indexes"]
                or any(
                    (page := graph.get_page(page_id)) is not None and page.operations
                    for page_id in entry["page_ids"]
                )
            ):
                target.test_status = "in_progress"

        replacement_ids = plan["replacement_ids"]
        graph.invalidate_cross_tests(
            {source_id, *replacement_ids}, "function boundary restructured; requires revalidation"
        )
        for page_id in plan["moved_page_ids"]:
            graph.remove_page_owner(page_id, source_id)
        if plan["retire_source"]:
            source = graph.get_node(source_id)
            # 剩下的只是导航壳页面；replacement 的 entry_page_refs 已保留所需入口。
            for page_id in list(source.page_ids if source is not None else []):
                graph.remove_page_owner(page_id, source_id)
            graph.function_dependencies = [
                dependency
                for dependency in graph.function_dependencies
                if source_id not in {dependency.src_function_id, dependency.dst_function_id}
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
    """Merge two or more existing Functions into one target.

    所有前提（来源数量、目标存在、来源未被本 patch 消费）都在修改前校验；
    不满足的条目整条跳过。
    """
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
        # 失效页面引用只是陈旧的成员关系：跳过即可，不影响合并其余部分。
        for page_id in [page_id for page_id in page_roles if graph.get_page(page_id) is None]:
            issues.append(f"{prefix} references inactive page {page_id}")
            page_roles.pop(page_id)
        for page_id, is_coverage_page in page_roles.items():
            page = graph.get_page(page_id)
            if is_coverage_page:
                graph.add_page(page, target_id)
            else:
                graph.add_entry_page(page, target_id)

        target.intra_attempt_count = sum(
            node.intra_attempts() for node in {node.id: node for node in [target, *source_nodes]}.values()
        )
        graph.invalidate_cross_tests({target_id, *source_ids}, "functions merged; requires revalidation")
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
    for item in graph.cross_test_results:
        src, dst = item.get("src_function_id"), item.get("dst_function_id")
        if src not in active_ids or dst not in active_ids or src == dst:
            issues.append("graph has dangling or self-referencing cross-test result")
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


def _resolve_plan_target_function_id(
    graph: FlowGraph,
    current_plan: dict[str, Any],
) -> str:
    target_function_id = str(current_plan.get("target_function_id", "")).strip()
    if graph.get_node(target_function_id) is not None:
        return graph.resolve_function_id(target_function_id)
    return ""


def _is_depth_plan(current_plan: dict[str, Any]) -> bool:
    """本轮规划是否为功能内深度场景（阶段由规划器写入 current_plan）。"""
    return str(current_plan.get("function_phase", "") or "").strip().lower() == "depth"


def _is_exploratory_navigation_plan(current_plan: dict[str, Any]) -> bool:
    """本轮导航是否为 D 级探索式导航（FFG 中没有到目标的已记录路径）。"""
    return str(current_plan.get("navigation_level", "") or "").strip().upper() == "D"


def _record_completed_intra_scenario(
    graph: FlowGraph,
    current_plan: dict[str, Any],
    test_scenario: str,
) -> str:
    """把完成的场景记到规划时的目标功能上。"""
    owner_id = _resolve_plan_target_function_id(graph, current_plan)
    if owner_id and test_scenario:
        graph.add_scenario(owner_id, test_scenario)
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


def _navigation_failed(execution_result: dict[str, Any]) -> bool:
    """导航阶段没有到达目标功能（测试阶段未开始）。"""
    return (
        str(execution_result.get("scenario_status", "")).strip() in {"", "skipped"}
        and str(execution_result.get("navigation_status", "")).strip() in NAVIGATION_FAILURE_STATUSES
    )


def _test_phase_started(execution_result: dict[str, Any]) -> bool:
    return str(execution_result.get("scenario_status", "")).strip() not in {"", "skipped"}


def _trace_test_data_values(trace: list[dict[str, Any]]) -> list[str]:
    """执行轨迹中成功输入的 FuncDroid_ 自建数据，按出现顺序去重。"""
    values: list[str] = []
    for step in trace or []:
        if not isinstance(step, dict) or step.get("action_failed"):
            continue
        if str(step.get("action_name", "")).strip() != "type":
            continue
        for value in TEST_DATA_PATTERN.findall(str(step.get("content", "") or "")):
            if value not in values:
                values.append(value)
    return values


def _is_screenshot_unavailable_execution(execution_result: dict[str, Any]) -> bool:
    return "screenshot_unavailable" in {
        str(execution_result.get("navigation_status", "")).strip(),
        str(execution_result.get("scenario_status", "")).strip(),
    }


def _left_target_app(execution_result: dict[str, Any]) -> bool:
    return bool(execution_result.get("crash_detected")) or bool(
        {"app_crashed", "left_app"}
        & {
            str(execution_result.get("navigation_status", "")).strip(),
            str(execution_result.get("scenario_status", "")).strip(),
        }
    )


def _execution_result_for_llm(execution_result: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(execution_result or {})
    sanitized.pop("crash_logcat", None)
    return sanitized


def _record_execution_outcome(
    graph: FlowGraph,
    state: AgentState,
    execution_result: dict[str, Any],
    yielded_structure: bool = True,
) -> dict[str, int]:
    """把本轮执行结果写回调度状态，返回新的 function_execution_failure_counts。

    - 执行节点基础设施异常（infrastructure_error）不是功能尝试，不改任何调度状态。
    - 跨功能：completed 记 done（不再重排），失败记 failed；导航类失败与截图不可用
      标为 retryable，由规划器有限次重试；D 级探索式导航失败不重试。
    - 功能内：只有测试阶段的 blocked（验证码、安全策略等目标功能自身的硬墙）标记
      unavailable；导航阶段的 blocked 属路径问题，与其它导航失败一样累计
      navigation_failure_count，达到上限才延后（deferred，reason=navigation_failed）；
      D 级探索式导航只尝试一次，失败即延后（reason=exploratory_navigation_failed）。
      截图不可用属环境问题不计失败；其余失败打上失败标记让出调度优先级。
      成功执行清除标记；但覆盖率过半后仍无任何结构产出（页面/功能/操作/绑定/覆盖率
      都没增长，典型是 WebView 长文这类无底洞页面）同样打标记。
      已处于终态的功能不因执行结果翻转状态。
    """
    failure_counts = dict(state.get("function_execution_failure_counts", {}) or {})
    if execution_result.get("infrastructure_error"):
        return failure_counts
    current_plan = state.get("current_plan", {}) or {}
    scenario_type = str(current_plan.get("scenario_type", "")).strip()
    failure_reason = _execution_failure_reason(execution_result)
    completed = execution_result.get("scenario_status") == "completed"

    if scenario_type == "cross_test":
        src_function_id = str(current_plan.get("src_function_id", "")).strip()
        dst_function_id = str(current_plan.get("dst_function_id", "")).strip()
        if not src_function_id or not dst_function_id or not (completed or failure_reason):
            return failure_counts
        resource = str(current_plan.get("resource_variable", "") or "").strip()
        test_scenario = (
            str(state.get("current_test_scenario", "") or "").strip()
            or str(execution_result.get("current_scenario", "") or "").strip()
            or "跨功能场景执行失败"
        )
        if completed:
            graph.record_cross_test(src_function_id, dst_function_id, resource, test_scenario, "done")
        else:
            # 没走到测试阶段（导航失败）或截图不可用：组合本身尚未被检验，允许重试；
            # D 级探索式导航失败说明图中没有路径，重试只会重复同样的探索。
            navigation_failed = _navigation_failed(execution_result)
            retryable = (
                navigation_failed and not _is_exploratory_navigation_plan(current_plan)
            ) or _is_screenshot_unavailable_execution(execution_result)
            graph.record_cross_test(
                src_function_id, dst_function_id, resource, test_scenario,
                "failed", reason=failure_reason, retryable=retryable,
            )
            if _left_target_app(execution_result) and graph.get_node(src_function_id) is not None:
                # 崩溃/离开 App 通常由源功能的写入触发，降低源功能调度优先级。
                failure_counts[src_function_id] = 1
            print(
                "[FFG 更新] 跨功能组合执行失败，记录结果"
                f"（{'允许有限次重试' if retryable else '不再重复规划'}）: "
                f"{src_function_id} -> {dst_function_id} | {failure_reason}"
            )
        return failure_counts

    if scenario_type != "intra_test":
        return failure_counts
    target_function_id = _resolve_plan_target_function_id(graph, current_plan)
    target_node = graph.get_node(target_function_id) if target_function_id else None
    if target_node is None:
        return failure_counts
    if _test_phase_started(execution_result):
        target_node.navigation_failure_count = 0
    if completed and not failure_reason:
        coverage = int(graph.function_coverage_details(target_function_id)["coverage_percent"])
        if yielded_structure or coverage < ZERO_YIELD_MIN_COVERAGE:
            failure_counts.pop(target_function_id, None)
        else:
            failure_counts[target_function_id] = 1
            print(
                "[FFG 更新] 功能本轮无新增页面或功能，降低调度优先级: "
                f"{target_function_id} | coverage={coverage}%"
            )
        return failure_counts
    if not failure_reason or _is_screenshot_unavailable_execution(execution_result):
        return failure_counts

    current_status = graph.get_node_test_status(target_function_id)
    if current_status in TERMINAL_FUNCTION_STATUSES:
        # 覆盖审计轮中已完成的 Function 不因执行失败翻转终端状态：
        # 状态翻转会改变审计签名，导致同一功能被反复全量重审并烧掉轮次。
        print(
            "[FFG 更新] 功能已处于终端状态，执行失败不改变其状态: "
            f"{target_function_id}({current_status}) | {failure_reason}"
        )
        return failure_counts
    if _navigation_failed(execution_result):
        target_node.navigation_failure_count += 1
        failure_counts[target_function_id] = 1
        if _is_exploratory_navigation_plan(current_plan):
            graph.set_node_test_status(
                target_function_id, "deferred", reason="exploratory_navigation_failed"
            )
            print(
                "[FFG 更新] 探索式导航（D 级）未到达目标功能，只尝试一次，延后该功能: "
                f"{target_function_id} | {failure_reason}"
            )
        elif target_node.navigation_failure_count >= NAVIGATION_FAILURE_DEFER_LIMIT:
            graph.set_node_test_status(target_function_id, "deferred", reason="navigation_failed")
            print(
                "[FFG 更新] 功能连续导航失败达到上限，延后并等待路径变化或收尾补测: "
                f"{target_function_id} | {target_node.navigation_failure_count}/"
                f"{NAVIGATION_FAILURE_DEFER_LIMIT} | {failure_reason}"
            )
        else:
            print(
                "[FFG 更新] 导航未到达目标功能，记录导航失败并降低调度优先级: "
                f"{target_function_id} | {target_node.navigation_failure_count}/"
                f"{NAVIGATION_FAILURE_DEFER_LIMIT} | {failure_reason}"
            )
        return failure_counts
    if (
        str(execution_result.get("scenario_status", "")).strip() == "blocked"
        and EXPLORATION_BOUNDARY_ERROR_MARKER in str(execution_result.get("scenario_error", "") or "")
    ):
        # 探索边界只拦截某个越界入口，功能本身仍可从其他控件继续测试；
        # 标 unavailable 会让整个功能（含其余未覆盖控件）不再被调度。
        failure_counts[target_function_id] = 1
        print(
            "[FFG 更新] 本轮触达探索边界，记录失败但保留功能可测: "
            f"{target_function_id} | {failure_reason}"
        )
        return failure_counts
    if str(execution_result.get("scenario_status", "")).strip() == "blocked":
        graph.set_node_test_status(target_function_id, "unavailable", reason="blocked")
        failure_counts.pop(target_function_id, None)
        print(
            "[FFG 更新] 当前功能遇到无法继续的阻塞，标记为 unavailable 并切换其它功能: "
            f"{target_function_id} | {failure_reason}"
        )
        return failure_counts
    failure_counts[target_function_id] = 1
    print(
        "[FFG 更新] 记录功能执行失败，降低调度优先级: "
        f"{target_function_id} | {failure_reason}"
    )
    return failure_counts


def _trace_has_observed_business_response(trace: list[dict]) -> bool:
    """A submitted action or model 'finished' alone cannot prove completion.

    Require a healthy subsequent screenshot before another business action or
    automatic recovery. Status-bar changes are excluded by the shared comparer.
    This proves an observed response, not the correctness of the business result.
    """
    for before, after in zip(trace, trace[1:]):
        if not isinstance(before, dict) or not isinstance(after, dict):
            continue
        if str(before.get("action_name", "")) not in {"click", "long_press", "type", "scroll", "swipe"}:
            continue
        if any(step.get(flag) for step in (before, after) for flag in (
            "action_failed", "is_interruption", "foreground_recovery", "external_context",
            "screenshot_health_error", "observation_only",
        )):
            continue
        change = screenshot_change_region(
            _page_screenshot(_extract_page(before)), _page_screenshot(_extract_page(after)),
        )
        if change is not None and change[1] is not None and change[0] >= 0.0002:
            return True
    return False


def _record_intra_attempt(graph: FlowGraph, state: AgentState) -> None:
    plan = state.get("current_plan", {}) or {}
    execution_result = state.get("execution_result") or {}
    status = str(execution_result.get("scenario_status", ""))
    # 导航失败（skipped）由 navigation_failure_count 单独计；执行节点基础设施异常不是功能尝试。
    if (
        plan.get("scenario_type") != "intra_test"
        or status in {"", "skipped"}
        or execution_result.get("infrastructure_error")
    ):
        return
    node = graph.get_node(_resolve_plan_target_function_id(graph, plan))
    if node is not None:
        node.record_intra_attempt()
        # 深度场景只要进入了测试阶段就计数，测试阶段失败同样算做过一次。
        if _is_depth_plan(plan):
            node.depth_scenario_count += 1
            if status == "completed" and _trace_has_observed_business_response(
                state.get("scenario_execution_trace", []) or [],
            ):
                node.successful_depth_scenario_count += 1


def _record_execution_facts(
    graph: FlowGraph,
    state: AgentState,
    execution_result: dict[str, Any],
    *,
    yielded_structure: bool = True,
    record_attempt_scenario: bool = True,
) -> dict[str, int]:
    """写入本轮执行事实：场景记录、跨功能结果、测试数据登记与功能状态。

    执行事实独立于模型结构补丁：模型调用失败或补丁为空时同样写入。
    返回 function_execution_failure_counts。
    """
    current_plan = state.get("current_plan", {}) or {}
    scenario_type = str(current_plan.get("scenario_type", "")).strip()
    test_scenario = str(state.get("current_test_scenario", "") or "").strip()
    test_trace = [
        step for step in list(state.get("scenario_execution_trace", []) or [])
        if isinstance(step, dict)
    ]
    test_has_device_action = _trace_has_device_action(test_trace)
    scenario_status = str(execution_result.get("scenario_status", "") or "").strip()
    if test_scenario and not execution_result.get("infrastructure_error"):
        if scenario_type == "intra_test":
            target_function_id = _resolve_plan_target_function_id(graph, current_plan)
            if target_function_id and (
                (record_attempt_scenario and test_trace and test_has_device_action)
                or (not test_trace and scenario_status in EXECUTION_FAILURE_STATUSES)
            ):
                # 尝试历史是执行证据：即使所有步骤随后都被过滤出建图也要保留。
                graph.add_scenario(target_function_id, test_scenario)
            if scenario_status == "completed" and test_has_device_action:
                _record_completed_intra_scenario(graph, current_plan, test_scenario)
            if target_function_id:
                graph.record_test_data(
                    target_function_id, _trace_test_data_values(test_trace),
                    iteration=int(state.get("iteration_count", 0) or 0), scenario=test_scenario,
                )
        elif scenario_type == "cross_test":
            src_function_id = str(current_plan.get("src_function_id", "")).strip()
            dst_function_id = str(current_plan.get("dst_function_id", "")).strip()
            if src_function_id and dst_function_id and scenario_status == "completed" and test_has_device_action:
                graph.add_cross_scenario(src_function_id, dst_function_id, test_scenario)
            if src_function_id:
                graph.record_test_data(
                    src_function_id, _trace_test_data_values(test_trace),
                    iteration=int(state.get("iteration_count", 0) or 0), scenario=test_scenario,
                )
    return _record_execution_outcome(
        graph, state, execution_result, yielded_structure=yielded_structure,
    )


def _apply_non_recordable_execution_outcome(
    graph: FlowGraph,
    state: AgentState,
    execution_result: dict[str, Any],
    *,
    reason: str = "no recordable trace",
) -> dict[str, Any]:
    _record_screenshot_unavailable_page(graph, state, execution_result)
    current_page = _map_current_page_after_update(graph, state)
    # 无结构补丁：执行事实只按轨迹本身记录（场景、跨功能结果、状态）。
    failure_counts = _record_execution_facts(graph, state, execution_result)
    return {
        "ffg": graph,
        "current_page": current_page,
        "function_execution_failure_counts": failure_counts,
        "ffg_update_decision": "finish",
        "ffg_update_reason": reason,
        "ffg_update_validation_issues": [],
    }


def _fill_unmapped_observations(
    graph: FlowGraph,
    trace: list[dict[str, Any]],
    trace_page_map: dict[int, str],
    current_plan: dict[str, Any],
    thumbnail_cache: _ScreenshotThumbnailCache | None = None,
) -> list[str]:
    """程序兜底：把模型没有对齐的观测步骤落到图上，返回新建的页面 id。

    先按截图相似度复用已有页面；仍无法对应、且其 Activity 在图中完全没有
    页面表示时，为本轮规划目标功能补建一个页面（同一 Activity 只建一个）。
    这样即使模型漏写对齐或调用失败，本轮执行到的页面、操作与跳转也会写入
    FFG；页面名称与控件可由后续轮次与页面审计补全。
    """
    owner_id = next(
        (
            resolved
            for key in ("target_function_id", "src_function_id", "dst_function_id")
            if (resolved := graph.resolve_function_id(str(current_plan.get(key, "") or "").strip()))
        ),
        "",
    )
    represented = {
        str(page.activity or "").strip()
        for page in graph.active_pages()
        if str(page.activity or "").strip()
    }
    auto_pages: dict[str, str] = {}
    created: list[str] = []
    for index, step in enumerate(trace, start=1):
        runtime_page = _extract_page(step)
        if runtime_page is None or _is_ignorable_observation_step(step):
            continue
        raw_step = int(step.get("raw_step_index", index) or index)
        mapped = str(trace_page_map.get(raw_step, "") or "")
        if mapped and graph.get_page(mapped) is not None:
            continue
        resolved = _resolve_observed_page_id(graph, step, raw_step, None, thumbnail_cache)
        if resolved:
            trace_page_map[raw_step] = resolved
            continue
        activity = str(_page_activity(runtime_page) or "").strip()
        if not owner_id or not activity:
            continue
        page_id = auto_pages.get(activity, "")
        if page_id:
            _merge_page_observation(graph.get_page(page_id), runtime_page)
        elif activity not in represented:
            page_id, _ = _allocate_page_id(graph, _next_numeric_id(list(graph.pages), "page_"))
            page = _clone_page(runtime_page, page_id)
            page.name = page.name or activity.rsplit(".", 1)[-1] or page_id
            page.description = page.description or "程序按未建模 Activity 自动补建的页面，名称与控件待后续复核"
            graph.add_page(page, owner_id)
            auto_pages[activity] = page_id
            created.append(page_id)
        else:
            continue
        trace_page_map[raw_step] = page_id
    if created:
        print(
            "[FFG 更新] 模型未覆盖的新 Activity 已由程序补建页面: "
            + ", ".join(f"{page_id}({graph.get_page(page_id).activity})" for page_id in created)
        )
    return created


def _repair_ffg_invariants(graph: FlowGraph, current_plan: dict[str, Any]) -> list[str]:
    """确定性修复结构不一致，返回修复前发现的问题（仅用于日志）。

    取代旧的“校验失败整轮回滚”：失效的成员、跳转与依赖由
    reconcile_page_membership 清理；失效的跨功能结果归档到历史；没有属主的
    页面挂到本轮规划目标或与之相邻页面的属主上，保证本轮观测不丢。
    """
    issues = validate_ffg_invariants(graph)
    if not issues:
        return []
    graph.reconcile_page_membership()
    active_ids = set(graph.functions)
    dangling = {
        (item.get("src_function_id"), item.get("dst_function_id"))
        for item in graph.cross_test_results
        if item.get("src_function_id") not in active_ids
        or item.get("dst_function_id") not in active_ids
        or item.get("src_function_id") == item.get("dst_function_id")
    }
    if dangling:
        retained = []
        for item in graph.cross_test_results:
            if (item.get("src_function_id"), item.get("dst_function_id")) in dangling:
                graph.cross_test_history.append({**item, "invalidation_reason": "inactive function reference"})
            else:
                retained.append(item)
        graph.cross_test_results = retained
    fallback_owner = next(
        (
            resolved
            for key in ("target_function_id", "src_function_id", "dst_function_id")
            if (resolved := graph.resolve_function_id(str(current_plan.get(key, "") or "").strip()))
        ),
        "",
    )
    owners = graph.page_owners
    for page_id, page in graph.pages.items():
        if owners.get(page_id):
            continue
        neighbor_owner = next(
            (
                owner
                for transition in graph.page_transitions
                if page_id in {transition.src_page_id, transition.dst_page_id}
                for owner in owners.get(
                    transition.dst_page_id if transition.src_page_id == page_id else transition.src_page_id,
                    [],
                )
            ),
            "",
        )
        owner_id = fallback_owner or neighbor_owner
        if owner_id:
            graph.add_page(page, owner_id)
    print("[FFG 更新] 已确定性修复结构不一致: " + "; ".join(issues[:8]))
    return issues


def _run_patch_stage(name: str, func, *args, default=None, **kwargs):
    """单个 patch 阶段尽力执行：异常只丢弃该阶段，其余阶段与执行事实照常写入。"""
    try:
        return func(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - 单个模型字段异常不应丢掉整轮观测
        print(f"[FFG 更新] patch 阶段 {name} 异常，已跳过该阶段: {type(exc).__name__}: {exc}")
        return default


def apply_ffg_patch(
    graph: FlowGraph,
    patch: dict[str, Any],
    trace: list[dict[str, Any]],
    current_plan: dict[str, Any] | None = None,
    thumbnail_cache: _ScreenshotThumbnailCache | None = None,
    *,
    fill_unmapped: bool = True,
    carried_function_refs: dict[str, str] | None = None,
    carried_page_refs: dict[str, str] | None = None,
    carried_trace_page_map: dict[int, str] | None = None,
) -> dict[str, Any]:
    """把模型 patch 尽力应用到图上，不整轮回滚。

    各阶段独立校验：非法条目被跳过并记入 validation_issues，合法部分照常
    写入；最后确定性修复结构不一致。fill_unmapped=True 时由程序为模型未对齐
    的观测兜底落图（最后一次模型调用之后才启用，之前把未对齐留给修补轮）。
    carried_* 让修补轮沿用上一轮已解析的临时引用与步骤映射。
    """
    current_plan = current_plan or {}
    control_naming_issues = _strip_enumerated_controls(patch)
    for issue in control_naming_issues:
        print(f"[FFG 更新] {issue}")
    functions_patch = _list_patch(patch.get("functions"))
    pages_patch = _list_patch(patch.get("pages"))
    dependencies_patch = _list_patch(patch.get("function_dependencies"))
    observation_alignment = _list_patch(patch.get("observation_alignment"))
    page_ownership_patch = _list_patch(patch.get("page_ownership"))
    function_restructures_patch = _list_patch(patch.get("function_restructures"))
    function_merges_patch = _list_patch(patch.get("function_merges"))

    function_refs: dict[str, str] = dict(carried_function_refs or {})
    page_refs: dict[str, str] = dict(carried_page_refs or {})
    trace_page_map: dict[int, str] = dict(carried_trace_page_map or {})
    persisted_evidence_refs = _persisted_restructure_function_refs(graph, function_restructures_patch)
    persisted_evidence_refs.update(_persisted_merge_function_refs(graph, function_merges_patch))
    existing_function_ids = set(graph.functions)

    _run_patch_stage(
        "functions", _apply_functions, graph, functions_patch, trace, function_refs, page_refs,
        _next_numeric_id(list(graph.functions), "func_"),
        persisted_evidence_refs=persisted_evidence_refs,
    )
    _run_patch_stage(
        "pages", _apply_pages, graph, pages_patch, trace, function_refs, page_refs,
        trace_page_map, _next_numeric_id(list(graph.pages), "page_"),
    )
    _run_patch_stage(
        "observation_alignment", _apply_observation_alignment,
        graph, observation_alignment, page_refs, trace_page_map,
    )
    auto_created_pages = _run_patch_stage(
        "fill_unmapped", _fill_unmapped_observations,
        graph, trace, trace_page_map, current_plan, thumbnail_cache, default=[],
    ) if fill_unmapped else []
    persisted_trace_images = _run_patch_stage(
        "persist_screenshots", _persist_aligned_trace_screenshots,
        graph, trace, trace_page_map, default=0,
    )
    _run_patch_stage(
        "page_ownership", _apply_page_ownership,
        graph, page_ownership_patch, trace, function_refs, page_refs,
    )
    _run_patch_stage(
        "separate_new_functions", _separate_new_function_pages_from_planned_target,
        graph, functions_patch, pages_patch, function_refs, page_refs, current_plan,
    )
    _run_patch_stage(
        "function_dependencies", _apply_function_dependencies,
        graph, dependencies_patch, function_refs,
    )
    merge_replacements, merge_issues = _run_patch_stage(
        "function_merges", _apply_function_merges,
        graph, function_merges_patch, function_refs, default=({}, []),
    )
    restructure_replacements, restructure_issues = _run_patch_stage(
        "function_restructures", _apply_function_restructures,
        graph, function_restructures_patch, function_refs, page_refs, default=({}, []),
    )
    function_replacements = {**merge_replacements, **restructure_replacements}
    # 只凭拆分/合并声明创建的新功能，若对应的拆分/合并被跳过，会留下没有任何
    # 页面与场景的空功能；这类节点没有证据支撑，直接移除。
    backed_only_ids = {
        function_refs[ref] for ref in persisted_evidence_refs if ref in function_refs
    }
    for function_id in backed_only_ids - existing_function_ids:
        node = graph.get_node(function_id)
        if node is not None and not node.page_ids and not node.scenarios:
            graph.functions.pop(function_id, None)
            print(f"[FFG 更新] 移除没有页面承接的新功能（对应拆分/合并未生效）: {function_id}")
    graph.reconcile_page_membership()
    repaired_issues = _repair_ffg_invariants(graph, current_plan)
    # 修复后 trace_page_map 只保留仍存在的页面（合并/拆分可能移除页面属主，但不删页面）。
    trace_page_map = {
        raw_step: page_id
        for raw_step, page_id in trace_page_map.items()
        if graph.get_page(page_id) is not None
    }
    validation_issues = list(dict.fromkeys([*merge_issues, *restructure_issues, *repaired_issues]))
    if validation_issues:
        print("[FFG 更新] patch 中被跳过或修复的内容: " + "; ".join(validation_issues[:8]))

    return {
        "graph": graph,
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
        "auto_created_pages": auto_created_pages,
        "control_naming_issues": control_naming_issues,
    }


_ISSUE_STEP_PATTERN = re.compile(r"(?:source_step\s+|alignment\s+)(\d+)", re.IGNORECASE)


def _issue_step_already_mapped(
    issue: str,
    trace_page_map: dict[int, str],
    graph: FlowGraph,
) -> bool:
    """对齐问题所指步骤已在前面轮次映射到存在的页面时，视为已解决。"""
    match = _ISSUE_STEP_PATTERN.search(str(issue or ""))
    if not match:
        return False
    page_id = trace_page_map.get(int(match.group(1)), "")
    return bool(page_id) and graph.get_page(page_id) is not None


def _open_update_issues(
    graph: FlowGraph,
    trace: list[dict[str, Any]],
    trace_page_map: dict[int, str],
    alignment_issues: list[str],
    patch_result: dict[str, Any],
) -> list[str]:
    """本轮应用后仍值得再问一次模型的问题。

    只保留模型能修、且不修会损失建图质量的问题：尚未映射到页面的观测步骤
    （否则只能由程序按 Activity 粗粒度补建）、列举式控件名、被跳过的功能
    拆分/合并。已在前面轮次映射的步骤不再重复要求。
    """
    unresolved_steps: set[int] = set()
    for index, step in enumerate(trace, start=1):
        if _extract_page(step) is None or _is_ignorable_observation_step(step):
            continue
        raw_step = int(step.get("raw_step_index", index) or index)
        page_id = trace_page_map.get(raw_step, "")
        if page_id and graph.get_page(page_id) is not None:
            continue
        if _resolve_observed_page_id(graph, step, raw_step):
            continue
        unresolved_steps.add(raw_step)
    issues: list[str] = []
    for issue in alignment_issues:
        match = _ISSUE_STEP_PATTERN.search(str(issue or ""))
        # 指向某一步的对齐问题，只有该步仍未落到任何页面时才值得再问。
        if match is None or int(match.group(1)) in unresolved_steps:
            issues.append(issue)
    mentioned = {
        int(match.group(1))
        for issue in issues
        if (match := _ISSUE_STEP_PATTERN.search(str(issue or "")))
    }
    issues.extend(
        f"missing observation_alignment for source_step {raw_step}"
        for raw_step in sorted(unresolved_steps - mentioned)
    )
    issues.extend(patch_result.get("control_naming_issues", []))
    issues.extend(
        issue
        for issue in patch_result.get("validation_issues", [])
        if str(issue).startswith(("function_restructures", "function_merges"))
    )
    return list(dict.fromkeys(issues))


def _build_repair_message_content(
    base_text_blocks: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    compact_trace_screenshots: dict[int, str],
    open_issues: list[str],
    graph: FlowGraph,
    candidate_map: dict[int, list[dict[str, Any]]],
    step_index: int,
) -> list[dict[str, Any]]:
    """修补轮请求：沿用首轮文字上下文，只附问题步骤及相邻步骤的截图。

    前一轮的合法内容已写入图中，模型只需针对剩余问题补充 patch；已映射
    的步骤以文字说明其对应页面，避免重复上传全部截图。
    """
    focus_steps = _refinement_focus_steps(open_issues, trace)
    content: list[dict[str, Any]] = list(base_text_blocks)
    content.append({
        "type": "text",
        "text": (
            f"【FFG 修补第 {step_index} 步】上一版 patch 的合法部分已写入 FFG（不回滚），"
            "本次只需针对下列未解决问题补充 patch；已写入的内容无需重复。\n"
            f"未解决问题: {json.dumps(open_issues, ensure_ascii=False)}\n"
            "修复方法：\n"
            "- missing observation_alignment：为该 STEP 给出 reuse page_x，或 create new_page_x "
            "并在 pages 中声明（source_step 指向页面首次可见的 STEP；同一新页面的后续 STEP 可 reuse 该 new_page_x）。\n"
            "- ignore alignment 类错误：ignore 只能对齐外部/系统原因观察；正常执行结果应 reuse 或 create。\n"
            "- cannot move non-coverage page / has no valid coverage pages：拆分只能移动源功能的 "
            "coverage 页面，改用正确的 page_x 或放弃该拆分。\n"
            "- 引用不存在的编号：func_x/page_x 只能引用输入上下文中列出的编号。\n"
            "- 控件名把多个控件列举成一条：在 pages 中用同一 page_ref 重新声明该页 controls，"
            "逐项写出可分别点击的入口；同质资源实例只写一个类型级名称。"
        ),
    })
    uploaded = 0
    for index, step in enumerate(trace, start=1):
        raw_index = int(step.get("raw_step_index", index) or index)
        in_focus = focus_steps is None or raw_index in focus_steps
        if not in_focus:
            continue
        candidates = ", ".join(
            str(item.get("page_id", "")) for item in candidate_map.get(raw_index, [])[:3]
        )
        content.append({
            "type": "text",
            "text": (
                _format_trace_step(step, raw_index, index)
                + (f"\n[RETRIEVED_CANDIDATES] {candidates}" if candidates else "")
            ),
        })
        screenshot = compact_trace_screenshots.get(raw_index, "")
        if screenshot and uploaded < FFG_UPDATE_MAX_REFINEMENT_IMAGES:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{screenshot}"},
            })
            uploaded += 1
    return content


def create_ffg_update_node(llm, device=None):
    """Create the single-pass FFG update node.

    每轮只调用一次多模态模型：patch 尽力应用，模型遗漏或写错的部分由程序
    兜底与确定性修复处理，不做重试或整轮回滚。
    """
    thumbnail_cache = _ScreenshotThumbnailCache()

    def ffg_update_node(state: AgentState) -> dict:
        raw_trace = _combined_execution_trace(state)
        trace = _filter_trace(raw_trace)
        graph = state.get("ffg")
        if not isinstance(graph, FlowGraph):
            graph = FlowGraph()
        _record_intra_attempt(graph, state)

        # 页面控件后台审计：先合并已完成的结果（在 before 快照前完成，
        # 使本轮 stats 只反映 LLM patch），再投喂新一批待审页面。
        _merge_page_audit_results(graph)
        _submit_page_audit_batch(graph, llm)

        before = _snapshot_graph(graph)
        execution_result = state.get("execution_result", {}) or {}
        current_plan = state.get("current_plan", {}) or {}
        test_scenario = str(state.get("current_test_scenario", "")).strip()
        coverage_target_id = _resolve_plan_target_function_id(graph, current_plan)
        coverage_before = (
            int(graph.function_coverage_details(coverage_target_id)["coverage_percent"])
            if coverage_target_id else 0
        )
        review_attempts = dict(state.get("granularity_review_attempts", {}) or {})
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
        if not trace:
            print("[FFG 更新] 没有可处理的轨迹步骤，改为记录本轮执行结果并更新功能状态。")
            return _apply_non_recordable_execution_outcome(graph, state, execution_result)

        thumbnail_cache.preload_graph(graph)
        candidate_map, step_similarity_cache = _build_observation_candidate_map(
            graph,
            current_plan,
            trace,
            thumbnail_cache=thumbnail_cache,
        )
        ffg_summary = _build_ffg_summary(
            graph,
            current_plan,
            trace,
            candidate_map=candidate_map,
            granularity_review_function_id=granularity_review_function_id,
        )
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
        base_text_blocks = list(message_content)
        # Every trace screenshot is first-hand structural evidence and may be a
        # new logical Page even when its Activity matches another step. Never
        # hide it behind an image budget.
        compact_trace_screenshots: dict[int, str] = {}
        for index, step in enumerate(trace, start=1):
            raw_step = int(step.get("raw_step_index", index) or index)
            screenshot = _page_screenshot(_extract_page(step)) or ""
            compact_trace_screenshots[raw_step] = (
                _compact_screenshot_for_llm(screenshot) if screenshot else ""
            )
        trace_screenshot_payloads = {
            screenshot for screenshot in compact_trace_screenshots.values() if screenshot
        }
        reference_candidates = _reference_pages_for_update(
            graph,
            current_plan,
            trace,
            candidate_map=candidate_map,
            thumbnail_cache=thumbnail_cache,
            step_similarity_cache=step_similarity_cache,
        )
        image_budget = FFG_UPDATE_MAX_TOTAL_IMAGES
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
                screenshot = (
                    _compact_screenshot_for_llm(reference_screenshot)
                    if reference_screenshot
                    else ""
                )
                if screenshot and screenshot not in trace_screenshot_payloads:
                    message_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{screenshot}",
                        },
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
            f"omitted_reference_images={len(reference_candidates) - len(image_reference_pages)}"
        )

        remaining_run_seconds = _remaining_run_seconds(state)
        available_llm_seconds = (
            None
            if remaining_run_seconds is None
            else max(0.0, remaining_run_seconds - FFG_UPDATE_TIME_SAFETY_SECONDS)
        )
        if available_llm_seconds is not None and available_llm_seconds <= 0:
            print(
                "[FFG 更新] 总运行时限已到，跳过多模态调用；"
                "原始轨迹已落盘，本轮按执行结果收尾。"
            )
            return _apply_non_recordable_execution_outcome(graph, state, execution_result)
        deadline_monotonic = (
            None if available_llm_seconds is None else time.monotonic() + available_llm_seconds
        )

        # 多轮修补：每轮把合法内容增量写入（不回滚）；仍有未解决问题时，再发一次
        # 只含相关步骤截图的修补请求。最后一次调用之后才由程序兜底落图，保证
        # 模型先有机会给出语义正确的页面名、控件与归属。
        carried_function_refs: dict[str, str] = {}
        carried_page_refs: dict[str, str] = {}
        carried_trace_page_map: dict[int, str] = {}
        model_patch_unavailable = False
        merged_bindings: list[dict[str, Any]] = []
        all_alignment_issues: list[str] = []
        all_validation_issues: list[str] = []
        all_naming_issues: list[str] = []
        patch_counts = {key: 0 for key in (
            "functions", "pages", "page_ownership", "function_restructures",
            "function_merges", "function_dependencies",
        )}
        function_replacements: dict[str, list[str]] = {}
        persisted_trace_images = 0
        previous_open_issues: list[str] = []
        open_issues: list[str] = []
        steps_run = 0
        for step_index in range(1, FFG_UPDATE_MAX_STEPS + 1):
            is_repair = step_index > 1
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                print("[FFG 更新] 运行时限将到，停止后续修补。")
                break
            remaining_llm_seconds = (
                None if deadline_monotonic is None else deadline_monotonic - time.monotonic()
            )
            max_attempts = FFG_UPDATE_LLM_RETRY_ATTEMPTS
            if remaining_llm_seconds is not None:
                max_attempts = max(
                    1, min(max_attempts, int(remaining_llm_seconds // FFG_UPDATE_LLM_TIMEOUT_SECONDS)),
                )
            content = (
                message_content
                if not is_repair
                else _build_repair_message_content(
                    base_text_blocks,
                    trace,
                    compact_trace_screenshots,
                    open_issues,
                    graph,
                    candidate_map,
                    step_index,
                )
            )
            if is_repair:
                print(f"[FFG 更新] 修补轮 {step_index}/{FFG_UPDATE_MAX_STEPS}: " + "; ".join(open_issues[:5]))
            steps_run = step_index
            patch = invoke_llm_json_with_retry(
                llm,
                [
                    SystemMessage(content=FFG_UPDATE_SYSTEM_PROMPT),
                    HumanMessage(content=content),
                ],
                log_prefix="[FFG 更新]",
                json_block_pattern=JSON_BLOCK_PATTERN,
                max_attempts=max_attempts,
                deadline_monotonic=deadline_monotonic,
            )
            if llm_json_result_failed(patch):
                if not is_repair:
                    model_patch_unavailable = True
                    print(
                        "[FFG 更新] 底层多模态调用已耗尽重试；本轮改由程序按轨迹落图"
                        "（已有页面按截图复用，新 Activity 补建页面），执行事实照常记录。"
                    )
                else:
                    print("[FFG 更新] 修补轮调用失败，保留已写入内容并由程序兜底。")
                patch = {}
            elif not patch:
                print("[FFG 更新] LLM 未返回有效 patch，本轮不增加结构；仍记录执行结果。")
                patch = {}

            _normalize_observation_alignment(patch, graph, trace, candidate_map)
            auto_fixed = _ensure_ignorable_alignments(patch, trace)
            if auto_fixed:
                print(f"[FFG 更新] 程序自动补齐外部/恢复步骤 ignore 对齐: {auto_fixed} 项")
            # 修补轮只需重申未解决的步骤；已在前面轮次映射的步骤视为已对齐。
            alignment_issues = [
                issue
                for issue in validate_observation_alignment(graph, patch, trace, candidate_map)
                if not _issue_step_already_mapped(issue, carried_trace_page_map, graph)
            ]
            # 首轮调用失败或修补轮无产出时，再问也无益，直接收尾。
            final_pass = (
                step_index >= FFG_UPDATE_MAX_STEPS
                or model_patch_unavailable
                or (is_repair and not patch)
            )
            patch_result = apply_ffg_patch(
                graph,
                patch,
                trace,
                current_plan=current_plan,
                thumbnail_cache=thumbnail_cache,
                fill_unmapped=False,
                carried_function_refs=carried_function_refs,
                carried_page_refs=carried_page_refs,
                carried_trace_page_map=carried_trace_page_map,
            )
            carried_function_refs = patch_result["function_refs"]
            carried_page_refs = patch_result["page_refs"]
            carried_trace_page_map = patch_result["trace_page_map"]
            persisted_trace_images += int(patch_result.get("persisted_trace_images") or 0)
            function_replacements.update(patch_result.get("function_replacements", {}))
            merged_bindings.extend(_list_patch(patch.get("operation_bindings")))
            for key in patch_counts:
                patch_counts[key] += len(_list_patch(patch.get(key)))
            all_alignment_issues.extend(alignment_issues)
            all_validation_issues.extend(patch_result.get("validation_issues", []))
            all_naming_issues.extend(patch_result.get("control_naming_issues", []))

            open_issues = _open_update_issues(
                graph, trace, carried_trace_page_map, alignment_issues, patch_result,
            )
            if not open_issues or final_pass:
                break
            if previous_open_issues and open_issues == previous_open_issues:
                print("[FFG 更新] 未解决问题与上一轮相同，模型无法修复，停止修补: " + "; ".join(open_issues[:3]))
                break
            previous_open_issues = open_issues

        # 模型调用用尽后：程序兜底未对齐观测，并对最终图做一次确定性修复。
        auto_created_pages = _fill_unmapped_observations(
            graph, trace, carried_trace_page_map, current_plan, thumbnail_cache,
        )
        if auto_created_pages:
            graph.reconcile_page_membership()
            persisted_trace_images += _persist_aligned_trace_screenshots(
                graph, trace, carried_trace_page_map,
            )
        all_validation_issues.extend(_repair_ffg_invariants(graph, current_plan))
        patch_result = {
            "trace_page_map": {
                raw_step: page_id
                for raw_step, page_id in carried_trace_page_map.items()
                if graph.get_page(page_id) is not None
            },
            "persisted_trace_images": persisted_trace_images,
            "function_replacements": function_replacements,
            "validation_issues": list(dict.fromkeys(all_validation_issues)),
            "control_naming_issues": list(dict.fromkeys(all_naming_issues)),
        }
        patch = {"operation_bindings": merged_bindings}
        alignment_issues = list(dict.fromkeys(all_alignment_issues))
        if patch_result.get("persisted_trace_images"):
            print(
                "[FFG 更新] 已将对齐到已有 Page 的轨迹截图持久化为状态证据: "
                f"{patch_result['persisted_trace_images']} 张"
            )
        for source_function_id, replacement_ids in function_replacements.items():
            review_attempts.pop(source_function_id, None)
            for replacement_id in replacement_ids:
                review_attempts.pop(replacement_id, None)
        _record_screenshot_unavailable_page(
            graph,
            state,
            execution_result,
            patch_result.get("trace_page_map", {}),
        )
        _apply_trace_operations_and_transitions(
            graph,
            trace,
            patch_result.get("trace_page_map", {}),
        )
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
            f"model_calls={steps_run}, "
            f"functions={patch_counts['functions']}, pages={patch_counts['pages']}, "
            f"page_ownership={patch_counts['page_ownership']}, "
            f"function_restructures={patch_counts['function_restructures']}, "
            f"function_merges={patch_counts['function_merges']}, "
            f"dependencies={patch_counts['function_dependencies']}"
        )

        mapped_current_page = _map_current_page_after_update(
            graph,
            state,
            patch_result.get("trace_page_map", {}),
        )

        missing_activities = _unrepresented_trace_activities(
            graph,
            trace,
            patch_result.get("trace_page_map", {}),
        )
        diagnostics = list(dict.fromkeys([
            *open_issues,
            *patch_result.get("validation_issues", []),
            *(
                ["执行轨迹中仍有未被任何 Page 表示的 Activity（无可归属功能）: "
                 + ", ".join(missing_activities)]
                if missing_activities else []
            ),
        ]))
        update_reason = (
            "model patch unavailable; trace applied by program"
            if model_patch_unavailable
            else "; ".join(diagnostics[:8])
        )

        after = _snapshot_graph(graph)
        print(
            "[FFG 更新] 增量: "
            f"功能={after['functions'] - before['functions']}, "
            f"页面={after['pages'] - before['pages']}, "
            f"操作={after['operations'] - before['operations']}, "
            f"页面跳转={after['page_transitions'] - before['page_transitions']}, "
            f"功能依赖={after['function_dependencies'] - before['function_dependencies']}"
        )

        coverage_after = (
            int(graph.function_coverage_details(coverage_target_id)["coverage_percent"])
            if coverage_target_id and graph.get_node(coverage_target_id) is not None else 0
        )
        # 有进展 = 任一结构事实增长：功能/页面/操作/跳转、新绑定或目标覆盖率上升。
        yielded_structure = (
            any(after[key] > before[key] for key in ("functions", "pages", "operations", "page_transitions"))
            or bound_operations > 0
            or coverage_after > coverage_before
        )
        function_failure_counts = _record_execution_facts(
            graph, state, execution_result, yielded_structure=yielded_structure,
        )

        return {
            "ffg": graph,
            "current_page": mapped_current_page,
            "function_execution_failure_counts": function_failure_counts,
            "granularity_review_attempts": review_attempts,
            "ffg_granularity_review_function_id": granularity_review_function_id,
            "ffg_granularity_review_metrics": granularity_review_metrics,
            "ffg_update_decision": "finish",
            "ffg_update_reason": update_reason,
            "ffg_update_validation_issues": diagnostics,
        }

    return ffg_update_node
