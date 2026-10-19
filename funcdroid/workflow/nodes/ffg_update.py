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
    FunctionalFlowGraph,
    FunctionDependency,
    FunctionalityNode,
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
# Count no structural progress only after half coverage; early exploration needs initial observations.
ZERO_YIELD_MIN_COVERAGE = 50
# Defer a functionality after consecutive navigation failures instead of endlessly rescheduling it.
NAVIGATION_FAILURE_DEFER_LIMIT = max(1, int(os.getenv("NAVIGATION_FAILURE_DEFER_LIMIT", "3")))
# Navigation failures, including crashes and app exits, do not test the target functionality itself.

NAVIGATION_FAILURE_STATUSES = {
    "blocked", "stalled", "incomplete", "failed_goal", "error", "left_app", "app_crashed",
}
TEST_DATA_PATTERN = re.compile(r"FuncDroid_[0-9A-Za-z_]+")
# Keep this marker aligned with scenario_exec.EXPLORATION_BOUNDARY_BLOCKED_END.
EXPLORATION_BOUNDARY_ERROR_MARKER = "(outside target app core functionality); returned automatically and ended the scenario"
# Reuse pages with the same Activity and sufficient control-family overlap to avoid inflated coverage denominators.

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
# Cap stored reference images only; unmatched current trace observations remain complete first-hand evidence.

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
# Bound initial plus repair calls. Repair only unresolved alignment/control/restructure issues and retain prior valid increments.


FFG_UPDATE_MAX_STEPS = max(1, int(os.getenv("FFG_UPDATE_MAX_STEPS", "3")))
FFG_UPDATE_MAX_REFINEMENT_IMAGES = max(
    1,
    int(os.getenv("FFG_UPDATE_MAX_REFINEMENT_IMAGES", "3")),
)
# Bound images per page-audit call, preferring unseen states; stored history has its own cap.


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
# Bound cross-round thumbnails so unchanged stored screenshots need not be decoded repeatedly.
FFG_THUMBNAIL_CACHE_SIZE = max(64, int(os.getenv("FFG_THUMBNAIL_CACHE_SIZE", "4096")))
# Minimum screenshot similarity required to match an existing page.
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
BLACK_SCREEN_PAGE_NAME = "Black or unavailable page"
# Accept the historical serialized marker without rewriting recorded page names.
_BLACK_SCREEN_PAGE_NAMES = frozenset({
    BLACK_SCREEN_PAGE_NAME,
    "\u9ed1\u5c4f\u6216\u9875\u9762\u4e0d\u53ef\u7528",
})


def is_black_screen_page_name(name: str) -> bool:
    """Recognize current and historical unavailable-screen page markers."""
    return name in _BLACK_SCREEN_PAGE_NAMES


BLACK_SCREEN_PAGE_DESCRIPTION = (
    "No valid interface could be captured after execution: a black, blank or repeatedly unavailable endpoint. "
    "Avoid its recorded incoming operations and scenarios when planning."
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
    graph: FunctionalFlowGraph,
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
        # External and automatic-recovery observations do not create target-app Pages; ignore alignment is authoritative.

        if _is_ignorable_observation_step(step):
            continue
        activity = str(_page_activity(_extract_page(step)) or "").strip()
        if activity and activity not in represented and activity not in missing:
            missing.append(activity)
    return missing


class _ScreenshotThumbnailCache:
    """Bounded cross-round grayscale-thumbnail LRU keyed by content digests. Candidate retrieval, references and observation matching reuse unchanged screenshots without retaining large source strings."""


    def __init__(self, size: int = 64, max_entries: int = FFG_THUMBNAIL_CACHE_SIZE):
        self._size = size
        self._max_entries = max(1, int(max_entries))
        self._thumbnails: OrderedDict[str, np.ndarray | None] = OrderedDict()

    def clear(self) -> None:
        self._thumbnails.clear()

    def preload_graph(self, graph: FunctionalFlowGraph) -> None:
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


# Shared process cache for callers without an update-node instance.
_SHARED_THUMBNAIL_CACHE = _ScreenshotThumbnailCache()


def _best_page_similarity(cache, screenshot: str | None, page: Page) -> float | None:
    """Return the highest similarity to any screenshot stored for a page; cache supplies similarity()."""
    scores = [
        value
        for stored in [page.representative_screenshot, *page.evidence_screenshots]
        if (value := cache.similarity(screenshot, stored)) is not None
    ]
    return max(scores, default=None)


def _compact_screenshot_for_llm(screenshot: str) -> str:
    """Use the process-wide image-compaction cache shared by FFG updating, defect review and page auditing."""


    return compact_screenshot(
        screenshot,
        max_side=FFG_UPDATE_IMAGE_MAX_SIDE,
        quality=FFG_UPDATE_IMAGE_QUALITY,
        source_label="FFG update",
    )


def _natural_language_controls(page: Page | dict[str, Any] | None) -> list[str]:
    """Read natural-language control descriptions supplied by the model or persisted page."""
    if isinstance(page, Page) and page.controls:
        existing = list(page.controls)
    elif isinstance(page, dict) and isinstance(page.get("controls"), list):
        existing = [str(item).strip() for item in page["controls"] if str(item).strip()]
    else:
        existing = []
    return existing


# ---------------------------------------------------------------------------
# Background page auditing: merge completed results and submit new work.
# ---------------------------------------------------------------------------
PAGE_AUDIT_ENABLED = os.getenv("PAGE_AUDIT_ENABLED", "1").strip().lower() in {
    "1", "true", "yes", "on",
}
# Global navigation controls are excluded, consistent with press_back not being a business operation.

# Preserve legacy UI-language families alongside English operation labels.
PAGE_AUDIT_EXCLUDED_FAMILIES = {"back", "close", "return", "\u8fd4\u56de", "\u5173\u95ed", "\u8fd4\u56de\u952e"}
# Bound audit removals to avoid clearing an inventory; transitions remain intact.
PAGE_AUDIT_MAX_REMOVALS = 5
PAGE_AUDIT_MAX_CONTROL_MERGES = 3


def _page_audit_fingerprint(page: Page) -> str:
    """Fingerprint screenshot digests and operations, excluding operation_bindings: binding changes add no audit value and must not trigger re-auditing every existing page."""


    parts = [screenshot_digest(page.representative_screenshot or "")]
    parts.extend(screenshot_digest(item or "") for item in page.evidence_screenshots)
    parts.append("\x00operations")
    parts.extend(str(item or "") for item in page.operations)
    digest = hashlib.sha1("|".join(parts).encode("utf-8", errors="replace")).hexdigest()
    return digest[:16]


def _page_audit_semantically_duplicate(candidate: str, existing: list[str]) -> bool:
    """Reject controls in an existing control family, using the same normalization as FFG coverage accounting."""


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


def _merge_page_audit_results(graph: FunctionalFlowGraph) -> int:
    """Validate and merge background audit results on the main thread. Reject semantic duplicates and global navigation controls, then report actual additions for saturation detection. Do not require positional words: legitimate album/camera controls otherwise lose bindings and coverage."""


    if not PAGE_AUDIT_ENABLED:
        return 0
    total_changes = 0
    results = page_audit_worker.collect_results()
    for result in results:
        page_id = str(result.get("page_id", "") or "")
        page = graph.get_page(page_id) if page_id else None
        if page is None:
            # Removed pages report zero additions so stale audit state cannot persist.
            page_audit_worker.record_result(page_id, 0)
            continue
        # Audit decisions use a submission-time snapshot; when observations changed, keep additions but skip stale removals and merges.

        result_fingerprint = str(result.get("fingerprint", "") or "")
        stale = bool(result_fingerprint) and result_fingerprint != _page_audit_fingerprint(page)
        if stale:
            print(
                f"[Page audit] Page changed during review; retain additions, skip removals and merges: page={page_id}"
            )
        existing = list(page.controls)
        added: list[str] = []
        for control in result.get("missing_controls") or []:
            text = str(control or "").strip()
            if not text:
                continue
            if control_family_key(text) in PAGE_AUDIT_EXCLUDED_FAMILIES:
                print(f"[Page audit] Reject global navigation control handled by back: {text!r}")
                continue
            if _control_name_enumerates_items(text):
                print(f"[Page audit] Reject a name enumerating multiple controls: {text!r}")
                continue
            if _page_audit_semantically_duplicate(text, existing + added):
                print(f"[Page audit] Reject semantic duplicate control: {text!r}")
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
                f"[Page audit] Add controls: page={page_id} "
                f"({page.name or 'unnamed'}), added={len(added)}: "
                + "; ".join(added[:5])
            )
        if removed:
            print(
                f"[Page audit] Remove invalid operations: page={page_id} "
                f"({page.name or 'unnamed'}), removed={len(removed)}: "
                + "; ".join(removed[:5])
            )
        if merged_controls:
            print(
                f"[Page audit] Merge equivalent resource controls: page={page_id} "
                f"({page.name or 'unnamed'}), groups={len(merged_controls)}: "
                + "; ".join(merged_controls[:3])
            )
    return total_changes


# Opposite switch states share positions but express different operations; never merge them as aliases.

OPPOSITE_STATE_WORDS = (
    ("enable", "disable"),
    ("show", "hide"),
    ("expand", "collapse"),
    ("play", "pause"),
    ("start", "stop"),
    ("lock", "unlock"),
    ("mute", "unmute"),
    ("favorite", "unfavorite"),
    ("follow", "unfollow"),
    ("\u5f00\u542f", "\u5173\u95ed"),
    ("\u6253\u5f00", "\u5173\u95ed"),
    ("\u542f\u7528", "\u7981\u7528"),
    ("\u663e\u793a", "\u9690\u85cf"),
    ("\u5c55\u5f00", "\u6536\u8d77"),
    ("\u64ad\u653e", "\u6682\u505c"),
    ("\u5f00\u59cb", "\u505c\u6b62"),
    ("\u9501\u5b9a", "\u89e3\u9501"),
    ("\u9759\u97f3", "\u53d6\u6d88\u9759\u97f3"),
    ("\u6536\u85cf", "\u53d6\u6d88\u6536\u85cf"),
    ("\u5173\u6ce8", "\u53d6\u6d88\u5173\u6ce8"),
)


def _sources_name_opposite_states(sources: list[str]) -> bool:
    for first, second in OPPOSITE_STATE_WORDS:
        # Strip the opposite phrase first when it contains the positive term, such as unfavorite versus favorite.
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
    """Apply exact high-confidence merges without losing bindings. Resource-instance merges need three sources and a shared business prefix. Physical-control aliases need two sources and must reuse an existing name."""


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
            print(f"[Page audit] Reject merge with missing sources: sources={sources!r}, target={target!r}")
            continue
        if synonym_merge and _sources_name_opposite_states(sources):
            print(f"[Page audit] Reject merge of opposite switch states: sources={sources!r}")
            continue
        if not synonym_merge:
            normalized_sources = [TEXT_NORMALIZE_PATTERN.sub("", source) for source in sources]
            common_prefix = os.path.commonprefix(normalized_sources)
            normalized_target = TEXT_NORMALIZE_PATTERN.sub("", target)
            if len(common_prefix) < 2 or common_prefix not in normalized_target:
                print(
                    "[Page audit] Reject control merge without a shared business prefix: "
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
    """Remove only exact listed operations, bounded by PAGE_AUDIT_MAX_REMOVALS; preserve transition records. Never remove an operation with a declared binding, which represents validated control coverage. Invalid bindings must be removed explicitly first."""


    bound_operations = page.bound_operation_texts()
    removed: list[str] = []
    seen: set[str] = set()
    for item in invalid_operations:
        text = str(item or "").strip()
        for suffix in (" [UNBOUND CONTROL]", "（\u672a\u7ed1\u5b9a\u63a7\u4ef6）"):
            if text.endswith(suffix):
                text = text[:-len(suffix)].rstrip()
                break
        if not text or text in seen:
            continue
        seen.add(text)
        if len(removed) >= PAGE_AUDIT_MAX_REMOVALS:
            break
        if text in bound_operations:
            print(
                f"[Page audit] Preserve bound business operation: {text!r}"
            )
            continue
        if page.remove_operation(text):
            removed.append(text)
        else:
            print(f"[Page audit] Reject removal that does not exactly match an operation: {text!r}")
    return removed


def _select_page_audit_screenshots(page_id: str, screenshots: list[str]) -> list[str]:
    """Within the per-audit cap, prefer unaudited screenshots, then the representative and latest states."""
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
    # Preserve screenshot order for chronological interpretation.
    order = {item: index for index, item in enumerate(screenshots)}
    return sorted(selected, key=lambda item: order[item])


def _submit_page_audit_batch(
    graph: FunctionalFlowGraph,
    llm=None,
    *,
    exclude_fingerprints: set[tuple[str, str]] | None = None,
) -> int:
    """Submit pages of completed functionalities first, then pages with fewer controls. The worker filters unchanged fingerprints and saturated pages."""


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
                    f"Page name: {page.name or 'unnamed'}\n"
                    f"Page description: {page.description or 'none'}\n"
                    "Recorded controls:\n"
                    + ("\n".join(f"- {c}" for c in page.controls) if page.controls else "(empty)")
                    + "\nRecorded operations (review [UNBOUND CONTROL] items first):\n"
                    + (
                        "\n".join(
                            f"- {o}" if o in bound_operations else f"- {o} [UNBOUND CONTROL]"
                            for o in page.operations
                        )
                        if page.operations
                        else "(empty)"
                    )
                    + "\nAudit the images: add missing interactive controls and remove unanchored or redundant operations."
                ),
            },
        ]
        # A Page includes stable, keyboard, dialog and expanded-panel states. Prioritize unseen states within each call; omitted images remain eligible for later audits.


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
                    f"[PAGE_STATE_IMAGES_OMITTED] Per-call limit omitted "
                    f"{omitted} images; unseen states were prioritized and remaining images await another audit."
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
                f"[Page audit] Submitted inventory review: page={page.id} "
                f"({page.name or 'unnamed'}), controls={len(page.controls)}, "
                f"operations={len(page.operations)}, images={used}"
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
    """Keep every page observation and let FFG updating decide its graph representation."""
    filtered: list[dict[str, Any]] = []
    for raw_index, step in enumerate(trace, start=1):
        if not isinstance(step, dict):
            continue
        copied = dict(step)
        copied["raw_step_index"] = raw_index
        filtered.append(copied)
    return filtered


def _combined_execution_trace(state: AgentState) -> list[dict[str, Any]]:
    """Combine navigation and test evidence in actual execution order."""
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


def _black_screen_page(graph: FunctionalFlowGraph) -> Page | None:
    return next(
        (page for page in graph.active_pages() if is_black_screen_page_name(page.name)),
        None,
    )


def _resolve_observed_page_id(
    graph: FunctionalFlowGraph,
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
        if is_black_screen_page_name(candidate.name):
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
        return "Black or unavailable page after scenario execution"
    action = str(step.get("action_description", "") or "").strip()
    if not action:
        action_name = str(step.get("action_name", "") or "Unknown action").strip()
        if step.get("x") is not None and step.get("y") is not None:
            action = f"{action_name}({step.get('x')},{step.get('y')})"
        elif str(step.get("content", "") or "").strip():
            action = f"{action_name}({str(step.get('content')).strip()})"
        elif str(step.get("direction", "") or "").strip():
            action = f"{action_name}({str(step.get('direction')).strip()})"
        else:
            action = action_name
    return f"{action}; result: black or unavailable page (screenshot health failed)"


def _record_screenshot_unavailable_page(
    graph: FunctionalFlowGraph,
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
        print("[FFG update] No functionality can own the black-screen page; retain execution outcome only.")
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
        print(f"[FFG update] Created shared unavailable-screen page: {page_id}")
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
            "[FFG update] Recorded black-screen path: "
            f"{source_page_id} --[{operation}]--> {black_page_id}"
        )
    else:
        print("[FFG update] Recorded black-screen page; no final valid source page could be resolved.")
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
    graph: FunctionalFlowGraph,
    state: AgentState,
    trace_page_map: dict[int, str] | None = None,
) -> Page | None:
    """Map the final device interface to a stable Page in the updated FFG."""
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
    graph: FunctionalFlowGraph,
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
    """Return operation text and whether to record it, using the same derivation for trace persistence and source_step operation bindings."""


    action_name = str(step.get("action_name", "")).strip()
    operation = str(step.get("action_description", "") or "").strip() or action_name
    # System back is global navigation, not a page business operation; preserve its transition edge only.

    record_operation = action_name != "press_back"
    if action_name == "press_back":
        operation = "Back"
    if step.get("action_failed"):
        error = str(step.get("action_error", "") or "").strip() or "Action execution failed"
        operation = f"Failed attempt: {operation}; error: {error}"
    elif step.get("foreground_recovery") or step.get("is_interruption"):
        operation = f"Automatic recovery: {operation}"
    return operation, record_operation


def _apply_trace_operations_and_transitions(
    graph: FunctionalFlowGraph,
    trace: list[dict[str, Any]],
    trace_page_map: dict[int, str] | None = None,
) -> list[str]:
    """Deterministically retain executed operations and directly observed page results in the available FFG structure."""
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
                    or "Restore target app"
                )
                operation = f"Automatic recovery: {operation}"
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
            # The dedicated exceptional edge records the action and unavailable-screen result.
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
        if is_black_screen_page_name(dst_page.name):
            # Maintain explicit failed outcomes on black-screen incoming edges.
            continue
        graph.add_page_transition(PageTransition(
            src_page_id=src_page_id,
            dst_page_id=dst_page_id,
            operation=operation,
        ))
        touch_page(dst_page_id)
    return touched_function_ids


def _apply_operation_bindings(
    graph: FunctionalFlowGraph,
    patch: dict[str, Any],
    trace: list[dict[str, Any]],
    trace_page_map: dict[int, str],
) -> int:
    """Apply explicit operation-to-control bindings after recording trace operations. Derive source_step text from the actual trace; explicit operation/control names must match existing page inventories. Skip invalid bindings without discarding the graph increment. Bindings describe observed facts and remain applicable whenever their pages and controls exist."""


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
                    # Non-business actions such as press_back have no inventory operation to bind.
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
                "[FFG update] Reject binding without a resolvable step/page: "
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
            # A previously recorded identical binding is normal, not a validation error.
            print(
                f"[FFG update] Skip already recorded operation binding: "
                f"page={page.id}, control={control!r}"
            )
        else:
            print(
                "[FFG update] Reject binding whose operation/control is absent from the page inventory: "
                f"page={page.id}, operation={operation_text!r}, control={control!r}"
            )
    return applied


def _auto_bind_unambiguous_trace_operations(
    graph: FunctionalFlowGraph,
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
                "[FFG update] Added an unambiguous operation binding: "
                f"page={page.id}, operation={stored_operation!r}, control={candidates[0]!r}"
            )
    return applied


def _clip_context_text(value: object, limit: int = FFG_CONTEXT_FIELD_MAX_CHARS) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    marker = "...[truncated]"
    return text[: max(1, limit - len(marker))] + marker


def _focused_update_function_ids(
    graph: FunctionalFlowGraph,
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
    graph: FunctionalFlowGraph,
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


def _select_functionality_independence_review(
    graph: FunctionalFlowGraph,
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
    graph: FunctionalFlowGraph,
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
    # Text similarity is expensive and weighted by 0.15. Skip candidates whose best possible score cannot enter the top limit; rankings remain exact.


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
    graph: FunctionalFlowGraph,
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
    graph: FunctionalFlowGraph,
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
    graph: FunctionalFlowGraph,
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
    # Trim noisy low-value candidates/index/dependencies before functionality boundary evidence needed for splits and merges.


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
    # Count omitted fields after trimming to match the actual model context.
    facts["context_scope"]["omitted_functions"] = max(
        0, len(graph.functions) - len(facts["functions"])
    )
    facts["context_scope"]["omitted_pages"] = max(
        0, len(graph.active_pages()) - len(facts["pages"])
    )
    serialized = json.dumps(facts, ensure_ascii=False, separators=(",", ":"))
    return "[Focused FFG fact snapshot]\n" + serialized


def _function_context(graph: FunctionalFlowGraph, function_id: object) -> dict[str, Any]:
    normalized_id = str(function_id or "").strip()
    node = graph.get_node(normalized_id)
    return {
        "id": graph.resolve_function_id(normalized_id) if node is not None else normalized_id,
        "name": (node.name or node.description) if node is not None else "",
        "exists_in_ffg": node is not None,
    }


def _build_plan_context(graph: FunctionalFlowGraph, current_plan: dict[str, Any]) -> str:
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
        "[Authoritative planning scope]\n"
        + json.dumps(facts, ensure_ascii=False, indent=2)
        + "\n- The func_x references above come from validated planning and take precedence over scenario prose."
        + "\n- scenario_type fixes this round's target, not the permanent owner of newly observed pages."
        + "\n- intra_test may discover an independent functionality. Intermediate pages may belong to the target; stable entries/results may be shared."
        + "\n- cross_test connects source and destination only with direct evidence."
        + "\n- Do not assign evidence to another func_x merely because its name is similar."
    )


def _build_function_boundary_context(
    graph: FunctionalFlowGraph,
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
        "[Functionality boundary review]\n"
        + json.dumps(facts, ensure_ascii=False, indent=2)
        + "\n- Classify every new Page as an intermediate state, shared page, or independent goal with its own complete flow."
        + "\n- With mandatory_boundary_review=true, do not silently assign every new Page to target_function."
        + "\n- When granularity_review_function_id is present, review all persisted pages of that functionality. Split independently verifiable subdomains sharing an entry using function_restructures, preferably a partial retire_source=false split of the clearest subdomain. Do not examine only new pages."
        + "\n- An Activity change is a review signal, not sufficient proof of a new functionality."
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
    graph: FunctionalFlowGraph,
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


def _snapshot_graph(graph: FunctionalFlowGraph) -> dict[str, int]:
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


# One enumerated name can hide independently clickable entries from coverage. Reject it for model/audit correction; illustrative examples and homogeneous instance types are not enumerations.


CONTROL_ENUMERATION_BRACKET_PATTERN = re.compile(r"[（(]([^（）()]*)[)）]")
CONTROL_EXAMPLE_PREFIX_PATTERN = re.compile(r"^\s*(?:\u4ee3\u8868|\u5982|\u4f8b\u5982|\u6bd4\u5982|e\.g\.|eg|such as)", re.IGNORECASE)


def _control_name_enumerates_items(name: str) -> bool:
    text = str(name or "").strip()
    if not text:
        return False
    for inner in CONTROL_ENUMERATION_BRACKET_PATTERN.findall(text):
        if CONTROL_EXAMPLE_PREFIX_PATTERN.match(inner) or inner.rstrip().endswith("\u7b49"):
            continue
        if re.search(r"[、/／;；]", inner) or len(re.findall(r"[,，]", inner)) >= 2:
            return True
    outside = CONTROL_ENUMERATION_BRACKET_PATTERN.sub("", text)
    return outside.count("、") >= 2


def _strip_enumerated_controls(patch: dict[str, Any]) -> list[str]:
    """Remove multi-entry control names and report issues requesting proper declarations."""
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
                "Declare independently clickable entries into different functionalities as separate controls; "
                "represent equivalent resource instances with one type-level name, without enumerating values"
            )
    return issues


def _normalize_observation_alignment(
    patch: dict[str, Any],
    graph: FunctionalFlowGraph,
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
            # Only new_page_x means create; updating an existing page_x while reusing it is valid.

            item["page_decision"] = "create"
            continue
        if decision == "create":
            # Normalize create for an existing page identifier without a new page patch to reuse.


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
            # Require independent retrieval evidence before converting to reuse, or a new page could be lost.

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


CONTROL_ENUMERATION_ISSUE_MARKER = "Control name enumerates multiple controls; entry not applied"


def _is_ignorable_observation_step(step: dict[str, Any]) -> bool:
    """Recognize unhealthy screenshots, automatic recovery/interruption and external observations. Any applicable flag is sufficient; these do not model target-app pages and require ignore alignment."""


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
    """Fill or correct ignore alignment for deterministically ignorable steps, even if the model omitted a segment. Return the number of corrected entries."""


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
    graph: FunctionalFlowGraph,
    patch: dict[str, Any],
    trace: list[dict[str, Any]],
    candidate_map: dict[int, list[dict[str, Any]]],
) -> list[str]:
    """Diagnose alignment quality for logs without blocking updates; pages patches and unmapped-observation fallback determine ownership."""


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
                # Later observations of a newly declared page may reuse its new_page_x reference.
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
            # source_step identifies the first observation; later steps may create/reuse the same declared temporary page.

            if not page_ref.startswith("new_page_"):
                issues.append(f"create alignment must use new_page_x at source_step {raw_step}")
            elif page_ref not in page_patch_refs:
                issues.append(f"create alignment lacks matching pages patch at source_step {raw_step}")
        elif not _is_ignorable_observation_step(step):
            issues.append(f"ignore alignment lacks an ignorable observation at source_step {raw_step}")
    return list(dict.fromkeys(issues))


def _resolve_function_ref(ref: str, temp_refs: dict[str, str], graph: FunctionalFlowGraph) -> str:
    ref = _normalize_ref(ref)
    if not ref:
        return ""
    resolved = graph.resolve_function_id(ref)
    if resolved and graph.get_node(resolved) is not None:
        return resolved
    return temp_refs.get(ref, "")


def _resolve_page_ref(ref: str, temp_refs: dict[str, str], graph: FunctionalFlowGraph) -> str:
    ref = _normalize_ref(ref)
    if not ref:
        return ""
    resolved = graph.resolve_page_id(ref)
    if resolved and graph.get_page(resolved) is not None:
        return resolved
    return temp_refs.get(ref, "")


def _allocate_function_id(graph: FunctionalFlowGraph, next_index: int) -> tuple[str, int]:
    function_id = f"func_{next_index}"
    while function_id in graph.functions:
        next_index += 1
        function_id = f"func_{next_index}"
    return function_id, next_index + 1


def _allocate_page_id(graph: FunctionalFlowGraph, next_index: int) -> tuple[str, int]:
    page_id = f"page_{next_index}"
    while page_id in graph.pages:
        next_index += 1
        page_id = f"page_{next_index}"
    return page_id, next_index + 1


def _function_match_key(value: object) -> str:
    return TEXT_NORMALIZE_PATTERN.sub("", str(value or "").strip().lower())


def _find_exact_duplicate_function_id(
    graph: FunctionalFlowGraph,
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
    graph: FunctionalFlowGraph,
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
        # Preserve the first stable representative image; later keyboard, dialog or media states remain additional evidence.


        if page.representative_screenshot:
            page.add_evidence_screenshot(screenshot)
        else:
            page.update_representative_screenshot(screenshot)
    page.activity = page.activity or _page_activity(source_page)


def _find_entity_generalized_page(
    graph: FunctionalFlowGraph,
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
    graph: FunctionalFlowGraph,
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
    graph: FunctionalFlowGraph,
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
    graph: FunctionalFlowGraph,
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
    graph: FunctionalFlowGraph,
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
                    "[FFG update] Skip functionality name/description update without current direct evidence: "
                    f"{existing_function_id}"
                )
            elif identity_fields:
                graph.update_function(existing_function_id, identity_fields)
            if resources and require_direct_evidence and not has_direct_evidence:
                print(
                    "[FFG update] Skip resource-variable update without current direct evidence: "
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
                "[FFG update] Skip new functionality without current direct evidence: "
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
                "[FFG update] Skip empty or exact duplicate functionality node: "
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
        node = FunctionalityNode(
            id=function_id,
            name=name or description or function_id,
            description=description or name or function_id,
            resource_variables=resources,
        )
        graph.add_node(node)
        function_refs[ref] = function_id
    return next_function_index


def _apply_pages(
    graph: FunctionalFlowGraph,
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
                "[FFG update] Skip page without original-step evidence: "
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
        # A resolved temporary reference updates its existing allocation rather than creating another page.

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
                "[FFG update] Reuse equivalent page after resource-instance generalization: "
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
                "[FFG update] Reuse page with matching Activity and control signature: "
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
    graph: FunctionalFlowGraph,
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
                "[FFG update] Skip ownership change without current direct evidence: "
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
    graph: FunctionalFlowGraph,
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
    graph: FunctionalFlowGraph,
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
    graph: FunctionalFlowGraph,
    item: dict[str, Any],
    prefix: str,
    function_refs: dict[str, str],
    page_refs: dict[str, str],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate a split without mutation. Drop invalid page/scenario references; skip a split without valid replacements. Retiring a source requires complete transfer of pages, scenarios, resources and dependencies."""


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
    graph: FunctionalFlowGraph,
    patch_items: list[dict[str, Any]],
    function_refs: dict[str, str],
    page_refs: dict[str, str],
) -> tuple[dict[str, list[str]], list[str]]:
    """Validate each coarse-functionality split before applying it. Transfer existing pages to finer goals; retire the source only after all pages, scenarios, resources and dependencies are preserved."""


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
            # Only navigation-shell pages remain; replacements retain their required entry references.
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
    graph: FunctionalFlowGraph,
    patch_items: list[dict[str, Any]],
    function_refs: dict[str, str],
) -> tuple[dict[str, list[str]], list[str]]:
    """Merge existing functionalities only after validating sources, target and prior patch consumption; skip invalid entries before mutation."""


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
        # Skip stale page membership without discarding the remaining merge.
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
    graph: FunctionalFlowGraph,
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
    graph: FunctionalFlowGraph,
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


def validate_ffg_invariants(graph: FunctionalFlowGraph) -> list[str]:
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
    graph: FunctionalFlowGraph,
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
    graph: FunctionalFlowGraph,
    current_plan: dict[str, Any],
) -> str:
    target_function_id = str(current_plan.get("target_function_id", "")).strip()
    if graph.get_node(target_function_id) is not None:
        return graph.resolve_function_id(target_function_id)
    return ""


def _is_depth_plan(current_plan: dict[str, Any]) -> bool:
    """Whether current_plan selects an intra-functionality depth scenario."""
    return str(current_plan.get("function_phase", "") or "").strip().lower() == "depth"


def _is_exploratory_navigation_plan(current_plan: dict[str, Any]) -> bool:
    """Whether navigation uses level D because the FFG lacks a recorded route to the target."""
    return str(current_plan.get("navigation_level", "") or "").strip().upper() == "D"


def _record_completed_intra_scenario(
    graph: FunctionalFlowGraph,
    current_plan: dict[str, Any],
    test_scenario: str,
) -> str:
    """Record the completed scenario on the functionality selected by the planner."""
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
        return navigation_error or f"Unexpected navigation status: {navigation_status}"
    return ""


def _navigation_failed(execution_result: dict[str, Any]) -> bool:
    """Whether navigation failed to reach the target before testing started."""
    return (
        str(execution_result.get("scenario_status", "")).strip() in {"", "skipped"}
        and str(execution_result.get("navigation_status", "")).strip() in NAVIGATION_FAILURE_STATUSES
    )


def _test_phase_started(execution_result: dict[str, Any]) -> bool:
    return str(execution_result.get("scenario_status", "")).strip() not in {"", "skipped"}


def _trace_test_data_values(trace: list[dict[str, Any]]) -> list[str]:
    """Collect successful FuncDroid_ test-data inputs, deduplicated in execution order."""
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
    graph: FunctionalFlowGraph,
    state: AgentState,
    execution_result: dict[str, Any],
    yielded_structure: bool = True,
) -> dict[str, int]:
    """Update scheduling from execution facts and return failure counts. Infrastructure errors are not attempts. Inter-functional outcomes are done/failed; navigation and screenshot failures can be retried within budget, except level-D navigation. Intra-functionality test blockers become unavailable; navigation failures defer only at their threshold, or immediately for level D. Screenshot failures are environmental. Other failures and successful attempts with no structural progress after half coverage lower priority. Terminal statuses remain unchanged."""


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
            or "Inter-functional scenario execution failed"
        )
        if completed:
            graph.record_cross_test(src_function_id, dst_function_id, resource, test_scenario, "done")
        else:
            # Navigation/screenshot failures leave the combination untested; level-D navigation has no known path and is not retried.

            navigation_failed = _navigation_failed(execution_result)
            retryable = (
                navigation_failed and not _is_exploratory_navigation_plan(current_plan)
            ) or _is_screenshot_unavailable_execution(execution_result)
            graph.record_cross_test(
                src_function_id, dst_function_id, resource, test_scenario,
                "failed", reason=failure_reason, retryable=retryable,
            )
            if _left_target_app(execution_result) and graph.get_node(src_function_id) is not None:
                # Crashes and app exits often follow source writes; lower source scheduling priority.
                failure_counts[src_function_id] = 1
            print(
                "[FFG update] Recorded failed inter-functional combination "
                f"({'bounded retry allowed' if retryable else 'do not reschedule'}): "
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
                "[FFG update] No new pages or functionalities this round; lower scheduling priority: "
                f"{target_function_id} | coverage={coverage}%"
            )
        return failure_counts
    if not failure_reason or _is_screenshot_unavailable_execution(execution_result):
        return failure_counts

    current_status = graph.get_node_test_status(target_function_id)
    if current_status in TERMINAL_FUNCTION_STATUSES:
        # Do not flip completed statuses during coverage auditing, which would change signatures and repeatedly re-audit the same functionality.

        print(
            "[FFG update] Preserve terminal functionality status despite execution failure: "
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
                "[FFG update] Level-D navigation failed; defer after its single attempt: "
                f"{target_function_id} | {failure_reason}"
            )
        elif target_node.navigation_failure_count >= NAVIGATION_FAILURE_DEFER_LIMIT:
            graph.set_node_test_status(target_function_id, "deferred", reason="navigation_failed")
            print(
                "[FFG update] Consecutive navigation limit reached; defer until path changes or final audit: "
                f"{target_function_id} | {target_node.navigation_failure_count}/"
                f"{NAVIGATION_FAILURE_DEFER_LIMIT} | {failure_reason}"
            )
        else:
            print(
                "[FFG update] Target not reached; record navigation failure and lower priority: "
                f"{target_function_id} | {target_node.navigation_failure_count}/"
                f"{NAVIGATION_FAILURE_DEFER_LIMIT} | {failure_reason}"
            )
        return failure_counts
    if (
        str(execution_result.get("scenario_status", "")).strip() == "blocked"
        and any(marker in str(execution_result.get("scenario_error", "") or "") for marker in (
            EXPLORATION_BOUNDARY_ERROR_MARKER,
            "\uff08\u975e\u76ee\u6807\u5e94\u7528\u6838\u5fc3\u529f\u80fd\uff09\uff0c\u5df2\u81ea\u52a8\u8fd4\u56de\u5e76\u7ed3\u675f\u5f53\u524d\u573a\u666f",
        ))
    ):
        # A boundary blocks one entry, not every remaining control in the functionality.

        failure_counts[target_function_id] = 1
        print(
            "[FFG update] Exploration boundary reached; record failure but keep functionality testable: "
            f"{target_function_id} | {failure_reason}"
        )
        return failure_counts
    if str(execution_result.get("scenario_status", "")).strip() == "blocked":
        graph.set_node_test_status(target_function_id, "unavailable", reason="blocked")
        failure_counts.pop(target_function_id, None)
        print(
            "[FFG update] Impassable functionality blocker; mark unavailable and switch targets: "
            f"{target_function_id} | {failure_reason}"
        )
        return failure_counts
    failure_counts[target_function_id] = 1
    print(
        "[FFG update] Record functionality execution failure and lower scheduling priority: "
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


def _record_intra_attempt(graph: FunctionalFlowGraph, state: AgentState) -> None:
    plan = state.get("current_plan", {}) or {}
    execution_result = state.get("execution_result") or {}
    status = str(execution_result.get("scenario_status", ""))
    # Navigation skips have a separate counter; infrastructure errors are not functionality attempts.
    if (
        plan.get("scenario_type") != "intra_test"
        or status in {"", "skipped"}
        or execution_result.get("infrastructure_error")
    ):
        return
    node = graph.get_node(_resolve_plan_target_function_id(graph, plan))
    if node is not None:
        node.record_intra_attempt()
        # Entering the test phase consumes a depth attempt, including failed attempts.
        if _is_depth_plan(plan):
            node.depth_scenario_count += 1
            if status == "completed" and _trace_has_observed_business_response(
                state.get("scenario_execution_trace", []) or [],
            ):
                node.successful_depth_scenario_count += 1


def _record_execution_facts(
    graph: FunctionalFlowGraph,
    state: AgentState,
    execution_result: dict[str, Any],
    *,
    yielded_structure: bool = True,
    record_attempt_scenario: bool = True,
) -> dict[str, int]:
    """Persist scenarios, inter-functional outcomes, test data and statuses even if model patching fails. Return functionality execution failure counts."""


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
                # Attempt history is execution evidence even when subsequent graph filters exclude all steps.
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
    graph: FunctionalFlowGraph,
    state: AgentState,
    execution_result: dict[str, Any],
    *,
    reason: str = "no recordable trace",
) -> dict[str, Any]:
    _record_screenshot_unavailable_page(graph, state, execution_result)
    current_page = _map_current_page_after_update(graph, state)
    # Without a structure patch, retain facts from the trace: scenarios, inter-functional outcomes and statuses.
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
    graph: FunctionalFlowGraph,
    trace: list[dict[str, Any]],
    trace_page_map: dict[int, str],
    current_plan: dict[str, Any],
    thumbnail_cache: _ScreenshotThumbnailCache | None = None,
) -> list[str]:
    """Reuse screenshot-matching pages, then create one fallback page for each unmodeled Activity under the planned target. Preserve executed operations and transitions despite omitted alignment or model failure; later rounds can refine names and controls."""


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
            page.description = page.description or "Fallback page for an unmodeled Activity; name and controls await review"
            graph.add_page(page, owner_id)
            auto_pages[activity] = page_id
            created.append(page_id)
        else:
            continue
        trace_page_map[raw_step] = page_id
    if created:
        print(
            "[FFG update] Created fallback page for an Activity omitted by the model: "
            + ", ".join(f"{page_id}({graph.get_page(page_id).activity})" for page_id in created)
        )
    return created


def _repair_ffg_invariants(graph: FunctionalFlowGraph, current_plan: dict[str, Any]) -> list[str]:
    """Repair invalid membership, transitions and dependencies, archive stale outcomes, and attach ownerless observations to planned or adjacent functionality owners. Return pre-repair issues for diagnostics instead of rolling back a whole round."""


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
    print("[FFG update] Repaired structural inconsistencies: " + "; ".join(issues[:8]))
    return issues


def _run_patch_stage(name: str, func, *args, default=None, **kwargs):
    """Apply each patch stage independently; a stage exception must not discard other stages or execution facts."""
    try:
        return func(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - one invalid model field must not discard the round
        print(f"[FFG update] Patch stage {name} failed and was skipped: {type(exc).__name__}: {exc}")
        return default


def apply_ffg_patch(
    graph: FunctionalFlowGraph,
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
    """Apply valid patch entries without whole-round rollback, record invalid entries and reconcile invariants. fill_unmapped enables deterministic fallback after model repair opportunities. carried_* preserves temporary references and step mappings between repair calls."""


    current_plan = current_plan or {}
    control_naming_issues = _strip_enumerated_controls(patch)
    for issue in control_naming_issues:
        print(f"[FFG update] {issue}")
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
    # Remove unsupported empty functionality nodes created solely for skipped splits or merges.

    backed_only_ids = {
        function_refs[ref] for ref in persisted_evidence_refs if ref in function_refs
    }
    for function_id in backed_only_ids - existing_function_ids:
        node = graph.get_node(function_id)
        if node is not None and not node.page_ids and not node.scenarios:
            graph.functions.pop(function_id, None)
            print(f"[FFG update] Remove empty functionality created for an unapplied split/merge: {function_id}")
    graph.reconcile_page_membership()
    repaired_issues = _repair_ffg_invariants(graph, current_plan)
    # Keep only existing pages in trace_page_map after invariant repair.
    trace_page_map = {
        raw_step: page_id
        for raw_step, page_id in trace_page_map.items()
        if graph.get_page(page_id) is not None
    }
    validation_issues = list(dict.fromkeys([*merge_issues, *restructure_issues, *repaired_issues]))
    if validation_issues:
        print("[FFG update] Skipped or repaired patch entries: " + "; ".join(validation_issues[:8]))

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
    graph: FunctionalFlowGraph,
) -> bool:
    """Treat alignment issues as resolved when earlier repair calls mapped the step to an existing page."""
    match = _ISSUE_STEP_PATTERN.search(str(issue or ""))
    if not match:
        return False
    page_id = trace_page_map.get(int(match.group(1)), "")
    return bool(page_id) and graph.get_page(page_id) is not None


def _open_update_issues(
    graph: FunctionalFlowGraph,
    trace: list[dict[str, Any]],
    trace_page_map: dict[int, str],
    alignment_issues: list[str],
    patch_result: dict[str, Any],
) -> list[str]:
    """Keep only model-repairable issues that affect graph quality: unmapped observations, enumerated controls and skipped splits/merges. Do not request already mapped steps again."""


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
        # Ask again about a step only if it still has no page mapping.
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
    graph: FunctionalFlowGraph,
    candidate_map: dict[int, list[dict[str, Any]]],
    step_index: int,
) -> list[dict[str, Any]]:
    """Reuse initial text context and attach only affected/adjacent screenshots. Earlier valid patches remain applied; mapped steps are described without uploading all images again."""


    focus_steps = _refinement_focus_steps(open_issues, trace)
    content: list[dict[str, Any]] = list(base_text_blocks)
    content.append({
        "type": "text",
        "text": (
            f"[FFG repair call {step_index}] Valid portions of the preceding patch were applied without rollback. "
            "Supply only the remaining corrections; do not repeat applied entries.\n"
            f"Unresolved issues: {json.dumps(open_issues, ensure_ascii=False)}\n"
            "Repair guidance:\n"
            "- missing observation_alignment: provide reuse page_x or create new_page_x "
            "declared in pages with source_step at its first visible occurrence; later steps may reuse that new_page_x.\n"
            "- ignore alignment: ignore is only for external/recovery observations; normal results must reuse or create.\n"
            "- cannot move non-coverage page / has no valid coverage pages: split only source-owned "
            "coverage pages; choose the correct page_x or omit the split.\n"
            "- unknown references: func_x/page_x must come from the supplied context.\n"
            "- enumerated control names: redeclare controls for the same page_ref, listing independent "
            "clickable entries separately; equivalent resource instances use one type-level name."
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
    """Create incremental FFG updating with bounded model repair calls and deterministic fallback. Retain valid patch entries across calls without whole-round rollback."""


    thumbnail_cache = _ScreenshotThumbnailCache()

    def ffg_update_node(state: AgentState) -> dict:
        raw_trace = _combined_execution_trace(state)
        trace = _filter_trace(raw_trace)
        graph = state.get("ffg")
        if not isinstance(graph, FunctionalFlowGraph):
            graph = FunctionalFlowGraph()
        _record_intra_attempt(graph, state)

        # Merge completed audits before the baseline snapshot so round statistics reflect only its patch; then submit new audits.

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
        ) = _select_functionality_independence_review(
            graph,
            current_plan,
            review_attempts,
        )
        if granularity_review_function_id:
            review_attempts[granularity_review_function_id] = int(
                granularity_review_metrics.get("review_attempt", 1) or 1
            )
            print(
                "[FFG update] Triggered functionality granularity review: "
                f"{granularity_review_function_id} | "
                f"pages={granularity_review_metrics.get('coverage_page_count', 0)}, "
                f"activities={granularity_review_metrics.get('activity_count', 0)}, "
                f"attempt={granularity_review_metrics.get('review_attempt', 1)}/"
                f"{granularity_review_metrics.get('max_review_attempts', 1)}"
            )

        print(
            "[FFG update] Input state: "
            f"scenario_status={execution_result.get('scenario_status', '')}, "
            f"navigation_trace_steps={len(state.get('navigation_execution_trace', []) or [])}, "
            f"scenario_trace_steps={len(state.get('scenario_execution_trace', []) or [])}, "
            f"combined_steps={len(raw_trace)}, processed_steps={len(trace)}"
        )
        if not trace:
            print("[FFG update] No usable trace steps; retain execution outcome and update functionality status.")
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
                    f"[Target app] name={state.get('app_name', '') or 'unknown'}, "
                    f"package={state.get('package_name', '') or 'unknown'}"
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
                    f"Scenario: {test_scenario}\n"
                    f"Execution result: {_execution_result_for_llm(execution_result)}\n"
                    "The steps below include navigation and test traces in actual order. "
                    "Inspect failed actions, interruptions, external pages, recovery and observation-only steps individually, "
                    "then use direct evidence to update Page, operation or PageTransition.\n"
                    "[STEP] index identifies the combined trace position in this round; "
                    "pages.source_step must use this index.\n"
                    "Each image is the pre-action state BEFORE_STEP_index. "
                    "The result of action i must be confirmed by a later STEP image. "
                    "A new result page must reference the later observation source_step, not the triggering action's step."
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
                    "[Stored FFG visual references] These are persisted Page images. "
                    "Compare them with existing textual descriptions when deciding whether to reuse page_x. "
                    "They are not current trace observations, "
                    "cannot serve as pages.source_step and do not prove this round's outcomes."
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
                        f"The image below precedes action={action_name or 'unknown'}."
                        if _page_screenshot(page)
                        else "This step has no usable screenshot."
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
                        f"[OBSERVATION_IMAGE_REUSE] BEFORE_STEP_{raw_index} and "
                        f"BEFORE_STEP_{reused_step} share the same screenshot; image not uploaded twice."
                    ),
                })

        message_content.append({
            "type": "text",
            "text": (
                "[Trace chronology checks]\n"
                "- STEP i shows the source page before action i; use the next available BEFORE_STEP image to judge its result.\n"
                "- A new page's source_step must identify the screenshot where it is visible. "
                "For example, a page first visible in BEFORE_STEP_3 after action 2 has source_step=3.\n"
                "- finished is a final observation, not a page operation or transition trigger.\n"
                "- executor_target_hint/executor_reasoning_hint are explanations only. "
                "Coordinates and actual screenshot targets take precedence when they disagree.\n"
                "- Coordinates use original_screenshot_size pixels. "
                "Uploaded images may be proportionally resized; map coordinates by dimension ratios.\n"
                "- A final side-effecting action without a subsequent observation cannot establish a result page, operation or transition."
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
            "[FFG update] Context size: "
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
                "[FFG update] Runtime deadline reached; skip multimodal invocation. "
                "Raw trace is persisted; finalize this round from its execution outcome."
            )
            return _apply_non_recordable_execution_outcome(graph, state, execution_result)
        deadline_monotonic = (
            None if available_llm_seconds is None else time.monotonic() + available_llm_seconds
        )

        # Apply each valid increment before a focused repair request; defer coarse deterministic fallback until the model has exhausted its semantic repair opportunities.


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
                print("[FFG update] Runtime deadline approaching; stop further repair calls.")
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
                print(f"[FFG update] Repair call {step_index}/{FFG_UPDATE_MAX_STEPS}: " + "; ".join(open_issues[:5]))
            steps_run = step_index
            patch = invoke_llm_json_with_retry(
                llm,
                [
                    SystemMessage(content=FFG_UPDATE_SYSTEM_PROMPT),
                    HumanMessage(content=content),
                ],
                log_prefix="[FFG update]",
                json_block_pattern=JSON_BLOCK_PATTERN,
                max_attempts=max_attempts,
                deadline_monotonic=deadline_monotonic,
            )
            if llm_json_result_failed(patch):
                if not is_repair:
                    model_patch_unavailable = True
                    print(
                        "[FFG update] Multimodal invocation exhausted retries; use deterministic trace mapping "
                        "(reuse matching screenshots, create unmodeled Activities) and retain execution facts."
                    )
                else:
                    print("[FFG update] Repair invocation failed; retain applied entries and use deterministic fallback.")
                patch = {}
            elif not patch:
                print("[FFG update] No valid model patch; retain execution outcomes without model structure additions.")
                patch = {}

            _normalize_observation_alignment(patch, graph, trace, candidate_map)
            auto_fixed = _ensure_ignorable_alignments(patch, trace)
            if auto_fixed:
                print(f"[FFG update] Filled external/recovery ignore alignments: {auto_fixed}")
            # Repair calls need restate unresolved steps only; earlier mappings remain valid.
            alignment_issues = [
                issue
                for issue in validate_observation_alignment(graph, patch, trace, candidate_map)
                if not _issue_step_already_mapped(issue, carried_trace_page_map, graph)
            ]
            # Stop when initial invocation failed or a repair produced nothing useful.
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
                print("[FFG update] Unresolved issues unchanged; stop model repair: " + "; ".join(open_issues[:3]))
                break
            previous_open_issues = open_issues

        # After bounded model calls, map remaining observations and repair final graph invariants.
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
                "[FFG update] Persisted trace screenshots aligned with existing Pages: "
                f"{patch_result['persisted_trace_images']} images"
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
                f"[FFG update] Recorded {bound_operations} explicit operation-control bindings "
                "(authoritative for control coverage)"
            )
        graph.recompute_all_coverages()
        print(
            "[FFG update] LLM patch: "
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
                ["Trace Activities still lack a Page and assignable functionality: "
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
            "[FFG update] Increment: "
            f"functionalities={after['functions'] - before['functions']}, "
            f"pages={after['pages'] - before['pages']}, "
            f"operations={after['operations'] - before['operations']}, "
            f"page_transitions={after['page_transitions'] - before['page_transitions']}, "
            f"function_dependencies={after['function_dependencies'] - before['function_dependencies']}"
        )

        coverage_after = (
            int(graph.function_coverage_details(coverage_target_id)["coverage_percent"])
            if coverage_target_id and graph.get_node(coverage_target_id) is not None else 0
        )
        # Progress includes new functionalities/pages/operations/transitions/bindings or increased target coverage.
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
