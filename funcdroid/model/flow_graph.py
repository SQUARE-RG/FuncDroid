# Unicode escapes preserve literal UI labels and legacy evidence matching.
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from dataclasses import dataclass, field

TEXT_NORMALIZE_PATTERN = re.compile(r"[\"'“”‘’（）()【】\[\]{}<>《》、，,。.：:；;!！?？\-_/\s]+")
# Legacy verification markers are removed on load and counted as depth attempts.
LEGACY_DEPTH_MARKER_PATTERN = re.compile(r"^\[(?:verify|mismatch):[^\]]*\]\s*")
MAX_EVIDENCE_SCREENSHOTS_PER_PAGE = max(
    0,
    int(os.getenv("FFG_MAX_SCREENSHOTS_PER_PAGE", "12")) - 1,
)
# Explicit Page.operation_bindings are authoritative. Legacy text matching can merge distinct controls or miss paraphrases; enable FFG_COVERAGE_LEGACY_MATCHING=1 only for compatibility.
COVERAGE_LEGACY_TEXT_MATCHING = (
    os.getenv("FFG_COVERAGE_LEGACY_MATCHING", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)

# Functionality status machine: completed requires successful scenarios and sufficient coverage. Other terminal states do not imply successful completion.
FUNCTION_STATUSES = {
    "not_started",
    "in_progress",
    "completed",
    "blocked",
    "deferred",
    "unavailable",
}
# Terminal statuses leave normal scheduling; only completed functionalities receive completion audits.
TERMINAL_FUNCTION_STATUSES = {"completed", "blocked", "deferred", "unavailable"}
# Statuses remaining in the test queue.
ACTIVE_FUNCTION_STATUSES = {"not_started", "in_progress"}


def _write_bytes_if_changed(path: str, data: bytes) -> None:
    if os.path.exists(path):
        try:
            if os.path.getsize(path) == len(data):
                with open(path, "rb") as existing:
                    if existing.read() == data:
                        return
        except OSError:
            pass
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _text_key(text: str) -> str:
    return TEXT_NORMALIZE_PATTERN.sub("", str(text or "").strip().lower())


def _is_same_text(left: str, right: str) -> bool:
    left_key = _text_key(left)
    right_key = _text_key(right)
    return bool(left_key and right_key and left_key == right_key)


def _prefer_richer_text(existing: str, candidate: str) -> str:
    existing = str(existing or "").strip()
    candidate = str(candidate or "").strip()
    if not existing:
        return candidate
    if not candidate:
        return existing
    if len(candidate) > len(existing) + 12:
        return candidate
    return existing


def _merge_exact_texts(values: list[str]) -> list[str]:
    merged: list[str] = []
    for value in values:
        value = str(value or "").strip()
        if not value:
            continue
        for index, existing in enumerate(merged):
            if _is_same_text(existing, value):
                merged[index] = _prefer_richer_text(existing, value)
                break
        else:
            merged.append(value)
    return merged


def _resolve_list_text(text: object, values: list[str]) -> str:
    """Return the original list entry matching text, preferring exact over normalized equality."""
    text = str(text or "").strip()
    if not text:
        return ""
    if text in values:
        return text
    for value in values:
        if _is_same_text(value, text):
            return value
    return ""


def _normalize_operation_bindings(
    bindings: list[dict] | None,
    operations: list[str],
    controls: list[str],
) -> list[dict]:
    """Keep bindings to recorded entries and deduplicate by (operation, control)."""
    normalized: list[dict] = []
    seen: set[tuple[str, str]] = set()
    bound_operations: set[str] = set()
    for binding in bindings or []:
        if not isinstance(binding, dict):
            continue
        operation = _resolve_list_text(binding.get("operation"), operations)
        control = _resolve_list_text(binding.get("control"), controls)
        if not operation or not control:
            continue
        key = (_text_key(operation), _text_key(control))
        if key in seen or key[0] in bound_operations:
            continue
        seen.add(key)
        bound_operations.add(key[0])
        normalized.append({"operation": operation, "control": control})
    return normalized


# Accept a closing parenthesis after a failed result, including screenshot-health failures, so failures cannot count as successful operations.
OPERATION_FAILURE_PATTERN = re.compile(r"^(?:\u5c1d\u8bd5)?(.+?)\u5931\u8d25(?:[：:].*)?[）)]?$")
OPERATION_RECOVERY_PATTERN = re.compile(
    r"automatic recovery|restore (?:the )?target app|open .+ (?:app|main page)|\u81ea\u52a8\u6062\u590d|\u6062\u590d\u76ee\u6807app|\u6253\u5f00.+\u5e94\u7528|\u6253\u5f00.+\u4e3b\u9875\u9762", re.IGNORECASE
)
ENGLISH_OPERATION_FAILURE_PATTERN = re.compile(
    r"^failed attempt:\s*(.*?)\s*(?:;\s*error:.*)?$|^(.+?)\s+failed(?:[:;].*)?$", re.IGNORECASE
)
CONTROL_INSTANCE_PATTERN = re.compile(
    r"\u4f1a\u8bdd\u6761\u76ee|\u8054\u7cfb\u4eba\u6761\u76ee|\u8054\u7cfb\u4eba\u5217\u8868|\u6587\u7ae0\u5361\u7247|\u5185\u5bb9\u5361\u7247|\u5546\u54c1\u5361\u7247|\u52a8\u6001\u5217\u8868\u6761\u76ee|\u5217\u8868\u9879"
)
CONTROL_NOISE_PATTERN = re.compile(
    r"\u4e00\u76f4\u53ef\u89c1\u7684?|\u5f53\u524d\u53ef\u89c1\u7684?|\u53ef\u70b9\u51fb\u7684?|\u9875\u9762\u4e2d\u7684?|\u9875\u9762\u4e0a|\u9876\u90e8\u5de5\u5177\u680f\u4e2d\u7684?|"
    r"\u9876\u90e8|\u5e95\u90e8\u5bfc\u822a\u680f\u7684?|\u5e95\u90e8|\u5de6\u4e0a\u89d2|\u53f3\u4e0a\u89d2|\u5de6\u4e0b\u89d2|\u53f3\u4e0b\u89d2|\u4e2d\u7684?|\u4e0a\u7684?"
)
CONTROL_WIDGET_PATTERN = re.compile(
    r"\u6309\u94ae|\u56fe\u6807|\u5165\u53e3\u6761\u76ee|\u5165\u53e3\u6309\u94ae|\u5165\u53e3|\u9009\u9879|\u63a7\u4ef6|\u533a\u57df|\u6807\u7b7e（\u9009\u4e2d\u72b6\u6001）|\u6807\u7b7e\(\u9009\u4e2d\u72b6\u6001\)|\u6807\u7b7e"
)
NON_INTERACTIVE_STATE_PATTERN = re.compile(
    r"loading indicator|publication time|time separator|unread badge state|status hint|hint text|system keyboard|\u52a0\u8f7d\u6307\u793a\u5668|\u53d1\u5e03\u65f6\u95f4|\u65f6\u95f4\u5206\u9694|\u672a\u8bfb\u7ea2\u70b9\u72b6\u6001|\u72b6\u6001\u63d0\u793a|\u63d0\u793a\u6587\u672c|\u7cfb\u7edf\u8f6f\u952e\u76d8", re.IGNORECASE
)
# Back-to-home and generic Back are distinct controls. Use the destination immediately after the first Back occurrence.
BACK_DESTINATION_PATTERN = re.compile(r"\u8fd4\u56de(?:\u5230)?(\u9996\u9875|\u4e3b\u9875|\u4e3b\u754c\u9762|\u684c\u9762|\u9876\u90e8)")


def _back_destination(text: str) -> str:
    index = text.find("\u8fd4\u56de")
    if index < 0:
        return ""
    match = BACK_DESTINATION_PATTERN.match(text, index)
    if not match:
        return ""
    return {
        "\u9996\u9875": "home", "\u4e3b\u9875": "home", "\u4e3b\u754c\u9762": "home",
        "\u684c\u9762": "desktop", "\u9876\u90e8": "top",
    }[match.group(1)]


def control_family_key(value: object) -> str:
    """Return a relaxed semantic family key for one visible control."""
    text = str(value or "").strip().lower()
    if not text:
        return ""
    english_back = re.fullmatch(
        r"(?:(?:top|upper)[ -]?(?:left|right)?(?: corner)?\s+)?back"
        r"(?:(?::| to )(home|main page|desktop|top))?(?: (?:button|arrow|icon|control))?", text,
    )
    if english_back:
        destination = english_back.group(1)
        if destination == "main page":
            destination = "home"
        return "back" + (":" + destination if destination else "")
    # English authored descriptions coexist with literal labels from any UI language.
    text = re.sub(r"\s*\(representative[: ].*?\)", "", text)
    text = re.sub(r"^(?:(?:currently )?visible|clickable|top toolbar)\s+", "", text)
    text = re.sub(r"\s+(?:button|icon|control|widget|entrypoint)$", "", text)
    text = re.sub(r"（\u4ee3\u8868[：:].*?）|\(\u4ee3\u8868[：:].*?\)", "", text)
    text = re.sub(r"[“”‘’\"']", "", text)
    text = re.sub(r"\d+\s*\u6761", "", text)
    # Extract the back destination before removing positional noise.
    back_destination = _back_destination(text)
    text = CONTROL_NOISE_PATTERN.sub("", text)

    # Visual state, item names and category choices do not create new
    # semantic families. One representative selection is sufficient.
    if "\u8868\u60c5" in text:
        # Search, delete/backspace and the panel toggle are separate fixed
        # semantic controls despite sharing the same emoji-related label.
        if "\u641c\u7d22" in text:
            return "\u8868\u60c5\u641c\u7d22"
        if any(token in text for token in ("\u5220\u9664", "\u9000\u683c")):
            return "\u8868\u60c5\u5220\u9664"
        if "\u6d88\u606f" in text:
            return "\u8868\u60c5\u6d88\u606f"
        if any(token in text for token in ("\u5206\u7c7b", "\u8868\u60c5\u9879", "\u9009\u62e9\u9762\u677f", "\u9009\u62e9\u9879")) or re.search(
            r"(?:\u7b2c\u4e00\u4e2a|\u9996\u4e2a|\u4efb\u610f\u4e00\u4e2a|\u540d\u4e3a.+\u7684)\u8868\u60c5",
            text,
        ):
            return "\u8868\u60c5\u9009\u62e9\u9879"
        if any(token in text for token in ("\u6309\u94ae", "\u56fe\u6807", "\u5165\u53e3")):
            return "\u8868\u60c5"
    if "\u53d1\u9001" in text and any(token in text for token in ("\u6309\u94ae", "\u56fe\u6807", "\u64cd\u4f5c")):
        return "\u53d1\u9001"

    if (
        any(token in text for token in ("\u6587\u672c\u8f93\u5165\u6846", "\u6587\u5b57\u8f93\u5165\u6846", "\u6d88\u606f\u8f93\u5165\u6846"))
        and "\u641c\u7d22" not in text
    ):
        return "\u6587\u672c\u8f93\u5165\u6846"

    if "\u5217\u8868\u9879" in text:
        # Named list categories are distinct evidence. Do not strip concrete
        # labels with an App-specific dictionary: that silently turns different
        # business rows into one generic list-item family.
        label = text.replace("\u5217\u8868\u9879", "")
        label = CONTROL_NOISE_PATTERN.sub("", label)
        label = TEXT_NORMALIZE_PATTERN.sub("", label)
        return f"\u5217\u8868\u9879:{label}" if label else "\u5217\u8868\u9879"
    if "\u4f1a\u8bdd" in text and (
        "\u6761\u76ee" in text
        or "\u5217\u8868" in text
        or re.search(r"(?:\u7684|\u4e2a|\u4efb\u4e00|\u4efb\u610f)\u4f1a\u8bdd$", text)
    ):
        # Instance-level conversation labels share one control family; repeated instance operations are merged.
        return "\u4f1a\u8bdd\u6761\u76ee"
    if "\u8054\u7cfb\u4eba" in text and ("\u6761\u76ee" in text or "\u5217\u8868" in text):
        return "\u8054\u7cfb\u4eba\u6761\u76ee"
    if "\u6587\u7ae0" in text and ("\u5361\u7247" in text or "\u6807\u9898" in text or "\u5c01\u9762" in text):
        return "\u6587\u7ae0\u5361\u7247"
    if "\u5185\u5bb9" in text and any(token in text for token in ("\u5361\u7247", "\u6807\u9898", "\u56fe\u7247", "\u7f29\u7565\u56fe")):
        return "\u5185\u5bb9\u5361\u7247"
    if "\u5206\u7c7b\u6807\u7b7e" in text:
        return "\u5206\u7c7b\u6807\u7b7e"
    if text.endswith("\u5206\u533a") or "\u5206\u533a\u6807\u9898" in text:
        return "\u5185\u5bb9\u5206\u533a"
    if "\u641c\u7d22" in text and any(token in text for token in ("\u653e\u5927\u955c", "\u6309\u94ae", "\u56fe\u6807", "\u8f93\u5165\u6846")):
        return "\u641c\u7d22\u8f93\u5165\u6846" if "\u8f93\u5165\u6846" in text else "\u641c\u7d22"
    if "\u8fd4\u56de" in text and (text == "\u8fd4\u56de" or back_destination or any(token in text for token in ("\u7bad\u5934", "\u6309\u94ae", "\u64cd\u4f5c"))):
        return "back" + (":" + back_destination if back_destination else "")
    if "\u66f4\u591a" in text and any(token in text for token in ("\u83dc\u5355", "\u6309\u94ae", "\u56fe\u6807")):
        return "\u66f4\u591a\u83dc\u5355"

    text = CONTROL_WIDGET_PATTERN.sub("", text)
    text = TEXT_NORMALIZE_PATTERN.sub("", text)
    return text


def control_is_coverage_candidate(value: object) -> bool:
    text = str(value or "").strip()
    return bool(text and not NON_INTERACTIVE_STATE_PATTERN.search(text))


def _operation_action_and_target(value: object) -> tuple[str, str]:
    text = str(value or "").strip()
    if not text:
        return "", ""
    text = re.sub(r"^automatic recovery:\s*", "", text, flags=re.IGNORECASE)
    english_failure = ENGLISH_OPERATION_FAILURE_PATTERN.match(text)
    if english_failure:
        text = (english_failure.group(1) or english_failure.group(2) or "").strip()
    if re.match(r"^(?:(?:click|tap|press)\s+(?:the\s+)?)?(?:back|press_back)\b", text, re.IGNORECASE):
        destination = re.search(r"\bback\s+to\s+(home|main page|desktop|top)\b", text, re.IGNORECASE)
        return "back", "back" + (":" + destination.group(1).lower() if destination else "")
    english_actions = (
        ("long_press", r"^(?:long[-_ ]press)\s+"),
        ("type", r"^(?:type|input|enter)\s+"),
        ("scroll", r"^(?:scroll|swipe|drag)\s+"),
        ("open", r"^(?:open|launch)\s+"),
        ("click", r"^(?:click|tap|select|switch to|check)\s+"),
    )
    for action, pattern in english_actions:
        if re.match(pattern, text, re.IGNORECASE):
            target = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE)
            target = re.split(r"\s*(?:;|->|→)\s*|\s+(?:then|and then)\s+", target, maxsplit=1)[0]
            return action, target.strip()
    text = re.sub(r"^\u81ea\u52a8\u6062\u590d[：:]?", "", text)
    failure_match = OPERATION_FAILURE_PATTERN.match(text)
    if failure_match:
        text = failure_match.group(1).strip()
    text = re.sub(r"^\u5c1d\u8bd5", "", text)
    if re.search(r"\u8fd4\u56de|press_back", text, re.IGNORECASE):
        destination = _back_destination(text)
        return "back", "back" + (":" + destination if destination else "")
    if re.match(r"^\u53d1\u9001", text):
        return "click", "\u53d1\u9001"
    action_patterns = (
        ("long_press", r"^(?:\u957f\u6309)"),
        ("type", r"^(?:\u5728.+?\u4e2d)?\u8f93\u5165|^\u8f93\u5165"),
        ("scroll", r"^(?:\u6ed1\u52a8|\u6eda\u52a8|\u4e0a\u6ed1|\u4e0b\u6ed1|\u5de6\u6ed1|\u53f3\u6ed1|\u62d6\u52a8|\u62d6\u62fd)"),
        ("open", r"^(?:\u6253\u5f00|\u542f\u52a8)"),
        ("click", r"^(?:\u70b9\u51fb|\u9009\u62e9|\u5207\u6362\u5230|\u52fe\u9009)"),
    )
    for action, pattern in action_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            target = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE).strip()
            target = re.split(r"(?:\u5e76|\u540e|，|,|；|;|→|->)", target, maxsplit=1)[0]
            return action, target
    return "action", text


def operation_family_key(value: object) -> str:
    """Canonical action+target key; outcome text does not create a new family."""
    action, target = _operation_action_and_target(value)
    if not action:
        return ""
    target_key = control_family_key(target)
    return f"{action}:{target_key or _text_key(target)}"


def operation_is_failure(value: object) -> bool:
    text = str(value or "").strip()
    return bool(
        OPERATION_FAILURE_PATTERN.match(text)
        or ENGLISH_OPERATION_FAILURE_PATTERN.match(text)
        or "\u52a8\u4f5c\u6267\u884c\u5931\u8d25" in text
        or "action execution failed" in text.lower()
    )


def operation_is_recovery(value: object) -> bool:
    return bool(OPERATION_RECOVERY_PATTERN.search(str(value or "")))


def _merge_operation_families(values: list[str]) -> list[str]:
    merged: list[str] = []
    keys: list[str] = []
    for value in values:
        value = str(value or "").strip()
        key = operation_family_key(value)
        if not value or not key:
            continue
        if key not in keys:
            keys.append(key)
            merged.append(value)
            continue
        index = keys.index(key)
        existing = merged[index]
        if operation_is_failure(existing) and not operation_is_failure(value):
            merged[index] = value
        elif operation_is_failure(existing) == operation_is_failure(value):
            merged[index] = _prefer_richer_text(existing, value)
    return merged


def _coverage_target_ratio(total_controls: int) -> float:
    if total_controls <= 4:
        return 1.0
    if total_controls <= 12:
        return 0.8
    return 0.7


def adjusted_control_coverage(total_controls: int, executed_operations: int) -> int:
    """Scale direct control/operation coverage: small functions need all."""
    total = max(0, int(total_controls or 0))
    handled = max(0, min(total, int(executed_operations or 0)))
    if total == 0:
        return 0
    raw_ratio = handled / total
    target_ratio = _coverage_target_ratio(total)
    if target_ratio >= 1.0:
        score = 100.0 * raw_ratio
    elif raw_ratio <= target_ratio:
        score = 90.0 * raw_ratio / target_ratio
    else:
        score = 90.0 + 10.0 * (raw_ratio - target_ratio) / (1.0 - target_ratio)
    return max(0, min(100, int(round(score))))


def _coverage_match_key(value: object) -> str:
    """Lightweight key used only to produce useful uncovered-control hints."""
    text = CONTROL_NOISE_PATTERN.sub("", str(value or "").strip().lower())
    text = CONTROL_WIDGET_PATTERN.sub("", text)
    return _text_key(text)


def _coverage_keys_refer_to_same_control(control_key: str, target_key: str) -> bool:
    """Decide whether an operation target identifies the same control.

    Allow descriptive qualifiers but reject numeric suffix extensions, such as control1 versus control12, that would incorrectly cover a different control."""
    if not control_key or not target_key:
        return False
    if control_key == target_key:
        return True
    shorter, longer = sorted((control_key, target_key), key=len)
    start = longer.find(shorter)
    if start < 0:
        return False
    tail = longer[start + len(shorter):]
    return not (shorter[-1:].isdigit() and tail[:1].isdigit())


def _control_has_binding(page: Page, control: str) -> bool:
    """Return whether a successful, non-recovery operation explicitly binds this control."""
    for binding in page.operation_bindings:
        operation = str(binding.get("operation", "") or "")
        if operation_is_failure(operation) or operation_is_recovery(operation):
            continue
        if _is_same_text(str(binding.get("control", "") or ""), control):
            return True
    return False


def _control_text_matched(page: Page, control: str) -> bool:
    """Legacy operation-family matching; see function_handled_controls."""
    control_match_key = _coverage_match_key(control)
    control_family = control_family_key(control)
    for operation in page.operations:
        if operation_is_recovery(operation):
            continue
        _action, target = _operation_action_and_target(operation)
        target_match_key = _coverage_match_key(target)
        target_family = operation_family_key(operation).partition(":")[2]
        families_agree = bool(
            control_family
            and target_family
            and control_family == target_family
        )
        families_conflict = bool(
            control_family and target_family and control_family != target_family
        )
        if families_agree or (
            not families_conflict
            and _coverage_keys_refer_to_same_control(control_match_key, target_match_key)
        ):
            return True
    return False


def _read_screenshot(value: object, screenshots_dir: str | None) -> str:
    ref = str(value or "").strip()
    if not ref:
        return ""
    if screenshots_dir:
        path = os.path.join(screenshots_dir, ref)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("ascii")
        # A missing referenced screenshot is invalid input, not inline image data. Fail during load for both seeds and resumes instead of corrupting later checkpoints.
        raise ValueError(f"Cannot load FFG: screenshot file is missing: {path}")
    return ref


# Cache only the latest content hash per existing screenshot path. Bound the cache and clear it when full; A-to-B-to-A writes must not reuse stale entries.
_SCREENSHOT_LAST_WRITTEN: dict[str, str] = {}
_SCREENSHOT_LAST_WRITTEN_MAX = 4096


def _write_screenshot(
    screenshot: str,
    filename: str,
    screenshots_dir: str | None,
) -> str:
    screenshot = str(screenshot or "").strip()
    if not screenshot:
        return ""
    if not screenshots_dir:
        return screenshot
    path = os.path.join(screenshots_dir, filename)
    content_key = ""
    try:
        # Hash the ASCII base64 text directly instead of decoding and rereading the image.
        content_key = hashlib.sha256(screenshot.encode("ascii")).hexdigest()
    except (AttributeError, UnicodeEncodeError):
        content_key = ""
    if content_key and _SCREENSHOT_LAST_WRITTEN.get(path) == content_key and os.path.exists(path):
        return filename
    _write_bytes_if_changed(path, base64.b64decode(screenshot))
    if content_key:
        if len(_SCREENSHOT_LAST_WRITTEN) >= _SCREENSHOT_LAST_WRITTEN_MAX:
            _SCREENSHOT_LAST_WRITTEN.clear()
        _SCREENSHOT_LAST_WRITTEN[path] = content_key
    else:
        _SCREENSHOT_LAST_WRITTEN.pop(path, None)
    return filename


@dataclass(slots=True, init=False)
class Page:
    """A stable logical page with a representative screenshot and bounded state history."""

    id: str | None = None
    name: str = ""
    description: str = ""
    representative_screenshot: str = ""
    evidence_screenshots: list[str] = field(default_factory=list)
    controls: list[str] = field(default_factory=list)
    operations: list[str] = field(default_factory=list)
    operation_bindings: list[dict] = field(default_factory=list)
    activity: str = ""

    def __init__(
        self,
        id: str | None = None,
        name: str = "",
        description: str = "",
        representative_screenshot: str = "",
        evidence_screenshots: list[str] | None = None,
        controls: list[str] | None = None,
        operations: list[str] | None = None,
        operation_bindings: list[dict] | None = None,
        activity: str = "",
    ):
        self.id = id
        self.name = str(name or "")
        self.description = str(description or "")
        self.representative_screenshot = str(representative_screenshot or "").strip()
        self.evidence_screenshots = list(evidence_screenshots or [])
        self.controls = list(controls or [])
        self.operations = list(operations or [])
        self.operation_bindings = [
            dict(binding) for binding in (operation_bindings or []) if isinstance(binding, dict)
        ]
        self.activity = str(activity or "")
        self.__post_init__()

    def __post_init__(self):
        self.representative_screenshot = str(self.representative_screenshot or "").strip()
        evidence = list(dict.fromkeys(
            str(value).strip()
            for value in self.evidence_screenshots
            if str(value).strip()
            and str(value).strip() != self.representative_screenshot
        ))
        self.evidence_screenshots = (
            evidence[-MAX_EVIDENCE_SCREENSHOTS_PER_PAGE:]
            if MAX_EVIDENCE_SCREENSHOTS_PER_PAGE > 0
            else []
        )
        self.controls = _merge_exact_texts(self.controls)
        self._remerge_operations()
        self.operation_bindings = _normalize_operation_bindings(
            self.operation_bindings,
            self.operations,
            self.controls,
        )

    def _remerge_operations(self) -> None:
        """Merge operation families and migrate bindings when the retained wording changes."""
        merged = _merge_operation_families(self.operations)
        if merged == self.operations:
            return
        old_to_new: dict[str, str] = {}
        for old in self.operations:
            if old in merged:
                continue
            family = operation_family_key(old)
            replacement = next(
                (
                    candidate
                    for candidate in merged
                    if operation_family_key(candidate) == family
                ),
                "",
            )
            if replacement:
                old_to_new[old] = replacement
        self.operations = merged
        for binding in self.operation_bindings:
            replacement = old_to_new.get(str(binding.get("operation", "") or ""))
            if replacement:
                binding["operation"] = replacement

    def update_representative_screenshot(self, screenshot: str):
        screenshot = str(screenshot or "").strip()
        if not screenshot or screenshot == self.representative_screenshot:
            return
        previous = self.representative_screenshot
        self.representative_screenshot = screenshot
        if previous:
            self.add_evidence_screenshot(previous)
        self.evidence_screenshots = [
            value for value in self.evidence_screenshots if value != screenshot
        ]

    def add_evidence_screenshot(self, screenshot: str):
        screenshot = str(screenshot or "").strip()
        if (
            not screenshot
            or screenshot == self.representative_screenshot
            or screenshot in self.evidence_screenshots
        ):
            return
        self.evidence_screenshots.append(screenshot)
        if len(self.evidence_screenshots) > MAX_EVIDENCE_SCREENSHOTS_PER_PAGE:
            del self.evidence_screenshots[0]

    def add_control(self, control: str):
        control = str(control or "").strip()
        if control:
            self.controls = _merge_exact_texts([*self.controls, control])

    def merge_controls(self, sources: list[str], target: str) -> bool:
        """Merge exact control instances into one type-level control.

        Page audit uses this only after strict source validation. Existing
        operation bindings are migrated so real execution evidence is preserved.
        """
        target = str(target or "").strip()
        exact_sources = list(dict.fromkeys(
            str(source or "").strip()
            for source in sources
            if str(source or "").strip() in self.controls
        ))
        if not target or len(exact_sources) < 2:
            return False
        source_set = set(exact_sources)
        first_index = min(self.controls.index(source) for source in exact_sources)
        retained = [control for control in self.controls if control not in source_set]
        if target not in retained:
            retained.insert(min(first_index, len(retained)), target)
        self.controls = _merge_exact_texts(retained)
        for binding in self.operation_bindings:
            if str(binding.get("control", "") or "") in source_set:
                binding["control"] = target
        self.operation_bindings = _normalize_operation_bindings(
            self.operation_bindings,
            self.operations,
            self.controls,
        )
        return True

    def add_operation(self, operation: str):
        operation = str(operation or "").strip()
        if operation:
            # Merge through _remerge_operations so retained wording and bindings move together; dangling bindings would otherwise disappear on reload.
            self.operations = [*self.operations, operation]
            self._remerge_operations()

    def remove_operation(self, operation: str) -> bool:
        """Remove an operation only when its exact recorded text matches.

        Page auditing cannot delete an operation through a paraphrase or invented name. Remove associated bindings while preserving page transitions."""
        operation = str(operation or "").strip()
        if operation and operation in self.operations:
            self.operations = [item for item in self.operations if item != operation]
            self.operation_bindings = [
                binding
                for binding in self.operation_bindings
                if not _is_same_text(binding.get("operation", ""), operation)
            ]
            return True
        return False

    def add_operation_binding(self, operation: str, control: str) -> bool:
        """Bind a recorded operation to a recorded control on this page.

        Explicit bindings are authoritative coverage evidence. Both sides must match the original lists; return False for an invalid binding so the caller can report and discard it."""
        operation = _resolve_list_text(operation, self.operations)
        control = _resolve_list_text(control, self.controls)
        if not operation or not control:
            return False
        key = (_text_key(operation), _text_key(control))
        for binding in self.operation_bindings:
            existing_key = (
                _text_key(binding.get("operation", "")),
                _text_key(binding.get("control", "")),
            )
            # One recorded operation describes one concrete action.  If an
            # over-broad operation family collapsed two controls, binding both
            # would falsely cover both; keep the first proven target only.
            if existing_key == key or existing_key[0] == key[0]:
                return False
        self.operation_bindings.append({"operation": operation, "control": control})
        return True

    def has_operation_binding(self, operation: str, control: str) -> bool:
        """Check whether this operation/control binding exists after normalization."""
        key = (
            _text_key(_resolve_list_text(operation, self.operations) or operation),
            _text_key(_resolve_list_text(control, self.controls) or control),
        )
        for binding in self.operation_bindings:
            existing_key = (
                _text_key(binding.get("operation", "")),
                _text_key(binding.get("control", "")),
            )
            if existing_key == key:
                return True
        return False

    def bound_operation_texts(self) -> set[str]:
        """Return original operation texts that already have explicit control bindings."""
        return {
            str(binding.get("operation", "") or "")
            for binding in self.operation_bindings
            if str(binding.get("operation", "") or "")
        }

    def to_dict(self, screenshots_dir: str | None = None) -> dict:
        representative_ref = ""
        evidence_refs: list[str] = []
        if self.id:
            representative_ref = _write_screenshot(
                self.representative_screenshot,
                f"{self.id}_representative.jpg",
                screenshots_dir,
            )
            for index, screenshot in enumerate(self.evidence_screenshots, start=1):
                ref = _write_screenshot(
                    screenshot,
                    f"{self.id}_evidence_{index}.jpg",
                    screenshots_dir,
                )
                if ref:
                    evidence_refs.append(ref)
        else:
            representative_ref = self.representative_screenshot
            evidence_refs = list(self.evidence_screenshots)
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "representative_screenshot": representative_ref,
            "evidence_screenshots": evidence_refs,
            "controls": list(self.controls),
            "operations": list(self.operations),
            "operation_bindings": [
                {
                    "operation": str(binding.get("operation", "") or ""),
                    "control": str(binding.get("control", "") or ""),
                }
                for binding in self.operation_bindings
            ],
            "activity": self.activity,
        }

    @classmethod
    def from_dict(cls, data: dict, screenshots_dir: str | None = None) -> Page:
        representative_ref = data.get("representative_screenshot", "")
        evidence_refs = data.get("evidence_screenshots", [])

        if not isinstance(evidence_refs, list):
            evidence_refs = []
        return cls(
            id=data.get("id"),
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            representative_screenshot=_read_screenshot(representative_ref, screenshots_dir),
            evidence_screenshots=[
                screenshot
                for value in evidence_refs
                if (screenshot := _read_screenshot(value, screenshots_dir))
            ],
            controls=[
                str(value).strip()
                for value in data.get("controls", [])
                if isinstance(value, str) and str(value).strip()
            ],
            operations=[
                str(value).strip()
                for value in data.get("operations", [])
                if isinstance(value, str) and str(value).strip()
            ],
            operation_bindings=[
                {
                    "operation": str(item.get("operation", "") or "").strip(),
                    "control": str(item.get("control", "") or "").strip(),
                }
                for item in data.get("operation_bindings", [])
                if isinstance(item, dict)
            ],
            activity=str(data.get("activity", "")),
        )


@dataclass(slots=True, frozen=True)
class PageTransition:
    src_page_id: str
    dst_page_id: str
    operation: str

    def to_dict(self) -> dict:
        return {
            "src_page_id": self.src_page_id,
            "dst_page_id": self.dst_page_id,
            "operation": self.operation,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PageTransition:
        return cls(
            src_page_id=str(data.get("src_page_id", "")).strip(),
            dst_page_id=str(data.get("dst_page_id", "")).strip(),
            operation=str(data.get("operation", "")).strip(),
        )


@dataclass(slots=True, frozen=True)
class FunctionDependency:
    src_function_id: str
    dst_function_id: str
    resource_variable: str

    def __post_init__(self):
        object.__setattr__(self, "src_function_id", str(self.src_function_id or "").strip())
        object.__setattr__(self, "dst_function_id", str(self.dst_function_id or "").strip())
        object.__setattr__(self, "resource_variable", str(self.resource_variable or "").strip())

    def to_dict(self) -> dict:
        return {
            "src_function_id": self.src_function_id,
            "dst_function_id": self.dst_function_id,
            "resource_variable": self.resource_variable,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FunctionDependency:
        return cls(
            src_function_id=str(data.get("src_function_id", "")).strip(),
            dst_function_id=str(data.get("dst_function_id", "")).strip(),
            resource_variable=str(data.get("resource_variable", "")).strip(),
        )


@dataclass(slots=True)
class FunctionNode:
    id: str
    name: str = ""
    description: str = ""
    page_ids: list[str] = field(default_factory=list)
    entry_page_ids: list[str] = field(default_factory=list)
    resource_variables: list[str] = field(default_factory=list)
    scenarios: list[str] = field(default_factory=list)
    test_status: str = "not_started"
    coverage_percent: int = 0
    intra_attempt_count: int = 0
    # Depth attempts consume budget; failed attempts do not prove completion.
    depth_scenario_count: int = 0
    successful_depth_scenario_count: int = 0
    # Consecutive navigation failures reset when testing starts; defer at the limit.
    navigation_failure_count: int = 0
    # Keep the last transition reason for reopening decisions and reports; empty means an ordinary completion transition.
    status_reason: str = ""

    def __post_init__(self):
        self.id = str(self.id or "").strip()
        self.name = str(self.name or "").strip() or self.description or self.id
        self.description = str(self.description or "").strip()
        self.page_ids = list(dict.fromkeys(
            str(value).strip() for value in self.page_ids if str(value).strip()
        ))
        self.entry_page_ids = list(dict.fromkeys(
            str(value).strip()
            for value in self.entry_page_ids
            if str(value).strip() in self.page_ids
        ))
        self.resource_variables = _merge_exact_texts(self.resource_variables)
        self.scenarios = _merge_exact_texts(self.scenarios)
        try:
            self.intra_attempt_count = max(0, int(self.intra_attempt_count or 0))
        except (TypeError, ValueError):
            self.intra_attempt_count = 0
        try:
            self.depth_scenario_count = max(0, int(self.depth_scenario_count or 0))
        except (TypeError, ValueError):
            self.depth_scenario_count = 0
        try:
            self.successful_depth_scenario_count = max(0, int(self.successful_depth_scenario_count or 0))
        except (TypeError, ValueError):
            self.successful_depth_scenario_count = 0
        try:
            self.navigation_failure_count = max(0, int(self.navigation_failure_count or 0))
        except (TypeError, ValueError):
            self.navigation_failure_count = 0
        status = str(self.test_status or "not_started").strip().lower()
        self.test_status = status if status in FUNCTION_STATUSES else "not_started"
        self.status_reason = str(self.status_reason or "").strip()
        try:
            self.coverage_percent = max(0, min(100, int(self.coverage_percent or 0)))
        except (TypeError, ValueError):
            self.coverage_percent = 0

    @property
    def coverage_page_ids(self) -> list[str]:
        entry_page_ids = set(self.entry_page_ids)
        return [page_id for page_id in self.page_ids if page_id not in entry_page_ids]

    def add_page_id(self, page_id: str, *, entry_only: bool = False):
        page_id = str(page_id or "").strip()
        if page_id and page_id not in self.page_ids:
            self.page_ids.append(page_id)
        if not page_id:
            return
        if entry_only:
            if page_id not in self.entry_page_ids:
                self.entry_page_ids.append(page_id)
        else:
            self.entry_page_ids = [value for value in self.entry_page_ids if value != page_id]

    def add_scenario(self, scenario: str):
        scenario = str(scenario or "").strip()
        if scenario:
            self.scenarios = _merge_exact_texts([*self.scenarios, scenario])

    def intra_attempts(self) -> int:
        # Old snapshots have only distinct scenario text; retain it as a lower bound.
        return max(self.intra_attempt_count, sum(not s.startswith("[cross:") for s in self.scenarios))

    def record_intra_attempt(self) -> None:
        self.intra_attempt_count = self.intra_attempts() + 1

    def add_resource_variable(self, resource_variable: str):
        resource_variable = str(resource_variable or "").strip()
        if resource_variable:
            self.resource_variables = _merge_exact_texts([
                *self.resource_variables,
                resource_variable,
            ])

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name or self.description or self.id,
            "description": self.description,
            "page_ids": list(self.page_ids),
            "entry_page_ids": list(self.entry_page_ids),
            "resource_variables": list(self.resource_variables),
            "scenarios": list(self.scenarios),
            "test_status": self.test_status,
            "coverage_percent": self.coverage_percent,
            "intra_attempt_count": self.intra_attempt_count,
            "depth_scenario_count": self.depth_scenario_count,
            "successful_depth_scenario_count": self.successful_depth_scenario_count,
            "navigation_failure_count": self.navigation_failure_count,
            "status_reason": self.status_reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FunctionNode:
        page_ids = [
            str(value).strip()
            for value in data.get("page_ids", [])
            if str(value).strip()
        ]
        scenarios: list[str] = []
        legacy_depth = 0
        for value in data.get("scenarios", []):
            if not isinstance(value, str) or not value.strip():
                continue
            # Migrate legacy verify/mismatch prefixes into depth-attempt counts.
            text, marked = LEGACY_DEPTH_MARKER_PATTERN.subn("", value.strip(), count=1)
            legacy_depth += marked
            if text.strip():
                scenarios.append(text.strip())
        return cls(
            id=str(data.get("id", "")).strip(),
            name=str(data.get("name", "")).strip(),
            description=str(data.get("description", "")).strip(),
            page_ids=page_ids,
            entry_page_ids=[
                str(value).strip()
                for value in data.get("entry_page_ids", [])
                if str(value).strip()
            ],
            resource_variables=[
                str(value).strip()
                for value in data.get("resource_variables", [])
                if isinstance(value, str) and str(value).strip()
            ],
            scenarios=scenarios,
            test_status=str(data.get("test_status", "not_started")).strip() or "not_started",
            coverage_percent=data.get("coverage_percent", 0),
            intra_attempt_count=data.get("intra_attempt_count", 0),
            depth_scenario_count=data.get("depth_scenario_count", legacy_depth),
            successful_depth_scenario_count=data.get("successful_depth_scenario_count", 0),
            navigation_failure_count=data.get("navigation_failure_count", 0),
            status_reason=str(data.get("status_reason", "") or ""),
        )


class FlowGraph:
    """Functional Flow Graph: functionalities, pages, page transitions, and resource dependencies."""

    def __init__(self):
        self.functions: dict[str, FunctionNode] = {}
        self.pages: dict[str, Page] = {}
        self.page_transitions: list[PageTransition] = []
        self._page_transition_set: set[PageTransition] = set()
        self._transition_index: dict[tuple[str, str], list[PageTransition]] | None = None
        self.function_dependencies: list[FunctionDependency] = []
        self._function_dependency_set: set[FunctionDependency] = set()
        # Legacy [cross:...] strings remain attempt history, never a cross outcome.
        self.cross_test_results: list[dict] = []
        self.cross_test_history: list[dict] = []
        # Record actually entered FuncDroid_ test values and their producing functionality for explicit reuse in inter-functional scenarios.
        self.test_data_records: list[dict] = []

    @property
    def page_owners(self) -> dict[str, list[str]]:
        owners: dict[str, list[str]] = {}
        for node in self.functions.values():
            for page_id in node.page_ids:
                if page_id in self.pages:
                    owners.setdefault(page_id, []).append(node.id)
        return owners

    def active_nodes(self) -> list[FunctionNode]:
        return list(self.functions.values())

    def test_status_counts(self) -> dict[str, int]:
        counts = {status: 0 for status in FUNCTION_STATUSES}
        for node in self.functions.values():
            status = node.test_status or "not_started"
            if status in counts:
                counts[status] += 1
        return counts

    def active_pages(self) -> list[Page]:
        return list(self.pages.values())

    def resolve_function_id(self, node_id: str) -> str:
        node_id = str(node_id or "").strip()
        return node_id if node_id in self.functions else ""

    def resolve_page_id(self, page_id: str) -> str:
        page_id = str(page_id or "").strip()
        return page_id if page_id in self.pages else ""

    def add_node(self, node: FunctionNode):
        if not node.name:
            node.name = node.description or node.id
        self.functions[node.id] = node

    def add_page(self, page: Page, node_id: str):
        if page.id is None:
            raise ValueError("A page must have an id before being added to the FFG")
        node = self.functions.get(node_id)
        if node is None:
            raise ValueError(f"Unknown functionality node: {node_id}")
        self.pages[page.id] = page
        node.add_page_id(page.id)

    def add_entry_page(self, page: Page, node_id: str):
        if page.id is None:
            raise ValueError("A page must have an id before being added to the FFG")
        node = self.functions.get(node_id)
        if node is None:
            raise ValueError(f"Unknown functionality node: {node_id}")
        self.pages[page.id] = page
        node.add_page_id(page.id, entry_only=True)

    def remove_page_owner(self, page_id: str, node_id: str) -> None:
        """Remove one Function owner while leaving the global Page intact."""
        page_id = self.resolve_page_id(page_id)
        node = self.functions.get(str(node_id or "").strip())
        if not page_id or node is None:
            return
        node.page_ids = [value for value in node.page_ids if value != page_id]
        node.entry_page_ids = [value for value in node.entry_page_ids if value != page_id]

    def _transition_bucket(self, src_page_id: str, dst_page_id: str) -> list[PageTransition]:
        """Index transitions by (source, destination), avoiding a full scan per insertion.

        Rebuild lazily if the transition list is replaced or its size diverges from the index."""
        index = getattr(self, "_transition_index", None)
        if (
            index is None
            or getattr(self, "_transition_index_source", None) is not self.page_transitions
            or getattr(self, "_transition_index_size", -1) != len(self.page_transitions)
        ):
            index = {}
            for existing in self.page_transitions:
                index.setdefault((existing.src_page_id, existing.dst_page_id), []).append(existing)
            self._transition_index = index
            self._transition_index_source = self.page_transitions
            self._transition_index_size = len(self.page_transitions)
        return index.setdefault((src_page_id, dst_page_id), [])

    def add_page_transition(self, transition: PageTransition):
        if not transition.src_page_id or not transition.dst_page_id or not transition.operation:
            return
        bucket = self._transition_bucket(transition.src_page_id, transition.dst_page_id)
        family = operation_family_key(transition.operation)
        for existing in list(bucket):
            if operation_family_key(existing.operation) == family:
                preferred = _merge_operation_families([
                    existing.operation,
                    transition.operation,
                ])[0]
                if preferred != existing.operation:
                    self._page_transition_set.discard(existing)
                    self.page_transitions.remove(existing)
                    bucket.remove(existing)
                    replacement = PageTransition(
                        src_page_id=existing.src_page_id,
                        dst_page_id=existing.dst_page_id,
                        operation=preferred,
                    )
                    self._page_transition_set.add(replacement)
                    self.page_transitions.append(replacement)
                    bucket.append(replacement)
                return
        if transition not in self._page_transition_set:
            self._page_transition_set.add(transition)
            self.page_transitions.append(transition)
            bucket.append(transition)
            self._transition_index_size = len(self.page_transitions)

    def get_transitions_from(self, src_page_id: str) -> list[PageTransition]:
        return [
            transition
            for transition in self.page_transitions
            if transition.src_page_id == src_page_id
        ]

    def get_function_page_transitions(self, function_id: str) -> list[PageTransition]:
        node = self.get_node(function_id)
        if node is None:
            return []
        page_ids = set(node.page_ids)
        return [
            transition
            for transition in self.page_transitions
            if transition.src_page_id in page_ids and transition.dst_page_id in page_ids
        ]

    def add_function_dependency(self, edge: FunctionDependency):
        if (
            edge.src_function_id not in self.functions
            or edge.dst_function_id not in self.functions
            or not edge.resource_variable
        ):
            return
        for existing in self.function_dependencies:
            if (
                existing.src_function_id == edge.src_function_id
                and existing.dst_function_id == edge.dst_function_id
                and _is_same_text(existing.resource_variable, edge.resource_variable)
            ):
                self.functions[edge.src_function_id].add_resource_variable(edge.resource_variable)
                self.functions[edge.dst_function_id].add_resource_variable(edge.resource_variable)
                return
        self._function_dependency_set.add(edge)
        self.function_dependencies.append(edge)
        self.functions[edge.src_function_id].add_resource_variable(edge.resource_variable)
        self.functions[edge.dst_function_id].add_resource_variable(edge.resource_variable)

    def add_scenario(self, function_id: str, description: str):
        node = self.get_node(function_id)
        if node is not None:
            node.add_scenario(description)

    def add_cross_scenario(self, src_function_id: str, dst_function_id: str, description: str):
        description = str(description or "").strip()
        if not description:
            return
        marker = f"[cross:{src_function_id}->{dst_function_id}] {description}"
        self.add_scenario(src_function_id, marker)
        self.add_scenario(dst_function_id, marker)

    @staticmethod
    def cross_test_key(src_function_id: str, dst_function_id: str, resource: str) -> tuple[str, str, str]:
        return (str(src_function_id).strip(), str(dst_function_id).strip(), _text_key(resource))

    def record_cross_test(
        self, src_function_id: str, dst_function_id: str, resource: str,
        scenario: str, status: str, *, reason: str = "",
        retryable: bool = False, attempts: int | None = None,
    ) -> None:
        """Persist the outcome of a directed resource flow separately from its attempts.

        Done means the test phase finished, not that the business outcome was correct. Failed means execution did not finish. Replace the latest result but accumulate attempts for bounded retries."""
        if status not in {"done", "failed"}:
            raise ValueError(f"Unknown inter-functional test outcome: {status}")
        key = self.cross_test_key(src_function_id, dst_function_id, resource)
        previous = next((
            item for item in self.cross_test_results
            if self.cross_test_key(item["src_function_id"], item["dst_function_id"], item["resource_variable"]) == key
        ), None)
        if attempts is None:
            attempts = int((previous or {}).get("attempts", 0) or 0) + 1
        self.cross_test_results = [item for item in self.cross_test_results if item is not previous]
        self.cross_test_results.append({
            "src_function_id": src_function_id, "dst_function_id": dst_function_id,
            "resource_variable": str(resource or "").strip(), "scenario": scenario,
            "status": status, "reason": reason,
            "retryable": bool(retryable) and status != "done",
            "attempts": max(1, int(attempts or 1)),
        })

    def record_test_data(self, function_id: str, values: list[str], *, iteration: int = 0, scenario: str = "") -> int:
        """Record self-created values actually entered by a functionality, once per functionality/value pair."""
        if self.get_node(function_id) is None:
            return 0
        existing = {(item["function_id"], item["value"]) for item in self.test_data_records}
        added = 0
        for value in values:
            value = str(value or "").strip()
            if not value or (function_id, value) in existing:
                continue
            existing.add((function_id, value))
            self.test_data_records.append({
                "function_id": function_id, "value": value,
                "iteration": int(iteration or 0), "scenario": str(scenario or "").strip(),
            })
            added += 1
        return added

    def function_test_data(self, function_id: str) -> list[dict]:
        return [dict(item) for item in self.test_data_records if item["function_id"] == function_id]

    def invalidate_cross_tests(self, function_ids: set[str], reason: str) -> None:
        """Preserve evidence whose function boundary changed, without granting new coverage."""
        retained = []
        for item in self.cross_test_results:
            if {item["src_function_id"], item["dst_function_id"]} & function_ids:
                self.cross_test_history.append({**item, "invalidation_reason": reason})
            else:
                retained.append(item)
        self.cross_test_results = retained

    def cross_test_status_counts(self) -> dict[str, int]:
        return {
            status: sum(item["status"] == status for item in self.cross_test_results)
            for status in ("done", "failed")
        }

    def set_node_test_status(self, node_id: str, status: str, reason: str = ""):
        """Apply a functionality status transition; reject unknown states explicitly."""
        normalized = str(status or "").strip().lower()
        if normalized not in FUNCTION_STATUSES:
            raise ValueError(f"Unknown functionality status: {status}")
        node = self.get_node(node_id)
        if node is None:
            return
        node.test_status = normalized
        node.status_reason = str(reason or "").strip()

    def get_node_test_status(self, node_id: str) -> str:
        node = self.get_node(node_id)
        return node.test_status if node is not None else "not_started"

    def is_node_terminal(self, node_id: str) -> bool:
        node = self.get_node(node_id)
        return bool(node and node.test_status in TERMINAL_FUNCTION_STATUSES)

    def get_node_coverage_percent(self, node_id: str) -> int:
        node = self.get_node(node_id)
        return node.coverage_percent if node is not None else 0

    def get_page(self, page_id: str) -> Page | None:
        return self.pages.get(self.resolve_page_id(page_id))

    def get_page_owners(self, page_id: str) -> list[str]:
        page_id = self.resolve_page_id(page_id) or str(page_id or "").strip()
        return list(self.page_owners.get(page_id, []))

    def get_page_coverage_owners(self, page_id: str) -> list[str]:
        page_id = self.resolve_page_id(page_id) or str(page_id or "").strip()
        return [
            node.id
            for node in self.functions.values()
            if page_id in node.coverage_page_ids
        ]

    def function_controls(self, node_id: str) -> dict[tuple[str, str], str]:
        """Return every deduplicated Page control owned by one Function."""
        node = self.get_node(node_id)
        if node is None:
            return {}
        controls: dict[tuple[str, str], str] = {}
        for page_id in node.coverage_page_ids:
            page = self.get_page(page_id)
            if page is None:
                continue
            for control in page.controls:
                if not control_is_coverage_candidate(control):
                    continue
                control_key = _text_key(control)
                if control_key:
                    controls[(page_id, control_key)] = control
        return controls

    def function_handled_controls(self, node_id: str) -> set[tuple[str, str]]:
        """Return controls hit by recorded business operations.

        Page.operation_bindings is authoritative and validated against original list entries. Optional legacy matching uses operation families, rejecting substring fallback when both nonempty families differ."""
        controls = self.function_controls(node_id)
        handled: set[tuple[str, str]] = set()
        for control_ref, control in controls.items():
            page_id, _control_key = control_ref
            page = self.get_page(page_id)
            if page is None:
                continue
            if _control_has_binding(page, control):
                handled.add(control_ref)
                continue
            if COVERAGE_LEGACY_TEXT_MATCHING and _control_text_matched(page, control):
                handled.add(control_ref)
        return handled

    def function_executed_operation_keys(self, node_id: str) -> set[tuple[str, str]]:
        """Distinct business operations of one Function, deduplicated per page.

        Repeating the same operation must not raise coverage, otherwise a single
        control clicked many times can push a Function to the completion gate
        while most of its controls were never touched.
        """
        node = self.get_node(node_id)
        if node is None:
            return set()
        executed: set[tuple[str, str]] = set()
        for page_id in node.coverage_page_ids:
            page = self.get_page(page_id)
            if page is None:
                continue
            resolved_page_id = self.resolve_page_id(page_id) or str(page_id or "").strip()
            for operation in page.operations:
                if operation_is_recovery(operation):
                    continue
                operation_key = operation_family_key(operation) or _text_key(operation)
                if operation_key:
                    executed.add((resolved_page_id, operation_key))
        return executed

    def function_coverage_details(self, node_id: str) -> dict[str, object]:
        """Compute coverage from distinct controls hit, not raw operation counts.

        Repeatedly operating one control must not satisfy coverage while other controls remain untouched."""
        controls = self.function_controls(node_id)
        total = len(controls)
        directly_handled = self.function_handled_controls(node_id)
        handled_count = len(directly_handled)
        unmatched_controls = [
            label for ref, label in controls.items() if ref not in directly_handled
        ]
        return {
            "total_controls": total,
            "executed_operations": handled_count,
            "coverage_percent": adjusted_control_coverage(total, handled_count),
            "uncovered_controls": unmatched_controls,
        }

    def recompute_function_coverage(self, node_id: str) -> int:
        node = self.get_node(node_id)
        if node is None:
            return 0
        coverage = int(self.function_coverage_details(node_id)["coverage_percent"])
        node.coverage_percent = coverage
        return coverage

    def recompute_all_coverages(self) -> None:
        for node_id in list(self.functions):
            self.recompute_function_coverage(node_id)

    def function_has_successful_business_operation(self, node_id: str) -> bool:
        node = self.get_node(node_id)
        if node is None or not node.scenarios:
            return False
        for page_id in node.coverage_page_ids:
            page = self.get_page(page_id)
            if page is None:
                continue
            for operation in page.operations:
                if operation_is_failure(operation) or operation_is_recovery(operation):
                    continue
                operation_key = operation_family_key(operation)
                action, _, target_key = operation_key.partition(":")
                if action in {"open", "back"}:
                    continue
                if target_key:
                    return True
        return False

    def reconcile_page_membership(self):
        active_page_ids = set(self.pages)
        for node in self.functions.values():
            node.page_ids = list(dict.fromkeys(
                page_id for page_id in node.page_ids if page_id in active_page_ids
            ))
            node.entry_page_ids = list(dict.fromkeys(
                page_id
                for page_id in node.entry_page_ids
                if page_id in node.page_ids
            ))
        retained_transitions = [
            transition
            for transition in self.page_transitions
            if transition.src_page_id in active_page_ids and transition.dst_page_id in active_page_ids
        ]
        self.page_transitions = retained_transitions
        self._page_transition_set = set(retained_transitions)
        retained_dependencies = [
            dependency
            for dependency in self.function_dependencies
            if dependency.src_function_id in self.functions and dependency.dst_function_id in self.functions
        ]
        self.function_dependencies = retained_dependencies
        self._function_dependency_set = set(retained_dependencies)
        for page in self.pages.values():
            page.controls = _merge_exact_texts(page.controls)
            page._remerge_operations()
            page.operation_bindings = _normalize_operation_bindings(
                page.operation_bindings,
                page.operations,
                page.controls,
            )
        rebuilt_transitions = list(self.page_transitions)
        self.page_transitions = []
        self._page_transition_set = set()
        for transition in rebuilt_transitions:
            self.add_page_transition(transition)
        self.recompute_all_coverages()

    def get_node(self, node_id: str) -> FunctionNode | None:
        return self.functions.get(self.resolve_function_id(node_id))

    def update_function(self, node_id: str, fields: dict) -> bool:
        node = self.get_node(node_id)
        if node is None or not isinstance(fields, dict):
            return False
        changed = False
        for key in ("name", "description"):
            value = str(fields.get(key, "") or "").strip()
            if value and getattr(node, key) != value:
                setattr(node, key, value)
                changed = True
        if isinstance(fields.get("resource_variables"), list):
            before = list(node.resource_variables)
            for value in fields["resource_variables"]:
                node.add_resource_variable(str(value))
            changed = changed or before != node.resource_variables
        return changed

    def update_page(self, page_id: str, fields: dict) -> bool:
        page = self.get_page(page_id)
        if page is None or not isinstance(fields, dict):
            return False
        changed = False
        for key in ("name", "description", "activity"):
            value = str(fields.get(key, "") or "").strip()
            if value and getattr(page, key) != value:
                setattr(page, key, value)
                changed = True
        if isinstance(fields.get("controls"), list):
            before = list(page.controls)
            for control in fields["controls"]:
                page.add_control(str(control))
            changed = changed or before != page.controls
        return changed

    def clone(self) -> FlowGraph:
        """Copy mutable graph structure while sharing immutable screenshot strings.

        Unlike a serialization round trip, this avoids text merging, transition rebuilding, and image decoding. The copy can be mutated independently."""
        graph = FlowGraph()
        for node_id, node in self.functions.items():
            graph.functions[node_id] = FunctionNode(
                id=node.id,
                name=node.name,
                description=node.description,
                page_ids=list(node.page_ids),
                entry_page_ids=list(node.entry_page_ids),
                resource_variables=list(node.resource_variables),
                scenarios=list(node.scenarios),
                test_status=node.test_status,
                coverage_percent=node.coverage_percent,
                intra_attempt_count=node.intra_attempt_count,
                depth_scenario_count=node.depth_scenario_count,
                successful_depth_scenario_count=node.successful_depth_scenario_count,
                navigation_failure_count=node.navigation_failure_count,
                status_reason=node.status_reason,
            )
        for page_id, page in self.pages.items():
            copied = Page.__new__(Page)
            copied.id = page.id
            copied.name = page.name
            copied.description = page.description
            copied.representative_screenshot = page.representative_screenshot
            copied.evidence_screenshots = list(page.evidence_screenshots)
            copied.controls = list(page.controls)
            copied.operations = list(page.operations)
            copied.operation_bindings = [dict(binding) for binding in page.operation_bindings]
            copied.activity = page.activity
            graph.pages[page_id] = copied
        graph.page_transitions = list(self.page_transitions)
        graph._page_transition_set = set(self._page_transition_set)
        graph.function_dependencies = list(self.function_dependencies)
        graph._function_dependency_set = set(self._function_dependency_set)
        graph.cross_test_results = [dict(item) for item in self.cross_test_results]
        graph.cross_test_history = [dict(item) for item in self.cross_test_history]
        graph.test_data_records = [dict(item) for item in self.test_data_records]
        return graph

    def to_dict(self, screenshots_dir: str | None = None) -> dict:
        return {
            "functions": [node.to_dict() for node in self.functions.values()],
            "pages": [
                page.to_dict(screenshots_dir=screenshots_dir)
                for page in self.pages.values()
            ],
            "page_transitions": [edge.to_dict() for edge in self.page_transitions],
            "function_dependencies": [edge.to_dict() for edge in self.function_dependencies],
            "cross_test_results": [dict(item) for item in self.cross_test_results],
            "cross_test_history": [dict(item) for item in self.cross_test_history],
            "test_data_records": [dict(item) for item in self.test_data_records],
        }

    @classmethod
    def from_dict(cls, data: dict, screenshots_dir: str | None = None) -> FlowGraph:
        graph = cls()
        function_items = data.get("functions", [])
        if not isinstance(function_items, list):
            function_items = []
        for node_data in function_items:
            if not isinstance(node_data, dict):
                continue
            node = FunctionNode.from_dict(node_data)
            if node.id:
                graph.add_node(node)

        page_items = data.get("pages", [])
        if not isinstance(page_items, list):
            page_items = []
        for page_data in page_items:
            if not isinstance(page_data, dict):
                continue
            page = Page.from_dict(page_data, screenshots_dir=screenshots_dir)
            if page.id:
                graph.pages[page.id] = page

        for item in data.get("page_transitions", []):
            if isinstance(item, dict):
                graph.add_page_transition(PageTransition.from_dict(item))

        for item in data.get("function_dependencies", []):
            if isinstance(item, dict):
                graph.add_function_dependency(FunctionDependency.from_dict(item))
        graph.reconcile_page_membership()
        graph.cross_test_history = [dict(item) for item in data.get("cross_test_history", []) or [] if isinstance(item, dict)]
        for item in data.get("cross_test_results", []) or []:
            if not isinstance(item, dict):
                continue
            src, dst = item.get("src_function_id"), item.get("dst_function_id")
            if src not in graph.functions or dst not in graph.functions or src == dst:
                graph.cross_test_history.append({**item, "invalidation_reason": "inactive function reference on load"})
                continue
            # Legacy verified/unverified statuses both indicate the test phase finished.
            status = {"verified": "done", "unverified": "done"}.get(item.get("status"), item.get("status"))
            if status not in {"done", "failed"}:
                continue
            graph.record_cross_test(src, dst, item.get("resource_variable", ""),
                                    item.get("scenario", ""), status,
                                    reason=item.get("reason", ""),
                                    retryable=bool(item.get("retryable", False)),
                                    attempts=int(item.get("attempts", 1) or 1))
        for item in data.get("test_data_records", []) or []:
            if isinstance(item, dict):
                graph.record_test_data(
                    str(item.get("function_id", "") or ""), [item.get("value", "")],
                    iteration=int(item.get("iteration", 0) or 0), scenario=item.get("scenario", ""),
                )
        return graph

    def save(self, output_dir: str):
        self.recompute_all_coverages()
        os.makedirs(output_dir, exist_ok=True)
        screenshots_dir = os.path.join(output_dir, "screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)
        serialized = json.dumps(
            self.to_dict(screenshots_dir=screenshots_dir),
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")
        _write_bytes_if_changed(os.path.join(output_dir, "ffg.json"), serialized)
        print(f"[Persistence] FFG saved to {output_dir}")
        print(
            f"  Functionality nodes: {len(self.functions)}, Pages: {len(self.pages)}, "
            f"Page transitions: {len(self.page_transitions)}, Resource dependencies: {len(self.function_dependencies)}"
        )

    @classmethod
    def load(cls, output_dir: str) -> FlowGraph:
        screenshots_dir = os.path.join(output_dir, "screenshots")
        if not os.path.exists(screenshots_dir):
            screenshots_dir = None
        with open(os.path.join(output_dir, "ffg.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("The root of ffg.json must be an object")
        for key in ("functions", "pages", "page_transitions", "function_dependencies"):
            if not isinstance(data.get(key), list):
                raise ValueError(f"Missing or invalid ffg.json structure field: {key}")
        graph = cls.from_dict(data, screenshots_dir=screenshots_dir)
        print(f"[Persistence] FFG loaded from {output_dir}")
        print(
            f"  Functionality nodes: {len(graph.functions)}, Pages: {len(graph.pages)}, "
            f"Page transitions: {len(graph.page_transitions)}, Resource dependencies: {len(graph.function_dependencies)}"
        )
        return graph


# Paper terminology aliases preserve existing imports and serialized formats.
FunctionalFlowGraph = FlowGraph
FunctionalityNode = FunctionNode
