"""Package-independent account-session protection for unattended execution.

The runner cannot restore credentials or verification codes after logout. Guard session-invalidating actions deterministically. Unlike the package-specific account-deletion guard, this module permits locating deletion settings without executing them. Disable with ACCOUNT_SESSION_GUARD=0."""

# Unicode escapes preserve literal UI labels and legacy evidence matching.
from __future__ import annotations

import os
import re

# Irrecoverable session-invalidating actions.
_ACCOUNT_SESSION_RE = re.compile(
    r"\u9000\u51fa\u767b\u5f55|\u9000\u51fa\u5f53\u524d\u8d26\u53f7|\u9000\u51fa\u8d26\u53f7|\u767b\u51fa|\u5207\u6362\u8d26\u53f7|"
    r"log\s*out|logout|sign\s*out|switch\s+account",
    re.IGNORECASE,
)

# These actions cannot click a control and are excluded from the guard.
_NON_COMMITTAL_ACTIONS = {"", "finished", "wait", "press_back", "scroll", "swipe", "open_app"}


def session_guard_enabled() -> bool:
    value = os.getenv("ACCOUNT_SESSION_GUARD", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


# Recognize adjacent prohibitions and coordinated lists. Clause punctuation ends the negation scope so a later affirmative action is still blocked.
_NEGATION_PREFIX_RE = re.compile(
    r"(?:\u4e0d\u8981|\u4e0d\u80fd|\u4e0d\u53ef|\u4e0d\u5f97|\u4e0d\u4f1a|\u7981\u6b62|\u907f\u514d|\u5207\u52ff|\u52ff|\u522b|\u4e0d|\u514d\u5f97|\u9632\u6b62|\u65e0\u9700|\u65e0\u987b|\u672a)"
    r"\s*(?:\u53bb|\u518d|\u8981|\u4f1a)?\s*(?:\u70b9\u51fb|\u70b9\u6309|\u70b9|\u6309|\u6267\u884c|\u8fdb\u884c|\u89e6\u53d1|\u9009\u62e9|\u64cd\u4f5c)?\s*"
    r"(?:[^，。；;,.!！?？\n]{0,16}(?:、|\u548c|\u6216|\u53ca|\u4ee5\u53ca|/))?"
    r"\s*[“\"'「『]?\s*$"
)
_ENGLISH_NEGATION_PREFIX_RE = re.compile(
    r"(?:do not|don['’]t|never|avoid|must not|cannot|will not|without)\s+"
    r"(?:(?:click|tap|press|select|perform|execute)\s+)?(?:[\w -]+\s+(?:or|and)\s+)?[\"']?$",
    re.IGNORECASE,
)
_NEGATION_LOOKBEHIND_CHARS = 80


def is_negated_mention(text: str, start: int) -> bool:
    """Return whether a match at start is immediately governed by a negation."""
    prefix = text[max(0, start - _NEGATION_LOOKBEHIND_CHARS):start]
    return bool(_NEGATION_PREFIX_RE.search(prefix) or _ENGLISH_NEGATION_PREFIX_RE.search(prefix))


def first_affirmative_match(pattern: re.Pattern, text: str) -> str:
    """Return the first non-negated matching substring, or an empty string."""
    for match in pattern.finditer(text):
        if not is_negated_mention(text, match.start()):
            return match.group(0)
    return ""


def account_session_violation(*texts: object) -> str:
    """Return a non-negated session-invalidating intent, independent of package."""
    if not session_guard_enabled():
        return ""
    combined = " ".join(str(text or "") for text in texts if text is not None)
    return first_affirmative_match(_ACCOUNT_SESSION_RE, combined)


def account_session_action_violation(action_name: object, *texts: object) -> str:
    """Check only actions capable of clicking or entering text.

    Pass the action description, not full reasoning that may merely restate a prohibition."""
    if str(action_name or "").strip().lower() in _NON_COMMITTAL_ACTIONS:
        return ""
    violation = account_session_violation(*texts)
    return f"Logout or account switching is forbidden: {violation}" if violation else ""
