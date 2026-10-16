"""Minimal hard safety guard for unattended WeChat exploration."""

from __future__ import annotations

import re

from workflow.account_safety import first_affirmative_match


WECHAT_PACKAGE = "com.tencent.mm"

# Keep the runtime guard intentionally small. Exploration breadth and prioritization
# belong to the model prompts; code only prevents losing access to the test account.
_WECHAT_ACCOUNT_DESTRUCTIVE_RE = re.compile(
    r"退出登录|登出(?:微信|账号|账户)?|"
    r"注销(?:微信|账号|账户)|删除(?:微信)?(?:账号|账户)|永久删除(?:账号|账户)|"
    r"log\s*out|sign\s*out|delete\s+(?:the\s+)?account|close\s+(?:the\s+)?account",
    re.IGNORECASE,
)


def wechat_read_only_violation(package_name: str, *texts: object) -> str:
    """Return only logout/account-deletion intent; all other actions are allowed."""
    if str(package_name or "").strip() != WECHAT_PACKAGE:
        return ""
    combined = " ".join(str(text or "") for text in texts if text is not None)
    # “不得执行注销账号、退出登录”这类约束复述不是动作意图，跳过否定语气中的命中。
    return first_affirmative_match(_WECHAT_ACCOUNT_DESTRUCTIVE_RE, combined)


def wechat_action_policy_violation(
    package_name: str,
    action_name: str,
    action_text: object = "",
) -> str:
    """Apply the same minimal account guard to any proposed action."""
    violation = wechat_read_only_violation(package_name, action_text)
    return f"禁止退出登录或删除账户: {violation}" if violation else ""
