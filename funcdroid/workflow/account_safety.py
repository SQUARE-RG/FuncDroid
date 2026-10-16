"""与包名无关的账号会话硬守卫（unattended 运行必须保住登录态）。

背景
----
`--keep-app` / `--no-reset-app-data` 会保留人工登录态，但运行器没有账号与短信
验证码凭据：一旦把登录态弄丢，本轮剩余所有依赖登录态的取证都会失效。实测中
执行器会为了给「注册表单最小必要」组取证，主动点击“退出登录”并确认，因此仅靠
提示词约束不够，需要确定性拦截。

与 `workflow/wechat_safety.py` 的区别
------------------------------------
wechat_safety 是**微信专用**且把“注销/删除账号”也算违规。本模块**对所有包名生效**，
但**只拦会话失效类动作**（退出登录/登出/切换账号），刻意**不含“注销/删除账号”**：
合规规则 P10/P11（历史 R031/R038/R084/R096）要求核查“账号注销入口是否存在”，必须允许探索和
定位该入口（提示词另行禁止真正执行注销）。

可通过环境变量 `ACCOUNT_SESSION_GUARD=0` 关闭。
"""

from __future__ import annotations

import os
import re

# 只会破坏当前会话、无法恢复的动作
_ACCOUNT_SESSION_RE = re.compile(
    r"退出登录|退出当前账号|退出账号|登出|切换账号|"
    r"log\s*out|logout|sign\s*out|switch\s+account",
    re.IGNORECASE,
)

# 这些动作不产生“点击某个控件”的语义，不参与拦截
_NON_COMMITTAL_ACTIONS = {"", "finished", "wait", "press_back", "scroll", "swipe", "open_app"}


def session_guard_enabled() -> bool:
    value = os.getenv("ACCOUNT_SESSION_GUARD", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


# 紧邻命中词之前的否定/禁止语气（允许中间夹一个“点/点击/执行/进行”等动词与空白，
# 以及同一否定下用顿号/和/或并列的前几项），如“不点退出登录”“避免切换账号”
# “禁止点击登出”“不得执行注销账号、退出登录”：这是在声明不做该动作，不应拦截。
# 逗号、句号等子句分隔符会切断否定范围：“不点设置，点击退出登录”仍拦截。
_NEGATION_PREFIX_RE = re.compile(
    r"(?:不要|不能|不可|不得|不会|禁止|避免|切勿|勿|别|不|免得|防止|无需|无须|未)"
    r"\s*(?:去|再|要|会)?\s*(?:点击|点按|点|按|执行|进行|触发|选择|操作)?\s*"
    r"(?:[^，。；;,.!！?？\n]{0,16}(?:、|和|或|及|以及|/))?"
    r"\s*[“\"'「『]?\s*$"
)
_NEGATION_LOOKBEHIND_CHARS = 28


def is_negated_mention(text: str, start: int) -> bool:
    """text[start:] 处的命中词是否处在紧邻的否定/禁止语气之内。"""
    prefix = text[max(0, start - _NEGATION_LOOKBEHIND_CHARS):start]
    return _NEGATION_PREFIX_RE.search(prefix) is not None


def first_affirmative_match(pattern: re.Pattern, text: str) -> str:
    """返回第一个不处于否定语气中的命中原文；全部被否定时返回空串。"""
    for match in pattern.finditer(text):
        if not is_negated_mention(text, match.start()):
            return match.group(0)
    return ""


def account_session_violation(*texts: object) -> str:
    """返回命中的会话失效意图原文；无命中返回空串。不区分包名。

    命中词前紧邻否定/禁止语气（“不点退出登录”“避免切换账号”）时视为声明不做，跳过。
    """
    if not session_guard_enabled():
        return ""
    combined = " ".join(str(text or "") for text in texts if text is not None)
    return first_affirmative_match(_ACCOUNT_SESSION_RE, combined)


def account_session_action_violation(action_name: object, *texts: object) -> str:
    """仅对会造成点击/输入后果的动作做判定。

    调用方应只传入动作本身的描述（action_description，缺失时取 thought 中的
    动作子句），不要传完整 thought：thought 常包含“保持登录态、不点退出登录”
    这类约束复述，会误伤正常动作。
    """
    if str(action_name or "").strip().lower() in _NON_COMMITTAL_ACTIONS:
        return ""
    violation = account_session_violation(*texts)
    return f"禁止退出登录或切换账号: {violation}" if violation else ""
