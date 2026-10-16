"""后台页面控件审计 worker。

设计动机：
- Page.controls 的补充目前依赖 FFG 更新 LLM 在复用页面时自觉声明，小参数
  模型几乎不会主动补充，形成"首次识别遗漏 → 永远没人补"的自锁循环。
  本 worker 用独立后台线程定期对页面截图做一次控件清单复核，把"补充
  controls"变成程序主动触发的确定性机制。
- 审计只挖掘已有证据（页面已保存截图），不产生新执行；输入是截图与
  现有清单，输出是"截图中真实可见但清单遗漏的可交互控件"。
- 与 bug_detection_worker 同构：后台线程只做 LLM 调用（算），不直接写
  FFG；合并动作由主线程在 ffg_update 节点入口完成（写），保持单写者。
- 退出条件：页面截图集合指纹未变化，或连续 DEFAULT_ZERO_ADD_LIMIT
  次审计零新增（且该页全部截图都已审计过），则标记 saturated 不再审计；
  连续 DEFAULT_FAILURE_LIMIT 次调用失败同样停止重试，直到页面指纹变化。
  已提交未完成的任务按 (page_id, fingerprint) 登记在途，不会被重复提交。
  状态仅存进程内，重启清零。

用法：
    from workflow.page_audit_worker import page_audit_worker

    page_audit_worker.configure(llm)
    page_audit_worker.submit(job)      # job 含 page_id 与已构建好的消息
    ...                                # ffg_update 节点入口：
    for job in page_audit_worker.collect_results(): ...
    page_audit_worker.record_result(page_id, added_count)
    page_audit_worker.flush(timeout)   # 运行结束前等待排空
"""

from __future__ import annotations

import json
import os
from typing import Any

from workflow.llm import invoke_llm_json_with_retry, llm_json_result_failed
from workflow.worker_utils import SerialWorker


DEFAULT_BATCH_SIZE = 3
DEFAULT_ZERO_ADD_LIMIT = 2
DEFAULT_PER_PAGE_LIMIT = 5
# 同一指纹连续失败达到该次数后不再重试（请求过大、模型持续输出非 JSON 等）；
# 页面截图或操作变化（指纹变化）后自动解除。
DEFAULT_FAILURE_LIMIT = max(1, int(os.getenv("PAGE_AUDIT_FAILURE_LIMIT", "2")))
# 审计在后台线程里执行，失败由 worker 自己的失败计数兜底，单次调用不重试。
PAGE_AUDIT_LLM_ATTEMPTS = max(1, int(os.getenv("PAGE_AUDIT_LLM_ATTEMPTS", "1")))


class PageAuditWorker(SerialWorker):
    """顺序执行页面控件审计 LLM 调用；队列满时丢弃（页面下轮会被重新挑选）。"""

    log_prefix = "[页面审计线程]"
    thread_name = "page-audit-worker"
    default_max_pending = 4
    drop_when_full = True

    def __init__(self) -> None:
        super().__init__()
        self._results: list[dict[str, Any]] = []
        # page_id -> {"fingerprint": str, "zero_streak": int, "saturated": bool,
        #             "inflight": str, "failed_fingerprint": str, "fail_streak": int,
        #             "audited_screenshots": set[str], "pending_screenshots": int}
        self._audit_state: dict[str, dict[str, Any]] = {}

    def configure(self, llm) -> None:
        self._configure_base(llm)

    # ------------------------------------------------------------------
    # 提交任务（主线程调用）
    # ------------------------------------------------------------------
    def filter_candidates(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """按饱和/指纹状态过滤候选页面，返回允许提交的子集。

        每个 candidate: {"page_id": str, "fingerprint": str, ...}。
        跳过：已 saturated；指纹与上次成功审计相同（截图集合未变化）；同一指纹
        仍在队列或执行中；同一指纹连续失败已达上限。
        """
        allowed: list[dict[str, Any]] = []
        with self._lock:
            for candidate in candidates:
                page_id = str(candidate.get("page_id", "") or "")
                fingerprint = str(candidate.get("fingerprint", ""))
                state = self._audit_state.get(page_id)
                if state is None:
                    allowed.append(candidate)
                    continue
                if state.get("saturated"):
                    continue
                if state.get("inflight") == fingerprint:
                    continue
                if state.get("fingerprint") == fingerprint and not state.get("pending_screenshots"):
                    continue
                if (
                    state.get("failed_fingerprint") == fingerprint
                    and int(state.get("fail_streak", 0) or 0) >= DEFAULT_FAILURE_LIMIT
                ):
                    continue
                allowed.append(candidate)
        return allowed

    def submit(self, job: dict[str, Any], timeout: float | None = None) -> bool:
        # 入队前登记在途：后台线程可能在 super().submit 返回前就处理完任务，
        # 事后登记会留下永不清除的 inflight，使该页面再也不会被审计。
        page_id = str(job.get("page_id", "") or "")
        fingerprint = str(job.get("fingerprint", "") or "")
        with self._lock:
            state = self._audit_state.setdefault(page_id, {})
            previous = state.get("inflight")
            state["inflight"] = fingerprint
        ok = super().submit(job, timeout=timeout)
        if not ok:
            with self._lock:
                state = self._audit_state.setdefault(page_id, {})
                if state.get("inflight") == fingerprint:
                    if previous:
                        state["inflight"] = previous
                    else:
                        state.pop("inflight", None)
        return ok

    def audited_screenshot_digests(self, page_id: str) -> set[str]:
        with self._lock:
            state = self._audit_state.get(str(page_id or ""), {})
            return set(state.get("audited_screenshots", set()))

    def _process_job(self, job: dict[str, Any]) -> None:
        page_id = str(job.get("page_id", "") or "")
        fingerprint = str(job.get("fingerprint", "") or "")
        messages = job.get("messages") or []
        with self._lock:
            llm = self._llm
        if llm is None or not messages:
            self._record_failure(page_id, fingerprint)
            return
        payload = invoke_llm_json_with_retry(
            llm,
            messages,
            log_prefix="[页面审计]",
            max_attempts=PAGE_AUDIT_LLM_ATTEMPTS,
        )
        if llm_json_result_failed(payload):
            print(f"[页面审计线程] LLM 调用或解析失败（page={page_id}），稍后按失败计数重试")
            self._record_failure(page_id, fingerprint)
            return
        missing, control_merges, invalid_operations = self._clean_audit_payload(payload)
        with self._lock:
            self._results.append({
                "page_id": page_id,
                "fingerprint": fingerprint,
                "missing_controls": missing,
                "control_merges": control_merges,
                "invalid_operations": invalid_operations,
            })
            # 指纹记账放在结果里，等主线程合并后 record_result 决定饱和。
            state = self._audit_state.setdefault(page_id, {})
            state["fingerprint"] = fingerprint
            if state.get("inflight") == fingerprint:
                state.pop("inflight", None)
            state["fail_streak"] = 0
            state.pop("failed_fingerprint", None)
            audited = set(state.get("audited_screenshots", set()))
            audited.update(str(item) for item in job.get("screenshot_digests", []) or [])
            state["audited_screenshots"] = audited
            state["pending_screenshots"] = int(job.get("pending_screenshots", 0) or 0)
        print(
            f"[页面审计线程] 页面清单复核完成: page={page_id}, "
            f"候选遗漏控件={len(missing)}, 候选待删操作={len(invalid_operations)}"
            f", 候选控件合并={len(control_merges)}"
        )

    def _clean_audit_payload(
        self,
        payload: dict[str, Any],
    ) -> tuple[list[str], list[dict[str, Any]], list[str]]:
        missing = self._clean_text_list(payload.get("missing_controls"))
        control_merges = self._clean_control_merges(payload.get("control_merges"))
        invalid_operations = self._clean_text_list(payload.get("invalid_operations"))
        return (
            missing[:DEFAULT_PER_PAGE_LIMIT],
            control_merges[:DEFAULT_PER_PAGE_LIMIT],
            invalid_operations[:DEFAULT_PER_PAGE_LIMIT],
        )

    def _parse_audit_payload_with_status(
        self,
        content: str,
    ) -> tuple[list[str], list[dict[str, Any]], list[str], bool]:
        """从文本解析审计结果（兼容入口；后台线程走 invoke_llm_json_with_retry）。"""
        text = str(content or "").strip()
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return [], [], [], False
        try:
            payload = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return [], [], [], False
        if not isinstance(payload, dict):
            return [], [], [], False
        return (*self._clean_audit_payload(payload), True)

    @staticmethod
    def _clean_control_merges(raw: object) -> list[dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        cleaned: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            sources = PageAuditWorker._clean_text_list(item.get("sources"))
            target = str(item.get("target", "") or "").strip()
            if len(sources) >= 2 and target:
                cleaned.append({"sources": sources, "target": target})
        return cleaned

    @staticmethod
    def _clean_text_list(raw: object) -> list[str]:
        if not isinstance(raw, list):
            return []
        cleaned: list[str] = []
        for item in raw:
            value = str(item or "").strip()
            if value and value not in cleaned:
                cleaned.append(value)
        return cleaned

    def _record_failure(self, page_id: str, fingerprint: str = "") -> None:
        """失败审计保持可重试，但同一指纹连续失败达到上限后暂停重试。

        不改变零新增计数；成功审计过的旧指纹作废，避免页面看起来已审计。
        """
        with self._lock:
            state = self._audit_state.setdefault(str(page_id or ""), {})
            state["fingerprint"] = ""
            if fingerprint and state.get("inflight") == fingerprint:
                state.pop("inflight", None)
            if fingerprint and state.get("failed_fingerprint") == fingerprint:
                state["fail_streak"] = int(state.get("fail_streak", 0) or 0) + 1
            else:
                state["failed_fingerprint"] = fingerprint
                state["fail_streak"] = 1
            if fingerprint and state["fail_streak"] >= DEFAULT_FAILURE_LIMIT:
                print(
                    f"[页面审计线程] 页面同一状态连续 {state['fail_streak']} 次审计失败，"
                    f"暂停重试直到页面变化: page={page_id}"
                )

    # ------------------------------------------------------------------
    # 结果读取 / 状态回报（主线程调用）
    # ------------------------------------------------------------------
    def collect_results(self) -> list[dict[str, Any]]:
        """取走已完成的审计结果（不改变饱和状态，等 record_result）。"""
        with self._lock:
            results = list(self._results)
            self._results = []
        return results

    def record_result(self, page_id: str, added: int) -> None:
        """主线程合并完成后回报实际新增数，驱动饱和判定。"""
        with self._lock:
            state = self._audit_state.setdefault(str(page_id or ""), {})
            streak = int(state.get("zero_streak", 0) or 0)
            streak = 0 if added > 0 else streak + 1
            state["zero_streak"] = streak
            # 还有截图没审计过时不判饱和：否则被单次上限省略的状态永远补不进来。
            if streak >= DEFAULT_ZERO_ADD_LIMIT and not state.get("pending_screenshots"):
                state["saturated"] = True
                print(
                    f"[页面审计线程] 页面连续 {DEFAULT_ZERO_ADD_LIMIT} 次审计无新增，"
                    f"标记饱和不再审计: page={page_id}"
                )

    def is_configured(self) -> bool:
        """Return whether final-output code may submit additional audits."""
        with self._lock:
            return self._llm is not None

    def record_failure(self, job: dict[str, Any], exc: BaseException) -> None:
        job = job or {}
        self._record_failure(
            str(job.get("page_id", "") or ""),
            str(job.get("fingerprint", "") or ""),
        )

    def reset(self) -> None:
        super().reset()
        with self._lock:
            self._results = []
            self._audit_state = {}


# 进程内单例，语义与 workflow/bug_detection_worker.py 一致。
page_audit_worker = PageAuditWorker()
