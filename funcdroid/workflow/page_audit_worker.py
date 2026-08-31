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
- 退出条件：页面截图集合指纹未变化，或连续 PAGE_AUDIT_ZERO_ADD_LIMIT
  次审计零新增，则标记 saturated 不再审计；状态仅存进程内，重启清零。

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
import queue
import threading
import time
import traceback
from typing import Any

from workflow.llm_usage import record_llm_failure, record_llm_request, record_llm_response


DEFAULT_MAX_PENDING = max(1, int(os.getenv("PAGE_AUDIT_MAX_PENDING", "4")))
DEFAULT_BATCH_SIZE = max(1, int(os.getenv("PAGE_AUDIT_BATCH_SIZE", "3")))
DEFAULT_ZERO_ADD_LIMIT = max(1, int(os.getenv("PAGE_AUDIT_ZERO_ADD_LIMIT", "2")))
DEFAULT_PER_PAGE_LIMIT = max(1, int(os.getenv("PAGE_AUDIT_PER_PAGE_LIMIT", "5")))
RESULT_PATTERN = json.JSONDecoder()


class PageAuditWorker:
    """单一后台线程 + 队列，顺序执行页面控件审计 LLM 调用。"""

    def __init__(self, max_pending: int = DEFAULT_MAX_PENDING) -> None:
        self._lock = threading.Lock()
        self._queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._llm = None
        self._pending = 0
        self._idle_cond = threading.Condition(self._lock)
        self._max_pending = max(1, int(max_pending))
        self._results: list[dict[str, Any]] = []
        # page_id -> {"fingerprint": str, "zero_streak": int, "saturated": bool}
        self._audit_state: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # 配置 / 生命周期
    # ------------------------------------------------------------------
    def configure(self, llm) -> None:
        with self._lock:
            self._llm = llm
        self._ensure_worker_started()

    def _ensure_worker_started(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            thread = threading.Thread(
                target=self._worker_loop,
                name="page-audit-worker",
                daemon=True,
            )
            self._thread = thread
        thread.start()

    # ------------------------------------------------------------------
    # 提交任务（主线程调用）
    # ------------------------------------------------------------------
    def filter_candidates(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """按饱和/指纹状态过滤候选页面，返回允许提交的子集。

        每个 candidate: {"page_id": str, "fingerprint": str, ...}。
        指纹与上次审计相同（截图集合未变化）或已 saturated 的候选被跳过。
        """
        allowed: list[dict[str, Any]] = []
        with self._lock:
            for candidate in candidates:
                page_id = str(candidate.get("page_id", "") or "")
                state = self._audit_state.get(page_id)
                if state is None:
                    allowed.append(candidate)
                    continue
                if state.get("saturated"):
                    continue
                if state.get("fingerprint") == str(candidate.get("fingerprint", "")):
                    continue
                allowed.append(candidate)
        return allowed

    def submit(self, job: dict[str, Any]) -> bool:
        """非阻塞提交单个审计任务；队列饱和时丢弃并返回 False。"""
        self._ensure_worker_started()
        with self._lock:
            if self._pending >= self._max_pending:
                return False
            self._pending += 1
        self._queue.put(job)
        return True

    # ------------------------------------------------------------------
    # 后台线程主循环
    # ------------------------------------------------------------------
    def _worker_loop(self) -> None:
        while True:
            job = self._queue.get()
            try:
                self._process_job(job)
            except Exception as exc:  # noqa: BLE001 - 后台线程需要吞掉异常继续消费
                print(f"[页面审计线程] 处理任务失败: {exc}")
                print(traceback.format_exc())
            finally:
                with self._idle_cond:
                    self._pending = max(0, self._pending - 1)
                    if self._pending == 0:
                        self._idle_cond.notify_all()

    def _process_job(self, job: dict[str, Any]) -> None:
        page_id = str(job.get("page_id", "") or "")
        fingerprint = str(job.get("fingerprint", "") or "")
        messages = job.get("messages") or []
        with self._lock:
            llm = self._llm
        if llm is None or not messages:
            self._record_failure(page_id)
            return
        record_llm_request(llm, operation="[页面审计]")
        try:
            response = llm.invoke(messages)
        except Exception as exc:  # noqa: BLE001
            record_llm_failure(llm, operation="[页面审计]")
            print(f"[页面审计线程] LLM 调用失败（page={page_id}）: {exc}")
            self._record_failure(page_id)
            return
        content = str(getattr(response, "content", "") or "")
        record_llm_response(llm, response, operation="[页面审计]")
        missing, control_merges, invalid_operations, parse_ok = self._parse_audit_payload_with_status(content)
        if not parse_ok:
            print(f"[页面审计线程] LLM 返回格式无效（page={page_id}），保留任务以便重试")
            self._record_failure(page_id)
            return
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
        print(
            f"[页面审计线程] 页面清单复核完成: page={page_id}, "
            f"候选遗漏控件={len(missing)}, 候选待删操作={len(invalid_operations)}"
            f", 候选控件合并={len(control_merges)}"
        )

    def _parse_audit_payload(self, content: str) -> tuple[list[str], list[str]]:
        missing, _control_merges, invalid_operations, _parse_ok = self._parse_audit_payload_with_status(content)
        return missing, invalid_operations

    def _parse_audit_payload_with_status(
        self,
        content: str,
    ) -> tuple[list[str], list[dict[str, Any]], list[str], bool]:
        text = str(content or "").strip()
        if not text:
            return [], [], [], False
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return [], [], [], False
        try:
            payload = RESULT_PATTERN.decode(text[start:end + 1])
        except json.JSONDecodeError:
            return [], [], [], False
        if not isinstance(payload, dict):
            return [], [], [], False
        missing = self._clean_text_list(payload.get("missing_controls"))
        control_merges = self._clean_control_merges(payload.get("control_merges"))
        invalid_operations = self._clean_text_list(payload.get("invalid_operations"))
        return (
            missing[:DEFAULT_PER_PAGE_LIMIT],
            control_merges[:DEFAULT_PER_PAGE_LIMIT],
            invalid_operations[:DEFAULT_PER_PAGE_LIMIT],
            True,
        )

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

    def _record_state(self, page_id: str, fingerprint: str, added: int) -> None:
        with self._lock:
            state = self._audit_state.setdefault(page_id, {})
            state["fingerprint"] = fingerprint
            streak = int(state.get("zero_streak", 0) or 0)
            streak = 0 if added > 0 else streak + 1
            state["zero_streak"] = streak
            if streak >= DEFAULT_ZERO_ADD_LIMIT:
                state["saturated"] = True

    def _record_failure(self, page_id: str) -> None:
        """Leave failed audits retryable without changing zero-result streaks."""
        with self._lock:
            state = self._audit_state.setdefault(str(page_id or ""), {})
            # Invalidate the prior fingerprint so a transient failure does not
            # make the unchanged page look successfully audited.
            state["fingerprint"] = ""

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
            if streak >= DEFAULT_ZERO_ADD_LIMIT:
                state["saturated"] = True
                print(
                    f"[页面审计线程] 页面连续 {DEFAULT_ZERO_ADD_LIMIT} 次审计无新增，"
                    f"标记饱和不再审计: page={page_id}"
                )

    def pending_count(self) -> int:
        with self._lock:
            return self._pending

    def is_configured(self) -> bool:
        """Return whether final-output code may submit additional audits."""
        with self._lock:
            return self._llm is not None

    def flush(self, timeout: float | None = None) -> bool:
        """等待队列中所有已提交任务处理完成。"""
        with self._idle_cond:
            if timeout is None:
                while self._pending > 0:
                    self._idle_cond.wait()
                return True
            deadline = time.monotonic() + timeout
            while self._pending > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._idle_cond.wait(remaining)
            return True

    def reset(self) -> None:
        """恢复到未配置的初始状态；仅供测试隔离使用。"""
        with self._lock:
            self._llm = None
            self._results = []
            self._audit_state = {}
            self._max_pending = DEFAULT_MAX_PENDING


# 进程内单例，语义与 workflow/bug_detection_worker.py 一致。
page_audit_worker = PageAuditWorker()
