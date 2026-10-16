"""后台检测线程的公共骨架：单一后台线程顺序消费队列。

缺陷检测、合规判定、页面审计都只读取已拍好的截图做 LLM 调用，不触碰设备，
因此放到后台线程与设备操作重叠。单线程顺序处理让结果合并天然无竞态。
"""

from __future__ import annotations

import os
import queue
import threading
import time
import traceback
import uuid
from typing import Any, Callable

# 队列满时 submit 等待空位的总时长上限（秒）；超时放弃并记录失败，避免后台线程
# 卡死时把主循环永久阻塞。<=0 表示不设上限（不推荐）。
SUBMIT_WAIT_TIMEOUT_SECONDS = float(os.getenv("WORKER_SUBMIT_WAIT_TIMEOUT_SECONDS", "600") or 600)
# 等待空位时每次 wait 的时间片：醒来后检查后台线程存活并按需重启。
_SUBMIT_WAIT_SLICE_SECONDS = 1.0
UNFINISHED_ERROR_TYPE = "ReviewUnfinished"


class SubmitTimeoutError(TimeoutError):
    """队列持续满载，submit 在超时内未等到空位。"""


def build_trace_job(state: dict[str, Any], *extra_fields: str, with_trace_id: bool = False) -> dict[str, Any]:
    """从工作流状态抽取判定任务的只读快照（缺陷检测/合规检测共用）。

    trace 元素生成后不再被原地修改，浅拷贝列表即可；extra_fields 为按字符串
    复制的附加字段。
    """
    navigation_trace = state.get("navigation_execution_trace", [])
    scenario_trace = state.get("scenario_execution_trace", [])
    job: dict[str, Any] = {}
    if with_trace_id:
        job["trace_id"] = uuid.uuid4().hex
    job.update({
        "iteration_count": int(state.get("iteration_count", 0) or 0),
        "navigation_execution_trace": list(navigation_trace) if isinstance(navigation_trace, list) else [],
        "scenario_execution_trace": list(scenario_trace) if isinstance(scenario_trace, list) else [],
        "execution_result": dict(state.get("execution_result", {}) or {}),
        "current_plan": dict(state.get("current_plan", {}) or {}),
    })
    for field in extra_fields:
        job[field] = str(state.get(field, "") or "")
    return job


class SerialWorker:
    """单一后台线程 + 有界待处理计数。子类实现 `_process_job`。"""

    log_prefix = "[后台线程]"
    thread_name = "serial-worker"
    default_max_pending = 8
    # True：队列满时丢弃并计数（页面审计可随时重提）；False：阻塞等待空位，不丢证据。
    drop_when_full = False

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._idle_cond = threading.Condition(self._lock)
        self._queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._llm = None
        self._on_batch_done: Callable[[], None] | None = None
        self._failures: list[dict[str, Any]] = []
        self._pending = 0
        self._rejected_count = 0
        self._max_pending = self.default_max_pending
        # 已提交但尚未处理完的任务（含正在处理的一个），flush 超时时据此留痕。
        self._outstanding: dict[int, dict[str, Any]] = {}
        self._timed_out_ids: set[int] = set()
        self._next_job_id = 0

    def _configure_base(
        self,
        llm,
        on_batch_done: Callable[[], None] | None = None,
        initial_failures: list[dict[str, Any]] | None = None,
    ) -> None:
        with self._lock:
            self._llm = llm
            self._on_batch_done = on_batch_done
            self._failures = list(initial_failures or [])
        self._ensure_worker_started()

    def _ensure_worker_started(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            thread = threading.Thread(target=self._worker_loop, name=self.thread_name, daemon=True)
            self._thread = thread
        thread.start()

    def submit(self, job: dict[str, Any], timeout: float | None = None) -> bool:
        """提交任务；成功入队返回 True。

        队列满时：drop_when_full 直接丢弃返回 False；否则按时间片等待空位，
        每片醒来检查后台线程是否存活（死亡则重启），总等待超过 timeout
        （默认 SUBMIT_WAIT_TIMEOUT_SECONDS）仍无空位则记录失败并返回 False。
        """
        self._ensure_worker_started()
        wait_limit = SUBMIT_WAIT_TIMEOUT_SECONDS if timeout is None else float(timeout)
        deadline = time.monotonic() + wait_limit if wait_limit > 0 else None
        timed_out = False
        with self._idle_cond:
            announced = False
            while self._pending >= self._max_pending:
                if self.drop_when_full:
                    self._rejected_count += 1
                    print(
                        f"{self.log_prefix} 队列已达上限，本次提交被丢弃: "
                        f"pending={self._pending}, 累计丢弃={self._rejected_count}"
                    )
                    return False
                if not announced:
                    print(f"{self.log_prefix} 待处理任务已达上限，等待空位: pending={self._pending}")
                    announced = True
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    timed_out = True
                    break
                slice_seconds = _SUBMIT_WAIT_SLICE_SECONDS if remaining is None else min(
                    _SUBMIT_WAIT_SLICE_SECONDS, remaining
                )
                self._idle_cond.wait(slice_seconds)
                if self._thread is None or not self._thread.is_alive():
                    # 后台线程意外退出时队列永远不会被消费；锁外重启后继续等待。
                    self._idle_cond.release()
                    try:
                        print(f"{self.log_prefix} 后台线程已退出，重新启动以继续消费队列")
                        self._ensure_worker_started()
                    finally:
                        self._idle_cond.acquire()
            if not timed_out:
                self._pending += 1
                pending = self._pending
                self._next_job_id += 1
                job_id = self._next_job_id
                self._outstanding[job_id] = job
        if timed_out:
            print(f"{self.log_prefix} 等待队列空位超时（{wait_limit:.0f}s），本次提交放弃并记入失败")
            self.record_failure(job, SubmitTimeoutError(f"等待判定队列空位超时（{wait_limit:.0f}s）"))
            return False
        self._queue.put((job_id, job))
        print(f"{self.log_prefix} 已提交异步任务: pending={pending}")
        return True

    def _worker_loop(self) -> None:
        while True:
            job_id, job = self._queue.get()
            try:
                self._process_job(job)
            except Exception as exc:  # noqa: BLE001 - 后台线程需要吞掉异常继续消费
                print(f"{self.log_prefix} 处理任务失败: {exc}")
                print(traceback.format_exc())
                self.record_failure(job, exc)
            finally:
                cleared = False
                with self._idle_cond:
                    self._pending = max(0, self._pending - 1)
                    self._outstanding.pop(job_id, None)
                    if job_id in self._timed_out_ids:
                        self._timed_out_ids.discard(job_id)
                        # flush 超时时记过“判定未完成”，任务随后结束：成功则结论已合并，
                        # 失败则上面已按真实异常留痕。两种情况都撤销那条“未完成”记录，
                        # 避免产物里同一任务既有结论又报未完成（或重复计失败）。
                        before = len(self._failures)
                        self._failures = [
                            failure for failure in self._failures
                            if not (
                                failure.get("error_type") == UNFINISHED_ERROR_TYPE
                                and failure.get("job_id") == job_id
                            )
                        ]
                        cleared = len(self._failures) != before
                    self._idle_cond.notify_all()
                if cleared:
                    self._batch_done()

    def _process_job(self, job: dict[str, Any]) -> None:
        raise NotImplementedError

    def _batch_done(self) -> None:
        with self._lock:
            callback = self._on_batch_done
        if callback is not None:
            try:
                callback()
            except Exception as exc:  # noqa: BLE001
                print(f"{self.log_prefix} 落盘回调失败: {exc}")

    def _failure_record(self, job: dict[str, Any], exc: BaseException) -> dict[str, Any]:
        return {
            "iteration_count": job.get("iteration_count", 0),
            "scenario": job.get("current_test_scenario", ""),
            "plan": dict(job.get("current_plan", {}) or {}),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    def _on_failure_locked(self, job: dict[str, Any], exc: BaseException) -> None:
        """子类钩子：在锁内同步更新自身结果状态。"""

    def _on_unfinished_locked(self, job: dict[str, Any], exc: BaseException) -> None:
        """子类钩子：flush 超时仍未完成的任务；默认与普通失败相同处理。"""
        self._on_failure_locked(job, exc)

    def record_failure(self, job: dict[str, Any], exc: BaseException) -> None:
        """Record a failed or rejected review independently of queue drainage."""
        job = job if isinstance(job, dict) else {}
        with self._lock:
            self._failures.append(self._failure_record(job, exc))
            self._on_failure_locked(job, exc)
        self._batch_done()

    def failures_snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(failure) for failure in self._failures]

    def pending_count(self) -> int:
        with self._lock:
            return self._pending

    def rejected_count(self) -> int:
        with self._lock:
            return self._rejected_count

    def flush(self, timeout: float | None = None) -> bool:
        """等待已提交任务全部完成；全部完成返回 True，超时返回 False。

        timeout=None 表示无限等待；timeout<=0 表示不等待，只报告当前是否已空闲。
        """
        with self._idle_cond:
            if timeout is not None and timeout <= 0:
                return self._pending <= 0
            deadline = None if timeout is None else time.monotonic() + timeout
            while self._pending > 0:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._idle_cond.wait(remaining)
            return True

    def record_unfinished(self, reason: str) -> int:
        """把仍未完成的任务记为“判定未完成”失败，保证产物可按迭代/条款追溯。

        同一任务只留痕一次；任务若之后仍在后台成功完成，其结论照常合并，并按
        job_id 撤销这条“判定未完成”记录。
        """
        with self._lock:
            unfinished = [
                (job_id, job) for job_id, job in self._outstanding.items()
                if job_id not in self._timed_out_ids
            ]
            for job_id, job in unfinished:
                self._timed_out_ids.add(job_id)
                exc = TimeoutError(reason)
                self._failures.append({
                    **self._failure_record(job, exc),
                    "error_type": UNFINISHED_ERROR_TYPE,
                    "job_id": job_id,
                })
                self._on_unfinished_locked(job, exc)
        if unfinished:
            self._batch_done()
        return len(unfinished)

    def reset(self) -> None:
        """恢复到未配置的初始状态；仅供测试隔离使用。"""
        with self._lock:
            self._llm = None
            self._on_batch_done = None
            self._failures = []
            self._pending = 0
            self._rejected_count = 0
            self._max_pending = self.default_max_pending
            self._outstanding = {}
            self._timed_out_ids = set()
            self._next_job_id = 0
            # 丢弃尚未被消费的任务；正在处理的任务完成后只会把 _pending 下限钳到 0。
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._idle_cond.notify_all()
