"""异步执行轨迹缺陷检测。

设计动机：
- `trace_defect_detection.analyze_execution_for_bugs` 只读取已经拍好的截图和执行结果，
  不触碰设备；而下一轮 `scenario_plan`/`scenario_exec` 需要立即用设备继续探索。
  把缺陷检测放到独立的后台线程，可以让"设备操作"和"LLM 视觉复核"两件事的耗时重叠，
  从而缩短整体运行时间，同时不改变缺陷检测本身的语义。
- 缺陷检测结果（detected_bugs/bug_trace_records）不参与 scenario_plan/ffg_update 的决策，
  只用于落盘报告，因此异步化不会影响主循环的正确性。
- 使用单一后台 worker 线程顺序消费队列，而不是线程池并发：detected_bugs 的去重依赖
  "上一轮已确认的缺陷列表"，多个检测请求并发写会产生合并竞态；单线程顺序处理可以
  天然避免这个问题，且实现简单。

用法：
    from workflow.bug_detection_worker import bug_detection_worker

    bug_detection_worker.configure(bug_detection_llm, on_batch_done=save_callback)
    bug_detection_worker.submit(job_state)   # 立即返回，不阻塞主循环
    ...
    bug_detection_worker.flush()             # 工作流结束前等待队列排空
    detected_bugs, bug_trace_records = bug_detection_worker.snapshot()
"""

from __future__ import annotations

import queue
import os
import threading
import time
import traceback
from typing import Any, Callable

from workflow.state import AgentState
from workflow.trace_defect_detection import analyze_execution_for_bugs


DEFAULT_MAX_PENDING = max(1, int(os.getenv("BUG_DETECTION_MAX_PENDING", "8")))
DEFAULT_SUBMIT_WAIT_SECONDS = float(
    os.getenv("BUG_DETECTION_SUBMIT_WAIT_SECONDS", "0") or 0
)


class BugDetectionWorker:
    """单一后台线程 + 队列，顺序执行 `analyze_execution_for_bugs`。"""

    def __init__(self, max_pending: int = DEFAULT_MAX_PENDING) -> None:
        self._lock = threading.Lock()
        self._queue: "queue.Queue[dict[str, Any] | object]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._llm = None
        self._on_batch_done: Callable[[], None] | None = None
        self._on_job_failed: Callable[[BaseException], None] | None = None
        self._detected_bugs: list[dict[str, Any]] = []
        self._bug_trace_records: dict[str, list[dict[str, Any]]] = {}
        # 未处理完的任务数，配合 Condition 实现可超时的 flush()。
        self._pending = 0
        self._idle_cond = threading.Condition(self._lock)
        self._max_pending = max(1, int(max_pending))

    # ------------------------------------------------------------------
    # 配置 / 生命周期
    # ------------------------------------------------------------------
    def configure(
        self,
        llm,
        *,
        on_batch_done: Callable[[], None] | None = None,
        on_job_failed: Callable[[BaseException], None] | None = None,
        initial_bugs: list[dict[str, Any]] | None = None,
        initial_records: dict[str, list[dict[str, Any]]] | None = None,
        max_pending: int | None = None,
    ) -> None:
        """配置本次运行使用的 LLM、落盘回调，并（重新）播种已有缺陷（断点续跑场景）。"""
        with self._lock:
            self._llm = llm
            self._on_batch_done = on_batch_done
            self._on_job_failed = on_job_failed
            self._detected_bugs = list(initial_bugs or [])
            self._bug_trace_records = dict(initial_records or {})
            if max_pending is not None:
                self._max_pending = max(1, int(max_pending))
        self._ensure_worker_started()

    def _ensure_worker_started(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            thread = threading.Thread(
                target=self._worker_loop,
                name="bug-detection-worker",
                daemon=True,
            )
            self._thread = thread
        thread.start()

    # ------------------------------------------------------------------
    # 提交任务
    # ------------------------------------------------------------------
    def submit(self, job_state: dict[str, Any]) -> bool:
        """Submit without dropping evidence; apply backpressure at the memory cap.

        ``BUG_DETECTION_SUBMIT_WAIT_SECONDS=0`` (default) waits until a queue
        slot is available. A positive value permits an explicit bounded wait;
        a negative value restores the old non-blocking/drop behavior.
        """
        self._ensure_worker_started()
        wait_seconds = DEFAULT_SUBMIT_WAIT_SECONDS
        deadline = time.monotonic() + wait_seconds if wait_seconds > 0 else None
        with self._idle_cond:
            announced = False
            while self._pending >= self._max_pending:
                if not announced:
                    print(
                        "[缺陷检测线程] 待处理任务已达内存上限，等待空位以保留本轮复核: "
                        f"pending={self._pending}, max_pending={self._max_pending}"
                    )
                    announced = True
                if wait_seconds < 0:
                    return False
                if deadline is None:
                    self._idle_cond.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    print("[缺陷检测线程] 等待空位超时，本轮复核未入队。")
                    return False
                self._idle_cond.wait(remaining)
            self._pending += 1
            pending = self._pending
        self._queue.put(job_state)
        print(
            "[缺陷检测线程] 已提交异步检测任务: "
            f"pending={pending}, queue_size={self._queue.qsize()}"
        )
        return True

    # ------------------------------------------------------------------
    # 后台线程主循环
    # ------------------------------------------------------------------
    def _worker_loop(self) -> None:
        while True:
            job_state = self._queue.get()
            try:
                self._process_job(job_state)
            except Exception as exc:  # noqa: BLE001 - 后台线程需要吞掉异常继续消费
                print(f"[缺陷检测线程] 处理任务失败，保留已有缺陷结果: {exc}")
                print(traceback.format_exc())
                callback = self._on_job_failed
                if callback is not None:
                    try:
                        callback(exc)
                    except Exception as callback_exc:  # noqa: BLE001
                        print(f"[缺陷检测线程] 失败回调本身也失败: {callback_exc}")
            finally:
                with self._idle_cond:
                    self._pending = max(0, self._pending - 1)
                    # Wake both flush waiters and producers blocked by the
                    # pending-memory cap whenever one slot becomes available.
                    self._idle_cond.notify_all()

    def _process_job(self, job_state: dict[str, Any]) -> None:
        with self._lock:
            llm = self._llm
            # 用当前权威缺陷列表覆盖任务快照里的旧值，保证顺序合并不丢更新。
            job_state["detected_bugs"] = list(self._detected_bugs)
            job_state["bug_trace_records"] = dict(self._bug_trace_records)
        updates = analyze_execution_for_bugs(llm, job_state)  # type: ignore[arg-type]
        callback: Callable[[], None] | None = None
        with self._lock:
            if isinstance(updates.get("detected_bugs"), list):
                self._detected_bugs = updates["detected_bugs"]
            if isinstance(updates.get("bug_trace_records"), dict):
                self._bug_trace_records = updates["bug_trace_records"]
            callback = self._on_batch_done
        if callback is not None:
            try:
                callback()
            except Exception as exc:  # noqa: BLE001
                print(f"[缺陷检测线程] 落盘回调失败: {exc}")

    # ------------------------------------------------------------------
    # 读取 / 等待
    # ------------------------------------------------------------------
    def snapshot(self) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
        """线程安全地读取当前权威缺陷列表和轨迹记录快照。"""
        with self._lock:
            return list(self._detected_bugs), dict(self._bug_trace_records)

    def pending_count(self) -> int:
        with self._lock:
            return self._pending

    def flush(self, timeout: float | None = None) -> bool:
        """等待队列中所有已提交任务处理完成。

        Returns:
            True 表示在超时前完成；timeout=None 时会一直阻塞直到完成。
        """
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
        """恢复到未配置的初始状态；仅供测试隔离使用，不面向生产调用。"""
        with self._lock:
            self._llm = None
            self._on_batch_done = None
            self._on_job_failed = None
            self._detected_bugs = []
            self._bug_trace_records = {}
            self._max_pending = DEFAULT_MAX_PENDING


def build_defect_detection_job(state: AgentState) -> dict[str, Any]:
    """从工作流状态中抽取一份缺陷检测所需的只读快照，避免共享可变引用。

    只复制 `analyze_execution_for_bugs` 会读取的字段；trace 中的元素（dict/Page）
    在生成后不会被工作流其它节点原地修改，浅拷贝列表/字典足以避免后续状态更新
    影响已提交的任务。
    """
    navigation_trace = state.get("navigation_execution_trace", [])
    scenario_trace = state.get("scenario_execution_trace", [])
    return {
        "navigation_execution_trace": list(navigation_trace) if isinstance(navigation_trace, list) else [],
        "scenario_execution_trace": list(scenario_trace) if isinstance(scenario_trace, list) else [],
        "execution_result": dict(state.get("execution_result", {}) or {}),
        "current_plan": dict(state.get("current_plan", {}) or {}),
        "current_navigation_scenario": str(state.get("current_navigation_scenario", "") or ""),
        "current_test_scenario": str(state.get("current_test_scenario", "") or ""),
        "current_testing_function_id": str(state.get("current_testing_function_id", "") or ""),
        "package_name": str(state.get("package_name", "") or ""),
    }


# 进程内单例，语义上与 `workflow/llm_usage.py` 的全局用量登记表一致。
bug_detection_worker = BugDetectionWorker()
