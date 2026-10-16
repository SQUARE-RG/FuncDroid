"""异步执行轨迹缺陷检测。

缺陷检测结果（detected_bugs/bug_trace_records）不参与 scenario_plan/ffg_update 的决策，
只用于落盘报告，因此异步化不会影响主循环的正确性。去重依赖“上一轮已确认的缺陷列表”，
由单一后台线程顺序合并。

用法：
    bug_detection_worker.configure(bug_detection_llm, on_batch_done=save_callback)
    bug_detection_worker.submit(job_state)   # 立即返回，不阻塞主循环
    bug_detection_worker.flush()             # 工作流结束前等待队列排空
    detected_bugs, bug_trace_records = bug_detection_worker.snapshot()
"""

from __future__ import annotations

from typing import Any, Callable

from workflow.state import AgentState
from workflow.trace_defect_detection import analyze_execution_for_bugs, copy_bug_record
from workflow.worker_utils import SerialWorker, build_trace_job


def _copy_bugs(bugs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [copy_bug_record(bug) if isinstance(bug, dict) else bug for bug in bugs]


def _copy_records(records: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    # 轨迹步骤生成后不再被原地修改；复制到列表层，避免与 worker 共享同一个列表。
    return {
        key: list(value) if isinstance(value, list) else value
        for key, value in records.items()
    }


class BugDetectionWorker(SerialWorker):
    """顺序执行 `analyze_execution_for_bugs`。"""

    log_prefix = "[缺陷检测线程]"
    thread_name = "bug-detection-worker"

    def __init__(self) -> None:
        super().__init__()
        self._detected_bugs: list[dict[str, Any]] = []
        self._bug_trace_records: dict[str, list[dict[str, Any]]] = {}

    def configure(
        self,
        llm,
        *,
        on_batch_done: Callable[[], None] | None = None,
        initial_bugs: list[dict[str, Any]] | None = None,
        initial_records: dict[str, list[dict[str, Any]]] | None = None,
        initial_failures: list[dict[str, Any]] | None = None,
    ) -> None:
        """配置本次运行使用的 LLM、落盘回调，并（重新）播种已有缺陷（断点续跑场景）。"""
        with self._lock:
            self._detected_bugs = _copy_bugs(list(initial_bugs or []))
            self._bug_trace_records = _copy_records(dict(initial_records or {}))
        self._configure_base(llm, on_batch_done, initial_failures)

    def _process_job(self, job_state: dict[str, Any]) -> None:
        with self._lock:
            llm = self._llm
            # 用当前权威缺陷列表覆盖任务快照里的旧值，保证顺序合并不丢更新。
            # 逐条复制：分析过程不得改动主线程快照可能仍在序列化的字典。
            job_state["detected_bugs"] = _copy_bugs(self._detected_bugs)
            job_state["bug_trace_records"] = _copy_records(self._bug_trace_records)
        updates = analyze_execution_for_bugs(llm, job_state)  # type: ignore[arg-type]
        with self._lock:
            # 整体替换为新列表/新字典，不原地修改已发布出去的对象。
            if isinstance(updates.get("detected_bugs"), list):
                self._detected_bugs = _copy_bugs(updates["detected_bugs"])
            if isinstance(updates.get("bug_trace_records"), dict):
                self._bug_trace_records = _copy_records(updates["bug_trace_records"])
        self._batch_done()

    def snapshot(self) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
        """线程安全地读取当前权威缺陷列表和轨迹记录快照（逐条复制，调用方可自由序列化/修改）。"""
        with self._lock:
            return _copy_bugs(self._detected_bugs), _copy_records(self._bug_trace_records)

    def reset(self) -> None:
        super().reset()
        with self._lock:
            self._detected_bugs = []
            self._bug_trace_records = {}


def build_defect_detection_job(state: AgentState) -> dict[str, Any]:
    """抽取缺陷检测所需的只读快照；trace 元素生成后不再被原地修改，浅拷贝即可。"""
    return build_trace_job(
        state,  # type: ignore[arg-type]
        "current_navigation_scenario",
        "current_test_scenario",
        "current_testing_function_id",
        "package_name",
    )


# 进程内单例。
bug_detection_worker = BugDetectionWorker()
