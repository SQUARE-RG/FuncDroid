"""异步执行轨迹合规判定。

与 bug_detection_worker 同构。检测结果除落盘报告外，还通过只读快照参与
scenario_plan 的合规闭环重试；不直接修改 FFG。结果是按 rule_id 幂等的条款
结论（compliance_findings + checklist status）。
"""

from __future__ import annotations

from typing import Any, Callable

from compliance.checkpoints import ComplianceChecklist
from workflow.compliance_detection import analyze_execution_for_compliance
from workflow.worker_utils import SerialWorker, build_trace_job


def plan_check_ids(plan: dict[str, Any] | None) -> list[str]:
    return [
        str(item).strip()
        for item in (plan or {}).get("compliance_check_ids", []) or []
        if str(item).strip()
    ]


class ComplianceDetectionWorker(SerialWorker):
    """顺序执行 `analyze_execution_for_compliance`。"""

    log_prefix = "[合规检测线程]"
    thread_name = "compliance-detection-worker"

    def __init__(self) -> None:
        super().__init__()
        self._on_trace: Callable[[dict[str, Any]], str] | None = None
        self._checklist = ComplianceChecklist()

    def configure(
        self,
        llm,
        *,
        on_batch_done: Callable[[], None] | None = None,
        on_trace: Callable[[dict[str, Any]], str] | None = None,
        initial_checklist: dict[str, Any] | None = None,
        initial_failures: list[dict[str, Any]] | None = None,
    ) -> None:
        with self._lock:
            self._on_trace = on_trace
            if initial_checklist is not None:
                self._checklist = ComplianceChecklist.from_state(initial_checklist)
        self._configure_base(llm, on_batch_done, initial_failures)

    # 状态标记（供规划器派发合规场景时调用，防止队列积压时重复调度）
    def mark_executing(
        self,
        rule_ids: list[str],
        evidence_signature: str | dict[str, str] = "",
    ) -> None:
        with self._lock:
            self._checklist.mark_executing(rule_ids, evidence_signature)

    def defer_checks(self, rule_ids: list[str], reason: str) -> None:
        """执行或入队失败从未产生判定结果：退回 pending 并归还本次重试预算。"""
        with self._lock:
            released = self._checklist.release_scheduled(rule_ids)
        if released:
            print(f"{self.log_prefix} {reason}：{released} 已退回待执行，不计入重试预算")

    def reopen_deferred(
        self,
        evidence_signature: str | dict[str, str],
        *,
        force: bool = False,
        rules_path: Any = None,
    ) -> tuple[list[str], dict[str, Any]]:
        """原子重开可重试条款，并返回重开编号与最新清单快照。"""
        with self._lock:
            reopened = self._checklist.reopen_deferred(
                evidence_signature,
                force=force,
                rules_path=rules_path,
            )
            return reopened, self._checklist.to_state()

    def _process_job(self, job_state: dict[str, Any]) -> None:
        with self._lock:
            llm = self._llm
            on_trace = self._on_trace
        trace_path = on_trace(job_state) if on_trace is not None else ""
        updates = analyze_execution_for_compliance(llm, job_state)  # type: ignore[arg-type]
        results = updates.get("check_results")
        if isinstance(results, list):
            if trace_path:
                results = [
                    {**finding, "trace_path": trace_path} if isinstance(finding, dict) else finding
                    for finding in results
                ]
            with self._lock:
                self._checklist.apply_results(results)
        self._batch_done()

    def _failure_record(self, job: dict[str, Any], exc: BaseException) -> dict[str, Any]:
        return {"check_ids": plan_check_ids(job.get("current_plan")), **super()._failure_record(job, exc)}

    def _on_failure_locked(self, job: dict[str, Any], exc: BaseException) -> None:
        # 调用/解析重试耗尽、flush 超时、入队超时都属于基础设施失败：失败本身已由
        # record_failure 单独留痕；这里没有产生真实判定，不能按 undetermined 消耗
        # 重试预算（mark_executing 时已 +1），退回 pending 并归还预算（见 M3）。
        released = self._checklist.release_scheduled(plan_check_ids(job.get("current_plan")))
        if released:
            print(f"{self.log_prefix} 判定任务未产生结论（{type(exc).__name__}）：{released} 已退回待执行，不计入重试预算")

    def snapshot(self) -> dict[str, Any]:
        """线程安全地读取当前检测点清单状态。"""
        with self._lock:
            return self._checklist.to_state()

    def reset(self) -> None:
        super().reset()
        with self._lock:
            self._on_trace = None
            self._checklist = ComplianceChecklist()


def build_compliance_detection_job(state: dict[str, Any]) -> dict[str, Any]:
    """从工作流状态抽取合规判定所需的只读快照。"""
    return build_trace_job(
        state,
        "current_test_scenario",
        "compliance_rules_path",
        with_trace_id=True,
    )


# 进程内单例。
compliance_worker = ComplianceDetectionWorker()
