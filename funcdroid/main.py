"""FuncDroid 独立缺陷检测与 GUI 合规检测的统一入口。

用法：
  # 测试当前设备前台打开的 App
  python main.py --device-id emulator-5554

  # 测试设备上已安装的指定 App
  python main.py --device-id emulator-5554 --package-name com.example.app

  # 使用轮次控制而非时间控制
  python main.py --device-id emulator-5554 --max-iterations 20 --total-time 0
"""

import argparse
import base64
import hashlib
import json
import os
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv

# Load local configuration before module-level and CLI defaults are evaluated.
load_dotenv()

from device import Driver  # noqa: E402
from model.flow_graph import FlowGraph, Page  # noqa: E402
from workflow.bug_detection_worker import bug_detection_worker  # noqa: E402
from workflow.compliance_worker import compliance_worker  # noqa: E402
from compliance.checkpoints import ComplianceChecklist  # noqa: E402
from compliance.report import save_compliance_report  # noqa: E402
from workflow.exploration_boundary import enabled_packages  # noqa: E402
from workflow.graph import build_graph  # noqa: E402
from workflow.page_audit_worker import page_audit_worker  # noqa: E402
from workflow.llm import _resolve_request_timeout, create_llm  # noqa: E402
from workflow.llm_usage import (  # noqa: E402
    configure_llm_usage,
    get_llm_usage_totals,
    save_llm_usage_summary,
)
from workflow.run_logging import setup_logcat_capture, setup_run_logging  # noqa: E402
from workflow.run_options import RunArgumentParser, resolve_run_options, validate_resume_options  # noqa: E402
from workflow.seed_graph import load_seed_graph  # noqa: E402
from workflow.state import AgentState  # noqa: E402
from workflow.trace_artifacts import (  # noqa: E402
    configure_trace_artifacts,
    trace_screenshot_count,
)

DEFAULT_TOTAL_TIME_MINUTES = 90
DEFAULT_MAX_ITERATIONS = 0  # 0 表示关闭轮次上限，默认以总运行时间为准
DEFAULT_EXPLORATION_MODE = "core"
DEFAULT_OUTPUT_ROOT = "experiments/output"
ACTIVITY_TIMELINE_HEARTBEAT_SECONDS = max(
    1.0,
    float(os.getenv("ACTIVITY_TIMELINE_HEARTBEAT_SECONDS", "300")),
)
# 缺陷检测在独立后台线程里跑（见 workflow/bug_detection_worker.py），只读已拍好的
# 截图，不占用设备；工作流结束/异常退出前需要等待队列排空，避免最后几轮的检测结果
# 来不及落盘。设置超时是为了防止某次 LLM 调用异常挂起导致进程无法退出。
BUG_DETECTION_FLUSH_TIMEOUT_SECONDS = max(
    0.0,
    float(os.getenv("BUG_DETECTION_FLUSH_TIMEOUT_SECONDS", "180")),
)
# 正常收尾时页面审计补交（等待在途审计 + 补交未审页面）的总时长上限（秒）。
# 补交本身会产生新的 LLM 调用；不设上限时，大图上的逐批补交能让收尾拖很久。
FINALIZE_DEADLINE_SECONDS = max(
    0.0,
    float(os.getenv("FINALIZE_DEADLINE_SECONDS", "300")),
)
# 保护 bugs.json / bugs/ 目录的读-合并-写：缺陷检测线程完成一批检测后会异步落盘，
# 可能与主线程的运行快照落盘重叠，需要用锁避免并发写坏文件。
_BUGS_SAVE_LOCK = threading.Lock()
_COMPLIANCE_SAVE_LOCK = threading.Lock()
_CHECKPOINT_FAILURE_LOG_INTERVAL_SECONDS = max(
    1.0,
    float(os.getenv("CHECKPOINT_FAILURE_LOG_INTERVAL_SECONDS", "60")),
)
_CHECKPOINT_FAILURES: dict[str, dict[str, float | int]] = {}
PHASE_LABELS = {
    "all": "自动选择功能内/跨功能",
    "intra": "仅功能内测试",
}
EXPLORATION_MODE_LABELS = {
    "core": "核心测试",
}


@dataclass
class RunResources:
    output_dir: str = ""
    run_log: Any = None
    logcat_capture: Any = None
    device: Any = None


class RunTerminationSignal(BaseException):
    """Control flow: must bypass model retries and ordinary node recovery."""
    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"received signal {signum}")


def _raise_run_termination(signum, _frame) -> None:
    raise RunTerminationSignal(int(signum))


def _normalize_device_id(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        raise argparse.ArgumentTypeError("device id must not be empty")
    if text.lower() in {"all", "auto", "multi"}:
        raise argparse.ArgumentTypeError("请传入具体设备编号；多设备请分别开多个终端运行")
    return text


def _nonnegative_int(value: object) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("必须是大于等于 0 的整数") from exc
    if normalized < 0:
        raise argparse.ArgumentTypeError("必须是大于等于 0 的整数")
    return normalized


def _nonnegative_float(value: object) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("必须是大于等于 0 的数字") from exc
    if normalized < 0:
        raise argparse.ArgumentTypeError("必须是大于等于 0 的数字")
    return normalized


def _build_initial_state(args, start_time: float) -> AgentState:
    """构建初始状态。"""
    total_time_limit = args.total_time * 60 if args.total_time > 0 else 0

    # args 已由 RunArgumentParser.parse_args / _run_single 统一归一化，这里不再重复解析。
    return AgentState(
        run_mode=args.mode,
        reset_app_data=args.reset_app_data,
        grant_permissions=bool(args.grant_permissions),
        permission_grant_report=None,
        app_name=args.app_name or args.package_name or "当前前台应用",
        package_name=args.package_name or "",
        current_navigation_scenario="",
        current_test_scenario="",
        current_page=None,
        current_plan={},
        current_testing_function_id="",
        current_testing_function_status="not_started",
        navigation_execution_trace=[],
        scenario_execution_trace=[],
        detected_bugs=[],
        bug_trace_records={},
        bug_detection_enabled=bool(args.bug_detection),
        compliance_detection_enabled=bool(args.compliance),
        compliance_targeted=bool(getattr(args, "compliance_targeted", False)),
        compliance_checklist={
            "status": {},
            "findings": [],
            "attempts": {},
            "evidence_signatures": {},
            "discovery_baselines": {},
        },
        compliance_rules_path=str(getattr(args, "rules", "") or ""),
        ffg=FlowGraph(),
        navigation_failure_count=0,
        scenario_exec_error_count=0,
        scenario_plan_error_count=0,
        ffg_update_error_count=0,
        screenshot_unavailable_count=0,
        target_app_unavailable_count=0,
        function_execution_failure_counts={},
        iteration_count=0,
        max_iterations=args.max_iterations,
        iteration_started_at=0,
        iteration_llm_requests_start=0,
        iteration_llm_tokens_start=0,
        start_time=start_time,
        total_time_limit=total_time_limit,
        workflow_complete=False,
        long_run_idle_count=0,
        run_phase=args.phase,
        exploration_mode=args.exploration_mode,
        completion_audit_signatures={},
        completion_audit_attempts={},
        deferred_retry_ids=[],
        unplannable_counts={},
    )


def _load_resume_state(args, output_dir: str, started_at: float) -> AgentState:
    """Restore durable exploration state while starting a fresh graph execution."""
    state = _build_initial_state(args, started_at)
    root = Path(output_dir)
    ffg_path = root / "ffg.json"
    if not ffg_path.is_file():
        raise FileNotFoundError(f"续跑目录缺少 ffg.json: {ffg_path}")
    state["ffg"] = FlowGraph.load(output_dir)

    result_path = root / "run_result.json"
    previous_elapsed = 0.0
    if result_path.is_file():
        with result_path.open("r", encoding="utf-8") as f:
            previous_result = json.load(f)
        previous_elapsed = max(0.0, float(previous_result.get("elapsed_seconds", 0) or 0))
        state["iteration_count"] = max(
            0,
            int(previous_result.get("iteration_count", 0) or 0),
        )
        state["defect_review_failures"] = list(previous_result.get("defect_review_failures", []) or [])
        state["compliance_review_failures"] = list(previous_result.get("compliance_review_failures", []) or [])
        if previous_result.get("seed_provenance"):
            state["seed_provenance"] = previous_result["seed_provenance"]
    state["start_time"] = started_at - previous_elapsed

    bugs_path = root / "bugs.json"
    if bugs_path.is_file():
        with bugs_path.open("r", encoding="utf-8") as f:
            bugs = json.load(f)
        if isinstance(bugs, list):
            state["detected_bugs"] = bugs
            state["bug_trace_records"] = _load_bug_trace_records(root, bugs)

    failures_path = root / "defect_review_failures.json"
    if failures_path.is_file():
        failures = json.loads(failures_path.read_text(encoding="utf-8"))
        if not isinstance(failures, list):
            raise ValueError("defect_review_failures.json 必须是数组")
        state["defect_review_failures"] = failures

    compliance_failures_path = root / "compliance_review_failures.json"
    if compliance_failures_path.is_file():
        compliance_failures = json.loads(compliance_failures_path.read_text(encoding="utf-8"))
        if not isinstance(compliance_failures, list):
            raise ValueError("compliance_review_failures.json 必须是数组")
        state["compliance_review_failures"] = compliance_failures

    compliance_path = root / "compliance_checklist.json"
    if compliance_path.is_file():
        with compliance_path.open("r", encoding="utf-8") as f:
            compliance_state = json.load(f)
        if isinstance(compliance_state, dict):
            checklist = ComplianceChecklist.from_state(compliance_state)
            recovered_checks = checklist.recover_interrupted()
            if recovered_checks:
                print(
                    "[断点续跑] 恢复中断的合规检测点为待执行: "
                    + ", ".join(recovered_checks)
                )
            state["compliance_checklist"] = checklist.to_state()

    timeline_path = root / "activity_timeline.json"
    if timeline_path.is_file():
        with timeline_path.open("r", encoding="utf-8") as f:
            timeline = json.load(f)
        if isinstance(timeline, dict):
            state["_activity_timeline"] = timeline

    # A previous segment may have completed or been interrupted. The durable FFG
    # determines what remains; the new process must always re-enter planning.
    state["workflow_complete"] = False
    print(
        "[断点续跑] 已恢复累计状态: "
        f"elapsed={previous_elapsed / 60:.1f} 分钟, "
        f"iterations={state['iteration_count']}, "
        f"functions={len(state['ffg'].functions)}, pages={len(state['ffg'].pages)}, "
        f"activities={len((state.get('_activity_timeline') or {}).get('activities', []))}, "
        f"bugs={len(state['detected_bugs'])}"
    )
    return state


def _load_bug_trace_records(root: Path, bugs: list[dict]) -> dict[str, list[dict]]:
    """Restore each bug's own persisted actions and images, never the latest scene."""
    records = {}
    bug_ids = {str(bug.get("bug_id", "")) for bug in bugs if isinstance(bug, dict)}
    for metadata_path in (root / "bugs").glob("*/bug.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        bug_id = str(metadata.get("bug_id", ""))
        trace_path = metadata_path.with_name("trace.json")
        if bug_id not in bug_ids or not trace_path.is_file():
            continue
        serialized = json.loads(trace_path.read_text(encoding="utf-8"))
        if not isinstance(serialized, list):
            raise ValueError(f"缺陷轨迹必须是数组: {trace_path}")
        trace = []
        for saved in serialized:
            if not isinstance(saved, dict):
                raise ValueError(f"缺陷轨迹含无效步骤: {trace_path}")
            step = dict(saved)
            step.pop("step_index", None)
            operation = step.pop("operation", {})
            parameters = operation.get("parameters", {})
            # The serialized top-level flags include defaults even when the
            # original action had no such parameters. Avoid inventing them.
            for field in ("foreground_recovery", "external_context"):
                if field not in parameters and step.get(field) is False:
                    step.pop(field)
            step.update(parameters)
            step["action_name"] = operation.get("action_name", "")
            page = dict(step.pop("page", {}) or {})
            reference = page.get("screenshot")
            if reference:
                image_path = (trace_path.parent / reference).resolve()
                if not image_path.is_relative_to(root.resolve()):
                    raise ValueError(f"缺陷截图引用超出运行目录: {trace_path}")
                page["screenshot"] = base64.b64encode(image_path.read_bytes()).decode("ascii")
            step["current_page"] = page
            trace.append(step)
        records[bug_id] = trace
    return records


def _atomic_write_json(path: Path | str, data: Any) -> None:
    """原子写 JSON：先写临时文件再 os.replace，避免运行中被 kill 留下截断文件。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f"{target.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, target)


def _atomic_write_bytes(path: Path | str, data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f"{target.name}.tmp")
    with tmp_path.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, target)


def _activity_from_page(page: Any) -> str:
    if isinstance(page, Page):
        return str(page.activity or "").strip()
    if isinstance(page, dict):
        return str(page.get("activity", "") or "").strip()
    return ""


def _observed_activities(state: dict) -> set[str]:
    activities: set[str] = set()
    graph = state.get("ffg")
    if isinstance(graph, FlowGraph):
        activities.update(
            activity
            for page in graph.active_pages()
            if (activity := _activity_from_page(page))
        )
    current_activity = _activity_from_page(state.get("current_page"))
    if current_activity:
        activities.add(current_activity)
    for trace_key in ("navigation_execution_trace", "scenario_execution_trace"):
        trace = state.get(trace_key, []) or []
        if not isinstance(trace, list):
            continue
        for step in trace:
            if not isinstance(step, dict):
                continue
            if step.get("external_context"):
                continue
            activity = _activity_from_page(step.get("current_page"))
            if activity:
                activities.add(activity)
    return activities


def _new_activity_timeline(state: dict) -> dict[str, Any]:
    start_time = float(state.get("start_time", 0) or time.time())
    return {
        "schema_version": 1,
        "started_at": datetime.fromtimestamp(start_time).astimezone().isoformat(),
        "updated_at": datetime.now().astimezone().isoformat(),
        "activities": [],
        "points": [],
    }


def _render_activity_timeline(timeline: dict[str, Any], output_path: Path) -> None:
    width, height = 1200, 700
    left, right, top, bottom = 110, 50, 75, 105
    plot_width = width - left - right
    plot_height = height - top - bottom
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    axis_color = (60, 60, 60)
    grid_color = (225, 225, 225)
    line_color = (220, 105, 35)
    points = [item for item in timeline.get("points", []) if isinstance(item, dict)]
    max_minutes = max(
        1.0,
        max((float(item.get("elapsed_minutes", 0) or 0) for item in points), default=0.0),
    )
    max_count = max(
        1,
        max((int(item.get("activity_count", 0) or 0) for item in points), default=0),
    )

    cv2.putText(
        canvas,
        "Discovered Android Activities Over Time",
        (left, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    for tick in range(6):
        x = left + round(plot_width * tick / 5)
        y = top + plot_height - round(plot_height * tick / 5)
        cv2.line(canvas, (x, top), (x, top + plot_height), grid_color, 1)
        cv2.line(canvas, (left, y), (left + plot_width, y), grid_color, 1)
        cv2.putText(
            canvas,
            f"{max_minutes * tick / 5:.1f}",
            (x - 18, top + plot_height + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            axis_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            str(round(max_count * tick / 5)),
            (left - 58, y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            axis_color,
            1,
            cv2.LINE_AA,
        )
    cv2.line(canvas, (left, top), (left, top + plot_height), axis_color, 2)
    cv2.line(
        canvas,
        (left, top + plot_height),
        (left + plot_width, top + plot_height),
        axis_color,
        2,
    )
    cv2.putText(
        canvas,
        "Elapsed time (minutes)",
        (left + plot_width // 2 - 95, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        axis_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Unique Activity count",
        (12, top - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        axis_color,
        1,
        cv2.LINE_AA,
    )

    plotted: list[tuple[int, int]] = []
    for item in points:
        elapsed = float(item.get("elapsed_minutes", 0) or 0)
        count = int(item.get("activity_count", 0) or 0)
        x = left + round(plot_width * elapsed / max_minutes)
        y = top + plot_height - round(plot_height * count / max_count)
        plotted.append((x, y))
    if len(plotted) >= 2:
        cv2.polylines(canvas, [np.asarray(plotted, dtype=np.int32)], False, line_color, 3)
    for point in plotted:
        cv2.circle(canvas, point, 5, line_color, -1, cv2.LINE_AA)
    current_count = len(timeline.get("activities", []) or [])
    cv2.putText(
        canvas,
        f"Current unique activities: {current_count}",
        (width - 360, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        line_color,
        2,
        cv2.LINE_AA,
    )
    success, encoded = cv2.imencode(".png", canvas)
    if not success:
        raise RuntimeError("activity timeline PNG encoding failed")
    _atomic_write_bytes(output_path, encoded.tobytes())


def _save_activity_timeline(
    state: dict,
    output_dir: str,
    timeline: dict[str, Any] | None = None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    timeline = timeline if isinstance(timeline, dict) else _new_activity_timeline(state)
    activities = {
        str(item).strip()
        for item in timeline.get("activities", [])
        if str(item).strip()
    }
    activities.update(_observed_activities(state))
    start_time = float(state.get("start_time", 0) or time.time())
    elapsed_seconds = max(0.0, time.time() - start_time)
    points = timeline.setdefault("points", [])
    last = points[-1] if points and isinstance(points[-1], dict) else {}
    # 注意 0 是合法计数：不能写成 `or -1`，否则零 Activity 时每次都判定为
    # “数量变化”，每个节点都重绘 PNG 并无界追加采样点。
    last_count = last.get("activity_count")
    try:
        last_count = int(last_count) if last_count is not None else -1
    except (TypeError, ValueError):
        last_count = -1
    count_changed = last_count != len(activities)
    heartbeat_due = (
        elapsed_seconds - float(last.get("elapsed_seconds", 0) or 0)
        >= ACTIVITY_TIMELINE_HEARTBEAT_SECONDS
    )
    root = Path(output_dir)
    should_persist = (
        not points
        or force
        or count_changed
        or heartbeat_due
        or not (root / "activity_timeline.json").is_file()
        or not (root / "activity_timeline.png").is_file()
    )
    if should_persist:
        timeline["activities"] = sorted(activities)
        timeline["updated_at"] = datetime.now().astimezone().isoformat()
        point = {
            "timestamp": timeline["updated_at"],
            "elapsed_seconds": round(elapsed_seconds, 3),
            "elapsed_minutes": round(elapsed_seconds / 60, 4),
            "activity_count": len(activities),
        }
        if (
            force
            and last
            and last_count == len(activities)
            and abs(float(last.get("elapsed_seconds", 0) or 0) - elapsed_seconds) < 1.0
        ):
            points[-1] = point
        else:
            points.append(point)
        _atomic_write_json(root / "activity_timeline.json", timeline)
        _render_activity_timeline(timeline, root / "activity_timeline.png")
    return timeline


def _sync_bug_state(state: dict) -> None:
    """用后台缺陷检测线程的权威数据刷新 state 里的镜像字段。"""
    if not bool(state.get("bug_detection_enabled", True)):
        return
    detected_bugs, bug_trace_records = bug_detection_worker.snapshot()
    state["detected_bugs"] = detected_bugs
    state["bug_trace_records"] = bug_trace_records
    state["defect_review_failures"] = bug_detection_worker.failures_snapshot()


def _flush_worker(worker, reason: str) -> None:
    """等待后台检测队列排空，超时则继续（避免异常挂起导致无法退出）。"""
    pending = worker.pending_count()
    if pending <= 0:
        return
    print(f"{worker.log_prefix} {reason}前等待 {pending} 个未完成的异步任务...")
    # 0 表示不等待（直接把未完成任务记为判定未完成），不能写成 `or None` 变成无限等待。
    if not worker.flush(timeout=BUG_DETECTION_FLUSH_TIMEOUT_SECONDS):
        recorded = worker.record_unfinished(f"{reason}前等待后台判定超时，判定未完成")
        print(
            f"{worker.log_prefix} 等待超时，仍有任务在后台运行；"
            f"已将 {recorded} 个未完成任务记入判定失败（判定未完成）。"
        )


def _sync_compliance_state(state: dict) -> None:
    """用后台合规检测线程的权威数据刷新 state 里的检测点清单。"""
    if not bool(state.get("compliance_detection_enabled", False)):
        return
    state["compliance_checklist"] = compliance_worker.snapshot()
    state["compliance_review_failures"] = compliance_worker.failures_snapshot()


def _save_compliance_outputs(state: dict, output_dir: str, prefix: str):
    """落盘合规检测点清单与最终报告（json + markdown）。"""
    if not bool(state.get("compliance_detection_enabled", False)):
        return
    with _COMPLIANCE_SAVE_LOCK:
        _save_compliance_outputs_locked(state, output_dir, prefix)


def _save_compliance_outputs_locked(state: dict, output_dir: str, prefix: str):
    _sync_compliance_state(state)
    _atomic_write_json(
        Path(output_dir) / "compliance_review_failures.json",
        state.get("compliance_review_failures", []),
    )
    try:
        checklist = ComplianceChecklist.from_state(state.get("compliance_checklist"))
        _atomic_write_json(
            os.path.join(output_dir, "compliance_checklist.json"),
            checklist.to_state(),
        )
        report = save_compliance_report(
            checklist,
            output_dir,
            app_name=str(state.get("app_name", "") or ""),
            package_name=str(state.get("package_name", "") or ""),
            rules_path=str(state.get("compliance_rules_path", "") or "").strip() or None,
        )
        summary = report.get("summary", {})
        print(
            f"{prefix} compliance_report 已保存: "
            f"processed={summary.get('processed', summary.get('executed', 0))}/"
            f"{summary.get('gui_total', 0)}, "
            f"resolved={summary.get('resolved', 0)}, "
            f"unresolved={summary.get('unresolved', summary.get('deferred', 0))}, "
            f"violation={summary.get('violation', 0)}, "
            f"compliant={summary.get('compliant', 0)}"
        )
    except Exception as exc:
        print(f"{prefix} compliance_report 保存失败: {exc}")


def _flush_page_audit(
    final_state: dict,
    *,
    interrupted: bool = False,
    deadline: float | None = None,
) -> None:
    """等待页面审计队列排空，并把剩余审计结果合并进图（落盘前调用）。

    后台审计结果平时由下一轮 ffg_update 节点入口合并；运行结束时没有
    "下一轮"，必须在此处直接合并，否则最后一批结果会丢失。

    interrupted=True（用户中断/终止信号）时只合并已完成的结果，不等待在途
    任务、也不补交新审计；正常收尾时等待与补交受 deadline（time.monotonic()
    时间点）约束，到点即停止补交。
    """
    graph = final_state.get("ffg")
    if isinstance(graph, FlowGraph):
        try:
            from workflow.nodes.ffg_update import (
                _merge_page_audit_results,
                _submit_page_audit_batch,
            )

            if interrupted:
                _merge_page_audit_results(graph)
                pending = page_audit_worker.pending_count()
                if pending > 0:
                    print(
                        f"[页面审计线程] 运行被中断：只合并已完成的审计结果，"
                        f"不再等待 {pending} 个在途任务、不补交新审计。"
                    )
                return

            # ffg_update 在应用本轮 trace 前提交审计，因此最后新增或刚增加
            # 截图的 Page 尚未入队。结尾补交当前未审计指纹；worker 的
            # 指纹和饱和状态会阻止重复空转。
            # A failed worker call deliberately leaves its fingerprint
            # unaudited so a later workflow iteration may retry it. During
            # final output there is no later iteration, so remember what this
            # flush already submitted; otherwise one persistent 502 resubmits
            # the same page forever and prevents artifact finalization.
            final_submitted_fingerprints: set[tuple[str, str]] = set()
            while True:
                pending = page_audit_worker.pending_count()
                if pending > 0:
                    timeout = BUG_DETECTION_FLUSH_TIMEOUT_SECONDS
                    if deadline is not None:
                        timeout = min(timeout, max(0.0, deadline - time.monotonic()))
                    print(
                        f"[页面审计线程] 保存最终输出前等待 {pending} "
                        "个未完成的审计任务..."
                    )
                    page_audit_worker.flush(timeout=timeout)
                _merge_page_audit_results(graph)
                if deadline is not None and time.monotonic() >= deadline:
                    print(
                        "[页面审计线程] 已到收尾时限 "
                        f"FINALIZE_DEADLINE_SECONDS={FINALIZE_DEADLINE_SECONDS:g}s，"
                        "停止补交剩余未审计页面。"
                    )
                    break
                submitted = _submit_page_audit_batch(
                    graph,
                    exclude_fingerprints=final_submitted_fingerprints,
                )
                if submitted <= 0:
                    break
                print(
                    f"[页面审计线程] 最终补交 {submitted} 个尚未审计的页面指纹。"
                )
        except Exception as exc:  # noqa: BLE001 - 落盘前兜底，失败不阻塞退出
            print(f"[页面审计线程] 最终合并审计结果失败: {exc}")


def _print_summary(state: dict, output_dir: str, elapsed_seconds: float):
    """打印运行结束的摘要。"""
    _sync_bug_state(state)
    elapsed_min = elapsed_seconds / 60
    ffg = state.get("ffg")

    print(f"\n{'='*60}")
    print("  FuncDroid 运行结束摘要")
    print(f"{'='*60}")
    print(f"  总用时: {elapsed_min:.1f} 分钟")
    exploration_mode = str(state.get("exploration_mode") or DEFAULT_EXPLORATION_MODE)
    print(f"  探索范围: {EXPLORATION_MODE_LABELS.get(exploration_mode, exploration_mode)}")

    if isinstance(ffg, FlowGraph):
        print(f"  FFG 功能节点数: {len(ffg.functions)}")
        print(f"  FFG 页面数: {len(ffg.pages)}")
        print(f"  页面跳转数: {len(ffg.page_transitions)}")
        print(f"  功能依赖数: {len(ffg.function_dependencies)}")
        status_counts = ffg.test_status_counts()
        print(
            "  FFG 状态分布: "
            f"completed={status_counts.get('completed', 0)}, "
            f"in_progress={status_counts.get('in_progress', 0)}, "
            f"not_started={status_counts.get('not_started', 0)}, "
            f"blocked={status_counts.get('blocked', 0)}, "
            f"deferred={status_counts.get('deferred', 0)}, "
            f"unavailable={status_counts.get('unavailable', 0)}"
        )

    timeline = state.get("_activity_timeline", {})
    activities = timeline.get("activities", []) if isinstance(timeline, dict) else []
    if not activities:
        activities = sorted(_observed_activities(state))
    print(f"  已发现 Android Activity: {len(activities)} 个")

    bugs = state.get("detected_bugs", [])
    if not isinstance(bugs, list):
        bugs = []
    if bugs:
        crash_bugs = [b for b in bugs if isinstance(b, dict) and b.get("type") == "crash"]
        non_crash_bugs = [b for b in bugs if isinstance(b, dict) and b.get("type") != "crash"]
        print(f"  检测到的缺陷: {len(bugs)} 个")
        print(f"    - 崩溃类: {len(crash_bugs)} 个")
        print(f"    - 非崩溃类: {len(non_crash_bugs)} 个")
    elif state.get("bug_detection_enabled", True):
        print("  检测到的缺陷: 0 个")
    else:
        print("  缺陷检测: 本次未运行")

    if bool(state.get("compliance_detection_enabled", False)):
        try:
            summary = ComplianceChecklist.from_state(
                state.get("compliance_checklist")
            ).summary(str(state.get("compliance_rules_path", "") or "").strip() or None)
            print(
                "  合规检测点: "
                f"{summary.get('processed', summary.get('executed', 0))}/"
                f"{summary['gui_total']} 已处理 | "
                f"已形成结论 {summary.get('resolved', 0)} | "
                f"符合 {summary['compliant']} | 违规 {summary['violation']} | "
                f"证据不足 {summary['undetermined']} | 不适用 {summary['not_applicable']} | "
                f"未决待新证据 {summary.get('unresolved', summary['deferred'])} | "
                f"尚未处理 {summary.get('untested', summary['pending'])}"
            )
        except Exception as exc:
            print(f"  合规检测统计失败: {exc}")

    print(f"  执行轮次: {state.get('iteration_count', 0)}")
    print(f"  原始轨迹截图: {trace_screenshot_count()} 张 (trace_screenshots/)")
    usage_totals = get_llm_usage_totals()
    if usage_totals:
        print(
            "  LLM token: "
            f"prompt={usage_totals.get('prompt_tokens', 0)}, "
            f"cached={usage_totals.get('cached_tokens', 0)}, "
            f"completion={usage_totals.get('completion_tokens', 0)}, "
            f"total={usage_totals.get('total_tokens', 0)}, "
            f"requests={usage_totals.get('requests', 0)}, "
            f"failures={usage_totals.get('failures', 0)}"
        )
    print(f"  输出目录: {output_dir}")
    print("  Activity 时间序列: activity_timeline.json / activity_timeline.png")
    print(f"{'='*60}\n")


def _save_final_outputs(state: dict, output_dir: str):
    """保存最终输出，不保存可恢复的工作流状态。"""
    _save_outputs(state, output_dir, prefix="[输出]")


def _save_outputs(state: dict, output_dir: str, prefix: str):
    os.makedirs(output_dir, exist_ok=True)
    ffg = state.get("ffg")
    if isinstance(ffg, FlowGraph):
        try:
            ffg.save(output_dir)
        except Exception as exc:
            print(f"{prefix} FFG 保存失败: {exc}")
    _save_bugs(state, output_dir, prefix=prefix)
    _save_compliance_outputs(state, output_dir, prefix=prefix)


def _save_bugs(state: dict, output_dir: str, prefix: str):
    if not state.get("bug_detection_enabled", True):
        return
    _sync_bug_state(state)
    with _BUGS_SAVE_LOCK:
        _atomic_write_json(
            Path(output_dir) / "defect_review_failures.json",
            state.get("defect_review_failures", []),
        )
        detected_bugs = state.get("detected_bugs", [])
        if not isinstance(detected_bugs, list):
            detected_bugs = []
        detected_bugs = _filter_high_confidence_bugs(detected_bugs, prefix=prefix)
        bugs_path = os.path.join(output_dir, "bugs.json")
        try:
            existing_bugs = []
            if os.path.exists(bugs_path):
                with open(bugs_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    existing_bugs = [bug for bug in loaded if isinstance(bug, dict)]
            existing_bugs = _filter_high_confidence_bugs(existing_bugs, prefix=prefix)
            merged_bugs = _merge_bugs_by_id(existing_bugs, detected_bugs)
            _atomic_write_json(bugs_path, merged_bugs)
            print(f"{prefix} bugs.json 已保存: {bugs_path}")
        except Exception as exc:
            print(f"{prefix} bugs.json 保存失败: {exc}")
        _save_bug_folders(state, output_dir, detected_bugs, prefix)


def _is_high_confidence_bug(bug: dict[str, Any]) -> bool:
    return str(bug.get("confidence", "")).strip().lower() == "high"


def _filter_high_confidence_bugs(bugs: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    high_confidence = [
        bug
        for bug in bugs
        if isinstance(bug, dict) and _is_high_confidence_bug(bug)
    ]
    skipped_count = len([bug for bug in bugs if isinstance(bug, dict)]) - len(high_confidence)
    if skipped_count:
        print(f"{prefix} 已过滤非 high 置信度缺陷: {skipped_count} 个")
    return high_confidence


def _merge_bugs_by_id(existing_bugs: list[dict[str, Any]], new_bugs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    index_by_id: dict[str, int] = {}

    for bug in existing_bugs:
        bug_id = str(bug.get("bug_id", "")).strip()
        if bug_id and bug_id not in index_by_id:
            index_by_id[bug_id] = len(merged)
        merged.append(dict(bug))

    for bug in new_bugs:
        bug_id = str(bug.get("bug_id", "")).strip()
        if bug_id and bug_id in index_by_id:
            merged[index_by_id[bug_id]] = dict(bug)
        else:
            if bug_id:
                index_by_id[bug_id] = len(merged)
            merged.append(dict(bug))

    return merged


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _decode_screenshot(screenshot: str) -> bytes | None:
    screenshot = str(screenshot or "").strip()
    if not screenshot:
        return None
    if screenshot.startswith("data:image") and "," in screenshot:
        screenshot = screenshot.split(",", 1)[1]
    try:
        return base64.b64decode(screenshot, validate=False)
    except Exception:
        return None


def _page_screenshot(page: Any) -> str:
    if isinstance(page, Page):
        return str(page.representative_screenshot or "")
    if isinstance(page, dict):
        representative = str(page.get("representative_screenshot", "") or "")
        if representative:
            return representative
        screenshots = page.get("screenshots")
        if isinstance(screenshots, list) and screenshots:
            return str(screenshots[0] or "")
        return str(
            page.get("screenshot")
            or page.get("screenshot_base64")
            or ""
        )
    return ""


def _serialize_page(page: Any, screenshot_ref: str) -> dict[str, Any]:
    if isinstance(page, Page):
        return {
            "id": page.id,
            "name": page.name,
            "description": page.description,
            "controls": list(page.controls),
            "operations": list(page.operations),
            "activity": page.activity,
            "screenshot": screenshot_ref,
        }
    if isinstance(page, dict):
        page_data = {
            key: _json_safe(value)
            for key, value in page.items()
            if key not in {"screenshot", "screenshot_base64"}
        }
        page_data["screenshot"] = screenshot_ref
        return page_data
    return {"screenshot": screenshot_ref}


def _repair_bug_trace_screenshots(trace: list[dict[str, Any]], bug_dir: str) -> None:
    screenshots_dir = os.path.join(bug_dir, "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)
    for index, step in enumerate(trace, start=1):
        if not isinstance(step, dict):
            continue
        page = step.get("current_page")
        screenshot = _page_screenshot(page)
        if not screenshot:
            continue
        screenshot_name = f"step_{index:03d}.jpg"
        screenshot_path = os.path.join(screenshots_dir, screenshot_name)
        if os.path.exists(screenshot_path):
            continue
        screenshot_bytes = _decode_screenshot(screenshot)
        if not screenshot_bytes:
            continue
        _atomic_write_bytes(screenshot_path, screenshot_bytes)


def _serialize_trace(trace: list[dict[str, Any]], bug_dir: str) -> list[dict[str, Any]]:
    screenshots_dir = os.path.join(bug_dir, "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)
    serialized: list[dict[str, Any]] = []
    action_param_fields = (
        "coordinate_space",
        "model_coordinate_space",
        "model_coordinates",
        "image_size",
        "screen_size",
        "x",
        "y",
        "x1",
        "y1",
        "x2",
        "y2",
        "duration",
        "content",
        "direction",
        "app_name",
        "foreground_recovery",
        "recovery_method",
        "final_recovery",
        "recovered_to_target_app",
        "external_context",
        "external_package",
        "caused_external_transition",
        "caused_target_crash",
        "previous_package",
        "target_package",
        "foreground_package",
        "captured_after_seconds",
    )
    for index, step in enumerate(trace, start=1):
        if not isinstance(step, dict):
            continue
        page = step.get("current_page")
        screenshot_ref = ""
        screenshot = _page_screenshot(page)
        screenshot_bytes = _decode_screenshot(screenshot)
        if screenshot_bytes:
            screenshot_name = f"step_{index:03d}.jpg"
            screenshot_ref = f"screenshots/{screenshot_name}"
            _atomic_write_bytes(
                os.path.join(screenshots_dir, screenshot_name),
                screenshot_bytes,
            )

        operation = {
            "action_name": str(step.get("action_name", "")).strip(),
            "parameters": {
                field: _json_safe(step[field])
                for field in action_param_fields
                if field in step
            },
        }
        serialized.append({
            "step_index": index,
            "raw_step_index": step.get("raw_step_index"),
            "trace_phase": step.get("trace_phase", ""),
            "operation": operation,
            "thought": step.get("thought", ""),
            "action_description": step.get("action_description", ""),
            "is_interruption": bool(step.get("is_interruption", False)),
            "action_failed": bool(step.get("action_failed", False)),
            "external_context": bool(step.get("external_context", False)),
            "foreground_recovery": bool(step.get("foreground_recovery", False)),
            "action_error": step.get("action_error", ""),
            "page": _serialize_page(page, screenshot_ref),
        })
    return serialized


def _save_compliance_trace(job: dict, output_dir: str) -> str:
    """Persist the exact review input; per-job step numbers are not globally unique."""
    relative = Path("compliance_traces") / job["trace_id"] / "trace.json"
    trace = list(job.get("navigation_execution_trace", [])) + list(job.get("scenario_execution_trace", []))
    trace_dir = Path(output_dir) / relative.parent
    steps = _serialize_trace(trace, str(trace_dir))
    _atomic_write_json(Path(output_dir) / relative, {
        "iteration_count": job.get("iteration_count", 0),
        "plan": job.get("current_plan", {}),
        "scenario": job.get("current_test_scenario", ""),
        "execution_result": job.get("execution_result", {}),
        "steps": steps,
    })
    return relative.as_posix()


# 缺陷产物指纹缓存：bug_id → (廉价键, 完整指纹, 截图字符串强引用)。
# 每次快照/每个检测批次都会对全部缺陷重算指纹，其中逐字节哈希 base64 截图
# 是主要开销。廉价键覆盖完整指纹的全部非截图输入（同样的 JSON 序列化），
# 截图只按对象身份 (id, len) 计入；缓存条目持有这些截图字符串的强引用，
# 保证缓存存活期间 id 不会被复用，字符串不可变，因此身份相同即内容相同。
_BUG_FINGERPRINT_CACHE: dict[str, tuple[str, str, tuple[str, ...]]] = {}
# 缺陷详情目录扫描缓存：bugs_dir → {"dirs": {bug_id: bug_dir}, "seen": {目录名}}。
# 只为尚未识别过的目录读取 bug.json；本进程新建目录时直接登记。
_BUG_DIR_SCAN_CACHE: dict[str, dict[str, Any]] = {}


def _bug_fingerprint_inputs(
    bug: dict[str, Any],
    trace: list[dict[str, Any]],
):
    """逐段产出完整指纹的输入：("json", bytes) 或 ("screenshot", str)。"""
    yield "json", json.dumps(
        _json_safe(bug),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    for index, step in enumerate(trace, start=1):
        if not isinstance(step, dict):
            continue
        page = step.get("current_page")
        screenshot = _page_screenshot(page)
        screenshot_ref = f"screenshots/step_{index:03d}.jpg" if screenshot else ""
        step_without_page = {
            key: value for key, value in step.items() if key != "current_page"
        }
        yield "json", json.dumps(
            _json_safe(step_without_page),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        yield "json", json.dumps(
            _serialize_page(page, screenshot_ref),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        yield "screenshot", screenshot


def _bug_artifact_fingerprint_cached(bug: dict[str, Any], trace: list[dict[str, Any]]) -> str:
    """与 _bug_artifact_fingerprint 结果完全一致，但未变化时跳过截图内容哈希。"""
    bug_id = str(bug.get("bug_id", "") or "").strip()
    if not bug_id:
        return _bug_artifact_fingerprint(bug, trace)
    cheap = hashlib.sha256()
    screenshots: list[str] = []
    for kind, value in _bug_fingerprint_inputs(bug, trace):
        if kind == "json":
            cheap.update(value)
        else:
            screenshots.append(value)
            cheap.update(f"\x00{id(value)}:{len(value)}\x00".encode("ascii"))
    cheap_key = cheap.hexdigest()
    cached = _BUG_FINGERPRINT_CACHE.get(bug_id)
    if cached is not None and cached[0] == cheap_key:
        return cached[1]
    fingerprint = _bug_artifact_fingerprint(bug, trace)
    _BUG_FINGERPRINT_CACHE[bug_id] = (cheap_key, fingerprint, tuple(screenshots))
    return fingerprint


def _scan_existing_bug_dirs(bugs_dir: str) -> dict[str, str]:
    """bug_id → 详情目录；只解析新出现的目录，已识别目录被删除时同步移除。"""
    cache = _BUG_DIR_SCAN_CACHE.setdefault(
        os.path.abspath(bugs_dir),
        {"dirs": {}, "seen": set()},
    )
    dirs: dict[str, str] = cache["dirs"]
    seen: set[str] = cache["seen"]
    names = set(os.listdir(bugs_dir))
    for name in list(seen - names):
        seen.discard(name)
        for bug_id, bug_dir in list(dirs.items()):
            if os.path.basename(bug_dir) == name:
                dirs.pop(bug_id, None)
    for name in sorted(names - seen):
        bug_dir = os.path.join(bugs_dir, name)
        bug_path = os.path.join(bug_dir, "bug.json")
        if not os.path.isdir(bug_dir) or not os.path.exists(bug_path):
            continue
        try:
            with open(bug_path, "r", encoding="utf-8") as f:
                existing_bug = json.load(f)
        except Exception:
            continue
        if not isinstance(existing_bug, dict):
            continue
        seen.add(name)
        bug_id = str(existing_bug.get("bug_id", "")).strip()
        if bug_id and bug_id not in dirs:
            dirs[bug_id] = bug_dir
    return dirs


def _register_bug_dir(bugs_dir: str, bug_id: str, bug_dir: str) -> None:
    """本进程刚写好 bug.json 的目录直接登记，下次无需再读回。"""
    cache = _BUG_DIR_SCAN_CACHE.setdefault(
        os.path.abspath(bugs_dir),
        {"dirs": {}, "seen": set()},
    )
    cache["seen"].add(os.path.basename(bug_dir))
    if bug_id:
        cache["dirs"].setdefault(bug_id, bug_dir)


def _bug_artifact_fingerprint(bug: dict[str, Any], trace: list[dict[str, Any]]) -> str:
    """Fingerprint exactly the bug metadata, trace metadata and screenshot content."""
    digest = hashlib.sha256()
    for kind, value in _bug_fingerprint_inputs(bug, trace):
        if kind == "json":
            digest.update(value)
        else:
            digest.update(value.encode("ascii", errors="ignore"))
    return digest.hexdigest()


def _bug_artifacts_unchanged(
    bug_dir: str,
    fingerprint: str,
    trace: list[dict[str, Any]],
) -> bool:
    fingerprint_path = os.path.join(bug_dir, ".artifact-fingerprint")
    try:
        if not os.path.isfile(os.path.join(bug_dir, "bug.json")):
            return False
        if not os.path.isfile(os.path.join(bug_dir, "trace.json")):
            return False
        with open(fingerprint_path, "r", encoding="ascii") as f:
            if f.read().strip() != fingerprint:
                return False
        for index, step in enumerate(trace, start=1):
            if not isinstance(step, dict) or not _page_screenshot(step.get("current_page")):
                continue
            screenshot_path = os.path.join(
                bug_dir,
                "screenshots",
                f"step_{index:03d}.jpg",
            )
            if not os.path.isfile(screenshot_path):
                return False
        return True
    except OSError:
        return False


def _save_bug_folders(
    state: dict,
    output_dir: str,
    detected_bugs: list[dict[str, Any]],
    prefix: str,
):
    if not detected_bugs:
        return
    bug_trace_records = state.get("bug_trace_records", {})
    if not isinstance(bug_trace_records, dict):
        bug_trace_records = {}

    bugs_dir = os.path.join(output_dir, "bugs")
    try:
        os.makedirs(bugs_dir, exist_ok=True)
        updated_count = 0
        unchanged_count = 0
        existing_dirs = _scan_existing_bug_dirs(bugs_dir)

        for index, bug in enumerate(detected_bugs, start=1):
            if not isinstance(bug, dict):
                continue
            bug_id = str(bug.get("bug_id", "")).strip()
            bug_dir = existing_dirs.get(bug_id, "") if bug_id else ""
            if not bug_dir:
                slug = _safe_run_dir_name(bug_id or f"bug{index}")
                bug_dir = os.path.join(bugs_dir, slug)
                suffix = 2
                while os.path.exists(bug_dir) and not os.path.isdir(bug_dir):
                    bug_dir = os.path.join(bugs_dir, f"{slug}_{suffix}")
                    suffix += 1
            os.makedirs(bug_dir, exist_ok=True)
            trace = bug_trace_records.get(bug_id) if bug_id else None
            if not isinstance(trace, list) or not trace:
                if os.path.isfile(os.path.join(bug_dir, "trace.json")):
                    # A missing in-memory association must never erase or replace
                    # an existing bug's evidence with an unrelated current scene.
                    unchanged_count += 1
                    continue
                trace = []
            fingerprint = _bug_artifact_fingerprint_cached(bug, trace)
            if _bug_artifacts_unchanged(bug_dir, fingerprint, trace):
                unchanged_count += 1
                continue

            bug_path = os.path.join(bug_dir, "bug.json")
            _atomic_write_json(bug_path, bug)
            _register_bug_dir(bugs_dir, bug_id, bug_dir)
            _repair_bug_trace_screenshots(trace, bug_dir)
            trace_data = _serialize_trace(trace, bug_dir)
            trace_path = os.path.join(bug_dir, "trace.json")
            _atomic_write_json(trace_path, trace_data)
            _atomic_write_bytes(
                os.path.join(bug_dir, ".artifact-fingerprint"),
                (fingerprint + "\n").encode("ascii"),
            )
            updated_count += 1
        print(
            f"{prefix} bug 详情目录已增量保存: {bugs_dir} "
            f"(更新={updated_count}, 未变化={unchanged_count})"
        )
    except Exception as exc:
        print(f"{prefix} bug 详情目录保存失败: {exc}")


def _load_json_artifact(path: Path, errors: list[str]) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        errors.append(f"{path.name} 无法读取: {exc}")
        return None


def _save_run_result(
    state: dict,
    output_dir: str,
    *,
    exit_code: int | None,
    elapsed_seconds: float,
) -> None:
    _sync_bug_state(state)
    _sync_compliance_state(state)
    ffg = state.get("ffg")
    bugs = state.get("detected_bugs", [])
    if not isinstance(bugs, list):
        bugs = []
    timeline = state.get("_activity_timeline", {})
    activities = (
        [str(item) for item in timeline.get("activities", []) if str(item).strip()]
        if isinstance(timeline, dict)
        else []
    )
    if not activities:
        activities = sorted(_observed_activities(state))
    now = datetime.now().astimezone().isoformat()
    result = {
        "status": "running" if exit_code is None else ("success" if exit_code == 0 else "failed"),
        "exit_code": exit_code,
        "updated_at": now,
        "finished_at": now if exit_code is not None else None,
        "elapsed_seconds": round(max(0.0, elapsed_seconds), 3),
        "app_name": str(state.get("app_name", "") or ""),
        "package_name": str(state.get("package_name", "") or ""),
        "mode": state.get("run_mode", "compliance" if state.get("compliance_detection_enabled", False) else "defect"),
        "reset_app_data": bool(state.get("reset_app_data", False)),
        "grant_permissions": bool(state.get("grant_permissions", False)),
        "permission_grant_report": state.get("permission_grant_report"),
        "seed_provenance": state.get("seed_provenance"),
        "compliance_detection_enabled": bool(state.get("compliance_detection_enabled", False)),
        "compliance_targeted": bool(state.get("compliance_targeted", False)),
        "compliance_summary": (ComplianceChecklist.from_state(state.get("compliance_checklist")).summary(
            state.get("compliance_rules_path") or None
        ) if state.get("compliance_detection_enabled", False) else None),
        "pending_reviews": {"defect": bug_detection_worker.pending_count() if state.get("bug_detection_enabled", True) else 0,
                            "compliance": compliance_worker.pending_count() if state.get("compliance_detection_enabled", False) else 0},
        "bug_detection_enabled": bool(state.get("bug_detection_enabled", True)),
        "defect_review_failures": state.get("defect_review_failures", []),
        "compliance_review_failures": state.get("compliance_review_failures", []),
        "workflow_complete": bool(state.get("workflow_complete", False)),
        # 节点连续异常导致的提前结束原因（正常完成时为空）。
        "workflow_stop_reason": str(state.get("workflow_stop_reason", "") or ""),
        "iteration_count": int(state.get("iteration_count", 0) or 0),
        "error_counts": {
            key: int(state.get(key, 0) or 0)
            for key in (
                "navigation_failure_count",
                "scenario_exec_error_count",
                "scenario_plan_error_count",
                "ffg_update_error_count",
                "screenshot_unavailable_count",
                "target_app_unavailable_count",
            )
        },
        "ffg_counts": _json_safe({
            "functions": len(ffg.functions) if isinstance(ffg, FlowGraph) else 0,
            "pages": len(ffg.pages) if isinstance(ffg, FlowGraph) else 0,
            "page_transitions": len(ffg.page_transitions) if isinstance(ffg, FlowGraph) else 0,
            "function_dependencies": len(ffg.function_dependencies) if isinstance(ffg, FlowGraph) else 0,
            "test_status": ffg.test_status_counts() if isinstance(ffg, FlowGraph) else {},
            "cross_test_status": ffg.cross_test_status_counts() if isinstance(ffg, FlowGraph) else {},
        }),
        "activity_discovery": {
            "count": len(activities),
            "activities": activities,
            "timeline_json": "activity_timeline.json",
            "timeline_image": "activity_timeline.png",
        },
        "high_confidence_bug_count": len([
            bug for bug in bugs if isinstance(bug, dict) and _is_high_confidence_bug(bug)
        ]),
        "llm_usage": get_llm_usage_totals(),
        "fatal_error": _json_safe(state.get("fatal_error")),
    }
    _atomic_write_json(Path(output_dir) / "run_result.json", result)


# 最近一次 manifest 的文件哈希：输出目录 → {相对路径: (size, mtime_ns, sha256)}。
# 产物有错误时 manifest 会连续生成两次（中间只重写 run_result.json），第二次
# 复用第一次的哈希，只重算 run_result.json 以及 size/mtime 变化过的文件。
_MANIFEST_FILE_HASHES: dict[str, dict[str, tuple[int, int, str]]] = {}


def _validate_final_artifacts(
    state: dict,
    output_dir: str,
    exit_code: int,
    *,
    reuse_hashes: dict[str, tuple[int, int, str]] | None = None,
) -> dict[str, Any]:
    """校验最终目录的结构、JSON 可读性和所有被引用文件。"""
    root = Path(output_dir)
    errors: list[str] = []
    warnings: list[str] = []
    if state.get("bug_detection_enabled", True):
        failures = state.get("defect_review_failures", []) or bug_detection_worker.failures_snapshot()
        if failures:
            errors.append(f"缺陷判定有 {len(failures)} 个失败任务，最终产物不完整")
    if state.get("compliance_detection_enabled", False):
        # H4: 合规判定失败此前完全不校验——调用/解析重试耗尽后会被
        # analyze_execution_for_compliance 悄悄写成 undetermined 的"模型未
        # 返回该条款结论"，与缺陷侧不同，产物校验也没有对应项，导致合规判定
        # 全线失败时仍可能被判定为 run_result.status=success。现在与缺陷侧
        # 对齐：compliance_worker 记录的基础设施失败必须体现为产物不完整。
        compliance_failures = (
            state.get("compliance_review_failures", []) or compliance_worker.failures_snapshot()
        )
        if compliance_failures:
            errors.append(f"合规判定有 {len(compliance_failures)} 个失败任务，最终产物不完整")
    required_files = [
        "activity_timeline.json",
        "activity_timeline.png",
        "llm_usage.json",
        "run.log",
        "run_result.json",
    ]
    if state.get("bug_detection_enabled", True):
        required_files.append("bugs.json")
    if state.get("compliance_detection_enabled", False):
        required_files.extend(["compliance_checklist.json", "compliance_report.json", "compliance_report.md"])
    if isinstance(state.get("ffg"), FlowGraph):
        required_files.append("ffg.json")
    for relative in required_files:
        path = root / relative
        if not path.is_file():
            errors.append(f"缺少必需产物: {relative}")
        elif path.stat().st_size <= 0:
            errors.append(f"必需产物为空: {relative}")

    activity_path = root / "activity_timeline.json"
    activity_data = _load_json_artifact(activity_path, errors) if activity_path.is_file() else None
    if isinstance(activity_data, dict):
        activities = activity_data.get("activities")
        points = activity_data.get("points")
        if not isinstance(activities, list) or any(not isinstance(item, str) for item in activities):
            errors.append("activity_timeline.json 的 activities 不是字符串列表")
        if not isinstance(points, list) or not points:
            errors.append("activity_timeline.json 缺少时间采样点")
        elif isinstance(activities, list):
            last = points[-1]
            if not isinstance(last, dict) or int(last.get("activity_count", -1)) != len(activities):
                errors.append("activity_timeline.json 最后采样点与 Activity 总数不一致")
    elif activity_data is not None:
        errors.append("activity_timeline.json 不是对象")
    activity_image_path = root / "activity_timeline.png"
    if activity_image_path.is_file() and cv2.imread(str(activity_image_path)) is None:
        errors.append("activity_timeline.png 无法解码")

    ffg_path = root / "ffg.json"
    if ffg_path.is_file():
        ffg_data = _load_json_artifact(ffg_path, errors)
        if isinstance(ffg_data, dict):
            functions = [item for item in ffg_data.get("functions", []) if isinstance(item, dict)]
            pages = [item for item in ffg_data.get("pages", []) if isinstance(item, dict)]
            page_ids = {str(item.get("id", "")) for item in pages}
            for page in pages:
                page_id = str(page.get("id", "") or "")
                representative = page.get("representative_screenshot", "")
                screenshots = page.get("evidence_screenshots")
                controls = page.get("controls")
                if not isinstance(screenshots, list):
                    errors.append(f"页面 {page_id} 的 evidence_screenshots 不是列表")
                    screenshots = []
                if not isinstance(controls, list) or any(not isinstance(value, str) for value in controls):
                    errors.append(f"页面 {page_id} 的 controls 不是自然语言字符串列表")
                for screenshot_ref in [representative, *screenshots]:
                    if not screenshot_ref:
                        continue
                    screenshot_path = root / "screenshots" / str(screenshot_ref)
                    if not screenshot_path.is_file() or screenshot_path.stat().st_size <= 0:
                        errors.append(f"页面 {page_id} 缺少截图文件: screenshots/{screenshot_ref}")
            for function in functions:
                function_id = str(function.get("id", "") or "")
                page_refs = function.get("page_ids", [])
                if not isinstance(page_refs, list):
                    errors.append(f"功能 {function_id} 的 page_ids 不是列表")
                    continue
                for page_id in page_refs:
                    if str(page_id) not in page_ids:
                        errors.append(f"功能 {function_id} 引用了不存在的页面 {page_id}")
                coverage_percent = function.get("coverage_percent", 0)
                if (
                    not isinstance(coverage_percent, int)
                    or isinstance(coverage_percent, bool)
                    or not 0 <= coverage_percent <= 100
                ):
                    errors.append(
                        f"功能 {function_id} 的 coverage_percent 必须是 0-100 的整数"
                    )

    bugs_path = root / "bugs.json"
    bugs_data = _load_json_artifact(bugs_path, errors) if bugs_path.is_file() else None
    if bugs_data is not None and not isinstance(bugs_data, list):
        errors.append("bugs.json 不是数组")
        bugs_data = []
    bug_detail_dirs: dict[str, Path] = {}
    bugs_root = root / "bugs"
    if bugs_root.is_dir():
        for bug_file in bugs_root.glob("*/bug.json"):
            bug_data = _load_json_artifact(bug_file, errors)
            if isinstance(bug_data, dict):
                bug_id = str(bug_data.get("bug_id", "") or "")
                if bug_id:
                    bug_detail_dirs[bug_id] = bug_file.parent
    for bug in bugs_data or []:
        if not isinstance(bug, dict):
            errors.append("bugs.json 含非对象条目")
            continue
        bug_id = str(bug.get("bug_id", "") or "")
        detail_dir = bug_detail_dirs.get(bug_id)
        if detail_dir is None:
            errors.append(f"缺陷 {bug_id or '<无 ID>'} 缺少详情目录")
            continue
        trace_path = detail_dir / "trace.json"
        trace_data = _load_json_artifact(trace_path, errors) if trace_path.is_file() else None
        if trace_data is None:
            errors.append(f"缺陷 {bug_id} 缺少 trace.json")
            continue
        if not isinstance(trace_data, list):
            errors.append(f"缺陷 {bug_id} 的 trace.json 不是数组")
            continue
        if not trace_data:
            warnings.append(f"缺陷 {bug_id} 的 trace 为空")
        for step in trace_data:
            if not isinstance(step, dict):
                continue
            screenshot_ref = str((step.get("page") or {}).get("screenshot", "") or "")
            if screenshot_ref:
                screenshot_path = detail_dir / screenshot_ref
                if not screenshot_path.is_file() or screenshot_path.stat().st_size <= 0:
                    errors.append(f"缺陷 {bug_id} 缺少轨迹截图: {screenshot_ref}")

    if state.get("compliance_detection_enabled", False):
        checklist_path = root / "compliance_checklist.json"
        report_path = root / "compliance_report.json"
        checklist = _load_json_artifact(checklist_path, errors) if checklist_path.is_file() else None
        report = _load_json_artifact(report_path, errors) if report_path.is_file() else None
        if not isinstance(checklist, dict) or not isinstance(report, dict):
            errors.append("合规清单或报告不是有效 JSON 对象")
        elif not isinstance(report.get("checks"), list):
            errors.append("合规报告缺少 checks 数组")
        else:
            for check in report["checks"]:
                if not isinstance(check, dict):
                    errors.append("合规报告含无效检测点")
                    continue
                relative = str(check.get("trace_path") or "")
                if not relative:
                    if check.get("evidence"):
                        warnings.append(f"旧合规结论 {check.get('rule_id')} 缺少可定位的执行轨迹")
                    continue
                trace_path = (root / relative).resolve()
                if not trace_path.is_relative_to(root.resolve()):
                    errors.append("合规证据引用超出运行目录")
                    continue
                trace = _load_json_artifact(trace_path, errors) if trace_path.is_file() else None
                if not isinstance(trace, dict) or not isinstance(trace.get("steps"), list):
                    errors.append(f"合规结论 {check.get('rule_id')} 缺少有效轨迹: {relative}")
                    continue
                steps = {step.get("step_index"): step for step in trace["steps"] if isinstance(step, dict)}
                for evidence in check.get("evidence", []):
                    step = steps.get(evidence.get("global_step")) if isinstance(evidence, dict) else None
                    screenshot = (step.get("page") or {}).get("screenshot") if step else None
                    image_path = (trace_path.parent / str(screenshot or "")).resolve()
                    if (not screenshot or not image_path.is_relative_to(root.resolve())
                            or not image_path.is_file() or not image_path.stat().st_size):
                        errors.append(f"合规结论 {check.get('rule_id')} 引用了无截图的步骤")

    for label, enabled, worker in (
        ("缺陷", state.get("bug_detection_enabled", True), bug_detection_worker),
        ("合规", state.get("compliance_detection_enabled", False), compliance_worker),
        ("页面审计", True, page_audit_worker),
    ):
        if enabled and worker.pending_count():
            errors.append(f"{label}后台任务尚未排空，最终产物不完整")

    # M4：页面审计队列饱和时的拒收此前完全静默；即使不阻断产物完整性判定，
    # 也要留一条可见的警告，避免"部分页面控件清单缺失"在事后完全无法察觉。
    rejected_page_audits = page_audit_worker.rejected_count()
    if rejected_page_audits:
        warnings.append(
            f"页面审计队列饱和导致 {rejected_page_audits} 次提交被拒收，"
            "部分页面控件清单可能不完整"
        )

    trace_index_path = root / "trace_screenshots" / "index.jsonl"
    trace_screenshot_records = 0
    if trace_index_path.is_file():
        try:
            with trace_index_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        errors.append(
                            f"trace_screenshots/index.jsonl 第 {line_number} 行无法解析: {exc}"
                        )
                        continue
                    if not isinstance(record, dict):
                        errors.append(
                            f"trace_screenshots/index.jsonl 第 {line_number} 行不是对象"
                        )
                        continue
                    trace_screenshot_records += 1
                    screenshot_ref = str(record.get("file", "") or "")
                    screenshot_path = root / screenshot_ref
                    if (
                        not screenshot_ref
                        or not screenshot_path.is_file()
                        or screenshot_path.stat().st_size <= 0
                    ):
                        errors.append(
                            f"原始轨迹截图索引第 {line_number} 行缺少文件: "
                            f"{screenshot_ref or '<空>'}"
                        )
        except OSError as exc:
            errors.append(f"无法读取原始轨迹截图索引: {exc}")

    if exit_code != 0:
        warnings.append("本次运行异常退出；产物是异常发生前的最后一致快照")

    files: list[dict[str, Any]] = []
    file_hashes: dict[str, tuple[int, int, str]] = {}
    for path in sorted(
        item
        for item in root.rglob("*")
        if item.is_file()
        and not item.name.endswith(".tmp")
        and item.name != "artifact_manifest.json"
    ):
        relative = path.relative_to(root).as_posix()
        stat = path.stat()
        size = stat.st_size
        previous = (reuse_hashes or {}).get(relative)
        if (
            previous is not None
            and relative != "run_result.json"
            and previous[0] == size
            and previous[1] == stat.st_mtime_ns
        ):
            digest = previous[2]
        else:
            digest = ""
            if size <= 64 * 1024 * 1024:
                hasher = hashlib.sha256()
                with path.open("rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        hasher.update(chunk)
                digest = hasher.hexdigest()
        file_hashes[relative] = (size, stat.st_mtime_ns, digest)
        files.append({"path": relative, "size_bytes": size, "sha256": digest or None})
    _MANIFEST_FILE_HASHES[str(root.resolve())] = file_hashes

    return {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "complete" if not errors else "incomplete",
        "run_status": "success" if exit_code == 0 else "failed",
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "trace_screenshot_count": trace_screenshot_records,
        "files": files,
    }


def _save_artifact_manifest(
    state: dict,
    output_dir: str,
    exit_code: int,
    *,
    reuse_previous: bool = False,
) -> dict[str, Any]:
    reuse_hashes = (
        _MANIFEST_FILE_HASHES.get(str(Path(output_dir).resolve()))
        if reuse_previous
        else None
    )
    manifest = _validate_final_artifacts(
        state,
        output_dir,
        exit_code,
        reuse_hashes=reuse_hashes,
    )
    _atomic_write_json(Path(output_dir) / "artifact_manifest.json", manifest)
    return manifest


def _should_save_run_snapshot(node_name: str, state: dict) -> bool:
    if node_name == "ffg_update":
        return True
    if node_name != "scenario_plan":
        return False
    scenario_type = (state.get("current_plan", {}) or {}).get("scenario_type")
    # L2：曾支持的 "ffg_audit_update" scenario_type 全仓已无生产者（规划器
    # 从未产出过这个值），是纯死分支；一并从消费端移除，避免误读为
    # "系统里存在这个能力"。
    return scenario_type == "skip"


def _save_run_snapshot(state: dict, output_dir: str):
    """保存运行中间产物：FFG、页面截图和 bugs，不保存断点续跑状态。"""
    _save_outputs(state, output_dir, prefix="[运行快照]")


def _resolve_foreground_package(device) -> str:
    print("[启动] 正在获取当前前台包名...")
    started_at = time.time()
    try:
        package_name = str(device.current_package() or "").strip()
        elapsed = time.time() - started_at
        if package_name:
            print(f"[启动] 当前前台包名: {package_name} ({elapsed:.2f}s)")
        else:
            print(f"[启动] 未获取到当前前台包名 ({elapsed:.2f}s)")
        return package_name
    except Exception as exc:
        print(f"[启动] 获取当前前台包名失败: {exc}")
        return ""


def _env_value(name: str, default: str) -> str:
    return os.getenv(name, default)


def _effective_max_exec_steps() -> int:
    return max(1, int(os.getenv("MAX_EXEC_STEPS", "15")))


def _effective_max_navigation_steps() -> int:
    return max(1, int(os.getenv("MAX_NAVIGATION_STEPS", "10")))


def _graph_recursion_limit(state: dict) -> int:
    """Return a framework guard that cannot pre-empt the configured run limits."""
    configured = str(os.getenv("LANGGRAPH_RECURSION_LIMIT", "")).strip()
    if configured:
        return max(100, int(configured))
    max_iterations = int(state.get("max_iterations", 0) or 0)
    if max_iterations > 0:
        # A workflow iteration crosses several top-level graph nodes. Leave
        # ample room for skipped (no-execution) iterations and final routing.
        return max(1000, max_iterations * 8 + 100)
    # LangGraph requires a recursion guard even when FuncDroid is time-bound.
    # Keep it practically unreachable and expose an env override for operators.
    return 1_000_000


def _safe_run_dir_name(app_name: str) -> str:
    name = str(app_name or "").strip() or "当前前台应用"
    for separator in {os.sep, os.altsep}:
        if separator:
            name = name.replace(separator, "_")
    return name or "当前前台应用"


def _allocate_default_output_dir(app_name: str) -> str:
    base_name = _safe_run_dir_name(app_name)
    root = Path(os.getenv("OUTPUT_ROOT", "").strip() or DEFAULT_OUTPUT_ROOT)
    suffix = 1
    while True:
        dirname = base_name if suffix == 1 else f"{base_name}_{suffix}"
        candidate = root / dirname
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return str(candidate)
        except FileExistsError:
            suffix += 1


def _run_android_shell_best_effort(device, command: str, label: str) -> tuple[str, bool]:
    try:
        return str(device.connector.shell(command) or ""), True
    except Exception as exc:
        print(f"[启动前清理] {label}失败，继续执行: {exc}")
        return "", False


def _reset_android_app_data_at_run_start(device, package_name: str) -> bool:
    """Reset app data once for a brand-new run, never from an inner scenario."""
    package_name = str(package_name or "").strip()
    if not package_name:
        return False
    if getattr(device, "operating_system", "") not in {"adb", "android"}:
        print("[启动前重置] 当前设备不支持自动清数据；请准备初始状态并使用 --no-reset-app-data。")
        return False

    print(f"[启动前重置] 清除目标应用数据（本次运行仅执行一次）: {package_name}")
    output, succeeded = _run_android_shell_best_effort(
        device,
        f"pm clear {package_name}",
        f"清除目标应用数据 {package_name}",
    )
    confirmed = succeeded and "success" in output.strip().lower()
    if confirmed:
        print(f"[启动前重置] pm clear 已确认成功: {output.strip()}")
    else:
        print(
            "[启动前重置] pm clear 未明确成功，拒绝在非全新状态下开始测试: "
            f"{output.strip() or '<空输出>'}"
        )
    return confirmed


def _grant_app_permissions_at_run_start(device, package_name: str) -> dict:
    """Grant the target app's runtime permissions once before the first launch.

    缺陷检测的目标是功能行为，逐条权限弹窗会占用场景步数并可能触发重复索权
    循环；因此在启动前把已申请且未授予的权限一次性补齐。单条授予失败不中止
    运行：设备可能把权限标记为不可变更（USER_FIXED）或使用非标准权限名。
    """
    package_name = str(package_name or "").strip()
    if not package_name:
        print("[启动前授权] 未指定目标包名（测试当前前台应用），跳过预授予权限。")
        return {}
    print(f"[启动前授权] 预授予目标应用运行期权限: {package_name}")
    try:
        report = device.grant_app_permissions(package_name) or {}
    except Exception as exc:  # noqa: BLE001 - 授权失败不应阻断本次运行
        print(f"[启动前授权] 预授予权限异常，继续执行: {exc}")
        return {"granted": [], "missing": [], "failed": [], "unavailable": True}
    if report.get("unavailable"):
        print("[启动前授权] 当前设备不支持自动授予权限，跳过；请手动确认授权状态。")
        return report
    granted = list(report.get("granted", []) or [])
    missing = list(report.get("missing", []) or [])
    failed = list(report.get("failed", []) or [])
    print(
        f"[启动前授权] 已授予 {len(granted)} 项运行期权限"
        + (f"，不可变更/普通权限 {len(missing)} 项" if missing else "")
        + (f"，授予失败 {len(failed)} 项: {', '.join(failed[:5])}" if failed else "")
    )
    if not granted and not missing and not failed:
        print("[启动前授权] 该应用没有可预授予的运行期权限（或权限已全部授予）。")
    return report


def _start_target_app(
    device,
    package_name: str,
    *,
    reset_app_data: bool = False,
    grant_permissions: bool = False,
) -> dict:
    device.stop_app(package_name)
    if reset_app_data and not _reset_android_app_data_at_run_start(device, package_name):
        raise RuntimeError("启动前无法确认目标应用数据已清除，已中止本次测试")
    if not reset_app_data:
        print("[启动] 保留目标应用数据，不执行 pm clear。")
    permission_report: dict = {}
    if grant_permissions:
        permission_report = _grant_app_permissions_at_run_start(device, package_name)
    print(f"[启动] 启动目标应用: {package_name}")
    device.start_app(package_name)
    time.sleep(float(os.getenv("APP_RESTART_DELAY", "2.0")))
    return permission_report


def _llm_timeout_value() -> str:
    # 与 create_llm 使用同一解析逻辑：未配置/非法时为有限默认超时，而非“不超时”。
    return f"{_resolve_request_timeout(None):g}"


def _exec_json_response_format_enabled() -> bool:
    """场景执行是否开启服务端 JSON 模式（EXEC_JSON_RESPONSE_FORMAT，默认开启）。"""
    return os.getenv("EXEC_JSON_RESPONSE_FORMAT", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _ensure_device_id(args) -> None:
    if args.device_id:
        return
    raise ValueError("必须通过 --device-id 显式指定要运行的设备编号，例如: --device-id emulator-5554")


def _init_runtime(args):
    print("[启动] 初始化规划/更新 LLM...")
    planning_llm = create_llm()
    # 场景执行是逐步骤的高频调用，思考强度单独配置（默认 none，关闭思考以降低单步延迟）。
    execution_effort = os.getenv("EXECUTION_REASONING_EFFORT", "none").strip().lower()
    # 场景执行要求纯 JSON 动作：开启服务端 JSON 模式可根治“模型把响应写成
    # 自然语言”的解析失败；端点不支持时执行器会自动关闭（EXEC_JSON_RESPONSE_FORMAT=0）。
    exec_json_format = _exec_json_response_format_enabled()
    print(
        f"[启动] 初始化场景执行 LLM（思考强度 {execution_effort or '服务端默认'}"
        f"{'，JSON 模式' if exec_json_format else ''}）..."
    )
    execution_llm = create_llm(
        reasoning_effort=execution_effort or None,
        json_response_format=exec_json_format,
    )
    # FFG 更新保留所有尚未匹配的轨迹图；本地多模态模型处理完整证据通常
    # 比规划慢，使用独立长超时并由总运行预算限制重试次数。
    ffg_update_timeout = float(os.getenv("FFG_UPDATE_LLM_TIMEOUT", "300") or 300)
    print(f"[启动] 初始化 FFG 更新 LLM（超时 {ffg_update_timeout:g}s）...")
    ffg_update_llm = create_llm(timeout_seconds=ffg_update_timeout)
    if bool(getattr(args, "bug_detection", True)) or bool(getattr(args, "compliance", False)):
        print("[启动] 初始化判定 LLM...")
        bug_detection_llm = create_llm()
    else:
        print("[启动] 合规判定与缺陷检测均关闭，跳过专用判定 LLM 初始化")
        bug_detection_llm = None
    print(f"[启动] 初始化设备驱动: {args.device_id}")
    device = Driver(args.device_id)
    print("[启动] 设备驱动初始化完成")
    return planning_llm, execution_llm, ffg_update_llm, bug_detection_llm, device


def _close_logcat_capture(
    resources: RunResources,
    *,
    announce: bool = True,
    suppress_errors: bool = True,
) -> None:
    if resources.logcat_capture is None:
        return
    try:
        if announce:
            print(f"[logcat] 停止完整 logcat 采集: {resources.logcat_capture.path}")
        resources.logcat_capture.close()
    except Exception as exc:
        if announce:
            print(f"[logcat] 停止完整 logcat 采集失败: {exc}")
        if not suppress_errors:
            raise
    finally:
        resources.logcat_capture = None


def _close_run_log(resources: RunResources) -> None:
    if resources.run_log is not None:
        resources.run_log.close()
        resources.run_log = None


def _resolve_target_app(args, device) -> bool:
    """Resolve the target package and return whether it was explicitly requested."""
    explicit_package = bool(str(args.package_name or "").strip())
    if not explicit_package:
        args.package_name = _resolve_foreground_package(device)
        if not args.package_name:
            raise RuntimeError(
                "未指定 --package-name，且无法识别当前前台 App；"
                "请先在设备上打开目标 App"
            )
        print(f"[启动] 使用当前前台 App: {args.package_name}")
    if not args.app_name:
        args.app_name = args.package_name
    return explicit_package


def _prepare_workflow_state(
    args, device, resources: RunResources, start_time: float,
    resume_state: AgentState | None = None,
) -> AgentState:
    if args.compliance:
        from compliance.rule_library import load_rules

        snapshot_path = Path(resources.output_dir).resolve() / "compliance_rules.json"
        if not bool(getattr(args, "resume_existing", False)) or not snapshot_path.is_file():
            _atomic_write_json(snapshot_path, {"rules": load_rules(args.rules)})
        args.rules = str(snapshot_path)
    if bool(getattr(args, "resume_existing", False)):
        state = resume_state if resume_state is not None else _load_resume_state(args, resources.output_dir, start_time)
        state["compliance_rules_path"] = str(args.rules or "")
        return state
    state = _build_initial_state(args, start_time)
    if resume_state is not None:
        state["ffg"] = resume_state["ffg"]
        state["seed_provenance"] = resume_state["seed_provenance"]
    return state


def _prepare_run_logging(args, device, resources: RunResources) -> None:
    if not resources.output_dir:
        resources.output_dir = _allocate_default_output_dir(
            str(args.app_name or args.package_name or "run")
        )
    if resources.run_log is None:
        resources.run_log = setup_run_logging(resources.output_dir)
        configure_llm_usage(
            resources.output_dir,
            reset=not bool(getattr(args, "resume_existing", False)),
        )
        configure_trace_artifacts(
            resources.output_dir,
            reset=not bool(getattr(args, "resume_existing", False)),
        )
    if resources.logcat_capture is None:
        is_resume = bool(getattr(args, "resume_existing", False))
        resources.logcat_capture = setup_logcat_capture(
            resources.output_dir,
            device,
            clear=not is_resume,
            append=is_resume,
        )


def _print_startup_banner(args, state: dict, resources: RunResources) -> None:
    time_label = f"{args.total_time} 分钟" if args.total_time > 0 else "无限制"
    # 与 workflow/llm.py 的取用顺序保持一致：通用变量优先，其次 DEEPSEEK_* 回退。
    llm_model = (
        os.getenv("MODEL", "").strip()
        or os.getenv("DEEPSEEK_MODEL", "").strip()
        or "未指定"
    )
    llm_timeout = _llm_timeout_value()
    ffg_update_timeout = os.getenv("FFG_UPDATE_LLM_TIMEOUT", "300").strip() or "300"

    print(f"\n{'='*60}")
    print("  FuncDroid 工作流启动")
    print(f"{'='*60}")
    print(f"  应用: {state.get('app_name')} ({state.get('package_name')})")
    print(f"  应用选择: {'指定包名' if args.package_name_explicit else '当前前台 App'}")
    print(f"  设备: {args.device_id}")
    print(f"  规划模式: {PHASE_LABELS[args.phase]}")
    exploration_mode = str(state.get("exploration_mode") or DEFAULT_EXPLORATION_MODE)
    print(
        f"  探索范围: {EXPLORATION_MODE_LABELS.get(exploration_mode, exploration_mode)}"
        f"（{exploration_mode}）"
    )
    boundary_packages = enabled_packages()
    if exploration_mode == "core":
        print(
            "  探索边界: "
            + (
                "已启用: " + ", ".join(boundary_packages)
                if boundary_packages
                else "未启用（boundary_rules.json 无启用规则）"
            )
        )
    print(f"  总时间: {time_label}")
    iteration_label = str(args.max_iterations) if args.max_iterations > 0 else "无限制"
    print(f"  最大轮次: {iteration_label}")
    print(f"  LangGraph 递归预算: {_graph_recursion_limit(state)}")
    print(f"  规划/更新模型: {llm_model}")
    print(f"  场景执行模型: {llm_model}")
    global_effort = os.getenv("LLM_REASONING_EFFORT", "low").strip().lower() or "服务端默认"
    exec_effort = os.getenv("EXECUTION_REASONING_EFFORT", "none").strip().lower() or "服务端默认"
    print(f"  思考强度: 其他阶段={global_effort}, 场景执行={exec_effort}")
    exec_json_format = _exec_json_response_format_enabled()
    print(f"  场景执行 JSON 模式: {'开启（服务端保证合法 JSON）' if exec_json_format else '关闭'}")
    bug_detection_enabled = bool(state.get("bug_detection_enabled", True))
    compliance_enabled = bool(state.get("compliance_detection_enabled", False))
    print(f"  检测模式: {args.mode}")
    print(f"  本次清除应用数据: {args.reset_app_data}")
    print(f"  启动前预授予权限: {bool(getattr(args, 'grant_permissions', False))}")
    if compliance_enabled:
        compliance_rule_count = ComplianceChecklist.from_state(
            state.get("compliance_checklist")
        ).summary(str(state.get("compliance_rules_path", "") or "").strip() or None)["gui_total"]
        print(f"  GUI 合规检测: 开启（{compliance_rule_count} 条规则）")
    print(
        "  缺陷检测: "
        + (f"开启（{llm_model}）" if bug_detection_enabled else "关闭")
    )
    print(
        "  LLM 超时: "
        f"常规={llm_timeout}s, "
        f"FFG更新={ffg_update_timeout}s"
    )
    print(f"  输出目录: {resources.output_dir}")
    print(f"  断点续跑: {'是' if bool(getattr(args, 'resume_existing', False)) else '否'}")
    print(f"  运行日志: {resources.run_log.path}")
    if resources.logcat_capture is not None:
        print(f"  完整 logcat: {resources.logcat_capture.path}")
    print(
        "  探索参数: "
        f"breadth_max_scenarios={_env_value('PLAN_BREADTH_MAX_SCENARIOS', '3')}, "
        f"depth_controls_per_scenario={_env_value('PLAN_DEPTH_CONTROLS_PER_SCENARIO', '5')}, "
        f"depth_scenarios={_env_value('PLAN_MIN_DEPTH_SCENARIOS', '1')}-{_env_value('PLAN_MAX_DEPTH_SCENARIOS', '6')}, "
        f"function_coverage_target={_env_value('PLAN_FUNCTION_COVERAGE_TARGET', '80')}%, "
        f"max_audit_images_per_function={_env_value('PLAN_MAX_AUDIT_IMAGES_PER_FUNCTION', '4')}"
    )
    print(
        "  时间参数: "
        f"action_delay={_env_value('ACTION_DELAY', '0.7')}s, "
        f"fast_action_delay={_env_value('FAST_ACTION_DELAY', '0.15')}s, "
        f"app_restart_delay={_env_value('APP_RESTART_DELAY', '2.0')}s, "
        f"max_exec_steps={_effective_max_exec_steps()}, "
        f"max_navigation_steps={_effective_max_navigation_steps()}"
    )
    print(
        "  检测参数: "
        "ffg_update=执行轨迹增量建图, "
        f"trace_defect_detection={'后台异步线程(全量轨迹+全部截图)' if bug_detection_enabled else 'off'}, "
        "crash_logcat=on, "
        f"full_logcat={'on' if resources.logcat_capture is not None else 'off'}"
    )
    print(f"{'='*60}\n")


def _merge_graph_step(final_state: dict, step: dict) -> str:
    node_name = ", ".join(step.keys()) if step else "unknown"
    for node_payload in step.values():
        if isinstance(node_payload, dict):
            final_state.update(node_payload)
    return node_name


def _compact_log_text(value: object, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _print_graph_step_status(node_name: str, final_state: dict) -> None:
    current_plan = final_state.get("current_plan", {}) or {}
    if node_name == "scenario_plan":
        scenario_type = str(current_plan.get("scenario_type", "") or "none")
        if scenario_type == "cross_test":
            scope = (
                f"{current_plan.get('src_function_id', '') or 'unknown'}→"
                f"{current_plan.get('dst_function_id', '') or 'unknown'}"
            )
        else:
            scope = str(current_plan.get("target_function_id", "") or "none")
        print(f"[状态] scenario_plan | type={scenario_type} | scope={scope}")
        navigation_scenario = _compact_log_text(
            final_state.get("current_navigation_scenario")
        )
        test_scenario = _compact_log_text(final_state.get("current_test_scenario"))
        if navigation_scenario:
            print(f"[计划] 导航: {navigation_scenario}")
        if test_scenario:
            print(f"[计划] 场景: {test_scenario}")
        return

    if node_name == "scenario_exec":
        execution_result = final_state.get("execution_result", {}) or {}
        current_page = final_state.get("current_page")
        page_id = str(getattr(current_page, "id", "") or "unknown")
        print(
            "[状态] scenario_exec | "
            f"navigation={execution_result.get('navigation_status', 'unknown')}"
            f"({int(execution_result.get('navigation_trace_length', 0) or 0)}步) | "
            f"scenario={execution_result.get('scenario_status', 'unknown')}"
            f"({int(execution_result.get('scenario_trace_length', 0) or 0)}步) | "
            f"page={page_id}"
        )
        error = _compact_log_text(
            execution_result.get("scenario_error")
            or execution_result.get("navigation_error")
        )
        if error:
            print(f"[状态] scenario_exec | error={error}")
        return

    if node_name == "ffg_update":
        ffg = final_state.get("ffg")
        function_count = len(ffg.functions) if isinstance(ffg, FlowGraph) else 0
        page_count = len(ffg.pages) if isinstance(ffg, FlowGraph) else 0
        current_function_id = str(
            final_state.get("current_testing_function_id", "") or "none"
        )
        current_function_status = str(
            final_state.get("current_testing_function_status", "") or "not_started"
        )
        print(
            "[状态] ffg_update | "
            f"decision={final_state.get('ffg_update_decision', 'finish')} | "
            f"graph={function_count}F/{page_count}P | "
            f"current_function={current_function_id}({current_function_status})"
        )
        issues = list(final_state.get("ffg_update_validation_issues", []) or [])
        if issues:
            print(f"[状态] ffg_update | validation={'; '.join(map(str, issues[:3]))}")
        return

    print(f"[状态] node={node_name}")


def _save_progress_best_effort(label: str, callback, fallback=None):
    """Keep transient progress-artifact failures outside the graph control flow."""
    try:
        result = callback()
    except Exception as exc:  # noqa: BLE001 - checkpoints must not stop testing
        now = time.monotonic()
        failure = _CHECKPOINT_FAILURES.setdefault(
            label,
            {"count": 0, "last_log_monotonic": 0.0},
        )
        failure["count"] = int(failure.get("count", 0) or 0) + 1
        last_log = float(failure.get("last_log_monotonic", 0.0) or 0.0)
        if last_log == 0 or now - last_log >= _CHECKPOINT_FAILURE_LOG_INTERVAL_SECONDS:
            failure["last_log_monotonic"] = now
            print(
                f"[运行快照] {label} 保存失败，主测试循环继续，后续节点会重试: "
                f"{exc} (累计失败={failure['count']})"
            )
        return fallback
    _CHECKPOINT_FAILURES.pop(label, None)
    return result


def _run_graph(
    graph,
    state: AgentState,
    resources: RunResources,
    final_state: dict | None = None,
) -> dict:
    if final_state is None:
        final_state = {}
    final_state.clear()
    final_state.update(dict(state))
    timeline = final_state.get("_activity_timeline")
    if not isinstance(timeline, dict):
        timeline = _new_activity_timeline(final_state)
    final_state["_activity_timeline"] = timeline

    # Materialize a useful artifact set before the first LLM/device step, then
    # refresh lightweight metadata after every graph node.
    _save_progress_best_effort(
        "FFG/缺陷快照",
        lambda: _save_run_snapshot(final_state, resources.output_dir),
    )
    _save_progress_best_effort("LLM 用量", save_llm_usage_summary)
    timeline = _save_progress_best_effort(
        "Activity 时间线",
        lambda: _save_activity_timeline(
            final_state,
            resources.output_dir,
            timeline,
            force=True,
        ),
        fallback=timeline,
    )
    final_state["_activity_timeline"] = timeline
    _save_progress_best_effort(
        "运行状态",
        lambda: _save_run_result(
            final_state,
            resources.output_dir,
            exit_code=None,
            elapsed_seconds=max(
                0.0,
                time.time() - float(final_state.get("start_time", 0) or time.time()),
            ),
        ),
    )
    stream_config = {"recursion_limit": _graph_recursion_limit(final_state)}
    for step in graph.stream(state, config=stream_config):
        node_name = _merge_graph_step(final_state, step)
        _print_graph_step_status(node_name, final_state)
        # 只有 FFG 更新后或跳过/收尾轮才是有意义的检查点：此时做完整快照、
        # 显式落盘用量与 run_result。其余节点（规划/执行）只走节流的用量保存，
        # 避免每个节点都深拷贝状态、fsync 多个文件。
        checkpoint = _should_save_run_snapshot(node_name, final_state) or bool(
            final_state.get("workflow_complete", False)
        )
        if checkpoint:
            _save_progress_best_effort(
                "FFG/缺陷快照",
                lambda: _save_run_snapshot(final_state, resources.output_dir),
            )
            _save_progress_best_effort("LLM 用量", save_llm_usage_summary)
        else:
            _save_progress_best_effort(
                "LLM 用量",
                lambda: save_llm_usage_summary(durable=False),
            )
        timeline = _save_progress_best_effort(
            "Activity 时间线",
            lambda: _save_activity_timeline(
                final_state,
                resources.output_dir,
                timeline,
            ),
            fallback=timeline,
        )
        final_state["_activity_timeline"] = timeline
        if checkpoint:
            _save_progress_best_effort(
                "运行状态",
                lambda: _save_run_result(
                    final_state,
                    resources.output_dir,
                    exit_code=None,
                    elapsed_seconds=max(
                        0.0,
                        time.time() - float(final_state.get("start_time", 0) or time.time()),
                    ),
                ),
            )
    return final_state


def _save_and_summarize(
    final_state: dict,
    output_dir: str,
    elapsed: float,
    *,
    interrupted: bool = False,
    deadline: float | None = None,
) -> None:
    """收尾落盘。interrupted 时不补交新的页面审计；deadline 缺省为
    现在起 FINALIZE_DEADLINE_SECONDS，约束页面审计的等待与补交总时长。"""
    if deadline is None:
        deadline = time.monotonic() + FINALIZE_DEADLINE_SECONDS
    _flush_worker(bug_detection_worker, "保存最终输出")
    _flush_worker(compliance_worker, "保存最终输出")
    _flush_page_audit(final_state, interrupted=interrupted, deadline=deadline)
    save_llm_usage_summary()
    _save_final_outputs(final_state, output_dir)
    final_state["_activity_timeline"] = _save_activity_timeline(
        final_state,
        output_dir,
        final_state.get("_activity_timeline"),
        force=True,
    )
    _print_summary(final_state, output_dir, elapsed)


def _stop_target_app_processes(args, resources: RunResources) -> None:
    """测试结束后停止目标应用的后台进程。

    连续测试多个 App 时，上一个应用常驻后台会占用内存，也可能在下一个
    App 测试期间被任务切换弹回前台干扰操作。只停止进程（app_stop），
    不清除数据、不卸载；启动前权限与登录态都不受影响。
    """
    device = resources.device
    package_name = str(args.package_name or "").strip()
    if device is None or not package_name:
        return
    try:
        print(f"[收尾] 测试结束，停止目标应用后台进程: {package_name}")
        device.stop_app(package_name)
    except Exception as exc:  # noqa: BLE001 - 收尾清理失败不影响运行结果
        print(f"[收尾] 停止目标应用后台进程失败（忽略继续）: {type(exc).__name__}: {exc}")


def _cleanup_run_resources(resources: RunResources) -> None:
    _close_logcat_capture(resources)
    _close_run_log(resources)


def _build_parser() -> argparse.ArgumentParser:
    parser = RunArgumentParser(description="FuncDroid 独立缺陷检测 / GUI 合规检测")
    parser.add_argument("--mode", choices=["defect", "compliance"], default=None,
                        help="本次检测模式；默认 defect，两种模式必须分开运行")
    parser.add_argument("--reset-app-data", action=argparse.BooleanOptionalAction, default=None,
                        help="启动前清除目标 App 数据；defect 默认保留，compliance 默认清除，续跑始终保留")
    parser.add_argument("--grant-permissions", action=argparse.BooleanOptionalAction, default=None,
                        help="启动前预授予目标 App 的运行期权限；defect 默认授予（专注功能探索），"
                             "compliance 默认不授予（权限申请本身是检测对象）")
    parser.add_argument("--device-id", type=_normalize_device_id, required=True,
                        help="设备 ID；必须显式指定，例如 emulator-5554。多设备请分别开多个终端运行")
    parser.add_argument("--app-name", default=None,
                        help="应用显示名称；默认使用目标包名")
    parser.add_argument("--package-name", default=None,
                        help="设备上已安装的应用包名；不传则测试当前前台打开的 App")
    parser.add_argument("--max-iterations", type=_nonnegative_int,
                        default=_nonnegative_int(os.getenv("MAX_ITERATIONS", str(DEFAULT_MAX_ITERATIONS))),
                        help="最大执行轮次（默认 0，表示不限制）")
    parser.add_argument("--total-time", type=_nonnegative_float,
                        default=_nonnegative_float(os.getenv("TOTAL_TIME", str(DEFAULT_TOTAL_TIME_MINUTES))),
                        help=f"总运行时间限制（分钟，默认 {DEFAULT_TOTAL_TIME_MINUTES}，0 表示不限制）")
    parser.add_argument("--output-dir", default=None,
                        help="新运行的输出目录；必须为空，默认在 experiments/output/ 下按应用名创建独立目录")
    parser.add_argument("--seed-ffg", default=None,
                        help="从成功运行继承页面和路径；新预算、新覆盖状态，保留应用数据")
    parser.add_argument(
        "--resume-output",
        default=None,
        help="从已有输出目录恢复同一张 FFG、Activity 时间线、缺陷和执行历史并继续累计运行",
    )
    parser.add_argument("--phase", choices=["all", "intra"], default="all",
                        help="规划模式: all=自动选择, intra=仅功能内")
    parser.add_argument(
        "--exploration-mode",
        choices=["core"],
        default=os.getenv("EXPLORATION_MODE", DEFAULT_EXPLORATION_MODE).strip() or DEFAULT_EXPLORATION_MODE,
        help="探索模式: core=仅探索目标应用核心功能，"
        "按 boundary_rules.json 拦截小程序等第三方服务内容（默认 core）",
    )
    parser.add_argument(
        "--bug-detection",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="缺陷模式是否进行轨迹判定；默认开启，可用 --no-bug-detection 仅探索",
    )
    parser.add_argument(
        "--compliance",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--rules",
        default=None,
        help="自定义合规规则库 JSON 路径；默认使用 compliance/rules.json",
    )
    parser.add_argument(
        "--compliance-targeted", action="store_true",
        help="只执行规则库中的合规目标；全部目标完成一次判定后结束，证据不足如实保留",
    )
    return parser


def _run_single(args) -> int:
    _ensure_device_id(args)
    resolve_run_options(args)
    validate_resume_options(args)
    if args.compliance:
        from compliance.rule_library import gui_rules

        if not gui_rules(args.rules):
            raise ValueError("合规规则库没有可执行的 GUI 检测条款")
    if args.output_dir and args.resume_output:
        raise ValueError("--output-dir 与 --resume-output 不能同时使用")
    resources = RunResources(output_dir=args.resume_output or args.output_dir or "")
    requested_root = Path(resources.output_dir) if resources.output_dir else None
    existing_ffg = bool(requested_root and (requested_root / "ffg.json").is_file())
    if args.resume_output and not existing_ffg:
        raise FileNotFoundError(f"--resume-output 指定的目录缺少 ffg.json: {requested_root}")
    if (
        args.output_dir
        and requested_root
        and requested_root.exists()
        and any(requested_root.iterdir())
    ):
        raise FileExistsError(
            f"新运行不允许覆盖已有非空目录: {requested_root}；"
            "请换一个新目录，或仅在灾难恢复时使用 --resume-output"
        )
    args.resume_existing = bool(args.resume_output)
    if args.resume_existing:
        print(f"[断点续跑] 显式恢复同一张 FFG: {requested_root}")
    final_state: dict = {}
    start_time = time.time()
    device = None
    exit_code = 0
    outputs_saved = False
    state_ready = False
    interrupted = False
    elapsed = 0.0
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _raise_run_termination)
    try:
        # Validate and read every durable resume input before logging or device
        # initialization can change files or application state.
        state = _load_resume_state(args, resources.output_dir, start_time) if args.resume_existing else None
        if getattr(args, "seed_ffg", None):
            graph, provenance = load_seed_graph(args.seed_ffg, args.package_name)
            state = {"ffg": graph, "seed_provenance": provenance}
        planning_llm, execution_llm, ffg_update_llm, bug_detection_llm, device = _init_runtime(args)
        resources.device = device
        args.package_name_explicit = _resolve_target_app(args, device)
        _prepare_run_logging(args, device, resources)
        permission_report: dict = {}
        if args.package_name_explicit:
            permission_report = _start_target_app(
                device,
                args.package_name,
                reset_app_data=args.reset_app_data,
                grant_permissions=bool(getattr(args, "grant_permissions", False)),
            )
            if permission_report:
                summary = {
                    "granted": len(permission_report.get("granted", []) or []),
                    "missing": len(permission_report.get("missing", []) or []),
                    "failed": len(permission_report.get("failed", []) or []),
                    "unavailable": bool(permission_report.get("unavailable", False)),
                }
                print(f"[启动前授权] 权限授予摘要: {summary}")
        state = _prepare_workflow_state(args, device, resources, start_time, resume_state=state)
        # 必须写在 _prepare_workflow_state 之后：它会重建 state，覆盖之前写入的字段。
        if permission_report:
            state["permission_grant_report"] = dict(permission_report)
        final_state.update(dict(state))
        state_ready = True
        if state.get("bug_detection_enabled", True):
            bug_detection_worker.configure(
                bug_detection_llm,
                on_batch_done=lambda: _save_bugs(final_state, resources.output_dir, prefix="[缺陷检测线程]"),
                initial_bugs=state.get("detected_bugs"),
                initial_records=state.get("bug_trace_records"),
                initial_failures=state.get("defect_review_failures"),
            )
        # 合规判定线程：读取执行轨迹 + 关联检测条款，输出条款级结论。
        if state.get("compliance_detection_enabled", False):
            compliance_worker.configure(
                bug_detection_llm,
                on_trace=lambda job: _save_compliance_trace(job, resources.output_dir),
                on_batch_done=lambda: _save_compliance_outputs(
                    final_state, resources.output_dir, prefix="[合规检测线程]"
                ),
                initial_checklist=state.get("compliance_checklist"),
                initial_failures=state.get("compliance_review_failures"),
            )
        # 页面控件后台审计使用与 FFG 更新相同的规划 LLM；队列与饱和
        # 状态见 workflow/page_audit_worker.py。
        page_audit_worker.configure(planning_llm)
        graph = build_graph(
            planning_llm,
            execution_llm,
            device,
            ffg_update_llm=ffg_update_llm,
        )
        _print_startup_banner(args, state, resources)
        _run_graph(graph, state, resources, final_state)
        if final_state.get("workflow_hard_stop"):
            # 连续异常触发硬上限说明基础设施不可恢复（如设备断连），不能记为成功。
            exit_code = 1
            print(f"[运行异常] {final_state.get('workflow_stop_reason', '节点连续异常')}，本次运行记为失败。")
        elapsed = time.time() - float(final_state.get("start_time", start_time) or start_time)
        _save_and_summarize(final_state, resources.output_dir, elapsed)
        outputs_saved = True
        print(f"[完成] 工作流执行结束，输出保存在: {resources.output_dir}")
    except (KeyboardInterrupt, RunTerminationSignal) as exc:
        exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 128 + exc.signum
        elapsed = time.time() - float(final_state.get("start_time", start_time) or start_time)
        final_state["fatal_error"] = {
            "type": type(exc).__name__,
            "message": "用户中断" if isinstance(exc, KeyboardInterrupt) else str(exc),
            "traceback": "",
        }
        interrupted = True
        print(f"[运行中断] 收到停止请求，正在保存当前 FFG、缺陷和用量数据（exit={exit_code}）。")
    except Exception as exc:
        exit_code = 1
        elapsed = time.time() - float(final_state.get("start_time", start_time) or start_time)
        final_state["fatal_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(f"[运行异常] 工作流遇到未捕获异常，已安全退出: {exc}")
        print(traceback.format_exc())
        # 不在这里保存：统一交给 finally 中带异常保护的保存路径，保证
        # 保存本身再抛异常时 _run_single 仍返回退出码而不是向上传播。
    finally:
        # 收尾落盘期间暂缓 SIGTERM：此时再收到终止信号若直接按默认行为杀进程，
        # 会在写 FFG/run_result/manifest 途中留下半成品。保存完成后再恢复原处理器。
        deferred_sigterms: list[int] = []

        def _defer_sigterm(signum, _frame) -> None:
            deferred_sigterms.append(int(signum))

        signal.signal(signal.SIGTERM, _defer_sigterm)
        try:
            exit_code = _finalize_run(
                args,
                resources,
                final_state,
                start_time=start_time,
                elapsed=elapsed,
                exit_code=exit_code,
                outputs_saved=outputs_saved,
                state_ready=state_ready,
                interrupted=interrupted,
            )
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm_handler)
            if deferred_sigterms:
                print(
                    f"[运行中断] 收尾保存期间收到 {len(deferred_sigterms)} 次终止信号，"
                    "已在保存完成后退出。"
                )
    return exit_code


def _finalize_run(
    args,
    resources: RunResources,
    final_state: dict,
    *,
    start_time: float,
    elapsed: float,
    exit_code: int,
    outputs_saved: bool,
    state_ready: bool,
    interrupted: bool,
) -> int:
    """运行收尾：保存最终输出、run_result 与 manifest，返回最终退出码。"""
    elapsed = elapsed or (
        time.time() - float(final_state.get("start_time", start_time) or start_time)
    )
    can_save = bool(resources.output_dir) and (state_ready or not args.resume_existing)
    if resources.output_dir and final_state and not outputs_saved and can_save:
        # _save_and_summarize 自身会等待判定线程排空，这里不再重复等待。
        try:
            _save_and_summarize(
                final_state,
                resources.output_dir,
                elapsed,
                interrupted=interrupted,
            )
        except Exception as exc:
            exit_code = 1
            print(f"[输出] 异常快照保存失败: {exc}")
    elif not outputs_saved:
        # 不保存最终输出时仍给后台判定一次排空机会（与原行为一致）。
        _flush_worker(bug_detection_worker, "生成最终运行结果")
        _flush_worker(compliance_worker, "生成最终运行结果")
    _stop_target_app_processes(args, resources)
    _cleanup_run_resources(resources)
    if resources.output_dir and can_save:
        try:
            save_llm_usage_summary()
            final_state.setdefault("start_time", start_time)
            artifact_root = Path(resources.output_dir)
            ffg_missing = (
                isinstance(final_state.get("ffg"), FlowGraph)
                and not (artifact_root / "ffg.json").is_file()
            )
            if ffg_missing or (final_state.get("bug_detection_enabled", True) and not (artifact_root / "bugs.json").is_file()):
                _save_outputs(final_state, resources.output_dir, prefix="[输出修复]")
            final_state["_activity_timeline"] = _save_activity_timeline(
                final_state,
                resources.output_dir,
                final_state.get("_activity_timeline"),
                force=True,
            )
            _save_run_result(
                final_state,
                resources.output_dir,
                exit_code=exit_code,
                elapsed_seconds=elapsed,
            )
            manifest = _save_artifact_manifest(
                final_state,
                resources.output_dir,
                exit_code,
            )
            if manifest["errors"] and exit_code == 0:
                exit_code = 1
                _save_run_result(
                    final_state,
                    resources.output_dir,
                    exit_code=exit_code,
                    elapsed_seconds=elapsed,
                )
                manifest = _save_artifact_manifest(
                    final_state,
                    resources.output_dir,
                    exit_code,
                    reuse_previous=True,
                )
            print(
                "[产物校验] "
                f"status={manifest['status']}, files={len(manifest['files'])}, "
                f"errors={len(manifest['errors'])}, warnings={len(manifest['warnings'])}"
            )
            for error in manifest["errors"]:
                print(f"[产物校验] 错误: {error}")
        except Exception as exc:
            exit_code = 1
            print(f"[产物校验] 生成 manifest 失败: {exc}")
    return exit_code


def main():
    parser = _build_parser()
    args = parser.parse_args()

    _ensure_device_id(args)
    try:
        exit_code = _run_single(args)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        # 启动前的参数/目录校验失败：给出一行可读原因，不打印内部调用栈。
        print(f"[参数错误] {exc}", file=sys.stderr)
        sys.exit(2)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
