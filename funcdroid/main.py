"""FuncDroid entrypoint for independent functional bug detection and GUI compliance testing.

Examples:
  python main.py --device-id emulator-5554
  python main.py --device-id emulator-5554 --package-name com.example.app
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
from model.flow_graph import FunctionalFlowGraph, Page  # noqa: E402
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
DEFAULT_MAX_ITERATIONS = 0  # Zero disables the iteration limit; the time budget remains active.
DEFAULT_EXPLORATION_MODE = "core"
DEFAULT_OUTPUT_ROOT = "experiments/output"
ACTIVITY_TIMELINE_HEARTBEAT_SECONDS = max(
    1.0,
    float(os.getenv("ACTIVITY_TIMELINE_HEARTBEAT_SECONDS", "300")),
)
# Background bug review reads captured screenshots without occupying the device.
# Drain the queue before shutdown so the last review results can be persisted.
# A bounded timeout prevents stalled model calls from blocking process exit.
BUG_DETECTION_FLUSH_TIMEOUT_SECONDS = max(
    0.0,
    float(os.getenv("BUG_DETECTION_FLUSH_TIMEOUT_SECONDS", "180")),
)
# Total shutdown budget in seconds for waiting on and submitting page audits.
# Submissions invoke the model, so large graphs need a bounded finalization budget.
FINALIZE_DEADLINE_SECONDS = max(
    0.0,
    float(os.getenv("FINALIZE_DEADLINE_SECONDS", "300")),
)
# Serialize read-merge-write operations on bugs.json and bug detail directories.
# Background review callbacks can otherwise race with main-thread snapshots.
_BUGS_SAVE_LOCK = threading.Lock()
_COMPLIANCE_SAVE_LOCK = threading.Lock()
_CHECKPOINT_FAILURE_LOG_INTERVAL_SECONDS = max(
    1.0,
    float(os.getenv("CHECKPOINT_FAILURE_LOG_INTERVAL_SECONDS", "60")),
)
_CHECKPOINT_FAILURES: dict[str, dict[str, float | int]] = {}
PHASE_LABELS = {
    "all": "Automatically select intra-functionality or inter-functional scenarios",
    "intra": "Intra-functionality testing only",
}
EXPLORATION_MODE_LABELS = {
    "core": "Core functionality testing",
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
        raise argparse.ArgumentTypeError("Specify one device ID; run separate terminals for multiple devices.")
    return text


def _nonnegative_int(value: object) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Must be a nonnegative integer") from exc
    if normalized < 0:
        raise argparse.ArgumentTypeError("Must be a nonnegative integer")
    return normalized


def _nonnegative_float(value: object) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Must be a finite nonnegative number") from exc
    if normalized < 0:
        raise argparse.ArgumentTypeError("Must be a finite nonnegative number")
    return normalized


def _build_initial_state(args, start_time: float) -> AgentState:
    """Build the initial workflow state."""
    total_time_limit = args.total_time * 60 if args.total_time > 0 else 0

    # RunArgumentParser and _run_single already normalize these options.
    return AgentState(
        run_mode=args.mode,
        reset_app_data=args.reset_app_data,
        grant_permissions=bool(args.grant_permissions),
        permission_grant_report=None,
        app_name=args.app_name or args.package_name or "Foreground app",
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
        ffg=FunctionalFlowGraph(),
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
        raise FileNotFoundError(f"Resume directory is missing ffg.json: {ffg_path}")
    state["ffg"] = FunctionalFlowGraph.load(output_dir)

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
            raise ValueError("defect_review_failures.json must be an array")
        state["defect_review_failures"] = failures

    compliance_failures_path = root / "compliance_review_failures.json"
    if compliance_failures_path.is_file():
        compliance_failures = json.loads(compliance_failures_path.read_text(encoding="utf-8"))
        if not isinstance(compliance_failures, list):
            raise ValueError("compliance_review_failures.json must be an array")
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
                    "[Resume] Reset interrupted compliance checks to pending: "
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
        "[Resume] Restored cumulative state: "
        f"elapsed={previous_elapsed / 60:.1f} minutes, "
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
            raise ValueError(f"Bug trace must be an array: {trace_path}")
        trace = []
        for saved in serialized:
            if not isinstance(saved, dict):
                raise ValueError(f"Bug trace contains an invalid step: {trace_path}")
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
                    raise ValueError(f"Bug screenshot reference escapes the run directory: {trace_path}")
                page["screenshot"] = base64.b64encode(image_path.read_bytes()).decode("ascii")
            step["current_page"] = page
            trace.append(step)
        records[bug_id] = trace
    return records


def _atomic_write_json(path: Path | str, data: Any) -> None:
    """Write JSON atomically using a temporary file and os.replace to avoid truncated artifacts."""
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
    if isinstance(graph, FunctionalFlowGraph):
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
    # Zero is a valid count. Using `or -1` would report a change on every empty graph.
    # That would redraw the chart and append timeline samples at every workflow node.
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
    """Refresh state mirrors from the authoritative background bug worker."""
    if not bool(state.get("bug_detection_enabled", True)):
        return
    detected_bugs, bug_trace_records = bug_detection_worker.snapshot()
    state["detected_bugs"] = detected_bugs
    state["bug_trace_records"] = bug_trace_records
    state["defect_review_failures"] = bug_detection_worker.failures_snapshot()


def _flush_worker(worker, reason: str) -> None:
    """Drain a background review queue with a bounded shutdown timeout."""
    pending = worker.pending_count()
    if pending <= 0:
        return
    print(f"{worker.log_prefix} {reason}: waiting for {pending} pending asynchronous jobs...")
    # Zero means no wait; `or None` would incorrectly turn it into an unlimited wait.
    if not worker.flush(timeout=BUG_DETECTION_FLUSH_TIMEOUT_SECONDS):
        recorded = worker.record_unfinished(f"{reason}: background review timed out and remains incomplete")
        print(
            f"{worker.log_prefix} timed out with jobs still running; "
            f"recorded {recorded} unfinished jobs as incomplete review failures."
        )


def _sync_compliance_state(state: dict) -> None:
    """Refresh the checklist from the authoritative background compliance worker."""
    if not bool(state.get("compliance_detection_enabled", False)):
        return
    state["compliance_checklist"] = compliance_worker.snapshot()
    state["compliance_review_failures"] = compliance_worker.failures_snapshot()


def _save_compliance_outputs(state: dict, output_dir: str, prefix: str):
    """Persist the compliance checklist and final JSON/Markdown reports."""
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
            f"{prefix} compliance_report saved: "
            f"processed={summary.get('processed', summary.get('executed', 0))}/"
            f"{summary.get('gui_total', 0)}, "
            f"resolved={summary.get('resolved', 0)}, "
            f"unresolved={summary.get('unresolved', summary.get('deferred', 0))}, "
            f"violation={summary.get('violation', 0)}, "
            f"compliant={summary.get('compliant', 0)}"
        )
    except Exception as exc:
        print(f"{prefix} failed to save compliance_report: {exc}")


def _flush_page_audit(
    final_state: dict,
    *,
    interrupted: bool = False,
    deadline: float | None = None,
) -> None:
    """Drain page audits and merge their results before persisting the final graph.

    The next FFG update normally merges completed audits. At shutdown there is
    no next iteration, so the final batch must be merged here. Interrupted runs
    only merge completed results. Normal waiting and additional submissions
    share the supplied monotonic deadline.
    """
    graph = final_state.get("ffg")
    if isinstance(graph, FunctionalFlowGraph):
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
                        f"[Page audit] Run interrupted: merge completed audits only; "
                        f"do not wait for {pending} in-flight jobs or submit new audits."
                    )
                return

            # The updater submits audits before applying the current trace, so the final
            # pages and newly attached screenshots may still need submission. Worker
            # fingerprints and saturation checks prevent duplicate submissions.
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
                        f"[Page audit] Waiting before final output for {pending} "
                        " pending audit jobs..."
                    )
                    page_audit_worker.flush(timeout=timeout)
                _merge_page_audit_results(graph)
                if deadline is not None and time.monotonic() >= deadline:
                    print(
                        "[Page audit] Finalization deadline reached "
                        f"FINALIZE_DEADLINE_SECONDS={FINALIZE_DEADLINE_SECONDS:g}s,"
                        "Stop submitting remaining unaudited pages."
                    )
                    break
                submitted = _submit_page_audit_batch(
                    graph,
                    exclude_fingerprints=final_submitted_fingerprints,
                )
                if submitted <= 0:
                    break
                print(
                    f"[Page audit] Submitted final batch of {submitted} unaudited page fingerprints."
                )
        except Exception as exc:  # noqa: BLE001 - Best-effort persistence must not block process exit.
            print(f"[Page audit] Failed to merge final audit results: {exc}")


def _print_summary(state: dict, output_dir: str, elapsed_seconds: float):
    """Print the final run summary."""
    _sync_bug_state(state)
    elapsed_min = elapsed_seconds / 60
    ffg = state.get("ffg")

    print(f"\n{'='*60}")
    print("  FuncDroid run summary")
    print(f"{'='*60}")
    print(f"  Total duration: {elapsed_min:.1f} minutes")
    exploration_mode = str(state.get("exploration_mode") or DEFAULT_EXPLORATION_MODE)
    print(f"  Exploration scope: {EXPLORATION_MODE_LABELS.get(exploration_mode, exploration_mode)}")

    if isinstance(ffg, FunctionalFlowGraph):
        print(f"  FFG functionalities: {len(ffg.functions)}")
        print(f"  FFG pages: {len(ffg.pages)}")
        print(f"  Page transitions: {len(ffg.page_transitions)}")
        print(f"  Inter-functional resource dependencies: {len(ffg.function_dependencies)}")
        status_counts = ffg.test_status_counts()
        print(
            "  FFG status distribution: "
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
    print(f"  Observed Android activities: {len(activities)}")

    bugs = state.get("detected_bugs", [])
    if not isinstance(bugs, list):
        bugs = []
    if bugs:
        crash_bugs = [b for b in bugs if isinstance(b, dict) and b.get("type") == "crash"]
        non_crash_bugs = [b for b in bugs if isinstance(b, dict) and b.get("type") != "crash"]
        print(f"  Detected bugs: {len(bugs)}")
        print(f"    - Crash bugs: {len(crash_bugs)}")
        print(f"    - Non-crash bugs: {len(non_crash_bugs)}")
    elif state.get("bug_detection_enabled", True):
        print("  Detected bugs: 0")
    else:
        print("  Bug detection: not run")

    if bool(state.get("compliance_detection_enabled", False)):
        try:
            summary = ComplianceChecklist.from_state(
                state.get("compliance_checklist")
            ).summary(str(state.get("compliance_rules_path", "") or "").strip() or None)
            print(
                "  Compliance checkpoints: "
                f"{summary.get('processed', summary.get('executed', 0))}/"
                f"{summary['gui_total']} processed | "
                f"conclusive {summary.get('resolved', 0)} | "
                f"compliant {summary['compliant']} | violations {summary['violation']} | "
                f"insufficient evidence {summary['undetermined']} | not applicable {summary['not_applicable']} | "
                f"awaiting new evidence {summary.get('unresolved', summary['deferred'])} | "
                f"not processed {summary.get('untested', summary['pending'])}"
            )
        except Exception as exc:
            print(f"  Failed to summarize compliance checks: {exc}")

    print(f"  Iterations: {state.get('iteration_count', 0)}")
    print(f"  Original trace screenshots: {trace_screenshot_count()} (trace_screenshots/)")
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
    print(f"  Output directory: {output_dir}")
    print("  Activity timeline: activity_timeline.json / activity_timeline.png")
    print(f"{'='*60}\n")


def _save_final_outputs(state: dict, output_dir: str):
    """Persist final outputs without a resumable workflow execution checkpoint."""
    _save_outputs(state, output_dir, prefix="[Output]")


def _save_outputs(state: dict, output_dir: str, prefix: str):
    os.makedirs(output_dir, exist_ok=True)
    ffg = state.get("ffg")
    if isinstance(ffg, FunctionalFlowGraph):
        try:
            ffg.save(output_dir)
        except Exception as exc:
            print(f"{prefix} failed to save FFG: {exc}")
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
            print(f"{prefix} bugs.json saved: {bugs_path}")
        except Exception as exc:
            print(f"{prefix} failed to save bugs.json: {exc}")
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
        print(f"{prefix} filtered bugs without high confidence: {skipped_count}")
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


# Bug artifact fingerprint cache: bug_id -> (cheap key, fingerprint, screenshots).
# Snapshots repeatedly fingerprint all bugs; hashing base64 screenshots dominates.
# The cheap key includes every non-image input with identical JSON serialization.
# Screenshot identity uses (id, length), retaining strong references in the cache.
# These immutable strings cannot reuse object IDs while the cache retains them.
_BUG_FINGERPRINT_CACHE: dict[str, tuple[str, str, tuple[str, ...]]] = {}
# Detail directory cache: bugs_dir -> {"dirs": {bug_id: bug_dir}, "seen": names}.
# Read bug.json only for new directories; register local writes directly.
_BUG_DIR_SCAN_CACHE: dict[str, dict[str, Any]] = {}


def _bug_fingerprint_inputs(
    bug: dict[str, Any],
    trace: list[dict[str, Any]],
):
    """Yield fingerprint inputs as ("json", bytes) or ("screenshot", str) segments."""
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
    """Match _bug_artifact_fingerprint while skipping unchanged screenshot hashes."""
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
    """Map bug IDs to detail directories, reading new entries and dropping deleted ones."""
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
    """Register a newly written bug directory without rereading its bug.json."""
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
            f"{prefix} incrementally saved bug detail directories: {bugs_dir} "
            f"(updated={updated_count}, unchanged={unchanged_count})"
        )
    except Exception as exc:
        print(f"{prefix} failed to save bug detail directories: {exc}")


def _load_json_artifact(path: Path, errors: list[str]) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        errors.append(f"{path.name} cannot be read: {exc}")
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
        # Reason for a hard stop after consecutive node errors; empty on normal completion.
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
            "functions": len(ffg.functions) if isinstance(ffg, FunctionalFlowGraph) else 0,
            "pages": len(ffg.pages) if isinstance(ffg, FunctionalFlowGraph) else 0,
            "page_transitions": len(ffg.page_transitions) if isinstance(ffg, FunctionalFlowGraph) else 0,
            "function_dependencies": len(ffg.function_dependencies) if isinstance(ffg, FunctionalFlowGraph) else 0,
            "test_status": ffg.test_status_counts() if isinstance(ffg, FunctionalFlowGraph) else {},
            "cross_test_status": ffg.cross_test_status_counts() if isinstance(ffg, FunctionalFlowGraph) else {},
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


# Manifest hash cache: output directory -> {relative path: (size, mtime_ns, sha256)}.
# Artifact errors trigger a second manifest after rewriting run_result.json.
# Reuse cached hashes for other files whose size and modification time are unchanged.
_MANIFEST_FILE_HASHES: dict[str, dict[str, tuple[int, int, str]]] = {}


def _validate_final_artifacts(
    state: dict,
    output_dir: str,
    exit_code: int,
    *,
    reuse_hashes: dict[str, tuple[int, int, str]] | None = None,
) -> dict[str, Any]:
    """Validate final output structure, readable JSON, and all referenced files."""
    root = Path(output_dir)
    errors: list[str] = []
    warnings: list[str] = []
    if state.get("bug_detection_enabled", True):
        failures = state.get("defect_review_failures", []) or bug_detection_worker.failures_snapshot()
        if failures:
            errors.append(f"Bug review has {len(failures)} failed jobs; final artifacts are incomplete")
    if state.get("compliance_detection_enabled", False):
        # Compliance infrastructure failures must invalidate the final artifacts too.
        # Exhausted model or parsing retries can leave checks undetermined, which
        # must not silently hide a failed review behind a successful run status.
        # Apply the same completeness rule used for bug review.
        # The worker retains these failures independently of checkpoint conclusions.
        compliance_failures = (
            state.get("compliance_review_failures", []) or compliance_worker.failures_snapshot()
        )
        if compliance_failures:
            errors.append(f"Compliance review has {len(compliance_failures)} failed jobs; final artifacts are incomplete")
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
    if isinstance(state.get("ffg"), FunctionalFlowGraph):
        required_files.append("ffg.json")
    for relative in required_files:
        path = root / relative
        if not path.is_file():
            errors.append(f"Missing required artifact: {relative}")
        elif path.stat().st_size <= 0:
            errors.append(f"Required artifact is empty: {relative}")

    activity_path = root / "activity_timeline.json"
    activity_data = _load_json_artifact(activity_path, errors) if activity_path.is_file() else None
    if isinstance(activity_data, dict):
        activities = activity_data.get("activities")
        points = activity_data.get("points")
        if not isinstance(activities, list) or any(not isinstance(item, str) for item in activities):
            errors.append("activity_timeline.json activities must be a list of strings")
        if not isinstance(points, list) or not points:
            errors.append("activity_timeline.json is missing time samples")
        elif isinstance(activities, list):
            last = points[-1]
            if not isinstance(last, dict) or int(last.get("activity_count", -1)) != len(activities):
                errors.append("activity_timeline.json final sample disagrees with the activity count")
    elif activity_data is not None:
        errors.append("activity_timeline.json must be an object")
    activity_image_path = root / "activity_timeline.png"
    if activity_image_path.is_file() and cv2.imread(str(activity_image_path)) is None:
        errors.append("activity_timeline.png cannot be decoded")

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
                    errors.append(f"Page {page_id} evidence_screenshots must be a list")
                    screenshots = []
                if not isinstance(controls, list) or any(not isinstance(value, str) for value in controls):
                    errors.append(f"Page {page_id} controls must be a list of natural-language strings")
                for screenshot_ref in [representative, *screenshots]:
                    if not screenshot_ref:
                        continue
                    screenshot_path = root / "screenshots" / str(screenshot_ref)
                    if not screenshot_path.is_file() or screenshot_path.stat().st_size <= 0:
                        errors.append(f"Page {page_id} is missing screenshot file: screenshots/{screenshot_ref}")
            for function in functions:
                function_id = str(function.get("id", "") or "")
                page_refs = function.get("page_ids", [])
                if not isinstance(page_refs, list):
                    errors.append(f"Functionality {function_id} page_ids must be a list")
                    continue
                for page_id in page_refs:
                    if str(page_id) not in page_ids:
                        errors.append(f"Functionality {function_id} references a missing page {page_id}")
                coverage_percent = function.get("coverage_percent", 0)
                if (
                    not isinstance(coverage_percent, int)
                    or isinstance(coverage_percent, bool)
                    or not 0 <= coverage_percent <= 100
                ):
                    errors.append(
                        f"Functionality {function_id} coverage_percent must be an integer from 0 to 100"
                    )

    bugs_path = root / "bugs.json"
    bugs_data = _load_json_artifact(bugs_path, errors) if bugs_path.is_file() else None
    if bugs_data is not None and not isinstance(bugs_data, list):
        errors.append("bugs.json must be an array")
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
            errors.append("bugs.json contains a non-object entry")
            continue
        bug_id = str(bug.get("bug_id", "") or "")
        detail_dir = bug_detail_dirs.get(bug_id)
        if detail_dir is None:
            errors.append(f"Bug {bug_id or '<no ID>'} is missing its detail directory")
            continue
        trace_path = detail_dir / "trace.json"
        trace_data = _load_json_artifact(trace_path, errors) if trace_path.is_file() else None
        if trace_data is None:
            errors.append(f"Bug {bug_id} is missing trace.json")
            continue
        if not isinstance(trace_data, list):
            errors.append(f"Bug {bug_id} trace.json must be an array")
            continue
        if not trace_data:
            warnings.append(f"Bug {bug_id} has an empty trace")
        for step in trace_data:
            if not isinstance(step, dict):
                continue
            screenshot_ref = str((step.get("page") or {}).get("screenshot", "") or "")
            if screenshot_ref:
                screenshot_path = detail_dir / screenshot_ref
                if not screenshot_path.is_file() or screenshot_path.stat().st_size <= 0:
                    errors.append(f"Bug {bug_id} is missing a trace screenshot: {screenshot_ref}")

    if state.get("compliance_detection_enabled", False):
        checklist_path = root / "compliance_checklist.json"
        report_path = root / "compliance_report.json"
        checklist = _load_json_artifact(checklist_path, errors) if checklist_path.is_file() else None
        report = _load_json_artifact(report_path, errors) if report_path.is_file() else None
        if not isinstance(checklist, dict) or not isinstance(report, dict):
            errors.append("Compliance checklist and report must be valid JSON objects")
        elif not isinstance(report.get("checks"), list):
            errors.append("Compliance report is missing the checks array")
        else:
            for check in report["checks"]:
                if not isinstance(check, dict):
                    errors.append("Compliance report contains an invalid checkpoint")
                    continue
                relative = str(check.get("trace_path") or "")
                if not relative:
                    if check.get("evidence"):
                        warnings.append(f"Legacy compliance conclusion {check.get('rule_id')} lacks a locatable execution trace")
                    continue
                trace_path = (root / relative).resolve()
                if not trace_path.is_relative_to(root.resolve()):
                    errors.append("Compliance evidence reference escapes the run directory")
                    continue
                trace = _load_json_artifact(trace_path, errors) if trace_path.is_file() else None
                if not isinstance(trace, dict) or not isinstance(trace.get("steps"), list):
                    errors.append(f"Compliance conclusion {check.get('rule_id')} is missing a valid trace: {relative}")
                    continue
                steps = {step.get("step_index"): step for step in trace["steps"] if isinstance(step, dict)}
                for evidence in check.get("evidence", []):
                    step = steps.get(evidence.get("global_step")) if isinstance(evidence, dict) else None
                    screenshot = (step.get("page") or {}).get("screenshot") if step else None
                    image_path = (trace_path.parent / str(screenshot or "")).resolve()
                    if (not screenshot or not image_path.is_relative_to(root.resolve())
                            or not image_path.is_file() or not image_path.stat().st_size):
                        errors.append(f"Compliance conclusion {check.get('rule_id')} references a step without a screenshot")

    for label, enabled, worker in (
        ("Bug review", state.get("bug_detection_enabled", True), bug_detection_worker),
        ("Compliance review", state.get("compliance_detection_enabled", False), compliance_worker),
        ("Page audit", True, page_audit_worker),
    ):
        if enabled and worker.pending_count():
            errors.append(f"{label} still has background jobs; final artifacts are incomplete")

    # Surface saturated-queue rejections even when they do not invalidate artifacts.
    # Otherwise missing page-control audits would be invisible after the run.
    rejected_page_audits = page_audit_worker.rejected_count()
    if rejected_page_audits:
        warnings.append(
            f"Page audit queue saturation rejected {rejected_page_audits} submissions; "
            "some page control inventories may be incomplete"
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
                            f"trace_screenshots/index.jsonl line {line_number} cannot be parsed: {exc}"
                        )
                        continue
                    if not isinstance(record, dict):
                        errors.append(
                            f"trace_screenshots/index.jsonl line {line_number} is not an object"
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
                            f"Original screenshot index line {line_number} references a missing file: "
                            f"{screenshot_ref or '<empty>'}"
                        )
        except OSError as exc:
            errors.append(f"Cannot read the original screenshot index: {exc}")

    if exit_code != 0:
        warnings.append("Run exited abnormally; artifacts contain the last consistent snapshot before the error")

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
    # The planner no longer produces the obsolete ffg_audit_update scenario type.
    # Its dead consumer branch was removed as well.
    # Do not imply that an unimplemented workflow capability is available.
    return scenario_type == "skip"


def _save_run_snapshot(state: dict, output_dir: str):
    """Persist intermediate FFG, screenshot, and bug artifacts without a workflow checkpoint."""
    _save_outputs(state, output_dir, prefix="[Snapshot]")


def _resolve_foreground_package(device) -> str:
    print("[Startup] Reading the foreground package...")
    started_at = time.time()
    try:
        package_name = str(device.current_package() or "").strip()
        elapsed = time.time() - started_at
        if package_name:
            print(f"[Startup] Foreground package: {package_name} ({elapsed:.2f}s)")
        else:
            print(f"[Startup] Could not determine the foreground package ({elapsed:.2f}s)")
        return package_name
    except Exception as exc:
        print(f"[Startup] Failed to read the foreground package: {exc}")
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
    name = str(app_name or "").strip() or "Foreground app"
    for separator in {os.sep, os.altsep}:
        if separator:
            name = name.replace(separator, "_")
    return name or "Foreground app"


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
        print(f"[Startup cleanup] {label} failed; continuing: {exc}")
        return "", False


def _reset_android_app_data_at_run_start(device, package_name: str) -> bool:
    """Reset app data once for a brand-new run, never from an inner scenario."""
    package_name = str(package_name or "").strip()
    if not package_name:
        return False
    if getattr(device, "operating_system", "") not in {"adb", "android"}:
        print("[Startup reset] Device cannot clear app data automatically; prepare its state and use --no-reset-app-data.")
        return False

    print(f"[Startup reset] Clear target app data once for this run: {package_name}")
    output, succeeded = _run_android_shell_best_effort(
        device,
        f"pm clear {package_name}",
        f"Clear target app data {package_name}",
    )
    confirmed = succeeded and "success" in output.strip().lower()
    if confirmed:
        print(f"[Startup reset] pm clear confirmed success: {output.strip()}")
    else:
        print(
            "[Startup reset] pm clear did not confirm success; cannot start from an unverified initial state: "
            f"{output.strip() or '<empty output>'}"
        )
    return confirmed


def _grant_app_permissions_at_run_start(device, package_name: str) -> dict:
    """Grant the target app runtime permissions once before the first launch.

    Functional testing grants requested permissions up front so repeated dialogs
    do not consume scenario steps. Individual failures do not abort the run:
    permissions may be USER_FIXED or use nonstandard names.
    """
    package_name = str(package_name or "").strip()
    if not package_name:
        print("[Startup permissions] No explicit target package; skip permission pre-granting.")
        return {}
    print(f"[Startup permissions] Pre-grant runtime permissions for: {package_name}")
    try:
        report = device.grant_app_permissions(package_name) or {}
    except Exception as exc:  # noqa: BLE001 - Individual permission failures must not abort exploration.
        print(f"[Startup permissions] Pre-granting failed; continuing: {exc}")
        return {"granted": [], "missing": [], "failed": [], "unavailable": True}
    if report.get("unavailable"):
        print("[Startup permissions] Automatic granting is unsupported; verify permissions manually.")
        return report
    granted = list(report.get("granted", []) or [])
    missing = list(report.get("missing", []) or [])
    failed = list(report.get("failed", []) or [])
    print(
        f"[Startup permissions] Granted {len(granted)} runtime permissions"
        + (f", fixed or normal permissions {len(missing)}" if missing else "")
        + (f", grant failures {len(failed)}: {', '.join(failed[:5])}" if failed else "")
    )
    if not granted and not missing and not failed:
        print("[Startup permissions] No runtime permissions need pre-granting.")
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
        raise RuntimeError("Could not confirm cleared app data before startup; aborting this run")
    if not reset_app_data:
        print("[Startup] Preserve target app data; do not run pm clear.")
    permission_report: dict = {}
    if grant_permissions:
        permission_report = _grant_app_permissions_at_run_start(device, package_name)
    print(f"[Startup] Launch target app: {package_name}")
    device.start_app(package_name)
    time.sleep(float(os.getenv("APP_RESTART_DELAY", "2.0")))
    return permission_report


def _llm_timeout_value() -> str:
    # Match create_llm: absent or invalid configuration uses a finite default timeout.
    return f"{_resolve_request_timeout(None):g}"


def _exec_json_response_format_enabled() -> bool:
    """Whether scenario execution requests server-side JSON mode (enabled by default)."""
    return os.getenv("EXEC_JSON_RESPONSE_FORMAT", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _ensure_device_id(args) -> None:
    if args.device_id:
        return
    raise ValueError("Specify a device explicitly with --device-id, for example --device-id emulator-5554")


def _init_runtime(args):
    print("[Startup] Initialize the planning/update model...")
    planning_llm = create_llm()
    # Configure execution reasoning separately; the default none minimizes per-step latency.
    execution_effort = os.getenv("EXECUTION_REASONING_EFFORT", "none").strip().lower()
    # Request server-side JSON mode for action responses to avoid prose-only output.
    # The executor disables this request automatically for unsupported endpoints.
    exec_json_format = _exec_json_response_format_enabled()
    print(
        f"[Startup] Initialize the execution model (reasoning effort {execution_effort or 'server default'}"
        f"{', JSON mode' if exec_json_format else ''})..."
    )
    execution_llm = create_llm(
        reasoning_effort=execution_effort or None,
        json_response_format=exec_json_format,
    )
    # FFG updates retain all unmatched trace images and can be slower than planning.
    # Use a separate timeout while the total run budget still bounds retries.
    ffg_update_timeout = float(os.getenv("FFG_UPDATE_LLM_TIMEOUT", "300") or 300)
    print(f"[Startup] Initialize the FFG update model (timeout {ffg_update_timeout:g}s)...")
    ffg_update_llm = create_llm(timeout_seconds=ffg_update_timeout)
    if bool(getattr(args, "bug_detection", True)) or bool(getattr(args, "compliance", False)):
        print("[Startup] Initialize the review model...")
        bug_detection_llm = create_llm()
    else:
        print("[Startup] Compliance and bug review are disabled; skip the dedicated review model")
        bug_detection_llm = None
    print(f"[Startup] Initialize device driver: {args.device_id}")
    device = Driver(args.device_id)
    print("[Startup] Device driver initialized")
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
            print(f"[logcat] Stop full logcat capture: {resources.logcat_capture.path}")
        resources.logcat_capture.close()
    except Exception as exc:
        if announce:
            print(f"[logcat] Failed to stop full logcat capture: {exc}")
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
                "No --package-name was supplied and the foreground app could not be identified; "
                "open the target app on the device first"
            )
        print(f"[Startup] Use the foreground app: {args.package_name}")
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
    time_label = f"{args.total_time} minutes" if args.total_time > 0 else "unlimited"
    # Match workflow/llm.py: generic settings take precedence over DEEPSEEK_* fallbacks.
    llm_model = (
        os.getenv("MODEL", "").strip()
        or os.getenv("DEEPSEEK_MODEL", "").strip()
        or "unspecified"
    )
    llm_timeout = _llm_timeout_value()
    ffg_update_timeout = os.getenv("FFG_UPDATE_LLM_TIMEOUT", "300").strip() or "300"

    print(f"\n{'='*60}")
    print("  FuncDroid workflow startup")
    print(f"{'='*60}")
    print(f"  App: {state.get('app_name')} ({state.get('package_name')})")
    print(f"  App selection: {'explicit package' if args.package_name_explicit else 'foreground app'}")
    print(f"  Device: {args.device_id}")
    print(f"  Planning scope: {PHASE_LABELS[args.phase]}")
    exploration_mode = str(state.get("exploration_mode") or DEFAULT_EXPLORATION_MODE)
    print(
        f"  Exploration scope: {EXPLORATION_MODE_LABELS.get(exploration_mode, exploration_mode)}"
        f"({exploration_mode})"
    )
    boundary_packages = enabled_packages()
    if exploration_mode == "core":
        print(
            "  Exploration boundary: "
            + (
                "enabled: " + ", ".join(boundary_packages)
                if boundary_packages
                else "disabled (no active boundary_rules.json entries)"
            )
        )
    print(f"  Time budget: {time_label}")
    iteration_label = str(args.max_iterations) if args.max_iterations > 0 else "unlimited"
    print(f"  Maximum iterations: {iteration_label}")
    print(f"  LangGraph recursion budget: {_graph_recursion_limit(state)}")
    print(f"  Planning/update model: {llm_model}")
    print(f"  Execution model: {llm_model}")
    global_effort = os.getenv("LLM_REASONING_EFFORT", "low").strip().lower() or "server default"
    exec_effort = os.getenv("EXECUTION_REASONING_EFFORT", "none").strip().lower() or "server default"
    print(f"  Reasoning effort: other stages={global_effort}, execution={exec_effort}")
    exec_json_format = _exec_json_response_format_enabled()
    print(f"  Execution JSON mode: {'enabled (server guarantees valid JSON)' if exec_json_format else 'disabled'}")
    bug_detection_enabled = bool(state.get("bug_detection_enabled", True))
    compliance_enabled = bool(state.get("compliance_detection_enabled", False))
    print(f"  Detection mode: {args.mode}")
    print(f"  Clear app data for this run: {args.reset_app_data}")
    print(f"  Pre-grant permissions before launch: {bool(getattr(args, 'grant_permissions', False))}")
    if compliance_enabled:
        compliance_rule_count = ComplianceChecklist.from_state(
            state.get("compliance_checklist")
        ).summary(str(state.get("compliance_rules_path", "") or "").strip() or None)["gui_total"]
        print(f"  GUI compliance testing: enabled ({compliance_rule_count} rules)")
    print(
        "  Bug detection: "
        + (f"enabled ({llm_model})" if bug_detection_enabled else "disabled")
    )
    print(
        "  Model timeouts: "
        f"standard={llm_timeout}s, "
        f"FFG update={ffg_update_timeout}s"
    )
    print(f"  Output directory: {resources.output_dir}")
    print(f"  Resume existing run: {'yes' if bool(getattr(args, 'resume_existing', False)) else 'no'}")
    print(f"  Run log: {resources.run_log.path}")
    if resources.logcat_capture is not None:
        print(f"  Full logcat: {resources.logcat_capture.path}")
    print(
        "  Exploration parameters: "
        f"breadth_max_scenarios={_env_value('PLAN_BREADTH_MAX_SCENARIOS', '3')}, "
        f"depth_controls_per_scenario={_env_value('PLAN_DEPTH_CONTROLS_PER_SCENARIO', '5')}, "
        f"depth_scenarios={_env_value('PLAN_MIN_DEPTH_SCENARIOS', '1')}-{_env_value('PLAN_MAX_DEPTH_SCENARIOS', '6')}, "
        f"function_coverage_target={_env_value('PLAN_FUNCTION_COVERAGE_TARGET', '80')}%, "
        f"max_audit_images_per_function={_env_value('PLAN_MAX_AUDIT_IMAGES_PER_FUNCTION', '4')}"
    )
    print(
        "  Timing parameters: "
        f"action_delay={_env_value('ACTION_DELAY', '0.7')}s, "
        f"fast_action_delay={_env_value('FAST_ACTION_DELAY', '0.15')}s, "
        f"app_restart_delay={_env_value('APP_RESTART_DELAY', '2.0')}s, "
        f"max_exec_steps={_effective_max_exec_steps()}, "
        f"max_navigation_steps={_effective_max_navigation_steps()}"
    )
    print(
        "  Detection parameters: "
        "ffg_update=incremental graph construction from execution traces, "
        f"trace_defect_detection={'background worker (complete traces and screenshots)' if bug_detection_enabled else 'off'}, "
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
        print(f"[State] scenario_plan | type={scenario_type} | scope={scope}")
        navigation_scenario = _compact_log_text(
            final_state.get("current_navigation_scenario")
        )
        test_scenario = _compact_log_text(final_state.get("current_test_scenario"))
        if navigation_scenario:
            print(f"[Plan] Navigation: {navigation_scenario}")
        if test_scenario:
            print(f"[Plan] Scenario: {test_scenario}")
        return

    if node_name == "scenario_exec":
        execution_result = final_state.get("execution_result", {}) or {}
        current_page = final_state.get("current_page")
        page_id = str(getattr(current_page, "id", "") or "unknown")
        print(
            "[State] scenario_exec | "
            f"navigation={execution_result.get('navigation_status', 'unknown')}"
            f"({int(execution_result.get('navigation_trace_length', 0) or 0)} steps) | "
            f"scenario={execution_result.get('scenario_status', 'unknown')}"
            f"({int(execution_result.get('scenario_trace_length', 0) or 0)} steps) | "
            f"page={page_id}"
        )
        error = _compact_log_text(
            execution_result.get("scenario_error")
            or execution_result.get("navigation_error")
        )
        if error:
            print(f"[State] scenario_exec | error={error}")
        return

    if node_name == "ffg_update":
        ffg = final_state.get("ffg")
        function_count = len(ffg.functions) if isinstance(ffg, FunctionalFlowGraph) else 0
        page_count = len(ffg.pages) if isinstance(ffg, FunctionalFlowGraph) else 0
        current_function_id = str(
            final_state.get("current_testing_function_id", "") or "none"
        )
        current_function_status = str(
            final_state.get("current_testing_function_status", "") or "not_started"
        )
        print(
            "[State] ffg_update | "
            f"decision={final_state.get('ffg_update_decision', 'finish')} | "
            f"graph={function_count}F/{page_count}P | "
            f"current_function={current_function_id}({current_function_status})"
        )
        issues = list(final_state.get("ffg_update_validation_issues", []) or [])
        if issues:
            print(f"[State] ffg_update | validation={'; '.join(map(str, issues[:3]))}")
        return

    print(f"[State] node={node_name}")


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
                f"[Snapshot] {label} failed to save; exploration continues and later nodes will retry: "
                f"{exc} (cumulative failures={failure['count']})"
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
        "FFG/bug snapshot",
        lambda: _save_run_snapshot(final_state, resources.output_dir),
    )
    _save_progress_best_effort("LLM usage", save_llm_usage_summary)
    timeline = _save_progress_best_effort(
        "Activity timeline",
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
        "Run status",
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
        # Full checkpoints are useful after FFG updates and skip/finalization rounds.
        # Persist usage and run_result there. Planning/execution only save throttled usage,
        # avoiding a deep state copy and multiple fsync calls at every workflow node.
        checkpoint = _should_save_run_snapshot(node_name, final_state) or bool(
            final_state.get("workflow_complete", False)
        )
        if checkpoint:
            _save_progress_best_effort(
                "FFG/bug snapshot",
                lambda: _save_run_snapshot(final_state, resources.output_dir),
            )
            _save_progress_best_effort("LLM usage", save_llm_usage_summary)
        else:
            _save_progress_best_effort(
                "LLM usage",
                lambda: save_llm_usage_summary(durable=False),
            )
        timeline = _save_progress_best_effort(
            "Activity timeline",
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
                "Run status",
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
    """Persist final outputs. Interrupted runs do not submit new page audits.
    The default monotonic deadline bounds total final audit waiting and submission."""
    if deadline is None:
        deadline = time.monotonic() + FINALIZE_DEADLINE_SECONDS
    _flush_worker(bug_detection_worker, "save final outputs")
    _flush_worker(compliance_worker, "save final outputs")
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
    """Stop the target app process after testing.

    This releases memory and prevents a previous app from interrupting the next
    run. Stop only the process: preserve app data, installation, permissions,
    and the account session.
    """
    device = resources.device
    package_name = str(args.package_name or "").strip()
    if device is None or not package_name:
        return
    try:
        print(f"[Finalize] Stop target app background processes: {package_name}")
        device.stop_app(package_name)
    except Exception as exc:  # noqa: BLE001 - Best-effort process cleanup does not change the run result.
        print(f"[Finalize] Failed to stop target app processes; continuing: {type(exc).__name__}: {exc}")


def _cleanup_run_resources(resources: RunResources) -> None:
    _close_logcat_capture(resources)
    _close_run_log(resources)


def _build_parser() -> argparse.ArgumentParser:
    parser = RunArgumentParser(description="FuncDroid functional bug detection / GUI compliance testing")
    parser.add_argument("--mode", choices=["defect", "compliance"], default=None,
                        help="Detection mode (default: defect); run each mode separately")
    parser.add_argument("--reset-app-data", action=argparse.BooleanOptionalAction, default=None,
                        help="Clear target app data before launch; default off for defect, on for compliance, always off when resuming")
    parser.add_argument("--grant-permissions", action=argparse.BooleanOptionalAction, default=None,
                        help="Pre-grant runtime permissions; enabled by default for defect testing, "
                             "disabled for compliance testing of consent and permission requests")
    parser.add_argument("--device-id", type=_normalize_device_id, required=True,
                        help="Required device ID, e.g. emulator-5554; use separate terminals for multiple devices")
    parser.add_argument("--app-name", default=None,
                        help="App display name (default: package name)")
    parser.add_argument("--package-name", default=None,
                        help="Installed app package; omit to test the current foreground app")
    parser.add_argument("--max-iterations", type=_nonnegative_int,
                        default=_nonnegative_int(os.getenv("MAX_ITERATIONS", str(DEFAULT_MAX_ITERATIONS))),
                        help="Maximum iterations (default: 0, unlimited)")
    parser.add_argument("--total-time", type=_nonnegative_float,
                        default=_nonnegative_float(os.getenv("TOTAL_TIME", str(DEFAULT_TOTAL_TIME_MINUTES))),
                        help=f"Total time budget in minutes (default: {DEFAULT_TOTAL_TIME_MINUTES}, 0 means unlimited)")
    parser.add_argument("--output-dir", default=None,
                        help="Empty output directory for a new run; defaults to an app-specific directory under experiments/output/")
    parser.add_argument("--seed-ffg", default=None,
                        help="Inherit pages and paths from a successful run with fresh budgets and coverage; preserve app data")
    parser.add_argument(
        "--resume-output",
        default=None,
        help="Resume the same FFG, activity timeline, bugs, and execution history from an existing output directory",
    )
    parser.add_argument("--phase", choices=["all", "intra"], default="all",
                        help="Planning scope: all=automatic, intra=intra-functionality only")
    parser.add_argument(
        "--exploration-mode",
        choices=["core"],
        default=os.getenv("EXPLORATION_MODE", DEFAULT_EXPLORATION_MODE).strip() or DEFAULT_EXPLORATION_MODE,
        help="Exploration mode: core tests target app functionalities; "
        "boundary_rules.json excludes third-party services such as embedded mini apps (default: core)",
    )
    parser.add_argument(
        "--bug-detection",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Review execution traces for bugs (default: enabled); use --no-bug-detection for exploration only",
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
        help="Custom compliance rule JSON path (default: compliance/rules.json)",
    )
    parser.add_argument(
        "--compliance-targeted", action="store_true",
        help="Only run rule-library compliance targets; stop after each has a verdict, retaining insufficient-evidence results",
    )
    return parser


def _run_single(args) -> int:
    _ensure_device_id(args)
    resolve_run_options(args)
    validate_resume_options(args)
    if args.compliance:
        from compliance.rule_library import gui_rules

        if not gui_rules(args.rules):
            raise ValueError("Compliance rule library has no executable GUI checkpoints")
    if args.output_dir and args.resume_output:
        raise ValueError("--output-dir and --resume-output cannot be used together")
    resources = RunResources(output_dir=args.resume_output or args.output_dir or "")
    requested_root = Path(resources.output_dir) if resources.output_dir else None
    existing_ffg = bool(requested_root and (requested_root / "ffg.json").is_file())
    if args.resume_output and not existing_ffg:
        raise FileNotFoundError(f"--resume-output directory is missing ffg.json: {requested_root}")
    if (
        args.output_dir
        and requested_root
        and requested_root.exists()
        and any(requested_root.iterdir())
    ):
        raise FileExistsError(
            f"A new run cannot overwrite a nonempty output directory: {requested_root};"
            "Choose a new directory, or use --resume-output for recovery"
        )
    args.resume_existing = bool(args.resume_output)
    if args.resume_existing:
        print(f"[Resume] Explicitly restore the same FFG: {requested_root}")
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
                print(f"[Startup permissions] Grant summary: {summary}")
        state = _prepare_workflow_state(args, device, resources, start_time, resume_state=state)
        # Set this after _prepare_workflow_state, which rebuilds and replaces state fields.
        if permission_report:
            state["permission_grant_report"] = dict(permission_report)
        final_state.update(dict(state))
        state_ready = True
        if state.get("bug_detection_enabled", True):
            bug_detection_worker.configure(
                bug_detection_llm,
                on_batch_done=lambda: _save_bugs(final_state, resources.output_dir, prefix="[Bug review]"),
                initial_bugs=state.get("detected_bugs"),
                initial_records=state.get("bug_trace_records"),
                initial_failures=state.get("defect_review_failures"),
            )
        # Review associated compliance rules against execution traces in the background.
        if state.get("compliance_detection_enabled", False):
            compliance_worker.configure(
                bug_detection_llm,
                on_trace=lambda job: _save_compliance_trace(job, resources.output_dir),
                on_batch_done=lambda: _save_compliance_outputs(
                    final_state, resources.output_dir, prefix="[Compliance review]"
                ),
                initial_checklist=state.get("compliance_checklist"),
                initial_failures=state.get("compliance_review_failures"),
            )
        # Page-control audits share the planning model used for FFG updates.
        # See workflow/page_audit_worker.py for queue and saturation behavior.
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
            # Consecutive node errors indicate unrecoverable infrastructure, not a successful run.
            exit_code = 1
            print(f"[Run error] {final_state.get('workflow_stop_reason', 'consecutive node failures')}; marking this run as failed.")
        elapsed = time.time() - float(final_state.get("start_time", start_time) or start_time)
        _save_and_summarize(final_state, resources.output_dir, elapsed)
        outputs_saved = True
        print(f"[Complete] Workflow finished; outputs saved to: {resources.output_dir}")
    except (KeyboardInterrupt, RunTerminationSignal) as exc:
        exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 128 + exc.signum
        elapsed = time.time() - float(final_state.get("start_time", start_time) or start_time)
        final_state["fatal_error"] = {
            "type": type(exc).__name__,
            "message": "User interrupted" if isinstance(exc, KeyboardInterrupt) else str(exc),
            "traceback": "",
        }
        interrupted = True
        print(f"[Interrupted] Stop requested; saving FFG, bugs, and usage (exit={exit_code})。")
    except Exception as exc:
        exit_code = 1
        elapsed = time.time() - float(final_state.get("start_time", start_time) or start_time)
        final_state["fatal_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(f"[Run error] Workflow exited after an uncaught exception: {exc}")
        print(traceback.format_exc())
        # Final persistence is handled by the guarded finally path below.
        # Even if saving fails, _run_single must return an exit code rather than propagate.
    finally:
        # Defer SIGTERM during final persistence to avoid partially written artifacts.
        # Restore the previous handler after FFG/run_result/manifest writes finish.
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
                    f"[Interrupted] Received {len(deferred_sigterms)} termination signals while saving; "
                    "exited after persistence completed."
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
    """Persist final outputs, run_result, and manifest, then return the final exit code."""
    elapsed = elapsed or (
        time.time() - float(final_state.get("start_time", start_time) or start_time)
    )
    can_save = bool(resources.output_dir) and (state_ready or not args.resume_existing)
    if resources.output_dir and final_state and not outputs_saved and can_save:
        # _save_and_summarize already drains review workers; do not wait twice.
        try:
            _save_and_summarize(
                final_state,
                resources.output_dir,
                elapsed,
                interrupted=interrupted,
            )
        except Exception as exc:
            exit_code = 1
            print(f"[Output] Failed to save the exception snapshot: {exc}")
    elif not outputs_saved:
        # Allow review workers one drain attempt even when final outputs are not saved.
        _flush_worker(bug_detection_worker, "generate the final run result")
        _flush_worker(compliance_worker, "generate the final run result")
    _stop_target_app_processes(args, resources)
    _cleanup_run_resources(resources)
    if resources.output_dir and can_save:
        try:
            save_llm_usage_summary()
            final_state.setdefault("start_time", start_time)
            artifact_root = Path(resources.output_dir)
            ffg_missing = (
                isinstance(final_state.get("ffg"), FunctionalFlowGraph)
                and not (artifact_root / "ffg.json").is_file()
            )
            if ffg_missing or (final_state.get("bug_detection_enabled", True) and not (artifact_root / "bugs.json").is_file()):
                _save_outputs(final_state, resources.output_dir, prefix="[Output repair]")
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
                "[Artifact validation] "
                f"status={manifest['status']}, files={len(manifest['files'])}, "
                f"errors={len(manifest['errors'])}, warnings={len(manifest['warnings'])}"
            )
            for error in manifest["errors"]:
                print(f"[Artifact validation] Error: {error}")
        except Exception as exc:
            exit_code = 1
            print(f"[Artifact validation] Failed to generate manifest: {exc}")
    return exit_code


def main():
    parser = _build_parser()
    args = parser.parse_args()

    _ensure_device_id(args)
    try:
        exit_code = _run_single(args)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        # Report startup validation failures concisely without an internal traceback.
        print(f"[Argument error] {exc}", file=sys.stderr)
        sys.exit(2)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
