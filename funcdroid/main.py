"""Pure defect detection workflow entrypoint.

Usage:
  # Test the App currently open in the foreground on the device
  python main.py --device-id emulator-5554

  # Test a specified App installed on the device
  python main.py --device-id emulator-5554 --package-name com.example.app

  # Use iteration control instead of time control
  python main.py --device-id emulator-5554 --max-iterations 20 --total-time 0
"""

import argparse
import base64
import hashlib
import json
import os
import re
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

# Load repository-local configuration before module-level and CLI defaults are
# evaluated. Explicit paths make startup independent of the caller's cwd.
PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(Path(__file__).with_name(".env"))
load_dotenv(PROJECT_DIR / ".env")

from device import Driver  # noqa: E402
from connector import ADB, HDC  # noqa: E402
from model.flow_graph import FlowGraph, Page  # noqa: E402
from workflow.bug_detection_worker import bug_detection_worker  # noqa: E402
from workflow.exploration_boundary import enabled_packages  # noqa: E402
from workflow.graph import build_graph  # noqa: E402
from workflow.page_audit_worker import page_audit_worker  # noqa: E402
from workflow.llm import create_llm  # noqa: E402
from workflow.llm_usage import (  # noqa: E402
    configure_llm_usage,
    get_llm_usage_summary,
    save_llm_usage_summary,
)
from workflow.run_logging import setup_logcat_capture, setup_run_logging  # noqa: E402
from workflow.state import AgentState  # noqa: E402
from workflow.trace_artifacts import (  # noqa: E402
    configure_trace_artifacts,
    trace_screenshot_count,
)

DEFAULT_TOTAL_TIME_MINUTES = 90
DEFAULT_MAX_ITERATIONS = 0  # 0 disables the iteration limit; total runtime is used by default
DEFAULT_EXPLORATION_MODE = "core"
DEFAULT_OUTPUT_ROOT = "output"
ACTIVITY_TIMELINE_HEARTBEAT_SECONDS = max(
    1.0,
    float(os.getenv("ACTIVITY_TIMELINE_HEARTBEAT_SECONDS", "300")),
)
# Defect detection runs in an independent background thread (see workflow/bug_detection_worker.py), reading only already captured
# screenshots without occupying the device; before workflow completion/abnormal exit, wait for the queue to drain to prevent missing results from the last few rounds
# from being persisted to disk. A timeout is set to prevent an abnormal LLM call hang from blocking process exit.
BUG_DETECTION_FLUSH_TIMEOUT_SECONDS = max(
    0.0,
    float(os.getenv("BUG_DETECTION_FLUSH_TIMEOUT_SECONDS", "180")),
)
# Protect read-merge-write of bugs.json / bugs/ directory: defect detection threads persist asynchronously after finishing a batch,
# which may overlap with snapshot persistence on the main thread; a lock is required to avoid concurrent write corruption.
_BUGS_SAVE_LOCK = threading.Lock()
_CHECKPOINT_FAILURE_LOG_INTERVAL_SECONDS = max(
    1.0,
    float(os.getenv("CHECKPOINT_FAILURE_LOG_INTERVAL_SECONDS", "60")),
)
_CHECKPOINT_FAILURES: dict[str, dict[str, float | int]] = {}
PHASE_LABELS = {
    "all": "Automatically select intra-function/cross-function",
    "intra": "Intra-function testing only",
}
EXPLORATION_MODE_LABELS = {
    "core": "Core testing",
}
RECENT_TASK_ID_PATTERNS = (
    re.compile(r"\btaskId=(\d+)\b"),
    re.compile(r"\bid=(\d+)\b"),
    re.compile(r"TaskRecord\{[^}]*\s#(\d+)\s"),
)


@dataclass
class RunResources:
    output_dir: str = ""
    run_log: Any = None
    logcat_capture: Any = None


class RunTerminationSignal(Exception):
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
        raise argparse.ArgumentTypeError("Please pass a specific device ID; for multiple devices, run in separate terminals")
    return text


def _nonnegative_int(value: object) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Must be an integer greater than or equal to 0") from exc
    if normalized < 0:
        raise argparse.ArgumentTypeError("Must be an integer greater than or equal to 0")
    return normalized


def _nonnegative_float(value: object) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Must be a number greater than or equal to 0") from exc
    if normalized < 0:
        raise argparse.ArgumentTypeError("Must be a number greater than or equal to 0")
    return normalized


def _build_initial_state(args, start_time: float) -> AgentState:
    """Build initial state."""
    total_time_limit = args.total_time * 60 if args.total_time > 0 else 0

    return AgentState(
        app_name=args.app_name or args.package_name or "Current foreground app",
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
        bug_detection_enabled=bool(getattr(args, "bug_detection", True)),
        ffg=FlowGraph(),
        navigation_failure_count=0,
        scenario_exec_error_count=0,
        scenario_plan_error_count=0,
        ffg_update_error_count=0,
        screenshot_unavailable_count=0,
        target_app_unavailable_count=0,
        function_execution_failure_counts={},
        cross_execution_failure_counts={},
        planner_noop_count=0,
        planner_feedback="",
        planner_rejected_scenario="",
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
        completion_audit_updates=0,
        completion_audit_signatures={},
        completion_audit_attempts={},
    )


def _load_resume_state(args, output_dir: str, started_at: float) -> AgentState:
    """Restore durable exploration state while starting a fresh graph execution."""
    state = _build_initial_state(args, started_at)
    root = Path(output_dir)
    ffg_path = root / "ffg.json"
    if not ffg_path.is_file():
        raise FileNotFoundError(f"Resume directory missing ffg.json: {ffg_path}")
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
    state["start_time"] = started_at - previous_elapsed

    bugs_path = root / "bugs.json"
    if bugs_path.is_file():
        with bugs_path.open("r", encoding="utf-8") as f:
            bugs = json.load(f)
        if isinstance(bugs, list):
            state["detected_bugs"] = bugs

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


def _atomic_write_json(path: Path | str, data: Any) -> None:
    """Atomically write JSON: write to temporary file first then os.replace to avoid leaving truncated files if killed during execution."""
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
    count_changed = int(last.get("activity_count", -1) or -1) != len(activities)
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
            and int(last.get("activity_count", -1)) == len(activities)
            and abs(float(last.get("elapsed_seconds", 0) or 0) - elapsed_seconds) < 1.0
        ):
            points[-1] = point
        else:
            points.append(point)
        _atomic_write_json(root / "activity_timeline.json", timeline)
        _render_activity_timeline(timeline, root / "activity_timeline.png")
    return timeline


def _sync_bug_state(state: dict) -> None:
    """Refresh mirror fields in state with authoritative data from the background defect detection thread."""
    if not bool(state.get("bug_detection_enabled", True)):
        return
    detected_bugs, bug_trace_records = bug_detection_worker.snapshot()
    state["detected_bugs"] = detected_bugs
    state["bug_trace_records"] = bug_trace_records


def _flush_bug_detection(reason: str) -> None:
    """Wait for background defect detection queue to drain, continue on timeout (avoids abnormal hangs blocking exit)."""
    pending = bug_detection_worker.pending_count()
    if pending <= 0:
        return
    print(f"[Defect Detection Thread] Waiting for {pending} pending asynchronous detection tasks before {reason}...")
    completed = bug_detection_worker.flush(timeout=BUG_DETECTION_FLUSH_TIMEOUT_SECONDS or None)
    if not completed:
        print(
            "[Defect Detection Thread] Wait timed out, detection tasks are still running in the background; "
            "artifacts from this run may lack bug findings from these tasks; check bugs.json later."
        )


def _flush_page_audit(final_state: dict) -> None:
    """Wait for page audit queue to drain, and merge remaining audit results into graph (called before disk save).

    Background audit results are normally merged at the next round ffg_update node entry; at the end of run there is no
    "next round", so they must be merged directly here, otherwise the last batch of results will be lost.
    """
    graph = final_state.get("ffg")
    if isinstance(graph, FlowGraph):
        try:
            from workflow.nodes.ffg_update import (
                _merge_page_audit_results,
                _submit_page_audit_batch,
            )

            # ffg_update submits audits before applying current trace, so newly added Pages or those
            # with newly added screenshots are not yet enqueued. Resubmit un-audited fingerprints at the end; worker's
            # fingerprints and saturation state will prevent redundant idle processing.
            # A failed worker call deliberately leaves its fingerprint
            # unaudited so a later workflow iteration may retry it. During
            # final output there is no later iteration, so remember what this
            # flush already submitted; otherwise one persistent 502 resubmits
            # the same page forever and prevents artifact finalization.
            final_submitted_fingerprints: set[tuple[str, str]] = set()
            while True:
                pending = page_audit_worker.pending_count()
                if pending > 0:
                    print(
                        f"[Page Audit Thread] Waiting for {pending} "
                        "pending audit tasks before saving final outputs..."
                    )
                    page_audit_worker.flush(
                        timeout=BUG_DETECTION_FLUSH_TIMEOUT_SECONDS or None
                    )
                _merge_page_audit_results(graph)
                submitted = _submit_page_audit_batch(
                    graph,
                    exclude_fingerprints=final_submitted_fingerprints,
                )
                if submitted <= 0:
                    break
                print(
                    f"[Page Audit Thread] Resubmitted {submitted} un-audited page fingerprints at the end."
                )
        except Exception as exc:  # noqa: BLE001 - Fallback before disk save, failure does not block exit
            print(f"[Page Audit Thread] Failed to merge audit results at the end: {exc}")


def _print_summary(state: dict, output_dir: str, elapsed_seconds: float):
    """Print execution summary at the end of run."""
    _sync_bug_state(state)
    elapsed_min = elapsed_seconds / 60
    ffg = state.get("ffg")

    print(f"\n{'='*60}")
    print("  FuncDroid Execution Summary")
    print(f"{'='*60}")
    print(f"  Total time: {elapsed_min:.1f} minutes")
    exploration_mode = str(state.get("exploration_mode") or DEFAULT_EXPLORATION_MODE)
    print(f"  Exploration scope: {EXPLORATION_MODE_LABELS.get(exploration_mode, exploration_mode)}")

    if isinstance(ffg, FlowGraph):
        print(f"  FFG Function nodes: {len(ffg.functions)}")
        print(f"  FFG Pages: {len(ffg.pages)}")
        print(f"  Page transitions: {len(ffg.page_transitions)}")
        print(f"  Function dependencies: {len(ffg.function_dependencies)}")
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
    print(f"  Discovered Android Activities: {len(activities)}")

    bugs = state.get("detected_bugs", [])
    if not isinstance(bugs, list):
        bugs = []
    if bugs:
        crash_bugs = [b for b in bugs if isinstance(b, dict) and b.get("type") == "crash"]
        non_crash_bugs = [b for b in bugs if isinstance(b, dict) and b.get("type") != "crash"]
        print(f"  Detected bugs: {len(bugs)}")
        print(f"    - Crash: {len(crash_bugs)}")
        print(f"    - Non-crash: {len(non_crash_bugs)}")
    else:
        print("  Detected bugs: 0")

    print(f"  Execution iterations: {state.get('iteration_count', 0)}")
    print(f"  Raw trace screenshots: {trace_screenshot_count()} (trace_screenshots/)")
    usage_summary = get_llm_usage_summary()
    usage_totals = usage_summary.get("totals", {}) if isinstance(usage_summary, dict) else {}
    if usage_totals:
        print(
            "  LLM token: "
            f"prompt={usage_totals.get('prompt_tokens', 0)}, "
            f"completion={usage_totals.get('completion_tokens', 0)}, "
            f"total={usage_totals.get('total_tokens', 0)}, "
            f"requests={usage_totals.get('requests', 0)}, "
            f"failures={usage_totals.get('failures', 0)}"
        )
    print(f"  Output directory: {output_dir}")
    print("  Activity timeline: activity_timeline.json / activity_timeline.png")
    print(f"{'='*60}\n")


def _save_final_outputs(state: dict, output_dir: str):
    """Save final outputs, without saving recoverable workflow state."""
    _save_outputs(state, output_dir, prefix="[Output]")


def _save_outputs(state: dict, output_dir: str, prefix: str):
    os.makedirs(output_dir, exist_ok=True)
    ffg = state.get("ffg")
    if isinstance(ffg, FlowGraph):
        try:
            ffg.save(output_dir)
        except Exception as exc:
            print(f"{prefix} Failed to save FFG: {exc}")
    _save_bugs(state, output_dir, prefix=prefix)


def _save_bugs(state: dict, output_dir: str, prefix: str):
    _sync_bug_state(state)
    with _BUGS_SAVE_LOCK:
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
            print(f"{prefix} Failed to save bugs.json: {exc}")
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
        print(f"{prefix} Filtered non-high confidence bugs: {skipped_count}")
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
        "external_app_used",
        "external_context",
        "external_package",
        "caused_external_transition",
        "caused_target_crash",
        "final_recovery_method",
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
            "is_interruption": bool(step.get("is_interruption", False)),
            "action_failed": bool(step.get("action_failed", False)),
            "external_context": bool(step.get("external_context", False)),
            "foreground_recovery": bool(step.get("foreground_recovery", False)),
            "action_error": step.get("action_error", ""),
            "page": _serialize_page(page, screenshot_ref),
        })
    return serialized


def _fallback_trace_for_bug(state: dict, bug: dict[str, Any]) -> list[dict[str, Any]]:
    scenario_trace = state.get("scenario_execution_trace", [])
    if isinstance(scenario_trace, list) and scenario_trace:
        return scenario_trace
    return []


def _bug_artifact_fingerprint(bug: dict[str, Any], trace: list[dict[str, Any]]) -> str:
    """Fingerprint exactly the bug metadata, trace metadata and screenshot content."""
    digest = hashlib.sha256()
    digest.update(json.dumps(
        _json_safe(bug),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8"))
    for index, step in enumerate(trace, start=1):
        if not isinstance(step, dict):
            continue
        page = step.get("current_page")
        screenshot = _page_screenshot(page)
        screenshot_ref = f"screenshots/step_{index:03d}.jpg" if screenshot else ""
        step_without_page = {
            key: value for key, value in step.items() if key != "current_page"
        }
        digest.update(json.dumps(
            _json_safe(step_without_page),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"))
        digest.update(json.dumps(
            _serialize_page(page, screenshot_ref),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"))
        digest.update(screenshot.encode("ascii", errors="ignore"))
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
        existing_dirs: dict[str, str] = {}
        for name in os.listdir(bugs_dir):
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
            bug_id = str(existing_bug.get("bug_id", "")).strip()
            if bug_id and bug_id not in existing_dirs:
                existing_dirs[bug_id] = bug_dir

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
            if not isinstance(trace, list):
                trace = _fallback_trace_for_bug(state, bug)
            fingerprint = _bug_artifact_fingerprint(bug, trace)
            if _bug_artifacts_unchanged(bug_dir, fingerprint, trace):
                unchanged_count += 1
                continue

            bug_path = os.path.join(bug_dir, "bug.json")
            _atomic_write_json(bug_path, bug)
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
            f"{prefix} Bug details directory saved incrementally: {bugs_dir} "
            f"(updated={updated_count}, unchanged={unchanged_count})"
        )
    except Exception as exc:
        print(f"{prefix} Failed to save bug details directory: {exc}")


def _load_json_artifact(path: Path, errors: list[str]) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        errors.append(f"Cannot read {path.name}: {exc}")
        return None


def _save_run_result(
    state: dict,
    output_dir: str,
    *,
    exit_code: int | None,
    elapsed_seconds: float,
) -> None:
    _sync_bug_state(state)
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
        "bug_detection_enabled": bool(state.get("bug_detection_enabled", True)),
        "workflow_complete": bool(state.get("workflow_complete", False)),
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
        "llm_usage": get_llm_usage_summary().get("totals", {}),
        "fatal_error": _json_safe(state.get("fatal_error")),
    }
    _atomic_write_json(Path(output_dir) / "run_result.json", result)


def _validate_final_artifacts(state: dict, output_dir: str, exit_code: int) -> dict[str, Any]:
    """Validate the structure of the final directory, JSON readability, and all referenced files."""
    root = Path(output_dir)
    errors: list[str] = []
    warnings: list[str] = []
    required_files = [
        "activity_timeline.json",
        "activity_timeline.png",
        "bugs.json",
        "llm_usage.json",
        "run.log",
        "run_result.json",
    ]
    if isinstance(state.get("ffg"), FlowGraph):
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
            errors.append("activities in activity_timeline.json is not a list of strings")
        if not isinstance(points, list) or not points:
            errors.append("activity_timeline.json is missing time sample points")
        elif isinstance(activities, list):
            last = points[-1]
            if not isinstance(last, dict) or int(last.get("activity_count", -1)) != len(activities):
                errors.append("The last sample point in activity_timeline.json is inconsistent with total Activity count")
    elif activity_data is not None:
        errors.append("activity_timeline.json is not an object")
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
                    errors.append(f"evidence_screenshots of page {page_id} is not a list")
                    screenshots = []
                if not isinstance(controls, list) or any(not isinstance(value, str) for value in controls):
                    errors.append(f"controls of page {page_id} is not a list of natural language strings")
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
                    errors.append(f"page_ids of function {function_id} is not a list")
                    continue
                for page_id in page_refs:
                    if str(page_id) not in page_ids:
                        errors.append(f"Function {function_id} references non-existent page {page_id}")
                coverage_percent = function.get("coverage_percent", 0)
                if (
                    not isinstance(coverage_percent, int)
                    or isinstance(coverage_percent, bool)
                    or not 0 <= coverage_percent <= 100
                ):
                    errors.append(
                        f"coverage_percent of function {function_id} must be an integer between 0-100"
                    )

    bugs_path = root / "bugs.json"
    bugs_data = _load_json_artifact(bugs_path, errors) if bugs_path.is_file() else None
    if bugs_data is not None and not isinstance(bugs_data, list):
        errors.append("bugs.json is not an array")
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
            errors.append("bugs.json contains non-object entries")
            continue
        bug_id = str(bug.get("bug_id", "") or "")
        detail_dir = bug_detail_dirs.get(bug_id)
        if detail_dir is None:
            errors.append(f"Bug {bug_id or '<No ID>'} is missing details directory")
            continue
        trace_path = detail_dir / "trace.json"
        trace_data = _load_json_artifact(trace_path, errors) if trace_path.is_file() else None
        if trace_data is None:
            errors.append(f"Bug {bug_id} is missing trace.json")
            continue
        if not isinstance(trace_data, list):
            errors.append(f"trace.json of bug {bug_id} is not an array")
            continue
        if not trace_data:
            warnings.append(f"trace of bug {bug_id} is empty")
        for step in trace_data:
            if not isinstance(step, dict):
                continue
            screenshot_ref = str((step.get("page") or {}).get("screenshot", "") or "")
            if screenshot_ref:
                screenshot_path = detail_dir / screenshot_ref
                if not screenshot_path.is_file() or screenshot_path.stat().st_size <= 0:
                    errors.append(f"Bug {bug_id} is missing trace screenshot: {screenshot_ref}")

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
                            f"Raw trace screenshot index line {line_number} is missing file: "
                            f"{screenshot_ref or '<empty>'}"
                        )
        except OSError as exc:
            errors.append(f"Cannot read raw trace screenshot index: {exc}")

    if exit_code != 0:
        warnings.append("This run exited abnormally; artifacts represent the last consistent snapshot before the exception")

    files: list[dict[str, Any]] = []
    for path in sorted(
        item
        for item in root.rglob("*")
        if item.is_file()
        and not item.name.endswith(".tmp")
        and item.name != "artifact_manifest.json"
    ):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = ""
        if size <= 64 * 1024 * 1024:
            hasher = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
        files.append({"path": relative, "size_bytes": size, "sha256": digest or None})

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


def _save_artifact_manifest(state: dict, output_dir: str, exit_code: int) -> dict[str, Any]:
    manifest = _validate_final_artifacts(state, output_dir, exit_code)
    _atomic_write_json(Path(output_dir) / "artifact_manifest.json", manifest)
    return manifest


def _should_save_run_snapshot(node_name: str, state: dict) -> bool:
    if node_name == "ffg_update":
        return True
    if node_name != "scenario_plan":
        return False
    scenario_type = (state.get("current_plan", {}) or {}).get("scenario_type")
    return scenario_type in {"bootstrap", "status_update", "ffg_audit_update"}


def _save_run_snapshot(state: dict, output_dir: str):
    """Save intermediate run artifacts: FFG, page screenshots, and bugs, without saving checkpoint resume state."""
    _save_outputs(state, output_dir, prefix="[Run Snapshot]")


def _resolve_foreground_package(device) -> str:
    print("[Startup] Getting current foreground package name...")
    started_at = time.time()
    try:
        package_name = str(device.current_package() or "").strip()
        elapsed = time.time() - started_at
        if package_name:
            print(f"[Startup] Current foreground package name: {package_name} ({elapsed:.2f}s)")
        else:
            print(f"[Startup] Failed to get current foreground package name ({elapsed:.2f}s)")
        return package_name
    except Exception as exc:
        print(f"[Startup] Failed to get current foreground package name: {exc}")
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
        # ample room for bootstrap/status-only iterations and final routing.
        return max(1000, max_iterations * 8 + 100)
    # LangGraph requires a recursion guard even when FuncDroid is time-bound.
    # Keep it practically unreachable and expose an env override for operators.
    return 1_000_000


def _safe_run_dir_name(app_name: str) -> str:
    name = str(app_name or "").strip() or "Current foreground app"
    for separator in {os.sep, os.altsep}:
        if separator:
            name = name.replace(separator, "_")
    return name or "Current foreground app"


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
        print(f"[Pre-startup Cleanup] {label} failed, continuing execution: {exc}")
        return "", False


def _extract_recent_task_ids(dumpsys_output: str) -> list[str]:
    task_ids: list[str] = []
    seen: set[str] = set()
    for line in dumpsys_output.splitlines():
        if "Recent #" not in line and "RecentTaskInfo" not in line and "TaskRecord" not in line:
            continue
        for pattern in RECENT_TASK_ID_PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            task_id = match.group(1)
            if task_id not in seen:
                seen.add(task_id)
                task_ids.append(task_id)
            break
    return task_ids


def _clear_android_background_tasks(device, package_name: str) -> None:
    if getattr(device, "operating_system", "") not in {"adb", "android"}:
        return

    print("[Pre-startup Cleanup] Clearing background tasks and preparing clean launch of target app.")
    _run_android_shell_best_effort(device, "input keyevent KEYCODE_HOME", "Return to home screen")
    dumpsys_output, _ = _run_android_shell_best_effort(
        device,
        "dumpsys activity recents",
        "Read recent tasks",
    )
    task_ids = _extract_recent_task_ids(dumpsys_output)
    removed = 0
    activity_help, _ = _run_android_shell_best_effort(
        device,
        "cmd activity help",
        "Read activity command capability",
    )
    if task_ids and "remove-task" not in activity_help:
        print("[Pre-startup Cleanup] Current Android version does not support remove-task, skipping itemized removal of recent tasks.")
    else:
        for task_id in task_ids[:50]:
            _, succeeded = _run_android_shell_best_effort(
                device,
                f"cmd activity remove-task {task_id}",
                f"Remove recent task {task_id}",
            )
            if succeeded:
                removed += 1
    _run_android_shell_best_effort(device, "am kill-all", "Clean up background processes")
    if package_name:
        _run_android_shell_best_effort(device, f"am force-stop {package_name}", f"Stop target app {package_name}")
    print(
        "[Pre-startup Cleanup] Background cleanup finished: "
        f"removed_recent_tasks={removed}/{min(len(task_ids), 50)}"
    )


def _start_app_after_background_cleanup(
    device,
    package_name: str,
) -> None:
    _clear_android_background_tasks(device, package_name)
    print(f"[Startup] Launching target app: {package_name}")
    device.start_app(package_name)
    time.sleep(float(os.getenv("APP_RESTART_DELAY", "2.0")))


def _llm_timeout_value() -> str:
    return os.getenv("LLM_REQUEST_TIMEOUT", "").strip() or "none"


def _ensure_device_id(args) -> None:
    if args.device_id:
        return

    detected: list[tuple[str, str]] = []
    errors: list[str] = []
    for backend, connector in (("adb", ADB), ("hdc", HDC)):
        try:
            detected.extend((backend, serial) for serial in connector.devices())
        except Exception as exc:
            errors.append(f"{backend}: {exc}")

    unique_serials = sorted({serial for _backend, serial in detected if serial})
    if len(unique_serials) == 1:
        args.device_id = unique_serials[0]
        backend = next(name for name, serial in detected if serial == args.device_id)
        print(f"[Startup] Automatically selected the only {backend} device: {args.device_id}")
        return
    if not unique_serials:
        detail = f" (detection errors: {'; '.join(errors)})" if errors else ""
        raise ValueError(
            "No available devices detected. Please enable USB debugging and verify devices via adb devices or hdc list targets"
            + detail
        )
    raise ValueError(
        "Multiple devices detected, please specify one via --device-id: "
        + ", ".join(unique_serials)
    )


def _init_runtime(args):
    print("[Startup] Initializing planning/update LLM...")
    planning_llm = create_llm()
    print("[Startup] Initializing scenario execution LLM...")
    execution_llm = create_llm()
    # FFG update retains all unmatched trace graphs; local multimodal models processing full evidence are usually
    # slower than planning, so use an independent long timeout and limit retries by total runtime budget.
    ffg_update_timeout = float(os.getenv("FFG_UPDATE_LLM_TIMEOUT", "300") or 300)
    print(f"[Startup] Initializing FFG update LLM (timeout {ffg_update_timeout:g}s)...")
    ffg_update_llm = create_llm(timeout_seconds=ffg_update_timeout)
    if bool(getattr(args, "bug_detection", True)):
        print("[Startup] Initializing bug detection LLM...")
        bug_detection_llm = create_llm()
    else:
        print("[Startup] Bug detection is disabled, skipping dedicated detection LLM initialization")
        bug_detection_llm = None
    print(f"[Startup] Initializing device driver: {args.device_id}")
    device = Driver(args.device_id)
    print("[Startup] Device driver initialization completed")
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
            print(f"[logcat] Stopping full logcat capture: {resources.logcat_capture.path}")
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
                "--package-name not specified and cannot identify current foreground App; "
                "please open the target App on the device first"
            )
        print(f"[Startup] Using current foreground App: {args.package_name}")
    if not args.app_name:
        args.app_name = args.package_name
    return explicit_package


def _prepare_workflow_state(args, device, resources: RunResources, start_time: float) -> AgentState:
    if bool(getattr(args, "resume_existing", False)):
        return _load_resume_state(args, resources.output_dir, start_time)
    return _build_initial_state(args, start_time)


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
    llm_model = os.getenv("MODEL", "").strip()
    llm_timeout = _llm_timeout_value()
    ffg_update_timeout = os.getenv("FFG_UPDATE_LLM_TIMEOUT", "300").strip() or "300"

    print(f"\n{'='*60}")
    print("  FuncDroid Workflow Started")
    print(f"{'='*60}")
    print(f"  App: {state.get('app_name')} ({state.get('package_name')})")
    print(f"  App selection: {'Specified package name' if args.package_name_explicit else 'Current foreground App'}")
    print(f"  Device: {args.device_id}")
    print(f"  Planning mode: {PHASE_LABELS[args.phase]}")
    exploration_mode = str(state.get("exploration_mode") or DEFAULT_EXPLORATION_MODE)
    print(
        f"  Exploration scope: {EXPLORATION_MODE_LABELS.get(exploration_mode, exploration_mode)}"
        f" ({exploration_mode})"
    )
    boundary_packages = enabled_packages()
    if exploration_mode == "core":
        print(
            "  Exploration boundary: "
            + (
                "Enabled: " + ", ".join(boundary_packages)
                if boundary_packages
                else "Disabled (no active rules in boundary_rules.json)"
            )
        )
    print(f"  Total time: {time_label}")
    iteration_label = str(args.max_iterations) if args.max_iterations > 0 else "unlimited"
    print(f"  Max iterations: {iteration_label}")
    print(f"  LangGraph recursion budget: {_graph_recursion_limit(state)}")
    print(f"  Planning/update model: {llm_model}")
    print(f"  Scenario execution model: {llm_model}")
    bug_detection_enabled = bool(state.get("bug_detection_enabled", True))
    print(
        "  Bug detection: "
        + (f"Enabled ({llm_model})" if bug_detection_enabled else "Disabled")
    )
    print(
        "  LLM timeout: "
        f"regular={llm_timeout}s, "
        f"ffg_update={ffg_update_timeout}s"
    )
    print(f"  Output directory: {resources.output_dir}")
    print(f"  Resume run: {'Yes' if bool(getattr(args, 'resume_existing', False)) else 'No'}")
    print(f"  Run log: {resources.run_log.path}")
    if resources.logcat_capture is not None:
        print(f"  Full logcat: {resources.logcat_capture.path}")
    print(
        "  Exploration params: "
        f"max_scenarios_per_function={_env_value('PLAN_MAX_SCENARIOS_PER_FUNCTION', '6')}, "
        f"function_coverage_target={_env_value('PLAN_FUNCTION_COVERAGE_TARGET', '80')}%, "
        f"max_audit_images_per_function={_env_value('PLAN_MAX_AUDIT_IMAGES_PER_FUNCTION', '4')}"
    )
    print(
        "  Timing params: "
        f"action_delay={_env_value('ACTION_DELAY', '0.7')}s, "
        f"fast_action_delay={_env_value('FAST_ACTION_DELAY', '0.15')}s, "
        f"app_restart_delay={_env_value('APP_RESTART_DELAY', '2.0')}s, "
        f"max_exec_steps={_effective_max_exec_steps()}, "
        f"max_navigation_steps={_effective_max_navigation_steps()}"
    )
    print(
        "  Detection params: "
        "ffg_update=incremental graph building from execution trace, "
        f"trace_defect_detection={'background async thread (full trace + all screenshots)' if bug_detection_enabled else 'off'}, "
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
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _print_graph_step_status(node_name: str, final_state: dict) -> None:
    current_plan = final_state.get("current_plan", {}) or {}
    if node_name == "scenario_plan":
        scenario_type = str(current_plan.get("scenario_type", "") or "none")
        if scenario_type == "cross_test":
            scope = (
                f"{current_plan.get('src_function_id', '') or 'unknown'}->"
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
        function_count = len(ffg.functions) if isinstance(ffg, FlowGraph) else 0
        page_count = len(ffg.pages) if isinstance(ffg, FlowGraph) else 0
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
                f"[Run Snapshot] Failed to save {label}, main test loop continues, subsequent nodes will retry: "
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
        "FFG/Bug snapshot",
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
        "Run state",
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
        _save_progress_best_effort("LLM usage", save_llm_usage_summary)
        if _should_save_run_snapshot(node_name, final_state):
            _save_progress_best_effort(
                "FFG/Bug snapshot",
                lambda: _save_run_snapshot(final_state, resources.output_dir),
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
        _save_progress_best_effort(
            "Run state",
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


def _save_and_summarize(final_state: dict, output_dir: str, elapsed: float) -> None:
    _flush_bug_detection("save final outputs")
    _flush_page_audit(final_state)
    save_llm_usage_summary()
    _save_final_outputs(final_state, output_dir)
    final_state["_activity_timeline"] = _save_activity_timeline(
        final_state,
        output_dir,
        final_state.get("_activity_timeline"),
        force=True,
    )
    _print_summary(final_state, output_dir, elapsed)


def _cleanup_run_resources(resources: RunResources) -> None:
    _close_logcat_capture(resources)
    _close_run_log(resources)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FuncDroid Pure Defect Detection Workflow")
    parser.add_argument("--device-id", type=_normalize_device_id, default=None,
                        help="Device ID; auto-selected when only one device is connected, must be specified for multiple devices")
    parser.add_argument("--app-name", default=None,
                        help="App display name; defaults to target package name")
    parser.add_argument("--package-name", default=None,
                        help="Package name of installed app on device; tests current foreground App if omitted")
    parser.add_argument("--max-iterations", type=_nonnegative_int,
                        default=_nonnegative_int(os.getenv("MAX_ITERATIONS", str(DEFAULT_MAX_ITERATIONS))),
                        help="Maximum execution iterations (default 0, unlimited)")
    parser.add_argument("--total-time", type=_nonnegative_float,
                        default=_nonnegative_float(os.getenv("TOTAL_TIME", str(DEFAULT_TOTAL_TIME_MINUTES))),
                        help=f"Total run time limit (minutes, default {DEFAULT_TOTAL_TIME_MINUTES}, 0 for unlimited)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for new run; must be empty, defaults to creating a separate directory under output/ by app name")
    parser.add_argument(
        "--resume-output",
        default=None,
        help="Resume cumulative run from an existing output directory with the same FFG, Activity timeline, bugs, and execution history",
    )
    parser.add_argument("--phase", choices=["all", "intra"], default="all",
                        help="Planning mode: all=automatic selection, intra=intra-function only")
    parser.add_argument(
        "--exploration-mode",
        choices=["core"],
        default=os.getenv("EXPLORATION_MODE", DEFAULT_EXPLORATION_MODE).strip() or DEFAULT_EXPLORATION_MODE,
        help="Exploration mode: core=explore target app core functions only, "
        "blocking third-party content like mini-programs if boundary_rules.json is configured (default core)",
    )
    parser.add_argument(
        "--bug-detection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to enable asynchronous bug detection; enabled by default, use --no-bug-detection to disable",
    )
    return parser


def _run_single(args) -> int:
    _ensure_device_id(args)
    if args.output_dir and args.resume_output:
        raise ValueError("--output-dir and --resume-output cannot be used together")
    resources = RunResources(output_dir=args.resume_output or args.output_dir or "")
    requested_root = Path(resources.output_dir) if resources.output_dir else None
    existing_ffg = bool(requested_root and (requested_root / "ffg.json").is_file())
    if args.resume_output and not existing_ffg:
        raise FileNotFoundError(f"Directory specified by --resume-output is missing ffg.json: {requested_root}")
    if (
        args.output_dir
        and requested_root
        and requested_root.exists()
        and any(requested_root.iterdir())
    ):
        raise FileExistsError(
            f"New runs are not allowed to overwrite existing non-empty directory: {requested_root}; "
            "please use a new directory, or use --resume-output only for disaster recovery"
        )
    args.resume_existing = bool(args.resume_output)
    if args.resume_existing:
        print(f"[Resume] Explicitly resuming the same FFG: {requested_root}")
    final_state: dict = {}
    start_time = time.time()
    device = None
    exit_code = 0
    outputs_saved = False
    elapsed = 0.0
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _raise_run_termination)
    try:
        planning_llm, execution_llm, ffg_update_llm, bug_detection_llm, device = _init_runtime(args)
        args.package_name_explicit = _resolve_target_app(args, device)
        _prepare_run_logging(args, device, resources)
        if args.package_name_explicit:
            _start_app_after_background_cleanup(device, args.package_name)
        state = _prepare_workflow_state(args, device, resources, start_time)
        final_state.update(dict(state))
        if state.get("bug_detection_enabled", True):
            bug_detection_worker.configure(
                bug_detection_llm,
                on_batch_done=lambda: _save_bugs(final_state, resources.output_dir, prefix="[Defect Detection Thread]"),
                initial_bugs=state.get("detected_bugs"),
                initial_records=state.get("bug_trace_records"),
            )
        # Background audit of page controls uses the same planning LLM as FFG update; queue and saturation
        # state can be seen in workflow/page_audit_worker.py.
        page_audit_worker.configure(planning_llm)
        graph = build_graph(
            planning_llm,
            execution_llm,
            device,
            ffg_update_llm=ffg_update_llm,
        )
        _print_startup_banner(args, state, resources)
        _run_graph(graph, state, resources, final_state)
        elapsed = time.time() - float(final_state.get("start_time", start_time) or start_time)
        _save_and_summarize(final_state, resources.output_dir, elapsed)
        outputs_saved = True
        print(f"[Done] Workflow execution finished, output saved in: {resources.output_dir}")
    except (KeyboardInterrupt, RunTerminationSignal) as exc:
        exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 128 + exc.signum
        elapsed = time.time() - float(final_state.get("start_time", start_time) or start_time)
        final_state["fatal_error"] = {
            "type": type(exc).__name__,
            "message": "User interrupted" if isinstance(exc, KeyboardInterrupt) else str(exc),
            "traceback": "",
        }
        print(f"[Run Interrupted] Received stop request, saving current FFG, bugs, and usage data (exit={exit_code}).")
    except Exception as exc:
        exit_code = 1
        elapsed = time.time() - float(final_state.get("start_time", start_time) or start_time)
        final_state["fatal_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(f"[Run Exception] Workflow encountered an uncaught exception and exited safely: {exc}")
        print(traceback.format_exc())
        if final_state:
            _save_and_summarize(final_state, resources.output_dir, elapsed)
            outputs_saved = True
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
        _flush_bug_detection("generate final run results")
        elapsed = elapsed or (
            time.time() - float(final_state.get("start_time", start_time) or start_time)
        )
        if resources.output_dir and final_state and not outputs_saved:
            try:
                _save_and_summarize(final_state, resources.output_dir, elapsed)
            except Exception as exc:
                exit_code = 1
                print(f"[Output] Failed to save exception snapshot: {exc}")
        _cleanup_run_resources(resources)
        if resources.output_dir:
            try:
                save_llm_usage_summary()
                final_state.setdefault("start_time", start_time)
                artifact_root = Path(resources.output_dir)
                ffg_missing = (
                    isinstance(final_state.get("ffg"), FlowGraph)
                    and not (artifact_root / "ffg.json").is_file()
                )
                if ffg_missing or not (artifact_root / "bugs.json").is_file():
                    _save_outputs(final_state, resources.output_dir, prefix="[Output Recovery]")
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
                    )
                print(
                    "[Artifact Validation] "
                    f"status={manifest['status']}, files={len(manifest['files'])}, "
                    f"errors={len(manifest['errors'])}, warnings={len(manifest['warnings'])}"
                )
                for error in manifest["errors"]:
                    print(f"[Artifact Validation] Error: {error}")
            except Exception as exc:
                exit_code = 1
                print(f"[Artifact Validation] Failed to generate manifest: {exc}")
    return exit_code


def main():
    parser = _build_parser()
    args = parser.parse_args()

    _ensure_device_id(args)
    sys.exit(_run_single(args))


if __name__ == "__main__":
    main()
