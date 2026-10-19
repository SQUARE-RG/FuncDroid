"""FuncDroid runtime visualization server using only the Python standard library.

    python web/server.py --port 8787

Monitor experiments/output/ under the project root, with one directory per run.
Override the root with FUNCDROID_OUTPUT_DIR or --output-dir."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

WEB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_DIR.parent
STATIC_DIR = WEB_DIR / "static"

# Unicode escapes retain historical log parsing without modifying saved artifacts.
_ROUND_HEADER_RE = re.compile(r"(?:\u7b2c\s*|Iteration\s+)(\d+)\s*(?:\u8f6e\s*)?[—-]\s*(?:\u6a21\u5f0f|scope):\s*(.*?)\s*\|\s*(?:\u63a2\u7d22\u8303\u56f4|exploration):\s*(.*)$")
_METRIC_LINE_RE = re.compile(r"^\[(?:\u8f6e\u6b21\u6307\u6807|Iteration metrics)\]\s*(.*)$")
_TAGGED_LINE_RE = re.compile(r"^\[([^\]]+)\]\s*(.*)$")
_STEP_RE = re.compile(
    r"^\[(?:\u6b65\u9aa4|Step)\s+(\d+)\](?:\s+\[(?:Interruption|\u4e2d\u65ad)\])?\s+(?:\u9875\u9762|page)=(.*?)\s+(?:\u52a8\u4f5c|action)=([^|]+?)(?:\s*\|\s*(?:\u601d\u8003|Assessment):\s*(.*))?$"
)
_EXEC_INPUT_RE = re.compile(
    r"^\[(?:\u573a\u666f\u6267\u884c|Scenario execution)\]\s+LLM (?:\u8f93\u5165\u6458\u8981|input summary):\s*phase=([^,]+),\s*step=(\d+).*?"
    r"current_page=([^,]+),\s*activity=([^,]+)"
)
_EXEC_START_PAGE_RE = re.compile(r"^\[(?:\u573a\u666f\u6267\u884c|Scenario execution)\]\s+(?:\u8d77\u59cb\u9875\u9762|Initial page|Starting page):\s*(.*)$")
_FFG_DELTA_RE = re.compile(
    r"^\[(?:FFG \u66f4\u65b0|FFG update)\]\s+(?:\u589e\u91cf|Increment):\s*(?:\u529f\u80fd|functionalities)=(\d+),\s*(?:\u9875\u9762|pages)=(\d+),\s*(?:\u64cd\u4f5c|operations)=(\d+),\s*"
    r"(?:\u9875\u9762\u8df3\u8f6c|\u9875\u5185\u8df3\u8f6c|page_transitions)=(\d+),\s*(?:\u529f\u80fd\u8df3\u8f6c=\d+,\s*)?(?:\u529f\u80fd\u4f9d\u8d56|function_dependencies)=(\d+)"
)
_STATE_RESULT_PREFIX = "[\u72b6\u6001\u66f4\u65b0] \u6267\u884c\u7ed3\u679c:"
_STATE_EXEC_RE = re.compile(
    r"^\[State\] scenario_exec \| navigation=([^ (]+)\((\d+) steps\) \| "
    r"scenario=([^ (]+)\((\d+) steps\) \| page=(.*)$"
)


def _read_json(path: Path, default=None):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001 - Return defaults for damaged artifacts.
        return default


def _parse_metric_value(raw: str):
    raw = raw.strip()
    try:
        if re.fullmatch(r"-?\d+", raw):
            return int(raw)
        if re.fullmatch(r"-?\d+\.\d+", raw):
            return float(raw)
    except Exception:  # noqa: BLE001, S110 - A malformed metric must not prevent parsing the remaining metrics.
        pass
    return raw


def _parse_kv_line(payload: str) -> dict:
    result = {}
    for chunk in payload.split(", "):
        if "=" not in chunk:
            continue
        key, _, value = chunk.partition("=")
        result[key.strip()] = _parse_metric_value(value)
    return result


def _normalize_coverage_percent(value: object) -> int:
    try:
        return max(0, min(100, int(value or 0)))
    except (TypeError, ValueError):
        return 0


def _normalize_ffg(data: dict | None) -> dict:
    """Convert legacy function_nodes/page_registry artifacts to the current Web structure."""
    data = data if isinstance(data, dict) else {}
    if "functions" in data or "pages" in data:
        normalized_functions = []
        for raw in data.get("functions", []) if isinstance(data.get("functions"), list) else []:
            if not isinstance(raw, dict):
                continue
            page_ids = [str(value).strip() for value in raw.get("page_ids", []) if str(value).strip()]
            legacy_entries = raw.get("entry_page_ids", [])
            if isinstance(legacy_entries, list):
                for value in legacy_entries:
                    page_id = str(value).strip()
                    if page_id and page_id not in page_ids:
                        page_ids.append(page_id)
            coverage_percent = _normalize_coverage_percent(raw.get("coverage_percent", 0))
            normalized_functions.append({
                **{
                    key: value
                    for key, value in raw.items()
                    if key not in {"entry_page_ids", "entry_page_id", "test_depth"}
                },
                "page_ids": page_ids,
                "coverage_percent": coverage_percent,
            })
        return {
            "functions": normalized_functions,
            "pages": data.get("pages") if isinstance(data.get("pages"), list) else [],
            "page_transitions": data.get("page_transitions") if isinstance(data.get("page_transitions"), list) else [],
            "function_dependencies": data.get("function_dependencies") if isinstance(data.get("function_dependencies"), list) else [],
        }

    functions = []
    transition_candidates = []
    for raw in data.get("function_nodes", []) if isinstance(data.get("function_nodes"), list) else []:
        if not isinstance(raw, dict):
            continue
        legacy_entry_ids = raw.get("entry_page_ids")
        if not isinstance(legacy_entry_ids, list):
            entry = str(raw.get("entry_page_id") or "").strip()
            legacy_entry_ids = [entry] if entry else []
        page_ids = [str(value).strip() for value in raw.get("page_ids", []) if str(value).strip()]
        for value in legacy_entry_ids:
            page_id = str(value).strip()
            if page_id and page_id not in page_ids:
                page_ids.append(page_id)
        scenarios = []
        for key in ("scenarios", "test_scenarios", "attempted_scenarios"):
            values = raw.get(key, [])
            if not isinstance(values, list):
                continue
            for value in values:
                if isinstance(value, dict):
                    value = value.get("description") or value.get("scenario") or ""
                if str(value).strip():
                    scenarios.append(str(value).strip())
        functions.append(
            {
                "id": str(raw.get("id") or "").strip(),
                "name": str(raw.get("name") or raw.get("description") or raw.get("id") or "").strip(),
                "description": str(raw.get("description") or "").strip(),
                "page_ids": page_ids,
                "resource_variables": [str(value).strip() for value in raw.get("resource_variables", []) if str(value).strip()],
                "scenarios": list(dict.fromkeys(scenarios)),
                "test_status": str(raw.get("test_status") or "not_started").strip(),
                "coverage_percent": _normalize_coverage_percent(raw.get("coverage_percent", 0)),
            }
        )
        transitions = raw.get("page_transitions", [])
        if isinstance(transitions, list):
            transition_candidates.extend(transitions)

    pages = []
    for raw in data.get("page_registry", []) if isinstance(data.get("page_registry"), list) else []:
        if not isinstance(raw, dict):
            continue
        screenshots = raw.get("screenshots") if isinstance(raw.get("screenshots"), list) else []
        screenshots = [str(value).strip() for value in screenshots if str(value).strip()]
        pages.append(
            {
                "id": str(raw.get("id") or "").strip(),
                "name": str(raw.get("name") or raw.get("id") or "").strip(),
                "description": str(raw.get("description") or "").strip(),
                "representative_screenshot": screenshots[0] if screenshots else "",
                "evidence_screenshots": screenshots[1:],
                "controls": [str(value).strip() for value in raw.get("controls", []) if str(value).strip()],
                "operations": [str(value).strip() for value in raw.get("operations", []) if str(value).strip()],
                "activity": str(raw.get("activity") or "").strip(),
            }
        )

    transitions = []
    seen_transitions = set()
    for raw in transition_candidates + (data.get("function_transitions", []) if isinstance(data.get("function_transitions"), list) else []):
        if not isinstance(raw, dict):
            continue
        item = {
            "src_page_id": str(raw.get("src_page_id") or raw.get("src") or "").strip(),
            "dst_page_id": str(raw.get("dst_page_id") or raw.get("dst") or "").strip(),
            "operation": str(raw.get("operation") or raw.get("action") or "").strip(),
        }
        key = tuple(item.values())
        if all(item.values()) and key not in seen_transitions:
            seen_transitions.add(key)
            transitions.append(item)

    dependencies = []
    for raw in data.get("function_dependencies", []) if isinstance(data.get("function_dependencies"), list) else []:
        if not isinstance(raw, dict):
            continue
        dependencies.append(
            {
                "src_function_id": str(raw.get("src_function_id") or raw.get("src") or "").strip(),
                "dst_function_id": str(raw.get("dst_function_id") or raw.get("dst") or "").strip(),
                "resource_variable": str(raw.get("resource_variable") or raw.get("description") or "").strip(),
            }
        )
    return {
        "functions": functions,
        "pages": pages,
        "page_transitions": transitions,
        "function_dependencies": dependencies,
    }


def _ffg_counts(ffg: dict, fallback: dict | None = None) -> dict:
    functions = ffg.get("functions", []) if isinstance(ffg, dict) else []
    pages = ffg.get("pages", []) if isinstance(ffg, dict) else []
    statuses = {"unavailable": 0, "blocked": 0, "not_started": 0, "completed": 0, "deferred": 0, "in_progress": 0}
    for function in functions:
        status = str(function.get("test_status") or "not_started").strip()
        if status in statuses:
            statuses[status] += 1
    raw = fallback if isinstance(fallback, dict) else {}
    return {
        "functions": len(functions),
        "pages": len(pages),
        "page_transitions": len(ffg.get("page_transitions", [])),
        "function_dependencies": len(ffg.get("function_dependencies", [])),
        "test_status": statuses if functions else raw.get("test_status", statuses),
    }


class OutputStore:
    """Scan output directories and read artifacts for individual runs."""

    def __init__(self, output_root: Path):
        self.output_root = output_root

    def list_runs(self) -> list[dict]:
        runs = []
        if not self.output_root.is_dir():
            return runs
        for entry in self.output_root.iterdir():
            if not entry.is_dir():
                continue
            run_result = _read_json(entry / "run_result.json", {}) or {}
            normalized_ffg = self.ffg(entry)
            bugs = _read_json(entry / "bugs.json", []) or []
            try:
                mtime = entry.stat().st_mtime
            except OSError:
                mtime = 0
            updated_at = run_result.get("updated_at") or datetime.fromtimestamp(mtime).astimezone().isoformat()
            runs.append(
                {
                    "name": entry.name,
                    "status": run_result.get("status", "unknown"),
                    "app_name": run_result.get("app_name", ""),
                    "package_name": run_result.get("package_name", ""),
                    "iteration_count": run_result.get("iteration_count", 0),
                    "elapsed_seconds": run_result.get("elapsed_seconds", 0),
                    "updated_at": updated_at,
                    "ffg_counts": _ffg_counts(normalized_ffg, run_result.get("ffg_counts", {})),
                    "high_confidence_bug_count": run_result.get("high_confidence_bug_count", 0),
                    "bug_count": len(bugs) if isinstance(bugs, list) else 0,
                    "has_ffg": (entry / "ffg.json").is_file(),
                    "has_compliance": (entry / "compliance_report.json").is_file(),
                    "mtime": mtime,
                }
            )
        runs.sort(key=lambda item: item.get("mtime", 0), reverse=True)
        return runs

    def _safe_run_dir(self, name: str) -> Path | None:
        if not name or "/" in name or "\\" in name or name in {".", ".."}:
            return None
        candidate = (self.output_root / name).resolve()
        try:
            candidate.relative_to(self.output_root.resolve())
        except ValueError:
            return None
        if not candidate.is_dir():
            return None
        return candidate

    def get_run_dir(self, name: str) -> Path | None:
        return self._safe_run_dir(name)

    def summary(self, run_dir: Path) -> dict:
        run_result = _read_json(run_dir / "run_result.json", {}) or {}
        normalized_ffg = self.ffg(run_dir)
        run_result["ffg_counts"] = _ffg_counts(normalized_ffg, run_result.get("ffg_counts", {}))
        manifest = _read_json(run_dir / "artifact_manifest.json", {}) or {}
        llm_usage = _read_json(run_dir / "llm_usage.json", {}) or {}
        bugs = _read_json(run_dir / "bugs.json", []) or []
        files = {}
        for fname in (
            "ffg.json",
            "bugs.json",
            "run_result.json",
            "run.log",
            "llm_usage.json",
            "activity_timeline.json",
            "activity_timeline.png",
            "artifact_manifest.json",
        ):
            fpath = run_dir / fname
            if fpath.is_file():
                stat = fpath.stat()
                files[fname] = {"size": stat.st_size, "mtime": stat.st_mtime}
        return {
            "run_result": run_result,
            "manifest": manifest,
            "llm_usage": llm_usage,
            "bug_count": len(bugs) if isinstance(bugs, list) else 0,
            "files": files,
        }

    def ffg(self, run_dir: Path) -> dict:
        return _normalize_ffg(_read_json(run_dir / "ffg.json", {}))

    def bugs(self, run_dir: Path) -> list:
        data = _read_json(run_dir / "bugs.json", []) or []
        return data if isinstance(data, list) else []

    def compliance(self, run_dir: Path) -> dict:
        """Read a compliance run report, or return an empty object when absent."""
        data = _read_json(run_dir / "compliance_report.json", {}) or {}
        return data if isinstance(data, dict) else {}

    def compliance_trace(self, run_dir: Path, rel_path: str) -> dict | None:
        """Read a referenced compliance trace restricted to the run directory."""
        if not rel_path or not rel_path.endswith(".json"):
            return None
        target = self.resolve_file(run_dir, rel_path)
        if target is None:
            return None
        data = _read_json(target, None)
        return data if isinstance(data, dict) else None

    def bug_detail(self, run_dir: Path, bug_id: str) -> dict | None:
        bugs_root = run_dir / "bugs"
        if not bugs_root.is_dir():
            return None
        target_dir = None
        for entry in bugs_root.iterdir():
            if not entry.is_dir():
                continue
            bug_json = _read_json(entry / "bug.json", {})
            if str((bug_json or {}).get("bug_id", "")) == bug_id or entry.name == bug_id:
                target_dir = entry
                break
        if target_dir is None:
            return None
        bug = _read_json(target_dir / "bug.json", {}) or {}
        trace = _read_json(target_dir / "trace.json", []) or []
        return {"bug": bug, "trace": trace, "dir_name": target_dir.name}

    def activity_timeline(self, run_dir: Path) -> dict:
        return _read_json(run_dir / "activity_timeline.json", {}) or {}

    def log_lines(self, run_dir: Path) -> list[str]:
        log_path = run_dir / "run.log"
        if not log_path.is_file():
            return []
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        return text.splitlines()

    def metrics(self, run_dir: Path) -> dict:
        lines = self.log_lines(run_dir)
        iterations = []
        rounds = []
        for line in lines:
            m = _METRIC_LINE_RE.match(line)
            if m:
                iterations.append(_parse_kv_line(m.group(1)))
                continue
            m2 = _ROUND_HEADER_RE.search(line)
            if m2:
                rounds.append(
                    {
                        "round": int(m2.group(1)),
                        "mode": m2.group(2).strip(),
                        "scope": m2.group(3).strip(),
                    }
                )
        return {"iterations": iterations, "rounds": rounds}

    def live_state(self, run_dir: Path) -> dict:
        """Reconstruct the current visualization state from the appended run.log.

        Parse stable tagged lines because there is no separate runtime event stream.
        Preserve ambiguous Activity names without inventing exact FFG page identities."""
        lines = self.log_lines(run_dir)
        graph = self.ffg(run_dir)
        page_by_id = {
            str(page.get("id")): page
            for page in graph.get("pages", [])
            if isinstance(page, dict) and page.get("id")
        }
        pages_by_activity: dict[str, list[dict]] = {}
        for page in page_by_id.values():
            activity = str(page.get("activity") or "").strip()
            if activity:
                pages_by_activity.setdefault(activity, []).append(page)

        rounds: list[dict] = []
        current: dict | None = None
        phase = "planning"
        pending_observation: dict[tuple[str, int], dict] = {}
        last_execution_round: dict | None = None

        def ensure_round() -> dict:
            nonlocal current
            if current is None:
                current = {
                    "round": 0,
                    "mode": "initializing",
                    "scope": "",
                    "stage": "initializing",
                    "scenario": "",
                    "scenario_type": "",
                    "trace": [],
                    "deltas": [],
                    "result": {},
                    "metric": {},
                }
                rounds.append(current)
            return current

        for line_no, line in enumerate(lines):
            header = _ROUND_HEADER_RE.search(line)
            if header:
                current = {
                    "round": int(header.group(1)),
                    "mode": header.group(2).strip(),
                    "scope": header.group(3).strip(),
                    "stage": "planning",
                    "scenario": "",
                    "scenario_type": "",
                    "trace": [],
                    "deltas": [],
                    "result": {},
                    "metric": {},
                }
                rounds.append(current)
                phase = "planning"
                pending_observation = {}
                continue

            item = ensure_round()
            if line.startswith(("[\u72b6\u6001\u66f4\u65b0] \u6267\u884c\u573a\u666f:", "[Plan] Scenario:")):
                item["scenario"] = line.partition(":")[2].strip()
            elif any(tag in line for tag in ("[\u573a\u666f\u6267\u884c\u5b50\u56fe]", "[Scenario execution subgraph]")):
                item["stage"] = "executing"
            elif line.startswith(("[\u573a\u666f\u6267\u884c] \u5df2\u8bbe\u7f6e\u573a\u666f logcat \u8d77\u70b9:", "[Scenario execution] Set scenario logcat start:")):
                phase = line.rsplit("=", 1)[-1].strip() or phase
                item["stage"] = "executing"
            else:
                start_match = _EXEC_START_PAGE_RE.match(line)
                if start_match:
                    page_id = start_match.group(1).strip()
                    if page_id and page_id not in {"\u672a\u77e5", "unknown", "None"}:
                        item["start_page_id"] = page_id

            input_match = _EXEC_INPUT_RE.match(line)
            if input_match:
                obs_phase = input_match.group(1).strip()
                obs_step = int(input_match.group(2))
                pending_observation[(obs_phase, obs_step)] = {
                    "activity": input_match.group(4).strip(),
                    "runtime_page": input_match.group(3).strip(),
                }
                phase = obs_phase
                item["stage"] = "executing"
                continue

            step_match = _STEP_RE.match(line)
            if step_match:
                step_no = int(step_match.group(1))
                page_id = step_match.group(2).strip()
                if page_id in {"\u672a\u77e5", "unknown", "None", ""}:
                    page_id = ""
                observation = pending_observation.get((phase, step_no), {})
                activity = str(observation.get("activity") or "").strip()
                candidates = pages_by_activity.get(activity, [])
                inferred = False
                if not page_id and len(candidates) == 1:
                    page_id = str(candidates[0].get("id") or "")
                    inferred = bool(page_id)
                item["trace"].append(
                    {
                        "sequence": len(item["trace"]) + 1,
                        "step": step_no,
                        "phase": phase,
                        "page_id": page_id,
                        "page_inferred": inferred,
                        "activity": activity,
                        "action": step_match.group(3).strip(),
                        "thought": (step_match.group(4) or "").strip(),
                        "line": line_no,
                    }
                )
                item["stage"] = "executing"
                last_execution_round = item
                continue

            delta_match = _FFG_DELTA_RE.match(line)
            if delta_match:
                item["deltas"].append(
                    {
                        "functions": int(delta_match.group(1)),
                        "pages": int(delta_match.group(2)),
                        "operations": int(delta_match.group(3)),
                        "transitions": int(delta_match.group(4)),
                        "dependencies": int(delta_match.group(5)),
                    }
                )
                item["stage"] = "updating_ffg"
                continue

            state_match = _STATE_EXEC_RE.match(line)
            if state_match:
                item["result"].update({
                    "navigation_status": state_match.group(1),
                    "navigation_trace_length": int(state_match.group(2)),
                    "scenario_status": state_match.group(3),
                    "scenario_trace_length": int(state_match.group(4)),
                })
                if state_match.group(5) in page_by_id:
                    item["metric"]["page"] = state_match.group(5)
                continue

            if line.startswith("[State] scenario_exec | error="):
                item["result"]["scenario_error"] = line.partition("error=")[2].strip()
                continue

            if line.startswith(_STATE_RESULT_PREFIX):
                try:
                    result = json.loads(line[len(_STATE_RESULT_PREFIX) :].strip())
                except (TypeError, ValueError, json.JSONDecodeError):
                    result = {}
                if isinstance(result, dict):
                    item["result"] = result
                    item["scenario"] = result.get("current_scenario") or item["scenario"]
                    item["scenario_type"] = result.get("scenario_type") or item["scenario_type"]
                continue

            metric_match = _METRIC_LINE_RE.match(line)
            if metric_match:
                item["metric"].update(_parse_kv_line(metric_match.group(1)))
                item["scenario_type"] = item["metric"].get("scenario_type") or item["scenario_type"]
                item["stage"] = "round_complete"

        latest = rounds[-1] if rounds else ensure_round()
        display_round = latest if latest.get("trace") else (last_execution_round or latest)
        latest_metric = latest.get("metric") or {}
        display_metric = display_round.get("metric") or {}
        display_trace = display_round.get("trace") or []
        last_trace = (display_trace or [{}])[-1]
        last_trace_page_id = str(last_trace.get("page_id") or "").strip()
        if last_trace_page_id not in page_by_id:
            last_trace_page_id = ""
        current_page_id = next(
            (
                candidate
                for value in (
                    latest_metric.get("page"),
                    display_metric.get("page"),
                    last_trace_page_id,
                    display_round.get("start_page_id") if not display_trace else "",
                )
                if (candidate := str(value or "").strip()) in page_by_id
            ),
            "",
        )
        current_page_match = "exact" if current_page_id else "none"
        current_activity = str(
            latest_metric.get("activity")
            or display_metric.get("activity")
            or (display_round.get("trace") or [{}])[-1].get("activity", "")
        ).strip()
        current_page = page_by_id.get(current_page_id)
        if current_page is None and current_activity:
            candidates = pages_by_activity.get(current_activity, [])
            if len(candidates) == 1:
                current_page = candidates[0]
                current_page_id = str(current_page.get("id") or "")
                current_page_match = "activity_unique"
            elif candidates:
                # Legacy artifacts lack the final page ID. Prefer the last discovered page for the Activity and label this as inferred.
                current_page = candidates[-1]
                current_page_id = str(current_page.get("id") or "")
                current_page_match = "activity_latest"

        delta_total = {key: 0 for key in ("functions", "pages", "operations", "transitions", "dependencies")}
        for delta in display_round.get("deltas", []):
            for key in delta_total:
                delta_total[key] += int(delta.get(key, 0) or 0)

        return {
            "latest_round": latest.get("round", 0),
            "display_round": display_round.get("round", 0),
            "is_latest_round": display_round is latest,
            "stage": latest.get("stage", "initializing"),
            "display_stage": display_round.get("stage", "initializing"),
            "mode": latest.get("mode", ""),
            "scope": latest.get("scope", ""),
            "scenario": display_round.get("scenario", ""),
            "scenario_type": display_round.get("scenario_type", ""),
            "current_page_id": current_page_id,
            "current_page": current_page or {},
            "current_page_match": current_page_match,
            "current_activity": current_activity,
            "trace": display_round.get("trace", []),
            "trace_round": display_round.get("round", 0),
            "trace_is_historical": display_round is not latest,
            "result": display_round.get("result", {}),
            "delta": delta_total,
            "log_total": len(lines),
        }

    def resolve_file(self, run_dir: Path, rel_path: str) -> Path | None:
        if not rel_path:
            return None
        candidate = (run_dir / rel_path).resolve()
        try:
            candidate.relative_to(run_dir.resolve())
        except ValueError:
            return None
        if not candidate.is_file():
            return None
        return candidate


class Handler(BaseHTTPRequestHandler):
    store: OutputStore = None  # injected below
    server_version = "FuncDroidVizServer/1.0"

    def log_message(self, fmt, *args):
        pass

    # ---------- helpers ----------
    def _send_json(self, data, status: int = 200):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str):
        self._send_json({"error": message}, status=status)

    def _send_file(self, path: Path, content_type: str | None = None):
        if not path.is_file():
            self._send_error_json(404, f"File not found: {path.name}")
            return
        ctype = content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_static(self, rel: str):
        rel = unquote(rel).lstrip("/")
        candidate = (STATIC_DIR / rel).resolve()
        try:
            candidate.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self._send_error_json(403, "Access denied")
            return
        self._send_file(candidate)

    # ---------- routing ----------
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        try:
            if path in ("/", "/index.html"):
                self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            elif path.startswith("/static/"):
                self._serve_static(path[len("/static/") :])
            elif path == "/favicon.ico":
                self._send_error_json(404, "no favicon")
            elif path == "/api/runs":
                self._send_json({"runs": self.store.list_runs()})
            elif path.startswith("/api/runs/"):
                self._route_run_api(path[len("/api/runs/") :], query)
            else:
                self._send_error_json(404, "Unknown path")
        except Exception as exc:  # noqa: BLE001 - Isolate request errors so the server remains available.
            self._send_error_json(500, f"Server error: {exc}")

    def _route_run_api(self, rest: str, query: dict):
        parts = [unquote(p) for p in rest.split("/") if p != ""]
        if not parts:
            self._send_error_json(400, "Missing run name")
            return
        run_name = parts[0]
        run_dir = self.store.get_run_dir(run_name)
        if run_dir is None:
            self._send_error_json(404, f"Run not found: {run_name}")
            return
        sub = parts[1] if len(parts) > 1 else ""

        if sub == "summary":
            self._send_json(self.store.summary(run_dir))
        elif sub == "ffg":
            self._send_json(self.store.ffg(run_dir))
        elif sub == "bugs":
            if len(parts) > 2:
                bug_id = parts[2]
                detail = self.store.bug_detail(run_dir, bug_id)
                if detail is None:
                    self._send_error_json(404, f"Defect not found: {bug_id}")
                else:
                    self._send_json(detail)
            else:
                self._send_json({"bugs": self.store.bugs(run_dir)})
        elif sub == "compliance":
            if len(parts) > 2 and parts[2] == "trace":
                rel = (query.get("path") or [""])[0]
                trace = self.store.compliance_trace(run_dir, rel)
                if trace is None:
                    self._send_error_json(404, "Compliance trace not found")
                else:
                    self._send_json(trace)
            else:
                self._send_json(self.store.compliance(run_dir))
        elif sub == "timeline":
            self._send_json(self.store.activity_timeline(run_dir))
        elif sub == "metrics":
            self._send_json(self.store.metrics(run_dir))
        elif sub == "live":
            self._send_json(self.store.live_state(run_dir))
        elif sub == "log":
            self._handle_log(run_dir, query)
        elif sub == "file":
            rel = (query.get("path") or [""])[0]
            target = self.store.resolve_file(run_dir, rel)
            if target is None:
                self._send_error_json(404, f"File not found: {rel}")
            else:
                self._send_file(target)
        else:
            self._send_error_json(404, "Unknown run subpath")

    def _handle_log(self, run_dir: Path, query: dict):
        since = int((query.get("since") or ["0"])[0] or 0)
        limit = int((query.get("limit") or ["500"])[0] or 500)
        lines = self.store.log_lines(run_dir)
        total = len(lines)
        if since > 0:
            start = min(since, total)
            payload = lines[start:]
        else:
            start = max(0, total - limit) if limit > 0 else 0
            payload = lines[start:]
        self._send_json({"total": total, "start_index": start, "lines": payload})


def build_handler(store: OutputStore):
    class BoundHandler(Handler):
        pass

    BoundHandler.store = store
    return BoundHandler


def main():
    parser = argparse.ArgumentParser(description="FuncDroid runtime visualization server")
    parser.add_argument("--host", default=os.getenv("FUNCDROID_WEB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("FUNCDROID_WEB_PORT", "8787")))
    parser.add_argument(
        "--output-dir",
        default=os.getenv("FUNCDROID_OUTPUT_DIR", str(PROJECT_ROOT / "experiments" / "output")),
        help="Run artifact root; defaults to experiments/output/ under the project root",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    store = OutputStore(output_root)
    handler_cls = build_handler(store)

    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    url = f"http://{args.host}:{args.port}/"
    print(f"[FuncDroid Viz] Monitoring directory: {output_root}")
    print(f"[FuncDroid Viz] Server started: {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[FuncDroid Viz] Shutting down server...")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
