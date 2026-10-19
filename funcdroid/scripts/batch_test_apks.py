#!/usr/bin/env python3
"""Install, test, and clean APKs on one Android emulator sequentially."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from dotenv import load_dotenv

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._batch_common import (  # noqa: E402 - retain legacy module exports
    BATCH_PROCESS_TERMINATE_GRACE,
    BATCH_PROCESS_TIMEOUT_GRACE,
    DEFAULT_BATCH_ADB_COMMAND_TIMEOUT,
    DEFAULT_BATCH_PROCESS_TIMEOUT,
    BatchInterrupted,
    append_jsonl,
    nonnegative_float as _nonnegative_float,
    nonnegative_int as _nonnegative_int,
    query_installed,
    resolve_process_timeout,
    run_artifacts_complete,
    run_command as _run,
    signal_handler as _signal_handler,
    signal_process_tree as _signal_process_tree,  # noqa: F401 - retain the importable alias
    terminate_process_tree,
)

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

PACKAGE_PATTERN = re.compile(
    r"package: name='(?P<package>[^']+)' versionCode='(?P<version_code>[^']*)' "
    r"versionName='(?P<version_name>[^']*)'"
)
LABEL_PATTERN = re.compile(r"application-label(?:-[^:]+)?:'([^']*)'")
MIN_SDK_PATTERN = re.compile(r"sdkVersion:'(\d+)'")
NATIVE_CODE_PATTERN = re.compile(r"native-code: (.+)")
QUOTED_VALUE_PATTERN = re.compile(r"'([^']+)'")
ITERATION_METRIC_PREFIX = "[Iteration metrics] "
# Accept original archived log markers without modifying historical logs.
LEGACY_ITERATION_METRIC_PREFIX = "[\u8f6e\u6b21\u6307\u6807] "
INTEGER_METRIC_FIELDS = {
    "iteration",
    "planner_noop_count",
    "llm_requests",
    "llm_tokens",
    "functions",
    "pages",
    "completed_functions",
    "attempted_scenarios",
}
FLOAT_METRIC_FIELDS = {"elapsed_seconds", "completion_rate"}
DEFAULT_EXCLUDED_PARTS = (
    "/Library/Android/sdk/",
    "/site-packages/uiautomator2/",
    "/.venv/",
    "/venv/",
)
DEFAULT_BATCH_ADB_INSTALL_TIMEOUT = max(
    DEFAULT_BATCH_ADB_COMMAND_TIMEOUT,
    float(os.getenv("BATCH_ADB_INSTALL_TIMEOUT", "300")),
)

@dataclass(frozen=True)
class ApkInfo:
    path: str
    sha256: str
    package_name: str
    version_code: str
    version_name: str
    app_label: str
    min_sdk: int
    native_codes: tuple[str, ...]


def _adb(
    device_id: str,
    *args: str,
    check: bool = False,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return _run(
        ["adb", "-s", device_id, *args],
        check=check,
        timeout=DEFAULT_BATCH_ADB_COMMAND_TIMEOUT if timeout is None else timeout,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_aapt(explicit_path: str = "") -> str:
    if explicit_path:
        candidate = Path(explicit_path).expanduser().resolve()
        if candidate.is_file():
            return str(candidate)
        raise FileNotFoundError(f"aapt does not exist: {candidate}")

    from_path = shutil.which("aapt")
    if from_path:
        return from_path

    sdk_root = Path(
        os.getenv("ANDROID_HOME")
        or os.getenv("ANDROID_SDK_ROOT")
        or Path.home() / "Library/Android/sdk"
    )
    candidates = sorted(sdk_root.glob("build-tools/*/aapt"))
    if not candidates:
        return ""
    stable_candidates = [candidate for candidate in candidates if "-" not in candidate.parent.name]
    return str((stable_candidates or candidates)[-1])


def discover_apks(roots: Iterable[str]) -> list[Path]:
    discovered: list[Path] = []
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        if root.is_file() and root.suffix.lower() == ".apk":
            candidates = [root]
        elif root.is_dir():
            candidates = sorted(root.rglob("*.apk"))
        else:
            print(f"[App queue] Skipping missing APK path: {root}")
            continue
        for candidate in candidates:
            normalized = str(candidate)
            if any(part in normalized for part in DEFAULT_EXCLUDED_PARTS):
                continue
            discovered.append(candidate)
    return discovered


def inspect_apk(path: Path, aapt: str) -> ApkInfo:
    if not aapt:
        return inspect_apk_with_python(path)
    # Some large/protected APKs make aapt abort after it has already emitted a
    # complete badging section. Treat parseable metadata as authoritative and
    # only reject the APK when the required package record is actually absent.
    result = _run([aapt, "dump", "badging", str(path)], timeout=60)
    package_match = PACKAGE_PATTERN.search(result.stdout)
    if package_match is None:
        raise ValueError(f"aapt returned no package metadata (returncode={result.returncode})")
    label_match = LABEL_PATTERN.search(result.stdout)
    sdk_match = MIN_SDK_PATTERN.search(result.stdout)
    native_match = NATIVE_CODE_PATTERN.search(result.stdout)
    native_codes = (
        tuple(QUOTED_VALUE_PATTERN.findall(native_match.group(1)))
        if native_match
        else ()
    )
    return ApkInfo(
        path=str(path),
        sha256=_sha256(path),
        package_name=package_match.group("package"),
        version_code=package_match.group("version_code"),
        version_name=package_match.group("version_name"),
        app_label=(label_match.group(1).strip() if label_match else path.stem),
        min_sdk=int(sdk_match.group(1)) if sdk_match else 1,
        native_codes=native_codes,
    )


def inspect_apk_with_python(path: Path) -> ApkInfo:
    """Read binary Android manifests when SDK build-tools are not installed."""
    from pyaxmlparser import APK

    apk = APK(str(path))
    package_name = apk.get_package()
    if not package_name:
        raise ValueError("APK has no valid Android package name")
    native_codes = tuple(sorted({
        name.split("/")[1] for name in apk.get_files()
        if name.startswith("lib/") and name.endswith(".so") and len(name.split("/")) == 3
    }))
    return ApkInfo(
        path=str(path), sha256=_sha256(path), package_name=package_name,
        version_code=str(apk.get_androidversion_code() or ""),
        version_name=str(apk.get_androidversion_name() or ""),
        app_label=str(apk.get_app_name() or path.stem),
        min_sdk=int(apk.get_min_sdk_version() or 1), native_codes=native_codes,
    )


def build_manifest(paths: Iterable[Path], aapt: str) -> tuple[list[ApkInfo], list[dict]]:
    manifest: list[ApkInfo] = []
    rejected: list[dict] = []
    seen_hashes: set[str] = set()
    for index, path in enumerate(paths, start=1):
        try:
            info = inspect_apk(path, aapt)
        except Exception as exc:
            rejected.append({"path": str(path), "reason": f"metadata_error: {exc}"})
            print(f"[App queue] APK metadata read failed ({index}): {path} | {exc}")
            continue
        if info.sha256 in seen_hashes:
            rejected.append({"path": str(path), "reason": "duplicate_sha256"})
            continue
        seen_hashes.add(info.sha256)
        manifest.append(info)
    return manifest, rejected


def device_capabilities(device_id: str) -> tuple[int, set[str]]:
    sdk = int(_adb(device_id, "shell", "getprop", "ro.build.version.sdk", check=True).stdout.strip())
    abilist = _adb(device_id, "shell", "getprop", "ro.product.cpu.abilist", check=True).stdout
    return sdk, {item.strip() for item in abilist.split(",") if item.strip()}


def compatibility_reason(info: ApkInfo, device_sdk: int, device_abis: set[str]) -> str:
    if info.min_sdk > device_sdk:
        return f"min_sdk_{info.min_sdk}_gt_device_{device_sdk}"
    if info.native_codes and not device_abis.intersection(info.native_codes):
        return "native_abi_mismatch"
    return ""


def is_installed(device_id: str, package_name: str) -> bool:
    """Query installation; any failure or timeout means unknown state and raises RuntimeError."""
    return query_installed(_adb, device_id, package_name)


def cleanup_package(device_id: str, package_name: str) -> dict:
    """Clear data and uninstall one tested package, then verify removal."""
    cleanup = {
        "package_name": package_name,
        "force_stop": False,
        "clear_data": False,
        "uninstall": False,
        "verified_absent": False,
        "errors": [],
    }
    if not package_name:
        cleanup["verified_absent"] = True
        return cleanup

    try:
        cleanup["force_stop"] = _adb(
            device_id, "shell", "am", "force-stop", package_name
        ).returncode == 0
    except BatchInterrupted:
        raise
    except Exception as exc:
        cleanup["errors"].append(f"force_stop: {exc}")

    try:
        installed = is_installed(device_id, package_name)
    except BatchInterrupted:
        raise
    except Exception as exc:
        cleanup["errors"].append(f"check_installed_before_cleanup: {exc}")
        # The state is unknown, so still attempt both cleanup operations.
        installed = True

    if installed:
        try:
            clear_result = _adb(device_id, "shell", "pm", "clear", package_name)
            cleanup["clear_data"] = clear_result.returncode == 0 and "Success" in clear_result.stdout
        except BatchInterrupted:
            raise
        except Exception as exc:
            cleanup["errors"].append(f"clear_data: {exc}")
        try:
            uninstall_result = _adb(device_id, "uninstall", package_name)
            cleanup["uninstall"] = (
                uninstall_result.returncode == 0 and "Success" in uninstall_result.stdout
            )
        except BatchInterrupted:
            raise
        except Exception as exc:
            cleanup["errors"].append(f"uninstall: {exc}")

    verification_failed = False
    try:
        cleanup["verified_absent"] = not is_installed(device_id, package_name)
    except BatchInterrupted:
        raise
    except Exception as exc:
        cleanup["errors"].append(f"verify_absent: {exc}")
        cleanup["verified_absent"] = False
        verification_failed = True
    if not cleanup["verified_absent"]:
        cleanup["errors"].append(
            "package_absence_unverified" if verification_failed else "package_still_installed"
        )
    return cleanup


def install_apk(device_id: str, info: ApkInfo) -> subprocess.CompletedProcess[str]:
    return _adb(
        device_id,
        "install",
        "-r",
        "-t",
        info.path,
        timeout=DEFAULT_BATCH_ADB_INSTALL_TIMEOUT,
    )


def safe_output_name(info: ApkInfo) -> str:
    version = re.sub(r"[^A-Za-z0-9._-]+", "_", info.version_name or info.version_code or "unknown")
    return f"{info.package_name}__{version}__{info.sha256[:10]}"


def completed_hashes(results_path: Path, mode: str = "defect") -> set[str]:
    if not results_path.exists():
        return set()
    completed: set[str] = set()
    for line in results_path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict) or item.get("mode", "defect") != mode:
            continue
        digest = str(item.get("sha256", ""))
        if not digest:
            continue
        completed.discard(digest)
        if (item.get("cleanup", {}).get("verified_absent")
                and item.get("test", {}).get("status") == "completed"
                and item.get("test", {}).get("artifacts_complete") is True):
            completed.add(digest)
    return completed


def parse_iteration_metrics(run_log: Path) -> list[dict]:
    """Parse stable key/value iteration records emitted by workflow.graph."""
    if not run_log.exists():
        return []
    records: list[dict] = []
    for line in run_log.read_text(encoding="utf-8", errors="replace").splitlines():
        prefix = next((item for item in (ITERATION_METRIC_PREFIX, LEGACY_ITERATION_METRIC_PREFIX) if item in line), None)
        if prefix is None:
            continue
        marker_index = line.find(prefix)
        payload = line[marker_index + len(prefix):]
        record: dict = {}
        for item in payload.split(", "):
            key, separator, value = item.partition("=")
            if not separator:
                continue
            key = key.strip()
            value = value.strip()
            try:
                if key in INTEGER_METRIC_FIELDS:
                    record[key] = int(value)
                elif key in FLOAT_METRIC_FIELDS:
                    record[key] = float(value)
                else:
                    record[key] = value
            except ValueError:
                record[key] = value
        if "iteration" in record:
            records.append(record)
    return records


def _max_consecutive(records: list[dict], fields: tuple[str, ...], *, ignore_empty=False) -> int:
    maximum = current = 0
    previous: tuple[object, ...] | None = None
    for record in records:
        signature = tuple(record.get(field) for field in fields)
        if ignore_empty and all(value in {None, "", "none"} for value in signature):
            previous = None
            current = 0
            continue
        current = current + 1 if signature == previous else 1
        previous = signature
        maximum = max(maximum, current)
    return maximum


def collect_run_metrics(app_output_dir: Path) -> dict:
    """Build a machine-readable performance, coverage, and stall summary."""
    run_log = app_output_dir / "run.log"
    records = parse_iteration_metrics(run_log)
    elapsed_values = sorted(
        float(item.get("elapsed_seconds", 0) or 0) for item in records
    )
    percentile_index = max(0, min(len(elapsed_values) - 1, int(len(elapsed_values) * 0.95)))
    final = records[-1] if records else {}
    same_scope_page_streak = _max_consecutive(
        records, ("scenario_type", "scope", "page"), ignore_empty=True
    )
    no_coverage_progress_streak = _max_consecutive(
        records,
        ("functions", "pages", "completed_functions", "attempted_scenarios"),
    )
    same_scope_without_progress_streak = _max_consecutive(
        records,
        (
            "scenario_type",
            "scope",
            "page",
            "functions",
            "pages",
            "completed_functions",
            "attempted_scenarios",
        ),
        ignore_empty=True,
    )

    usage_totals: dict = {}
    usage_path = app_output_dir / "llm_usage.json"
    if usage_path.exists():
        try:
            usage_payload = json.loads(usage_path.read_text(encoding="utf-8"))
            if isinstance(usage_payload, dict) and isinstance(usage_payload.get("totals"), dict):
                usage_totals = usage_payload["totals"]
        except (OSError, json.JSONDecodeError):
            pass

    activity_discovery = {"count": 0, "activities": [], "point_count": 0}
    activity_path = app_output_dir / "activity_timeline.json"
    if activity_path.exists():
        try:
            activity_payload = json.loads(activity_path.read_text(encoding="utf-8"))
            if isinstance(activity_payload, dict):
                activities = activity_payload.get("activities", [])
                points = activity_payload.get("points", [])
                if isinstance(activities, list):
                    activity_discovery["activities"] = [
                        str(item) for item in activities if str(item).strip()
                    ]
                    activity_discovery["count"] = len(activity_discovery["activities"])
                if isinstance(points, list):
                    activity_discovery["point_count"] = len(points)
        except (OSError, json.JSONDecodeError):
            pass

    log_text = run_log.read_text(encoding="utf-8", errors="replace") if run_log.exists() else ""
    summary = {
        "iteration_count": len(records),
        "iteration_elapsed_seconds": {
            "average": round(sum(elapsed_values) / len(elapsed_values), 3) if elapsed_values else 0.0,
            "p95": round(elapsed_values[percentile_index], 3) if elapsed_values else 0.0,
            "maximum": round(max(elapsed_values), 3) if elapsed_values else 0.0,
        },
        "llm": {
            "requests": int(usage_totals.get("requests", 0) or 0),
            "failures": int(usage_totals.get("failures", 0) or 0),
            "prompt_tokens": int(usage_totals.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage_totals.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage_totals.get("total_tokens", 0) or 0),
        },
        "coverage": {
            "active_functions": int(final.get("functions", 0) or 0),
            "active_pages": int(final.get("pages", 0) or 0),
            "completed_functions": int(final.get("completed_functions", 0) or 0),
            "function_completion_rate": float(final.get("completion_rate", 0) or 0),
            "attempted_scenarios": int(final.get("attempted_scenarios", 0) or 0),
        },
        "activity_discovery": activity_discovery,
        "stall": {
            "max_same_scope_page_streak": same_scope_page_streak,
            "max_same_scope_page_without_progress_streak": same_scope_without_progress_streak,
            "max_no_coverage_progress_streak": no_coverage_progress_streak,
            "planner_duplicate_skips": log_text.count("LLM generated a similar historical scenario") + log_text.count("\u004c\u004c\u004d \u751f\u6210\u4e86\u76f8\u4f3c\u5386\u53f2\u573a\u666f"),
            "forced_progress_events": log_text.count("Forced progress") + log_text.count("\u5f3a\u5236\u63a8\u8fdb"),
            "node_exceptions": log_text.count("[Node error]") + log_text.count("[\u8282\u70b9\u5f02\u5e38]"),
            "suspected": same_scope_without_progress_streak >= 4,
        },
        "status_counts": {},
        "iterations": records,
    }
    for record in records:
        status = str(record.get("scenario_status", "none") or "none")
        summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1
    app_output_dir.mkdir(parents=True, exist_ok=True)
    (app_output_dir / "iteration-metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _resolve_process_timeout(
    total_time_minutes: float,
    explicit_timeout_seconds: float | None,
) -> float:
    return resolve_process_timeout(
        total_time_minutes,
        explicit_timeout_seconds,
        grace_seconds=BATCH_PROCESS_TIMEOUT_GRACE,
        unlimited_timeout_seconds=DEFAULT_BATCH_PROCESS_TIMEOUT,
    )


def _terminate_process_tree(
    process: subprocess.Popen,
    grace_seconds: float = BATCH_PROCESS_TERMINATE_GRACE,
) -> None:
    terminate_process_tree(process, grace_seconds, log_prefix="[App queue]")


def resolve_seed_dir(root: str, package_name: str) -> str:
    """Find a completed run for this package beneath root to supply as --seed-ffg.

    load_seed_graph requires run_result.json with status=success and
    artifact_manifest.json with status=complete. Apply the same checks here
    rather than passing an invalid seed directory to main.py.
    """
    if not root:
        return ""
    base = Path(root).expanduser()
    if not base.is_dir():
        return ""
    for candidate in sorted(base.iterdir()):
        if not candidate.is_dir():
            continue
        result_path = candidate / "run_result.json"
        manifest_path = candidate / "artifact_manifest.json"
        if not result_path.exists() or not manifest_path.exists():
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if result.get("package_name") != package_name:
            continue
        if result.get("status") != "success" or manifest.get("status") != "complete":
            continue
        return str(candidate)
    return ""


def run_one(
    info: ApkInfo,
    *,
    device_id: str,
    total_time_minutes: float,
    max_iterations: int,
    batch_output_dir: Path,
    project_root: Path,
    process_timeout_seconds: float | None = None,
    mode: str = "defect",
    keep_app: bool = False,
    seed_ffg_root: str = "",
) -> dict:
    """Run one app in an APK batch.

    keep_app=True skips clearing and uninstalling before and after the run and
    requires a preinstalled app with its session prepared manually. Defect testing
    may depend on authentication, which the default clean-install-test-clean cycle
    would erase. Preserving the app allows those runs without bypassing this runner.
    """
    if mode not in {"defect", "compliance"}:
        raise ValueError(f"Unsupported detection mode: {mode}")
    started_at = datetime.now().isoformat()
    started_monotonic = time.monotonic()
    app_output_dir = batch_output_dir / safe_output_name(info)
    # main.py requires an empty new-run directory. Keep previous evidence on retry.
    if app_output_dir.exists():
        archive = batch_output_dir / "previous-attempts"
        archive.mkdir(exist_ok=True)
        app_output_dir.rename(archive / f"{app_output_dir.name}__{time.time_ns()}")
    result: dict = {
        **asdict(info),
        "mode": mode,
        "started_at": started_at,
        "output_dir": str(app_output_dir),
        "install": {},
        "test": {},
        "cleanup": {},
    }

    if keep_app:
        result["pre_cleanup"] = {"skipped": "keep_app", "verified_absent": True}
    else:
        # A stale copy of the same package would make the test state non-reproducible.
        pre_cleanup = cleanup_package(device_id, info.package_name)
        result["pre_cleanup"] = pre_cleanup
        if not pre_cleanup["verified_absent"]:
            result["test"] = {"status": "skipped", "reason": "pre_cleanup_failed"}
            result["cleanup"] = pre_cleanup
            return result

    try:
        if keep_app:
            result["install"] = {"skipped": "keep_app"}
            if not is_installed(device_id, info.package_name):
                result["test"] = {
                    "status": "skipped",
                    "reason": "keep_app_requires_preinstalled_app",
                }
                return result
        else:
            install_result = install_apk(device_id, info)
            result["install"] = {
                "returncode": install_result.returncode,
                "stdout": install_result.stdout[-4000:],
                "stderr": install_result.stderr[-4000:],
            }
            if install_result.returncode != 0 or not is_installed(device_id, info.package_name):
                result["test"] = {"status": "skipped", "reason": "install_failed"}
                return result

        command = [
            sys.executable,
            str(project_root / "main.py"),
            "--mode", mode,
            "--device-id",
            device_id,
            "--package-name",
            info.package_name,
            "--app-name",
            info.app_label or info.package_name,
            "--phase",
            "all",
            "--max-iterations",
            str(max_iterations),
            "--total-time",
            str(total_time_minutes),
            "--output-dir",
            str(app_output_dir),
        ]
        if keep_app:
            command.append("--no-reset-app-data")
        seed_dir = resolve_seed_dir(seed_ffg_root, info.package_name)
        if seed_dir:
            command.extend(["--seed-ffg", seed_dir])
            result["seed_ffg"] = seed_dir
            print(f"[App queue] Reusing the preceding run's FFG as a seed: {seed_dir}")
        elif seed_ffg_root:
            print(f"[App queue] No usable FFG seed found; starting fresh exploration: {info.package_name}")
        print(f"[App queue] Starting: {info.package_name} | {info.path}")
        hard_timeout = _resolve_process_timeout(
            total_time_minutes,
            process_timeout_seconds,
        )
        test_process = subprocess.Popen(
            command,
            cwd=project_root,
            start_new_session=os.name == "posix",
        )
        try:
            test_returncode = test_process.wait(timeout=hard_timeout)
        except subprocess.TimeoutExpired:
            print(
                "[App queue] App test exceeded its hard timeout; terminating the process tree: "
                f"{info.package_name} | timeout={hard_timeout:g}s"
            )
            _terminate_process_tree(test_process)
            result["test"] = {
                "status": "timed_out",
                "reason": "process_hard_timeout",
                "timeout_seconds": hard_timeout,
                "command": command,
            }
            return result
        except BatchInterrupted as exc:
            _terminate_process_tree(test_process)
            result["test"] = {
                "status": "interrupted",
                "reason": str(exc),
                "command": command,
            }
            result["batch_interrupted"] = True
            return result
        artifacts_complete = run_artifacts_complete(app_output_dir, mode, info.package_name)
        result["test"] = {
            "status": "completed" if test_returncode == 0 and artifacts_complete else "failed",
            "artifacts_complete": artifacts_complete,
            "returncode": test_returncode,
            "command": command,
        }
    except BatchInterrupted as exc:
        # Interrupts during installation or verification must also stop the queue;
        # do not let the ordinary exception handler consume them and continue.
        result["test"] = {"status": "interrupted", "reason": str(exc)}
        result["batch_interrupted"] = True
    except Exception as exc:
        result["test"] = {"status": "failed", "reason": str(exc)}
    finally:
        try:
            result["metrics"] = collect_run_metrics(app_output_dir)
        except Exception as exc:
            result["metrics_error"] = str(exc)
        finally:
            if keep_app:
                # keep_app preserves the session for later runs, without clearing or uninstalling.
                # verified_absent=True bypasses the queue's cleanup-failure stop condition;
                # it means cleanup was deliberately skipped, not that absence was verified.
                result["cleanup"] = {"skipped": "keep_app", "verified_absent": True}
            else:
                try:
                    result["cleanup"] = cleanup_package(device_id, info.package_name)
                except BatchInterrupted as exc:
                    # A second interrupt during cleanup leaves cleanup unverified; the queue must stop.
                    result["cleanup"] = {
                        "package_name": info.package_name,
                        "verified_absent": False,
                        "errors": [f"cleanup_interrupted: {exc}"],
                    }
                    result["batch_interrupted"] = True
        result["finished_at"] = datetime.now().isoformat()
        result["elapsed_seconds"] = round(time.monotonic() - started_monotonic, 3)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sequentially install, test, and clean local APKs")
    parser.add_argument("--mode", choices=("defect", "compliance"), default="defect")
    parser.add_argument("--device-id", help="Required for device testing; unnecessary for offline manifest generation")
    parser.add_argument("--apk-root", action="append", required=True,
                        help="APK file or directory; may be supplied multiple times")
    parser.add_argument("--output-dir", default="experiments/output/apk-batch")
    parser.add_argument("--total-time", type=_nonnegative_float, default=30.0,
                        help="FuncDroid time budget for each app, in minutes")
    parser.add_argument(
        "--process-timeout",
        type=_nonnegative_float,
        default=0.0,
        help=(
            "Hard timeout per app subprocess in minutes; 0 derives it from "
            "total-time plus grace, or BATCH_PROCESS_TIMEOUT when total-time=0"
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=_nonnegative_int,
        default=0,
        help="Workflow iteration cap per app; 0 uses only the time budget",
    )
    parser.add_argument("--aapt", default="")
    parser.add_argument("--limit", type=_nonnegative_int, default=0,
                        help="Run only the first N manifest entries; 0 means all")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--keep-app",
        action="store_true",
        help=(
            "Skip data clearing and uninstalling before and after testing. Every app must be preinstalled "
            "and manually signed in when required; the default clean-install-test-clean cycle removes "
            "sessions needed for authenticated defect testing. This mode processes only prepared apps "
            "and is unsuitable for an unattended list requiring manual login."
        ),
    )
    parser.add_argument(
        "--seed-ffg-root",
        default="",
        help=(
            "Find a completed run beneath this directory by package name (run_result.json=success "
            "and artifact_manifest.json=complete) and pass it as --seed-ffg. "
            "Defect testing can reuse the same app's earlier exploration graph to focus on business logic; "
            "apps without a usable seed start fresh exploration."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Record failed or incomplete app tests and continue, rather than stopping for review at the "
            "first incomplete run. Cleanup-verification failures still stop the queue to prevent "
            "residual state from affecting subsequent apps. Suitable for unattended APK batches."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.manifest_only and not args.device_id:
        parser.error("Device testing requires --device-id; use --manifest-only for an offline manifest")
    project_root = Path(__file__).resolve().parents[1]
    batch_output_dir = Path(args.output_dir).expanduser().resolve()
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    results_path = batch_output_dir / "batch-results.jsonl"
    manifest_path = batch_output_dir / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("mode", "defect") != args.mode:
            parser.error("Output directory belongs to another detection mode; specify separate --output-dir values")
    elif results_path.exists():
        parser.error("Output directory has no batch manifest, so its mode cannot be verified; use a new --output-dir")

    aapt = find_aapt(args.aapt)
    print(f"[App queue] APK metadata parser: {aapt or 'pyaxmlparser'}")
    apk_paths = discover_apks(args.apk_root)
    manifest, rejected = build_manifest(apk_paths, aapt)
    if not args.manifest_only:
        # Match create_llm() in workflow/llm.py: prefer the shared endpoint settings,
        # then permit DEEPSEEK_* fallbacks. Otherwise valid fallback-only configurations
        # would be incorrectly rejected before starting a runnable batch.
        missing = [
            primary
            for primary, fallback in (
                ("API_KEY", "DEEPSEEK_API_KEY"),
                ("BASE_URL", "DEEPSEEK_BASE_URL"),
                ("MODEL", "DEEPSEEK_MODEL"),
            )
            if not os.getenv(primary, "").strip() and not os.getenv(fallback, "").strip()
        ]
        if missing:
            parser.error("Missing model configuration; fill .env first: " + ", ".join(missing))
    device_sdk, device_abis = (None, set()) if args.manifest_only else device_capabilities(args.device_id)

    manifest_payload = {
        "generated_at": datetime.now().isoformat(),
        "mode": args.mode,
        "metadata_parser": aapt or "pyaxmlparser",
        "device_id": args.device_id,
        "device_sdk": device_sdk,
        "device_abis": sorted(device_abis),
        "apks": [asdict(item) for item in manifest],
        "rejected": rejected,
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[App queue] Manifest ready: discovered={len(apk_paths)}, unique={len(manifest)}, "
        f"rejected={len(rejected)}, manifest={manifest_path}"
    )
    if args.manifest_only:
        return 0 if manifest and not rejected else 1

    if not manifest:
        print("[App queue] No runnable APKs")
        return 1

    completed = set() if args.no_resume else completed_hashes(results_path, args.mode)
    pending = [item for item in manifest if item.sha256 not in completed]
    if args.limit > 0:
        pending = pending[:args.limit]

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    exit_code = 1 if rejected else 0
    try:
        for index, info in enumerate(pending, start=1):
            reason = compatibility_reason(info, device_sdk, device_abis)
            if reason:
                try:
                    skip_cleanup = {"verified_absent": not is_installed(args.device_id, info.package_name)}
                except BatchInterrupted:
                    raise
                except RuntimeError as exc:
                    # An installation-state query failure affects only this skipped item, not the whole batch.
                    skip_cleanup = {"verified_absent": False, "errors": [f"verify_absent: {exc}"]}
                payload = {
                    **asdict(info),
                    "mode": args.mode,
                    "test": {"status": "skipped", "reason": reason},
                    "cleanup": skip_cleanup,
                }
                append_jsonl(results_path, payload)
                print(f"[App queue] Skipping incompatible APK ({index}/{len(pending)}): {info.path} | {reason}")
                exit_code = 1
                continue
            result = run_one(
                info,
                device_id=args.device_id,
                total_time_minutes=args.total_time,
                max_iterations=args.max_iterations,
                batch_output_dir=batch_output_dir,
                project_root=project_root,
                mode=args.mode,
                keep_app=args.keep_app,
                seed_ffg_root=args.seed_ffg_root,
                process_timeout_seconds=(
                    args.process_timeout * 60 if args.process_timeout > 0 else None
                ),
            )
            append_jsonl(results_path, result)
            if not result.get("cleanup", {}).get("verified_absent"):
                print(f"[App queue] Cleanup verification failed; stopping before the next app: {info.package_name}")
                exit_code = 2
                break
            if result.get("batch_interrupted"):
                print("[App queue] Current app cleaned up; queue stopped safely")
                exit_code = 130
                break
            if result.get("test", {}).get("status") != "completed":
                if args.continue_on_error:
                    print(
                        f"[App queue] Incomplete test recorded; continuing to the next app: {info.package_name}"
                    )
                    exit_code = 1
                    continue
                print(f"[App queue] Incomplete test; stopping the queue for review: {info.package_name}")
                exit_code = 1
                break
    except BatchInterrupted as exc:
        print(f"[App queue] Queue interrupted: {exc}")
        exit_code = 130
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
