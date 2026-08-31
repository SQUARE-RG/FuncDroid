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


PACKAGE_PATTERN = re.compile(
    r"package: name='(?P<package>[^']+)' versionCode='(?P<version_code>[^']*)' "
    r"versionName='(?P<version_name>[^']*)'"
)
LABEL_PATTERN = re.compile(r"application-label(?:-[^:]+)?:'([^']*)'")
MIN_SDK_PATTERN = re.compile(r"sdkVersion:'(\d+)'")
NATIVE_CODE_PATTERN = re.compile(r"native-code: (.+)")
QUOTED_VALUE_PATTERN = re.compile(r"'([^']+)'")
ITERATION_METRIC_PREFIX = "[轮次指标] "
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
DEFAULT_BATCH_ADB_COMMAND_TIMEOUT = max(
    1.0,
    float(os.getenv("BATCH_ADB_COMMAND_TIMEOUT", "30")),
)
DEFAULT_BATCH_ADB_INSTALL_TIMEOUT = max(
    DEFAULT_BATCH_ADB_COMMAND_TIMEOUT,
    float(os.getenv("BATCH_ADB_INSTALL_TIMEOUT", "300")),
)
DEFAULT_BATCH_PROCESS_TIMEOUT = max(
    60.0,
    float(os.getenv("BATCH_PROCESS_TIMEOUT", "7200")),
)
BATCH_PROCESS_TIMEOUT_GRACE = max(
    0.0,
    float(os.getenv("BATCH_PROCESS_TIMEOUT_GRACE", "300")),
)
BATCH_PROCESS_TERMINATE_GRACE = max(
    1.0,
    float(os.getenv("BATCH_PROCESS_TERMINATE_GRACE", "15")),
)


class BatchInterrupted(RuntimeError):
    """Raised on SIGINT/SIGTERM so package cleanup still runs."""


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


def _run(
    command: Sequence[str],
    *,
    check: bool = False,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=check,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


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
        raise FileNotFoundError(f"aapt 不存在: {candidate}")

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
        raise FileNotFoundError("未找到 aapt，请设置 ANDROID_HOME 或传入 --aapt")
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
            print(f"[应用队列] 跳过不存在的 APK 路径: {root}")
            continue
        for candidate in candidates:
            normalized = str(candidate)
            if any(part in normalized for part in DEFAULT_EXCLUDED_PARTS):
                continue
            discovered.append(candidate)
    return discovered


def inspect_apk(path: Path, aapt: str) -> ApkInfo:
    # Some large/protected APKs make aapt abort after it has already emitted a
    # complete badging section. Treat parseable metadata as authoritative and
    # only reject the APK when the required package record is actually absent.
    result = _run([aapt, "dump", "badging", str(path)], timeout=60)
    package_match = PACKAGE_PATTERN.search(result.stdout)
    if package_match is None:
        raise ValueError(f"aapt 未返回 package 元数据 (returncode={result.returncode})")
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


def build_manifest(paths: Iterable[Path], aapt: str) -> tuple[list[ApkInfo], list[dict]]:
    manifest: list[ApkInfo] = []
    rejected: list[dict] = []
    seen_hashes: set[str] = set()
    for index, path in enumerate(paths, start=1):
        try:
            info = inspect_apk(path, aapt)
        except Exception as exc:
            rejected.append({"path": str(path), "reason": f"metadata_error: {exc}"})
            print(f"[应用队列] APK 元数据读取失败 ({index}): {path} | {exc}")
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
    result = _adb(device_id, "shell", "pm", "path", package_name)
    return result.returncode == 0 and "package:" in result.stdout


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
    except Exception as exc:
        cleanup["errors"].append(f"force_stop: {exc}")

    try:
        installed = is_installed(device_id, package_name)
    except Exception as exc:
        cleanup["errors"].append(f"check_installed_before_cleanup: {exc}")
        # The state is unknown, so still attempt both cleanup operations.
        installed = True

    if installed:
        try:
            clear_result = _adb(device_id, "shell", "pm", "clear", package_name)
            cleanup["clear_data"] = clear_result.returncode == 0 and "Success" in clear_result.stdout
        except Exception as exc:
            cleanup["errors"].append(f"clear_data: {exc}")
        try:
            uninstall_result = _adb(device_id, "uninstall", package_name)
            cleanup["uninstall"] = (
                uninstall_result.returncode == 0 and "Success" in uninstall_result.stdout
            )
        except Exception as exc:
            cleanup["errors"].append(f"uninstall: {exc}")

    verification_failed = False
    try:
        cleanup["verified_absent"] = not is_installed(device_id, package_name)
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


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        file_obj.flush()
        os.fsync(file_obj.fileno())


def completed_hashes(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()
    completed: set[str] = set()
    for line in results_path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("cleanup", {}).get("verified_absent") and item.get("sha256"):
            completed.add(str(item["sha256"]))
    return completed


def parse_iteration_metrics(run_log: Path) -> list[dict]:
    """Parse stable key/value iteration records emitted by workflow.graph."""
    if not run_log.exists():
        return []
    records: list[dict] = []
    for line in run_log.read_text(encoding="utf-8", errors="replace").splitlines():
        marker_index = line.find(ITERATION_METRIC_PREFIX)
        if marker_index < 0:
            continue
        payload = line[marker_index + len(ITERATION_METRIC_PREFIX):]
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
            "planner_duplicate_skips": log_text.count("LLM 生成了相似历史场景"),
            "forced_progress_events": log_text.count("强制推进"),
            "node_exceptions": log_text.count("[节点异常]"),
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
    if explicit_timeout_seconds is not None:
        return max(1.0, float(explicit_timeout_seconds))
    if total_time_minutes > 0:
        return total_time_minutes * 60 + BATCH_PROCESS_TIMEOUT_GRACE
    return DEFAULT_BATCH_PROCESS_TIMEOUT


def _signal_process_tree(process: subprocess.Popen, sig: int) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, sig)
            return
        except ProcessLookupError:
            return
        except OSError:
            # Fall back to signalling the direct child if the process group is
            # unavailable (for example if session creation failed externally).
            pass
    if sig == signal.SIGTERM:
        process.terminate()
    else:
        process.kill()


def _terminate_process_tree(
    process: subprocess.Popen,
    grace_seconds: float = BATCH_PROCESS_TERMINATE_GRACE,
) -> None:
    if process.poll() is not None:
        return
    _signal_process_tree(process, signal.SIGTERM)
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    _signal_process_tree(process, signal.SIGKILL)
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        print(
            "[应用队列] 子进程在强制终止后仍未退出，放弃继续等待: "
            f"pid={process.pid}"
        )


def run_one(
    info: ApkInfo,
    *,
    device_id: str,
    total_time_minutes: float,
    max_iterations: int,
    batch_output_dir: Path,
    project_root: Path,
    process_timeout_seconds: float | None = None,
) -> dict:
    started_at = datetime.now().isoformat()
    started_monotonic = time.monotonic()
    app_output_dir = batch_output_dir / safe_output_name(info)
    result: dict = {
        **asdict(info),
        "started_at": started_at,
        "output_dir": str(app_output_dir),
        "install": {},
        "test": {},
        "cleanup": {},
    }

    # A stale copy of the same package would make the test state non-reproducible.
    pre_cleanup = cleanup_package(device_id, info.package_name)
    result["pre_cleanup"] = pre_cleanup
    if not pre_cleanup["verified_absent"]:
        result["test"] = {"status": "skipped", "reason": "pre_cleanup_failed"}
        result["cleanup"] = pre_cleanup
        return result

    try:
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
        print(f"[应用队列] 开始运行: {info.package_name} | {info.path}")
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
                "[应用队列] App 测试进程超过硬超时，终止进程树: "
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
        result["test"] = {
            "status": "completed" if test_returncode == 0 else "failed",
            "returncode": test_returncode,
            "command": command,
        }
    except Exception as exc:
        result["test"] = {"status": "failed", "reason": str(exc)}
    finally:
        result["metrics"] = collect_run_metrics(app_output_dir)
        result["cleanup"] = cleanup_package(device_id, info.package_name)
        result["finished_at"] = datetime.now().isoformat()
        result["elapsed_seconds"] = round(time.monotonic() - started_monotonic, 3)
    return result


def _signal_handler(signum, _frame):
    raise BatchInterrupted(f"received_signal_{signum}")


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按顺序安装、测试并彻底清理本地 APK")
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--apk-root", action="append", required=True,
                        help="APK 文件或目录；可以重复传入")
    parser.add_argument("--output-dir", default="output/apk-batch")
    parser.add_argument("--total-time", type=float, default=30.0,
                        help="每个 App 的 FuncDroid 时间预算（分钟）")
    parser.add_argument(
        "--process-timeout",
        type=_nonnegative_float,
        default=0.0,
        help=(
            "每个 App 子进程的硬超时（分钟）；默认 0 表示自动使用 "
            "total-time 加宽限期，total-time=0 时使用 BATCH_PROCESS_TIMEOUT"
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=_nonnegative_int,
        default=0,
        help="每个 App 的工作流轮次上限；默认 0 表示仅受时间预算约束",
    )
    parser.add_argument("--aapt", default="")
    parser.add_argument("--limit", type=int, default=0,
                        help="只运行清单前 N 个；0 表示全部")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    batch_output_dir = Path(args.output_dir).expanduser().resolve()
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    results_path = batch_output_dir / "batch-results.jsonl"

    aapt = find_aapt(args.aapt)
    apk_paths = discover_apks(args.apk_root)
    manifest, rejected = build_manifest(apk_paths, aapt)
    device_sdk, device_abis = device_capabilities(args.device_id)

    manifest_payload = {
        "generated_at": datetime.now().isoformat(),
        "device_id": args.device_id,
        "device_sdk": device_sdk,
        "device_abis": sorted(device_abis),
        "apks": [asdict(item) for item in manifest],
        "rejected": rejected,
    }
    manifest_path = batch_output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[应用队列] 清单完成: discovered={len(apk_paths)}, unique={len(manifest)}, "
        f"rejected={len(rejected)}, manifest={manifest_path}"
    )
    if args.manifest_only:
        return 0

    completed = set() if args.no_resume else completed_hashes(results_path)
    pending = [item for item in manifest if item.sha256 not in completed]
    if args.limit > 0:
        pending = pending[:args.limit]

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    exit_code = 0
    try:
        for index, info in enumerate(pending, start=1):
            reason = compatibility_reason(info, device_sdk, device_abis)
            if reason:
                payload = {
                    **asdict(info),
                    "test": {"status": "skipped", "reason": reason},
                    "cleanup": {"verified_absent": not is_installed(args.device_id, info.package_name)},
                }
                append_jsonl(results_path, payload)
                print(f"[应用队列] 跳过不兼容 APK ({index}/{len(pending)}): {info.path} | {reason}")
                continue
            result = run_one(
                info,
                device_id=args.device_id,
                total_time_minutes=args.total_time,
                max_iterations=args.max_iterations,
                batch_output_dir=batch_output_dir,
                project_root=project_root,
                process_timeout_seconds=(
                    args.process_timeout * 60 if args.process_timeout > 0 else None
                ),
            )
            append_jsonl(results_path, result)
            if not result.get("cleanup", {}).get("verified_absent"):
                print(f"[应用队列] 清理校验失败，停止队列以避免污染下一个 App: {info.package_name}")
                exit_code = 2
                break
            if result.get("batch_interrupted"):
                print("[应用队列] 当前 App 已清理，队列安全停止")
                exit_code = 130
                break
    except BatchInterrupted as exc:
        print(f"[应用队列] 队列被中断: {exc}")
        exit_code = 130
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
