"""One installation, two independent budgets, shared navigation knowledge.

Run with: python -m scripts.staged_apk_experiment --help
"""

import argparse
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

from scripts import batch_test_apks as batch


def positive_minutes(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Stage budget must be a finite positive number")
    return number


def write_result(path, result):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def stage_command(info, device_id, mode, minutes, output, seed=None):
    command = [sys.executable, "-u", "main.py", "--device-id", device_id,
               "--package-name", info.package_name, "--app-name", info.app_label,
               "--mode", mode, "--total-time", str(minutes), "--max-iterations", "0",
               "--output-dir", str(output)]
    if seed:
        command.extend(["--seed-ffg", str(seed), "--no-reset-app-data"])
    return command


def snapshot_sources(project, archive):
    sources = [project / "main.py", *[p for folder in (
        "workflow", "scripts", "model", "compliance", "tools", "automator", "connector")
        for p in (project / folder).rglob("*") if p.suffix in {".py", ".json"}]]
    hashes = {}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as snapshot:
        for path in sorted(sources):
            relative = str(path.relative_to(project))
            data = path.read_bytes()
            hashes[relative] = hashlib.sha256(data).hexdigest()
            snapshot.writestr(relative, data)
    return hashes


def network_state(device_id):
    result = batch._adb(device_id, "shell", "dumpsys", "connectivity")
    match = re.search(r"Active default network:\s*(\S+)", result.stdout)
    if result.returncode or not match:
        return {"status": "unknown"}
    network = match.group(1)
    return {"status": "disconnected" if network.lower() == "none" else "connected",
            "default_network": network,
            "scope": "Default route only; does not prove app backend availability"}


def run_experiment(args):
    project = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).resolve()
    # Never erase a pre-existing installation or experiment as a side effect.
    info = batch.inspect_apk(Path(args.apk).resolve(), "")
    if batch.is_installed(args.device_id, info.package_name):
        raise ValueError(f"Target package is already installed; refusing to erase it as a fresh experiment: {info.package_name}")
    sdk, abis = batch.device_capabilities(args.device_id)
    reason = batch.compatibility_reason(info, sdk, abis)
    if reason:
        raise ValueError(f"APK is incompatible with the device: {reason}")
    network = network_state(args.device_id)
    if getattr(args, "require_network", False) and network["status"] != "connected":
        raise ValueError("The experiment requires a default device network connection; connect the device first")
    output.mkdir(parents=True, exist_ok=False)
    result_path = output / "experiment.json"
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project,
                              capture_output=True, text=True, check=True).stdout.strip()
    diff = subprocess.run(["git", "diff", "HEAD"], cwd=project,
                          capture_output=True, check=True).stdout
    (output / "implementation.patch").write_bytes(diff)
    # Include new Python files too; git diff alone excludes untracked work.
    source_hashes = snapshot_sources(project, output / "source-initial.zip")
    result = {"status": "running", "apk": asdict(info), "device_id": args.device_id,
              "device_sdk": sdk, "device_abis": sorted(abis), "network": network,
              "git_revision": revision, "source_hashes": source_hashes,
              "started_at": time.time(), "stages": [],
              "scope": "GUI screening; findings require independent reproduction"}
    write_result(result_path, result)
    attempted_install = False
    old_handlers = {sig: signal.signal(sig, batch._signal_handler)
                    for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        attempted_install = True
        installed = batch.install_apk(args.device_id, info)
        result["install"] = {"returncode": installed.returncode,
                             "stdout": installed.stdout, "stderr": installed.stderr}
        if installed.returncode != 0 or not batch.is_installed(args.device_id, info.package_name):
            raise RuntimeError("Experiment APK installation failed")
        seed = None
        for mode, minutes in (("compliance", args.compliance_minutes), ("defect", args.defect_minutes)):
            stage_output = output / mode
            command = stage_command(info, args.device_id, mode, minutes, stage_output, seed)
            stage = {"mode": mode, "budget_minutes": minutes, "command": command,
                     "output_dir": str(stage_output), "status": "running", "started_at": time.time()}
            stage["source_hashes"] = snapshot_sources(project, output / f"source-{mode}.zip")
            stage["network"] = network_state(args.device_id)
            if getattr(args, "require_network", False) and stage["network"]["status"] != "connected":
                raise RuntimeError("Default device network unavailable before the stage; stopping the experiment")
            result["stages"].append(stage)
            write_result(result_path, result)
            with (output / f"{mode}-console.log").open("w", encoding="utf-8") as log:
                child = subprocess.Popen(command, cwd=project, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=os.name == "posix")
                stage_timeout = minutes * 60 + 300
                try:
                    returncode = child.wait(timeout=stage_timeout)
                except BaseException as exc:
                    batch._terminate_process_tree(child)
                    # Record subprocess timeouts separately from user interruptions.
                    if isinstance(exc, subprocess.TimeoutExpired):
                        stage["status"] = "timeout"
                        stage["timeout_seconds"] = stage_timeout
                    else:
                        stage["status"] = "interrupted"
                    stage["finished_at"] = time.time()
                    try:
                        stage["metrics"] = batch.collect_run_metrics(stage_output)
                    except Exception as exc:
                        stage["metrics_error"] = str(exc)
                    raise
            stage.update(returncode=returncode, finished_at=time.time())
            stage["metrics"] = batch.collect_run_metrics(stage_output)
            complete = returncode == 0 and batch.run_artifacts_complete(stage_output, mode, info.package_name)
            stage["status"] = "completed" if complete else "failed"
            write_result(result_path, result)
            if not complete:
                raise RuntimeError(f"The {mode} stage did not finalize completely; stopping subsequent stages")
            seed = stage_output
        result["status"] = "completed"
    except (Exception, KeyboardInterrupt) as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if attempted_install:
            try:
                result["cleanup"] = batch.cleanup_package(args.device_id, info.package_name)
                if not result["cleanup"].get("verified_absent"):
                    result["status"] = "failed"
            except Exception as exc:
                result.update(status="failed", cleanup_error=str(exc))
        result["finished_at"] = time.time()
        write_result(result_path, result)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apk", required=True)
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--compliance-minutes", type=positive_minutes, default=60)
    parser.add_argument("--defect-minutes", type=positive_minutes, default=60)
    parser.add_argument("--require-network", action="store_true",
                        help="Require a default device network connection; this does not verify backend reachability")
    return run_experiment(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
