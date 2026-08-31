import json
import subprocess
from dataclasses import replace

from scripts import batch_test_apks


def test_discover_apks_excludes_sdk_and_venv(tmp_path):
    normal = tmp_path / "apps" / "example.apk"
    sdk = tmp_path / "Library" / "Android" / "sdk" / "system" / "system.apk"
    venv = tmp_path / "venv" / "lib" / "helper.apk"
    for path in (normal, sdk, venv):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"apk")

    discovered = batch_test_apks.discover_apks([str(tmp_path)])

    assert discovered == [normal]


def test_compatibility_rejects_new_sdk_and_wrong_abi():
    base = batch_test_apks.ApkInfo(
        path="example.apk",
        sha256="abc",
        package_name="com.example",
        version_code="1",
        version_name="1.0",
        app_label="Example",
        min_sdk=29,
        native_codes=(),
    )
    wrong_abi = replace(base, min_sdk=21, native_codes=("x86",))

    assert batch_test_apks.compatibility_reason(base, 28, {"arm64-v8a"}) == "min_sdk_29_gt_device_28"
    assert batch_test_apks.compatibility_reason(wrong_abi, 28, {"arm64-v8a"}) == "native_abi_mismatch"


def test_inspect_apk_accepts_parseable_badging_when_aapt_aborts(tmp_path, monkeypatch):
    apk = tmp_path / "large.apk"
    apk.write_bytes(b"apk")
    stdout = (
        "package: name='com.example.large' versionCode='7' versionName='1.2.3'\n"
        "sdkVersion:'23'\n"
        "application-label:'Large App'\n"
        "native-code: 'arm64-v8a' 'x86_64'\n"
    )
    monkeypatch.setattr(
        batch_test_apks,
        "_run",
        lambda *_args, **_kwargs: batch_test_apks.subprocess.CompletedProcess(
            [], -6, stdout, ""
        ),
    )

    info = batch_test_apks.inspect_apk(apk, "aapt")

    assert info.package_name == "com.example.large"
    assert info.min_sdk == 23
    assert info.native_codes == ("arm64-v8a", "x86_64")


def test_cleanup_clears_data_uninstalls_and_verifies(monkeypatch):
    installed = {"value": True}
    commands = []

    def fake_adb(_device_id, *args, check=False):
        commands.append(args)
        if args[:3] == ("shell", "pm", "path"):
            stdout = "package:/data/app/base.apk\n" if installed["value"] else ""
            return batch_test_apks.subprocess.CompletedProcess(args, 0, stdout, "")
        if args[:3] == ("shell", "pm", "clear"):
            return batch_test_apks.subprocess.CompletedProcess(args, 0, "Success\n", "")
        if args[0] == "uninstall":
            installed["value"] = False
            return batch_test_apks.subprocess.CompletedProcess(args, 0, "Success\n", "")
        return batch_test_apks.subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(batch_test_apks, "_adb", fake_adb)

    result = batch_test_apks.cleanup_package("emulator-5554", "com.example")

    assert result["force_stop"] is True
    assert result["clear_data"] is True
    assert result["uninstall"] is True
    assert result["verified_absent"] is True
    assert ("shell", "am", "force-stop", "com.example") in commands
    assert ("shell", "pm", "clear", "com.example") in commands
    assert ("uninstall", "com.example") in commands


def test_completed_hashes_requires_verified_cleanup(tmp_path):
    results = tmp_path / "batch-results.jsonl"
    results.write_text(
        '{"sha256":"clean","cleanup":{"verified_absent":true}}\n'
        '{"sha256":"dirty","cleanup":{"verified_absent":false}}\n',
        encoding="utf-8",
    )

    assert batch_test_apks.completed_hashes(results) == {"clean"}


def test_collect_run_metrics_summarizes_speed_usage_coverage_and_stalls(tmp_path):
    output_dir = tmp_path / "app"
    output_dir.mkdir()
    metric_rows = [
        "iteration=1, outcome=bootstrap, scenario_type=bootstrap, scope=func_1, "
        "page=page_1, activity=Main, planner_noop_count=0, elapsed_seconds=2.000, "
        "llm_requests=1, llm_tokens=100, functions=1, pages=1, "
        "completed_functions=0, completion_rate=0.000, attempted_scenarios=0, scenario_status=none",
        "iteration=2, outcome=executed, scenario_type=intra_test, scope=func_1, "
        "page=page_1, activity=Main, planner_noop_count=0, elapsed_seconds=4.000, "
        "llm_requests=3, llm_tokens=300, functions=1, pages=1, "
        "completed_functions=0, completion_rate=0.000, attempted_scenarios=1, scenario_status=completed",
        "iteration=3, outcome=executed, scenario_type=intra_test, scope=func_1, "
        "page=page_1, activity=Main, planner_noop_count=0, elapsed_seconds=6.000, "
        "llm_requests=2, llm_tokens=200, functions=2, pages=2, "
        "completed_functions=1, completion_rate=0.500, attempted_scenarios=2, scenario_status=completed",
    ]
    (output_dir / "run.log").write_text(
        "\n".join(f"2026-08-01 [轮次指标] {row}" for row in metric_rows),
        encoding="utf-8",
    )
    (output_dir / "llm_usage.json").write_text(
        json.dumps({"totals": {"requests": 6, "failures": 1, "total_tokens": 600}}),
        encoding="utf-8",
    )
    (output_dir / "activity_timeline.json").write_text(
        json.dumps({
            "activities": ["MainActivity", "DetailActivity"],
            "points": [
                {"elapsed_seconds": 0, "activity_count": 1},
                {"elapsed_seconds": 10, "activity_count": 2},
            ],
        }),
        encoding="utf-8",
    )

    summary = batch_test_apks.collect_run_metrics(output_dir)

    assert summary["iteration_count"] == 3
    assert summary["iteration_elapsed_seconds"]["average"] == 4.0
    assert summary["llm"]["requests"] == 6
    assert summary["coverage"] == {
        "active_functions": 2,
        "active_pages": 2,
        "completed_functions": 1,
        "function_completion_rate": 0.5,
        "attempted_scenarios": 2,
    }
    assert summary["activity_discovery"] == {
        "count": 2,
        "activities": ["MainActivity", "DetailActivity"],
        "point_count": 2,
    }
    assert summary["stall"]["max_same_scope_page_streak"] == 2
    assert summary["stall"]["max_same_scope_page_without_progress_streak"] == 1
    assert summary["stall"]["max_no_coverage_progress_streak"] == 1
    assert summary["stall"]["suspected"] is False
    assert summary["status_counts"] == {"none": 1, "completed": 2}
    assert (output_dir / "iteration-metrics.json").exists()


def test_coverage_plateau_alone_is_not_reported_as_a_stall(tmp_path):
    output_dir = tmp_path / "app"
    output_dir.mkdir()
    rows = [
        (
            f"iteration={index}, outcome=executed, scenario_type=cross_test, "
            f"scope=func_{index}>func_{index + 1}, page=none, activity=Main, "
            "planner_noop_count=0, elapsed_seconds=2.000, llm_requests=1, "
            "llm_tokens=10, functions=3, pages=3, completed_functions=3, "
            "completion_rate=1.000, attempted_scenarios=4, scenario_status=completed"
        )
        for index in range(1, 7)
    ]
    (output_dir / "run.log").write_text(
        "\n".join(f"[轮次指标] {row}" for row in rows),
        encoding="utf-8",
    )

    summary = batch_test_apks.collect_run_metrics(output_dir)

    assert summary["stall"]["max_no_coverage_progress_streak"] == 6
    assert summary["stall"]["max_same_scope_page_without_progress_streak"] == 1
    assert summary["stall"]["suspected"] is False


def test_run_one_terminates_timed_out_process_and_continues_cleanup(tmp_path, monkeypatch):
    info = batch_test_apks.ApkInfo(
        path="example.apk",
        sha256="abc",
        package_name="com.example",
        version_code="1",
        version_name="1.0",
        app_label="Example",
        min_sdk=21,
        native_codes=(),
    )
    cleanup_calls = []
    popen_kwargs = {}
    terminated = []

    def fake_cleanup(device_id, package_name):
        cleanup_calls.append((device_id, package_name))
        return {"verified_absent": True, "errors": []}

    class FakeProcess:
        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(["main.py"], timeout)

    def fake_popen(_command, **kwargs):
        popen_kwargs.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(batch_test_apks, "cleanup_package", fake_cleanup)
    monkeypatch.setattr(
        batch_test_apks,
        "install_apk",
        lambda *_args: subprocess.CompletedProcess([], 0, "Success", ""),
    )
    monkeypatch.setattr(batch_test_apks, "is_installed", lambda *_args: True)
    monkeypatch.setattr(batch_test_apks, "collect_run_metrics", lambda *_args: {})
    monkeypatch.setattr(batch_test_apks.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        batch_test_apks,
        "_terminate_process_tree",
        lambda process: terminated.append(process),
    )

    result = batch_test_apks.run_one(
        info,
        device_id="emulator-5554",
        total_time_minutes=30,
        max_iterations=100,
        batch_output_dir=tmp_path,
        project_root=tmp_path,
        process_timeout_seconds=7,
    )

    assert result["test"]["status"] == "timed_out"
    assert result["test"]["reason"] == "process_hard_timeout"
    assert result["test"]["timeout_seconds"] == 7.0
    assert len(terminated) == 1
    assert len(cleanup_calls) == 2
    assert popen_kwargs["start_new_session"] == (batch_test_apks.os.name == "posix")


def test_process_timeout_uses_time_budget_grace_and_finite_fallback():
    assert batch_test_apks._resolve_process_timeout(2, None) == (
        120 + batch_test_apks.BATCH_PROCESS_TIMEOUT_GRACE
    )
    assert batch_test_apks._resolve_process_timeout(0, None) == (
        batch_test_apks.DEFAULT_BATCH_PROCESS_TIMEOUT
    )
    assert batch_test_apks._resolve_process_timeout(2, 9) == 9.0


def test_batch_adb_commands_always_receive_a_timeout(monkeypatch):
    captured = {}

    def fake_run(command, *, check=False, timeout=None):
        captured.update(command=command, check=check, timeout=timeout)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(batch_test_apks, "_run", fake_run)

    batch_test_apks._adb("emulator-5554", "shell", "getprop")

    assert captured["command"] == [
        "adb",
        "-s",
        "emulator-5554",
        "shell",
        "getprop",
    ]
    assert captured["timeout"] == batch_test_apks.DEFAULT_BATCH_ADB_COMMAND_TIMEOUT
