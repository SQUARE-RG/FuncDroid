# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
import json
import subprocess
from dataclasses import replace

import pytest

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
        if args[:4] == ("shell", "pm", "list", "packages"):
            stdout = "package:com.example\n" if installed["value"] else ""
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


def test_completed_hashes_requires_success_artifacts_cleanup_and_latest_mode(tmp_path):
    results = tmp_path / "batch-results.jsonl"
    def record(digest, *, clean=True, status="completed", artifacts=True, mode="defect"):
        batch_test_apks.append_jsonl(results, {
            "sha256": digest, "mode": mode,
            "cleanup": {"verified_absent": clean},
            "test": {"status": status, "artifacts_complete": artifacts},
        })
    record("clean")
    record("dirty", clean=False)
    record("failed", status="failed")
    record("missing", artifacts=False)
    record("retry")
    record("retry", status="timed_out")
    record("compliance", mode="compliance")
    assert batch_test_apks.completed_hashes(results) == {"clean"}
    assert batch_test_apks.completed_hashes(results, "compliance") == {"compliance"}


def test_collect_run_metrics_summarizes_speed_usage_coverage_and_stalls(tmp_path):
    output_dir = tmp_path / "app"
    output_dir.mkdir()
    metric_rows = [
        "iteration=1, outcome=skip, scenario_type=skip, scope=func_1, "
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
        "\n".join(f"2026-08-01 [\u8f6e\u6b21\u6307\u6807] {row}" for row in metric_rows),
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
        "\n".join(f"[\u8f6e\u6b21\u6307\u6807] {row}" for row in rows),
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


def test_manifest_only_is_offline_and_modes_cannot_share_output(tmp_path, monkeypatch):
    apk = tmp_path / 'example.apk'
    apk.write_bytes(b'apk')
    info = batch_test_apks.ApkInfo(str(apk), 'abc', 'com.example', '1', '1', 'Example', 21, ())
    monkeypatch.setattr(batch_test_apks, 'find_aapt', lambda _: 'aapt')
    monkeypatch.setattr(batch_test_apks, 'build_manifest', lambda *_: ([info], []))
    monkeypatch.setattr(batch_test_apks, 'device_capabilities', lambda *_: pytest.fail('offline must not access device'))
    monkeypatch.setattr(batch_test_apks, 'run_one', lambda *_a, **_k: pytest.fail('offline must not test'))
    output = tmp_path / 'batch'
    command = ['--apk-root', str(apk), '--output-dir', str(output), '--manifest-only']
    assert batch_test_apks.main([*command, '--mode', 'compliance']) == 0
    original = (output / 'manifest.json').read_text()
    payload = json.loads(original)
    assert payload['mode'] == 'compliance'
    assert payload['device_sdk'] is None
    assert payload['apks'][0]['package_name'] == 'com.example'
    with pytest.raises(SystemExit):
        batch_test_apks.main([*command, '--mode', 'defect'])
    assert (output / 'manifest.json').read_text() == original


@pytest.mark.parametrize('mode', ['defect', 'compliance'])
@pytest.mark.parametrize('artifacts', [False, True])
def test_run_one_requires_real_outputs_and_preserves_previous_attempt(tmp_path, monkeypatch, mode, artifacts):
    info = batch_test_apks.ApkInfo('example.apk', 'abc', 'com.example', '1', '1', 'Example', 21, ())
    output = tmp_path / batch_test_apks.safe_output_name(info)
    output.mkdir()
    (output / 'evidence.txt').write_text('previous evidence')
    captured = []
    monkeypatch.setattr(batch_test_apks, 'cleanup_package', lambda *_: {'verified_absent': True})
    monkeypatch.setattr(batch_test_apks, 'install_apk', lambda *_: subprocess.CompletedProcess([], 0, 'Success', ''))
    monkeypatch.setattr(batch_test_apks, 'is_installed', lambda *_: True)

    class Process:
        def wait(self, timeout=None):
            if artifacts:
                output.mkdir()
                (output / 'run_result.json').write_text(json.dumps({
                    'status': 'success', 'mode': mode, 'package_name': info.package_name,
                }))
                (output / 'artifact_manifest.json').write_text('{"status":"complete"}')
                (output / ('compliance_report.json' if mode == 'compliance' else 'bugs.json')).write_text('{}')
            return 0

    def popen(command, **kwargs):
        captured.extend(command)
        assert not output.exists()
        return Process()

    monkeypatch.setattr(batch_test_apks.subprocess, 'Popen', popen)
    result = batch_test_apks.run_one(info, device_id='device', total_time_minutes=1,
                                   max_iterations=1, batch_output_dir=tmp_path,
                                   project_root=tmp_path, mode=mode)
    assert captured[captured.index('--mode') + 1] == mode
    assert result['test']['status'] == ('completed' if artifacts else 'failed')
    assert result['test']['artifacts_complete'] is artifacts
    saved = list((tmp_path / 'previous-attempts').glob('*/evidence.txt'))
    assert len(saved) == 1 and saved[0].read_text() == 'previous evidence'
    if artifacts:
        other = 'defect' if mode == 'compliance' else 'compliance'
        assert not batch_test_apks.run_artifacts_complete(output, other, info.package_name)
        assert not batch_test_apks.run_artifacts_complete(output, mode, 'com.other')


@pytest.mark.parametrize('value', ['nan', 'inf', '-1'])
def test_batch_rejects_unbounded_or_negative_numeric_options(value):
    with pytest.raises(SystemExit):
        batch_test_apks._build_parser().parse_args(['--apk-root', '.', '--total-time', value])


def test_keep_app_skips_cleanup_and_install_but_requires_preinstalled_app(tmp_path, monkeypatch):
    """--keep-app skips data cleanup, uninstall, and installation; require an already installed package so manually authenticated sessions can be reused."""
    info = batch_test_apks.ApkInfo('example.apk', 'abc', 'com.example', '1', '1', 'Example', 21, ())
    cleanup_calls = []
    install_calls = []
    monkeypatch.setattr(batch_test_apks, 'cleanup_package',
                         lambda *a: cleanup_calls.append(a) or {'verified_absent': True})
    monkeypatch.setattr(batch_test_apks, 'install_apk',
                         lambda *a: install_calls.append(a) or subprocess.CompletedProcess([], 0, 'Success', ''))
    monkeypatch.setattr(batch_test_apks, 'is_installed', lambda *_: True)
    monkeypatch.setattr(batch_test_apks, 'collect_run_metrics', lambda *_: {})

    class Process:
        def wait(self, timeout=None):
            (tmp_path / batch_test_apks.safe_output_name(info)).mkdir(exist_ok=True)
            return 0

    monkeypatch.setattr(batch_test_apks.subprocess, 'Popen', lambda *a, **k: Process())

    result = batch_test_apks.run_one(
        info, device_id='device', total_time_minutes=1, max_iterations=1,
        batch_output_dir=tmp_path, project_root=tmp_path, keep_app=True,
    )

    assert cleanup_calls == []
    assert install_calls == []
    assert result['pre_cleanup']['skipped'] == 'keep_app'
    assert result['install']['skipped'] == 'keep_app'
    assert result['cleanup']['skipped'] == 'keep_app'
    # keep_app must not accidentally trigger the queue's cleanup-verification stop condition.
    assert result['cleanup']['verified_absent'] is True


def test_keep_app_skips_missing_app_instead_of_installing_it(tmp_path, monkeypatch):
    """Skip a missing package explicitly under keep_app. Automatic installation would violate the preauthenticated-session prerequisite."""
    info = batch_test_apks.ApkInfo('example.apk', 'abc', 'com.example', '1', '1', 'Example', 21, ())
    monkeypatch.setattr(batch_test_apks, 'cleanup_package', lambda *_: pytest.fail('\u4e0d\u5e94\u8c03\u7528\u6e05\u7406'))
    monkeypatch.setattr(batch_test_apks, 'install_apk', lambda *_: pytest.fail('\u4e0d\u5e94\u8c03\u7528\u5b89\u88c5'))
    monkeypatch.setattr(batch_test_apks, 'is_installed', lambda *_: False)

    result = batch_test_apks.run_one(
        info, device_id='device', total_time_minutes=1, max_iterations=1,
        batch_output_dir=tmp_path, project_root=tmp_path, keep_app=True,
    )

    assert result['test'] == {
        'status': 'skipped', 'reason': 'keep_app_requires_preinstalled_app',
    }


def test_python_apk_parser_fallback_reads_metadata_and_native_libraries(tmp_path, monkeypatch):
    import pyaxmlparser
    apk = tmp_path / 'example.apk'
    apk.write_bytes(b'apk bytes')

    class FakeApk:
        def __init__(self, path):
            assert path == str(apk)

        def get_package(self):
            return 'com.example'

        def get_androidversion_code(self):
            return '12'

        def get_androidversion_name(self):
            return '1.2'

        def get_app_name(self):
            return 'Example'

        def get_min_sdk_version(self):
            return '23'

        def get_files(self):
            return ['lib/arm64-v8a/liba.so', 'lib/arm64-v8a/libb.so', 'lib/x86_64/liba.so', 'assets/readme']

    monkeypatch.setattr(pyaxmlparser, 'APK', FakeApk)
    info = batch_test_apks.inspect_apk(apk, '')
    assert info.package_name == 'com.example'
    assert info.version_name == '1.2' and info.version_code == '12'
    assert info.min_sdk == 23
    assert info.native_codes == ('arm64-v8a', 'x86_64')
    assert info.sha256 == batch_test_apks._sha256(apk)


def test_batch_stops_after_failed_run_even_when_cleanup_succeeds(tmp_path, monkeypatch):
    info = batch_test_apks.ApkInfo('example.apk', 'abc', 'com.example', '1', '1', 'Example', 21, ())
    monkeypatch.setattr(batch_test_apks, 'find_aapt', lambda _: '')
    monkeypatch.setattr(batch_test_apks, 'build_manifest', lambda *_: ([info, replace(info, sha256='def')], []))
    monkeypatch.setattr(batch_test_apks, 'device_capabilities', lambda *_: (30, {'arm64-v8a'}))
    monkeypatch.setattr(batch_test_apks.signal, 'signal', lambda *_: None)
    for key in ['API_KEY', 'BASE_URL', 'MODEL']:
        monkeypatch.setenv(key, 'test')
    called = []

    def failed_run(item, **kwargs):
        called.append((item.sha256, kwargs['mode']))
        return {'sha256': item.sha256, 'test': {'status': 'failed'}, 'cleanup': {'verified_absent': True}}

    monkeypatch.setattr(batch_test_apks, 'run_one', failed_run)
    status = batch_test_apks.main(['--mode', 'compliance', '--device-id', 'device',
                                  '--apk-root', str(tmp_path), '--output-dir', str(tmp_path / 'output')])
    assert status == 1
    assert called == [('abc', 'compliance')]


def test_continue_on_error_records_failure_and_runs_remaining_apps(tmp_path, monkeypatch):
    info = batch_test_apks.ApkInfo('example.apk', 'abc', 'com.example', '1', '1', 'Example', 21, ())
    monkeypatch.setattr(batch_test_apks, 'find_aapt', lambda _: '')
    monkeypatch.setattr(batch_test_apks, 'build_manifest', lambda *_: ([info, replace(info, sha256='def')], []))
    monkeypatch.setattr(batch_test_apks, 'device_capabilities', lambda *_: (30, {'arm64-v8a'}))
    monkeypatch.setattr(batch_test_apks.signal, 'signal', lambda *_: None)
    for key in ['API_KEY', 'BASE_URL', 'MODEL']:
        monkeypatch.setenv(key, 'test')
    called = []

    def first_fails(item, **kwargs):
        called.append(item.sha256)
        if item.sha256 == 'abc':
            return {'sha256': item.sha256, 'test': {'status': 'failed'}, 'cleanup': {'verified_absent': True}}
        return {'sha256': item.sha256, 'test': {'status': 'completed'}, 'cleanup': {'verified_absent': True}}

    monkeypatch.setattr(batch_test_apks, 'run_one', first_fails)
    status = batch_test_apks.main(['--mode', 'compliance', '--device-id', 'device',
                                  '--apk-root', str(tmp_path), '--output-dir', str(tmp_path / 'output'),
                                  '--continue-on-error'])
    assert status == 1
    assert called == ['abc', 'def']


def test_continue_on_error_still_stops_on_cleanup_failure(tmp_path, monkeypatch):
    info = batch_test_apks.ApkInfo('example.apk', 'abc', 'com.example', '1', '1', 'Example', 21, ())
    monkeypatch.setattr(batch_test_apks, 'find_aapt', lambda _: '')
    monkeypatch.setattr(batch_test_apks, 'build_manifest', lambda *_: ([info, replace(info, sha256='def')], []))
    monkeypatch.setattr(batch_test_apks, 'device_capabilities', lambda *_: (30, {'arm64-v8a'}))
    monkeypatch.setattr(batch_test_apks.signal, 'signal', lambda *_: None)
    for key in ['API_KEY', 'BASE_URL', 'MODEL']:
        monkeypatch.setenv(key, 'test')
    called = []

    def dirty_cleanup(item, **kwargs):
        called.append(item.sha256)
        return {'sha256': item.sha256, 'test': {'status': 'failed'}, 'cleanup': {'verified_absent': False}}

    monkeypatch.setattr(batch_test_apks, 'run_one', dirty_cleanup)
    status = batch_test_apks.main(['--mode', 'compliance', '--device-id', 'device',
                                  '--apk-root', str(tmp_path), '--output-dir', str(tmp_path / 'output'),
                                  '--continue-on-error'])
    assert status == 2
    assert called == ['abc']


def test_missing_model_configuration_rejected_before_device_access(tmp_path, monkeypatch):
    monkeypatch.setattr(batch_test_apks, 'find_aapt', lambda _: '')
    monkeypatch.setattr(batch_test_apks, 'build_manifest', lambda *_: ([], []))
    for key in ('API_KEY', 'BASE_URL', 'MODEL', 'DEEPSEEK_API_KEY', 'DEEPSEEK_BASE_URL', 'DEEPSEEK_MODEL'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(batch_test_apks, 'device_capabilities', lambda *_: pytest.fail('must validate config first'))
    with pytest.raises(SystemExit):
        batch_test_apks.main(['--device-id', 'device', '--apk-root', str(tmp_path),
                             '--output-dir', str(tmp_path / 'output')])


def test_deepseek_fallback_configuration_satisfies_preflight_check(tmp_path, monkeypatch):
    """DEEPSEEK fallback credentials alone must satisfy batch preflight, matching create_llm configuration precedence."""
    monkeypatch.setattr(batch_test_apks, 'find_aapt', lambda _: '')
    monkeypatch.setattr(batch_test_apks, 'build_manifest', lambda *_: ([], []))
    for key in ('API_KEY', 'BASE_URL', 'MODEL'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'key')
    monkeypatch.setenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
    monkeypatch.setenv('DEEPSEEK_MODEL', 'deepseek-flash')
    monkeypatch.setattr(batch_test_apks, 'device_capabilities', lambda *_: (30, {'arm64-v8a'}))
    status = batch_test_apks.main(['--device-id', 'device', '--apk-root', str(tmp_path),
                                  '--output-dir', str(tmp_path / 'output')])
    # An empty manifest returns nonzero, but configured fallback credentials must pass preflight.
    assert status == 1


def test_metric_failure_cannot_prevent_package_cleanup(tmp_path, monkeypatch):
    info = batch_test_apks.ApkInfo('example.apk', 'abc', 'com.example', '1', '1', 'Example', 21, ())
    cleanup_calls = []

    def cleanup(*args):
        cleanup_calls.append(args)
        return {'verified_absent': True}

    def broken_metrics(_output):
        raise OSError('metrics disk error')

    monkeypatch.setattr(batch_test_apks, 'cleanup_package', cleanup)
    monkeypatch.setattr(batch_test_apks, 'install_apk', lambda *_: subprocess.CompletedProcess([], 1, '', 'install error'))
    monkeypatch.setattr(batch_test_apks, 'collect_run_metrics', broken_metrics)
    result = batch_test_apks.run_one(info, device_id='device', total_time_minutes=1,
                                   max_iterations=1, batch_output_dir=tmp_path,
                                   project_root=tmp_path)
    assert result['test']['reason'] == 'install_failed'
    assert result['metrics_error'] == 'metrics disk error'
    assert len(cleanup_calls) == 2 and result['cleanup']['verified_absent']
