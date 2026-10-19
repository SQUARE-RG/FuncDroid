# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
"""Regression tests for main, graph, llm, llm_usage, seed_graph, and run_logging."""

import base64
import json
import signal
import time

import pytest

import main
from model.flow_graph import FunctionalFlowGraph, FunctionalityNode, Page
from workflow import graph as workflow_graph
from workflow import llm as llm_module
from workflow import llm_usage
from workflow.run_logging import _RotatingBinarySink, _RotatingTextSink
from workflow.seed_graph import load_seed_graph


@pytest.fixture(autouse=True)
def _no_proxy_env(monkeypatch):
    for name in ("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.delenv(name, raising=False)


# 1. Functional Flow Graph seed.

def _seed_run(root, *, file_refs=True):
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="f1", name="\u7ffb\u8bd1", test_status="completed",
                                scenarios=["\u5df2\u6d4b\u8bd5"], depth_scenario_count=4,
                                successful_depth_scenario_count=4))
    graph.add_page(Page(id="p1", controls=["\u7ffb\u8bd1"],
                        representative_screenshot=base64.b64encode(b"image bytes").decode()), "f1")
    if file_refs:
        graph.save(str(root))
    else:
        root.mkdir(parents=True, exist_ok=True)
        (root / "ffg.json").write_text(json.dumps(graph.to_dict()), encoding="utf-8")
    (root / "run_result.json").write_text(json.dumps({"package_name": "com.example", "status": "success"}))
    (root / "artifact_manifest.json").write_text('{"status":"complete"}')


def test_seed_resets_depth_scenario_count(tmp_path):
    _seed_run(tmp_path)
    graph, _ = load_seed_graph(str(tmp_path), "com.example")
    assert graph.functions["f1"].depth_scenario_count == 0
    assert graph.functions["f1"].successful_depth_scenario_count == 0


def test_seed_with_file_refs_skips_redundant_full_decode(tmp_path, monkeypatch):
    _seed_run(tmp_path)
    import workflow.seed_graph as seed_module

    monkeypatch.setattr(
        seed_module.base64, "b64decode",
        lambda *_a, **_k: pytest.fail("file-ref \u622a\u56fe\u5df2\u7531 FlowGraph.load \u6821\u9a8c，\u4e0d\u5e94\u518d\u5168\u91cf\u89e3\u7801"),
    )
    graph, _ = load_seed_graph(str(tmp_path), "com.example")
    assert graph.pages["p1"].representative_screenshot


def test_seed_with_broken_inline_screenshot_is_rejected(tmp_path):
    _seed_run(tmp_path, file_refs=False)
    data = json.loads((tmp_path / "ffg.json").read_text())
    data["pages"][0]["representative_screenshot"] = "not*base64!"
    (tmp_path / "ffg.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="FFG seed screenshot is missing or corrupt"):
        load_seed_graph(str(tmp_path), "com.example")


# 2. Default LLM timeout.

class _FakeChatOpenAI:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__class__.instances.append(self)


@pytest.mark.parametrize("env_value,explicit,expected", [
    (None, None, 180.0),
    ("", None, 180.0),
    ("0", None, 180.0),
    ("-5", None, 180.0),
    ("abc", None, 180.0),
    ("45", None, 45.0),
    ("45", 300, 300.0),
    (None, 0, 180.0),
])
def test_create_llm_always_sends_finite_timeout(monkeypatch, env_value, explicit, expected):
    _FakeChatOpenAI.instances = []
    monkeypatch.setattr(llm_module, "_ReasoningAwareChatOpenAI", _FakeChatOpenAI)
    monkeypatch.setattr(llm_module, "register_llm", lambda *_a, **_k: None)
    for name, value in {"API_KEY": "k", "BASE_URL": "https://x/v1", "MODEL": "m",
                        "LLM_REASONING_EFFORT": ""}.items():
        monkeypatch.setenv(name, value)
    if env_value is None:
        monkeypatch.delenv("LLM_REQUEST_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("LLM_REQUEST_TIMEOUT", env_value)
    llm_module.create_llm(timeout_seconds=explicit)
    assert _FakeChatOpenAI.instances[0].kwargs["request_timeout"] == expected


# 3. Fallback JSON parsing.

def test_parse_json_fallback_returns_last_top_level_object():
    text = 'prefix {"a": {"evidence": {"x": 1}}} middle {"b": 2, "n": {"c": 3}} tail'
    assert llm_module._parse_json_response(text) == {"b": 2, "n": {"c": 3}}


def test_parse_json_fallback_skips_invalid_braces_and_non_dict_values():
    text = 'note { not json } then {"ok": true} and [1, 2]'
    assert llm_module._parse_json_response(text) == {"ok": True}
    with pytest.raises(json.JSONDecodeError):
        llm_module._parse_json_response("no object { here")


def test_parse_json_fallback_is_linear_for_deep_nesting():
    # Do not reparse every nested object; deep nesting with many objects must remain fast.
    nested = '{"a": ' * 300 + "1" + "}" * 300
    text = "\u8bf4\u660e\u6587\u5b57 " + " ".join([nested] * 30)
    started = time.perf_counter()
    result = llm_module._parse_json_response(text)
    assert time.perf_counter() - started < 1.0
    assert isinstance(result, dict) and "a" in result


# 4. Node exception backoff and hard limit.

def test_skip_plan_and_threshold_helpers(monkeypatch):
    assert workflow_graph._skip_plan() == {
        "scenario_type": "skip", "target_function_id": "",
        "src_function_id": "", "dst_function_id": "",
    }
    monkeypatch.setattr(workflow_graph, "STOP_AFTER_CONSECUTIVE_NODE_ERRORS", False)
    monkeypatch.setattr(workflow_graph, "HARD_STOP_CONSECUTIVE_NODE_ERRORS", 5)
    assert workflow_graph._threshold_reached("ffg_update", 3, 3)[0] is False
    stop, hard, reason = workflow_graph._threshold_reached("ffg_update", 5, 3)
    assert stop and hard and "hard limit" in reason
    monkeypatch.setattr(workflow_graph, "STOP_AFTER_CONSECUTIVE_NODE_ERRORS", True)
    stop, hard, reason = workflow_graph._threshold_reached("ffg_update", 3, 3)
    assert stop and not hard and "threshold" in reason


@pytest.mark.parametrize("fallback,counter", [
    (workflow_graph._fallback_scenario_plan, "scenario_plan_error_count"),
    (workflow_graph._fallback_scenario_exec, "scenario_exec_error_count"),
    (workflow_graph._fallback_ffg_update, "ffg_update_error_count"),
])
def test_fallbacks_back_off_and_hard_stop_even_when_soft_stop_disabled(monkeypatch, fallback, counter):
    sleeps = []
    monkeypatch.setattr(workflow_graph._time, "sleep", sleeps.append)
    monkeypatch.setattr(workflow_graph, "STOP_AFTER_CONSECUTIVE_NODE_ERRORS", False)
    monkeypatch.setattr(workflow_graph, "HARD_STOP_CONSECUTIVE_NODE_ERRORS", 6)
    monkeypatch.setattr(workflow_graph, "NODE_ERROR_BACKOFF_MAX_SECONDS", 10.0)
    state = {"current_plan": {}}
    for _ in range(5):
        state.update(fallback(state, RuntimeError("adb offline")))
        assert state["workflow_complete"] is False
    # Retry the first exception immediately, then back off for 2/4/8 seconds with a cap.
    assert sleeps == [2.0, 4.0, 8.0, 10.0]
    state.update(fallback(state, RuntimeError("adb offline")))
    assert state[counter] == 6
    assert state["workflow_complete"] is True
    assert state["workflow_hard_stop"] is True
    assert "hard limit" in state["workflow_stop_reason"]


def test_time_limit_round_does_not_increment_iteration_count():
    planner = workflow_graph._create_scenario_plan_node_with_iteration(
        lambda _s: pytest.fail("\u65f6\u957f\u5df2\u7528\u5c3d\u65f6\u4e0d\u5e94\u8c03\u7528\u89c4\u5212\u5668")
    )
    result = planner({"iteration_count": 4, "start_time": time.time() - 100, "total_time_limit": 10})
    assert result["workflow_complete"] is True
    assert result["iteration_count"] == 4


# 5. Repeated failed audits must not prevent long-running exploration from converging.

def test_long_run_converges_when_audits_only_resubmit_same_fingerprints(monkeypatch):
    graph = FunctionalFlowGraph()
    graph.add_node(FunctionalityNode(id="f1", test_status="completed"))
    submitted_calls = []

    def fake_submit(_graph, llm=None, *, exclude_fingerprints=None):
        # Simulate a persistently failing page that is resubmitted unless its fingerprint is excluded.
        submitted_calls.append(set(exclude_fingerprints or ()))
        key = ("p1", "fp1")
        if exclude_fingerprints is not None and key in exclude_fingerprints:
            return 0
        if exclude_fingerprints is not None:
            exclude_fingerprints.add(key)
        return 1

    monkeypatch.setattr(workflow_graph, "LONG_RUN_IDLE_SECONDS", 0)
    monkeypatch.setattr(workflow_graph, "_merge_page_audit_results", lambda _g: 0)
    monkeypatch.setattr(workflow_graph, "_submit_page_audit_batch", fake_submit)
    monkeypatch.setattr(workflow_graph.page_audit_worker, "pending_count", lambda: 0)
    planner = workflow_graph._create_scenario_plan_node_with_iteration(
        lambda _s: {"ffg": graph, "workflow_complete": True, "current_plan": {"scenario_type": "done"}}
    )
    state = {"ffg": graph, "iteration_count": 1, "start_time": time.time(),
             "total_time_limit": 3600, "max_iterations": 0}
    first = planner(state)
    assert first["workflow_complete"] is False  # Wait for the first submission of a new fingerprint.
    state.update(first)
    second = planner(state)
    assert second["workflow_complete"] is True  # Repeated failure of the same fingerprint no longer blocks convergence.
    assert second["long_run_audit_fingerprints"] == []


# 6. Throttle metrics for skipped iterations.

def test_skip_round_metrics_are_throttled(monkeypatch):
    printed = []
    monkeypatch.setattr(workflow_graph, "_print_iteration_metrics", lambda _s, outcome: printed.append(outcome))
    monkeypatch.setattr(workflow_graph, "SKIP_ROUND_METRICS_EVERY", 5)
    monkeypatch.setattr(workflow_graph, "_CONSECUTIVE_SKIP_ROUNDS", 0)
    skip_state = {"current_plan": {"scenario_type": "skip"}, "iteration_count": 1}
    for _ in range(10):
        assert workflow_graph._route_after_plan(skip_state) == "continue"
    assert printed == ["skip", "skip", "skip"]  # The first, fifth, and tenth consecutive skipped iterations.
    assert workflow_graph._route_after_plan(
        {"current_plan": {"scenario_type": "intra_test"}, "iteration_count": 2}
    ) == "scenario_exec"
    assert workflow_graph._CONSECUTIVE_SKIP_ROUNDS == 0


# 7. Lightweight LLM usage totals and throttled persistence.

def test_usage_totals_and_throttled_saves_skip_fsync(monkeypatch, tmp_path):
    llm_usage.configure_llm_usage(str(tmp_path))
    llm = object()
    fsyncs = []
    writes = []
    real_save = llm_usage._save_locked

    def tracking_save(*, fsync=True):
        writes.append(fsync)
        return real_save(fsync=fsync)

    monkeypatch.setattr(llm_usage, "_save_locked", tracking_save)
    monkeypatch.setattr(llm_usage.os, "fsync", lambda fd: fsyncs.append(fd))
    monkeypatch.setattr(llm_usage, "_SAVE_EVERY_N", 5)
    monkeypatch.setattr(llm_usage, "_SAVE_MIN_INTERVAL", 3600.0)
    for _ in range(4):
        llm_usage.record_llm_request(llm, operation="op")
    assert writes == []  # Below both the change threshold and time interval: do not write.
    llm_usage.record_llm_request(llm, operation="op")
    assert writes == [False] and fsyncs == []  # The frequent path does not fsync.
    totals = llm_usage.get_llm_usage_totals()
    assert totals["requests"] == 5
    totals["requests"] = 999
    assert llm_usage.get_llm_usage_totals()["requests"] == 5  # Return a copy.
    llm_usage.save_llm_usage_summary(durable=False)
    assert writes == [False]  # No dirty data: do not write.
    llm_usage.save_llm_usage_summary()
    assert writes[-1] is True and fsyncs  # Explicit saves still fsync.


# 8. _run_graph writes run_result only at checkpoints.

def test_run_graph_writes_run_result_only_at_checkpoints(monkeypatch, tmp_path):
    calls = {"run_result": 0, "snapshot": 0, "durable_usage": 0, "throttled_usage": 0}
    monkeypatch.setattr(main, "_save_run_result", lambda *a, **k: calls.__setitem__("run_result", calls["run_result"] + 1))
    monkeypatch.setattr(main, "_save_run_snapshot", lambda *a, **k: calls.__setitem__("snapshot", calls["snapshot"] + 1))
    monkeypatch.setattr(main, "_save_activity_timeline", lambda state, out, timeline=None, **k: timeline or {})

    def fake_usage(*, durable=True):
        key = "durable_usage" if durable else "throttled_usage"
        calls[key] += 1

    monkeypatch.setattr(main, "save_llm_usage_summary", fake_usage)

    class Graph:
        def stream(self, state, config):
            yield {"scenario_plan": {"current_plan": {"scenario_type": "intra_test"}}}
            yield {"scenario_exec": {"execution_result": {}}}
            yield {"ffg_update": {"ffg": FunctionalFlowGraph()}}
            yield {"scenario_plan": {"current_plan": {"scenario_type": "skip"}}}

    main._run_graph(Graph(), {"start_time": time.time()}, main.RunResources(output_dir=str(tmp_path)))
    # One initial write plus ffg_update and a skipped planning iteration.
    assert calls["run_result"] == 3
    assert calls["snapshot"] == 3
    assert calls["durable_usage"] == 3
    assert calls["throttled_usage"] == 2


# 9. Defect detail directory cache.

def _bug_trace(seed: int):
    screenshot = base64.b64encode(bytes([seed]) * 64).decode()
    return [{"action_name": "click", "x": 1, "y": 2,
             "current_page": Page(id="page_1", name="\u8be6\u60c5", representative_screenshot=screenshot)}]


def test_bug_fingerprint_cache_matches_full_fingerprint_and_detects_changes(monkeypatch):
    main._BUG_FINGERPRINT_CACHE.clear()
    bug = {"bug_id": "BUG-901", "confidence": "high", "title": "\u65e0\u54cd\u5e94"}
    trace = _bug_trace(1)
    expected = main._bug_artifact_fingerprint(bug, trace)
    assert main._bug_artifact_fingerprint_cached(bug, trace) == expected
    full_calls = []
    real = main._bug_artifact_fingerprint
    monkeypatch.setattr(main, "_bug_artifact_fingerprint",
                        lambda b, t: full_calls.append(1) or real(b, t))
    assert main._bug_artifact_fingerprint_cached(bug, trace) == expected
    assert full_calls == []  # Unchanged data hits the cache without hashing screenshots again.
    changed_bug = dict(bug, title="\u6539\u4e86")
    assert main._bug_artifact_fingerprint_cached(changed_bug, trace) == real(changed_bug, trace)
    new_trace = _bug_trace(2)
    assert main._bug_artifact_fingerprint_cached(changed_bug, new_trace) == real(changed_bug, new_trace)
    assert len(full_calls) == 2


def test_bug_dir_scan_reads_each_bug_json_once(monkeypatch, tmp_path):
    main._BUG_DIR_SCAN_CACHE.clear()
    main._BUG_FINGERPRINT_CACHE.clear()
    bug = {"bug_id": "BUG-902", "confidence": "high"}
    state = {"bug_trace_records": {"BUG-902": _bug_trace(3)}}
    main._save_bug_folders(state, str(tmp_path), [bug], "[\u6d4b\u8bd5]")
    bug_dir = tmp_path / "bugs" / "BUG-902"
    assert (bug_dir / "trace.json").is_file()

    real_open = open
    opened = []

    def tracking_open(path, *args, **kwargs):
        if str(path).endswith("bug.json"):
            opened.append(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", tracking_open)
    main._save_bug_folders(state, str(tmp_path), [bug], "[\u6d4b\u8bd5]")
    assert opened == []  # Already registered directories do not reread bug.json.
    # New directories created externally are still discovered.
    other = tmp_path / "bugs" / "external"
    other.mkdir()
    (other / "bug.json").write_text(json.dumps({"bug_id": "BUG-903"}))
    assert main._scan_existing_bug_dirs(str(tmp_path / "bugs"))["BUG-903"] == str(other)


# 10. Zero-count Activity timeline.

def test_activity_timeline_with_zero_activities_does_not_redraw_every_node(monkeypatch, tmp_path):
    rendered = []
    monkeypatch.setattr(main, "_render_activity_timeline", lambda timeline, path: rendered.append(1) or path.write_bytes(b"png"))
    state = {"start_time": time.time(), "ffg": FunctionalFlowGraph()}
    timeline = main._save_activity_timeline(state, str(tmp_path), force=True)
    for _ in range(5):
        timeline = main._save_activity_timeline(state, str(tmp_path), timeline)
    assert len(timeline["points"]) == 1
    assert len(rendered) == 1


# 11/12. Finalization: no audit resubmission after interruption, zero timeout, and protected saves.

def test_interrupted_flush_page_audit_never_submits(monkeypatch):
    from workflow.nodes import ffg_update

    merged = []
    monkeypatch.setattr(ffg_update, "_merge_page_audit_results", lambda g: merged.append(1) or 0)
    monkeypatch.setattr(ffg_update, "_submit_page_audit_batch",
                        lambda *a, **k: pytest.fail("\u4e2d\u65ad\u65f6\u4e0d\u5e94\u8865\u4ea4\u65b0\u5ba1\u8ba1"))
    monkeypatch.setattr(main.page_audit_worker, "pending_count", lambda: 2)
    monkeypatch.setattr(main.page_audit_worker, "flush",
                        lambda timeout=None: pytest.fail("\u4e2d\u65ad\u65f6\u4e0d\u5e94\u7b49\u5f85\u5728\u9014\u5ba1\u8ba1"))
    main._flush_page_audit({"ffg": FunctionalFlowGraph()}, interrupted=True)
    assert merged == [1]


def test_flush_page_audit_stops_submitting_at_deadline(monkeypatch):
    from workflow.nodes import ffg_update

    submits = []
    monkeypatch.setattr(ffg_update, "_merge_page_audit_results", lambda g: 0)
    monkeypatch.setattr(ffg_update, "_submit_page_audit_batch", lambda *a, **k: submits.append(1) or 1)
    monkeypatch.setattr(main.page_audit_worker, "pending_count", lambda: 0)
    main._flush_page_audit({"ffg": FunctionalFlowGraph()}, deadline=time.monotonic() - 1)
    assert submits == []


def test_flush_worker_zero_timeout_means_no_wait(monkeypatch):
    seen = []

    class Worker:
        log_prefix = "[\u6d4b\u8bd5]"

        def pending_count(self):
            return 1

        def flush(self, timeout=None):
            seen.append(timeout)
            return False

        def record_unfinished(self, reason):
            return 1

    monkeypatch.setattr(main, "BUG_DETECTION_FLUSH_TIMEOUT_SECONDS", 0.0)
    main._flush_worker(Worker(), "\u4fdd\u5b58")
    assert seen == [0.0]


def _offline(monkeypatch, graph):
    llm = object()
    monkeypatch.setattr(main, "_init_runtime", lambda _: (llm, llm, llm, llm, object()))
    monkeypatch.setattr(main, "_start_target_app", lambda *a, **k: None)
    monkeypatch.setattr(main, "setup_logcat_capture", lambda *a, **k: None)
    monkeypatch.setattr(main, "build_graph", lambda *a, **k: graph)


def _options(*extra):
    return main._build_parser().parse_args(["--device-id", "device-1", *extra])


def test_run_single_returns_exit_code_when_saving_after_exception_fails(monkeypatch, tmp_path):
    class Graph:
        def stream(self, state, config):
            raise RuntimeError("graph boom")
            yield  # pragma: no cover

    _offline(monkeypatch, Graph())
    calls = []

    def broken_save(*a, **k):
        calls.append(k.get("interrupted"))
        raise OSError("disk full")

    monkeypatch.setattr(main, "_save_and_summarize", broken_save)
    code = main._run_single(_options("--package-name", "com.example", "--output-dir", str(tmp_path)))
    assert code == 1
    assert calls == [False]  # Save once through the protected finally path.


def test_interrupt_passes_flag_and_sigterm_is_deferred_during_save(monkeypatch, tmp_path):
    class Graph:
        def stream(self, state, config):
            yield {"scenario_plan": {"iteration_count": 2}}
            raise KeyboardInterrupt

    _offline(monkeypatch, Graph())
    observed = {}
    real_save = main._save_and_summarize

    def save(*args, **kwargs):
        observed["interrupted"] = kwargs.get("interrupted")
        handler = signal.getsignal(signal.SIGTERM)
        observed["handler_is_default"] = handler in (signal.SIG_DFL, None)
        # SIGTERM received while saving must not interrupt the save.
        handler(signal.SIGTERM, None)
        observed["survived"] = True
        return real_save(*args, **kwargs)

    monkeypatch.setattr(main, "_save_and_summarize", save)
    previous = signal.getsignal(signal.SIGTERM)
    code = main._run_single(_options("--package-name", "com.example", "--output-dir", str(tmp_path)))
    assert code == 130
    assert observed == {"interrupted": True, "handler_is_default": False, "survived": True}
    assert signal.getsignal(signal.SIGTERM) == previous
    assert json.loads((tmp_path / "run_result.json").read_text())["exit_code"] == 130


# 13. Reuse hashes in the manifest.

def test_second_manifest_reuses_hashes_except_run_result(monkeypatch, tmp_path):
    (tmp_path / "a.bin").write_bytes(b"a" * 100)
    (tmp_path / "run_result.json").write_text("{}")
    state = {"bug_detection_enabled": False}
    first = main._save_artifact_manifest(state, str(tmp_path), 0)
    (tmp_path / "run_result.json").write_text('{"exit_code": 1}')
    hashed = []
    real_open = type(tmp_path).open

    def tracking_open(self, mode="r", *args, **kwargs):
        if "b" in mode and "r" in mode:
            hashed.append(self.name)
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(type(tmp_path), "open", tracking_open)
    second = main._save_artifact_manifest(state, str(tmp_path), 1, reuse_previous=True)
    assert hashed == ["run_result.json"]
    by_path = {item["path"]: item["sha256"] for item in second["files"]}
    first_by_path = {item["path"]: item["sha256"] for item in first["files"]}
    assert by_path["a.bin"] == first_by_path["a.bin"]
    assert by_path["run_result.json"] != first_by_path["run_result.json"]


# 14. Timestamp in rotated log headers.

def test_rotation_header_timestamp_is_generated_at_rotate_time(monkeypatch, tmp_path):
    import workflow.run_logging as run_logging

    stamps = iter(["2026-01-01T00:00:00", "2026-01-01T05:00:00", "2026-01-01T09:00:00"])

    class FakeDatetime:
        @staticmethod
        def now():
            class _Stamp:
                def isoformat(self_inner):
                    return next(stamps)
            return _Stamp()

    monkeypatch.setattr(run_logging, "datetime", FakeDatetime)
    sink = _RotatingTextSink(str(tmp_path / "run.log"), max_bytes=10, backup_count=3)
    sink.write("a" * 8)
    sink.write("b" * 8)  # First rotation.
    sink.write("c" * 200)  # Second rotation.
    sink.close()
    first_header = (tmp_path / "run.log.1").read_text(encoding="utf-8")
    second_header = (tmp_path / "run.log").read_text(encoding="utf-8")
    assert "2026-01-01T00:00:00" in first_header
    assert "2026-01-01T05:00:00" in second_header

    binary = _RotatingBinarySink(str(tmp_path / "logcat.log"), max_bytes=4, backup_count=2,
                                 continuation_header=b"HDR\n")
    binary.write(b"1234")
    binary.write(b"5678")
    binary.close()
    assert (tmp_path / "logcat.log").read_bytes() == b"HDR\n5678"


# 15. Miscellaneous regressions.

def test_build_initial_state_does_not_reresolve_options(monkeypatch):
    args = _options("--mode", "defect", "--package-name", "com.example")
    monkeypatch.setattr(main, "resolve_run_options", lambda *_: pytest.fail("\u4e0d\u5e94\u91cd\u590d\u89e3\u6790"))
    state = main._build_initial_state(args, 1.0)
    assert state["bug_detection_enabled"] is True


@pytest.mark.parametrize("value,expected", [("1", True), ("on", True), ("0", False), ("off", False)])
def test_exec_json_response_format_helper(monkeypatch, value, expected):
    monkeypatch.setenv("EXEC_JSON_RESPONSE_FORMAT", value)
    assert main._exec_json_response_format_enabled() is expected


def test_llm_timeout_banner_reports_finite_default(monkeypatch):
    monkeypatch.delenv("LLM_REQUEST_TIMEOUT", raising=False)
    assert main._llm_timeout_value() == "180"


def test_hard_stop_marks_run_failed_with_reason(monkeypatch, tmp_path):
    class Graph:
        def stream(self, state, config):
            yield {"ffg_update": {
                "workflow_complete": True,
                "workflow_hard_stop": True,
                "workflow_stop_reason": "ffg_update \u8fde\u7eed\u5f02\u5e38\u8fbe\u5230hard limit 30/30，\u5224\u5b9a\u4e3a\u4e0d\u53ef\u6062\u590d\u6545\u969c",
            }}

    _offline(monkeypatch, Graph())
    code = main._run_single(_options("--package-name", "com.example", "--output-dir", str(tmp_path)))
    assert code == 1
    result = json.loads((tmp_path / "run_result.json").read_text())
    assert result["status"] == "failed"
    assert "hard limit" in result["workflow_stop_reason"]
