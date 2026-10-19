# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
from types import SimpleNamespace
import json

from workflow.nodes.scenario_exec import _parse_action_json, _validate_action_plan
from workflow.compliance_detection import _validate_functionality_evidence
from workflow import llm_usage
import pytest
import numpy as np
from model.flow_graph import Page
from tools.utils import encode_image_to_jpg_base64


def test_prose_does_not_select_nested_object():
    result = _parse_action_json('\u89e3\u91ca {"action_name":"click","x":500,"y":600,"target":{"label":"\u62d2\u7edd"}}')
    assert result['action_name'] == 'click'
    _validate_action_plan(result, image_size=(723, 1600))


@pytest.mark.parametrize('value', ['', '\u70b9\u51fb\u53f3\u4fa7\u62d2\u7edd\u6309\u94ae'])
def test_empty_and_prose_cannot_become_actions(value):
    with pytest.raises(ValueError):
        _parse_action_json(value)


def test_functionality_needs_executed_action_and_later_observation():
    item = {'functionality_evidence': {'denial_step': 1, 'functional_action_step': 2,
            'result_observation_step': 3, 'observed_result': '\u663e\u793a\u7ffb\u8bd1\u7ed3\u679c'}}
    steps = [{'raw_step_index': 1, 'action_name': 'click'},
             {'raw_step_index': 2, 'action_name': 'click'},
             {'raw_step_index': 3, 'action_name': 'finished'}]
    for step in steps:
        step["current_page"] = Page(representative_screenshot=encode_image_to_jpg_base64(np.full((32, 32, 3), 180, dtype=np.uint8)))
    assert _validate_functionality_evidence(item, steps)
    assert not _validate_functionality_evidence(item, steps[:2])
    steps[1]['action_failed'] = True
    assert not _validate_functionality_evidence(item, steps)
    assert not _validate_functionality_evidence({}, steps)


def test_diagnostics_allowlist(tmp_path, monkeypatch):
    """Persist only allowlisted diagnostic fields to prevent credential leakage. Preserve reasoning text explicitly because it explains conclusions from reasoning models."""
    monkeypatch.setattr(llm_usage, '_OUTPUT_DIR', str(tmp_path))
    response = SimpleNamespace(content='', response_metadata={'finish_reason': 'length', 'api_key': 'secret'},
                               additional_kwargs={'reasoning_content': 'private reasoning'})
    llm_usage.record_llm_diagnostic(response, error='empty_content')
    text = (tmp_path / 'llm_responses.jsonl').read_text()
    row = json.loads(text)
    assert row['finish_reason'] == 'length' and row['has_reasoning']
    assert row['reasoning'] == 'private reasoning'
    assert 'secret' not in text


def test_diagnostics_reasoning_is_truncated(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_usage, '_OUTPUT_DIR', str(tmp_path))
    monkeypatch.setattr(llm_usage, 'LLM_DIAGNOSTIC_REASONING_MAX_CHARS', 12)
    response = SimpleNamespace(
        content='{}',
        response_metadata={},
        additional_kwargs={'reasoning_content': 'x' * 100},
    )
    llm_usage.record_llm_diagnostic(response)
    row = json.loads((tmp_path / 'llm_responses.jsonl').read_text())
    assert row['reasoning'] == 'x' * 12 and len(row['reasoning']) == 12



def test_home_visible_is_not_functional_success(monkeypatch):
    from workflow import compliance_detection as detection
    monkeypatch.setattr(detection, '_compact_screenshot', lambda value: value)
    monkeypatch.setattr(detection, 'invoke_llm_json_with_retry', lambda *a, **k: {
        'check_results': [{'check_id': 'R093', 'verdict': 'compliant',
                          'description': '\u62d2\u7edd\u540e\u9996\u9875\u5165\u53e3\u4ecd\u53ef\u89c1，\u57fa\u672c\u529f\u80fd\u53ef\u7528',
                          'evidence': [{'global_step': 1, 'role': 'before'}]}]})
    result = detection.analyze_execution_for_compliance(None, {
        'current_plan': {'compliance_check_ids': ['R093']},
        'scenario_execution_trace': [{'current_page': {'screenshot': 'image'}, 'action_name': 'click'}]})
    assert result['check_results'][0]['verdict'] == 'undetermined'


def test_tap_refine_response_is_persisted_to_diagnostics(tmp_path, monkeypatch):
    """Persist tap-refinement responses and duration in llm_responses.jsonl; usage totals alone cannot reconstruct per-call latency or model output."""
    from workflow.nodes.scenario_exec import _refine_tap_point

    monkeypatch.setattr(llm_usage, '_OUTPUT_DIR', str(tmp_path))
    response = SimpleNamespace(
        content='{"found": true, "x": 50, "y": 60}',
        response_metadata={}, additional_kwargs={})

    class _FakeCaller:
        def invoke(self, messages):
            return response

    class _FakeLLM:
        def bind(self, timeout=None):
            return _FakeCaller()

    screenshot = encode_image_to_jpg_base64(np.full((160, 160, 3), 200, dtype=np.uint8))
    plan = {'action_name': 'click', 'x': 80, 'y': 80, 'thought': '\u70b9\u51fb\u6309\u94ae', 'action_description': '\u70b9\u51fb\u6309\u94ae'}
    assert _refine_tap_point(_FakeLLM(), screenshot, plan) is not None
    rows = [json.loads(line) for line in (tmp_path / 'llm_responses.jsonl').read_text().splitlines()]
    assert rows[-1]['operation'] == '[Tap refinement]'
    assert rows[-1]['elapsed_seconds'] >= 0


def test_tap_refine_failure_is_persisted_to_diagnostics(tmp_path, monkeypatch):
    from workflow.nodes.scenario_exec import _refine_tap_point

    monkeypatch.setattr(llm_usage, '_OUTPUT_DIR', str(tmp_path))

    class _BrokenCaller:
        def invoke(self, messages):
            raise RuntimeError('boom')

    class _FakeLLM:
        def bind(self, timeout=None):
            return _BrokenCaller()

    screenshot = encode_image_to_jpg_base64(np.full((160, 160, 3), 200, dtype=np.uint8))
    plan = {'action_name': 'click', 'x': 80, 'y': 80, 'thought': '\u70b9\u51fb\u6309\u94ae', 'action_description': '\u70b9\u51fb\u6309\u94ae'}
    assert _refine_tap_point(_FakeLLM(), screenshot, plan) is None
    row = json.loads((tmp_path / 'llm_responses.jsonl').read_text().splitlines()[-1])
    assert row['operation'] == '[Tap refinement]' and 'RuntimeError' in row['error']
