# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
"""Offline regressions for the 2026-09-28 audit's observed contract failures."""
from copy import deepcopy

import numpy as np
import pytest

from model.flow_graph import FlowGraph, FunctionNode, FunctionDependency, Page
from tools.utils import encode_image_to_jpg_base64
from workflow import graph as workflow_graph
from workflow import trace_defect_detection as detector
from workflow import compliance_detection as compliance
from workflow.nodes import ffg_update as updater, scenario_plan as planner


@pytest.fixture
def image():
    return encode_image_to_jpg_base64(np.full((32, 32, 3), 180, dtype=np.uint8))


def trace(image, actions):
    return [dict(action_name=a, trace_phase='test', current_page=Page(representative_screenshot=image)) for a in actions]


def candidate(action=1, after=2):
    return dict(type='logic_error', confidence='high', description='\u4e00\u6b21\u64cd\u4f5c\u4ea7\u751f\u91cd\u590d\u6761\u76ee',
                expected='\u4e00\u4e2a\u6761\u76ee', actual='\u5217\u8868\u51fa\u73b0\u4e24\u4e2a\u5b8c\u5168\u76f8\u540c\u7684\u65b0\u6761\u76ee',
                evidence_category='single_action_resource_mutation',
                evidence=[dict(global_step=action, role='action'), dict(global_step=after, role='after')])


@pytest.mark.parametrize('better', [True, False])
def test_deduplicated_bug_and_trace_are_replaced_together(image, better):
    old = {**candidate(), 'bug_id': 'BUG-001'}
    old_trace = trace(image, ['click', 'finished'])
    new = candidate(2, 3) if better else candidate()
    new_trace = trace(image, ['wait', 'click', 'finished']) if better else old_trace
    bugs = detector._dedupe_bugs([old, new])
    records = detector._update_bug_trace_records({'bug_trace_records': {'BUG-001': old_trace}}, bugs, [new], new_trace)
    assert bugs[0]['evidence'] == (new if better else old)['evidence']
    assert records['BUG-001'] == (new_trace if better else old_trace)
    assert old['evidence'][0]['global_step'] == 1  # old worker snapshots remain immutable
    action = bugs[0]['evidence'][0]['global_step']
    assert records['BUG-001'][action - 1]['action_name'] == 'click'


def cross_graph():
    graph = FlowGraph()
    for n in range(1, 5):
        graph.add_node(FunctionNode(id=f'func_{n}', name=f'\u4e1a\u52a1{n}'))
    graph.add_function_dependency(FunctionDependency('func_3', 'func_1', '\u5171\u4eab\u6570\u636e'))
    graph.record_cross_test('func_3', 'func_1', '\u5171\u4eab\u6570\u636e', '\u4ea7\u751f\u5e76\u8bfb\u56de', 'done')
    graph.record_cross_test('func_1', 'func_4', '\u5176\u4ed6\u6570\u636e', '\u4e0d\u53d7\u5f71\u54cd', 'done')
    return graph


def test_merge_preserves_invalidated_evidence_across_patch_and_reload():
    graph = cross_graph()
    original = deepcopy(graph.cross_test_results[0])
    result = updater.apply_ffg_patch(graph, {'function_merges': [{'source_function_refs': ['func_2', 'func_3'], 'target_function_ref': 'func_2'}]}, [])
    assert result['validation_issues'] == []
    assert not updater.validate_ffg_invariants(graph)
    assert len(graph.cross_test_results) == 1
    assert graph.cross_test_history[0].items() >= original.items()
    assert ('func_2', 'func_1', 'Functional dependency: \u5171\u4eab\u6570\u636e') in planner._pending_cross_pairs(graph)
    snapshot = graph.to_dict()
    restored = FlowGraph.from_dict(snapshot)
    updater.apply_ffg_patch(restored, {}, [])
    assert restored.to_dict() == snapshot


def test_split_archives_evidence_and_invalid_split_is_skipped():
    graph = cross_graph()
    # No dependencies need redistribution in this minimal split fixture.
    graph.function_dependencies = []
    graph._function_dependency_set = set()
    graph.add_page(Page(id='page_3', name='\u6e90\u9875\u9762'), 'func_3')
    graph.add_page(Page(id='page_5', name='\u53e6\u4e00\u9875\u9762'), 'func_3')
    patch = {'functions': [{'function_ref': 'new_func_1', 'name': '\u62c6\u5206\u4e1a\u52a1', 'resource_variables': ['\u5171\u4eab\u6570\u636e']}, {'function_ref': 'new_func_2', 'name': '\u53e6\u4e00\u4e1a\u52a1'}],
             'function_restructures': [{'source_function_ref': 'func_3', 'retire_source': True,
                                       'replacements': [{'function_ref': 'new_func_1', 'page_refs': ['page_3']},
                                                        {'function_ref': 'new_func_2', 'page_refs': ['page_5']}]}]}
    invalid = deepcopy(patch)
    invalid['function_restructures'][0]['replacements'][0]['page_refs'] = ['missing']
    invalid['function_restructures'][0]['replacements'][1]['page_refs'] = ['missing']
    before_owners = graph.page_owners
    # Skip an entirely invalid split atomically; preserve the source functionality and page ownership.
    assert updater.apply_ffg_patch(graph, invalid, [])['validation_issues']
    assert 'func_3' in graph.functions
    assert graph.page_owners['page_3'] == before_owners['page_3']
    assert graph.page_owners['page_5'] == before_owners['page_5']
    assert updater.apply_ffg_patch(graph, patch, [])['validation_issues'] == []
    assert 'func_3' not in graph.functions
    assert len(graph.cross_test_history) == 1
    assert FlowGraph.from_dict(graph.to_dict()).cross_test_history == graph.cross_test_history


def test_legacy_dangling_cross_evidence_is_preserved_for_review():
    graph = cross_graph()
    del graph.functions['func_3']
    assert any('cross-test' in issue for issue in updater.validate_ffg_invariants(graph))
    restored = FlowGraph.from_dict(graph.to_dict())
    assert len(restored.cross_test_results) == 1
    assert restored.cross_test_history[0]['src_function_id'] == 'func_3'
    assert restored.cross_test_history[0]['status'] == 'done'


def attempt_state(graph):
    return {'ffg': graph, 'current_plan': {'scenario_type': 'intra_test', 'target_function_id': 'func_1'},
            'current_test_scenario': '\u70b9\u51fb\u6309\u94ae\u5e76\u9a8c\u8bc1\u7ed3\u679c',
            'execution_result': {'scenario_status': 'failed_goal', 'scenario_error': 'injection failed'}}


def test_repeated_failure_consumes_attempt_budget_and_survives_reload(monkeypatch):
    graph = FlowGraph()
    graph.add_node(FunctionNode(id='func_1', test_status='in_progress'))
    state = attempt_state(graph)
    monkeypatch.setattr(updater, '_merge_page_audit_results', lambda *a: 0)
    monkeypatch.setattr(updater, '_submit_page_audit_batch', lambda *a: None)
    node = updater.create_ffg_update_node(None)
    for _ in range(20):
        state.update(node(state))
    restored = FlowGraph.from_dict(graph.to_dict())
    assert len(restored.get_node('func_1').scenarios) == 1
    assert planner._node_intra_attempt_count(restored.get_node('func_1')) == 20


def test_update_pass_counts_attempt_once_per_round(monkeypatch):
    graph = FlowGraph()
    graph.add_node(FunctionNode(id='func_1'))
    state = attempt_state(graph)
    monkeypatch.setattr(updater, '_merge_page_audit_results', lambda *a: 0)
    monkeypatch.setattr(updater, '_submit_page_audit_batch', lambda *a: None)
    node = updater.create_ffg_update_node(None)
    state.update(node(state))
    assert graph.get_node('func_1').intra_attempt_count == 1


def test_navigation_skips_do_not_consume_intra_attempts():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id='func_1', scenarios=['\u65e7\u573a\u666f']))
    state = attempt_state(graph)
    state['execution_result']['scenario_status'] = 'skipped'
    updater._apply_non_recordable_execution_outcome(graph, state, state['execution_result'])
    assert planner._node_intra_attempt_count(graph.get_node('func_1')) == 1


def test_planner_exception_obeys_iteration_limit(monkeypatch):
    calls = []
    def broken(_):
        calls.append(1)
        raise RuntimeError('offline failure')
    monkeypatch.setattr(workflow_graph, 'STOP_AFTER_CONSECUTIVE_NODE_ERRORS', False)
    wrapped = workflow_graph._safe_node('scenario_plan', workflow_graph._create_scenario_plan_node_with_iteration(broken), workflow_graph._fallback_scenario_plan)
    state = {'iteration_count': 0, 'max_iterations': 1, 'total_time_limit': 0}
    for _ in range(4):
        state.update(wrapped(state))
    assert len(calls) == 1
    assert state['iteration_count'] == 1
    assert state['workflow_complete']


def test_audit_failure_is_not_converted_to_completion():
    graph = FlowGraph()
    graph.add_node(FunctionNode(id='func_1', test_status='completed'))
    response = planner._skip_response(graph, None)
    assert not response['workflow_complete']


@pytest.mark.parametrize('action,accepted', [('click', True), ('wait', False), ('finished', False)])
def test_resource_mutation_requires_real_action(image, monkeypatch, action, accepted):
    monkeypatch.setattr(detector, 'invoke_llm_json_with_retry', lambda *a, **k: {'bugs': [candidate()]})
    result = detector.analyze_execution_for_bugs(None, {'scenario_execution_trace': trace(image, [action, 'finished']),
        'execution_result': {'scenario_status': 'completed'}, 'current_test_scenario': '\u89c2\u5bdf\u5217\u8868',
        'current_plan': {'scenario_type': 'intra_test', 'target_function_id': 'func_1'}})
    assert bool(result['detected_bugs']) is accepted


@pytest.mark.parametrize('bad_image', ['', 'not-image'])
def test_defect_evidence_requires_decodable_images(bad_image):
    assert not detector._bug_has_required_trace_evidence(candidate(), trace(bad_image, ['click', 'finished']))


@pytest.mark.parametrize('condition,verdict', [('healthy', 'compliant'), ('black', 'undetermined'),
    ('health_flag', 'undetermined'), ('invalid', 'undetermined'), ('later_action_failed', 'compliant')])
def test_functionality_requires_usable_observation(image, monkeypatch, condition, verdict):
    steps = trace(image, ['click', 'click', 'finished'])
    if condition == 'black':
        steps[-1]['current_page'] = Page(representative_screenshot=encode_image_to_jpg_base64(np.zeros((32, 32, 3), dtype=np.uint8)))
    elif condition == 'invalid':
        steps[-1]['current_page'] = Page(representative_screenshot='not-image')
    elif condition == 'health_flag':
        steps[-1]['screenshot_health_error'] = '\u622a\u56fe\u4e0d\u53ef\u7528'
    elif condition == 'later_action_failed':
        steps[-1]['action_failed'] = True  # the pre-action observation remains usable
    item = {'check_id': 'P07', 'verdict': 'compliant', 'description': '\u62d2\u7edd\u540e\u529f\u80fd\u6b63\u5e38',
            'evidence': [{'global_step': 1, 'role': 'action'}, {'global_step': 3, 'role': 'after'}],
            'functionality_evidence': {'denial_step': 1, 'functional_action_step': 2,
                                      'result_observation_step': 3, 'observed_result': '\u529f\u80fd\u6210\u529f'}}
    monkeypatch.setattr(compliance, 'invoke_llm_json_with_retry', lambda *a, **k: {'check_results': [item]})
    result = compliance.analyze_execution_for_compliance(None, {'current_plan': {'compliance_check_ids': ['P07']}, 'scenario_execution_trace': steps})
    assert result['check_results'][0]['verdict'] == verdict


def test_problem_match_is_not_labeled_reproduction():
    from scripts.batch_compliance_run import _conclusion_text
    text = _conclusion_text({'official': {'version_relation': 'same'}, 'violations': ['P01'],
                             'per_issue': {'\u95ee\u9898': ['P01']}, 'candidate_issues': ['\u95ee\u9898'], 'extra': []})
    assert 'related official issues detected' in text and 'pending manual review' in text and 'reproduced' not in text


def test_historical_report_rejects_current_rule_library(monkeypatch):
    from scripts import gen_experiment_report as report
    monkeypatch.setattr(report.glob, 'glob', lambda *a: ['offline/app'])
    monkeypatch.setattr(report, 'load_json', lambda *a: {'checks': [{'rule_id': 'P01'}]})
    with pytest.raises(ValueError, match='57 R-rules'):
        report.load_compliance()
