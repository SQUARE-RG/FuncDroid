# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
import json
from types import SimpleNamespace

import pytest

from compliance.checkpoints import ComplianceChecklist
from compliance.rule_library import load_rules
from model.flow_graph import FlowGraph, FunctionNode
from workflow.nodes.scenario_plan import create_scenario_plan_node, _targeted_compliance_stop
from workflow import compliance_worker as worker_module
from main import _build_parser


@pytest.mark.parametrize('verdict', ['violation', 'compliant', 'undetermined'])
def test_targeted_run_ends_with_original_verdict_before_unrelated_exploration(tmp_path, monkeypatch, verdict):
    path = tmp_path / 'rules.json'
    path.write_text(json.dumps({'rules': [r for r in load_rules() if r['rule_id'] == 'P16']}))
    checklist = ComplianceChecklist()
    checklist.mark_executing(['P16'])
    checklist.apply_results([{'check_id': 'P16', 'verdict': verdict, 'description': '\u89c2\u5bdf\u7ed3\u679c', 'evidence': []}])
    monkeypatch.setattr(worker_module, 'compliance_worker', SimpleNamespace(
        pending_count=lambda: 0, snapshot=checklist.to_state,
    ))
    graph = FlowGraph()
    graph.add_node(FunctionNode(id='unrelated', name='\u65e0\u5173\u9605\u8bfb\u529f\u80fd'))
    result = create_scenario_plan_node(None, None)({
        'run_mode': 'compliance', 'compliance_targeted': True,
        'compliance_rules_path': str(path), 'ffg': graph,
    })
    assert result['workflow_complete']
    assert result['current_test_scenario'] == ''
    assert result['compliance_checklist']['findings'][0]['verdict'] == verdict


def test_pending_review_does_not_start_unrelated_work(monkeypatch):
    monkeypatch.setattr(worker_module, 'compliance_worker', SimpleNamespace(
        pending_count=lambda: 1, flush=lambda timeout: False,
    ))
    result = create_scenario_plan_node(None, None)({'run_mode': 'compliance', 'compliance_targeted': True})
    assert not result['workflow_complete']
    assert result['current_plan']['scenario_type'] == 'skip'
    assert result['current_test_scenario'] == ''


def test_normal_run_and_unchecked_target_are_not_stopped(monkeypatch):
    assert _targeted_compliance_stop({'run_mode': 'compliance'}) is None
    monkeypatch.setattr(worker_module, 'compliance_worker', SimpleNamespace(
        pending_count=lambda: 0, snapshot=lambda: {},
    ))
    assert _targeted_compliance_stop({'run_mode': 'compliance', 'compliance_targeted': True}) is None


def test_targeted_flag_rejects_defect_mode():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(['--mode', 'defect', '--compliance-targeted'])
