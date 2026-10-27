"""Applied-source checks reuse real live captures and never execute during reconciliation."""
from copy import deepcopy
import sys

import pytest
from pydantic import ValidationError

from services.semantic.contracts import CaptureReconciliation, CaptureSnapshot, ResolvedExecutionPlan
from services.semantic.source_reconciliation import reconcile_capture
from services.validation.host_snapshot import snapshot
from tests.source_review.test_semantic_candidate import prepare, replay, literal_proposal, SOURCE
import tests.source_review.test_candidate_activation_postconditions as activation
import tests.source_review.test_candidate_reconnection_postconditions as reconnect
import tests.source_review.test_candidate_insertion_postconditions as insertion
import tests.source_review.test_candidate_deletion_postconditions as deletion
import tests.source_review.test_candidate_constructor_postconditions as constructor
import tests.source_review.test_candidate_configuration_postconditions as configuration


def recapture(root, source, control, run_id='live-after', execution=None):
    (root / 'model.py').write_bytes(source.encode())
    plan = ResolvedExecutionPlan.model_validate(snapshot(dict(root=str(root.resolve()), executable=sys.executable,
        environmentRevision=0, session='test-session', workspaceId='live-workspace', runId=run_id, generation=4,
        modelFile='model.py', className='Model', timeoutMs=30000,
        execution=execution or {'inputs': {'args': [{'shape': [2, 4], 'dtype': 'float32', 'fill': 'ones'}]}}))['plan'])
    process, envelope = replay(root.resolve(), plan, control)
    assert process.returncode in (0, 4), process.stderr
    return envelope['result']['diagnostics']['semantic_capture']


def applied(root, source=SOURCE, propose=literal_proposal):
    workspace = root / 'live'; plan, _ = prepare(workspace, source)
    process, envelope = replay(workspace.resolve(), plan, root)
    assert process.returncode == 0, process.stderr
    base = CaptureSnapshot.model_validate(envelope['result']['diagnostics']['semantic_capture'])
    edit, after = propose(base, source)[:2]
    current = recapture(workspace, after, root)
    return dict(version=1, base=base.model_dump(mode='json'), sources={'model.py': source},
        intent=edit, reviewedFile='model.py', reviewedSource=after, current=current)


@pytest.fixture(scope='module')
def observed(tmp_path_factory):
    return applied(tmp_path_factory.mktemp('applied-source'))


def test_live_capture_proves_requested_effect_with_its_own_source_and_attempt_identity(observed):
    result = reconcile_capture(observed)
    assert result.status == 'checked' and result.reason == 'reviewed_source'
    assert result.captureId == observed['current']['id'] != observed['base']['id']
    assert result.sourceId == observed['current']['source']['id'] != observed['base']['source']['id']
    assert result.postconditions.status == 'verified'
    assert result.postconditions.executionId == 'live-after'
    assert result.postconditions.recipe == 'binary_scalar_v1'
    assert result.inputs == result.state == 'same'


@pytest.mark.parametrize('source,propose,recipe', [
    (activation.FUNCTIONAL, lambda b, s: activation.proposal(b, s, 'functional'), 'functional_activation_v1'),
    (activation.MODULE, lambda b, s: activation.proposal(b, s, 'module'), 'module_activation_v1'),
    (reconnect.SOURCE, reconnect.proposal, 'straight_line_tensor_v1'),
    (insertion.SOURCE, insertion.proposal, 'straight_line_unary_insert_v1'),
    (deletion.SIMPLE, deletion.proposal, 'straight_line_unary_delete_v1'),
    (constructor.SOURCE, lambda b, s: constructor.proposal(b, s, ['args', 0], {'kind': 'int', 'value': '9'}), 'constructor_scalar_v1'),
    (configuration.SOURCE, lambda b, s: configuration.proposal(b, s, ['args', 0], {'kind': 'int', 'value': '900'}), 'configuration_scalar_v1'),
])
def test_all_supported_source_recipes_reappear_in_real_saved_source_captures(tmp_path, source, propose, recipe):
    payload = applied(tmp_path, source, propose)
    result = reconcile_capture(payload)
    assert result.status == 'checked' and result.postconditions.status == 'verified'
    assert result.postconditions.recipe == recipe
    assert result.postconditions.captureId == result.captureId


@pytest.mark.parametrize('change,reason', [
    ('workspace', 'different_model'), ('session', 'different_model'), ('class', 'different_model'),
    ('model', 'different_model'), ('same_capture', 'capture_not_new'), ('same_attempt', 'capture_not_new'),
    ('environment', 'environment_changed'), ('dependency', 'dependencies_changed'),
])
def test_stale_or_unrelated_evidence_cannot_be_reconciled(observed, change, reason):
    payload = deepcopy(observed); current = payload['current']; source = current['source']
    if change == 'workspace': source['workspaceId'] = 'other-workspace'
    if change == 'session': source['workspaceSessionId'] = 'other-session'
    if change == 'class': source['className'] = 'Other'
    if change == 'model': source['modelId'] = 'other-model'
    if change == 'same_capture': current['id'] = payload['base']['id']
    if change == 'same_attempt': current['attemptId'] = payload['base']['attemptId']
    if change == 'environment': source['environmentDigest'] = 'a' * 64
    if change == 'dependency': source['files'].append(dict(path='helper.py', digest='b' * 64, role='source', documentVersion=None))
    result = reconcile_capture(payload)
    assert result.status == 'unavailable' and result.reason == reason and result.postconditions is None
    assert result.inputs == result.state == 'unavailable'


def test_undo_and_later_source_edits_never_reuse_the_applied_capture_proof(tmp_path):
    payload = applied(tmp_path)
    payload['current'] = recapture(tmp_path / 'live', SOURCE, tmp_path, 'live-undo')
    assert reconcile_capture(payload).reason == 'source_reverted'
    payload['current'] = recapture(tmp_path / 'live', SOURCE.replace('h + 1', 'h + 3'), tmp_path, 'live-other')
    assert reconcile_capture(payload).reason == 'source_changed'
    payload['current'] = recapture(tmp_path / 'live', payload['reviewedSource'], tmp_path, 'live-redo')
    assert reconcile_capture(payload).postconditions.status == 'verified'


def test_observed_edit_check_keeps_incomplete_dependency_inventory_explicit(observed):
    result = reconcile_capture(observed)
    # External imports commonly leave the workspace dependency inventory partial.
    # This cannot become a claim that every possible dependency was compared.
    assert result.status == 'checked' and result.dependencyScope == 'partial'


def test_shape_failure_stays_incomplete_instead_of_becoming_an_applied_edit_pass(tmp_path):
    from tests.source_review.test_semantic_configuration_edits import CONFIG_SOURCE
    payload = applied(tmp_path, CONFIG_SOURCE, lambda b, s: configuration.proposal(b, s, ['args', 0], {'kind': 'int', 'value': '8'}))
    result = reconcile_capture(payload)
    assert result.status == 'checked' and payload['current']['status'] == 'partial'
    assert result.postconditions.status == 'unavailable' and result.postconditions.reason == 'capture_incomplete'
    assert result.state == 'different'


def test_wrong_observed_effect_fails_even_with_exact_source_and_successful_forward(observed):
    payload = deepcopy(observed)
    payload['current']['operations'][0]['arguments']['args'][1]['value'] = '999'
    result = reconcile_capture(payload)
    assert result.status == 'checked' and result.postconditions.status == 'failed'
    assert result.postconditions.reason == 'observed_literal_mismatch'


def test_changed_actual_inputs_are_reported_separately_from_the_observed_edit(tmp_path):
    payload = applied(tmp_path)
    payload['current'] = recapture(tmp_path / 'live', payload['reviewedSource'], tmp_path, 'live-other-input',
        {'inputs': {'args': [{'shape': [3, 4], 'dtype': 'float32', 'fill': 'zeros'}]}})
    result = reconcile_capture(payload)
    assert result.postconditions.status == 'verified' and result.inputs == 'different'


@pytest.mark.parametrize('change', ['source', 'reviewed', 'intent', 'base', 'file', 'extra', 'buffers'])
def test_reconciliation_rederives_the_original_reviewed_edit_instead_of_trusting_a_claim(observed, change):
    payload = deepcopy(observed)
    if change == 'source': payload['sources']['model.py'] += '\n# changed'
    if change == 'reviewed': payload['reviewedSource'] += '\n# unrelated'
    if change == 'intent': payload['intent']['value'] = {'kind': 'int', 'value': '777'}
    if change == 'base': payload['base']['id'] = 'other-base'
    if change == 'file': payload['reviewedFile'] = 'other.py'
    if change == 'extra': payload['approved'] = True
    if change == 'buffers': payload['current']['source']['sourceMode'] = 'buffers'
    with pytest.raises(ValueError): reconcile_capture(payload)


@pytest.mark.parametrize('change', ['status', 'reason', 'capture', 'missing'])
def test_contract_requires_exact_capture_attribution_and_source_check(observed, change):
    result = reconcile_capture(observed).model_dump(mode='json')
    if change == 'status': result['status'] = 'unavailable'
    if change == 'reason': result['reason'] = 'source_changed'
    if change == 'capture': result['postconditions']['captureId'] = 'other'
    if change == 'missing': result['postconditions'] = None
    with pytest.raises(ValidationError): CaptureReconciliation.model_validate(result)
