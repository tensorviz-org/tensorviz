"""Reconnection proves a source use within each run, never cross-run ValueRef IDs."""
from copy import deepcopy
from dataclasses import replace

import pytest

from services.semantic.candidate import stage_candidate
from services.semantic.candidate_postconditions import check_postconditions
from services.semantic.source_edits import CapturedEditResolver, SourceEditRefused
from tests.source_review.test_semantic_candidate import prepare, replay
from tests.source_review.test_semantic_candidate_worker import payload, replay as private_replay
from tests.source_review.test_semantic_reconnect_edits import SOURCE, CHAIN, SHARED, intent
from tests.source_review.test_semantic_source_edits import apply


def proposal(base, source, port=1, name='h'):
    resolver = CapturedEditResolver(base, {'model.py': source})
    op = next(op for op in base.operations if op.target == 'aten.add.Tensor'
        and op.arguments.args[0] == op.arguments.args[1])
    cap = next(cap for cap in resolver.reconnect_capabilities(op.id) if cap.argumentPath == ['args', port])
    edit = intent(cap, name)
    return edit, apply(source, resolver.plan(edit)), cap


def candidate(root, source=SOURCE, port=1, name='h'):
    plan, base = prepare(root / 'original', source)
    edit, after, cap = proposal(base, source, port, name)
    with stage_candidate(str(root / 'original'), plan, base, edit, 'model.py', after) as staged:
        process, envelope = replay(staged.workspace, staged.execution.plan, root)
        assert process.returncode == 0, process.stderr
        return staged, envelope['result']['diagnostics']['semantic_capture'], cap


@pytest.fixture(scope='module')
def observed(tmp_path_factory):
    return candidate(tmp_path_factory.mktemp('candidate-reconnect'), SHARED)


def test_each_shared_call_uses_its_own_requested_value(observed):
    staged, capture, cap = observed
    result = check_postconditions(staged.execution, staged.postconditions, capture)
    assert result.status == 'verified' and result.reason == 'observed_requested_argument'
    assert result.recipe == cap.recipe and result.scope == 'observed_expression'
    assert result.observedOperations == 2
    assert result.bindingId == staged.postconditions.expression.id
    operations = [op for op in capture['operations'] if result.bindingId in op['sourceBindingIds']]
    assert operations[0]['arguments']['args'][1] != operations[1]['arguments']['args'][1]


@pytest.mark.parametrize('port', [0, 1])
@pytest.mark.parametrize('source', [SOURCE, CHAIN,
    SOURCE.replace('b = a + a', "label = '🚀'; b = a + a").replace('\n', '\r\n'),
    SOURCE.replace('b = a + a', 'b = (a +\n             a)'),
])
def test_exact_operand_uses_latest_lexical_value_with_real_candidate_capture(tmp_path, source, port):
    staged, capture, _ = candidate(tmp_path, source, port)
    result = check_postconditions(staged.execution, staged.postconditions, capture)
    assert result.status == 'verified' and result.observedOperations == 1


def test_keyword_arguments_are_resolved_in_candidate_call_scope(tmp_path):
    source = SHARED.replace('def forward(self, h):\n        a = h + 1',
        'def forward(self, h, *, other):\n        a = h + other', 1)
    source = source.replace('return self.b(self.a(h))', 'return self.a(other=h + 2, h=h)')
    staged, capture, _ = candidate(tmp_path, source, name='other')
    assert check_postconditions(staged.execution, staged.postconditions, capture).status == 'verified'


def test_wrong_selected_operand_fails_and_edit_authorization_remains_strict(observed):
    staged, original, _ = observed
    capture = deepcopy(original)
    ops = [op for op in capture['operations'] if staged.postconditions.expression.id in op['sourceBindingIds']]
    # A correct first occurrence cannot hide the second using its old operand.
    ops[-1]['arguments']['args'][1] = deepcopy(ops[-1]['arguments']['args'][0])
    result = check_postconditions(staged.execution, staged.postconditions, capture)
    assert result.status == 'failed' and result.reason == 'observed_argument_mismatch'
    assert result.observedOperations == 2
    resolver = CapturedEditResolver(capture, {'model.py': staged.postconditions.source})
    with pytest.raises(SourceEditRefused, match='ambiguous_source_flow'):
        resolver.reconnect_capabilities(ops[-1]['id'])


@pytest.mark.parametrize('change,reason', [
    ('partial', 'capture_incomplete'), ('identity', 'capture_identity_mismatch'),
    ('wrong_source', 'ambiguous_observation'), ('other_operand', 'ambiguous_observation'),
    ('target', 'ambiguous_observation'), ('unknown_value', 'ambiguous_observation'),
    ('extra_binding', 'source_mapping_unavailable'), ('missing_event', 'ambiguous_observation'),
    ('mapping', 'source_mapping_unavailable'), ('unobserved', 'expression_not_observed'),
])
def test_incomplete_or_ambiguous_flow_cannot_certify_the_request(observed, change, reason):
    staged, original, _ = observed
    capture = deepcopy(original)
    proof = staged.postconditions
    ops = [op for op in capture['operations'] if proof.expression.id in op['sourceBindingIds']]
    if change == 'partial': capture['status'] = 'partial'
    if change == 'identity': capture['attemptId'] = 'wrong'
    if change == 'wrong_source': proof = replace(proof, source=proof.source + '# different source')
    if change == 'other_operand': ops[-1]['arguments']['args'][0] = deepcopy(ops[-1]['arguments']['args'][1])
    if change == 'target': ops[-1]['target'] = 'aten.mul.Tensor'
    if change == 'unknown_value': ops[-1]['arguments']['args'][1] = {'kind': 'int', 'value': '1'}
    if change == 'extra_binding':
        alternate = proof.expression.model_dump(mode='json'); alternate['id'] = 'alternate'
        capture['sourceBindings'].append(alternate)
        ops[-1]['sourceBindingIds'] = [alternate['id']]
    if change == 'missing_event': ops[-1]['sourceBindingIds'] = []
    if change == 'mapping': proof = replace(proof, expression=None)
    if change == 'unobserved':
        for op in ops: op['sourceBindingIds'] = []
    result = check_postconditions(staged.execution, proof, capture)
    assert result.status == 'unavailable' and result.reason == reason
    assert result.observedOperations == 0


def test_all_runtime_identifiers_can_change_without_breaking_correspondence(observed):
    staged, capture, _ = observed
    identifiers = {item['id']: 'changed:' + item['id'] for kind in
        ['operations', 'invocations', 'modules', 'definitions', 'values'] for item in capture[kind]}
    def rename(value):
        if isinstance(value, dict): return {key: rename(item) for key, item in value.items()}
        if isinstance(value, list): return [rename(item) for item in value]
        return identifiers.get(value, value) if isinstance(value, str) else value
    assert check_postconditions(staged.execution, staged.postconditions, rename(capture)).status == 'verified'


def test_worker_publishes_reconnection_verdict_without_changing_live_source(tmp_path):
    plan, base = prepare(tmp_path / 'original')
    edit, after, _ = proposal(base, SOURCE)
    process, envelope = private_replay((tmp_path / 'original').resolve(), payload(plan, base, edit, after), tmp_path)
    assert process.returncode == 0, process.stderr
    result = envelope['worker']['candidatePostconditions']
    assert result['status'] == 'verified' and result['reason'] == 'observed_requested_argument'
    assert (tmp_path / 'original' / 'model.py').read_text() == SOURCE


def test_successful_run_that_skips_the_edited_block_is_unverified(tmp_path):
    source = 'from pathlib import Path\n' + SHARED.replace('return self.b(self.a(h))',
        'if Path(__file__).parent.name == "workspace":\n            return h\n        return self.b(self.a(h))')
    staged, capture, _ = candidate(tmp_path, source)
    assert capture['status'] == 'complete'
    result = check_postconditions(staged.execution, staged.postconditions, capture)
    assert result.status == 'unavailable' and result.reason == 'expression_not_observed'


def test_shape_failure_keeps_partial_evidence_without_certifying_reconnection(tmp_path):
    source = CHAIN.replace('self.first = nn.Linear(4, 4)', 'self.first = nn.Linear(4, 2)').replace(
        'self.last = nn.Linear(4, 2)', 'self.last = nn.Linear(2, 2)')
    plan, base = prepare(tmp_path / 'original', source)
    edit, after, _ = proposal(base, source, name='x')
    with stage_candidate(str(tmp_path / 'original'), plan, base, edit, 'model.py', after) as staged:
        process, envelope = replay(staged.workspace, staged.execution.plan, tmp_path)
        assert process.returncode == 4, process.stderr
        capture = envelope['result']['diagnostics']['semantic_capture']
        result = check_postconditions(staged.execution, staged.postconditions, capture)
    assert capture['status'] == 'partial'
    assert result.status == 'unavailable' and result.reason == 'capture_incomplete'
