"""Requested semantics require exact source and capture evidence, not run success."""
from copy import deepcopy
from dataclasses import replace

import pytest
from pydantic import ValidationError

from services.semantic.candidate import stage_candidate
from services.semantic.candidate_postconditions import check_postconditions
from services.semantic.contracts import CandidatePostconditions
from services.semantic.source_edits import CapturedEditResolver
from tests.source_review.test_semantic_candidate import prepare, literal_proposal, replay
from tests.source_review.test_semantic_source_edits import SOURCE, apply, intent


@pytest.fixture(scope='module')
def candidate(tmp_path_factory):
    root = tmp_path_factory.mktemp('postconditions')
    plan, base = prepare(root / 'original')
    edit, after = literal_proposal(base)
    with stage_candidate(str(root / 'original'), plan, base, edit, 'model.py', after) as staged:
        process, envelope = replay(staged.workspace, staged.execution.plan, root)
        assert process.returncode == 0, process.stderr
        capture = envelope['result']['diagnostics']['semantic_capture']
        return staged, capture


def test_literal_evidence_is_scoped_to_the_mapped_expression(candidate):
    staged, capture = candidate
    result = check_postconditions(staged.execution, staged.postconditions, capture)
    assert result.status == 'verified' and result.reason == 'observed_requested_literal'
    assert result.scope == 'observed_expression' and result.observedOperations == 1
    assert result.captureId == capture['id']
    assert result.bindingId == staged.postconditions.expression.id


@pytest.mark.parametrize('source', [
    SOURCE.replace('a = h + 1', "prefix = '🚀'; a = h + 1"),
    SOURCE.replace('a = h + 1', 'a = (h +\n             1)'),
    SOURCE.replace('\n', '\r\n'),
    SOURCE.replace('a = h + 1', 'a = h\n        for _ in range(3):\n            a = a + 1'),
])
def test_mapping_handles_columns_line_endings_and_repeated_occurrences(tmp_path, source):
    plan, base = prepare(tmp_path / 'original', source)
    resolver = CapturedEditResolver(base, {'model.py': source})
    cap = resolver.literal_capability(next(op.id for op in base.operations
        if op.target == 'aten.add.Tensor' and op.arguments.args[1].kind == 'int'))
    edit = intent(cap, {'kind': 'float', 'value': -123.5})
    with stage_candidate(str(tmp_path / 'original'), plan, base, edit, 'model.py', apply(source, resolver.plan(edit))) as staged:
        process, envelope = replay(staged.workspace, staged.execution.plan, tmp_path)
        assert process.returncode == 0, process.stderr
        capture = envelope['result']['diagnostics']['semantic_capture']
        result = check_postconditions(staged.execution, staged.postconditions, capture)
    assert result.status == 'verified'
    assert result.observedOperations == (3 if 'range(3)' in source else 1)
    if 'range(3)' in source:
        # The first matching event cannot hide a later occurrence of the same
        # source expression that still recorded the old operand.
        capture['operations'][1]['arguments']['args'][1] = {'kind': 'int', 'value': '1'}
        assert check_postconditions(staged.execution, staged.postconditions, capture).status == 'failed'


def test_wrong_recorded_literal_fails_even_when_runtime_and_capture_succeed(candidate):
    staged, original = candidate
    capture = deepcopy(original)
    capture['operations'][0]['arguments']['args'][1]['value'] = '1'
    result = check_postconditions(staged.execution, staged.postconditions, capture)
    assert result.status == 'failed' and result.reason == 'observed_literal_mismatch'


@pytest.mark.parametrize('change,reason', [
    ('partial', 'capture_incomplete'), ('missing', 'capture_unavailable'),
    ('identity', 'capture_identity_mismatch'), ('plan', 'capture_identity_mismatch'),
    ('binding', 'source_mapping_unavailable'), ('unobserved', 'expression_not_observed'),
    ('duplicate_binding', 'source_mapping_unavailable'),
    ('derived', 'ambiguous_observation'), ('mapping', 'source_mapping_unavailable'),
    ('unsupported', 'unsupported_recipe'),
])
def test_unproven_results_never_pass(candidate, change, reason):
    staged, original = candidate
    capture = deepcopy(original)
    proof = staged.postconditions
    if change == 'partial': capture['status'] = 'partial'
    if change == 'missing': capture = None
    if change == 'identity': capture['attemptId'] = 'different-run'
    if change == 'plan': capture['planDigest'] = '0' * 64
    if change == 'binding':
        next(b for b in capture['sourceBindings'] if b['id'] == proof.expression.id)['anchors'][0]['tokenDigest'] = '0' * 64
    if change == 'duplicate_binding':
        duplicate = deepcopy(next(b for b in capture['sourceBindings'] if b['id'] == proof.expression.id))
        duplicate['id'] = 'binding:ambiguous-alternative'
        capture['sourceBindings'].append(duplicate)
        capture['operations'][-1]['sourceBindingIds'] = [duplicate['id']]
    if change == 'unobserved': capture['operations'][0]['sourceBindingIds'] = []
    if change == 'derived': capture['operations'][0]['target'] = 'aten.mul.Tensor'
    if change == 'mapping': proof = replace(proof, expression=None)
    if change == 'unsupported': proof = None
    result = check_postconditions(staged.execution, proof, capture)
    assert result.status == 'unavailable' and result.reason == reason
    assert result.bindingId is None and result.observedOperations == 0


def test_operation_ids_and_other_identical_literals_cannot_supply_correspondence(candidate):
    staged, original = candidate
    capture = deepcopy(original)
    # An observed multiplication elsewhere has the requested literal too, but
    # neither its occurrence nor reusing the base ID proves the selected expression.
    capture['operations'][0]['sourceBindingIds'] = []
    assert capture['operations'][-1]['arguments']['args'][1] == {'kind': 'int', 'value': '2'}
    assert check_postconditions(staged.execution, staged.postconditions, capture).status == 'unavailable'


def test_renamed_operation_ids_do_not_break_source_correspondence(candidate):
    staged, capture = candidate
    identifiers = {op['id']: 'renamed:' + op['id'] for op in capture['operations']}
    def rename(value):
        if isinstance(value, dict): return {key: rename(item) for key, item in value.items()}
        if isinstance(value, list): return [rename(item) for item in value]
        return identifiers.get(value, value) if isinstance(value, str) else value
    assert check_postconditions(staged.execution, staged.postconditions, rename(capture)).status == 'verified'


@pytest.mark.parametrize('change', ['status', 'reason', 'capture', 'binding', 'count', 'recipe', 'extra'])
def test_contract_rejects_unsupported_certification(candidate, change):
    staged, capture = candidate
    value = check_postconditions(staged.execution, staged.postconditions, capture).model_dump(mode='json')
    if change == 'status': value['status'] = 'unavailable'
    if change == 'reason': value['reason'] = 'observed_literal_mismatch'
    if change == 'capture': value['captureId'] = None
    if change == 'binding': value['bindingId'] = None
    if change == 'count': value['observedOperations'] = 0
    if change == 'recipe': value['recipe'] = None
    if change == 'extra': value['privateSource'] = 'never permitted'
    with pytest.raises(ValidationError):
        CandidatePostconditions.model_validate(value)
