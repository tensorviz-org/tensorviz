"""Deletion needs the surviving source value flow, not absence of an old event."""
from copy import deepcopy
from dataclasses import replace

import pytest

from services.semantic.candidate import stage_candidate
from services.semantic.candidate_postconditions import check_postconditions
from services.semantic.source_edits import CapturedEditResolver
from tests.source_review.test_semantic_candidate import prepare, replay
from tests.source_review.test_semantic_candidate_worker import payload, replay as private_replay
from tests.source_review.test_semantic_reconnect_edits import CHAIN, SHARED
from tests.source_review.test_semantic_source_edits import apply
from tests.source_review.test_semantic_unary_edits import deletion

SIMPLE = '''import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def forward(self, h):
        return F.gelu(h)
'''
REPEATED = 'from torch.nn import functional as F\n' + SHARED.replace('a = h + 1', 'a = F.gelu(h)')


def proposal(base, source, name='gelu'):
    resolver = CapturedEditResolver(base, {'model.py': source})
    cap = resolver.delete_unary_capability(next(op.id for op in base.operations if op.target == 'aten.' + name + '.default'))
    edit = deletion(cap)
    return edit, apply(source, resolver.plan(edit)), cap


def candidate(root, source=SIMPLE, name='gelu'):
    plan, base = prepare(root / 'original', source)
    edit, after, cap = proposal(base, source, name)
    with stage_candidate(str(root / 'original'), plan, base, edit, 'model.py', after) as staged:
        process, envelope = replay(staged.workspace, staged.execution.plan, root)
        assert process.returncode == 0, process.stderr
        return staged, envelope['result']['diagnostics']['semantic_capture'], cap


@pytest.fixture(scope='module')
def observed(tmp_path_factory):
    return candidate(tmp_path_factory.mktemp('candidate-deletion'), REPEATED)


def test_every_shared_call_proves_the_surviving_bypass_flow(observed):
    staged, capture, cap = observed
    result = check_postconditions(staged.execution, staged.postconditions, capture)
    assert result.status == 'verified' and result.reason == 'observed_requested_deletion'
    assert result.recipe == cap.recipe and result.observedOperations == 2
    assert result.bindingId == staged.postconditions.expression.id
    assert staged.postconditions.expression.kind == 'definition'
    assert staged.postconditions.replacement.kind == 'argument'
    assert not any(op['target'] == 'aten.gelu.default' for op in capture['operations'])


@pytest.mark.parametrize('name', ['relu', 'gelu', 'silu', 'sigmoid', 'tanh'])
def test_returning_the_input_has_positive_call_and_return_evidence_without_operations(tmp_path, name):
    staged, capture, _ = candidate(tmp_path, SIMPLE.replace('gelu', name), name)
    assert capture['status'] == 'complete' and capture['operations'] == []
    assert capture['invocations'][0]['output'] == capture['inputs']['args'][0]
    result = check_postconditions(staged.execution, staged.postconditions, capture)
    assert result.status == 'verified' and result.observedOperations == 1


@pytest.mark.parametrize('source', [CHAIN,
    CHAIN.replace('h = F.gelu(input=h)', 'h = F.gelu(input=h, approximate="tanh")'),
    CHAIN.replace('h = F.gelu(input=h)', "label = '🚀'; h = F.gelu(input=h)").replace('\n', '\r\n'),
    CHAIN.replace('h = F.gelu(input=h)', 'h = h').replace('b = a + a', 'b = a + F.gelu(a)'),
    SIMPLE.replace('from torch.nn import functional as F', 'from torch.nn.functional import gelu as action').replace('F.gelu(h)', 'action(h)'),
])
def test_assignments_nested_uses_options_and_exact_source_columns_are_preserved(tmp_path, source):
    staged, capture, _ = candidate(tmp_path, source)
    assert check_postconditions(staged.execution, staged.postconditions, capture).status == 'verified'


@pytest.mark.parametrize('change', ['operand', 'return', 'event', 'partial', 'binding', 'source', 'replacement'])
def test_missing_old_activation_does_not_prove_a_bypass_with_broken_remaining_flow(observed, change):
    staged, original, _ = observed
    capture = deepcopy(original); proof = staged.postconditions
    assert not any(op['target'] == 'aten.gelu.default' for op in capture['operations'])
    if change == 'operand': capture['operations'][-1]['arguments']['args'][0] = deepcopy(capture['inputs']['args'][0])
    if change == 'return': capture['invocations'][-1]['output'] = deepcopy(capture['inputs']['args'][0])
    if change == 'event': capture['operations'][0]['sourceBindingIds'] = []
    if change == 'partial': capture['status'] = 'partial'
    if change == 'binding':
        alternate = proof.expression.model_dump(mode='json'); alternate['id'] = 'competing-definition'
        capture['sourceBindings'].append(alternate)
    if change == 'source': proof = replace(proof, source=proof.source + '# changed')
    if change == 'replacement': proof = replace(proof, replacement=None)
    result = check_postconditions(staged.execution, proof, capture)
    assert result.status == 'unavailable' and result.observedOperations == 0


def test_skipping_the_deleted_block_never_certifies_its_surviving_flow(tmp_path):
    source = 'from pathlib import Path\n' + REPEATED.replace('return self.b(self.a(h))',
        'if Path(__file__).parent.name == "workspace":\n            return h\n        return self.b(self.a(h))')
    staged, capture, _ = candidate(tmp_path, source)
    assert capture['status'] == 'complete'
    result = check_postconditions(staged.execution, staged.postconditions, capture)
    assert result.status == 'unavailable'


def test_runtime_identifiers_never_prove_deletion_correspondence(observed):
    staged, capture, _ = observed
    identifiers = {item['id']: 'changed:' + item['id'] for kind in
        ['operations', 'invocations', 'modules', 'definitions', 'values'] for item in capture[kind]}
    def rename(value):
        if isinstance(value, dict): return {key: rename(item) for key, item in value.items()}
        if isinstance(value, list): return [rename(item) for item in value]
        return identifiers.get(value, value) if isinstance(value, str) else value
    assert check_postconditions(staged.execution, staged.postconditions, rename(capture)).status == 'verified'


def test_an_ambiguous_shared_definition_cannot_hide_an_observed_call(observed):
    staged, original, _ = observed
    capture = deepcopy(original)
    definition = next(item for item in capture['definitions'] if item['sourceBindingIds'] == [staged.postconditions.expression.id])
    alternate = deepcopy(definition); alternate['id'] = 'ambiguous-definition'
    alternate['sourceBindingIds'].append(capture['operations'][0]['sourceBindingIds'][0])
    capture['definitions'].append(alternate)
    modules = [item for item in capture['modules'] if item['definitionId'] == definition['id']]
    modules[-1]['definitionId'] = alternate['id']
    result = check_postconditions(staged.execution, staged.postconditions, capture)
    assert result.status == 'unavailable' and result.observedOperations == 0


def test_private_worker_publishes_deletion_evidence_without_writing_live_source(tmp_path):
    plan, base = prepare(tmp_path / 'original', SIMPLE)
    edit, after, _ = proposal(base, SIMPLE)
    process, envelope = private_replay((tmp_path / 'original').resolve(), payload(plan, base, edit, after), tmp_path)
    assert process.returncode == 0, process.stderr
    verdict = envelope['worker']['candidatePostconditions']
    assert verdict['status'] == 'verified' and verdict['reason'] == 'observed_requested_deletion'
    assert (tmp_path / 'original' / 'model.py').read_text() == SIMPLE
