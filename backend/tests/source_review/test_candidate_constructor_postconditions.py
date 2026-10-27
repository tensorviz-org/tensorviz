"""Constructor checks require bound entry values at the exact edited source call."""
from copy import deepcopy
from dataclasses import replace

import pytest
from pydantic import ValidationError

from services.semantic.candidate import stage_candidate
from services.semantic.candidate_postconditions import check_postconditions
from services.semantic.contracts import CandidatePostconditions
from services.semantic.source_edits import CapturedEditResolver
from tests.source_review.test_semantic_candidate import prepare, replay
from tests.source_review.test_semantic_candidate_worker import payload, replay as private_replay
from tests.source_review.test_semantic_source_edits import apply
from tests.source_review.test_semantic_constructor_edits import intent

SOURCE = '''import torch
class Block(torch.nn.Module):
    def __init__(this, width=4, /, *sizes, bias=2, **options):
        super().__init__()
        width = 999
    def forward(self, h):
        return h + 1
def block():
    return Block(4, 6, width=3, bias=2, scale=0.25)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b, self.unused = block(), block(), block()
    def forward(self, h):
        return self.b(self.a(h)) + self.a(h)
'''


def proposal(base, source, path, value):
    resolver = CapturedEditResolver(base, {'model.py': source})
    cap = next(cap for cap in resolver.constructor_literal_capabilities(base.invocations[1].id)
        if cap.sourceArgumentPath == path)
    edit = intent(cap, value)
    return edit, apply(source, resolver.plan(edit))


def candidate(root, source=SOURCE, path=None, value=None):
    plan, base = prepare(root / 'original', source)
    edit, after = proposal(base, source, path or ['args', 0], value or {'kind': 'int', 'value': '9'})
    with stage_candidate(str(root / 'original'), plan, base, edit, 'model.py', after) as staged:
        process, envelope = replay(staged.workspace, staged.execution.plan, root)
        assert process.returncode == 0, process.stderr
        capture = envelope['result']['diagnostics']['semantic_capture']
    return staged, capture


@pytest.fixture(scope='module')
def observed(tmp_path_factory):
    return candidate(tmp_path_factory.mktemp('constructor-postconditions'))


def test_counts_every_retained_initializer_including_uncalled_and_shared_instances(observed):
    staged, capture = observed
    result = check_postconditions(staged.execution, staged.postconditions, capture)
    assert result.status == 'verified' and result.reason == 'observed_requested_constructor'
    assert result.scope == 'observed_initializer_entries' and result.observedInitializers == 3
    assert result.observedOperations == 0
    # The body overwrites width. Neither current attributes, call count nor the
    # forward's output can supply or replace this initializer-entry observation.
    assert result.bindingId == staged.postconditions.expression.id


@pytest.mark.parametrize('path,value', [
    (['args', 1], {'kind': 'int', 'value': '8'}),
    (['kwargs', 'width'], {'kind': 'int', 'value': '11'}),
    (['kwargs', 'bias'], {'kind': 'float', 'value': '-0'}),
    (['kwargs', 'scale'], {'kind': 'int', 'value': '9007199254740993'}),
])
def test_positional_only_variadic_and_keyword_slots_use_the_actual_signature(tmp_path, path, value):
    staged, capture = candidate(tmp_path, path=path, value=value)
    assert check_postconditions(staged.execution, staged.postconditions, capture).status == 'verified'


@pytest.mark.parametrize('source', [
    SOURCE.replace('return Block(4, 6, width=3, bias=2, scale=0.25)', 'label = "🚀"; return Block(4, 6, width=3, bias=2, scale=0.25)'),
    SOURCE.replace('Block(4, 6, width=3, bias=2, scale=0.25)', 'Block(\n        4, 6, width=3, bias=2, scale=0.25\n    )'),
    SOURCE.replace('\n', '\r\n'),
    SOURCE.replace('def block():', 'Alias = Block\ndef block():').replace('return Block(4', 'return Alias(4'),
])
def test_exact_source_mapping_preserves_unicode_multiline_crlf_and_class_aliases(tmp_path, source):
    staged, capture = candidate(tmp_path, source)
    assert check_postconditions(staged.execution, staged.postconditions, capture).status == 'verified'


def test_a_single_wrong_initializer_fails_even_when_forward_succeeds(observed):
    staged, original = observed
    capture = deepcopy(original)
    modules = [item for item in capture['modules'] if staged.postconditions.expression.id in item['constructorBindingIds']]
    modules[-1]['initializerArguments']['positional'][0]['value']['value'] = '4'
    result = check_postconditions(staged.execution, staged.postconditions, capture)
    assert result.status == 'failed' and result.reason == 'observed_constructor_mismatch'
    assert result.observedInitializers == 3


@pytest.mark.parametrize('change,reason', [
    ('missing_entry', 'initializer_arguments_unavailable'), ('opaque', 'initializer_arguments_unavailable'),
    ('bounded_entry', 'initializer_arguments_unavailable'),
    ('missing_slot', 'initializer_arguments_unavailable'), ('class', 'ambiguous_observation'),
    ('partial', 'capture_incomplete'), ('competing_binding', 'source_mapping_unavailable'),
    ('unobserved', 'expression_not_observed'), ('mapping', 'source_mapping_unavailable'),
])
def test_incomplete_or_ambiguous_initializer_evidence_never_passes(observed, change, reason):
    staged, original = observed
    capture, proof = deepcopy(original), staged.postconditions
    modules = [item for item in capture['modules'] if proof.expression.id in item['constructorBindingIds']]
    if change == 'missing_entry': modules[-1]['initializerArguments'] = None
    if change == 'bounded_entry':
        from services.semantic.initializer_arguments import unavailable
        modules[-1]['initializerArguments'] = unavailable('byte_limit')
    if change == 'opaque': modules[-1]['initializerArguments']['positional'][0]['value'] = dict(kind='opaque', reason='unsupported_type', typeName=None)
    if change == 'missing_slot': proof = replace(proof, argument=('args', 99))
    if change == 'class':
        modules[-1]['constructorClassVerified'] = False
        modules[-1]['initializerArguments'] = None
    if change == 'partial': capture['status'] = 'partial'
    if change == 'competing_binding':
        other = deepcopy(next(binding for binding in capture['sourceBindings'] if binding['id'] == proof.expression.id))
        other['id'] = 'constructor:competing'
        capture['sourceBindings'].append(other)
        modules[-1]['constructorBindingIds'] = [other['id']]
    if change == 'unobserved':
        for module in modules: module.update(constructorClassVerified=False, constructorBindingIds=[], initializerArguments=None)
    if change == 'mapping': proof = replace(proof, expression=None)
    result = check_postconditions(staged.execution, proof, capture)
    assert result.status == 'unavailable' and result.reason == reason
    assert result.observedInitializers == result.observedOperations == 0


def test_skipped_forward_does_not_erase_retained_initializer_entry_evidence(tmp_path):
    source = 'from pathlib import Path\n' + SOURCE.replace('def forward(self, h):\n        return self.b',
        'def forward(self, h):\n        if Path(__file__).parent.name == "workspace": return h\n        return self.b')
    staged, capture = candidate(tmp_path, source)
    assert len(capture['invocations']) == 1
    assert check_postconditions(staged.execution, staged.postconditions, capture).observedInitializers == 3


def test_candidate_that_never_constructs_at_the_edited_site_is_unverified(tmp_path):
    source = 'from pathlib import Path\n' + SOURCE.replace('def block():',
        'def block():\n    if Path(__file__).parent.name == "workspace": return Block(9)')
    staged, capture = candidate(tmp_path, source)
    result = check_postconditions(staged.execution, staged.postconditions, capture)
    assert result.status == 'unavailable' and result.reason == 'expression_not_observed'


def test_initializer_entry_check_does_not_claim_successful_initialization(tmp_path):
    source = SOURCE.replace('width = 999', 'global saved\n        saved = this\n        raise RuntimeError("initializer stopped")')
    source = source.replace('    return Block(4, 6, width=3, bias=2, scale=0.25)',
        '    try:\n        return Block(4, 6, width=3, bias=2, scale=0.25)\n    except RuntimeError:\n        return saved')
    staged, capture = candidate(tmp_path, source)
    result = check_postconditions(staged.execution, staged.postconditions, capture)
    assert result.status == 'verified' and result.scope == 'observed_initializer_entries'
    assert result.observedInitializers == 3 and result.observedOperations == 0


@pytest.mark.parametrize('change', ['scope', 'operations', 'initializers', 'reason', 'recipe'])
def test_initializer_contract_never_presents_entry_count_as_computational_proof(observed, change):
    staged, capture = observed
    value = check_postconditions(staged.execution, staged.postconditions, capture).model_dump(mode='json')
    if change == 'scope': value['scope'] = 'observed_expression'
    if change == 'operations': value['observedOperations'] = 1
    if change == 'initializers': value['observedInitializers'] = 0
    if change == 'reason': value['reason'] = 'observed_requested_literal'
    if change == 'recipe': value['recipe'] = 'binary_scalar_v1'
    with pytest.raises(ValidationError): CandidatePostconditions.model_validate(value)


def test_private_worker_publishes_constructor_entry_verdict(tmp_path):
    plan, base = prepare(tmp_path / 'original', SOURCE)
    edit, after = proposal(base, SOURCE, ['kwargs', 'width'], {'kind': 'int', 'value': '11'})
    process, envelope = private_replay((tmp_path / 'original').resolve(), payload(plan, base, edit, after), tmp_path)
    assert process.returncode == 0, process.stderr
    result = CandidatePostconditions.model_validate(envelope['worker']['candidatePostconditions'])
    assert result.status == 'verified' and result.observedInitializers == 3
