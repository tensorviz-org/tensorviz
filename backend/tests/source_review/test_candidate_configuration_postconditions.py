"""Shared definitions need all observed direct arguments, not just one selected call."""
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
from tests.source_review.test_semantic_configuration_edits import intent
from tests.source_review.test_semantic_source_edits import apply

SOURCE = '''import torch
class Block(torch.nn.Module):
    def __init__(this, width=4, /, *sizes, **options):
        super().__init__()
        width = 999
    def forward(self, h): return h
amount = 4
def block(): return Block(amount, amount, width=amount)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.unused = block(), block()
        self.b = Block(amount)
        self.shared = self.a
    def forward(self, h): return self.b(self.a(self.a(h)))
'''


def proposal(base, source, path, value):
    resolver = CapturedEditResolver(base, {'model.py': source})
    cap = next(cap for cap in resolver.configuration_literal_capabilities(base.invocations[1].id)
        if cap.sourceArgumentPath == path)
    edit = intent(cap, value)
    return edit, apply(source, resolver.plan(edit))


def candidate(root, source=SOURCE, path=None, value=None):
    plan, base = prepare(root / 'original', source)
    edit, after = proposal(base, source, path or ['args', 0], value or {'kind': 'int', 'value': '900'})
    with stage_candidate(str(root / 'original'), plan, base, edit, 'model.py', after) as staged:
        process, envelope = replay(staged.workspace, staged.execution.plan, root)
        assert process.returncode == 0, process.stderr
        capture = envelope['result']['diagnostics']['semantic_capture']
    return staged, capture


@pytest.fixture(scope='module')
def observed(tmp_path_factory):
    return candidate(tmp_path_factory.mktemp('configuration-postconditions'))


def test_shared_definition_checks_every_argument_and_distinct_retained_instance(observed):
    staged, capture = observed
    result = check_postconditions(staged.execution, staged.postconditions, capture)
    assert result.status == 'verified' and result.reason == 'observed_requested_configuration'
    assert result.scope == 'observed_configuration_arguments'
    assert result.observedInitializers == 3 and result.observedOperations == 0
    assert len(result.configurationUses) == 4
    assert len({use.bindingId for use in result.configurationUses}) == 2
    assert capture['operations'] == []  # Entry values are independent of forward events.


@pytest.mark.parametrize('path,value', [
    (['args', 1], {'kind': 'float', 'value': '-0'}),
    (['kwargs', 'width'], {'kind': 'int', 'value': '9007199254740993'}),
])
def test_variadic_and_positional_only_keyword_slots(tmp_path, path, value):
    staged, capture = candidate(tmp_path, path=path, value=value)
    assert check_postconditions(staged.execution, staged.postconditions, capture).status == 'verified'


@pytest.mark.parametrize('source', [
    SOURCE.replace('amount = 4\ndef block(): return', 'def block():\n    amount = 4; return').replace('self.b = Block(amount)', 'self.b = block()'),
    SOURCE.replace('amount = 4\ndef block(): return', 'def block():\n    label = "🚀"; amount = 4; return').replace('self.b = Block(amount)', 'self.b = block()'),
    SOURCE.replace('Block(amount, amount, width=amount)', 'Block(\n    amount, amount, width=amount\n)'),
    SOURCE.replace('\n', '\r\n'),
    SOURCE.replace('def block():', 'Alias = Block\ndef block():').replace('return Block(amount', 'return Alias(amount'),
])
def test_module_local_same_line_multiline_unicode_crlf_and_alias_mapping(tmp_path, source):
    staged, capture = candidate(tmp_path, source)
    assert check_postconditions(staged.execution, staged.postconditions, capture).status == 'verified'


def test_wrong_value_in_other_constructor_fails_the_whole_observed_check(observed):
    staged, original = observed
    capture = deepcopy(original)
    module = next(item for item in capture['modules'] if 'b' in item['paths'])
    assert staged.postconditions.expression.id not in module['constructorBindingIds']
    module['initializerArguments']['positional'][0]['value']['value'] = '4'
    result = check_postconditions(staged.execution, staged.postconditions, capture)
    assert result.status == 'failed' and result.reason == 'observed_configuration_mismatch'
    assert result.observedInitializers == 3 and len(result.configurationUses) == 4


@pytest.mark.parametrize('argument', ['amount + 0', '*[amount]', '**{"width": amount}'])
def test_computed_or_expanded_observed_references_do_not_inherit_direct_argument_pass(tmp_path, argument):
    staged, capture = candidate(tmp_path, SOURCE.replace('self.b = Block(amount)', f'self.b = Block({argument})'))
    result = check_postconditions(staged.execution, staged.postconditions, capture)
    assert result.status == 'unavailable' and result.reason == 'configuration_reference_unavailable'
    assert result.configurationUses == []


@pytest.mark.parametrize('change,reason', [
    ('missing', 'initializer_arguments_unavailable'), ('opaque', 'initializer_arguments_unavailable'),
    ('bounded', 'initializer_arguments_unavailable'), ('class', 'ambiguous_observation'),
    ('partial', 'capture_incomplete'), ('competing', 'source_mapping_unavailable'),
    ('unobserved', 'expression_not_observed'), ('mapping', 'source_mapping_unavailable'),
])
def test_missing_or_ambiguous_evidence_never_passes(observed, change, reason):
    staged, original = observed
    capture, proof = deepcopy(original), staged.postconditions
    module = next(item for item in capture['modules'] if 'b' in item['paths'])
    if change == 'missing': module['initializerArguments'] = None
    if change == 'opaque': module['initializerArguments']['positional'][0]['value'] = dict(kind='opaque', reason='unsupported_type', typeName=None)
    if change == 'bounded':
        from services.semantic.initializer_arguments import unavailable
        module['initializerArguments'] = unavailable('byte_limit')
    if change == 'class': module.update(constructorClassVerified=False, initializerArguments=None)
    if change == 'partial': capture['status'] = 'partial'
    if change == 'competing':
        other = deepcopy(next(binding for binding in capture['sourceBindings'] if binding['id'] in module['constructorBindingIds']))
        other['id'] = 'constructor:competing'
        capture['sourceBindings'].append(other)
    if change == 'unobserved':
        for item in capture['modules']:
            if proof.expression.id in item['constructorBindingIds']:
                item.update(constructorClassVerified=False, constructorBindingIds=[], initializerArguments=None)
    if change == 'mapping': proof = replace(proof, definition=None)
    result = check_postconditions(staged.execution, proof, capture)
    assert result.status == 'unavailable' and result.reason == reason
    assert result.observedInitializers == result.observedOperations == 0 and result.configurationUses == []


def test_other_reads_remain_unverified_and_uncalled_instances_are_counted(tmp_path):
    source = SOURCE.replace('def block():', 'unused = amount * 2\ndef never_called(): return Block(amount + 1)\ndef block():')
    staged, capture = candidate(tmp_path, source)
    assert check_postconditions(staged.execution, staged.postconditions, capture).observedInitializers == 3


def test_real_dependent_linear_layers_receive_the_new_dimension(tmp_path):
    from tests.source_review.test_semantic_configuration_edits import CONFIG_SOURCE
    source = CONFIG_SOURCE.replace('nn.Linear(width, width,', 'nn.Linear(4, width,')
    staged, capture = candidate(tmp_path, source, ['args', 1], {'kind': 'int', 'value': '8'})
    result = check_postconditions(staged.execution, staged.postconditions, capture)
    assert result.status == 'verified' and result.observedInitializers == 2
    assert len(result.configurationUses) == 2
    first = capture['invocations'][1]
    assert next(value for value in capture['values'] if value['id'] == first['output']['id'])['shape'] == [2, 8]


def test_skipped_selected_construction_cannot_pass_from_other_configuration_uses(tmp_path):
    source = 'from pathlib import Path\n' + SOURCE.replace('def block(): return',
        'def block():\n    if Path(__file__).parent.name == "workspace": return Block(900)\n    return')
    staged, capture = candidate(tmp_path, source)
    result = check_postconditions(staged.execution, staged.postconditions, capture)
    assert result.status == 'unavailable' and result.reason == 'expression_not_observed'


@pytest.mark.parametrize('change', ['scope', 'operations', 'count', 'reason', 'recipe', 'empty', 'duplicate', 'slot', 'selected', 'huge'])
def test_configuration_verdict_requires_bounded_consistent_argument_witnesses(observed, change):
    staged, capture = observed
    value = check_postconditions(staged.execution, staged.postconditions, capture).model_dump(mode='json')
    if change == 'scope': value['scope'] = 'observed_initializer_entries'
    if change == 'operations': value['observedOperations'] = 1
    if change == 'count': value['observedInitializers'] = 0
    if change == 'reason': value['reason'] = 'observed_requested_constructor'
    if change == 'recipe': value['recipe'] = 'constructor_scalar_v1'
    if change == 'empty': value['configurationUses'] = []
    if change == 'duplicate': value['configurationUses'].append(value['configurationUses'][0])
    if change == 'slot': value['configurationUses'][0]['sourceArgumentPath'] = ['args', -1]
    if change == 'selected': value['bindingId'] = 'other'
    if change == 'huge': value['configurationUses'] *= 100
    with pytest.raises(ValidationError): CandidatePostconditions.model_validate(value)


def test_private_worker_publishes_configuration_witnesses(tmp_path):
    plan, base = prepare(tmp_path / 'original', SOURCE)
    edit, after = proposal(base, SOURCE, ['kwargs', 'width'], {'kind': 'int', 'value': '11'})
    process, envelope = private_replay((tmp_path / 'original').resolve(), payload(plan, base, edit, after), tmp_path)
    assert process.returncode == 0, process.stderr
    result = CandidatePostconditions.model_validate(envelope['worker']['candidatePostconditions'])
    assert result.status == 'verified' and result.observedInitializers == 3 and len(result.configurationUses) == 4
