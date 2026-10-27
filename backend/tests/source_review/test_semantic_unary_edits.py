"""Structural source edits preserve exact operand uses and recapture from source."""
from copy import deepcopy

import pytest
from pydantic import ValidationError

from services.semantic.contracts import InsertUnaryCapability, DeleteUnaryCapability
from services.semantic.source_edits import CapturedEditResolver, SourceEditRefused
from tests.source_review.test_semantic_source_edits import SOURCE, setup, apply
from tests.source_review.test_semantic_reconnect_edits import CHAIN, SHARED


def add(capture):
    return next(op for op in capture.operations if op.target == 'aten.add.Tensor' and op.arguments.args[0] == op.arguments.args[1])


def insertion(cap, value='gelu'):
    return dict(kind='insert_unary_op', capabilityId=cap.id, target=cap.target.model_dump(mode='json'),
        argumentPath=cap.argumentPath, expected=cap.expected.model_dump(mode='json'), value=value, scope='source_expression')


def deletion(cap):
    return dict(kind='delete_unary_op', capabilityId=cap.id, target=cap.target.model_dump(mode='json'),
        argumentPath=cap.argumentPath, expected=cap.expected.model_dump(mode='json'),
        value=cap.input.value.model_dump(mode='json'), scope='source_expression')


@pytest.mark.parametrize('name', ['relu', 'gelu', 'silu', 'sigmoid', 'tanh'])
@pytest.mark.parametrize('port', [0, 1])
def test_insertion_wraps_only_the_selected_repeated_operand_and_recaptures(name, port, tmp_path):
    capture, resolver = setup(tmp_path, CHAIN)
    caps = resolver.insert_unary_capabilities(add(capture).id)
    assert [cap.argumentPath for cap in caps] == [['args', 0], ['args', 1]]
    cap = caps[port]
    assert cap.expectedName == 'a' and cap.importPlan is None and cap.namespace == 'F'
    assert cap.scope.observedOperations == cap.scope.observedInvocations == cap.scope.observedModuleInstances == 1
    plan = resolver.plan(insertion(cap, name))
    expected = f'b = F.{name}(a) + a' if port == 0 else f'b = a + F.{name}(a)'
    after = apply(CHAIN, plan)
    assert plan['validation'] == 'static_only' and len(plan['edits']) == 1
    assert after == CHAIN.replace('b = a + a', expected)
    changed, _ = setup(tmp_path, after)
    old_targets = [op.target for op in capture.operations]
    new_targets = [op.target for op in changed.operations]
    index = next(i for i, op in enumerate(capture.operations) if op.id == add(capture).id)
    assert new_targets == old_targets[:index] + [f'aten.{name}.default'] + old_targets[index:]
    unary, binary = changed.operations[index:index + 2]
    assert unary.arguments.args == [capture.operations[index].arguments.args[port]]
    assert binary.arguments.args[port] == unary.output
    assert binary.arguments.args[1 - port] == unary.arguments.args[0]
    assert all(not op.effects for op in changed.operations)


@pytest.mark.parametrize('name', ['relu', 'gelu', 'silu', 'sigmoid', 'tanh'])
@pytest.mark.parametrize('spelling', ['F.{name}(h)', 'F.{name}(input=h)'])
def test_deletion_bypasses_only_the_call_and_preserves_assignments_and_later_uses(tmp_path, name, spelling):
    source = CHAIN.replace('F.gelu(input=h)', spelling.format(name=name))
    capture, resolver = setup(tmp_path, source)
    op = next(op for op in capture.operations if op.target == f'aten.{name}.default')
    cap = resolver.delete_unary_capability(op.id)
    assert cap.input.name == 'h' and cap.expected == op.output and cap.input.value == op.arguments.args[0]
    assert cap.argumentPath == ['args', 0]
    plan = resolver.plan(deletion(cap))
    after = apply(source, plan)
    assert plan['validation'] == 'static_only' and len(plan['edits']) == 1
    assert after == source.replace(spelling.format(name=name), 'h')
    changed, _ = setup(tmp_path, after)
    assert [op.target for op in changed.operations] == [other.target for other in capture.operations if other.id != op.id]
    first_add = next(other for other in changed.operations if other.target == 'aten.add.Tensor')
    assert first_add.arguments.args[0] == changed.invocations[1].output
    assert add(changed).arguments.args == [first_add.output, first_add.output]


def test_insert_then_delete_recaptures_the_original_program_without_a_synthetic_graph_edit(tmp_path):
    capture, resolver = setup(tmp_path)
    cap = resolver.insert_unary_capabilities(add(capture).id)[1]
    inserted = apply(SOURCE, resolver.plan(insertion(cap)))
    changed, after_resolver = setup(tmp_path, inserted)
    unary = next(op for op in changed.operations if op.target == 'aten.gelu.default')
    deleted = apply(inserted, after_resolver.plan(deletion(after_resolver.delete_unary_capability(unary.id))))
    assert deleted == SOURCE
    final, _ = setup(tmp_path, deleted)
    assert [op.model_dump(exclude={'sourceBindingIds'}) for op in final.operations] == [op.model_dump(exclude={'sourceBindingIds'}) for op in capture.operations]


def test_insert_and_delete_include_all_shared_forward_instances(tmp_path):
    capture, resolver = setup(tmp_path, SHARED)
    cap = resolver.insert_unary_capabilities(add(capture).id)[1]
    assert cap.scope.observedOperations == cap.scope.observedInvocations == cap.scope.observedModuleInstances == 2
    inserted = apply(SHARED, resolver.plan(insertion(cap, 'relu')))
    changed, after_resolver = setup(tmp_path, inserted)
    activations = [op for op in changed.operations if op.target == 'aten.relu.default']
    assert len(activations) == 2
    cap = after_resolver.delete_unary_capability(activations[0].id)
    assert cap.scope.observedOperations == cap.scope.observedInvocations == cap.scope.observedModuleInstances == 2
    assert apply(inserted, after_resolver.plan(deletion(cap))) == SHARED


def test_insertion_reuses_proven_namespaces_and_adds_only_an_exact_fresh_import(tmp_path):
    source = SOURCE.replace('import torch', 'from torch.nn import Module').replace('torch.nn.Module', 'Module')
    source = '# coding: utf-8\n"""Keep docstring."""\nfrom __future__ import annotations\n' + source
    source += '\n_tensorviz_functional = 3\n'
    source = source.replace('\n', '\r\n')
    capture, resolver = setup(tmp_path, source)
    cap = resolver.insert_unary_capabilities(add(capture).id)[1]
    assert cap.namespace == '_tensorviz_functional_1'
    assert cap.importPlan.model_dump() == dict(module='torch.nn.functional', line=4, lineEnding='\r\n')
    plan = resolver.plan(insertion(cap))
    after = apply(source, plan)
    assert len(plan['edits']) == 2
    assert after == source.replace('from torch.nn import Module', 'import torch.nn.functional as _tensorviz_functional_1\r\nfrom torch.nn import Module').replace('b = a + a', 'b = a + _tensorviz_functional_1.gelu(a)')
    changed, _ = setup(tmp_path, after)
    assert changed.status == 'complete'


def test_deleting_a_direct_imported_activation_never_adds_a_replacement_namespace(tmp_path):
    source = 'from torch.nn import Module\nfrom torch.nn.functional import gelu as smooth\nclass Model(Module):\n    def forward(self, h):\n        a = smooth(h)\n        return a + a\n'
    capture, resolver = setup(tmp_path, source)
    cap = resolver.delete_unary_capability(capture.operations[0].id)
    plan = resolver.plan(deletion(cap))
    assert len(plan['edits']) == 1
    assert apply(source, plan) == source.replace('a = smooth(h)', 'a = h')


@pytest.mark.parametrize('option', ['F.relu(input=h, inplace=False)', 'F.gelu(h, approximate="tanh")'])
def test_deletion_removes_nonmutating_literal_options_with_the_selected_call(tmp_path, option):
    source = CHAIN.replace('F.gelu(input=h)', option)
    capture, resolver = setup(tmp_path, source)
    op = next(op for op in capture.operations if op.target in ('aten.relu.default', 'aten.gelu.default'))
    cap = resolver.delete_unary_capability(op.id)
    after = apply(source, resolver.plan(deletion(cap)))
    assert after == source.replace(option, 'h')
    assert setup(tmp_path, after)[0].status == 'complete'


@pytest.mark.parametrize('expression', ['F.relu(h, inplace=True)', 'F.gelu(h + 1)', 'F.gelu(\n            h\n        )'])
def test_deletion_refuses_mutating_computed_or_multiline_inputs(tmp_path, expression):
    source = CHAIN.replace('F.gelu(input=h)', expression)
    capture, resolver = setup(tmp_path, source)
    op = next(op for op in capture.operations if op.target in ('aten.relu_.default', 'aten.gelu.default'))
    with pytest.raises(SourceEditRefused, match='unsupported_unary_deletion'):
        resolver.delete_unary_capability(op.id)


@pytest.mark.parametrize('transform', [
    lambda s: s.replace('b = a + a', 'if True:\n            b = a + a'),
    lambda s: s.replace('h = F.gelu(input=h)', 'if True:\n            h = F.gelu(input=h)'),
    lambda s: s.replace('return self.last(b)', 'other = F.relu(x, inplace=True)\n        return self.last(b)'),
])
def test_structural_edits_refuse_control_flow_or_mutation_anywhere_in_the_scope(tmp_path, transform):
    capture, resolver = setup(tmp_path, transform(CHAIN))
    with pytest.raises(SourceEditRefused):
        resolver.insert_unary_capabilities(add(capture).id)
    with pytest.raises(SourceEditRefused):
        resolver.delete_unary_capability(next(op.id for op in capture.operations if op.target == 'aten.gelu.default'))


@pytest.mark.parametrize('code', ['def helper():\n    globals()', 'def helper():\n    setattr(torch, "extra", None)',
    'def helper():\n    torch.nn.functional.relu = None', 'def helper(self):\n    self.F.relu = None',
    'def helper():\n    return torch.__dict__'])
def test_new_unary_callees_refuse_source_visible_namespace_mutation_and_reflection(tmp_path, code):
    capture, resolver = setup(tmp_path, SOURCE + '\n' + code + '\n')
    with pytest.raises(SourceEditRefused, match='unproven_unary_namespace'):
        resolver.insert_unary_capabilities(add(capture).id)


@pytest.mark.parametrize('kind', ['insert', 'delete'])
@pytest.mark.parametrize('field', ['port', 'expected', 'capture', 'scope', 'name', 'value'])
def test_forged_structural_intents_never_widen_the_source_recipe(tmp_path, kind, field):
    capture, resolver = setup(tmp_path, CHAIN)
    cap = resolver.insert_unary_capabilities(add(capture).id)[1] if kind == 'insert' else resolver.delete_unary_capability(
        next(op.id for op in capture.operations if op.target == 'aten.gelu.default'))
    request = insertion(cap) if kind == 'insert' else deletion(cap)
    if field == 'port':
        request['argumentPath'] = ['args', 0 if kind == 'insert' else 1]
    elif field == 'expected':
        request['expected'] = {'kind': 'value', 'id': 'wrong'}
    elif field == 'capture':
        request['target']['captureId'] = 'other'
    elif field == 'scope':
        request['scope'] = 'one_call'
    elif field == 'name':
        request['name'] = 'danger()'
    else:
        request['value'] = 'danger()' if kind == 'insert' else capture.inputs.args[0].model_dump(mode='json')
    with pytest.raises((ValidationError, SourceEditRefused)):
        resolver.plan(request)


def test_structural_contracts_refuse_cross_file_ranges_future_definitions_and_imports(tmp_path):
    capture, resolver = setup(tmp_path, CHAIN)
    inserted = resolver.insert_unary_capabilities(add(capture).id)[1].model_dump(mode='json')
    deleted = resolver.delete_unary_capability(next(op.id for op in capture.operations if op.target == 'aten.gelu.default')).model_dump(mode='json')
    for contract, original in [(InsertUnaryCapability, inserted), (DeleteUnaryCapability, deleted)]:
        broken = deepcopy(original)
        broken['binding']['anchors'][0]['file'] = 'other.py'
        with pytest.raises(ValidationError):
            contract.model_validate(broken)
    inserted['importPlan'] = dict(module='torch.nn', line=1, lineEnding='\n')
    with pytest.raises(ValidationError):
        InsertUnaryCapability.model_validate(inserted)
    deleted['input']['binding']['anchors'][0]['endLine'] = 99
    with pytest.raises(ValidationError):
        DeleteUnaryCapability.model_validate(deleted)


def test_unicode_names_and_comments_survive_insert_delete_roundtrip(tmp_path):
    source = SOURCE.replace('a =', '变量 =').replace('a + a', '变量 + 变量').replace('b = 变量 + 变量', 'b = 变量 + 变量  # keep').replace('\n', '\r\n')
    capture, resolver = setup(tmp_path, source)
    cap = resolver.insert_unary_capabilities(add(capture).id)[1]
    inserted = apply(source, resolver.plan(insertion(cap)))
    changed, after_resolver = setup(tmp_path, inserted)
    unary = next(op for op in changed.operations if op.target == 'aten.gelu.default')
    deleted = apply(inserted, after_resolver.plan(deletion(after_resolver.delete_unary_capability(unary.id))))
    assert deleted == source


def test_stale_source_cannot_supply_a_structural_capability(tmp_path):
    capture, _ = setup(tmp_path, CHAIN)
    with pytest.raises(SourceEditRefused, match='stale_source'):
        CapturedEditResolver(capture, {'model.py': CHAIN + '# changed\n'})


def test_publication_keeps_replacement_when_deletion_needs_a_stronger_scope(tmp_path):
    source = CHAIN.replace('h = F.gelu(input=h)', 'if True:\n            h = F.gelu(input=h)')
    capture, resolver = setup(tmp_path, source)
    op = next(op for op in capture.operations if op.target == 'aten.gelu.default')
    assert [cap.command for cap in resolver.capabilities(op.id)] == ['replace_activation']


def test_publication_keeps_reconnection_when_insertion_cannot_prove_a_namespace(tmp_path):
    capture, resolver = setup(tmp_path, SOURCE + '\ndef helper():\n    globals()\n')
    assert [cap.command for cap in resolver.capabilities(add(capture).id)] == ['reconnect_argument', 'reconnect_argument']
