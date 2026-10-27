"""Activation changes preserve authored arguments and every unrelated source byte."""
from copy import deepcopy

import pytest
from pydantic import ValidationError

from services.semantic.contracts import ActivationEditCapability, ReplaceActivationEdit
from services.semantic.source_edits import CapturedEditResolver, SourceEditRefused
from tests.source_review.test_semantic_source_edits import setup, apply


SOURCE = '''import torch
import torch.nn.functional as F
class Model(torch.nn.Module):
    def forward(self, h):
        a = F.relu(  # preserve call formatting
            h + 1,
        )
        return a + a
'''
NAMES = ['relu', 'gelu', 'silu', 'sigmoid', 'tanh']


def capability(record, resolver, name='relu'):
    return resolver.activation_capability(next(op.id for op in record.operations if op.target == f'aten.{name}.default'))


def intent(cap, value='gelu'):
    return dict(kind='replace_activation', capabilityId=cap.id, target=cap.target.model_dump(mode='json'),
        expected=cap.expected, value=value, scope='source_expression')


@pytest.mark.parametrize('old', NAMES)
@pytest.mark.parametrize('new', NAMES)
def test_each_activation_round_trip_preserves_input_and_repeated_output_use(tmp_path, old, new):
    source = SOURCE.replace('F.relu', f'F.{old}')
    before, resolver = setup(tmp_path, source)
    cap = capability(before, resolver, old)
    assert cap.namespace == 'F'
    result = resolver.plan_activation(intent(cap, new))
    assert result['validation'] == 'static_only'
    if old == new:
        assert result['edits'] == []
    after = apply(source, result)
    assert after == source.replace(f'F.{old}', f'F.{new}')
    captured, _ = setup(tmp_path, after)
    assert [op.target for op in captured.operations] == ['aten.add.Tensor', f'aten.{new}.default', 'aten.add.Tensor']
    assert captured.operations[1].arguments.args == before.operations[1].arguments.args
    assert captured.operations[2].arguments.args[0] == captured.operations[2].arguments.args[1]


@pytest.mark.parametrize('declaration,namespace', [
    ('import torch.nn.functional as func', 'func'),
    ('from torch.nn import functional as func', 'func'),
    ('from torch import nn', 'nn.functional'),
    ('import torch as tensor', 'tensor.nn.functional'),
    ('import torch.nn as neural', 'neural.functional'),
    ('', 'torch.nn.functional'),
])
def test_proven_namespace_imports_reuse_existing_bindings(tmp_path, declaration, namespace):
    source = SOURCE.replace('import torch.nn.functional as F', declaration).replace('F.relu', namespace + '.relu')
    record, resolver = setup(tmp_path, source)
    cap = capability(record, resolver)
    assert cap.namespace == namespace
    after = apply(source, resolver.plan_activation(intent(cap)))
    assert after == source.replace(namespace + '.relu', namespace + '.gelu')


@pytest.mark.parametrize('shadow', [
    'F = torch.nn.functional',
    'def helper(F): return F',
    'def helper():\n    import torch.nn.functional as F\n    return F',
    'if True:\n    import torch.nn.functional as F',
    'class Other:\n    F = torch.nn.functional',
    'def helper():\n    try: pass\n    except Exception as F: pass',
    'def helper(value):\n    match value:\n        case {**F}: return F',
    'def helper(value):\n    match value:\n        case [*F]: return F',
    'def helper():\n    F.relu = torch.relu',
    'def helper():\n    setattr(F, "relu", torch.relu)',
    'from math import *',
])
def test_shadowed_conditional_rebound_or_mutated_namespaces_refuse(tmp_path, shadow):
    source = SOURCE + '\n' + shadow + '\n'
    record, resolver = setup(tmp_path, source)
    with pytest.raises(SourceEditRefused, match='unproven_activation_import'):
        capability(record, resolver)


@pytest.mark.parametrize('expression', ['F.relu(*(h,))', 'torch.relu(h)', 'h.relu()', 'torch.nn.ReLU()(h)'])
def test_unsupported_activation_parameters_and_call_forms_are_not_silently_dropped(tmp_path, expression):
    source = SOURCE.replace('a = F.relu(  # preserve call formatting\n            h + 1,\n        )', 'a = ' + expression)
    record, resolver = setup(tmp_path, source)
    with pytest.raises(SourceEditRefused, match='unsupported_activation_expression|unproven_activation_import'):
        resolver.activation_capability(record.operations[0].id)


def test_direct_import_reuses_proven_namespace_without_rewriting_shared_import(tmp_path):
    source = SOURCE.replace('import torch.nn.functional as F', 'from torch.nn.functional import relu as activation').replace('F.relu', 'activation')
    record, resolver = setup(tmp_path, source)
    cap = capability(record, resolver)
    assert cap.namespace == 'torch.nn.functional' and cap.importPlan is None
    assert apply(source, resolver.plan_activation(intent(cap))) == source.replace('a = activation(', 'a = torch.nn.functional.gelu(')


def test_shared_helpers_disclose_all_executions_and_change_together(tmp_path):
    source = '''import torch
from torch.nn import functional as F
def activate(h):
    return F.relu(h)
class Block(torch.nn.Module):
    def forward(self, h):
        for _ in range(2):
            h = activate(h)
        return h
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b = Block(), Block()
    def forward(self, h):
        return self.b(self.a(h))
'''
    record, resolver = setup(tmp_path, source)
    cap = capability(record, resolver)
    assert cap.scope.observedOperations == 4
    assert cap.scope.observedInvocations == 2
    assert cap.scope.observedModuleInstances == 2
    assert cap.scope.includesUnobservedExecutions
    after = apply(source, resolver.plan_activation(intent(cap)))
    new, _ = setup(tmp_path, after)
    assert len(new.operations) == 4 and all(op.target == 'aten.gelu.default' for op in new.operations)


def test_unicode_crlf_and_multiline_callee_comments_are_preserved(tmp_path):
    source = SOURCE.replace('a = F.relu', 'label = "😀\u2028"; a = F.relu').replace('\n', '\r\n')
    record, resolver = setup(tmp_path, source)
    assert apply(source, resolver.plan_activation(intent(capability(record, resolver)))) == source.replace('F.relu', 'F.gelu')
    source = SOURCE.replace('F.relu', '(F\n            # keep callee comment\n            .relu)')
    record, resolver = setup(tmp_path, source)
    with pytest.raises(SourceEditRefused, match='unsupported_activation_expression'):
        capability(record, resolver)


def test_stale_identity_derived_events_and_unsafe_names_cannot_make_patches(tmp_path):
    record, resolver = setup(tmp_path, SOURCE)
    cap = capability(record, resolver)
    for patch in [dict(capabilityId='wrong'), dict(expected='tanh'), dict(target={**cap.target.model_dump(mode='json'), 'sourceId': 'wrong'})]:
        with pytest.raises(SourceEditRefused, match='stale_capability'):
            resolver.plan_activation({**intent(cap), **patch})
    for patch in [dict(value='eval'), dict(scope='selected_call'), dict(code='F.tanh(h)')]:
        with pytest.raises(ValidationError):
            ReplaceActivationEdit.model_validate({**intent(cap), **patch})
    payload = record.model_dump(mode='json')
    payload['operations'][0]['sourceBindingIds'] = payload['operations'][1]['sourceBindingIds']
    with pytest.raises(SourceEditRefused, match='derived_operation'):
        capability(record, CapturedEditResolver(payload, {'model.py': SOURCE}))


def test_capability_contract_rejects_noncontained_or_cross_snapshot_callee(tmp_path):
    record, resolver = setup(tmp_path, SOURCE)
    cap = capability(record, resolver).model_dump(mode='json')
    for change in [lambda p: p['binding'].update(snapshotId='other'),
                   lambda p: p['binding'].update(quality='candidate'),
                   lambda p: p.update(namespace='F; danger()'),
                   lambda p: p.update(namespace='F\n'),
                   lambda p: p['binding']['anchors'][0].update(startLine=2, endLine=2)]:
        payload = deepcopy(cap)
        change(payload)
        with pytest.raises(ValidationError):
            ActivationEditCapability.model_validate(payload)
