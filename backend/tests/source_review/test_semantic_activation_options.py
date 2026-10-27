"""Replacing activations must preserve written options, storage effects and inputs."""
from copy import deepcopy

import pytest
from pydantic import ValidationError

from services.semantic.contracts import ActivationEditCapability
from services.semantic.source_edits import CapturedEditResolver, SourceEditRefused
from tests.source_review.test_semantic_module_activation_edits import setup, CLASSES
from tests.source_review.test_semantic_source_edits import apply
from tests.source_review.test_semantic_activation_edits import intent


def source(expression, module=False, named_input=False):
    return '''import torch
from torch import nn
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        CONSTRUCTOR
    def forward(self, h):
        h = h + 1
        view = h.view(-1)
        a = EXPRESSION  # preserve me
        return a, view
'''.replace('CONSTRUCTOR', 'self.activation = ' + expression if module else 'pass').replace(
        'EXPRESSION', ('self.activation(input=h)' if named_input else 'self.activation(h)') if module else expression)


def capability(capture, resolver, module=False):
    return resolver.capabilities(capture.invocations[1].id if module else capture.operations[2].id)[0]


@pytest.mark.parametrize('module', [False, True])
@pytest.mark.parametrize('old,new', [('relu', 'silu'), ('silu', 'relu')])
@pytest.mark.parametrize('inplace', [False, True])
@pytest.mark.parametrize('positional', [False, True])
def test_compatible_replacements_preserve_options_alias_effects_and_source(tmp_path, module, old, new, inplace, positional):
    callee = 'nn.' + CLASSES[old] if module else 'F.' + old
    expression = callee + '(' + ('' if module else 'h, ') + ('' if positional else 'inplace=') + str(inplace) + ')'
    before = source(expression, module, named_input=True)
    capture, resolver = setup(tmp_path, before)
    cap = capability(capture, resolver, module)
    assert cap.options.inplace is inplace and cap.options.approximate is None
    assert cap.compatibleActivations == ['relu', 'silu']
    assert resolver.plan(intent(cap, old))['edits'] == []
    after = apply(before, resolver.plan(intent(cap, new)))
    replacement = 'nn.' + CLASSES[new] if module else 'F.' + new
    assert after == before.replace(callee, replacement)
    changed, changed_resolver = setup(tmp_path, after)
    assert capability(changed, changed_resolver, module).expected == new
    previous, updated = capture.operations[2], changed.operations[2]
    assert updated.target == f'aten.{new}{"_" if inplace else ""}.default'
    assert previous.arguments == updated.arguments and previous.output == updated.output
    assert previous.effects == updated.effects
    if inplace:
        assert [effect.kind for effect in updated.effects] == ['write', 'alias_update']
        assert changed.output.items[1] == updated.effects[1].after
    else:
        assert not updated.effects
    with pytest.raises(SourceEditRefused, match='incompatible_activation_options'):
        resolver.plan(intent(cap, 'gelu'))


@pytest.mark.parametrize('name', CLASSES)
def test_keyword_inputs_retain_spelling_and_support_unrestricted_default_choices(tmp_path, name):
    before = source('F.' + name + '(input=h)')
    capture, resolver = setup(tmp_path, before)
    cap = capability(capture, resolver)
    assert cap.compatibleActivations == list(CLASSES)
    assert cap.options.inplace is cap.options.approximate is None
    after = apply(before, resolver.plan(intent(cap, 'tanh')))
    assert after == before.replace('F.' + name + '(', 'F.tanh(')
    assert setup(tmp_path, after)[0].status == 'complete'


@pytest.mark.parametrize('module', [False, True])
@pytest.mark.parametrize('approximate', ['none', 'tanh'])
def test_gelu_approximation_is_never_dropped_or_passed_to_an_incompatible_activation(tmp_path, module, approximate):
    expression = ('nn.GELU(' if module else 'F.gelu(input=h, ') + f'approximate="{approximate}")'
    before = source(expression, module)
    capture, resolver = setup(tmp_path, before)
    cap = capability(capture, resolver, module)
    assert cap.options.approximate == approximate and cap.compatibleActivations == ['gelu']
    assert apply(before, resolver.plan(intent(cap, 'gelu'))) == before
    for name in ['relu', 'silu', 'sigmoid', 'tanh']:
        with pytest.raises(SourceEditRefused, match='incompatible_activation_options'):
            resolver.plan(intent(cap, name))


def test_positional_constructor_approximation_and_multiline_comments_stay_intact(tmp_path):
    before = source('nn.GELU(\n            "tanh",  # keep approximation\n        )', module=True)
    capture, resolver = setup(tmp_path, before)
    cap = capability(capture, resolver, True)
    assert cap.options.approximate == 'tanh' and resolver.plan(intent(cap, 'gelu'))['edits'] == []
    before = source('F.relu(\n            inplace=True,  # retain effect\n            input=h,\n        )').replace('\n', '\r\n')
    capture, resolver = setup(tmp_path, before)
    assert apply(before, resolver.plan(intent(capability(capture, resolver), 'silu'))) == before.replace('F.relu', 'F.silu')


@pytest.mark.parametrize('module', [False, True])
@pytest.mark.parametrize('expression', ['bool(0)', '0', 'None'])
def test_computed_or_non_boolean_inplace_options_never_become_proven_booleans(tmp_path, module, expression):
    before = source(('nn.ReLU(' if module else 'F.relu(h, ') + 'inplace=' + expression + ')', module)
    capture, resolver = setup(tmp_path, before)
    with pytest.raises(SourceEditRefused, match='unsupported_activation_'):
        if module:
            resolver.module_activation_capability(capture.invocations[1].id)
        else:
            resolver.activation_capability(capture.operations[2].id)


@pytest.mark.parametrize('change', ['effect', 'storage', 'path', 'output', 'omissions'])
def test_missing_or_inconsistent_mutation_evidence_cannot_authorize_an_inplace_replacement(tmp_path, change):
    before = source('F.relu(h, inplace=True)')
    capture, resolver = setup(tmp_path, before)
    payload = capture.model_dump(mode='json')
    op = payload['operations'][2]
    if change == 'effect':
        op['effects'] = []
    elif change == 'storage':
        for value in payload['values']:
            value['storageId'] = None
    elif change == 'path':
        op['effects'][0]['argumentPath'] = ['args', 99]
    elif change == 'output':
        op['output'] = {'kind': 'none'}
    else:
        payload['coverage']['omissions'] = ['item_limit']
        payload['status'] = 'partial'
    with pytest.raises((SourceEditRefused, ValidationError)):
        capability(capture, CapturedEditResolver(payload, {'model.py': before}))


def test_capability_contract_cannot_offer_choices_that_drop_or_confuse_authored_options(tmp_path):
    before = source('nn.ReLU(inplace=True)', module=True)
    capture, resolver = setup(tmp_path, before)
    cap = capability(capture, resolver, True).model_dump(mode='json')
    for patch in [dict(compatibleActivations=['relu', 'gelu', 'silu']), dict(compatibleActivations=['silu', 'relu']),
                  dict(expected='gelu'), dict(options={'inplace': True, 'approximate': 'none'}),
                  dict(options={'inplace': 1})]:
        with pytest.raises(ValidationError):
            ActivationEditCapability.model_validate({**deepcopy(cap), **patch})
