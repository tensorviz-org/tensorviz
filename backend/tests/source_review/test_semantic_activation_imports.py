"""Direct imported activations preserve other uses and review any new import."""
import ast
from copy import deepcopy

import pytest
from pydantic import ValidationError

from services.semantic.contracts import ActivationEditCapability
from services.semantic.source_edits import SourceEditRefused
from tests.source_review.test_semantic_source_edits import setup as functional_setup, apply
from tests.source_review.test_semantic_activation_edits import capability as functional_capability, intent, NAMES
from tests.source_review.test_semantic_module_activation_edits import setup as module_setup, capability as module_capability, CLASSES


FUNCTIONAL = '''"""Preserve this module docstring."""
from __future__ import annotations
# keep this import comment
from torch.nn import Module
from torch.nn.functional import relu as activate
class Model(Module):
    def forward(self, h):
        a = activate(h)  # selected call
        b = activate(h)  # unrelated use of the same import
        return a + b
'''
MODULE = '''"""Preserve this module docstring."""
from __future__ import annotations
from torch.nn import Module, ReLU as Activation
class Model(Module):
    def __init__(self):
        super().__init__()
        self.a = Activation()  # selected constructor
        self.b = Activation()  # unrelated use of the same import
    def forward(self, h):
        return self.b(self.a(h))
'''


def prepare(tmp_path, module=False, source=None):
    setup = module_setup if module else functional_setup
    record, resolver = setup(tmp_path, source if source is not None else MODULE if module else FUNCTIONAL)
    cap = module_capability(record, resolver) if module else functional_capability(record, resolver)
    return record, resolver, cap


@pytest.mark.parametrize('module', [False, True])
@pytest.mark.parametrize('new', NAMES)
def test_required_import_and_selected_callee_round_trip_without_rebinding_other_uses(tmp_path, module, new):
    source = MODULE if module else FUNCTIONAL
    before, resolver, cap = prepare(tmp_path, module)
    assert cap.importPlan is not None and cap.importPlan.line > 2
    assert cap.namespace == ('_tensorviz_nn' if module else '_tensorviz_functional')
    plan = resolver.plan(intent(cap, new))
    after = apply(source, plan)
    if new == 'relu':
        assert not plan['edits'] and after == source
    else:
        assert len(plan['edits']) == 2
        imported = f'import {cap.importPlan.module} as {cap.namespace}\n'
        expected = source.replace('        self.a = Activation()', f'        self.a = {cap.namespace}.{CLASSES[new]}()') if module else source.replace('a = activate(h)', f'a = {cap.namespace}.{new}(h)')
        lines = expected.split('\n')
        lines.insert(cap.importPlan.line - 1, imported[:-1])
        assert after == '\n'.join(lines)
        assert ast.get_docstring(ast.parse(after)) == 'Preserve this module docstring.'
    updated, _ = (module_setup if module else functional_setup)(tmp_path, after)
    assert [op.target for op in updated.operations] == ([f'aten.{new}.default', 'aten.relu.default'] if module else [f'aten.{new}.default', 'aten.relu.default', 'aten.add.Tensor'])
    assert before.operations[-1].arguments.args == updated.operations[-1].arguments.args


@pytest.mark.parametrize('module', [False, True])
def test_existing_namespace_avoids_a_new_import_and_leaves_the_alias_import_unchanged(tmp_path, module):
    source = (MODULE if module else FUNCTIONAL).replace('from torch.nn import Module', 'import torch.nn as N\nfrom torch.nn import Module')
    _, resolver, cap = prepare(tmp_path, module, source)
    assert cap.namespace == ('N' if module else 'N.functional') and cap.importPlan is None
    after = apply(source, resolver.plan(intent(cap)))
    assert 'as Activation' in after if module else 'as activate' in after
    assert len(resolver.plan(intent(cap))['edits']) == 1


@pytest.mark.parametrize('module', [False, True])
def test_new_namespace_avoids_identifiers_in_all_scopes_and_preserves_crlf_unicode(tmp_path, module):
    source = MODULE if module else FUNCTIONAL
    base = '_tensorviz_nn' if module else '_tensorviz_functional'
    source += f'\ndef unused({base}):\n    {base}_1 = "😀"\n    return {base}_1\n'
    source = '#!/usr/bin/env python\n# coding: utf-8\n' + source
    source = source.replace('\n', '\r\n').rstrip('\r\n')
    _, resolver, cap = prepare(tmp_path, module, source)
    assert cap.namespace == base + '_2' and cap.importPlan.lineEnding == '\r\n'
    after = apply(source, resolver.plan(intent(cap)))
    assert after.startswith('#!/usr/bin/env python\r\n# coding: utf-8\r\n')
    assert '\n' not in after.replace('\r\n', '') and not after.endswith('\n')
    assert f'{base}_1 = "😀"' in after
    (module_setup if module else functional_setup)(tmp_path, after)


@pytest.mark.parametrize('extra', [
    '\ndef unused(activate): return activate\n',
    '\ndef unused(): return globals()\n',
    '\ndef unused(): return locals()\n',
    '\ndef unused(): exec("pass")\n',
    '\ndef unused(value): return value.__dict__\n',
    '\nfrom math import *\n',
])
def test_ambiguous_or_dynamic_namespace_never_receives_import_edit_authority(tmp_path, extra):
    with pytest.raises(SourceEditRefused, match='unproven_activation_import'):
        prepare(tmp_path, source=FUNCTIONAL + extra)


def test_future_semicolon_boundary_is_refused_instead_of_moving_an_import_before_future(tmp_path):
    source = FUNCTIONAL.replace('from __future__ import annotations\n# keep this import comment\n', 'from __future__ import annotations; ')
    with pytest.raises(SourceEditRefused, match='unsafe_import_insertion'):
        prepare(tmp_path, source=source)


def test_contract_does_not_authorize_arbitrary_imports_or_insertion_after_callee(tmp_path):
    _, _, capability = prepare(tmp_path)
    for change in [lambda p: p['importPlan'].update(module='os'),
                   lambda p: p['importPlan'].update(line=capability.expression.anchors[0].startLine + 1),
                   lambda p: p['importPlan'].update(lineEnding='\nexec("pass")\n'),
                   lambda p: p.update(namespace='torch.nn.functional'),
                   lambda p: p['importPlan'].update(module='torch.nn')]:
        payload = deepcopy(capability.model_dump(mode='json'))
        change(payload)
        with pytest.raises(ValidationError):
            ActivationEditCapability.model_validate(payload)
