"""Module activation recipes use actual construction and invocation evidence."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from pydantic import ValidationError

from services.ingest.trace import trace_model
from services.semantic.contracts import ActivationEditCapability, CaptureSnapshot
from services.semantic.source_edits import CapturedEditResolver, SourceEditRefused
from tests.source_review.test_semantic_source_bindings import prepared, run
from tests.source_review.test_semantic_source_edits import apply
from tests.source_review.test_semantic_activation_edits import intent


CLASSES = dict(relu='ReLU', gelu='GELU', silu='SiLU', sigmoid='Sigmoid', tanh='Tanh')
SOURCE = '''import torch
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.ReLU(  # preserve constructor formatting
        )
        self.shared = self.a
    def forward(self, h):
        a = self.a(h + 1)
        return self.shared(a + a)
'''


def setup(tmp_path, source=SOURCE):
    namespace, context = prepared(tmp_path, source)
    with context.source_evidence.construction.observe():
        model = namespace['Model']()
    record = run(model, context)
    return record, CapturedEditResolver(record, {'model.py': source})


def capability(record, resolver):
    return resolver.capability(record.invocations[1].id)


@pytest.mark.parametrize('old', CLASSES)
@pytest.mark.parametrize('new', CLASSES)
def test_module_round_trip_changes_only_constructor_and_all_shared_calls(tmp_path, old, new):
    source = SOURCE.replace('nn.ReLU', 'nn.' + CLASSES[old])
    before, resolver = setup(tmp_path, source)
    cap = capability(before, resolver)
    assert cap.recipe == 'module_activation_v1' and cap.target.kind == 'invocation'
    assert resolver.capabilities(before.invocations[1].id) == [cap]
    assert cap.expected == old and cap.namespace == 'nn'
    assert cap.expression.kind == 'constructor'
    assert cap.expression.anchors[0].scope == 'Model.__init__'
    assert cap.scope.observedInvocations == cap.scope.observedOperations == 2
    assert cap.scope.observedModuleInstances == 1
    result = resolver.plan(intent(cap, new))
    assert result['validation'] == 'static_only'
    after = apply(source, result)
    assert after == source.replace('nn.' + CLASSES[old], 'nn.' + CLASSES[new])
    if old == new:
        assert result['edits'] == []
    updated, updated_resolver = setup(tmp_path, after)
    assert capability(updated, updated_resolver).expected == new
    assert [op.target for op in updated.operations] == [
        'aten.add.Tensor', f'aten.{new}.default', 'aten.add.Tensor', f'aten.{new}.default']
    assert updated.operations[2].arguments.args[0] == updated.operations[2].arguments.args[1]
    assert updated.operations[1].arguments.args == before.operations[1].arguments.args


def test_shared_creation_expression_counts_instances_including_unobserved_instances(tmp_path):
    source = '''import torch
from torch import nn
def create_activation():
    return nn.ReLU()
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = create_activation()
        self.b = create_activation()
        self.unused = create_activation()
    def forward(self, h):
        return self.b(self.a(h))
'''
    record, resolver = setup(tmp_path, source)
    cap = capability(record, resolver)
    assert cap.scope.observedInvocations == 2 and cap.scope.observedModuleInstances == 3
    assert cap.scope.includesUnobservedExecutions
    assert apply(source, resolver.plan(intent(cap))) == source.replace('nn.ReLU()', 'nn.GELU()')


@pytest.mark.parametrize('declaration,namespace', [
    ('import torch.nn as layers', 'layers'), ('import torch as tensor', 'tensor.nn'),
    ('', 'torch.nn'),
])
def test_module_namespaces_use_verified_existing_imports(tmp_path, declaration, namespace):
    source = SOURCE.replace('from torch import nn', declaration).replace('nn.', namespace + '.')
    record, resolver = setup(tmp_path, source)
    cap = capability(record, resolver)
    assert cap.namespace == namespace
    assert apply(source, resolver.plan(intent(cap))) == source.replace(namespace + '.ReLU', namespace + '.GELU')


@pytest.mark.parametrize('constructor', ['nn.ReLU(inplace=bool(0))', 'nn.GELU(approximate=str("tanh"))', 'nn.Identity()'])
def test_explicit_options_and_unsupported_classes_do_not_lose_their_meaning(tmp_path, constructor):
    source = SOURCE.replace('nn.ReLU(  # preserve constructor formatting\n        )', constructor)
    record, resolver = setup(tmp_path, source)
    with pytest.raises(SourceEditRefused, match='unsupported_activation_constructor'):
        capability(record, resolver)
    if constructor != 'nn.Identity()':
        with pytest.raises(SourceEditRefused, match='unsupported_activation_constructor'):
            resolver.capabilities(record.invocations[1].id)


@pytest.mark.parametrize('mutation', [
    'self.a.inplace = True', 'self.a.forward = lambda x: x + 1',
    'self.a.register_forward_hook(lambda module, args, output: output + 1)',
    'self.a.register_forward_pre_hook(lambda module, args: (args[0] + 1,))',
])
def test_modified_module_execution_cannot_claim_plain_constructor_replacement(tmp_path, mutation):
    source = SOURCE.replace('self.shared = self.a', 'self.shared = self.a\n        ' + mutation)
    record, resolver = setup(tmp_path, source)
    with pytest.raises(SourceEditRefused, match='derived_operation|incomplete_or_mutating_observation'):
        capability(record, resolver)


def test_missing_construction_and_unproven_namespace_refuse(tmp_path):
    namespace, context = prepared(tmp_path, SOURCE)
    record = run(namespace['Model'](), context)
    with pytest.raises(SourceEditRefused, match='ambiguous_constructor'):
        capability(record, CapturedEditResolver(record, {'model.py': SOURCE}))
    source = SOURCE + '\ndef helper(nn): return nn\n'
    record, resolver = setup(tmp_path, source)
    with pytest.raises(SourceEditRefused, match='unproven_activation_import'):
        capability(record, resolver)


def test_qualified_class_evidence_and_exact_constructor_binding_are_required(tmp_path):
    record, _ = setup(tmp_path)
    for change in [lambda p: p['definitions'][1].update(qualifiedName='custom.ReLU'),
                   lambda p: next(b for b in p['sourceBindings'] if b['kind'] == 'constructor')['anchors'][0].update(tokenDigest='f' * 64)]:
        payload = record.model_dump(mode='json')
        change(payload)
        with pytest.raises(SourceEditRefused, match='derived_operation|source_binding_mismatch'):
            capability(record, CapturedEditResolver(payload, {'model.py': SOURCE}))


def test_incomplete_recording_does_not_certify_a_pure_activation_per_shared_call(tmp_path):
    namespace, context = prepared(tmp_path, SOURCE)
    with context.source_evidence.construction.observe():
        model = namespace['Model']()
    context = replace(context, limits=context.limits.model_copy(update={'operations': 2}))
    result = trace_model(model, (torch.ones(2, 4),), tier='real', semantic_context=context)
    record = CaptureSnapshot.model_validate(result.semantic_capture)
    assert record.coverage.omissions
    with pytest.raises(SourceEditRefused, match='incomplete_or_mutating_observation'):
        capability(record, CapturedEditResolver(record, {'model.py': SOURCE}))


def test_module_capability_rejects_functional_target_or_call_expression_substitution(tmp_path):
    record, resolver = setup(tmp_path)
    cap = capability(record, resolver).model_dump(mode='json')
    for change in [lambda p: p['target'].update(kind='operation'),
                   lambda p: p.update(recipe='functional_activation_v1'),
                   lambda p: p['expression'].update(kind='call')]:
        payload = deepcopy(cap)
        change(payload)
        with pytest.raises(ValidationError):
            ActivationEditCapability.model_validate(payload)


def test_unicode_and_crlf_keep_exact_source_and_multiline_callee_comments_refuse(tmp_path):
    source = SOURCE.replace('self.a = nn.ReLU', 'label = "😀"; self.a = nn.ReLU').replace('\n', '\r\n')
    record, resolver = setup(tmp_path, source)
    assert apply(source, resolver.plan(intent(capability(record, resolver)))) == source.replace('nn.ReLU', 'nn.GELU')
    source = SOURCE.replace('nn.ReLU', '(nn\n            # keep me\n            .ReLU)')
    record, resolver = setup(tmp_path, source)
    with pytest.raises(SourceEditRefused, match='unsupported_activation_constructor'):
        capability(record, resolver)
