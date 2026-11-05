"""Constructor argument positions belong to saved syntax, never forward ports."""
from copy import deepcopy

import pytest
import torch
from pydantic import ValidationError

from services.ingest.trace import trace_model
from services.semantic.contracts import CaptureSnapshot, ConstructorLiteralCapability, SetConstructorLiteralEdit
from services.semantic.source_edits import CapturedEditResolver, SourceEditRefused
from tests.source_review.test_semantic_source_bindings import prepared
from tests.source_review.test_semantic_source_edits import apply


SOURCE = '''import torch
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(4, 4, bias=False)  # preserve constructor and comment
        self.shared = self.first
        self.last = nn.Linear(4, 2)
    def forward(self, h):
        a = self.first(h)
        b = self.shared(h)
        return self.last(a + b)
'''


def setup(tmp_path, source=SOURCE):
    namespace, context = prepared(tmp_path, source)
    with context.source_evidence.construction.observe():
        model = namespace['Model']()
    result = trace_model(model, (torch.ones(2, 4),), tier='real', semantic_context=context)
    capture = CaptureSnapshot.model_validate(result.semantic_capture)
    return capture, CapturedEditResolver(capture, {'model.py': source})


def capabilities(capture, resolver):
    return resolver.constructor_literal_capabilities(capture.invocations[1].id)


def intent(cap, value=None):
    return dict(kind='set_constructor_literal', capabilityId=cap.id, target=cap.target.model_dump(mode='json'),
        sourceArgumentPath=cap.sourceArgumentPath, expected=cap.expected.model_dump(mode='json'),
        value=value or dict(kind='int', value='8'), scope='source_expression')


def test_exact_positional_constructor_patch_preserves_sharing_and_does_not_silently_repair_shapes(tmp_path):
    capture, resolver = setup(tmp_path)
    caps = capabilities(capture, resolver)
    assert [cap.sourceArgumentPath for cap in caps] == [['args', 0], ['args', 1]]
    cap = caps[1]
    assert cap.constructor == 'Linear' and cap.target.kind == 'invocation'
    assert cap.expression.kind == 'constructor' and cap.binding.kind == 'argument'
    assert cap.scope.observedInvocations == 2 and cap.scope.observedModuleInstances == 1
    assert cap.scope.observedOperations >= 2 and cap.scope.includesUnobservedExecutions
    plan = resolver.plan(intent(cap))
    assert plan['validation'] == 'static_only'
    after = apply(SOURCE, plan)
    assert after == SOURCE.replace('nn.Linear(4, 4,', 'nn.Linear(4, 8,')
    changed, _ = setup(tmp_path, after)
    assert changed.status == 'partial'  # The downstream Linear still expects width 4.
    assert changed.invocations[-1].status == 'failed'
    assert [module.paths for module in changed.modules] == [module.paths for module in capture.modules]
    assert 'self.last = nn.Linear(4, 2)' in after


def test_failed_forward_can_propose_a_direct_constructor_repair(tmp_path):
    source = SOURCE.replace('nn.Linear(4, 4,', 'nn.Linear(3, 4,')
    capture, resolver = setup(tmp_path, source)
    assert capture.status == 'partial'
    cap = capabilities(capture, resolver)[0]
    after = apply(source, resolver.plan(intent(cap, dict(kind='int', value='4'))))
    assert after == SOURCE and setup(tmp_path, after)[0].status == 'complete'


@pytest.mark.parametrize('declaration,callee', [
    ('from torch.nn import Linear as Dense', 'Dense'),
    ('import torch.nn as layers', 'layers.Linear'),
    ('', 'torch.nn.Linear'),
])
def test_proven_namespaces_and_direct_constructor_aliases_keep_actual_source_argument_paths(tmp_path, declaration, callee):
    source = SOURCE.replace('from torch import nn', declaration).replace('nn.Module', 'torch.nn.Module')
    source = source.replace('nn.Linear', callee).replace(callee + '(4, 4,', callee + '(in_features=4, out_features=4,')
    capture, resolver = setup(tmp_path, source)
    caps = capabilities(capture, resolver)
    assert [cap.sourceArgumentPath for cap in caps] == [['kwargs', 'in_features'], ['kwargs', 'out_features']]
    assert apply(source, resolver.plan(intent(caps[1]))) == source.replace('out_features=4', 'out_features=8')


def test_zero_operation_eval_call_can_edit_its_authored_dropout_probability(tmp_path):
    source = '''import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = torch.nn.Dropout(p=0.25)
    def forward(self, h):
        return self.drop(h)
'''
    capture, resolver = setup(tmp_path, source)
    cap = capabilities(capture, resolver)[0]
    assert cap.scope.observedOperations == 0 and cap.scope.observedInvocations == 1
    assert cap.expected.kind == 'float' and cap.sourceArgumentPath == ['kwargs', 'p']
    assert apply(source, resolver.plan(intent(cap, dict(kind='float', value=0.5)))) == source.replace('p=0.25', 'p=0.5')


def test_helper_constructor_scope_counts_all_instances_and_calls_without_specializing_one(tmp_path):
    source = SOURCE.replace('class Model', 'def make():\n    return nn.Linear(4, 4, bias=False)\nclass Model')
    source = source.replace('self.first = nn.Linear(4, 4, bias=False)', 'self.first = make()')
    source = source.replace('self.shared = self.first', 'self.shared = make()\n        self.unused = make()')
    capture, resolver = setup(tmp_path, source)
    cap = capabilities(capture, resolver)[1]
    assert cap.scope.observedModuleInstances == 3 and cap.scope.observedInvocations == 2
    assert cap.expression.anchors[0].scope == 'make'
    assert apply(source, resolver.plan(intent(cap))) == source.replace('return nn.Linear(4, 4,', 'return nn.Linear(4, 8,')


@pytest.mark.parametrize('expression,reason', [
    ('nn.Linear(width, width, bias=False)', 'no_constructor_literals'),
    ('nn.Linear(2 + 2, int(4), bias=False)', 'no_constructor_literals'),
    ('nn.Linear(*(4, 4), bias=False)', 'ambiguous_constructor_arguments'),
    ('nn.Linear(4, **{"out_features": 4})', 'ambiguous_constructor_arguments'),
])
def test_config_variables_computed_values_and_splats_never_become_literals(tmp_path, expression, reason):
    source = SOURCE.replace('class Model', 'width = 4\nclass Model').replace('nn.Linear(4, 4, bias=False)', expression)
    capture, resolver = setup(tmp_path, source)
    with pytest.raises(SourceEditRefused, match=reason):
        capabilities(capture, resolver)


def test_legacy_captures_cannot_assume_custom_classes_or_shadowed_namespaces(tmp_path):
    for source in [SOURCE + '\ndef helper(nn): return nn\n',
                   SOURCE.replace('class Model', 'class Dense(nn.Linear): pass\nclass Model').replace('nn.Linear(4, 4,', 'Dense(4, 4,')]:
        capture, resolver = setup(tmp_path, source)
        for module in capture.modules:
            module.constructorClassVerified = False
        resolver = CapturedEditResolver(capture, {'model.py': source})
        with pytest.raises(SourceEditRefused, match='unproven_constructor'):
            capabilities(capture, resolver)


@pytest.mark.parametrize('declaration,callee', [
    ('class Dense(nn.Linear): pass', 'Dense'),
    ('class 稠密(nn.Linear): pass', '稠密'),
    ('factory = nn.Linear', 'factory'),
    ('import types\nnamespace = types.ModuleType("layers")\nnamespace.Dense = nn.Linear', 'namespace.Dense'),
])
def test_observed_class_identity_enables_custom_and_aliased_constructor_literals(tmp_path, declaration, callee):
    source = SOURCE.replace('class Model', declaration + '\nclass Model').replace('nn.Linear(4, 4,', callee + '(4, 4,')
    capture, resolver = setup(tmp_path, source)
    first = next(m for m in capture.modules if 'first' in m.paths)
    assert first.constructorClassVerified
    cap = capabilities(capture, resolver)[1]
    assert cap.scope.observedInvocations == 2
    assert apply(source, resolver.plan(intent(cap))) == source.replace(callee + '(4, 4,', callee + '(4, 8,')
    assert not resolver.plan(intent(cap, dict(kind='int', value='4')))['edits']


def test_custom_configuration_reference_stays_a_reference_and_changed_literal_recaptures(tmp_path):
    source = '''import torch
class Scale(torch.nn.Module):
    def __init__(self, width, *, factor):
        super().__init__()
        self.width, self.factor = width, factor
    def forward(self, h):
        return h * self.factor
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        width = 4
        self.first = Scale(width, factor=2)  # preserve variable and comment
    def forward(self, h):
        return self.first(h)
'''
    capture, resolver = setup(tmp_path, source)
    caps = capabilities(capture, resolver)
    assert len(caps) == 1 and caps[0].sourceArgumentPath == ['kwargs', 'factor']
    after = apply(source, resolver.plan(intent(caps[0], dict(kind='int', value='3'))))
    assert after == source.replace('factor=2', 'factor=3')
    changed, _ = setup(tmp_path, after)
    assert changed.status == 'complete' and changed.operations[0].arguments.args[1].value == '3'


@pytest.mark.parametrize('factory,expression,reason', [
    ('def make(width):\n    return nn.Linear(width, width, bias=False)', 'make(4)', 'no_constructor_literals'),
    ('from functools import partial\nmake = partial(nn.Linear, bias=False)', 'make(4, 4)', 'unproven_constructor'),
])
def test_general_factory_arguments_do_not_masquerade_as_constructor_arguments(tmp_path, factory, expression, reason):
    source = SOURCE.replace('class Model', factory + '\nclass Model').replace('nn.Linear(4, 4, bias=False)', expression)
    capture, resolver = setup(tmp_path, source)
    with pytest.raises(SourceEditRefused, match=reason):
        capabilities(capture, resolver)


def test_custom_shared_constructor_requires_class_proof_for_every_retained_instance(tmp_path):
    source = SOURCE.replace('class Model', 'class Dense(nn.Linear): pass\ndef make():\n    return Dense(4, 4, bias=False)\nclass Model')
    source = source.replace('self.first = nn.Linear(4, 4, bias=False)', 'self.first = make()').replace('self.shared = self.first', 'self.shared = make()')
    capture, _ = setup(tmp_path, source)
    next(m for m in capture.modules if 'shared' in m.paths).constructorClassVerified = False
    with pytest.raises(SourceEditRefused, match='unproven_constructor'):
        capabilities(capture, CapturedEditResolver(capture, {'model.py': source}))


def test_identically_named_but_distinct_custom_classes_at_one_site_remain_ambiguous(tmp_path):
    source = SOURCE.replace('class Model', 'def make():\n    class Dense(nn.Linear): pass\n    return Dense(4, 4, bias=False)\nclass Model')
    source = source.replace('self.first = nn.Linear(4, 4, bias=False)', 'self.first = make()').replace('self.shared = self.first', 'self.shared = make()')
    capture, resolver = setup(tmp_path, source)
    instances = [m for m in capture.modules if any(p in ['first', 'shared'] for p in m.paths)]
    assert all(m.constructorClassVerified for m in instances)
    assert len({m.definitionId for m in instances}) == 2
    with pytest.raises(SourceEditRefused, match='ambiguous_constructor'):
        capabilities(capture, resolver)


def test_noop_keeps_numeric_spelling_and_unicode_crlf_ranges_remain_exact(tmp_path):
    source = SOURCE.replace('self.first =', 'label = "😀"; self.first =').replace('nn.Linear(4, 4,', 'nn.Linear(+4, 0x4,').replace('\n', '\r\n')
    capture, resolver = setup(tmp_path, source)
    cap = capabilities(capture, resolver)[1]
    unchanged = resolver.plan(intent(cap, dict(kind='int', value='4')))
    assert not unchanged['edits'] and apply(source, unchanged) == source
    assert apply(source, resolver.plan(intent(cap))) == source.replace('0x4', '8')


def test_identity_expected_value_and_source_path_cannot_be_forged(tmp_path):
    capture, resolver = setup(tmp_path)
    cap = capabilities(capture, resolver)[1]
    for patch in [dict(capabilityId='unknown'), dict(expected=dict(kind='int', value='3')),
                  dict(sourceArgumentPath=['kwargs', 'width']), dict(target={**cap.target.model_dump(mode='json'), 'sourceId':'old'})]:
        with pytest.raises(SourceEditRefused, match='stale_capability'):
            resolver.plan({**intent(cap), **patch})
    for patch in [dict(sourceArgumentPath=['args', 256]), dict(sourceArgumentPath=['kwargs', '']),
                  dict(sourceArgumentPath=['output']), dict(argumentPath=['args', 0]),
                  dict(target={**cap.target.model_dump(mode='json'), 'kind':'operation'})]:
        with pytest.raises(ValidationError):
            SetConstructorLiteralEdit.model_validate({**intent(cap), **patch})
    for change in [lambda p:p['binding'].update(kind='call'),
                   lambda p:p['scope'].update(observedInvocations=0),
                   lambda p:p['binding']['anchors'][0].update(startLine=1, endLine=1)]:
        payload = deepcopy(cap.model_dump(mode='json')); change(payload)
        with pytest.raises(ValidationError):
            ConstructorLiteralCapability.model_validate(payload)


@pytest.mark.parametrize('padding', ['files', 'bytes'])
def test_document_discovers_exact_constructor_edits_with_host_workspace_padding(tmp_path, padding):
    from services.semantic.document import capture_document
    from services.validation.source import MAX_BYTES, MAX_FILES
    import hashlib

    capture, _ = setup(tmp_path)
    sources = {'model.py': SOURCE}
    text = 'unused = True\n' if padding == 'files' else '#' + 'x' * MAX_BYTES
    for index in range(MAX_FILES if padding == 'files' else 1):
        path = f'unrelated_{index}.py'
        sources[path] = text
        capture.source.files.append(capture.source.files[0].model_copy(update={
            'path': path, 'digest': hashlib.sha256(text.encode()).hexdigest()}))
    capture.source.dependencyScope = 'partial'
    document, _ = capture_document(capture, name='Model', graph_id='host-workspace', model_file='model.py',
        source_digests={file.path: file.digest for file in capture.source.files},
        source_graph={'nodes': [], 'edges': []}, source_spans={}, source_texts=sources)
    caps = [cap for node in document.semantic.nodes for cap in node.metadata.get('sourceEditCapabilities', [])
        if cap['command'] == 'set_constructor_literal' and cap['sourceArgumentPath'] == ['args', 1]]
    assert caps
    assert all(cap['expression']['quality'] == 'exact' for cap in caps)
    assert all(node.metadata.get('sourceEditLimitation') != 'source_limit' for node in document.semantic.nodes)
    resolver = CapturedEditResolver(capture, sources)
    cap = capabilities(capture, resolver)[1]
    assert apply(SOURCE, resolver.plan(intent(cap))) == SOURCE.replace('nn.Linear(4, 4,', 'nn.Linear(4, 8,')
    with pytest.raises(SourceEditRefused, match='stale_source'):
        CapturedEditResolver(capture, {**sources, 'model.py': SOURCE + '\n'})


@pytest.mark.parametrize('budget', ['files', 'bytes', 'file_bytes'])
def test_relevant_model_and_binding_sources_still_obey_edit_budgets(tmp_path, monkeypatch, budget):
    from services.semantic import source_edits

    capture, _ = setup(tmp_path)
    sources = {'model.py': SOURCE}
    if budget == 'files':
        # Retain the model even when every binding points to another file.
        for binding in capture.sourceBindings:
            for anchor in binding.anchors:
                anchor.file = 'helper.py'
        sources['helper.py'] = SOURCE
        capture.source.files.append(capture.source.files[0].model_copy(update={'path': 'helper.py'}))
        monkeypatch.setattr(source_edits, 'MAX_FILES', 1)
    else:
        monkeypatch.setattr(source_edits, 'MAX_BYTES' if budget == 'bytes' else 'MAX_FILE_BYTES', len(SOURCE.encode()) - 1)
    with pytest.raises(SourceEditRefused, match='source_limit'):
        CapturedEditResolver(capture, sources)
