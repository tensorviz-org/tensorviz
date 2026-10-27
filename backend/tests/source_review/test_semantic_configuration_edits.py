"""Configuration edits preserve references and disclose their wider source scope."""
import ast
from copy import deepcopy

import pytest
from pydantic import ValidationError

from services.semantic.contracts import ConfigurationLiteralCapability, SetConfigurationLiteralEdit
from services.semantic.source_configuration import ConfigurationIndex
from services.semantic.source_edits import CapturedEditResolver, SourceEditRefused
from tests.source_review.test_semantic_constructor_edits import SOURCE, setup
from tests.source_review.test_semantic_source_edits import apply


CONFIG_SOURCE = SOURCE.replace('self.first = nn.Linear(4, 4,',
    'width = 4  # shared configuration\n        self.first = nn.Linear(width, width,').replace(
    'self.last = nn.Linear(4, 2)', 'self.last = nn.Linear(width, 2)')


def capabilities(capture, resolver):
    return resolver.configuration_literal_capabilities(capture.invocations[1].id)


def intent(cap, value=None):
    return dict(kind='set_configuration_literal', capabilityId=cap.id, target=cap.target.model_dump(mode='json'),
        sourceArgumentPath=cap.sourceArgumentPath, expected=cap.expected.model_dump(mode='json'),
        value=value or dict(kind='int', value='8'), scope='configuration_definition')


def test_local_configuration_edits_only_definition_and_exposes_all_source_references(tmp_path):
    capture, resolver = setup(tmp_path, CONFIG_SOURCE)
    caps = capabilities(capture, resolver)
    assert [cap.sourceArgumentPath for cap in caps] == [['args', 0], ['args', 1]]
    assert caps[0].id != caps[1].id and caps[0].binding == caps[1].binding
    cap = caps[1]
    assert cap.symbol == 'width' and cap.binding.kind == 'configuration' and cap.reference.kind == 'argument'
    assert cap.binding.anchors[0].scope == 'Model.__init__'
    assert cap.scope.model_dump() == dict(kind='configuration_definition', affects='all_references',
        knownSourceReferences=3, observedConstructorSites=2, observedOperations=6,
        observedInvocations=3, observedModuleInstances=2, includesUnobservedReferences=True)
    plan = resolver.plan(intent(cap))
    assert plan['validation'] == 'static_only' and len(plan['edits']) == 1
    after = apply(CONFIG_SOURCE, plan)
    assert after == CONFIG_SOURCE.replace('width = 4', 'width = 8')
    # Both constructors still read width; the selected capture input is now incompatible.
    changed, _ = setup(tmp_path, after)
    assert changed.status == 'partial' and changed.invocations[-1].status == 'failed'
    assert [module.paths for module in capture.modules] == [module.paths for module in changed.modules]


def test_module_configuration_keeps_external_and_uncalled_reads_visible(tmp_path):
    source = CONFIG_SOURCE.replace('        width = 4  # shared configuration\n', '').replace(
        'class Model', 'width: int = 4  # shared configuration\nunused = width * 2\nclass Model')
    source = source.replace('self.shared = self.first', 'self.shared = self.first\n        self.unused = nn.Linear(width, width)')
    capture, resolver = setup(tmp_path, source)
    cap = capabilities(capture, resolver)[0]
    assert cap.binding.anchors[0].scope == '<module>'
    assert cap.scope.knownSourceReferences == 6 and cap.scope.observedConstructorSites == 3
    assert cap.scope.observedModuleInstances == 3 and cap.scope.observedInvocations == 3
    assert apply(source, resolver.plan(intent(cap))) == source.replace('width: int = 4', 'width: int = 8')


def test_config_used_in_computed_constructor_args_counts_impact_without_offering_inline_edit(tmp_path):
    source = CONFIG_SOURCE.replace('self.last = nn.Linear(width, 2)', 'self.last = nn.Linear(width + 0, 2)')
    capture, resolver = setup(tmp_path, source)
    cap = capabilities(capture, resolver)[0]
    assert cap.scope.observedConstructorSites == 2 and cap.scope.observedInvocations == 3
    with pytest.raises(SourceEditRefused, match='no_configuration_literals'):
        resolver.configuration_literal_capabilities(capture.invocations[-1].id)


def test_custom_constructor_and_descendant_scope_do_not_double_count_operations(tmp_path):
    source = CONFIG_SOURCE.replace('class Model', '''class Dense(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.inner = nn.Linear(width, width, bias=False)
    def forward(self, h):
        return self.inner(h)
class Model''').replace('self.first = nn.Linear(width, width, bias=False)', 'self.first = Dense(width)')
    # A same-named parameter in another function is not a binding in Model.__init__.
    capture, resolver = setup(tmp_path, source)
    cap = capabilities(capture, resolver)[0]
    assert cap.scope.observedInvocations == 3 and cap.scope.observedOperations == 6
    assert cap.scope.observedConstructorSites == 2 and cap.scope.knownSourceReferences == 2
    assert apply(source, resolver.plan(intent(cap))) == source.replace('width = 4', 'width = 8')


def test_overlapping_parent_child_configuration_scope_counts_each_operation_once(tmp_path):
    source = CONFIG_SOURCE.replace('        width = 4  # shared configuration\n', '').replace('class Model', '''width = 4
class Dense(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.inner = nn.Linear(width, width, bias=False)
    def forward(self, h):
        return self.inner(h)
class Model''').replace('self.first = nn.Linear(width, width, bias=False)', 'self.first = Dense(width)')
    capture, resolver = setup(tmp_path, source)
    cap = capabilities(capture, resolver)[0]
    assert cap.scope.observedModuleInstances == 3 and cap.scope.observedConstructorSites == 3
    assert cap.scope.observedInvocations == 5 and cap.scope.observedOperations == 6


def test_recapture_reconstructs_both_dependent_layers_from_the_edited_definition(tmp_path):
    source = CONFIG_SOURCE.replace('nn.Linear(width, width,', 'nn.Linear(4, width,')
    capture, resolver = setup(tmp_path, source)
    after = apply(source, resolver.plan(intent(capabilities(capture, resolver)[0])))
    changed, changed_resolver = setup(tmp_path, after)
    assert capture.status == changed.status == 'complete'
    cap = capabilities(changed, changed_resolver)[0]
    assert cap.expected.value == '8' and cap.scope.observedConstructorSites == 2
    first = changed.invocations[1]
    assert next(value for value in changed.values if value.id == first.output.id).shape == [2, 8]


@pytest.mark.parametrize('token,expected', [('0.25', {'kind': 'float', 'value': 0.25}),
    ('-0.0', {'kind': 'float', 'value': '-0'}), ('+0.0', {'kind': 'float', 'value': 0.0}),
    ('+0x0', {'kind': 'int', 'value': '0'})])
def test_numeric_types_noop_spelling_and_crlf_unicode_anchors(tmp_path, token, expected):
    source = '''import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        label = "😀"; 比率 = TOKEN  # keep me
        self.drop = torch.nn.Dropout(p=比率)
    def forward(self, h):
        return self.drop(h)
'''.replace('TOKEN', token).replace('\n', '\r\n')
    capture, resolver = setup(tmp_path, source)
    cap = capabilities(capture, resolver)[0]
    assert cap.expected.model_dump() == expected and cap.symbol == '比率'
    assert cap.scope.observedOperations == 0 and cap.sourceArgumentPath == ['kwargs', 'p']
    assert resolver.plan(intent(cap, expected))['edits'] == []
    after = apply(source, resolver.plan(intent(cap, dict(kind='float', value=0.5))))
    assert after == source.replace('比率 = ' + token, '比率 = 0.5')


@pytest.mark.parametrize('change', ['id', 'capture', 'path', 'expected', 'scope', 'extra'])
def test_stale_and_forged_intents_cannot_redirect_configuration_patch(tmp_path, change):
    capture, resolver = setup(tmp_path, CONFIG_SOURCE)
    payload = intent(capabilities(capture, resolver)[0])
    if change == 'id':
        payload['capabilityId'] += ':forged'
    elif change == 'capture':
        payload['target']['captureId'] += ':forged'
    elif change == 'path':
        payload['sourceArgumentPath'] = ['args', 1]  # Same value and definition, different proven reference.
    elif change == 'expected':
        payload['expected'] = dict(kind='int', value='5')
    elif change == 'scope':
        payload['scope'] = 'source_expression'
    else:
        payload['binding'] = 'other.py'
    with pytest.raises((ValidationError, SourceEditRefused)):
        resolver.plan(payload)


def test_stale_source_and_forged_binding_relations_are_rejected(tmp_path):
    capture, resolver = setup(tmp_path, CONFIG_SOURCE)
    with pytest.raises(SourceEditRefused, match='stale_source'):
        CapturedEditResolver(capture, {'model.py': CONFIG_SOURCE.replace('width = 4', 'width = 8')})
    cap = capabilities(capture, resolver)[0].model_dump(mode='json')
    for field, attr, value in [('reference', 'file', 'other.py'), ('binding', 'scope', 'other'),
                               ('reference', 'startLine', 1), ('binding', 'endLine', 99)]:
        broken = deepcopy(cap)
        broken[field]['anchors'][0][attr] = value
        with pytest.raises(ValidationError):
            ConfigurationLiteralCapability.model_validate(broken)
    assert SetConfigurationLiteralEdit.model_validate(intent(capabilities(capture, resolver)[0]))


@pytest.mark.parametrize('source', [
    'width = 4\nwidth = 4\nx = Layer(width)',
    'width = 4\nif condition: width = 8\nx = Layer(width)',
    'width = other = 4\nx = Layer(width)',
    'width = 4\ndef f(width):\n    return Layer(width)',
    'width = 4\ndef f(width=4):\n    return Layer(width)',
    'from config import width\nx = Layer(width)',
    'def f():\n    width = 4\n    def g():\n        return Layer(width)',
    'width = 4\nx = [Layer(width) for _ in range(2)]',
    'width = 4\nx = lambda: Layer(width)',
    'class C:\n    width = 4\n    x = Layer(width)',
    'width = 4\nexec(code)\nx = Layer(width)',
    'width = 4\nns = globals()\nx = Layer(width)',
    'width = 4\nx = obj.__dict__\ny = Layer(width)',
    'width = 4\nsetattr(obj, key, value)\nx = Layer(width)',
    'from config import *\nwidth = 4\nx = Layer(width)',
    'width = 4\ndef f():\n    global width\n    return Layer(width)',
    'width = 4\nwidth += 1\nx = Layer(width)',
    'width = 4\ndel width\nx = Layer(width)',
    'width = 4\ntry: pass\nexcept Exception as width: pass\nx = Layer(width)',
    'width = 4\nmatch obj:\n    case {**width}: pass\nx = Layer(width)',
    'width = 4\nmatch obj:\n    case [*width]: pass\nx = Layer(width)',
    'width = 4\nfor width in values: pass\nx = Layer(width)',
    'width = 4\nwith ctx as width: pass\nx = Layer(width)',
    'width = 4\nif (width := 8): pass\nx = Layer(width)',
    'x = Layer(width)\nwidth = 4',
    'def f():\n    return Layer(width)\nwidth = 4',
    'width: int\nx = Layer(width)',
    'x = Layer(width)',
])
def test_lexical_ambiguity_and_dynamic_namespaces_never_claim_a_configuration_definition(source):
    tree = ast.parse(source)
    reference = next(node.args[0] for node in ast.walk(tree)
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'Layer')
    assert ConfigurationIndex(tree).resolve(reference, 'scope') is None


@pytest.mark.parametrize('definition,reference', [('width = 2 + 2', 'width'),
    ('width = int(4)', 'width'), ('width = [4]', 'width[0]'),
    ('width = 4', 'width + 0'), ('width = True', 'width')])
def test_computed_definitions_and_argument_expressions_are_never_replaced_with_measured_values(tmp_path, definition, reference):
    source = CONFIG_SOURCE.replace('width = 4', definition).replace('nn.Linear(width, width,', f'nn.Linear({reference}, {reference},')
    if definition == 'width = [4]':
        source = source.replace('nn.Linear(width, 2)', 'nn.Linear(width[0], 2)')
    if definition == 'width = True':
        # Keep model construction valid while exercising rejection of a boolean configuration.
        source = source.replace('nn.Linear(width, width, bias=False)', 'nn.Dropout(p=width)').replace('nn.Linear(width, 2)', 'nn.Linear(4, 2)')
    capture, resolver = setup(tmp_path, source)
    with pytest.raises(SourceEditRefused, match='no_configuration_literals'):
        capabilities(capture, resolver)
