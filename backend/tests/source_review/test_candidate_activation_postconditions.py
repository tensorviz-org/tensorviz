"""Activation postconditions reuse source proofs, known kernels and call boundaries."""
from copy import deepcopy

import pytest

from services.semantic.candidate import stage_candidate
from services.semantic.candidate_postconditions import check_postconditions
from services.semantic.contracts import CandidatePostconditions
from services.semantic.source_edits import CapturedEditResolver
from tests.source_review.test_semantic_candidate import prepare, replay
from tests.source_review.test_semantic_source_edits import apply
from tests.source_review.test_semantic_activation_edits import SOURCE as FUNCTIONAL, intent
from tests.source_review.test_semantic_module_activation_edits import SOURCE as MODULE
from tests.source_review.test_semantic_candidate_worker import payload, replay as private_replay


def proposal(capture, source, kind, value='gelu'):
    resolver = CapturedEditResolver(capture, {'model.py': source})
    cap = (resolver.activation_capability(next(op.id for op in capture.operations if op.target == 'aten.relu.default'))
        if kind == 'functional' else resolver.module_activation_capability(capture.invocations[1].id))
    edit = intent(cap, value)
    return edit, apply(source, resolver.plan(edit)), cap


def candidate(root, source, kind, value='gelu'):
    plan, base = prepare(root / 'original', source)
    edit, after, cap = proposal(base, source, kind, value)
    with stage_candidate(str(root / 'original'), plan, base, edit, 'model.py', after) as staged:
        process, envelope = replay(staged.workspace, staged.execution.plan, root)
        assert process.returncode == 0, process.stderr
        capture = envelope['result']['diagnostics']['semantic_capture']
    return staged, capture, cap


@pytest.fixture(scope='module', params=['functional', 'module'])
def observed(request, tmp_path_factory):
    root = tmp_path_factory.mktemp('candidate-activation')
    return candidate(root, FUNCTIONAL if request.param == 'functional' else MODULE, request.param)


def test_known_activation_is_observed_at_exact_edited_source(observed):
    staged, capture, cap = observed
    result = check_postconditions(staged.execution, staged.postconditions, capture)
    assert result.status == 'verified' and result.reason == 'observed_requested_activation'
    assert result.recipe == cap.recipe and result.scope == 'observed_expression'
    assert result.observedOperations == cap.scope.observedOperations
    assert result.bindingId == staged.postconditions.expression.id


@pytest.mark.parametrize('kind,source', [('functional', FUNCTIONAL), ('module', MODULE)])
@pytest.mark.parametrize('replacement', ['silu', 'sigmoid', 'tanh'])
def test_all_other_supported_replacements_use_the_existing_capture_engine(tmp_path, kind, source, replacement):
    staged, capture, _ = candidate(tmp_path, source, kind, replacement)
    result = check_postconditions(staged.execution, staged.postconditions, capture)
    assert result.status == 'verified'


@pytest.mark.parametrize('kind,source', [
    ('functional', '''from torch.nn import Module
from torch.nn.functional import relu as activation
class Model(Module):
    def forward(self, h):
        label = "🚀"; a = activation(h)
        return a + a
'''),
    ('module', '''from torch.nn import Module, ReLU as Activation
class Model(Module):
    def __init__(self):
        super().__init__()
        label = "🚀"; self.a = Activation()
        self.shared = self.a
    def forward(self, h):
        return self.shared(self.a(h))
'''),
])
@pytest.mark.parametrize('ending', ['\n', '\r\n'])
def test_required_import_unicode_and_line_endings_map_the_real_expression(tmp_path, kind, source, ending):
    staged, capture, cap = candidate(tmp_path, source.replace('\n', ending), kind)
    assert cap.importPlan is not None
    result = check_postconditions(staged.execution, staged.postconditions, capture)
    assert result.status == 'verified'
    assert staged.postconditions.expression.anchors[0].startLine == cap.expression.anchors[0].startLine + 1
    assert staged.postconditions.expression.anchors[0].startColumn == cap.expression.anchors[0].startColumn


@pytest.mark.parametrize('kind', ['functional', 'module'])
@pytest.mark.parametrize('inplace', ['True', 'False'])
def test_preserves_explicit_mutation_behavior_and_requires_its_write_evidence(tmp_path, kind, inplace):
    if kind == 'functional':
        source = 'import torch\nfrom torch.nn import functional as F\nclass Model(torch.nn.Module):\n    def forward(self, h):\n        return F.relu(h, inplace=' + inplace + ')\n'
    else:
        source = MODULE.replace('nn.ReLU(  # preserve constructor formatting\n        )', 'nn.ReLU(inplace=' + inplace + ')')
    plan, base = prepare(tmp_path / 'original', source)
    resolver = CapturedEditResolver(base, {'model.py': source})
    cap = (resolver.activation_capability(base.operations[0].id) if kind == 'functional'
        else resolver.module_activation_capability(base.invocations[1].id))
    edit = intent(cap, 'silu')
    with stage_candidate(str(tmp_path / 'original'), plan, base, edit, 'model.py', apply(source, resolver.plan(edit))) as staged:
        process, envelope = replay(staged.workspace, staged.execution.plan, tmp_path)
        assert process.returncode == 0, process.stderr
        capture = envelope['result']['diagnostics']['semantic_capture']
        assert check_postconditions(staged.execution, staged.postconditions, capture).status == 'verified'
        if inplace == 'True':
            next(op for op in capture['operations'] if op['target'] == 'aten.silu_.default')['effects'] = []
            assert check_postconditions(staged.execution, staged.postconditions, capture).status == 'unavailable'


def test_a_known_different_activation_fails_despite_successful_runtime(observed):
    staged, original, _ = observed
    capture = deepcopy(original)
    for op in capture['operations']:
        if op['target'] == 'aten.gelu.default':
            op['target'] = 'aten.relu.default'; op['arguments']['kwargs'] = {}
    for definition in capture['definitions']:
        if definition['qualifiedName'] == 'torch.nn.modules.activation.GELU':
            definition['qualifiedName'] = 'torch.nn.modules.activation.ReLU'
    result = check_postconditions(staged.execution, staged.postconditions, capture)
    assert result.status == 'failed' and result.reason == 'observed_activation_mismatch'


@pytest.mark.parametrize('change', ['partial', 'derived', 'options', 'duplicate_binding'])
def test_partial_derived_or_ambiguous_evidence_cannot_certify_the_change(observed, change):
    staged, original, _ = observed
    capture = deepcopy(original)
    if change == 'partial': capture['status'] = 'partial'
    op = next(op for op in capture['operations'] if op['target'] == 'aten.gelu.default')
    if change == 'derived': op['target'] = 'aten.custom.default'
    if change == 'options': op['arguments']['kwargs'] = {'approximate': {'kind': 'str', 'value': 'tanh'}}
    if change == 'duplicate_binding':
        binding = deepcopy(next(b for b in capture['sourceBindings'] if b['id'] == staged.postconditions.expression.id))
        binding['id'] = 'binding:competing'
        capture['sourceBindings'].append(binding)
    assert check_postconditions(staged.execution, staged.postconditions, capture).status == 'unavailable'


def test_module_shared_calls_all_participate_in_the_postcondition(tmp_path):
    staged, capture, _ = candidate(tmp_path, MODULE, 'module')
    assert check_postconditions(staged.execution, staged.postconditions, capture).observedOperations == 2
    calls = [op for op in capture['operations'] if op['target'] == 'aten.gelu.default']
    calls[-1]['target'] = 'aten.relu.default'; calls[-1]['arguments']['kwargs'] = {}
    # A class/kernel disagreement is ambiguous, not evidence for a different class.
    assert check_postconditions(staged.execution, staged.postconditions, capture).status == 'unavailable'


def test_one_constructor_expression_tracks_multiple_instances_but_counts_only_observed_calls(tmp_path):
    source = '''import torch
from torch import nn
def activation():
    return nn.ReLU()
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b, self.unused = activation(), activation(), activation()
    def forward(self, h):
        return self.b(self.a(h))
'''
    staged, capture, cap = candidate(tmp_path, source, 'module')
    assert cap.scope.observedModuleInstances == 3
    result = check_postconditions(staged.execution, staged.postconditions, capture)
    assert result.status == 'verified' and result.observedOperations == 2


@pytest.mark.parametrize('kind,source', [('functional', FUNCTIONAL), ('module', MODULE)])
def test_successful_candidate_without_the_selected_call_remains_unverified(tmp_path, kind, source):
    # Relocation-sensitive source is allowed to run, but candidate success must
    # not certify the branch that was observed only in the original workspace.
    source = 'from pathlib import Path\n' + source.replace('def forward(self, h):',
        'def forward(self, h):\n        if Path(__file__).parent.name == "workspace":\n            return h')
    staged, capture, _ = candidate(tmp_path, source, kind)
    assert capture['status'] == 'complete'
    result = check_postconditions(staged.execution, staged.postconditions, capture)
    assert result.status == 'unavailable' and result.reason == 'expression_not_observed'


def test_operation_invocation_and_instance_ids_do_not_establish_correspondence(observed):
    staged, capture, _ = observed
    identifiers = {item['id']: 'renamed:' + item['id'] for kind in ['operations', 'invocations', 'modules', 'definitions', 'values']
        for item in capture[kind]}
    def rename(value):
        if isinstance(value, dict): return {key: rename(item) for key, item in value.items()}
        if isinstance(value, list): return [rename(item) for item in value]
        return identifiers.get(value, value) if isinstance(value, str) else value
    assert check_postconditions(staged.execution, staged.postconditions, rename(capture)).status == 'verified'


def test_private_worker_publishes_scoped_activation_evidence(tmp_path):
    plan, base = prepare(tmp_path / 'original', MODULE)
    edit, after, _ = proposal(base, MODULE, 'module')
    process, envelope = private_replay((tmp_path / 'original').resolve(), payload(plan, base, edit, after), tmp_path)
    assert process.returncode == 0, process.stderr
    result = CandidatePostconditions.model_validate(envelope['worker']['candidatePostconditions'])
    assert result.status == 'verified' and result.observedOperations == 2
