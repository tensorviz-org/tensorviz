"""An inserted activation must consume and feed the exact reviewed operand use."""
from copy import deepcopy

import pytest

from services.semantic.candidate import stage_candidate
from services.semantic.candidate_postconditions import check_postconditions
from services.semantic.source_edits import CapturedEditResolver
from tests.source_review.test_semantic_candidate import prepare, replay
from tests.source_review.test_semantic_candidate_worker import payload, replay as private_replay
from tests.source_review.test_semantic_reconnect_edits import SOURCE, CHAIN, SHARED
from tests.source_review.test_semantic_source_edits import apply
from tests.source_review.test_semantic_unary_edits import insertion


def proposal(base, source, port=1, name='gelu'):
    resolver = CapturedEditResolver(base, {'model.py': source})
    target = next(op for op in base.operations if op.target == 'aten.add.Tensor' and op.arguments.args[1].kind == 'value')
    cap = next(cap for cap in resolver.insert_unary_capabilities(target.id) if cap.argumentPath == ['args', port])
    edit = insertion(cap, name)
    return edit, apply(source, resolver.plan(edit)), cap


def candidate(root, source=SOURCE, port=1, name='gelu'):
    plan, base = prepare(root / 'original', source)
    edit, after, cap = proposal(base, source, port, name)
    with stage_candidate(str(root / 'original'), plan, base, edit, 'model.py', after) as staged:
        process, envelope = replay(staged.workspace, staged.execution.plan, root)
        assert process.returncode == 0, process.stderr
        return staged, envelope['result']['diagnostics']['semantic_capture'], cap


@pytest.fixture(scope='module')
def observed(tmp_path_factory):
    return candidate(tmp_path_factory.mktemp('candidate-insertion'), SHARED)


def test_inserted_operation_feeds_every_observed_shared_operand(observed):
    staged, capture, cap = observed
    result = check_postconditions(staged.execution, staged.postconditions, capture)
    assert result.status == 'verified' and result.reason == 'observed_requested_insertion'
    assert result.recipe == cap.recipe and result.scope == 'observed_expression'
    assert result.observedOperations == 2 and result.bindingId == staged.postconditions.expression.id


@pytest.mark.parametrize('port', [0, 1])
@pytest.mark.parametrize('name', ['relu', 'gelu', 'silu', 'sigmoid', 'tanh'])
def test_each_supported_activation_feeds_only_the_selected_operand(tmp_path, port, name):
    staged, capture, cap = candidate(tmp_path, CHAIN, port, name)
    result = check_postconditions(staged.execution, staged.postconditions, capture)
    assert result.status == 'verified' and result.observedOperations == 1
    parent = next(op for op in capture['operations'] if result.bindingId in op['sourceBindingIds'])
    unary = next(op for op in capture['operations'] if op['output'] == parent['arguments']['args'][port])
    assert unary['target'] == 'aten.' + name + '.default'
    assert unary['arguments']['args'][0] == parent['arguments']['args'][1 - port]


@pytest.mark.parametrize('ending', ['\n', '\r\n'])
def test_required_import_unicode_and_multiline_parent_remain_exact(tmp_path, ending):
    source = SOURCE.replace('import torch', 'from torch.nn import Module').replace('torch.nn.Module', 'Module')
    source = source.replace('b = a + a', "label = '🚀'; b = (a +\n             a)").replace('\n', ending)
    staged, capture, cap = candidate(tmp_path, source)
    assert cap.importPlan is not None
    result = check_postconditions(staged.execution, staged.postconditions, capture)
    assert result.status == 'verified'
    assert staged.postconditions.expression.anchors[0].startLine == cap.expression.anchors[0].startLine + 1


def test_existing_activation_on_other_operand_cannot_supply_insertion_proof(tmp_path):
    source = SOURCE.replace('b = a + a', 'b = a + torch.nn.functional.gelu(a)')
    staged, original, _ = candidate(tmp_path, source, port=0)
    assert check_postconditions(staged.execution, staged.postconditions, original).status == 'verified'
    capture = deepcopy(original)
    parent = next(op for op in capture['operations'] if staged.postconditions.expression.id in op['sourceBindingIds'])
    parent['arguments']['args'][0] = deepcopy(parent['arguments']['args'][1])
    result = check_postconditions(staged.execution, staged.postconditions, capture)
    assert result.status == 'failed' and result.reason == 'observed_insertion_mismatch'


def test_successful_operation_that_does_not_feed_the_selected_use_fails(observed):
    staged, original, _ = observed
    capture = deepcopy(original)
    parent = [op for op in capture['operations'] if staged.postconditions.expression.id in op['sourceBindingIds']][-1]
    unary = next(op for op in capture['operations'] if op['output'] == parent['arguments']['args'][1])
    parent['arguments']['args'][1] = deepcopy(unary['arguments']['args'][0])
    result = check_postconditions(staged.execution, staged.postconditions, capture)
    assert result.status == 'failed' and result.reason == 'observed_insertion_mismatch'
    assert result.observedOperations == 2


@pytest.mark.parametrize('change', ['kernel', 'input', 'other_operand', 'binding', 'partial', 'parent_binding', 'duplicate'])
def test_incomplete_or_ambiguous_inserted_flow_never_passes(observed, change):
    staged, original, _ = observed
    capture = deepcopy(original)
    parent = next(op for op in capture['operations'] if staged.postconditions.expression.id in op['sourceBindingIds'])
    unary = next(op for op in capture['operations'] if op['output'] == parent['arguments']['args'][1])
    if change == 'kernel': unary['target'] = 'aten.relu.default'
    if change == 'input': unary['arguments']['args'][0] = deepcopy(capture['inputs']['args'][0])
    if change == 'other_operand': parent['arguments']['args'][0] = deepcopy(parent['arguments']['args'][1])
    if change == 'binding': unary['sourceBindingIds'] = []
    if change == 'partial': capture['status'] = 'partial'
    if change == 'parent_binding': parent['sourceBindingIds'] = []
    if change == 'duplicate':
        alternate = deepcopy(next(binding for binding in capture['sourceBindings'] if binding['id'] == unary['sourceBindingIds'][0]))
        alternate['id'] = 'alternate'; capture['sourceBindings'].append(alternate); unary['sourceBindingIds'] = ['alternate']
    result = check_postconditions(staged.execution, staged.postconditions, capture)
    assert result.status == 'unavailable' and result.observedOperations == 0


def test_runtime_identifiers_do_not_establish_inserted_operation_correspondence(observed):
    staged, capture, _ = observed
    identifiers = {item['id']: 'changed:' + item['id'] for kind in
        ['operations', 'invocations', 'modules', 'definitions', 'values'] for item in capture[kind]}
    def rename(value):
        if isinstance(value, dict): return {key: rename(item) for key, item in value.items()}
        if isinstance(value, list): return [rename(item) for item in value]
        return identifiers.get(value, value) if isinstance(value, str) else value
    assert check_postconditions(staged.execution, staged.postconditions, rename(capture)).status == 'verified'


def test_worker_publishes_scoped_insertion_without_touching_live_source(tmp_path):
    plan, base = prepare(tmp_path / 'original')
    edit, after, _ = proposal(base, SOURCE)
    process, envelope = private_replay((tmp_path / 'original').resolve(), payload(plan, base, edit, after), tmp_path)
    assert process.returncode == 0, process.stderr
    result = envelope['worker']['candidatePostconditions']
    assert result['status'] == 'verified' and result['reason'] == 'observed_requested_insertion'
    assert (tmp_path / 'original' / 'model.py').read_text() == SOURCE
