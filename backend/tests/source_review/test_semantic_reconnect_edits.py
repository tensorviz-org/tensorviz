"""Reconnection selects one use of a proven available tensor, not a producer set."""
from copy import deepcopy

import pytest
from pydantic import ValidationError

from services.semantic.contracts import ReconnectArgumentCapability, ReconnectArgumentEdit
from services.semantic.source_edits import CapturedEditResolver, SourceEditRefused
from tests.source_review.test_semantic_source_edits import SOURCE, setup, apply


def capability(capture, resolver, index=1, port=1):
    return next(cap for cap in resolver.reconnect_capabilities(capture.operations[index].id) if cap.argumentPath == ['args', port])


def intent(cap, name='h'):
    return dict(kind='reconnect_argument', capabilityId=cap.id, target=cap.target.model_dump(mode='json'),
        argumentPath=cap.argumentPath, expected=cap.expected.model_dump(mode='json'),
        value=next(choice.value.model_dump(mode='json') for choice in cap.choices if choice.name == name), scope='source_expression')


def test_repeated_operand_reconnects_only_the_selected_port_and_recaptures_it(tmp_path):
    capture, resolver = setup(tmp_path)
    left, right = resolver.reconnect_capabilities(capture.operations[1].id)
    assert left.argumentPath == ['args', 0] and right.argumentPath == ['args', 1]
    assert left.expected == right.expected and left.id != right.id
    assert [choice.name for choice in right.choices] == ['a', 'h']
    assert right.scope.observedOperations == right.scope.observedInvocations == right.scope.observedModuleInstances == 1
    plan = resolver.plan(intent(right))
    assert plan['validation'] == 'static_only' and len(plan['edits']) == 1
    after = apply(SOURCE, plan)
    assert after == SOURCE.replace('b = a + a', 'b = a + h')
    changed, _ = setup(tmp_path, after)
    assert changed.operations[1].arguments.args == [capture.operations[1].arguments.args[0], capture.inputs.args[0]]
    assert changed.operations[0].arguments == capture.operations[0].arguments
    assert changed.operations[2].arguments == capture.operations[2].arguments
    assert resolver.plan(intent(right, 'a'))['edits'] == []


def test_publication_keeps_literal_edit_and_port_specific_reconnections(tmp_path):
    capture, resolver = setup(tmp_path)
    mixed = resolver.capabilities(capture.operations[0].id)
    assert [cap.command for cap in mixed] == ['set_literal', 'reconnect_argument', 'insert_unary_op']
    assert mixed[0].argumentPath == ['args', 1] and mixed[1].argumentPath == ['args', 0]
    repeated = resolver.capabilities(capture.operations[1].id)
    assert [cap.argumentPath for cap in repeated] == [['args', 0], ['args', 1], ['args', 0], ['args', 1]]
    assert [cap.command for cap in repeated] == ['reconnect_argument', 'reconnect_argument', 'insert_unary_op', 'insert_unary_op']


def test_publication_preserves_flow_refusal_for_unsupported_control_flow(tmp_path):
    capture, resolver = setup(tmp_path, SOURCE.replace('b = a + a', 'if True:\n            b = a + a'))
    # Literal editing remains available even when the stronger flow proof refuses.
    assert [cap.command for cap in resolver.capabilities(capture.operations[0].id)] == ['set_literal']
    with pytest.raises(SourceEditRefused, match='unsupported_source_scope'):
        resolver.capabilities(capture.operations[1].id)


CHAIN = '''import torch
from torch import nn
from torch.nn import functional as F
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(4, 4)
        self.last = nn.Linear(4, 2)
    def forward(self, x):
        h = self.first(x)
        h = F.gelu(input=h)
        a = h + 1
        b = a + a
        return self.last(b)
'''


def test_module_outputs_and_reassigned_names_use_the_latest_source_value(tmp_path):
    capture, resolver = setup(tmp_path, CHAIN)
    target = next(op for op in capture.operations if op.target == 'aten.add.Tensor' and op.arguments.args[0] == op.arguments.args[1])
    cap = resolver.reconnect_capabilities(target.id)[1]
    assert [choice.name for choice in cap.choices] == ['a', 'x', 'h']
    h = next(choice for choice in cap.choices if choice.name == 'h')
    gelu = next(op for op in capture.operations if op.target == 'aten.gelu.default')
    assert h.value == gelu.output and h.value != capture.invocations[1].output
    assert h.binding.anchors[0].startLine == 11
    assert capture.invocations[1].output not in [choice.value for choice in cap.choices]
    after = apply(CHAIN, resolver.plan(intent(cap)))
    assert after == CHAIN.replace('b = a + a', 'b = a + h')
    changed, _ = setup(tmp_path, after)
    assert changed.status == 'complete'
    assert next(op for op in changed.operations if op.id == target.id).arguments.args[1] == gelu.output


def test_available_values_can_legitimately_fail_shapes_without_implicit_repairs(tmp_path):
    from tests.source_review.test_semantic_constructor_edits import setup as capture_attempt
    source = CHAIN.replace('self.first = nn.Linear(4, 4)', 'self.first = nn.Linear(4, 2)').replace(
        'self.last = nn.Linear(4, 2)', 'self.last = nn.Linear(2, 2)')
    capture, resolver = setup(tmp_path, source)
    target = next(op for op in capture.operations if op.target == 'aten.add.Tensor' and op.arguments.args[0] == op.arguments.args[1])
    cap = resolver.reconnect_capabilities(target.id)[1]
    plan = resolver.plan(intent(cap, 'x'))
    after = apply(source, plan)
    assert plan['validation'] == 'static_only' and after == source.replace('b = a + a', 'b = a + x')
    changed, _ = capture_attempt(tmp_path, after)
    assert changed.status == 'partial' and any(op.status == 'failed' for op in changed.operations)


SHARED = '''import torch
class Block(torch.nn.Module):
    def forward(self, h):
        a = h + 1
        b = a + a
        return b * 2
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b = Block(), Block()
    def forward(self, h):
        return self.b(self.a(h))
'''


def test_shared_forward_definition_changes_all_observed_instances_with_explicit_scope(tmp_path):
    capture, resolver = setup(tmp_path, SHARED)
    cap = capability(capture, resolver)
    assert cap.scope.observedOperations == cap.scope.observedInvocations == cap.scope.observedModuleInstances == 2
    assert cap.scope.includesUnobservedExecutions
    after = apply(SHARED, resolver.plan(intent(cap)))
    changed, _ = setup(tmp_path, after)
    for call in changed.invocations[1:]:
        ops = [op for op in changed.operations if op.invocationId == call.id]
        assert ops[1].arguments.args == [ops[0].output, call.arguments.args[0]]


def test_an_uninstrumented_direct_forward_use_prevents_a_partial_scope_claim(tmp_path):
    source = SHARED.replace('return self.b(self.a(h))', 'return self.b.forward(self.a(h))')
    capture, resolver = setup(tmp_path, source)
    with pytest.raises(SourceEditRefused, match='ambiguous_source_scope'):
        capability(capture, resolver)


def test_keyword_inputs_are_bound_by_names_instead_of_dict_or_forward_port_order(tmp_path):
    source = SHARED.replace('def forward(self, h):\n        a = h + 1', 'def forward(self, h, *, other):\n        a = h + other', 1)
    source = source.replace('return self.b(self.a(h))', 'return self.a(other=h + 2, h=h)')
    capture, resolver = setup(tmp_path, source)
    target = next(op for op in capture.operations if op.target == 'aten.add.Tensor' and op.arguments.args[0] == op.arguments.args[1])
    cap = resolver.reconnect_capabilities(target.id)[1]
    assert [choice.name for choice in cap.choices] == ['a', 'h', 'other']
    call = capture.invocations[1]
    assert next(choice.value for choice in cap.choices if choice.name == 'other') == call.arguments.kwargs['other']
    after = apply(source, resolver.plan(intent(cap, 'other')))
    assert after == source.replace('b = a + a', 'b = a + other')
    assert setup(tmp_path, after)[0].status == 'complete'


def test_aliases_do_not_create_duplicate_values_or_rewrite_an_unchanged_operand(tmp_path):
    source = SOURCE.replace('b = a + a', 'alias = a\n        b = alias + a')
    capture, resolver = setup(tmp_path, source)
    caps = resolver.reconnect_capabilities(capture.operations[1].id)
    assert [choice.name for choice in caps[0].choices] == ['alias', 'h']
    assert [choice.name for choice in caps[1].choices] == ['a', 'h']
    assert resolver.plan(intent(caps[0], 'alias'))['edits'] == []
    assert apply(source, resolver.plan(intent(caps[1]))) == source.replace('b = alias + a', 'b = alias + h')


def test_later_results_and_overwritten_values_are_not_reconnection_choices(tmp_path):
    source = SOURCE.replace('return b * 2', 'later = h + 2\n        return b * later')
    capture, resolver = setup(tmp_path, source)
    cap = capability(capture, resolver)
    assert 'b' not in [choice.name for choice in cap.choices] and 'later' not in [choice.name for choice in cap.choices]
    payload = intent(cap)
    payload['value'] = capture.operations[2].output.model_dump(mode='json')
    with pytest.raises(SourceEditRefused, match='unavailable_source_value'):
        resolver.plan(payload)


@pytest.mark.parametrize('source', [
    SOURCE.replace('b = a + a', 'if True:\n            b = a + a'),
    SOURCE.replace('b = a + a', 'for _ in range(1):\n            b = a + a'),
    SOURCE.replace('return b * 2', 'if True:\n            return b * 2'),
    SOURCE.replace('b = a + a', 'try:\n            b = a + a\n        finally:\n            pass'),
    SOURCE.replace('b = a + a', 'with torch.no_grad():\n            b = a + a'),
    SOURCE.replace('b = a + a', 'self.temp = a\n        b = a + a'),
    SOURCE.replace('b = a + a', 'alias, other = a, a\n        b = a + a'),
    SOURCE.replace('b = a + a', 'a += 1\n        b = a + a'),
    SOURCE.replace('a = h + 1', 'a = h.relu()'),
    SOURCE.replace('a = h + 1', 'a = sum([h for _ in range(1)])'),
    SOURCE.replace('def forward(self, h):', 'def forward(self, h, unused=None):'),
])
def test_control_flow_ambiguous_bindings_and_unknown_calls_refuse_instead_of_guessing(tmp_path, source):
    capture, resolver = setup(tmp_path, source)
    target = next(op for op in capture.operations if op.target == 'aten.add.Tensor' and op.arguments.args[0] == op.arguments.args[1])
    with pytest.raises(SourceEditRefused):
        resolver.reconnect_capabilities(target.id)


def test_mutation_anywhere_in_the_scope_prevents_stale_available_value_claims(tmp_path):
    source = SOURCE.replace('return b * 2', 'v = torch.nn.functional.relu(h, inplace=True)\n        return b * 2')
    capture, resolver = setup(tmp_path, source)
    with pytest.raises(SourceEditRefused, match='mutating_source_scope'):
        capability(capture, resolver)


def test_scalar_bindings_are_checked_but_never_offered_as_tensor_values(tmp_path):
    source = SOURCE.replace('a = h + 1', 'scale = 1\n        a = h + scale')
    capture, resolver = setup(tmp_path, source)
    cap = capability(capture, resolver)
    assert [choice.name for choice in cap.choices] == ['a', 'h']


def test_unicode_and_crlf_use_exact_operand_ranges_and_noop_keeps_spelling(tmp_path):
    source = SOURCE.replace('self, h', 'self, 输入').replace('h + 1', '输入 + 1')
    source = source.replace('a =', '变量 =').replace('a + a', '变量 + 变量').replace('\n', '\r\n')
    capture, resolver = setup(tmp_path, source)
    cap = capability(capture, resolver)
    assert cap.expectedName == '变量'
    assert resolver.plan(intent(cap, '变量'))['edits'] == []
    assert apply(source, resolver.plan(intent(cap, '输入'))) == source.replace('变量 + 变量', '变量 + 输入')


@pytest.mark.parametrize('change', ['port', 'value', 'capture', 'scope', 'name'])
def test_forged_intents_cannot_change_other_ports_or_inject_source_names(tmp_path, change):
    capture, resolver = setup(tmp_path)
    cap = capability(capture, resolver)
    payload = intent(cap)
    if change == 'port':
        payload['argumentPath'] = ['args', 0]
    elif change == 'value':
        payload['expected'] = capture.inputs.args[0].model_dump(mode='json')
    elif change == 'capture':
        payload['target']['captureId'] = 'other'
    elif change == 'scope':
        payload['scope'] = 'selected_call'
    else:
        payload['name'] = 'danger()'
    with pytest.raises((ValidationError, SourceEditRefused)):
        resolver.plan(payload)


def test_source_value_contracts_reject_nonlocal_and_future_definition_choices(tmp_path):
    capture, resolver = setup(tmp_path)
    cap = capability(capture, resolver).model_dump(mode='json')
    for field, value in [('file', 'other.py'), ('scope', 'Other.forward'), ('endLine', 99)]:
        payload = deepcopy(cap)
        payload['choices'][0]['binding']['anchors'][0][field] = value
        with pytest.raises(ValidationError):
            ReconnectArgumentCapability.model_validate(payload)
    payload = intent(capability(capture, resolver))
    for argumentPath in [['args', True], ['args', 2], ['kwargs', 'input'], ['args', 1, 0]]:
        with pytest.raises(ValidationError):
            ReconnectArgumentEdit.model_validate({**payload, 'argumentPath': argumentPath})


def test_stale_source_and_unmapped_or_changed_parameters_cannot_grant_reconnection(tmp_path):
    capture, resolver = setup(tmp_path)
    with pytest.raises(SourceEditRefused, match='stale_source'):
        CapturedEditResolver(capture, {'model.py': SOURCE.replace('h + 1', 'h + 2')})
    payload = capture.model_dump(mode='json')
    payload['invocations'][0]['arguments']['args'][0] = {'kind': 'int', 'value': '4'}
    with pytest.raises((ValidationError, SourceEditRefused)):
        capability(capture, CapturedEditResolver(payload, {'model.py': SOURCE}))


def test_availability_budgets_fail_explicitly_without_dropping_shared_calls(tmp_path, monkeypatch):
    capture, resolver = setup(tmp_path, SHARED)
    monkeypatch.setattr('services.semantic.source_flow.MAX_CALLS', 1)
    with pytest.raises(SourceEditRefused, match='incomplete_source_flow'):
        capability(capture, resolver)
