"""Numerical evidence survives in-place changes without rounding integer tensors."""
import base64
import struct

import torch
from torch import nn

from services.ingest.execution_values import capture_output
from services.ingest.trace import trace_model


def test_retained_values_do_not_alias_mutable_outputs():
    tensor = torch.tensor([[1., 2.], [3., 4.]])
    captured = capture_output({'logits': tensor})
    before = captured['leaves'][0]
    tensor.add_(10)
    assert struct.unpack('<4f', base64.b64decode(before['values']['data'])) == (1., 2., 3., 4.)
    assert capture_output({'logits': tensor})['digest'] != captured['digest']
    assert captured['allFinite'] is True


def test_int64_and_complex_values_are_not_coerced_to_json_numbers():
    captured = capture_output((torch.tensor([2**60 + 7], dtype=torch.int64), torch.tensor([1 + 2j])))
    integer, complex_value = captured['leaves']
    assert struct.unpack('<q', base64.b64decode(integer['values']['data']))[0] == 2**60 + 7
    assert struct.unpack('<2f', base64.b64decode(complex_value['values']['data'])) == (1., 2.)
    assert integer['dtype'] == 'int64' and complex_value['dtype'] == 'complex64'


def test_nonfinite_outputs_and_value_budget_are_explicit():
    captured = capture_output(torch.tensor([float('nan'), float('inf'), 3.]), max_bytes=0)
    leaf = captured['leaves'][0]
    assert leaf['digest'] and captured['digest']
    assert leaf['nonfinite'] == 2 and captured['allFinite'] is False
    assert leaf['values'] is None and leaf['unavailable'] == 'output_value_budget'
    limited = capture_output(torch.ones(10), max_bytes=4, include_values=False)
    assert limited['digest'] is None and limited['leaves'][0]['unavailable'] == 'module_digest_budget'


def test_first_observed_values_follow_module_completion_order():
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(inplace=True), nn.Linear(4, 2))
    x = torch.ones(2, 3)
    plain = trace_model(model, (x,), tier='real')
    captured = trace_model(model, (x,), tier='real', record_evidence=True)
    assert [op['op'] for op in captured.ops] == [op['op'] for op in plain.ops]
    calls = sorted(captured.evidence['moduleCalls'], key=lambda c: c['completedIndex'])
    assert [call['modulePath'] for call in calls] == ['0', '1', '2', '']
    assert all(call['values']['digest'] for call in calls)
    assert captured.evidence['outputs']['leaves'][0]['shape'] == [2, 2]


def test_scalar_and_container_contracts_are_preserved():
    captured = capture_output({'value': 2**80 + 1, 'none': None, 'nested': [True, 3.5]})
    assert captured['leaves'][0]['value'] == str(2**80 + 1)
    assert captured['containers'][0]['keys'] == ['value', 'none', 'nested']
    assert capture_output({'a': 1, 'b': 2})['digest'] != capture_output({'b': 2, 'a': 1})['digest']
    unavailable = capture_output(object())
    assert unavailable['digest'] is None and unavailable['allFinite'] is None


def test_byte_capture_does_not_require_numpy(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Tensor.numpy must not be needed to inspect a run')
    monkeypatch.setattr(torch.Tensor, 'numpy', forbidden)
    captured = capture_output(torch.tensor([1.5, 2.5]))
    assert captured['digest'] and captured['allFinite']
    assert struct.unpack('<2f', base64.b64decode(captured['leaves'][0]['values']['data'])) == (1.5, 2.5)


def test_measurement_errors_do_not_turn_a_successful_forward_into_a_failed_model(monkeypatch):
    import services.ingest.execution_values as values
    def broken(tensor):
        raise ValueError('private measurement details')
    monkeypatch.setattr(values, 'tensor_digest', broken)
    model = nn.Linear(3, 2)
    captured = trace_model(model, (torch.ones(2, 3),), tier='real', record_evidence=True)
    assert captured.error is None
    outputs = captured.evidence['outputs']
    assert outputs['digest'] is None and outputs['allFinite'] is None
    assert outputs['leaves'][0]['unavailable'] == 'ValueError'
    assert 'private' not in str(outputs)
