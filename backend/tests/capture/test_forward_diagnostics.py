"""Only the actual failing operator may supply expected tensor contracts."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from services.ingest.trace import trace_model
from services.semantic.contracts import Diagnostic


class Call(torch.nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function
    def forward(self, x):
        return self.function(x)


@pytest.mark.parametrize('function,value,code,expected,actual,field', [
    (lambda x: x @ torch.ones(5, 2), torch.ones(2, 3), 'invalid_shape', [None, 5], [2, 3], ['args', 0]),
    (lambda x: torch.addmm(torch.ones(2), x, torch.ones(5, 2)), torch.ones(2, 3),
     'invalid_shape', [None, 5], [2, 3], ['args', 1]),
    (lambda x: torch.bmm(x, torch.ones(4, 5, 2)), torch.ones(2, 3, 5),
     'invalid_shape', [4, None, 5], [2, 3, 5], ['args', 0]),
    (lambda x: torch.nn.functional.conv2d(x, torch.ones(2, 4, 3, 3)), torch.ones(1, 3, 8, 8),
     'invalid_shape', [None, 4, None, None], [1, 3, 8, 8], ['args', 0]),
    (lambda x: torch.nn.functional.conv_transpose2d(x, torch.ones(4, 2, 3, 3)), torch.ones(1, 3, 8, 8),
     'invalid_shape', [None, 4, None, None], [1, 3, 8, 8], ['args', 0]),
    (lambda x: torch.nn.functional.embedding(x, torch.ones(5, 3)), torch.ones(2, 3),
     'invalid_dtype', None, None, ['args', 1]),
    (lambda x: x @ torch.ones(3, 2, dtype=torch.float64), torch.ones(2, 3),
     'invalid_dtype', None, None, ['args', 0]),
])
def test_actual_failed_operator_supplies_bounded_contract(function, value, code, expected, actual, field):
    traced = trace_model(Call(function), (value,), tier='real', record_evidence=True)
    assert traced.error is not None
    diagnostic = Diagnostic.model_validate(traced.diagnostic)
    assert (diagnostic.code, diagnostic.expectedShape, diagnostic.actualShape, diagnostic.field) == (code, expected, actual, field)
    assert traced.evidence['failure']['diagnostic'] == traced.diagnostic
    if code == 'invalid_dtype':
        assert diagnostic.actualDtype == 'torch.float32'


def test_error_text_cannot_invent_dimensions():
    def operation(x):
        raise RuntimeError('mat1 shape 999 must be 123; PRIVATE_PAYLOAD')
    traced = trace_model(Call(operation), (torch.ones(2, 3),), tier='real')
    assert traced.diagnostic['code'] == 'execution_failed'
    assert traced.diagnostic['expectedShape'] is None
    assert 'PRIVATE' not in json.dumps(traced.diagnostic)


def test_caught_failure_is_not_the_later_terminal_assertion():
    def operation(x):
        try:
            x @ torch.ones(5, 2)
        except RuntimeError:
            pass
        assert False, 'different assertion'
    traced = trace_model(Call(operation), (torch.ones(2, 3),), tier='real', record_evidence=True)
    assert traced.diagnostic['code'] == 'assertion_failed'
    assert traced.diagnostic['expectedShape'] is None
    assert traced.evidence['failure'] is None


def test_wrapped_failure_keeps_shape_only_with_explicit_exception_link():
    def operation(x):
        try:
            x @ torch.ones(5, 2)
        except RuntimeError as error:
            raise ValueError('wrapped') from error
    traced = trace_model(Call(operation), (torch.ones(2, 3),), tier='real')
    assert traced.diagnostic['code'] == 'invalid_shape'


def test_successful_recovery_has_no_terminal_diagnostic():
    def operation(x):
        try:
            x @ torch.ones(5, 2)
        except RuntimeError:
            return x.sin()
    traced = trace_model(Call(operation), (torch.ones(2, 3),), tier='real')
    assert traced.error is None and traced.diagnostic is None


def test_suppressed_exception_context_does_not_certify_a_shape_explanation():
    def operation(x):
        try:
            x @ torch.ones(5, 2)
        except RuntimeError:
            raise ValueError('replacement error') from None
    traced = trace_model(Call(operation), (torch.ones(2, 3),), tier='real')
    assert traced.diagnostic['code'] == 'execution_failed'


@pytest.mark.parametrize("planned", [False, True])
def test_worker_envelope_keeps_shape_failure_and_partial_capture(tmp_path, planned):
    file = tmp_path / 'shape_net.py'
    file.write_text('''import torch
class Net(torch.nn.Module):
    def forward(self, x):
        return x.sin() @ torch.ones(5, 2)
''')
    spec = {'inputs': {'args': [{'shape': [2, 3], 'dtype': 'float32', 'fill': 'ones'}]}}
    invocation = ['--execution', json.dumps(spec)]
    environment = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
    if planned:
        from services.validation.host_snapshot import snapshot
        from services.semantic.contracts import ResolvedExecutionPlan
        from services.semantic.execution_plan import applicability_digest
        plan = snapshot(dict(root=str(tmp_path), executable=sys.executable, environmentRevision=0,
            session='diagnostic-test', workspaceId='diagnostic-workspace', runId='diagnostic-run', generation=0,
            modelFile=file.name, className='Net', execution=spec, timeoutMs=15000))['plan']
        private = tmp_path / 'plan.json'
        private.write_text(json.dumps(dict(version=1, plan=plan,
            applicabilityDigest=applicability_digest(ResolvedExecutionPlan.model_validate(plan)))))
        private.chmod(0o600)
        environment['TENSORVIZ_EXECUTION_PLAN'] = str(private)
        invocation = ['--resolved-execution']
    process = subprocess.run([sys.executable, '-B', '-m', 'services.ingest.worker', '--file', str(file),
        '--cls', 'Net', '--workspace', str(tmp_path), *invocation], env=environment,
        cwd=Path(__file__).parent.resolve().parents[1], capture_output=True, text=True, timeout=30)
    envelope = json.loads(process.stdout)
    assert process.returncode == 4, envelope
    assert envelope['result'] is not None
    assert envelope['error']['diagnostic']['expectedShape'] == [None, 5]
    assert envelope['result']['diagnostics']['run_evidence']['failure']['diagnostic'] == envelope['error']['diagnostic']
    if planned:
        capture = envelope['result']['diagnostics']['semantic_capture']
        assert capture['status'] == 'partial'
        operation = next(op for op in capture['operations'] if op['status'] == 'failed')
        diagnostic = next(d for d in capture['diagnostics'] if d['id'] == operation['diagnosticId'])
        assert capture['invocations'][0]['diagnosticId'] == diagnostic['id']
        assert {k: v for k, v in diagnostic.items() if k != 'id'} == {
            k: v for k, v in envelope['error']['diagnostic'].items() if k != 'id'}
