"""Facts about a failing operation and its actual contributing tensor paths."""
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn

from services.ingest.trace import trace_model

BACKEND = Path(__file__).parent.resolve().parent.parent


class ResidualMismatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.AvgPool2d(2)
        self.unreached = nn.Conv2d(1, 2, 1)

    def forward(self, x):
        short = self.down(x)
        combined = short + x
        return self.unreached(combined)


def test_failure_retains_both_shapes_source_and_contributing_path():
    result = trace_model(ResidualMismatch(), (torch.ones(1, 1, 56, 56),), tier='real',
                         frame_roots=(str(Path(__file__).parent.parent),), record_evidence=True)
    evidence = result.evidence
    failure = evidence['failure']
    assert failure['operation'] == 'aten.add.Tensor'
    assert failure['status'] == 'failed' and failure['outputs'] == []
    assert [t['shape'] for t in failure['inputs']] == [[1, 1, 28, 28], [1, 1, 56, 56]]
    assert failure['source']['file'] == __file__ and failure['source']['function'] == 'forward'
    assert 'short + x' in Path(__file__).read_text().splitlines()[failure['source']['line'] - 1]
    producer = next(op for op in evidence['operations'] if op['operationId'] == failure['inputs'][0]['producerOperationId'])
    assert producer['modulePath'] == 'down' and producer['shapeChanged']
    assert producer['inputs'][0]['inputPath'] == 'args[0]'
    assert failure['inputs'][1]['inputPath'] == 'args[0]'
    coverage = {m['modulePath']: m for m in evidence['coverage']}
    assert coverage['down']['reached'] and coverage['down']['completedCalls'] == 1
    assert not coverage['unreached']['reached']
    assert coverage['']['reached'] and coverage['']['completedCalls'] == 0
    # The failed op has no output and must not enter successful graph dataflow.
    assert not any(op['op'] == 'aten.add.Tensor' for op in result.ops)


def test_evidence_does_not_add_operations_or_change_outputs():
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    inputs = (torch.ones(2, 3),)
    plain = trace_model(model, inputs, tier='real')
    captured = trace_model(model, inputs, tier='real', record_evidence=True)
    assert captured.io == plain.io and captured.error == plain.error is None
    assert [op['op'] for op in captured.ops] == [op['op'] for op in plain.ops]
    assert all(c['status'] == 'completed' for c in captured.evidence['moduleCalls'])
    assert plain.evidence is None


def test_truncation_keeps_failure_and_does_not_invent_unreached_status(monkeypatch):
    import services.ingest.execution_evidence as module
    monkeypatch.setattr(module, 'MAX_OPERATIONS', 1)
    monkeypatch.setattr(module, 'MAX_MODULE_CALLS', 1)
    result = trace_model(ResidualMismatch(), (torch.ones(1, 1, 56, 56),), tier='real', record_evidence=True)
    evidence = result.evidence
    assert len(evidence['operations']) == 1
    assert evidence['failure']['operation'] == 'aten.add.Tensor'
    assert {'operations', 'module_inventory', 'module_calls'} <= set(evidence['truncated'])
    assert evidence['moduleCallCount'] == 2


def test_python_failure_is_not_falsely_attributed_to_the_last_successful_operation():
    class Bad(nn.Module):
        def forward(self, x):
            x = x + 1
            raise ValueError('example deliberately withheld from structured evidence')
    result = trace_model(Bad(), (torch.ones(2),), tier='real', record_evidence=True)
    assert result.error and result.error_frames
    assert result.evidence['failure'] is None
    assert result.evidence['operations'][-1]['status'] == 'completed'
    assert result.evidence['moduleCalls'][0]['status'] == 'failed'
    assert 'deliberately withheld' not in json.dumps(result.evidence)


def test_real_worker_serializes_partial_evidence_and_reproducible_setup(tmp_path):
    file = tmp_path / 'net.py'
    file.write_text('''import torch
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 2)
    def forward(self, x):
        return self.fc(x)
''')
    spec = {'inputs': {'args': [{'shape': [2, 4], 'dtype': 'float32', 'fill': 'ones'}]}}
    process = subprocess.run([sys.executable, '-m', 'services.ingest.worker', '--file', str(file),
        '--cls', 'Net', '--workspace', str(tmp_path), '--execution', json.dumps(spec)],
        cwd=BACKEND, env={**os.environ, 'KMP_DUPLICATE_LIB_OK': 'TRUE'}, capture_output=True, text=True, timeout=90)
    envelope = json.loads(process.stdout)
    assert process.returncode == 4
    diagnostics = envelope['result']['diagnostics']
    assert diagnostics['execution']['state']['digest']
    failure = diagnostics['run_evidence']['failure']
    assert failure['modulePath'] == 'fc' and failure['source']['file'] == str(file)
    assert any(t['shape'] == [2, 4] for t in failure['inputs'])


def test_same_shape_merge_is_not_reported_as_a_dimension_change():
    class Merge(nn.Module):
        def forward(self, x):
            return (x + x).relu()
    result = trace_model(Merge(), (torch.ones(2, 3),), tier='real', record_evidence=True)
    merge = next(op for op in result.evidence['operations'] if op['operation'] == 'aten.add.Tensor')
    assert merge['shapeChanged'] is False
    assert 'inputPath' not in merge['outputs'][0] and 'producerOperationId' not in merge['outputs'][0]
    relu = next(op for op in result.evidence['operations'] if op['operation'] == 'aten.relu.default')
    assert relu['inputs'][0]['producerOperationId'] == merge['operationId']


def test_caught_child_failure_unwinds_before_fallback_and_parent_operations():
    class Bad(nn.Module):
        def forward(self, x):
            raise ValueError('handled')
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.bad = Bad()
            self.fc = nn.Linear(3, 2)
        def forward(self, x):
            try:
                self.bad(x)
            except ValueError:
                x = self.fc(x)  # Successful call while sys.exc_info() is active.
            return x + 1
    model = Net()
    result = trace_model(model, (torch.ones(1, 3),), tier='real', record_evidence=True)
    calls = {c['modulePath']: c for c in result.evidence['moduleCalls']}
    assert result.error is None
    assert calls['bad']['status'] == 'failed' and calls['bad']['outputs'] == []
    assert calls['bad']['disposition'] == 'recovered'
    assert calls['fc']['status'] == calls['']['status'] == 'completed'
    assert result.evidence['operations'][-1]['modulePath'] == ''
    assert result.io['bad']['out'] is None and result.io['']['out'] == [1, 2]
    assert all('_call_impl' not in m.__dict__ for m in model.modules())


def test_recursive_global_hook_failure_and_none_output_remain_distinct():
    class Recursive(nn.Module):
        def forward(self, x, depth=0):
            if depth == 0:
                try:
                    self(x, 1)
                except ValueError:
                    pass
            return None
    model = Recursive()
    def reject_child(mod, args):
        if len(args) == 2:
            raise ValueError('global pre-hook')
    hook = nn.modules.module.register_module_forward_pre_hook(reject_child)
    try:
        result = trace_model(model, (torch.ones(1),), tier='real', record_evidence=True)
    finally:
        hook.remove()
    calls = result.evidence['moduleCalls']
    assert [c['status'] for c in calls] == ['completed', 'failed']
    assert calls[0]['outputs'] == calls[1]['outputs'] == []
    assert result.evidence['coverage'][0]['completedCalls'] == 1
    assert result.evidence['coverage'][0]['failedCalls'] == 1
    assert '_call_impl' not in model.__dict__


def test_recovered_tensor_error_is_not_a_terminal_failure():
    class Net(nn.Module):
        def forward(self, x):
            try:
                x + torch.ones(4)
            except RuntimeError:
                return x.relu()
    result = trace_model(Net(), (torch.ones(2, 3),), tier='real', record_evidence=True)
    assert result.error is None and result.evidence['failure'] is None
    observed = result.evidence['lastObservedFailure']
    assert observed['disposition'] == 'recovered' and observed['outputs'] == []
    assert result.evidence['moduleCalls'][0]['status'] == 'completed'


def test_worker_does_not_blame_the_unreached_layer_after_a_functional_failure(tmp_path):
    file = tmp_path / 'net.py'
    file.write_text("import torch\nfrom torch import nn\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.down = nn.AvgPool2d(2)\n        self.tail = nn.Conv2d(1, 2, 1)\n    def forward(self, x):\n        return self.tail(self.down(x) + x)\n")
    process = subprocess.run([sys.executable, '-m', 'services.ingest.worker', '--file', str(file),
        '--cls', 'Net', '--tier', 'real', '--workspace', str(tmp_path),
        '--execution', json.dumps({'inputs': {'args': [{'shape': [1, 1, 56, 56], 'dtype': 'float32', 'fill': 'ones'}]}})], cwd=BACKEND,
        env={**os.environ, 'KMP_DUPLICATE_LIB_OK': 'TRUE'}, capture_output=True, text=True, timeout=90)
    envelope = json.loads(process.stdout)
    assert process.returncode == 4
    assert envelope['error']['node_id'] is None
    evidence = envelope['result']['diagnostics']['run_evidence']
    assert evidence['failure']['operation'] == 'aten.add.Tensor'
    assert not next(row for row in evidence['coverage'] if row['modulePath'] == 'tail')['reached']
