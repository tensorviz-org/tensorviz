"""Unknown forward contracts must not run an invented zero-argument invocation."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from services.ingest.inputs import required_arity
from services.ingest.service import _introspect_and_trace
from services.validation.setup import ExecutionRequirement


class Zero(torch.nn.Module):
    def forward(this):
        return torch.ones(1)


class Variadic(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("must not guess a call")


class Keyword(torch.nn.Module):
    def forward(self, *, tokens):
        return tokens


class Positional(torch.nn.Module):
    def forward(this, x, y=None, *, scale=1):
        return x


class Four(torch.nn.Module):
    def forward(self, a, b, c, d):
        raise AssertionError("automatic search cannot supply four inputs")


def test_arity_distinguishes_known_zero_from_unknown_and_named_inputs():
    assert required_arity(Zero()) == 0
    assert required_arity(Variadic()) is None
    assert required_arity(Keyword()) is None
    assert required_arity(Positional()) == 1
    assert required_arity(Four()) == 4


def test_more_required_inputs_than_the_search_supports_need_an_explicit_recipe():
    with pytest.raises(ExecutionRequirement) as error:
        _introspect_and_trace('model', Four(), 2, None, {}, tier='real')
    assert error.value.diagnostic['code'] == 'missing_argument'


@pytest.mark.parametrize('model', [Variadic, Keyword])
def test_unknown_signature_does_not_execute_a_guessed_forward(model):
    with pytest.raises(ExecutionRequirement) as error:
        _introspect_and_trace('model', model(), 2, None, {}, tier='real')
    assert error.value.diagnostic['code'] == 'unknown_signature'
    assert error.value.diagnostic['field'] == ['inputs']
    assert error.value.diagnostic['remedy'] == 'provide_inputs'


def test_failed_search_prefix_never_claims_that_its_input_succeeded(monkeypatch):
    from services.ingest import inputs
    class Broken(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 4)
        def forward(self, x):
            h = self.fc(x)
            raise ValueError('private error after a real partial prefix')
    monkeypatch.setattr(inputs, 'MAX_CANDIDATES', 1)
    graph, diagnostics, error = _introspect_and_trace('model', Broken(), 2, None, {}, tier='real')
    assert error is not None
    assert diagnostics.input_provenance == 'failed-candidate'
    assert any(node['data']['parameters'].get('_input_provenance') == 'failed-candidate' for node in graph['nodes'])


def test_failed_search_without_hook_evidence_cannot_become_success(monkeypatch):
    from services.ingest import inputs
    class EarlyFailure(torch.nn.Module):
        def __call__(self, x):
            raise RuntimeError('failure before eager module hooks')
        def forward(self, x):
            raise AssertionError('not reached')
    monkeypatch.setattr(inputs, 'MAX_CANDIDATES', 1)
    _, diagnostics, error = _introspect_and_trace('model', EarlyFailure(), 2, None, {}, tier='real')
    assert error is not None
    assert diagnostics.tier == 'tree-only'
    assert diagnostics.input_provenance == 'failed-candidate'


@pytest.mark.parametrize('explicit', [False, True])
def test_variadic_worker_requires_recipe_and_then_uses_exact_inputs(tmp_path, explicit):
    file = tmp_path / 'variadic.py'
    file.write_text('''from pathlib import Path
import torch
class Model(torch.nn.Module):
    def forward(self, *args, **kwargs):
        Path(__file__).with_suffix('.called').touch()
        return args[0].sin()
''')
    args = [sys.executable, '-B', '-m', 'services.ingest.worker', '--file', str(file), '--cls', 'Model', '--json']
    if explicit:
        args += ['--execution', json.dumps({'inputs': {'args': [{'shape': [2, 4], 'dtype': 'float32', 'fill': 'ones'}]}}),
                 '--workspace', str(tmp_path)]
    process = subprocess.run(args, cwd=Path(__file__).parent.resolve().parents[1], capture_output=True, text=True, timeout=30)
    result = json.loads(process.stdout)
    assert process.returncode == (0 if explicit else 3), process.stderr
    assert file.with_suffix('.called').exists() == explicit
    if not explicit:
        assert result['error']['diagnostic']['code'] == 'unknown_signature'
        assert result['error']['diagnostic']['remedy'] == 'provide_inputs'
        assert result['result'] is None
