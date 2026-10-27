"""Setup refuses exact bad requests without guessing or leaking exception prose."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from services.semantic.contracts import Diagnostic
from services.validation.setup import bind_call, ExecutionRequirement, setup_stage


def signature(x, /, y, *, tokens):
    raise AssertionError("binding must never call the function")


@pytest.mark.parametrize('args,kwargs,code,names', [
    ((), {}, 'missing_argument', ['x', 'y', 'tokens']),
    ((1, 2), {'token': 3}, 'unexpected_argument', ['token']),
    ((1, 2), {'y': 3, 'tokens': 4}, 'invalid_binding', ['y']),
    ((1, 2, 3), {'tokens': 4}, 'invalid_binding', []),
    ((), {'x': 1, 'y': 2, 'tokens': 3}, 'unexpected_argument', ['x']),
])
def test_binding_facts_use_python_parameter_kinds(args, kwargs, code, names):
    with pytest.raises(ExecutionRequirement) as error:
        bind_call(signature, args, kwargs, 'inputs')
    diagnostic = Diagnostic.model_validate(error.value.diagnostic)
    assert diagnostic.code == code and diagnostic.names == names
    assert diagnostic.phase == 'binding' and diagnostic.remedy == 'provide_inputs'


def test_explicit_variadic_and_positional_only_with_kwargs_are_legal():
    def explicit(x, /, *args, **kwargs):
        raise AssertionError('must not execute')
    bind_call(explicit, (1, 2), {'x': 3, 'anything': 4}, 'inputs')


def test_unknown_signature_is_distinct_from_invalid_bindings(monkeypatch):
    import inspect
    monkeypatch.setattr(inspect, 'signature', lambda target: (_ for _ in ()).throw(ValueError('private')))
    with pytest.raises(ExecutionRequirement) as error:
        bind_call(object(), (), {}, 'factory')
    assert error.value.diagnostic['code'] == 'unknown_signature'
    assert 'private' not in json.dumps(error.value.diagnostic)


def test_import_name_and_exception_prose_are_not_unbounded_diagnostics():
    with pytest.raises(ExecutionRequirement) as error:
        with setup_stage('import', ['model'], 'review_dependencies'):
            raise ModuleNotFoundError('private secret value', name='bad\nprivate')
    assert error.value.diagnostic['names'] == []
    assert 'private' not in json.dumps(error.value.diagnostic)


def test_import_time_code_errors_are_not_missing_dependencies():
    with pytest.raises(ExecutionRequirement) as error:
        with setup_stage('import', ['model'], 'review_dependencies', 'unresolved_import'):
            raise ValueError('private model configuration')
    assert error.value.diagnostic['code'] == 'execution_failed'
    assert error.value.diagnostic['names'] == ['ValueError']
    assert 'private' not in json.dumps(error.value.diagnostic)


def test_wrapped_missing_import_keeps_the_actual_module_name():
    with pytest.raises(ExecutionRequirement) as error:
        with setup_stage('import', ['model'], 'review_dependencies', 'unresolved_import'):
            try:
                raise ModuleNotFoundError('private', name='einops')
            except ModuleNotFoundError:
                raise RuntimeError('library lazy-import wrapper')
    assert error.value.diagnostic['code'] == 'missing_dependency'
    assert error.value.diagnostic['names'] == ['einops']


MODEL = '''from pathlib import Path
import torch
class Net(torch.nn.Module):
    def __init__(self, dims=3):
        super().__init__()
    def forward(self, x):
        Path(__file__).with_suffix('.called').touch()
        return x.sin()
def build(width=3): return Net(width)
'''


@pytest.mark.parametrize('source,extra,phase,code,field,names', [
    (MODEL.replace('dims=3', 'dims'), {}, 'binding', 'missing_argument', ['init'], ['dims']),
    (MODEL, {'constructorArgs': {'typo': 3}}, 'binding', 'unexpected_argument', ['init'], ['typo']),
    (MODEL.replace('return Net(width)', 'return torch.nn.Identity()'),
     {'setup': {'file': 'diagnostic_net.py', 'callable': 'build'}}, 'construction', 'invalid_factory', ['factory'], []),
    (MODEL.replace('width=3', 'width'), {'setup': {'file': 'diagnostic_net.py', 'callable': 'build'}},
     'binding', 'missing_argument', ['factory'], ['width']),
    (MODEL.replace('self, x', 'self, x, *, tokens'), {}, 'binding', 'missing_argument', ['inputs'], ['tokens']),
    ('import tensorviz_test_absent_dependency\n' + MODEL, {}, 'import', 'missing_dependency', ['model'], ['tensorviz_test_absent_dependency']),
    ('raise ValueError("private import failure")\n' + MODEL, {}, 'import', 'execution_failed', ['model'], ['ValueError']),
    (MODEL, {'device': 'cuda:99'}, 'environment', 'unavailable_device', ['device'], []),
    (MODEL, {'inputs': {'fixture': 'absent.pt'}}, 'configuration', 'missing_file', ['inputs', 'fixture'], []),
    (MODEL.replace('super().__init__()', 'raise RuntimeError("private constructor data")'), {},
     'construction', 'execution_failed', ['init'], []),
    (MODEL, {'inputs': {'fixture': 'malformed.pt'}},
     'materialization', 'invalid_config', ['inputs'], []),
])
def test_worker_returns_typed_setup_failure_and_never_calls_forward(tmp_path, source, extra, phase, code, field, names):
    file = tmp_path / 'diagnostic_net.py'
    file.write_text(source)
    (tmp_path / 'malformed.pt').write_bytes(b'not a weights-only fixture')
    spec = {'inputs': {'args': [{'shape': [2, 3], 'dtype': 'float32', 'fill': 'ones'}]}, **extra}
    process = subprocess.run([sys.executable, '-B', '-m', 'services.ingest.worker', '--file', str(file),
        '--cls', 'Net', '--workspace', str(tmp_path), '--execution', json.dumps(spec)],
        cwd=Path(__file__).parent.resolve().parents[1], capture_output=True, text=True, timeout=30)
    envelope = json.loads(process.stdout)
    assert process.returncode == 3, envelope
    diagnostic = Diagnostic.model_validate(envelope['error']['diagnostic'])
    assert (diagnostic.phase, diagnostic.code, diagnostic.field, diagnostic.names) == (phase, code, field, names)
    assert 'private' not in diagnostic.model_dump_json()
    assert envelope['result'] is None
    assert not file.with_suffix('.called').exists()
