"""Runtime proposals describe an example; only subsequent capture proves it works."""
import pytest
import torch
from torch import nn

from services.ingest.graph_example import propose_input
from services.validation.model_run import materialize


class Model(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.features = nn.Sequential(layer)
    def forward(self, x):
        return self.features(x)


@pytest.mark.parametrize('layer,shape,dtype', [
    (nn.Linear(12, 4), [1, 12], 'float32'),
    (nn.Conv1d(2, 4, 3), [1, 2, 64], 'float32'),
    (nn.Conv2d(5, 4, 3), [1, 5, 64, 64], 'float32'),
    (nn.Conv3d(2, 4, 3), [1, 2, 16, 16, 16], 'float32'),
    (nn.Conv2d(3, 4, 14, stride=14), [1, 3, 70, 70], 'float32'),
    (nn.Embedding(19, 8), [1, 8], 'int64'),
    (nn.GRU(7, 8, batch_first=True), [1, 8, 7], 'float32'),
    (nn.LSTM(7, 8, batch_first=False), [8, 1, 7], 'float32'),
    (nn.Linear(12, 4).double(), [1, 12], 'float64'),
])
def test_constructed_layers_produce_bounded_bindings_that_run(layer, shape, dtype):
    model = Model(layer).eval()
    proposal = propose_input(model)
    value = proposal['binding']['kwargs']['x']
    assert value['shape'] == shape and value['dtype'] == dtype
    assert proposal['binding']['args'] == []
    from services.validation.contracts import TensorValue
    tensor = materialize(TensorValue.model_validate(value), torch, torch.device('cpu'))
    with torch.no_grad():
        result = model(tensor)
    if isinstance(result, tuple): result = result[0]
    assert torch.isfinite(result).all()


def test_declared_image_size_overrides_generic_example():
    model = Model(nn.Conv2d(3, 4, 16, stride=16))
    model.image_size = 224
    assert propose_input(model)['binding']['kwargs']['x']['shape'] == [1, 3, 224, 224]


def test_positional_only_forward_keeps_positional_binding():
    class Positional(Model):
        def forward(self, x, /): return self.features(x)
    binding = propose_input(Positional(nn.Linear(7, 2)))['binding']
    assert binding['kwargs'] == {} and binding['args'][0]['shape'] == [1, 7]


@pytest.mark.parametrize('kind', ['two_inputs', 'variadic', 'opaque', 'oversized', 'invalid_size'])
def test_unsupported_or_oversized_contracts_do_not_generate_a_recipe(kind):
    model = Model(nn.Linear(4, 2))
    if kind == 'two_inputs': model.forward = lambda x, context: x
    if kind == 'variadic': model.forward = lambda *args: args
    if kind == 'opaque': model = Model(nn.Identity())
    if kind in ('oversized', 'invalid_size'):
        model = Model(nn.Conv2d(3, 4, 3))
        model.image_size = 10000 if kind == 'oversized' else -1
    with pytest.raises(ValueError): propose_input(model)


def test_proposal_does_not_call_forward_or_claim_success():
    model = Model(nn.Linear(4, 2))
    def forward(x): raise AssertionError('not a capture')
    model.forward = forward
    proposal = propose_input(model)
    assert set(proposal) == {'binding', 'detail'}
    assert 'example' in proposal['detail'].lower()


def _run_probe(tmp_path, source, *, explicit=False, mutate=False, factory=False, probe=True):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    from services.semantic.execution_plan import profile_execution
    from services.validation.contracts import ModelProfile
    from services.validation.host_snapshot import snapshot
    root = tmp_path / 'workspace'
    (root / 'models').mkdir(parents=True, exist_ok=True)
    file = root / 'models/model.py'
    file.write_text(source)
    if factory:
        (root / 'factory.py').write_text('from models.model import Net\ndef make_model(): return Net()\n')
    profile = dict(kind='model', model='models.model:Net', mode='eval', device='cpu',
                   grad_enabled=False, seed=0, inputs=dict(args=[], kwargs={}))
    profile.update(factory=dict(file='factory.py', callable='make_model', kwargs={}) if factory else {},
                   **({} if factory else {'init': dict(args=[], kwargs={})}))
    if not factory: profile.pop('factory')
    if explicit:
        profile['inputs']['kwargs']['x'] = dict(kind='tensor', shape=[1, 4], dtype='float32', requires_grad=False, generator=dict(kind='full', value=0))
    execution = profile_execution(ModelProfile.model_validate(profile))
    raw = snapshot(dict(root=str(root), executable=sys.executable, environmentRevision=0,
        session='input-probe', workspaceId='input-probe', runId='input-probe', generation=1,
        modelFile='models/model.py', className='Net', timeoutMs=30000, execution=execution))
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'version': 1, 'plan': raw['plan'], 'applicabilityDigest': raw['applicabilityDigest']}))
    plan.chmod(0o600)
    if mutate: file.write_text(source + '\n# changed after preparation\n')
    backend = Path(__file__).parent.resolve().parents[1]
    result = subprocess.run([sys.executable, str(backend/'services/ingest/bootstrap.py'),
        '--file', str(file), '--cls', 'Net', '--workspace', str(root), '--resolved-execution',
        *(['--prepare-inputs'] if probe else []), '--json'], cwd=backend,
        env={**os.environ, 'PYTHONPATH': '', 'TENSORVIZ_EXECUTION_PLAN': str(plan)},
        capture_output=True, text=True, timeout=40)
    return result.returncode, json.loads(result.stdout)


_PROBE_MODEL = '''from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for _ in range(1): layers.append(nn.Linear(4, 2))
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        raise RuntimeError("deliberate forward failure")
'''


@pytest.mark.parametrize('factory', [False, True])
def test_private_worker_prepares_namespace_model_without_calling_forward(tmp_path, factory):
    code, envelope = _run_probe(tmp_path, _PROBE_MODEL, factory=factory)
    assert code == 0, envelope
    assert envelope['result']['graphExample']['binding']['kwargs']['x']['shape'] == [1, 4]
    assert envelope['result']['diagnostics']['execution']['planCurrent'] is True
    assert 'graph' not in envelope['result'], 'preparation must never masquerade as a capture'


def test_probe_success_does_not_hide_a_subsequent_forward_failure(tmp_path):
    code, envelope = _run_probe(tmp_path, _PROBE_MODEL)
    assert code == 0
    code, envelope = _run_probe(tmp_path, _PROBE_MODEL, explicit=True, probe=False)
    assert code == 4
    assert envelope['error'] is not None


@pytest.mark.parametrize('explicit,mutate', [(True, False), (False, True)])
def test_private_probe_refuses_explicit_inputs_and_stale_source_before_import(tmp_path, explicit, mutate):
    marker = tmp_path / 'must-not-import'
    source = f'from pathlib import Path\nPath({str(marker)!r}).touch()\n' + _PROBE_MODEL
    code, envelope = _run_probe(tmp_path, source, explicit=explicit, mutate=mutate)
    assert code != 0 and envelope['result'] is None
    assert not marker.exists()


class EncoderDecoder(nn.Module):
    main_input_name = 'input_ids'

    @property
    def dummy_inputs(self):
        return {'input_ids': torch.tensor([[2, 4, 0]]),
                'decoder_input_ids': torch.tensor([[0, 2]])}

    def forward(self, input_ids=None, decoder_input_ids=None):
        assert input_ids is not None and decoder_input_ids is not None
        return input_ids.sum() + decoder_input_ids.sum()


def test_declared_example_preserves_multiple_optional_inputs_and_exact_values():
    from services.validation.contracts import TensorValue
    model = EncoderDecoder()
    proposal = propose_input(model)
    assert proposal['binding']['args'] == []
    kwargs = {key: materialize(TensorValue.model_validate(value), torch, torch.device('cpu'))
              for key, value in proposal['binding']['kwargs'].items()}
    assert all(torch.equal(value, model.dummy_inputs[key]) for key, value in kwargs.items())
    assert model(**kwargs).item() == 8
    assert 'dummy_inputs' in proposal['detail']


def test_declared_example_handles_required_inputs_without_guessing_context():
    class Required(EncoderDecoder):
        def forward(self, input_ids, decoder_input_ids):
            return input_ids, decoder_input_ids
    assert set(propose_input(Required())['binding']['kwargs']) == {'input_ids', 'decoder_input_ids'}


def test_inherited_text_example_cannot_replace_declared_image_input():
    from types import SimpleNamespace
    class Vision(Model):
        main_input_name = 'pixel_values'
        dummy_inputs = {'input_ids': torch.tensor([[1, 2]])}
        def forward(self, pixel_values=None, **kwargs):
            assert pixel_values.shape[-2:] == (28, 28)
            return self.features(pixel_values)
    model = Vision(nn.Conv2d(3, 4, 14, stride=14))
    model.config = SimpleNamespace(image_size=28)
    binding = propose_input(model)['binding']
    assert set(binding['kwargs']) == {'pixel_values'}
    assert binding['kwargs']['pixel_values']['shape'] == [1, 3, 28, 28]


@pytest.mark.parametrize('example', [
    {'input_ids': torch.ones(4097, dtype=torch.int64)},
    {'input_ids': torch.tensor([float('nan')])},
    {'input_ids': object()},
    {'input_ids': torch.ones(1)},  # Missing required decoder input.
])
def test_invalid_declared_examples_stay_unresolved(example):
    class Required(EncoderDecoder):
        dummy_inputs = example
        def forward(self, input_ids, decoder_input_ids): return input_ids
    with pytest.raises(ValueError):
        propose_input(Required())


def test_optional_forward_without_an_input_contract_keeps_its_defaults():
    class Optional(Model):
        def forward(self, x=None):
            return torch.ones(1) if x is None else self.features(x)
    assert propose_input(Optional(nn.Linear(4, 2)))['binding'] == dict(args=[], kwargs={})
