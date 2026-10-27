"""Native attention contracts supply complete examples without changing the model."""
import json
import tomllib

import pytest
import torch
from torch import nn

from services.ide.graph_setup import discover
from services.ide.graph_setup_profiles import prepare_profile
from services.ingest.graph_example import propose_input
from services.semantic.projection import project_capture
from services.validation.contracts import Binding
from services.validation.model_run import materialize
from tests.capture.test_graph_inherited_defaults import _worker
from tests.discovery.test_graph_setup_profiles import snapshot as sources


def _model(family, batch_first=False):
    if family == 'attention':
        return nn.MultiheadAttention(16, 4, kdim=12, vdim=20, batch_first=batch_first).eval()
    if family == 'decoder':
        return nn.TransformerDecoderLayer(16, 4, dim_feedforward=32, batch_first=batch_first).eval()
    return nn.Transformer(16, 4, num_encoder_layers=1, num_decoder_layers=1,
                          dim_feedforward=32, batch_first=batch_first).eval()


@pytest.mark.parametrize('family', ['transformer', 'attention', 'decoder'])
@pytest.mark.parametrize('batch_first', [False, True])
def test_native_contract_proposes_every_input_and_preserves_dimensions(family, batch_first):
    model = _model(family, batch_first).double()
    proposal = propose_input(model)
    binding = Binding.model_validate(proposal['binding'])
    kwargs = {k: materialize(v, torch, torch.device('cpu')) for k, v in binding.kwargs.items()}
    assert binding.args == []
    expected = {'query': (4, 16), 'key': (8, 12), 'value': (8, 20)} if family == 'attention' else (
        {'tgt': (4, 16), 'memory': (8, 16)} if family == 'decoder' else {'src': (8, 16), 'tgt': (4, 16)})
    assert kwargs.keys() == expected.keys(), 'Masks, causal hints and other optional arguments keep their defaults.'
    for name, (length, width) in expected.items():
        assert kwargs[name].shape == ((1, length, width) if batch_first else (length, 1, width))
        assert kwargs[name].dtype == torch.float64
    with torch.no_grad():
        output = model(**kwargs)
    if family == 'attention':
        output, weights = output
        assert weights is not None, 'The proposal must not override need_weights to hide an output.'
        assert torch.isfinite(weights).all()
    assert output.shape == ((1, 4, 16) if batch_first else (4, 1, 16))
    assert torch.isfinite(output).all()
    assert 'example' in proposal['detail'].lower() and 'capture' in proposal['detail'].lower()


@pytest.mark.parametrize('family', ['transformer', 'attention', 'decoder'])
def test_preparation_does_not_run_attention_forward(family):
    model = _model(family)
    handle = model.register_forward_pre_hook(lambda *args: pytest.fail('Preparation called forward.'))
    try:
        assert propose_input(model)['binding']['kwargs']
    finally:
        handle.remove()


def test_arbitrary_multi_input_model_does_not_acquire_the_attention_contract():
    class Similar(nn.Module):
        def __init__(self):
            super().__init__()
            self.d_model = 16
            self.batch_first = True
            self.linear = nn.Linear(16, 16)

        def forward(self, src, tgt):
            return src + tgt

    with pytest.raises(ValueError):
        propose_input(Similar())


def test_subclass_overriding_forward_does_not_acquire_the_library_contract():
    class Changed(nn.MultiheadAttention):
        def forward(self, query, key, value):
            return query

    with pytest.raises(ValueError):
        propose_input(Changed(16, 4))


def test_declared_dummy_inputs_keep_priority_over_the_native_example():
    model = _model('transformer')
    model.dummy_inputs = dict(src=torch.ones(2, 1, 16), tgt=torch.ones(1, 1, 16))
    kwargs = propose_input(model)['binding']['kwargs']
    assert kwargs['src']['shape'] == [2, 1, 16]
    assert kwargs['tgt']['shape'] == [1, 1, 16]
    assert kwargs['src']['generator']['kind'] == 'values'


@pytest.mark.parametrize('value', [0, True, 349526])
def test_invalid_or_oversized_stored_dimensions_require_explicit_inputs(value):
    model = _model('transformer')
    model.d_model = value
    with pytest.raises(ValueError):
        propose_input(model)


@pytest.mark.parametrize('family', ['attention', 'decoder'])
def test_constructed_attention_factory_captures_all_forward_inputs(tmp_path, family):
    api = 'MultiheadAttention' if family == 'attention' else 'TransformerDecoderLayer'
    source = (f'from torch.nn import {api}\nclass Net({api}): pass\n'
              f'def make_model() -> Net:\n    return Net(16, 4, batch_first=True)\n')
    request = sources(source)
    resolution = discover(request)['resolution']
    assert resolution.get('runtimeInputs') is True
    profile = resolution['draft']['profile']
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'model.py').write_text(source)
    code, envelope = _worker(tmp_path, workspace, profile, probe=True)
    assert code == 0, envelope
    profile['inputs'] = envelope['result']['graphExample']['binding']
    prepared = prepare_profile(request, '', profile)
    assert prepared['ok'], prepared
    (workspace / 'tensorviz.toml').write_text(prepared['config'])
    saved = tomllib.loads(prepared['config'])['profiles'][prepared['profile']]
    code, envelope = _worker(tmp_path, workspace, saved, probe=False)
    assert code == 0, envelope
    capture = envelope['result']['diagnostics']['semantic_capture']
    assert capture['status'] == 'complete', capture['coverage']
    root = next(v for v in capture['invocations'] if v['id'] == capture['rootInvocationId'])
    assert root['status'] == 'completed'
    graph = project_capture(capture)
    assert graph.nodes
    print(json.dumps(dict(family=api, nodes=len(graph.nodes), capture=capture['status'])))


def test_full_default_transformer_opens_from_saved_inputs_and_recaptures_an_edit(tmp_path):
    source = 'from torch.nn import Transformer\nclass Net(Transformer): pass\n'
    request = sources(source)
    resolution = discover(request)['resolution']
    assert resolution.get('runtimeInputs') is True
    profile = resolution['draft']['profile']
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'model.py').write_text(source)
    code, envelope = _worker(tmp_path, workspace, profile, probe=True)
    assert code == 0, envelope
    assert 'graph' not in envelope['result']
    profile['inputs'] = envelope['result']['graphExample']['binding']
    assert profile['inputs']['kwargs']['src']['shape'] == [8, 1, 512]
    assert profile['inputs']['kwargs']['tgt']['shape'] == [4, 1, 512]
    prepared = prepare_profile(request, '', profile)
    assert prepared['ok'], prepared
    (workspace / 'tensorviz.toml').write_text(prepared['config'])
    saved = tomllib.loads(prepared['config'])['profiles'][prepared['profile']]
    assert saved['init'] == dict(args=[], kwargs={}) and 'factory' not in saved
    for length in (4, 3):
        saved['inputs']['kwargs']['tgt']['shape'][0] = length
        code, envelope = _worker(tmp_path, workspace, saved, probe=False)
        assert code == 0, envelope
        capture = envelope['result']['diagnostics']['semantic_capture']
        assert capture['status'] == 'complete', capture['coverage']
        root = next(v for v in capture['invocations'] if v['id'] == capture['rootInvocationId'])
        assert root['status'] == 'completed'
        output = next(v for v in capture['values'] if v['id'] == capture['output']['id'])
        assert output['shape'] == [length, 1, 512]
        graph = project_capture(capture)
        assert len(graph.nodes) > 100
        print(json.dumps(dict(family='Transformer', target_length=length,
                              nodes=len(graph.nodes), capture=capture['status'])))
