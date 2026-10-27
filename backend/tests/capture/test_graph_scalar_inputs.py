"""Declared scalar types can complete an example call, not certify its semantics."""
import inspect
import json
import tomllib
from typing import Optional, Union

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
from tests.capture.test_graph_runtime_inputs import Model
from tests.discovery.test_graph_setup_profiles import snapshot as sources


def _call(model, proposal):
    binding = Binding.model_validate(proposal['binding'])
    args = [materialize(value, torch, torch.device('cpu')) for value in binding.args]
    kwargs = {name: materialize(value, torch, torch.device('cpu')) for name, value in binding.kwargs.items()}
    inspect.signature(model.forward).bind(*args, **kwargs)
    with torch.no_grad():
        return model(*args, **kwargs)


@pytest.mark.parametrize('annotation,value', [
    (int, 1), (float, 1.0), (bool, False),
    (Union[torch.Tensor, float, int], 1), (torch.Tensor | float, 1.0),
])
def test_tensor_and_typed_scalar_form_one_editable_verified_example(annotation, value):
    class Timed(Model):
        def forward(self, sample, timestep, return_dict=True):
            assert type(timestep) is type(value) and timestep == value
            assert return_dict is True, 'Optional arguments retain the library default.'
            return self.features(sample)
    Timed.forward.__annotations__['timestep'] = annotation
    model = Timed(nn.Conv2d(3, 4, 3)).eval()
    proposal = propose_input(model)
    assert set(proposal['binding']['kwargs']) == {'sample', 'timestep'}
    assert 'example' in proposal['detail'].lower() and 'scalar' in proposal['detail'].lower()
    assert _call(model, proposal).shape == (1, 4, 62, 62)


def test_scalar_and_tensor_positional_order_is_preserved():
    class Positional(Model):
        def forward(self, step: int, sample, /, *, scale: float):
            assert type(step) is int and type(scale) is float
            return self.features(sample) * step * scale
    proposal = propose_input(Positional(nn.Linear(7, 2)))
    assert proposal['binding']['args'][0] == 1
    assert proposal['binding']['args'][1]['shape'] == [1, 7]
    assert set(proposal['binding']['kwargs']) == {'scale'}
    assert _call(Positional(nn.Linear(7, 2)), proposal).shape == (1, 2)


def test_required_scalars_do_not_need_a_tensor_layer():
    class Scalar(nn.Module):
        def forward(self, count: int, enabled: bool):
            assert enabled is False
            return torch.ones(count)
    assert _call(Scalar(), propose_input(Scalar())).shape == (1,)


def test_declared_optional_primary_tensor_can_accompany_required_scalar():
    class OptionalInput(Model):
        main_input_name = 'sample'
        def forward(self, step: int, sample: Optional[torch.Tensor] = None):
            return self.features(sample) * step
    assert _call(OptionalInput(nn.Linear(7, 2)), propose_input(OptionalInput(nn.Linear(7, 2)))).shape == (1, 2)


@pytest.mark.parametrize('annotation', [str, 'int', 'do_not_execute()', Optional[int], Union[int, str], list[int]])
def test_opaque_or_ambiguous_scalar_contracts_stay_unresolved(annotation):
    class Unknown(Model):
        def forward(self, sample, context):
            raise AssertionError('Preparation must not call forward.')
    Unknown.forward.__annotations__['context'] = annotation
    with pytest.raises(ValueError):
        propose_input(Unknown(nn.Linear(7, 2)))


def test_multiple_tensor_arguments_are_not_filled_with_scalars():
    class Conditional(Model):
        def forward(self, sample, step: int, context: torch.Tensor):
            pass
    with pytest.raises(ValueError):
        propose_input(Conditional(nn.Linear(7, 2)))


def test_variadic_forward_stays_unresolved_without_a_primary_contract():
    class Variadic(Model):
        def forward(self, sample, step: int, **kwargs):
            pass
    with pytest.raises(ValueError):
        propose_input(Variadic(nn.Linear(7, 2)))


def test_scalar_proposal_does_not_execute_forward_or_claim_success():
    class Fails(Model):
        def forward(self, sample, step: int):
            raise RuntimeError('The example does not fit this forward.')
    model = Fails(nn.Linear(7, 2))
    proposal = propose_input(model)
    assert set(proposal) == {'binding', 'detail'}
    with pytest.raises(RuntimeError, match='does not fit'):
        _call(model, proposal)


@pytest.mark.parametrize('family', ['UNet2DModel', 'AutoencoderKL'])
def test_default_diffusion_model_opens_from_saved_example_without_overrides(tmp_path, family):
    pytest.importorskip('diffusers')
    source = f'from diffusers import {family}\nclass Net({family}): pass\n'
    request = sources(source)
    resolution = discover(request)['resolution']
    assert resolution.get('runtimeInputs') is True
    profile = resolution['draft']['profile']
    workspace = tmp_path/'workspace'
    workspace.mkdir()
    (workspace/'model.py').write_text(source)
    code, envelope = _worker(tmp_path, workspace, profile, probe=True)
    assert code == 0, envelope
    assert 'graph' not in envelope['result']
    proposal = envelope['result']['graphExample']
    profile['inputs'] = proposal['binding']
    assert profile['inputs']['kwargs']['sample']['shape'] == [1, 3, 64, 64]
    if family == 'UNet2DModel':
        assert profile['inputs']['kwargs']['timestep'] == 1
    prepared = prepare_profile(request, '', profile)
    assert prepared['ok'], prepared
    (workspace/'tensorviz.toml').write_text(prepared['config'])
    saved = tomllib.loads(prepared['config'])['profiles'][prepared['profile']]
    assert saved['init'] == dict(args=[], kwargs={}) and 'factory' not in saved
    code, envelope = _worker(tmp_path, workspace, saved, probe=False)
    assert code == 0, envelope
    capture = envelope['result']['diagnostics']['semantic_capture']
    assert capture['status'] == 'complete'
    graph = project_capture(capture)
    assert len(graph.nodes) > 10
    output = capture['output']
    output_id = output['entries'][0]['value']['id']
    tensor = next(value for value in capture['values'] if value['id'] == output_id)
    assert tensor['shape'] == [1, 3, 64, 64]
    print(json.dumps(dict(family=family, nodes=len(graph.nodes), coverage=capture['status'], output_shape=tensor['shape'])))


def test_optional_positional_defaults_are_not_overwritten_by_a_later_primary_input():
    class Gap(Model):
        main_input_name = 'sample'
        def forward(self, flag=False, sample=None, /):
            pass
    with pytest.raises(ValueError, match='Skipped positional defaults'):
        propose_input(Gap(nn.Linear(7, 2)))


@pytest.mark.parametrize('annotation', [str, 'int', 'do_not_execute()', Optional[int], Union[int, str], list[int]])
def test_resolving_one_scalar_does_not_turn_an_opaque_argument_into_a_tensor(annotation):
    class Unknown(Model):
        def forward(self, count: int, context):
            pass
    Unknown.forward.__annotations__['context'] = annotation
    with pytest.raises(ValueError):
        propose_input(Unknown(nn.Linear(7, 2)))
