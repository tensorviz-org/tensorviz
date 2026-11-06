"""Library-specific input contracts must survive saving and real capture."""
import json
import hashlib
import tomllib

import pytest
import torch
from torch import nn

pytest.importorskip('diffusers')
from diffusers import UNet2DConditionModel

from services.ide.graph_setup import discover
from services.ide.graph_setup_profiles import prepare_profile
from services.ingest.graph_example import propose_input
from services.semantic.projection import project_capture
from services.semantic.document import capture_document
from services.ide.local_browser_wire import decode_message, encode_message, MAX_WIRE_BYTES
from services.validation.contracts import Binding
from services.validation.model_run import materialize
from tests.capture.test_graph_inherited_defaults import _worker
from tests.discovery.test_graph_setup_profiles import snapshot as sources


SMALL = dict(sample_size=16, block_out_channels=(32, 64), layers_per_block=1,
             down_block_types=('CrossAttnDownBlock2D', 'DownBlock2D'),
             up_block_types=('UpBlock2D', 'CrossAttnUpBlock2D'), cross_attention_dim=16,
             attention_head_dim=4, norm_num_groups=8)
VARIANTS = {
    'text': {},
    'projected_text': dict(encoder_hid_dim=12),
    'pooled_text': dict(addition_embed_type='text', addition_embed_type_num_heads=4),
    'text_time': dict(addition_embed_type='text_time', addition_time_embed_dim=8,
                      projection_class_embeddings_input_dim=24),
    'optional_timestep_condition': dict(time_cond_proj_dim=12),
}


def _model(variant='text', **kwargs):
    return UNet2DConditionModel(**{**SMALL, **VARIANTS[variant], **kwargs}).eval()


@pytest.mark.parametrize('variant', VARIANTS)
def test_declared_conditioning_dimensions_produce_a_real_finite_output(variant):
    model = _model(variant)
    proposal = propose_input(model)
    binding = Binding.model_validate(proposal['binding'])
    kwargs = {key: materialize(value, torch, torch.device('cpu')) for key, value in binding.kwargs.items()}
    assert kwargs['sample'].shape == (1, 4, 16, 16)
    assert kwargs['encoder_hidden_states'].shape == (1, 8, 12 if variant == 'projected_text' else 16)
    assert kwargs['timestep'] == 1
    assert not {'return_dict', 'timestep_cond', 'class_labels'} & kwargs.keys(), 'Optional defaults stay intact.'
    if variant == 'text_time':
        # Any compatible nonempty example is allowed; this is not recovered image metadata.
        extra = kwargs['added_cond_kwargs']
        assert extra['text_embeds'].shape[-1] + extra['time_ids'].shape[-1] * 8 == 24
    else:
        assert 'added_cond_kwargs' not in kwargs
    with torch.no_grad():
        output = model(**kwargs).sample
    assert output.shape == (1, 4, 16, 16) and torch.isfinite(output).all()
    assert 'example' in proposal['detail'].lower() and 'capture' in proposal['detail'].lower()


def test_adapter_does_not_run_forward_during_preparation():
    model = _model()
    handle = model.register_forward_pre_hook(lambda *args: pytest.fail('Preparation ran forward.'))
    try:
        assert propose_input(model)['binding']['kwargs']['encoder_hidden_states']['shape'] == [1, 8, 16]
    finally:
        handle.remove()


def test_rectangular_stored_sample_size_and_uniform_per_block_widths():
    model = _model(sample_size=(16, 24), cross_attention_dim=(16, 16))
    binding = propose_input(model)['binding']['kwargs']
    assert binding['sample']['shape'] == [1, 4, 16, 24]
    assert binding['encoder_hidden_states']['shape'] == [1, 8, 16]


@pytest.mark.parametrize('override', [
    dict(num_class_embeds=3),
    dict(encoder_hid_dim=16, encoder_hid_dim_type='text_image_proj'),
    dict(addition_embed_type='text_image'),
    dict(cross_attention_dim=(16, 24)),
    dict(sample_size=10000),
    dict(sample_size=0),
    dict(sample_size=(16,)),
    dict(addition_embed_type='text_time', addition_time_embed_dim=8,
         projection_class_embeddings_input_dim=8),
])
def test_unsupported_or_oversized_contracts_return_to_setup(override):
    with pytest.raises(ValueError):
        propose_input(_model(**override))


def test_overridden_forward_is_not_treated_as_the_library_contract():
    class Changed(UNet2DConditionModel):
        def forward(self, sample, timestep, encoder_hidden_states):
            return sample
    with pytest.raises(ValueError):
        propose_input(Changed(**SMALL))


def test_similar_argument_names_and_attributes_are_not_a_library_contract():
    class Pretender(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_in = nn.Conv2d(4, 4, 3)
            self.config = dict(cross_attention_dim=16)
        def forward(self, sample, timestep: int, encoder_hidden_states):
            pass
    with pytest.raises(ValueError):
        propose_input(Pretender())


@pytest.mark.parametrize('variant', ['full_defaults', 'projected_text', 'text_time'])
def test_conditional_inputs_survive_saved_profile_capture(tmp_path, variant):
    # The default UNet initializes hundreds of millions of weights on CPU before
    # capture. Give that integration fixture its own bounded budget on CI runners.
    timeout_seconds = 180 if variant == 'full_defaults' else 30
    source = 'from diffusers import UNet2DConditionModel\nclass Net(UNet2DConditionModel): pass\n'
    if variant != 'full_defaults':
        kwargs = {**SMALL, **VARIANTS[variant]}
        arguments = ', '.join(f'{key}={value!r}' for key, value in kwargs.items())
        source += f'\ndef make_model() -> Net:\n    return Net({arguments})\n'
    request = sources(source)
    resolution = discover(request)['resolution']
    assert resolution.get('runtimeInputs') is True
    profile = resolution['draft']['profile']
    workspace = tmp_path/'workspace'
    workspace.mkdir()
    (workspace/'model.py').write_text(source)
    code, envelope = _worker(tmp_path, workspace, profile, probe=True, timeout_seconds=timeout_seconds)
    assert code == 0, envelope
    assert 'graph' not in envelope['result']
    profile['inputs'] = envelope['result']['graphExample']['binding']
    prepared = prepare_profile(request, '', profile)
    assert prepared['ok'], prepared
    (workspace/'tensorviz.toml').write_text(prepared['config'])
    saved = tomllib.loads(prepared['config'])['profiles'][prepared['profile']]
    assert saved['inputs'] == profile['inputs']
    code, envelope = _worker(tmp_path, workspace, saved, probe=False, timeout_seconds=timeout_seconds)
    assert code == 0, envelope
    capture = envelope['result']['diagnostics']['semantic_capture']
    (tmp_path/'capture.json').write_text(json.dumps(capture))
    print(json.dumps(dict(variant=variant, status=capture['status'], omissions=capture['coverage']['omissions'], operations=len(capture['operations']), invocations=len(capture['invocations']))))
    assert capture['status'] == 'complete'
    output_id = capture['output']['entries'][0]['value']['id']
    output = next(value for value in capture['values'] if value['id'] == output_id)
    size = 64 if variant == 'full_defaults' else 16
    assert output['shape'] == [1, 4, size, size]
    graph = project_capture(capture)
    assert len(graph.nodes) > 100
    # A successful capture is insufficient if its display document is refused.
    document, _ = capture_document(capture, name='Net', graph_id='conditional-test',
        model_file='model.py', source_digests={'model.py': hashlib.sha256(source.encode()).hexdigest()},
        source_graph={'nodes': [], 'edges': []}, source_spans={}, source_texts={'model.py': source})
    payload = document.model_dump(mode='json', by_alias=True)
    wire = encode_message(payload)
    assert len(wire.encode()) < MAX_WIRE_BYTES
    assert decode_message(wire) == payload
    assert len(document.semantic.nodes) == len(graph.nodes)
    assert len(document.semantic.edges) == len(graph.edges)
    print(json.dumps(dict(variant=variant, nodes=len(graph.nodes), coverage=capture['status'], output_shape=output['shape'])))


class UNetWrapper(nn.Module):
    def __init__(self, child):
        super().__init__()
        self.model = child

    def forward(self, sample, timestep, encoder_hidden_states):
        pytest.fail('Preparation ran wrapper forward.')


def test_direct_child_unet_proposes_wrapper_inputs_without_forward():
    model = UNetWrapper(_model())
    handle = model.model.register_forward_pre_hook(lambda *args: pytest.fail('Preparation ran child forward.'))
    try:
        binding = Binding.model_validate(propose_input(model)['binding'])
        assert set(binding.kwargs) == {'sample', 'timestep', 'encoder_hidden_states'}
        assert binding.kwargs['sample'].shape == [1, 4, 16, 16]
        assert binding.kwargs['encoder_hidden_states'].shape == [1, 8, 16]
    finally:
        handle.remove()


@pytest.mark.parametrize('kind', ['ambiguous', 'renamed', 'nested', 'overridden'])
def test_unet_wrapper_requires_single_direct_compatible_library_child(kind):
    child = _model()
    model = UNetWrapper(child)
    if kind == 'ambiguous':
        model.other = _model()
    elif kind == 'renamed':
        model.forward = lambda sample, timestep, context: None
    elif kind == 'nested':
        model.model = nn.Sequential(child)
    else:
        child.forward = lambda sample, timestep, encoder_hidden_states: sample
    with pytest.raises(ValueError):
        propose_input(model)


def test_unet_wrapper_filters_optional_kwargs_and_adapts_positional_parameters():
    class Positional(UNetWrapper):
        def forward(self, sample, /, timestep=1):
            pytest.fail('Preparation ran forward.')
    binding = propose_input(Positional(_model()))['binding']
    assert binding['args'][0]['shape'] == [1, 4, 16, 16]
    assert binding['kwargs'] == {'timestep': 1}
