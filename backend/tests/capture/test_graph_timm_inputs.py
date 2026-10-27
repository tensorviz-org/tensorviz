"""The installed ViT image contract must survive a real forward."""
import pytest
import torch
from torch import nn

pytest.importorskip('timm')
from timm.models.vision_transformer import VisionTransformer

from services.ingest.graph_example import propose_input
from services.validation.contracts import Binding
from services.validation.model_run import materialize


def _model(**kwargs):
    return VisionTransformer(**{**dict(img_size=32, patch_size=8, in_chans=3,
        num_classes=4, embed_dim=16, depth=1, num_heads=2, mlp_ratio=2), **kwargs}).eval()


@pytest.mark.parametrize('size,channels,dynamic', [(32, 3, False), ((16, 24), 1, False), (16, 3, True)])
def test_declared_patch_embedding_dimensions_produce_a_real_finite_output(size, channels, dynamic):
    model = _model(img_size=size, in_chans=channels, dynamic_img_size=dynamic)
    proposal = propose_input(model)
    binding = Binding.model_validate(proposal['binding'])
    kwargs = {key: materialize(value, torch, torch.device('cpu')) for key, value in binding.kwargs.items()}
    spatial = (size, size) if type(size) is int else size
    assert kwargs['x'].shape == (1, channels, *spatial), 'The ViT rejects a generic convolution image size.'
    with torch.no_grad():
        output = model(**kwargs)
    assert output.shape == (1, 4) and torch.isfinite(output).all()
    assert 'timm' in proposal['detail'] and 'capture' in proposal['detail']


def test_inherited_forward_uses_the_library_contract_without_running_forward_during_preparation():
    class Inherited(VisionTransformer):
        pass
    model = Inherited(img_size=32, patch_size=8, embed_dim=16, depth=1, num_heads=2)
    handle = model.register_forward_pre_hook(lambda *args: pytest.fail('Preparation ran forward.'))
    try:
        assert propose_input(model)['binding']['kwargs']['x']['shape'] == [1, 3, 32, 32]
    finally:
        handle.remove()


@pytest.mark.parametrize('size', [None, (0, 32), (32,), (32, True), (10000, 10000)])
def test_missing_invalid_or_oversized_patch_metadata_stays_unresolved(size):
    model = _model()
    model.patch_embed.img_size = size
    with pytest.raises(ValueError):
        propose_input(model)


def test_custom_patch_embedding_stays_unresolved():
    model = _model()
    model.patch_embed = nn.Conv2d(3, 16, 8, stride=8)
    with pytest.raises(ValueError):
        propose_input(model)


def test_replaced_forward_does_not_claim_the_timm_contract():
    class Changed(VisionTransformer):
        def forward(self, x):
            return x
    proposal = propose_input(Changed(img_size=32, patch_size=8, embed_dim=16, depth=1, num_heads=2))
    assert 'timm' not in proposal['detail']
