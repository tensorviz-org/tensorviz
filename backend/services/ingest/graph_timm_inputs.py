"""Editable image examples for the installed timm ViT forward contract."""
from __future__ import annotations

import math

import torch


def propose_timm_input(model):
    forward = getattr(model.forward, '__func__', None)
    if getattr(forward, '__module__', None) != 'timm.models.vision_transformer':
        return None
    from timm.layers.patch_embed import PatchEmbed
    from timm.models.vision_transformer import VisionTransformer
    if not isinstance(model, VisionTransformer) or forward is not VisionTransformer.forward:
        return None
    patch = vars(model).get('_modules', {}).get('patch_embed')
    if type(patch) is not PatchEmbed:
        raise ValueError('The timm ViT needs an explicit input for its custom patch embedding.')
    projection = vars(patch).get('_modules', {}).get('proj')
    spatial = vars(patch).get('img_size')
    if (type(projection) is not torch.nn.Conv2d or type(spatial) not in (tuple, list)
            or len(spatial) != 2 or any(type(n) is not int or n < 1 for n in spatial)):
        raise ValueError('The timm ViT patch dimensions need an explicit input.')
    shape = [1, projection.in_channels, *spatial]
    dtype = str(projection.weight.dtype).removeprefix('torch.')
    if math.prod(shape) > 4_194_304 or dtype not in ('float16', 'float32', 'float64', 'bfloat16'):
        raise ValueError('The timm ViT input example exceeds the supported tensor budget or dtype.')
    tensor = dict(kind='tensor', shape=shape, dtype=dtype, requires_grad=False,
                  generator=dict(kind='normal', mean=0, std=1, seed=0))
    return dict(binding=dict(args=[], kwargs=dict(x=tensor)),
                detail='Example image from the installed timm ViT patch-embedding dimensions; batch 1. '
                'These are editable examples, not recovered task settings. The capture must verify this candidate.')
