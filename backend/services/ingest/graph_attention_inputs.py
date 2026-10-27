"""Complete editable inputs for native PyTorch attention interfaces."""
from __future__ import annotations

import math

import torch
from torch import nn


def propose_attention_input(model):
    forward = getattr(model.forward, '__func__', None)
    metadata = vars(model)
    if isinstance(model, nn.Transformer) and forward is nn.Transformer.forward:
        width = metadata.get('d_model')
        dimensions = dict(src=(8, width), tgt=(4, width))
    elif isinstance(model, nn.MultiheadAttention) and forward is nn.MultiheadAttention.forward:
        dimensions = dict(query=(4, metadata.get('embed_dim')), key=(8, metadata.get('kdim')),
                          value=(8, metadata.get('vdim')))
    elif isinstance(model, nn.TransformerDecoderLayer) and forward is nn.TransformerDecoderLayer.forward:
        attention = metadata.get('_modules', {}).get('self_attn')
        if type(attention) is not nn.MultiheadAttention:
            raise ValueError('The decoder attention dimensions need explicit inputs.')
        metadata = vars(attention)
        dimensions = dict(tgt=(4, metadata.get('embed_dim')), memory=(8, metadata.get('embed_dim')))
    else:
        return None
    batch_first = metadata.get('batch_first')
    if type(batch_first) is not bool:
        raise ValueError('The attention batch layout needs explicit inputs.')
    weight = next(model.parameters(), None)
    dtype = str(weight.dtype).removeprefix('torch.') if weight is not None else None
    if dtype not in ('float16', 'float32', 'float64', 'bfloat16'):
        raise ValueError('The attention model needs an explicit input dtype.')
    kwargs = {}
    elements = 0
    for name, (length, width) in dimensions.items():
        if type(width) is not int or width < 1:
            raise ValueError('Declared attention dimensions need explicit inputs.')
        shape = [1, length, width] if batch_first else [length, 1, width]
        elements += math.prod(shape)
        if elements > 4_194_304:
            raise ValueError('The input example exceeds the supported tensor budget.')
        kwargs[name] = dict(kind='tensor', shape=shape, dtype=dtype, requires_grad=False,
                            generator=dict(kind='normal', mean=0, std=1, seed=0))
    return dict(binding=dict(args=[], kwargs=kwargs),
                detail='Example inputs from the native PyTorch attention API and constructed dimensions: '
                'batch 1, 4 query or target tokens, 8 source or context tokens. '
                'These are editable examples, not recovered task settings. The capture must verify this candidate.')
