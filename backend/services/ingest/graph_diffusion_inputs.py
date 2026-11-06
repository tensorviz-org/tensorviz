"""Editable examples for the installed conditional UNet API, never a forward check."""
from __future__ import annotations

import inspect
import math

import torch


def propose_diffusion_input(model):
    proposal = _propose_library_input(model)
    if proposal is not None:
        return proposal
    signature = inspect.signature(model.forward)
    parameters = list(signature.parameters.values())
    matches = []
    # Only direct composition establishes a candidate; never search descendants.
    for index, (_, child) in enumerate(model.named_children()):
        if index >= 4096:
            raise ValueError('The constructed model exceeds the input inspection budget.')
        proposal = _propose_library_input(child)
        if proposal is None:
            continue
        values = proposal['binding']['kwargs']
        binding = dict(args=[], kwargs={})
        for position, parameter in enumerate(parameters):
            if parameter.name not in values:
                continue
            if parameter.kind is parameter.POSITIONAL_ONLY:
                # Skipping a positional default would change the argument meaning.
                if len(binding['args']) != position:
                    break
                binding['args'].append(values[parameter.name])
            elif parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY):
                binding['kwargs'][parameter.name] = values[parameter.name]
        else:
            try:
                signature.bind(*binding['args'], **binding['kwargs'])
            except TypeError:
                continue
            if binding['args'] or binding['kwargs']:
                matches.append(dict(binding=binding, detail=proposal['detail']))
                if len(matches) > 1:
                    raise ValueError('Multiple conditional UNet children need an explicit input binding.')
    return matches[0] if matches else None


def _propose_library_input(model):
    forward = getattr(model.forward, '__func__', None)
    if getattr(forward, '__module__', None) != 'diffusers.models.unets.unet_2d_condition':
        return None
    # Import only for this library's actual forward. Similar names on arbitrary
    # models, and subclasses that replace forward, do not establish this contract.
    from diffusers.configuration_utils import FrozenDict
    from diffusers.models.embeddings import Timesteps, TimestepEmbedding
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
    if not isinstance(model, UNet2DConditionModel) or forward is not UNet2DConditionModel.forward:
        return None
    config = vars(model).get('_internal_dict')
    if type(config) is not FrozenDict or type(model.conv_in) is not torch.nn.Conv2d:
        raise ValueError('Conditional UNet input metadata is unavailable.')
    if model.class_embedding is not None:
        raise ValueError('This conditional UNet needs explicit class-label inputs.')
    projection_type = config.get('encoder_hid_dim_type')
    projection = model.encoder_hid_proj
    if projection_type is None and projection is None:
        width = config.get('cross_attention_dim')
        if type(width) in (tuple, list):
            if not width or any(type(n) is not int or n != width[0] for n in width):
                raise ValueError('Different conditioning widths need explicit inputs.')
            width = width[0]
    elif projection_type == 'text_proj' and type(projection) is torch.nn.Linear:
        width = projection.in_features
    else:
        raise ValueError('This conditional UNet needs explicit image or custom conditioning inputs.')
    addition = config.get('addition_embed_type')
    if addition not in (None, 'text', 'text_time'):
        raise ValueError('This conditional UNet needs explicit additional conditioning inputs.')
    size = config.get('sample_size')
    spatial = [64, 64] if size is None else [size, size] if type(size) is int else size
    if type(spatial) not in (tuple, list) or len(spatial) != 2:
        raise ValueError('Declared sample dimensions need an explicit input.')
    dtype = str(model.conv_in.weight.dtype).removeprefix('torch.')
    if dtype not in ('float16', 'float32', 'float64', 'bfloat16'):
        raise ValueError('The conditional UNet needs an explicit input dtype.')
    elements = 0

    def tensor(shape, *, integer=False):
        nonlocal elements
        if any(type(n) is not int or n < 1 for n in shape):
            raise ValueError('Declared conditioning dimensions need explicit inputs.')
        elements += math.prod(shape)
        if elements > 4_194_304:
            raise ValueError('The input example exceeds the supported tensor budget.')
        return dict(kind='tensor', shape=shape, dtype='int64' if integer else dtype,
                    requires_grad=False, generator=dict(kind='full', value=1) if integer
                    else dict(kind='normal', mean=0, std=1, seed=0))

    kwargs = dict(sample=tensor([1, model.conv_in.in_channels, *spatial]), timestep=1,
                  encoder_hidden_states=tensor([1, 8, width]))
    if addition == 'text_time':
        if (type(model.add_time_proj) is not Timesteps or type(model.add_embedding) is not TimestepEmbedding
                or type(model.add_embedding.linear_1) is not torch.nn.Linear):
            raise ValueError('The text/time conditioning contract needs explicit inputs.')
        # Choose one time ID and solve the remaining pooled-text width jointly.
        # This is a compatible example, not recovered SDXL image metadata.
        time_width = model.add_time_proj.num_channels
        total_width = model.add_embedding.linear_1.in_features
        if type(time_width) is not int or time_width < 1 or type(total_width) is not int:
            raise ValueError('Declared text/time dimensions need explicit inputs.')
        kwargs['added_cond_kwargs'] = dict(kind='dict', items=dict(
            text_embeds=tensor([1, total_width - time_width]),
            time_ids=tensor([1, 1], integer=True)))
    return dict(binding=dict(args=[], kwargs=kwargs),
                detail='Example inputs from the installed conditional UNet API and constructed dimensions: '
                'batch 1, 8 context tokens, timestep 1; text/time mode uses one time ID and a compatible pooled-text width. '
                'These are editable examples, not recovered task settings. The capture must verify this candidate.')
