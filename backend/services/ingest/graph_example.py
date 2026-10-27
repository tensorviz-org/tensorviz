"""Propose an editable input binding from a constructed model, never a success verdict."""
from __future__ import annotations

import inspect
import math
from types import UnionType
from typing import Union, get_args, get_origin

import torch


def propose_constructor(target) -> dict:
    """A required config may use its declared class; other gaps need setup."""
    signature = inspect.signature(target)
    try:
        signature.bind()
        return {}
    except TypeError:
        pass
    parameter = signature.parameters.get('config')
    if (parameter is None or parameter.default is not inspect.Parameter.empty
            or parameter.kind not in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)):
        raise ValueError('No supported required config parameter was declared.')
    # Check all other required arguments before constructing any config object.
    signature.bind(config=None)
    config_class = inspect.getattr_static(target, 'config_class', None)
    if not isinstance(config_class, type):
        raise ValueError('The model must declare a config_class type with defaults.')
    inspect.signature(config_class).bind()
    return {'config': config_class()}


def _scalar_example(annotation):
    # Do not evaluate string/forward annotations or choose among opaque types.
    choices = get_args(annotation) if get_origin(annotation) in (Union, UnionType) else (annotation,)
    if not all(any(choice is allowed for allowed in (int, float, bool, torch.Tensor)) for choice in choices):
        return None
    for kind, value in ((int, 1), (float, dict(kind='float', value=1.0)), (bool, False)):
        if any(choice is kind for choice in choices):
            return value
    return None


def propose_input(model: torch.nn.Module) -> dict:
    signature = inspect.signature(model.forward)
    parameters = list(signature.parameters.values())
    required = [p for p in parameters if p.default is inspect.Parameter.empty
                and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    named = {p.name: p for p in parameters if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
    primary = inspect.getattr_static(model, 'main_input_name', None)
    primary = named.get(primary) if isinstance(primary, str) else None
    # This is an execution worker: the library's example property may run code,
    # but forward is never called. Do not accept unrelated keys via **kwargs.
    example = getattr(model, 'dummy_inputs', None)
    if isinstance(example, dict) and example and example.keys() <= named.keys() and (
            primary is None or primary.name in example):
        from ..validation.contracts import Binding
        try:
            signature.bind(**example)
        except TypeError as error:
            raise ValueError('The declared input example does not bind to forward.') from error
        if len(example) > 64 or any(type(value) is not torch.Tensor or value.layout != torch.strided
                or value.device.type != 'cpu' for value in example.values()):
            raise ValueError('Declared input examples must contain bounded CPU tensors.')
        if sum(value.numel() for value in example.values()) > 4096:
            raise ValueError('The declared input example exceeds the literal-value budget.')
        binding = dict(args=[], kwargs={key: dict(kind='tensor', shape=list(value.shape),
            dtype=str(value.dtype).removeprefix('torch.'), requires_grad=False,
            generator=dict(kind='values', values=value.detach().reshape(-1).tolist()))
            for key, value in example.items()})
        Binding.model_validate(binding)
        return dict(binding=binding, detail='Example inputs from the constructed model\'s dummy_inputs; '
                    'the capture must verify this candidate. Change inputs to edit it.')
    from .graph_attention_inputs import propose_attention_input
    from .graph_diffusion_inputs import propose_diffusion_input
    from .graph_timm_inputs import propose_timm_input
    library = propose_attention_input(model)
    if library is None:
        library = propose_diffusion_input(model)
    if library is None:
        library = propose_timm_input(model)
    if library is not None:
        from ..validation.contracts import Binding
        signature.bind(*library['binding']['args'], **library['binding']['kwargs'])
        Binding.model_validate(library['binding'])
        return library
    scalars = {p.name: value for p in required if (value := _scalar_example(p.annotation)) is not None}
    unresolved = [p for p in required if p.name not in scalars]

    def complete(values, detail):
        from ..validation.contracts import Binding
        binding = dict(args=[], kwargs={})
        for index, parameter in enumerate(parameters):
            if parameter.name not in values:
                continue
            if parameter.kind is parameter.POSITIONAL_ONLY:
                if len(binding['args']) != index:
                    raise ValueError('Skipped positional defaults need an explicit input binding.')
                binding['args'].append(values[parameter.name])
            else:
                binding['kwargs'][parameter.name] = values[parameter.name]
        signature.bind(*binding['args'], **binding['kwargs'])
        Binding.model_validate(binding)
        scalar_detail = ' Required scalar examples use declared types (int=1, float=1.0, bool=False).' if scalars else ''
        return dict(binding=binding, detail=f'Example {detail}.{scalar_detail} '
                    'The capture must verify this candidate. Change inputs to edit it.')

    variadic = any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in parameters)
    if primary is not None and primary.name not in scalars and all(p.name == primary.name for p in unresolved):
        field = primary
    elif len(unresolved) == 1 and not variadic:
        field = unresolved[0]
    elif not unresolved and not variadic:
        return complete(scalars, 'call uses forward defaults and declared scalar types' if scalars
                        else 'call uses the forward defaults; no declared tensor input example was available')
    else:
        raise ValueError('Forward needs a declared input example or one supported tensor input with typed scalars.')
    if scalars and field.annotation is not inspect.Parameter.empty:
        choices = get_args(field.annotation) if get_origin(field.annotation) in (Union, UnionType) else (field.annotation,)
        if (not any(choice is torch.Tensor for choice in choices)
                or any(choice is not torch.Tensor and choice is not type(None) for choice in choices)):
            raise ValueError('The remaining argument does not declare a supported tensor type.')
    supported = (torch.nn.Linear, torch.nn.Embedding, torch.nn.RNN, torch.nn.GRU, torch.nn.LSTM,
                 torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
    layer = None
    for index, module in enumerate(model.modules()):
        if index >= 4096:
            raise ValueError('The constructed model exceeds the input inspection budget.')
        if type(module) in supported:
            layer = module
            break
    if layer is None:
        raise ValueError('No supported tensor input boundary was found in the constructed model.')
    generator = dict(kind='normal', mean=0, std=1, seed=0)
    dtype = str(next(layer.parameters()).dtype).removeprefix('torch.')
    if type(layer) is torch.nn.Embedding:
        if layer.num_embeddings < 1:
            raise ValueError('An empty vocabulary cannot supply an example token.')
        shape, dtype = [1, 8], 'int64'
        generator = dict(kind='randint', low=0, high=min(layer.num_embeddings, 256), seed=0)
    elif type(layer) is torch.nn.Linear:
        shape = [1, layer.in_features]
    elif type(layer) in (torch.nn.RNN, torch.nn.GRU, torch.nn.LSTM):
        shape = [1, 8, layer.input_size] if layer.batch_first else [8, 1, layer.input_size]
    else:
        rank = len(layer.kernel_size)
        spatial = None
        # Read plain stored metadata; do not invoke user properties to choose a size.
        metadata = dict(vars(model))
        config = metadata.get('config')
        if config is not None:
            try:
                stored = vars(config)
                for name in ('image_size', 'img_size'):
                    if name in stored:
                        metadata.setdefault(name, stored[name])
            except TypeError:
                pass  # A non-object config supplies no stored image metadata.
        for name in ('image_size', 'img_size'):
            if name in metadata:
                value = metadata[name]
                spatial = [value] * rank if type(value) is int else value
                if (not isinstance(spatial, (tuple, list)) or len(spatial) != rank
                        or any(type(n) is not int or n < 1 for n in spatial)):
                    raise ValueError('Declared image dimensions need an explicit input.')
                break
        if spatial is None:
            size = 16 if rank == 3 else 64
            spatial = [max(size, dilation * (kernel - 1) + 1)
                       for kernel, dilation in zip(layer.kernel_size, layer.dilation)]
            if layer.kernel_size == layer.stride:
                spatial = [math.ceil(n / stride) * stride for n, stride in zip(spatial, layer.stride)]
        shape = [1, layer.in_channels, *spatial]
    if (any(type(n) is not int or n < 1 for n in shape) or math.prod(shape) > 4_194_304
            or dtype not in ('float16', 'float32', 'float64', 'bfloat16', 'int64')):
        raise ValueError('The input example exceeds the supported tensor budget.')
    tensor = dict(kind='tensor', shape=list(shape), dtype=dtype,
                  requires_grad=False, generator=generator)
    return complete({**scalars, field.name: tensor}, f'input from constructed {type(layer).__name__} dimensions')


def run_probe(args, envelope, plan, plan_digest):
    """Private preparation only. A later saved-profile capture owns all measurements."""
    from contextlib import redirect_stdout
    import sys
    from .execution_setup import prepare_execution
    from .execution_spec import workspace_directory
    from .worker import EXIT_OK, EXIT_CONSTRUCT_FAILED, _fail
    from ..semantic.execution_plan import verify_saved_plan

    proposal = {}
    def inputs(model):
        proposal.update(propose_input(model))
        return proposal['binding']
    def constructor(target):
        kwargs = propose_constructor(target)
        if kwargs:
            proposal['configDefaults'] = True
        return kwargs

    try:
        with redirect_stdout(sys.stderr):
            verify_saved_plan(plan, args.workspace)
            prepare_execution(args.file, args.cls, args.workspace, plan.execution,
                              import_root=workspace_directory(args.workspace, plan.execution['cwd']),
                              input_proposal=inputs,
                              constructor_proposal=constructor if args.prepare_config_defaults else None)
            verify_saved_plan(plan, args.workspace)
        if proposal.get('configDefaults'):
            proposal['detail'] = ('Example uses config_class() defaults, saved in an editable Python factory. '
                                  + proposal['detail'])[:512]
        envelope.result = {'graphExample': proposal,
                           'diagnostics': {'execution': {'planDigest': plan_digest, 'planCurrent': True}}}
        return envelope.emit(EXIT_OK)
    except Exception as error:
        _fail(envelope, error)
        return envelope.emit(EXIT_CONSTRUCT_FAILED)
