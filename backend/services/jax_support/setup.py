"""JAX bindings and callable construction, only in the approved capture worker."""
import inspect
import math
import functools
import sys
from pathlib import Path

from ..validation import contracts as c
from ..validation.setup import bind_call, load_callable, load_module, setup_stage, workspace_directory


def callable_in_trace(function):
    """Give NNX state wrappers the current trace context without copying arrays."""
    nnx = sys.modules.get('flax.nnx')
    if nnx is None:
        return function
    if isinstance(function, functools.partial):
        args, kwargs = nnx.clone((function.args, function.keywords))
        return functools.partial(callable_in_trace(function.func), *args, **kwargs)
    if inspect.ismethod(function) and isinstance(function.__self__, nnx.Module):
        return getattr(nnx.clone(function.__self__), function.__name__)
    return nnx.clone(function) if isinstance(function, nnx.Module) else function


def resolve_device(name):
    import jax
    if name == 'cpu':
        backend, index = 'cpu', 0
    elif name.startswith('cuda:') and name[5:].isdigit():
        backend, index = 'cuda', int(name[5:])
    else:
        raise ValueError('JAX setup requires cpu or cuda:N.')
    try:
        devices = jax.devices(backend)
    except RuntimeError as error:
        raise ValueError(f'JAX device {name} is unavailable; check the selected Python and JAX accelerator installation.') from error
    if index >= len(devices):
        raise ValueError(f'JAX device {name} is unavailable; {len(devices)} {backend} devices are visible.')
    return devices[index]


def materialize(value):
    import jax
    import jax.numpy as jnp
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, c.FloatValue):
        return value.value
    if isinstance(value, c.NoneValue):
        return None
    if isinstance(value, c.SequenceValue):
        items = [materialize(item) for item in value.items]
        return tuple(items) if value.kind == 'tuple' else items
    if isinstance(value, c.DictValue):
        return {key: materialize(item) for key, item in value.items.items()}
    if value.requires_grad:
        raise ValueError('JAX arrays do not have requires_grad; use a transformed JAX callable for gradients.')
    if value.dtype in ('float64', 'int64') and not jax.config.x64_enabled:
        raise ValueError('Enable JAX_ENABLE_X64=1 in the capture environment for 64-bit inputs.')
    if math.prod(value.shape) > 4_194_304:
        raise ValueError('JAX input exceeds 4,194,304 elements.')
    dtype, gen = getattr(jnp, value.dtype), value.generator
    if isinstance(gen, c.Values):
        return jnp.asarray(gen.values, dtype=dtype).reshape(value.shape)
    if isinstance(gen, c.Full):
        return jnp.full(value.shape, gen.value, dtype=dtype)
    key = jax.random.key(gen.seed)
    if isinstance(gen, c.RandInt):
        return jax.random.randint(key, value.shape, gen.low, gen.high, dtype=dtype)
    if isinstance(gen, c.Normal):
        return jax.random.normal(key, value.shape, dtype=dtype) * gen.std + gen.mean
    return jax.random.uniform(key, value.shape, dtype=dtype, minval=gen.low, maxval=gen.high)


def binding(raw):
    value = c.Binding.model_validate(raw)
    elements = 0
    def count(item):
        nonlocal elements
        if isinstance(item, c.TensorValue):
            elements += math.prod(item.shape)
        elif isinstance(item, c.SequenceValue):
            for child in item.items:
                count(child)
        elif isinstance(item, c.DictValue):
            for child in item.items.values():
                count(child)
    for item in [*value.args, *value.kwargs.values()]:
        count(item)
    if elements > 4_194_304:
        raise ValueError('JAX inputs exceed 4,194,304 total elements.')
    return tuple(materialize(item) for item in value.args), {key: materialize(item) for key, item in value.kwargs.items()}


def input_binding(spec):
    def tensor(value):
        fill = value.get('fill', 'randn')
        return dict(kind='tensor', shape=value['shape'], dtype=value.get('dtype', 'float32'),
                    requires_grad=False, generator=(dict(kind='normal', mean=0, std=1, seed=0)
                    if fill == 'randn' else dict(kind='full', value=int(fill == 'ones'))))
    return dict(args=[tensor(item) for item in spec.get('args', [])],
                kwargs={key: tensor(item) for key, item in spec.get('kwargs', {}).items()})


def check_inputs(args, kwargs):
    import jax
    import numpy as np
    leaves, _ = jax.tree_util.tree_flatten((args, kwargs))
    if len(leaves) > 4096:
        raise ValueError('JAX inputs exceed 4096 leaves.')
    arrays = [item for item in leaves if isinstance(item, (jax.Array, np.ndarray))]
    if len(arrays) > 256 or sum(item.size for item in arrays) > 4_194_304:
        raise ValueError('JAX inputs exceed 256 arrays or 4,194,304 elements.')
    if any(type(item) not in (bool, int, float, str) and not isinstance(item, (jax.Array, np.ndarray)) for item in leaves):
        raise ValueError('JAX inputs require arrays, literal scalars, and Python containers.')


def prepare(file, name, workspace, spec):
    import jax
    if spec['mode'] != 'eval' or spec['gradEnabled']:
        raise ValueError('JAX tracing requires eval and grad_enabled = false; express transforms in the callable.')
    if 'checkpoint' in spec or 'fixture' in spec['inputs']:
        raise ValueError('Use a JAX setup/input factory to load parameters or fixtures.')
    with jax.default_device(resolve_device(spec['device'])):
        directory = workspace_directory(workspace, spec['cwd'])
        target = getattr(load_module(Path(file).resolve(), root=directory), name)
        if not callable(target) or Path(inspect.getfile(inspect.unwrap(target))).resolve() != Path(file).resolve():
            raise ValueError('Select a JAX function or Flax class defined in this file.')
        with setup_stage('materialization', ['inputs'], 'provide_inputs', 'invalid_binding'):
            if 'factory' in spec['inputs']:
                factory = spec['inputs']['factory']
                function = load_callable(workspace, factory)
                bind_call(function, (), factory.get('kwargs', {}), 'inputs')
                payload = function(**factory.get('kwargs', {}))
                if not isinstance(payload, dict):
                    raise ValueError('Input factory must return a named-input mapping or args/kwargs binding.')
                if set(payload) != {'args', 'kwargs'}:
                    payload = dict(args=(), kwargs=payload)
                args, kwargs = payload['args'], payload['kwargs']
                if not isinstance(args, (tuple, list)) or not isinstance(kwargs, dict) or not all(type(key) is str for key in kwargs):
                    raise ValueError('Input factory requires positional args and named kwargs.')
            else:
                args, kwargs = binding(spec['inputs'].get('binding', input_binding(spec['inputs'])))
            check_inputs(args, kwargs)
        state = {}
        with setup_stage('construction', ['init'], 'edit_profile', 'invalid_factory'):
            if 'setup' in spec:
                factory = spec['setup']
                function = load_callable(workspace, factory)
                bind_call(function, (), factory.get('kwargs', {}), 'factory')
                model = function(**factory.get('kwargs', {}))
                if not callable(model):
                    raise ValueError('JAX setup factory must return a callable.')
                if isinstance(target, type) and type(model) is not target:
                    raise ValueError('A Flax setup factory must return the selected class, bound if using saved variables.')
            elif isinstance(target, type):
                init_args, init_kwargs = binding(spec.get('constructorBinding', dict(args=[], kwargs=spec.get('constructorArgs', {}))))
                bind_call(target, init_args, init_kwargs, 'init')
                model = target(*init_args, **init_kwargs)
            else:
                init = spec.get('constructorBinding', dict(args=[], kwargs=spec.get('constructorArgs', {})))
                if init.get('args') or init.get('kwargs'):
                    raise ValueError('JAX functions use inputs, not constructor arguments.')
                model = target
            if isinstance(target, type):
                from flax import linen as nn
                if isinstance(model, nn.Module):
                    if model.scope is None:
                        if spec['seed'] is None:
                            raise ValueError('Provide seed for Flax initialization or a factory returning a bound module.')
                        variables = model.init(jax.random.key(spec['seed']), *args, **kwargs)
                        model = model.bind(variables)
                    for path, array in jax.tree_util.tree_flatten_with_path(model.variables)[0]:
                        parts = [str(getattr(key, 'key', getattr(key, 'idx', ''))) for key in path]
                        state[id(array)] = dict(kind='parameter' if parts[0] == 'params' else 'buffer', name='/'.join(parts))
                else:
                    from flax import nnx
                    if not isinstance(model, nnx.Module):
                        raise ValueError('JAX class targets must be Flax Linen or NNX modules.')
            bind_call(model, args, kwargs, 'inputs')
    return model, tuple(args), kwargs, state
