"""Reproducible, explicit setup for a single isolated execution worker."""
from __future__ import annotations

import ctypes
import hashlib
import inspect
import json
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch._subclasses.fake_tensor import is_fake

from .execution_spec import validate_execution_spec, workspace_file
from .tensor_inputs import make_tensor_inputs
from ..validation.setup import (MAX_FIXTURE_BYTES, load_module, load_callable, factory_model,
                                configure_runtime, bind_call, setup_stage, ExecutionRequirement)

MAX_CHECKPOINT_BYTES = 2 * 1024 * 1024 * 1024


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class TensorValuesUnavailable(ValueError):
    """The tensor has shape, dtype and device but no values to read or digest.

    ``str(error)`` is the reason: ``fake_tensor`` or ``meta_tensor``.
    """


def values_unavailable(tensor: torch.Tensor) -> str | None:
    """Why ``tensor`` has no readable values, or None for an ordinary tensor.

    A FakeTensor's ``data_ptr()`` is not the address of any values (it is 0, or
    unrelated memory), so ``ctypes.string_at`` on it hashes arbitrary memory or
    crashes the worker. A meta tensor has no storage at all.
    """
    if is_fake(tensor):
        return 'fake_tensor'
    if tensor.device.type == 'meta':
        return 'meta_tensor'
    return None


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Copy exactly the live contiguous CPU tensor extent; no NumPy dependency.

    The local tensor keeps its allocation alive for the duration of string_at.
    CUDA/MPS transfers finish before reading and no address is exposed in evidence.
    Fake and meta tensors raise TensorValuesUnavailable before any memory is read.
    """
    reason = values_unavailable(tensor)
    if reason is not None:
        raise TensorValuesUnavailable(reason)
    cpu = tensor.detach().resolve_conj().resolve_neg().to('cpu').contiguous()
    if is_fake(cpu):  # a real tensor copied while a FakeTensorMode is active
        raise TensorValuesUnavailable('fake_tensor')
    size = cpu.numel() * cpu.element_size()
    return ctypes.string_at(cpu.data_ptr(), size) if size else b''


def tensor_digest(tensor: torch.Tensor) -> str:
    if tensor.layout != torch.strided or tensor.is_quantized:
        raise ValueError('Tensor fingerprints currently require dense, unquantized tensors.')
    reason = values_unavailable(tensor)
    if reason is not None:  # even when empty: a fake state never claims a value digest
        raise TensorValuesUnavailable(reason)
    digest = hashlib.sha256()
    digest.update(json.dumps([str(tensor.dtype), list(tensor.shape)]).encode())
    flat = tensor.detach().resolve_conj().resolve_neg().reshape(-1)
    # Stream parameter values so a large checkpoint does not need a second CPU copy.
    for start in range(0, flat.numel(), 262_144):
        digest.update(tensor_bytes(flat[start:start + 262_144]))
    return digest.hexdigest()


def describe_tree(value: Any, aliases: dict | None = None) -> dict:
    """Identity and shape, never a tensor's raw values or a repr of user objects."""
    aliases = {} if aliases is None else aliases
    if isinstance(value, torch.Tensor):
        storage = (str(value.device), value.untyped_storage()._cdata)
        if storage not in aliases:
            aliases[storage] = len(aliases)
        return {'kind': 'tensor', 'storageGroup': aliases[storage], 'storageOffset': value.storage_offset(), 'shape': list(value.shape), 'dtype': str(value.dtype),
                'device': str(value.device), 'stride': list(value.stride()),
                'requiresGrad': value.requires_grad, 'bytes': value.numel() * value.element_size(),
                'digest': tensor_digest(value)}
    if isinstance(value, (tuple, list)):
        return {'kind': type(value).__name__, 'items': [describe_tree(item, aliases) for item in value]}
    if isinstance(value, dict):
        return {'kind': 'dict', 'keys': list(value), 'items': {key: describe_tree(item, aliases) for key, item in value.items()}}
    encoded = json.dumps(value, allow_nan=False, sort_keys=True, separators=(',', ':'))
    return {'kind': type(value).__name__, 'digest': hashlib.sha256(encoded.encode()).hexdigest()}


def tree_digest(description: dict) -> str:
    return hashlib.sha256(json.dumps(description, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def state_identity(model: torch.nn.Module) -> dict:
    try:
        description = describe_tree(dict(model.state_dict()))
        return {'digest': tree_digest(description), 'parametersAndBuffers': description, 'unavailable': None}
    except (ValueError, TypeError, RuntimeError) as exc:
        reason = str(exc) if isinstance(exc, TensorValuesUnavailable) else type(exc).__name__
        return {'digest': None, 'parametersAndBuffers': None, 'unavailable': reason}


def rng_identity(device: torch.device) -> dict:
    """Record built-in RNGs actually present at the start of execution."""
    result = {'torchCpuDigest': tensor_digest(torch.get_rng_state()),
              'pythonDigest': hashlib.sha256(json.dumps(random.getstate()).encode()).hexdigest(),
              'device': str(device), 'deviceDigest': None, 'numpy': 'not_loaded',
              'numpyDigest': None, 'complete': True}
    np = sys.modules.get('numpy')
    if np is not None:
        state = np.random.get_state()
        digest = hashlib.sha256(json.dumps([state[0], *state[2:]]).encode())
        digest.update(state[1].tobytes())
        result.update(numpy=str(np.__version__), numpyDigest=digest.hexdigest())
    try:
        if device.type == 'cuda':
            result['deviceDigest'] = tensor_digest(torch.cuda.get_rng_state(device))
        elif device.type == 'mps':
            result['deviceDigest'] = tensor_digest(torch.mps.get_rng_state())
    except (RuntimeError, AttributeError) as error:
        result.update(complete=False, deviceUnavailable=type(error).__name__)
    return result


def environment(device: torch.device) -> dict:
    details = None
    if device.type == 'cuda':
        properties = torch.cuda.get_device_properties(device)
        details = {'name': properties.name, 'totalMemoryBytes': properties.total_memory,
                   'capability': list(torch.cuda.get_device_capability(device))}
    return {'python': platform.python_version(), 'interpreter': sys.executable,
            'torch': str(torch.__version__), 'numpy': getattr(sys.modules.get('numpy'), '__version__', None),
            'platform': platform.system(), 'platformRelease': platform.release(), 'architecture': platform.machine(),
            'pythonImplementation': platform.python_implementation(),
            'cudaBuild': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'device': str(device), 'deviceDetails': details, 'threads': torch.get_num_threads(),
            'deterministicAlgorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnnBenchmark': torch.backends.cudnn.benchmark,
            'float32MatmulPrecision': torch.get_float32_matmul_precision(),
            'cudnnAllowTf32': torch.backends.cudnn.allow_tf32, 'cudaMatmulAllowTf32': torch.backends.cuda.matmul.allow_tf32,
            'interopThreads': torch.get_num_interop_threads(), 'mkldnnEnabled': torch.backends.mkldnn.enabled}


def _load_tensor_file(path: Path) -> tuple[Any, str]:
    # No arbitrary-pickle fallback. A custom setup recipe can explicitly load a
    # project's specialized format, under the same Workspace Trust execution gate.
    before = file_digest(path)
    value = torch.load(path, map_location='cpu', weights_only=True)
    if file_digest(path) != before:
        raise ValueError('An execution file changed while it was being loaded.')
    return value, before


def _check_batch(value: Any, max_elements: int | None = None) -> None:
    entries = tensors = size = elements = 0
    def visit(item: Any, depth: int = 0) -> None:
        nonlocal entries, tensors, size, elements
        entries += 1
        if depth > 16 or entries > 4096:
            raise ValueError('The fixture is too deeply nested or has too many entries.')
        if isinstance(item, torch.Tensor):
            tensors += 1
            elements += item.numel()
            if max_elements is not None and elements > max_elements:
                raise ValueError('Input factory exceeds the total tensor-element budget.')
            size += item.numel() * item.element_size()
            if tensors > 256 or size > MAX_FIXTURE_BYTES:
                raise ValueError('The fixture exceeds 256 tensors or 64 MiB of tensor data.')
            if item.layout != torch.strided or item.is_quantized:
                raise ValueError('Fixture tensors must be dense and unquantized.')
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child, depth + 1)
        elif isinstance(item, dict) and all(isinstance(k, str) for k in item):
            for child in item.values():
                visit(child, depth + 1)
        elif item is None or type(item) in (bool, int, float, str):
            if len(json.dumps(item, allow_nan=False)) > 8192:
                raise ValueError('A fixture literal exceeds 8 KiB.')
        else:
            raise ValueError('Fixture data must contain tensors and JSON literals only.')
    visit(value)


def factory_inputs(workspace: str, spec: dict, device: torch.device) -> tuple[tuple, dict]:
    """Execute one explicit input fixture in the worker; preserve values and aliases."""
    from ..validation.setup import workspace_file
    file = workspace_file(workspace, spec['file'], MAX_FIXTURE_BYTES)
    before = file_digest(file)
    function = load_callable(workspace, spec)
    bind_call(function, (), spec.get('kwargs', {}), 'inputs')
    payload = function(**spec.get('kwargs', {}))
    if not isinstance(payload, dict):
        raise ValueError('An input factory must return named inputs or an args/kwargs binding.')
    if set(payload) != {'args', 'kwargs'}:
        payload = {'args': (), 'kwargs': payload}
    if (not isinstance(payload['args'], (tuple, list)) or not isinstance(payload['kwargs'], dict)
            or any(not isinstance(key, str) for key in payload['kwargs'])):
        raise ValueError('Input factory args must be a sequence and kwargs must have string keys.')
    _check_batch(payload, max_elements=4_194_304)
    if file_digest(file) != before:
        raise ValueError('Input factory source changed during setup.')
    moved = move_tree(payload, device)
    return tuple(moved['args']), moved['kwargs']


def move_tree(value: Any, device: torch.device, memo: dict[int, torch.Tensor] | None = None) -> Any:
    memo = {} if memo is None else memo
    if isinstance(value, torch.Tensor):
        if id(value) not in memo:
            memo[id(value)] = value.to(device)
        return memo[id(value)]
    if isinstance(value, tuple):
        return tuple(move_tree(v, device, memo) for v in value)
    if isinstance(value, list):
        return [move_tree(v, device, memo) for v in value]
    if isinstance(value, dict):
        return {k: move_tree(v, device, memo) for k, v in value.items()}
    return value


@dataclass
class PreparedExecution:
    model: torch.nn.Module
    args: tuple
    kwargs: dict
    batch: Any
    evidence: dict


def _profile_binding(binding: dict, torch, device, field="inputs") -> tuple[tuple, dict]:
    """Materialise a saved tensorviz.toml binding exactly as the profile wrote it.

    The profile grammar is richer than this module's own tensor descriptions — it carries
    `full` with any value, `randint` bounds, literal `values`, per-tensor seeds and
    `requires_grad` — so it is validated and built by the contract's own materialiser
    rather than approximated into zeros/ones/randn. Approximating would substitute an
    input the user did not ask for, which is the exact failure explicit profiles exist to
    prevent.
    """
    from ..validation import contracts as _contracts
    from ..validation.model_run import materialize
    with setup_stage("materialization", [field], "edit_profile", "invalid_config"):
        parsed = _contracts.Binding.model_validate(binding)
        args = tuple(materialize(item, torch, device) for item in parsed.args)
        kwargs = {name: materialize(item, torch, device) for name, item in parsed.kwargs.items()}
    return args, kwargs


def prepare_execution(file: str, cls: str, workspace: str, spec: dict,
                      default_constructor: dict | None = None, *,
                      import_root: Path | None = None, input_proposal=None, constructor_proposal=None) -> PreparedExecution:
    spec = validate_execution_spec(spec)
    if input_proposal is not None and (spec['inputs'] != {'binding': {'args': [], 'kwargs': {}}}
            or spec['device'] != 'cpu' or spec['mode'] != 'eval' or spec['gradEnabled'] or 'checkpoint' in spec):
        raise ValueError('Input preparation cannot replace explicit inputs or checkpoint settings.')
    if constructor_proposal is not None and (input_proposal is None or 'setup' in spec
            or spec.get('constructorArgs') or spec.get('constructorBinding') != {'args': [], 'kwargs': {}}):
        raise ValueError('Config preparation cannot replace an explicit constructor or factory.')
    workspace_path = Path(workspace).resolve(strict=True)
    file_path = Path(file).resolve(strict=True)
    if not workspace_path.is_dir() or not file_path.is_relative_to(workspace_path):
        raise ValueError('The model must be inside the execution workspace.')
    # Resolve every requested file before importing the model or setup code.
    def requested_file(field, descriptor, key, limit):
        if key not in descriptor:
            return None
        with setup_stage("configuration", [field, key], "edit_profile", "invalid_config"):
            return workspace_file(workspace, descriptor[key], limit)
    fixture = requested_file('inputs', spec['inputs'], 'fixture', MAX_FIXTURE_BYTES)
    checkpoint = requested_file('checkpoint', spec.get('checkpoint', {}), 'file', MAX_CHECKPOINT_BYTES)
    setup_file = requested_file('factory', spec.get('setup', {}), 'file', MAX_FIXTURE_BYTES)
    input_file = requested_file('inputs', spec['inputs'].get('factory', {}), 'file', MAX_FIXTURE_BYTES)
    with setup_stage("environment", ["environment"], "select_environment"):
        device = configure_runtime(torch, spec)
    source_digest = file_digest(file_path)
    setup_digest = file_digest(setup_file) if setup_file else None
    input_digest = file_digest(input_file) if input_file else None
    constructor = spec.get('constructorArgs', default_constructor or {})
    constructor_args: tuple = ()
    if 'constructorBinding' in spec:
        # An explicit `init` from the saved profile outranks anything remembered or
        # declared in a comment: the profile is the record of the invocation meant.
        binding = _profile_binding(spec['constructorBinding'], torch, device, "init")
        constructor_args, constructor = binding[0], binding[1]
    with setup_stage("import", ["model"], "fix_source", "invalid_source"):
        target = getattr(load_module(file_path, root=import_root), cls)
    if 'setup' in spec:
        model = factory_model(workspace, spec['setup'], target, file_path)
    else:
        if constructor_proposal is not None:
            with setup_stage("construction", ["init"], "edit_profile"):
                constructor = constructor_proposal(target)
        bind_call(target, constructor_args, constructor, "init")
        with setup_stage("construction", ["init"], "edit_profile"):
            model = target(*constructor_args, **constructor)
    if (not isinstance(model, torch.nn.Module) or type(model).__name__ != cls
            or Path(inspect.getfile(type(model))).resolve() != file_path):
        raise ExecutionRequirement("construction", "invalid_factory", ["model"], "provide_factory")
    checkpoint_info = None
    if checkpoint:
        with setup_stage("materialization", ["checkpoint"], "edit_profile", "invalid_config"):
            state, digest = _load_tensor_file(checkpoint)
            if 'key' in spec['checkpoint']:
                state = state[spec['checkpoint']['key']]
            report = model.load_state_dict(state, strict=spec['checkpoint']['strict'])
        checkpoint_info = {**spec['checkpoint'], 'digest': digest,
                           'missingKeys': report.missing_keys, 'unexpectedKeys': report.unexpected_keys}
    with setup_stage("materialization", ["device"], "select_environment"):
        model.to(device)
    with setup_stage("construction", ["mode"], "edit_profile"):
        if spec['mode'] != 'preserve':
            model.train(spec['mode'] == 'train')
    if input_proposal is not None:
        spec['inputs'] = {'binding': input_proposal(model)}
    fixture_digest = None
    with setup_stage("materialization", ["inputs"], "edit_profile", "invalid_config"):
        if fixture:
            payload, fixture_digest = _load_tensor_file(fixture)
            if not isinstance(payload, dict) or payload.keys() - {'args', 'kwargs', 'batch'}:
                raise ValueError('A fixture must contain args, kwargs and optionally batch.')
            args, kwargs, batch = payload.get('args', ()), payload.get('kwargs', {}), payload.get('batch')
            if not isinstance(args, (list, tuple)) or not isinstance(kwargs, dict) or any(not isinstance(k, str) for k in kwargs):
                raise ValueError('Fixture args must be a sequence and kwargs a string-keyed object.')
            _check_batch(payload)
            moved = move_tree({'args': tuple(args), 'kwargs': kwargs, 'batch': batch}, device)
            args, kwargs, batch = moved['args'], moved['kwargs'], moved['batch']
        elif 'factory' in spec['inputs']:
            args, kwargs = factory_inputs(workspace, spec['inputs']['factory'], device)
            batch = None
        elif 'binding' in spec['inputs']:
            args, kwargs = _profile_binding(spec['inputs']['binding'], torch, device)
            batch = None
        else:
            descriptions = spec['inputs']
            keys = list(descriptions['kwargs'])
            all_specs = descriptions['args'] + [descriptions['kwargs'][k] for k in keys]
            # The existing factory accepts up to eight inputs; preserve one generator
            # across the full explicit request instead of restarting it for each input.
            generator = torch.Generator(device='cpu').manual_seed(spec['seed']) if spec['seed'] is not None else None
            tensors = []
            for description in all_specs:
                if description['fill'] == 'randn':
                    tensor = torch.randn(description['shape'], dtype=getattr(torch, description['dtype']), generator=generator, device='cpu')
                else:
                    tensor = make_tensor_inputs([description])[0][0]
                tensors.append(tensor.to(device))
            n = len(descriptions['args'])
            args, kwargs, batch = tuple(tensors[:n]), dict(zip(keys, tensors[n:])), None
    bind_call(model.forward, args, kwargs, "inputs")
    bind_call(model.__call__, args, kwargs, "inputs")
    if (file_digest(file_path) != source_digest or (setup_file and file_digest(setup_file) != setup_digest)
            or (input_file and file_digest(input_file) != input_digest)):
        raise ValueError('Model or setup source changed during execution setup.')
    observed = describe_tree({'args': args, 'kwargs': kwargs, 'batch': batch})
    evidence = {'version': 1, 'sourceDigest': source_digest,
                'setup': {'factory': spec.get('setup'), 'factoryDigest': setup_digest,
                          'constructorArgs': None if setup_file else constructor, 'checkpoint': checkpoint_info},
                'inputs': {'description': observed, 'digest': tree_digest(observed),
                           'fixture': spec['inputs'].get('fixture'), 'fixtureDigest': fixture_digest,
                           'profileBinding': 'binding' in spec['inputs'],
                           **({'factory': spec['inputs']['factory'], 'factoryDigest': input_digest} if input_file else {})},
                'state': state_identity(model), 'seed': spec['seed'], 'mode': spec['mode'],
                'moduleModes': {name: module.training for name, module in model.named_modules()},
                'gradEnabled': spec['gradEnabled'], 'cwd': spec['cwd'], 'environment': environment(device),
                'rngDigest': tensor_digest(torch.get_rng_state()), 'rng': rng_identity(device)}
    return PreparedExecution(model, args, kwargs, batch, evidence)
