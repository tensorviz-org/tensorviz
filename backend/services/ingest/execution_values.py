"""Bounded immutable numerical evidence, captured outside Torch dispatch recording."""
from __future__ import annotations

import base64
import hashlib
import json
import math
import sys
from typing import Any

import torch
from torch.utils._python_dispatch import _disable_current_modes

from .execution_setup import tensor_bytes, tensor_digest

MAX_OUTPUT_BYTES = 4 * 1024 * 1024
MAX_OUTPUT_LEAVES = 256
MAX_MODULE_DIGEST_BYTES = 64 * 1024 * 1024
NUMERIC_DTYPES = {'float16', 'bfloat16', 'float32', 'float64', 'int8', 'uint8', 'int16', 'int32', 'int64', 'bool', 'complex64', 'complex128'}


def capture_output(value: Any, max_bytes: int = MAX_OUTPUT_BYTES, include_values: bool = True) -> dict:
    leaves: list[dict] = []
    containers: list[dict] = []
    unavailable: list[str] = []
    remaining = max_bytes
    visited = 0

    def visit(item: Any, path: str, depth: int = 0) -> None:
        nonlocal remaining, visited
        visited += 1
        if depth > 16 or visited > 4096 or len(leaves) >= MAX_OUTPUT_LEAVES:
            unavailable.append('output_structure_limit')
            return
        if isinstance(item, torch.Tensor):
            dtype = str(item.dtype).removeprefix('torch.')
            size = item.numel() * item.element_size()
            leaf = {'path': path, 'kind': 'tensor', 'shape': list(item.shape), 'dtype': dtype,
                    'device': str(item.device), 'numel': item.numel(), 'bytes': size,
                    'digest': None, 'nonfinite': None, 'values': None, 'unavailable': None}
            try:
                if item.layout != torch.strided or item.is_quantized:
                    raise ValueError('unsupported_tensor_layout')
                if not include_values and size > remaining:
                    raise ValueError('module_digest_budget')
                leaf['digest'] = tensor_digest(item)
                leaf['nonfinite'] = int((~torch.isfinite(item)).sum().item()) if item.is_floating_point() or item.is_complex() else 0
                if include_values and size <= remaining and dtype in NUMERIC_DTYPES:
                    raw = tensor_bytes(item)
                    leaf['values'] = {'encoding': 'base64', 'byteOrder': sys.byteorder, 'data': base64.b64encode(raw).decode('ascii')}
                    remaining -= size
                elif include_values:
                    leaf['unavailable'] = 'output_value_budget' if size > remaining else 'unsupported_numeric_dtype'
                else:
                    remaining -= size
            except (TypeError, ValueError, RuntimeError) as error:
                leaf['unavailable'] = str(error) if type(error) is ValueError and str(error) in ('unsupported_tensor_layout', 'module_digest_budget') else type(error).__name__
            leaves.append(leaf)
        elif isinstance(item, (tuple, list)):
            containers.append({'path': path, 'kind': 'tuple' if isinstance(item, tuple) else 'list', 'length': len(item)})
            for index, child in enumerate(item):
                if visited >= 4096 or len(leaves) >= MAX_OUTPUT_LEAVES:
                    unavailable.append('output_structure_limit'); break
                visit(child, f'{path}[{index}]', depth + 1)
        elif isinstance(item, dict) and all(isinstance(key, str) and len(key) <= 512 for key in item):
            if len(item) > 4096:
                unavailable.append('output_structure_limit'); return
            containers.append({'path': path, 'kind': 'dict', 'keys': list(item)})
            for key, child in item.items():
                if visited >= 4096 or len(leaves) >= MAX_OUTPUT_LEAVES:
                    unavailable.append('output_structure_limit'); break
                visit(child, f'{path}[{json.dumps(key)}]', depth + 1)
        elif item is None or type(item) in (bool, int, float, str):
            encoded = json.dumps(item, allow_nan=True)
            if len(encoded) > 8192:
                unavailable.append('output_literal_limit'); return
            leaf = {'path': path, 'kind': type(item).__name__, 'digest': hashlib.sha256(encoded.encode()).hexdigest(),
                    'nonfinite': int(type(item) is float and not math.isfinite(item))}
            # Preserve integers as decimal text so JavaScript cannot round int64.
            if type(item) in (int, float, bool):
                leaf['value'] = str(item) if type(item) is int else item if leaf['nonfinite'] == 0 else str(item)
            leaves.append(leaf)
        else:
            unavailable.append('unsupported_output_type')

    # Hooks fire inside the provenance mode. Measurement must not create new
    # graph operations, retain mutable tensors, or build an autograd graph.
    with _disable_current_modes(), torch.no_grad():
        try:
            visit(value, 'output')
        except Exception as error:
            unavailable.append(f'measurement_error:{type(error).__name__}')
    identity = {'containers': containers, 'leaves': [{key: value for key, value in leaf.items()
                if key in ('path', 'kind', 'shape', 'dtype', 'digest')} for leaf in leaves]}
    complete = not unavailable and all(leaf.get('digest') is not None for leaf in leaves)
    return {'containers': containers, 'leaves': leaves, 'truncated': sorted(set(unavailable)),
            'digest': hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() if complete else None,
            'bytesCaptured': max_bytes - remaining,
            'allFinite': all(leaf.get('nonfinite') == 0 for leaf in leaves) if complete and all(leaf.get('nonfinite') is not None for leaf in leaves) else None}
