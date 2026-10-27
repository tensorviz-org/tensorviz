"""Bounded tensor descriptions. Validation uses only the standard library."""
from __future__ import annotations

import math
from typing import Any

DTYPES = {"float32", "float64", "float16", "bfloat16", "int64", "int32", "bool"}
FLOAT_DTYPES = {"float32", "float64", "float16", "bfloat16"}


def validate_tensor_inputs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not 1 <= len(value) <= 8:
        raise ValueError("Supply one to eight positional tensor descriptions.")
    total = 0
    for item in value:
        if not isinstance(item, dict) or set(item) != {"shape", "dtype", "fill"}:
            raise ValueError("Each tensor requires only shape, dtype and fill.")
        shape, dtype, fill = item["shape"], item["dtype"], item["fill"]
        if (not isinstance(shape, list) or len(shape) > 8
                or any(type(dim) is not int or not 1 <= dim <= 1_000_000 for dim in shape)):
            raise ValueError("A shape has at most eight positive integer dimensions, each at most 1,000,000.")
        if not isinstance(dtype, str) or dtype not in DTYPES:
            raise ValueError("Unsupported tensor dtype.")
        if not isinstance(fill, str) or fill not in {"zeros", "ones", "randn"}:
            raise ValueError("Unsupported tensor fill.")
        if fill == "randn" and dtype not in FLOAT_DTYPES:
            raise ValueError("randn requires a floating-point dtype.")
        total += math.prod(shape)
    if total > 4_194_304:
        raise ValueError("Inputs may contain at most 4,194,304 elements in total.")
    return value


def make_tensor_inputs(value: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    inputs = validate_tensor_inputs(value)
    import torch

    # A local generator makes repeated requests reproducible without changing the
    # model's global RNG. These inputs describe one concrete execution, not all paths.
    generator = torch.Generator(device="cpu").manual_seed(0)
    tensors = []
    for item in inputs:
        dtype = getattr(torch, item["dtype"])
        if item["fill"] == "randn":
            tensor = torch.randn(item["shape"], dtype=dtype, generator=generator, device="cpu")
        else:
            tensor = torch.full(item["shape"], 1 if item["fill"] == "ones" else 0, dtype=dtype, device="cpu")
        tensors.append(tensor)
    return tuple(tensors), {}
