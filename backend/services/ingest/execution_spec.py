"""Execution request validation, deliberately safe to import before Torch or user code."""
from __future__ import annotations

import math
import re
from typing import Any

from ..validation.setup import (IDENTIFIER, _object, _literals, callable_spec,
                                relative_path, workspace_file, workspace_directory)
from .tensor_inputs import validate_tensor_inputs


def validate_execution_spec(value: Any) -> dict:
    value = _object(value, {"framework", "inputs", "constructorArgs", "constructorBinding", "setup", "checkpoint", "device", "mode", "seed", "deterministic", "gradEnabled", "cwd"}, {"inputs"})
    if 'framework' in value and value['framework'] != 'jax':
        raise ValueError('Unsupported execution framework.')
    result = {"device": "cpu", "mode": "eval", "seed": 0, "deterministic": True, "gradEnabled": False, "cwd": ".", **value}
    if result["cwd"] != ".":
        relative_path(result["cwd"])
    if "constructorBinding" in result:
        if "constructorArgs" in result:
            raise ValueError("Use a saved profile init binding or constructorArgs, not both.")
        from ..validation import contracts as _contracts
        _contracts.Binding.model_validate(result["constructorBinding"])
    if not isinstance(result["device"], str) or not re.fullmatch(r"cpu|mps|cuda(?::[0-9]{1,3})?", result["device"]):
        raise ValueError("Choose cpu, mps, cuda or cuda:N.")
    if result["mode"] not in ("eval", "train", "preserve"):
        raise ValueError("Choose eval, train or preserve mode.")
    if ((result["seed"] is not None and (type(result["seed"]) is not int or not 0 <= result["seed"] <= 2**53 - 1))
            or type(result["deterministic"]) is not bool or type(result["gradEnabled"]) is not bool):
        raise ValueError("Seed and deterministic settings are invalid.")
    inputs = _object(result["inputs"], {"fixture", "args", "kwargs", "binding", "factory"})
    if 'factory' in inputs:
        if set(inputs) != {'factory'}:
            raise ValueError('Use an input factory or explicit inputs, not both.')
        result['inputs'] = {'factory': callable_spec(inputs['factory'])}
    elif "binding" in inputs:
        # A saved tensorviz.toml profile, passed through verbatim rather than translated
        # into this module's three-fill vocabulary. The profile grammar can express
        # `full = -1.0`, `randint`, `values` and per-tensor seeds; approximating those
        # would be inventing an input, which is the whole thing explicit profiles exist to
        # stop. services.validation.contracts is the authority, and model_run.materialize
        # builds them.
        if set(inputs) != {"binding"}:
            raise ValueError("Use a saved profile binding, a fixture or tensor descriptions, not a mixture.")
        from ..validation import contracts as _contracts
        _contracts.Binding.model_validate(inputs["binding"])
    elif "fixture" in inputs:
        if set(inputs) != {"fixture"}:
            raise ValueError("Use a fixture or tensor descriptions, not both.")
        relative_path(inputs["fixture"])
    else:
        args, kwargs = inputs.get("args", []), inputs.get("kwargs", {})
        if not isinstance(args, list) or not isinstance(kwargs, dict) or any(not isinstance(k, str) or not IDENTIFIER.fullmatch(k) for k in kwargs):
            raise ValueError("Tensor inputs require args and named kwargs.")
        tensors = args + list(kwargs.values())
        if len(tensors) > 32:
            raise ValueError("Use at most 32 input tensors.")
        for tensor in tensors:
            validate_tensor_inputs([tensor])
        if sum(math.prod(t["shape"]) for t in tensors) > 4_194_304:
            raise ValueError("Inputs exceed 4,194,304 elements.")
        result["inputs"] = {"args": args, "kwargs": kwargs}
    if "constructorArgs" in value:
        _literals(value["constructorArgs"])
    if "setup" in value:
        if "constructorArgs" in value or "constructorBinding" in value:
            raise ValueError("Use either setup or constructor arguments.")
        result["setup"] = callable_spec(value["setup"])
    if "checkpoint" in value:
        checkpoint = _object(value["checkpoint"], {"file", "key", "strict"}, {"file"})
        relative_path(checkpoint["file"])
        if type(checkpoint.get("strict", True)) is not bool or ("key" in checkpoint and (
                not isinstance(checkpoint["key"], str) or not 1 <= len(checkpoint["key"]) <= 128)):
            raise ValueError("Invalid checkpoint key or strict setting.")
        result["checkpoint"] = {"strict": True, **checkpoint}
    return result


def validate_execution_job(value: Any) -> dict:
    """Graph capture has no profiler or training-step diagnostic job."""
    kind = value.get("kind") if isinstance(value, dict) else None
    if kind in ("profile", "training"):
        raise ValueError("Diagnostic profiling and training-step jobs are not supported.")
    raise ValueError("Unknown diagnostic job.")
