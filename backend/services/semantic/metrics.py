"""Bounded metadata reads; never retain tensors or invoke module iterators."""
from __future__ import annotations

import torch
from torch._subclasses.fake_tensor import FakeTensor

MAX_SAFE_INTEGER = 2**53 - 1
MAX_PARAMETER_ITEMS = 100000  # total module/parameter visits across this capture


def logical_bytes(tensor):
    """Caller already established an ordinary concrete strided tensor."""
    result = torch.Tensor.numel(tensor) * torch.Tensor.element_size(tensor)
    return result if type(result) is int and 0 <= result <= MAX_SAFE_INTEGER else None


class ParameterCounter:
    def __init__(self):
        self.remaining = MAX_PARAMETER_ITEMS

    def visit(self):
        if self.remaining == 0:
            return False
        self.remaining -= 1
        return True

    def measure(self, module):
        # Re-read at every entry: registration and shape can change between calls.
        # Local identities deduplicate sharing/cycles without storing model objects.
        modules, parameters = set(), set()
        pending, total = [iter((module,))], 0
        end = object()
        try:
            while pending:
                child = next(pending[-1], end)
                if child is end:
                    pending.pop()
                    continue
                if not self.visit():
                    return dict(parameterCountUnavailable="item_limit")
                if child is None or id(child) in modules:
                    continue
                modules.add(id(child))
                state = object.__getattribute__(child, "__dict__")
                own, children = state.get("_parameters"), state.get("_modules")
                if not isinstance(own, dict) or not isinstance(children, dict):
                    return dict(parameterCountUnavailable="unsupported_type")
                for tensor in dict.values(own):
                    if not self.visit():
                        return dict(parameterCountUnavailable="item_limit")
                    if tensor is None or id(tensor) in parameters:
                        continue
                    if type(tensor) not in (torch.Tensor, torch.nn.Parameter, FakeTensor):
                        return dict(parameterCountUnavailable="unsupported_type")
                    count = torch.Tensor.numel(tensor)
                    if type(count) is not int or not 0 <= count <= MAX_SAFE_INTEGER - total:
                        return dict(parameterCountUnavailable="unsupported_type")
                    total += count
                    parameters.add(id(tensor))
                pending.append(iter(dict.values(children)))
        except Exception:
            # Optional metrics must not replace the forward's outcome or stop
            # faithful argument/operation recording when introspection is unsupported.
            return dict(parameterCountUnavailable="unsupported_type")
        return dict(parameterCount=total)
