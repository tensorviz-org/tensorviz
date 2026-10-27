"""Conservative failure facts from actual dispatched operands, never error prose.

This is not an input generator or a general shape inference engine. Unknown
operators and unsupported tensor metadata retain a generic execution diagnostic.
"""
from __future__ import annotations

from .contracts import Diagnostic


def forward_diagnostic(error: BaseException, observed=None) -> dict:
    cause, seen = error, set()
    while cause is not None and id(cause) not in seen and len(seen) < 32:
        if observed is not None and cause is observed[0]:
            return observed[1]
        seen.add(id(cause))
        cause = cause.__cause__ or (None if cause.__suppress_context__ else cause.__context__)
    return Diagnostic(id="forward", phase="forward", code=("assertion_failed" if isinstance(error, AssertionError)
        else "execution_failed"), field=[], names=[], sourceBindingId=None, remedy="inspect_failure").model_dump(mode="json")


def operation_diagnostic(operation: str, args: tuple, kwargs: dict, error: BaseException) -> dict:
    import torch
    from torch._subclasses.fake_tensor import FakeTensor
    diagnostic = forward_diagnostic(error)

    def argument(index, name):
        return (args[index], ["args", index]) if len(args) > index else (kwargs.get(name), ["kwargs", name])

    def tensor(value):
        return (type(value) in (torch.Tensor, torch.nn.Parameter, FakeTensor) and value.layout == torch.strided
                and len(value.shape) <= 32 and all(type(size) is int and 0 <= size <= 2**53 - 1 for size in value.shape))

    def shape(field, actual, expected):
        diagnostic.update(code="invalid_shape", field=field, actualShape=list(actual), expectedShape=expected)

    def dtype(field, actual, expected=None):
        diagnostic.update(code="invalid_dtype", field=field, actualDtype=str(actual),
                          expectedDtype=str(expected) if expected is not None else None)

    try:
        if operation in ("aten.mm.default", "aten.addmm.default", "aten.bmm.default"):
            index = 1 if operation == "aten.addmm.default" else 0
            left, field = argument(index, "mat1" if index else "self")
            right, _ = argument(index + 1, "mat2")
            rank = 3 if operation == "aten.bmm.default" else 2
            if tensor(left) and tensor(right) and left.dim() == right.dim() == rank:
                if left.shape[-1] != right.shape[-2] or (rank == 3 and left.shape[0] != right.shape[0]):
                    expected = [None] * rank
                    expected[-1] = right.shape[-2]
                    if rank == 3:
                        expected[0] = right.shape[0]
                    shape(field, left.shape, expected)
                elif left.dtype != right.dtype:
                    dtype(field, left.dtype, right.dtype)
        elif operation == "aten.convolution.default":
            value, field = argument(0, "input")
            weight, _ = argument(1, "weight")
            transposed, _ = argument(6, "transposed")
            groups, _ = argument(8, "groups")
            if (tensor(value) and tensor(weight) and value.dim() == weight.dim() and value.dim() >= 3
                    and type(transposed) is bool and type(groups) is int and groups > 0):
                channels = weight.shape[0] if transposed else weight.shape[1] * groups
                if value.shape[1] != channels:
                    expected = [None] * value.dim()
                    expected[1] = channels
                    shape(field, value.shape, expected)
        elif operation == "aten.embedding.default":
            indices, field = argument(1, "indices")
            if tensor(indices) and indices.dtype not in (torch.int32, torch.int64):
                # Both integer dtypes are allowed; do not invent a single expected dtype.
                dtype(field, indices.dtype)
        return Diagnostic.model_validate(diagnostic).model_dump(mode="json")
    except Exception:
        # Observability must not mask or change the actual model exception.
        return forward_diagnostic(error)
