"""Single source of truth for all PyTorch op specifications.

Each op is described by an OpSpec that encodes:
  - accepted input ranks and arity
  - expected in/out dtypes
  - which params must be known for shape inference (defining_params)
  - param constraints (param_schema)
  - an analytic shape function (shape_fn)
  - a torch.nn.Module factory (build_fn)

Usage::

    from core.op_registry import spec_for, all_specs

    spec = spec_for("conv2d")          # alias-aware lookup
    out  = spec.shape_fn(params, in_shape, [in_shape])
    layer = spec.build_fn(params)

Dependency rule: this module NEVER imports from services/.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Stored parameter values
# ──────────────────────────────────────────────────────────────────────────────

def normalized_parameter(value: Any) -> Any:
    """A stored parameter as the number(s) it readably contains; else unchanged.

    The canvas stores one field in several shapes: the editor writes ``3``, the
    code importer writes ``[3, 3]``, and a field the user TYPES arrives as the
    string they typed — ``"3"`` or ``"3,3"``. torch constructors accept the first
    two and reject both strings ("type must be tuple of ints, but got str"), so
    every plane that turns a stored parameter into a real torch kwarg — both
    build_layers and the code renderer — has to read the typed form first. This
    is that reading, and it lives here so those planes cannot drift apart: the
    graph export verifies has to be the graph the generated file rebuilds.

    Int-ness is preserved on purpose. ``kernel_size=(3.0, 3.0)`` is rejected by
    torch just as flatly as the string was, so reading these as floats (the way
    ``ModelValidationService._numbers_in`` does, which only ever compares them
    against zero) would move the failure rather than fix it.

    Anything not readably numeric is returned untouched — ``padding="same"`` and
    ``mode="nearest"`` are strings that mean themselves.
    """
    if not isinstance(value, str):
        return value
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        return value
    numbers: List[Any] = []
    for part in parts:
        try:
            numbers.append(int(part))
            continue
        except ValueError:
            pass
        try:
            numbers.append(float(part))
        except ValueError:
            return value
    return numbers[0] if len(numbers) == 1 else tuple(numbers)


# ──────────────────────────────────────────────────────────────────────────────
# Shared math helpers (ported from CanonicalValidationService static methods)
# ──────────────────────────────────────────────────────────────────────────────

def _pair(value: Any) -> Tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(v, int) for v in value):
        return value[0], value[1]
    return 1, 1


def _ntuple(value: Any, n: int) -> Optional[Tuple[int, ...]]:
    if isinstance(value, int):
        return tuple([value] * n)
    if isinstance(value, (list, tuple)) and len(value) == n and all(isinstance(v, int) for v in value):
        return tuple(value)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Shape functions
# All have signature:
#   (params: dict, input_shape: list[int], all_inputs: list[list[int]]) -> list[int] | None
# Returns None when the op cannot produce a valid output shape with those inputs/params.
# ──────────────────────────────────────────────────────────────────────────────

def _shape_passthrough(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
    """Output shape == input shape (activations, dropout, norm in passthrough mode)."""
    return inp if inp else None


def _shape_linear(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
    if not inp:
        return None
    in_features = params.get("in_features")
    out_features = params.get("out_features")
    if not isinstance(in_features, int) or not isinstance(out_features, int):
        return None
    if inp[-1] != in_features:
        return None
    return inp[:-1] + [out_features]


def _shape_conv_nd(dim: int):
    def _fn(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
        expected = dim + 2
        if len(inp) != expected:
            return None
        c_in = inp[1]
        spatial_in = inp[2:]
        out_channels = params.get("out_channels")
        in_channels = params.get("in_channels")
        if not isinstance(out_channels, int) or out_channels <= 0:
            return None
        if isinstance(in_channels, int) and in_channels != c_in:
            return None

        padding_raw = params.get("padding", 0)
        # Handle string padding modes ("same" / "valid").
        if isinstance(padding_raw, str):
            pad_mode = padding_raw.lower()
            kernel = _ntuple(params.get("kernel_size", 1), dim)
            stride = _ntuple(params.get("stride", 1), dim)
            if kernel is None or stride is None:
                return None
            out_spatial: List[int] = []
            for i, (s, k, st) in enumerate(zip(spatial_in, kernel, stride)):
                if pad_mode == "same":
                    # torch rejects strided "same" padding.
                    if st != 1:
                        return None
                    # PyTorch "same" pads to keep output = input size.
                    o = math.ceil(s / st)
                elif pad_mode == "valid":
                    dilation = _ntuple(params.get("dilation", 1), dim)
                    if dilation is None:
                        return None
                    d = dilation[i]
                    o = math.floor((s - d * (k - 1) - 1) / st + 1)
                else:
                    return None
                if o <= 0:
                    return None
                out_spatial.append(o)
            return [inp[0], out_channels, *out_spatial]

        kernel = _ntuple(params.get("kernel_size", 1), dim)
        stride = _ntuple(params.get("stride", 1), dim)
        padding = _ntuple(padding_raw, dim)
        dilation = _ntuple(params.get("dilation", 1), dim)
        if None in (kernel, stride, padding, dilation):
            return None
        out_spatial = []
        for s, k, st, p, d in zip(spatial_in, kernel, stride, padding, dilation):
            o = math.floor((s + 2 * p - d * (k - 1) - 1) / st + 1)
            if o <= 0:
                return None
            out_spatial.append(o)
        return [inp[0], out_channels, *out_spatial]
    return _fn


def _shape_conv_transpose_nd(dim: int):
    def _fn(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
        expected = dim + 2
        if len(inp) != expected:
            return None
        c_in = inp[1]
        spatial_in = inp[2:]
        out_channels = params.get("out_channels")
        in_channels = params.get("in_channels")
        if not isinstance(out_channels, int) or out_channels <= 0:
            return None
        if isinstance(in_channels, int) and in_channels != c_in:
            return None
        kernel = _ntuple(params.get("kernel_size", 1), dim)
        stride = _ntuple(params.get("stride", 1), dim)
        padding = _ntuple(params.get("padding", 0), dim)
        output_padding = _ntuple(params.get("output_padding", 0), dim)
        dilation = _ntuple(params.get("dilation", 1), dim)
        if None in (kernel, stride, padding, output_padding, dilation):
            return None
        out_spatial: List[int] = []
        for s, k, st, p, op_pad, d in zip(spatial_in, kernel, stride, padding, output_padding, dilation):
            o = (s - 1) * st - 2 * p + d * (k - 1) + op_pad + 1
            if o <= 0:
                return None
            out_spatial.append(o)
        return [inp[0], out_channels, *out_spatial]
    return _fn


def _shape_padnd(ndim: int):
    """Padding ops: output == input with each spatial dim grown by its per-side padding.

    Input layout is [N, C, *spatial] (ndim spatial dims). `padding` is either an int
    (same amount on every side of every spatial dim) or a tuple of length 2*ndim following
    torch's ordering — last spatial dim first: 1d (left,right); 2d (left,right,top,bottom);
    3d (left,right,top,bottom,front,back). Pair i applies to spatial dim (ndim-1-i).
    """
    def _fn(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
        expected = ndim + 2
        if len(inp) != expected:
            return None
        padding = params.get("padding", 0)
        growth = [0] * ndim  # amount added to each spatial dim (inp[2 + i])
        if isinstance(padding, int):
            growth = [2 * padding] * ndim
        elif (isinstance(padding, (list, tuple)) and len(padding) == 2 * ndim
              and all(isinstance(v, int) for v in padding)):
            for i in range(ndim):
                growth[ndim - 1 - i] = padding[2 * i] + padding[2 * i + 1]
        else:
            return None
        out = list(inp)
        for i in range(ndim):
            out[2 + i] += growth[i]
            if out[2 + i] <= 0:
                return None
        return out
    return _fn


def _shape_pool_nd(dim: int):
    def _fn(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
        expected = dim + 2
        if len(inp) != expected:
            return None
        c_in = inp[1]
        spatial_in = inp[2:]
        kernel = _ntuple(params.get("kernel_size", 2), dim)
        # stride=None (or omitted) defaults to kernel_size, matching torch.
        stride_raw = params.get("stride")
        if stride_raw is None:
            stride_raw = params.get("kernel_size", 2)
        stride = _ntuple(stride_raw, dim)
        padding = _ntuple(params.get("padding", 0), dim)
        dilation = _ntuple(params.get("dilation", 1), dim)
        if None in (kernel, stride, padding, dilation):
            return None
        ceil_mode = bool(params.get("ceil_mode", False))
        out_spatial: List[int] = []
        for s, k, st, p, d in zip(spatial_in, kernel, stride, padding, dilation):
            numer = (s + 2 * p - d * (k - 1) - 1) / st + 1
            o = math.ceil(numer) if ceil_mode else math.floor(numer)
            # torch drops a ceil-mode window that would start entirely in the right padding.
            if ceil_mode and (o - 1) * st >= s + p:
                o -= 1
            if o <= 0:
                return None
            out_spatial.append(o)
        return [inp[0], c_in, *out_spatial]
    return _fn


def _shape_fractional_maxpool2d(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
    """FractionalMaxPool2d supports output_size/output_ratio, else kernel-style fallback."""
    if len(inp) != 4:
        return None
    n, c, h, w = inp

    out_size = params.get("output_size")
    if out_size is not None:
        if isinstance(out_size, int):
            out_h, out_w = out_size, out_size
        else:
            size_tuple = _ntuple(out_size, 2)
            if size_tuple is None:
                return None
            out_h, out_w = size_tuple
        if out_h <= 0 or out_w <= 0:
            return None
        return [n, c, out_h, out_w]

    out_ratio = params.get("output_ratio")
    if out_ratio is not None:
        if isinstance(out_ratio, (int, float)):
            rh = rw = float(out_ratio)
        elif isinstance(out_ratio, (list, tuple)) and len(out_ratio) == 2:
            try:
                rh = float(out_ratio[0])
                rw = float(out_ratio[1])
            except (TypeError, ValueError):
                return None
        else:
            return None
        if not (0.0 < rh <= 1.0 and 0.0 < rw <= 1.0):
            return None
        out_h = max(1, int(math.floor(h * rh)))
        out_w = max(1, int(math.floor(w * rw)))
        return [n, c, out_h, out_w]

    # Fallback to kernel/stride-style shape inference.
    return _shape_pool_nd(2)(params, inp, all_inputs)


def _shape_adaptive_pool_nd(dim: int):
    def _fn(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
        expected = dim + 2
        if len(inp) != expected:
            return None
        c_in = inp[1]
        out = params.get("output_size", [1] * dim if dim > 1 else 1)
        if dim == 1 and isinstance(out, int):
            out_spatial = [out]
        else:
            t = _ntuple(out, dim)
            if t is None:
                return None
            out_spatial = list(t)
        if any(o <= 0 for o in out_spatial):
            return None
        return [inp[0], c_in, *out_spatial]
    return _fn


def _shape_flatten(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
    if not inp:
        return None
    start_dim = params.get("start_dim", 1)
    end_dim = params.get("end_dim", -1)
    rank = len(inp)
    if end_dim < 0:
        end_dim = rank + end_dim
    if start_dim < 0:
        start_dim = rank + start_dim
    if start_dim < 0 or end_dim >= rank or start_dim > end_dim:
        return None
    flattened = 1
    for d in inp[start_dim : end_dim + 1]:
        flattened *= d
    return inp[:start_dim] + [flattened] + inp[end_dim + 1:]


def _shape_unflatten(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
    if not inp:
        return None
    dim = params.get("dim", 1)
    unflattened_size = params.get("unflattened_size")
    if not isinstance(dim, int):
        return None
    rank = len(inp)
    if dim < 0:
        dim += rank
    if dim < 0 or dim >= rank:
        return None
    if isinstance(unflattened_size, tuple):
        unflattened_size = list(unflattened_size)
    if not isinstance(unflattened_size, list) or not unflattened_size:
        return None
    if not all(isinstance(v, int) and v > 0 for v in unflattened_size):
        return None
    product = 1
    for v in unflattened_size:
        product *= v
    if product != inp[dim]:
        return None
    return inp[:dim] + unflattened_size + inp[dim + 1:]


def _shape_batchnorm_nd(channel_pos: int = 1, min_rank: int = 2):
    def _fn(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
        if len(inp) < min_rank:
            return None
        num_features = params.get("num_features")
        if not isinstance(num_features, int) or num_features != inp[channel_pos]:
            return None
        return inp
    return _fn


def _shape_groupnorm(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
    if len(inp) < 2:
        return None
    num_groups = params.get("num_groups")
    num_channels = params.get("num_channels")
    if not isinstance(num_groups, int) or num_groups <= 0:
        return None
    if not isinstance(num_channels, int) or num_channels != inp[1]:
        return None
    if num_channels % num_groups != 0:
        return None
    return inp


def _shape_layernorm(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
    normalized_shape = params.get("normalized_shape")
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]
    if isinstance(normalized_shape, tuple):
        normalized_shape = list(normalized_shape)
    if not isinstance(normalized_shape, list) or not normalized_shape:
        return None
    if len(normalized_shape) > len(inp):
        return None
    if inp[-len(normalized_shape):] != normalized_shape:
        return None
    return inp


def _shape_embedding(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
    num_embeddings = params.get("num_embeddings")
    embedding_dim = params.get("embedding_dim")
    if not isinstance(num_embeddings, int) or num_embeddings <= 0:
        return None
    if not isinstance(embedding_dim, int) or embedding_dim <= 0:
        return None
    return [*inp, embedding_dim]


def _shape_rnn(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
    input_size = params.get("input_size")
    hidden_size = params.get("hidden_size")
    batch_first = params.get("batch_first", False)
    if not isinstance(input_size, int) or input_size <= 0:
        return None
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        return None
    # A bidirectional RNN concatenates its two directions, so the feature width it
    # emits is 2*hidden_size. Without this the analytic plane sized every
    # `*_bi_block`'s projection for half the tensor torch actually produces —
    # thirteen blocks that validated clean and then died on the first matmul. It was
    # pinned as a known analytic/torch divergence rather than fixed;
    # `core/shape_inference.py` has always doubled it correctly, so the two shape
    # rules in this repo disagreed with each other.
    if params.get("bidirectional"):
        hidden_size *= 2
    if len(inp) == 2:
        first, features = inp
        if features != input_size:
            return None
        if batch_first:
            return [first, 1, hidden_size]
        return [first, hidden_size]
    if len(inp) != 3:
        return None
    if batch_first:
        batch, seq_len, features = inp
        if features != input_size:
            return None
        return [batch, seq_len, hidden_size]
    seq_len, batch, features = inp
    if features != input_size:
        return None
    return [seq_len, batch, hidden_size]


def _shape_multihead_attention(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
    if len(inp) not in {2, 3}:
        return None
    embed_dim = params.get("embed_dim")
    if not isinstance(embed_dim, int) or embed_dim <= 0:
        return None
    if inp[-1] != embed_dim:
        return None
    return inp


def _shape_transformer_layer(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
    if len(inp) != 3:
        return None
    d_model = params.get("d_model")
    if not isinstance(d_model, int) or d_model <= 0:
        return None
    if inp[-1] != d_model:
        return None
    return inp


def _shape_add(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
    if not all_inputs:
        return None
    head = all_inputs[0]
    for shape in all_inputs[1:]:
        if shape != head:
            return None
    return head


def _shape_cat(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
    if not all_inputs:
        return None
    rank = len(all_inputs[0])
    if any(len(s) != rank for s in all_inputs):
        return None
    dim = params.get("dim", 1)
    if not isinstance(dim, int):
        return None
    if dim < 0:
        dim = rank + dim
    if dim < 0 or dim >= rank:
        return None
    output = list(all_inputs[0])
    total = output[dim]
    for shape in all_inputs[1:]:
        for i in range(rank):
            if i != dim and shape[i] != output[i]:
                return None
        total += shape[dim]
    output[dim] = total
    return output


def _shape_upsample(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
    if len(inp) < 3:
        return None
    c_in = inp[1]
    spatial_in = inp[2:]
    dim = len(spatial_in)
    size = params.get("size")
    if size is not None:
        if isinstance(size, int):
            out_spatial = [size] * dim
        else:
            t = _ntuple(size, dim)
            if t is None:
                return None
            out_spatial = list(t)
        if any(o <= 0 for o in out_spatial):
            return None
        return [inp[0], c_in, *out_spatial]
    scale = params.get("scale_factor", 2)
    if isinstance(scale, (int, float)):
        scales = [float(scale)] * dim
    elif isinstance(scale, (list, tuple)) and len(scale) == dim:
        scales = [float(s) for s in scale]
    else:
        return None
    out_spatial = [int(math.floor(s * sc)) for s, sc in zip(spatial_in, scales)]
    if any(o <= 0 for o in out_spatial):
        return None
    return [inp[0], c_in, *out_spatial]


def _shape_pixel_shuffle(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
    if len(inp) != 4:
        return None
    n, c, h, w = inp
    r = params.get("upscale_factor")
    if not isinstance(r, int) or r <= 0:
        return None
    if c % (r * r) != 0:
        return None
    return [n, c // (r * r), h * r, w * r]


def _shape_pixel_unshuffle(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
    if len(inp) != 4:
        return None
    n, c, h, w = inp
    r = params.get("downscale_factor")
    if not isinstance(r, int) or r <= 0:
        return None
    if h % r != 0 or w % r != 0:
        return None
    return [n, c * r * r, h // r, w // r]


def _shape_channel_shuffle(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
    if len(inp) < 3:
        return None
    groups = params.get("groups", 1)
    if not isinstance(groups, int) or groups <= 0:
        return None
    if inp[1] % groups != 0:
        return None
    return inp


def _shape_unfold(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
    if len(inp) != 4:
        return None
    n, c, h, w = inp
    kernel = _ntuple(params.get("kernel_size", 1), 2)
    dilation = _ntuple(params.get("dilation", 1), 2)
    padding = _ntuple(params.get("padding", 0), 2)
    stride = _ntuple(params.get("stride", 1), 2)
    if None in (kernel, dilation, padding, stride):
        return None
    kh, kw = kernel
    dh, dw = dilation
    ph, pw = padding
    sh, sw = stride
    out_h = math.floor((h + 2 * ph - dh * (kh - 1) - 1) / sh + 1)
    out_w = math.floor((w + 2 * pw - dw * (kw - 1) - 1) / sw + 1)
    if out_h <= 0 or out_w <= 0:
        return None
    return [n, c * kh * kw, out_h * out_w]


def _shape_fold(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
    if len(inp) != 3:
        return None
    n, c_kk, l = inp
    output_size = params.get("output_size")
    if isinstance(output_size, int):
        out_h, out_w = output_size, output_size
    else:
        out_tuple = _ntuple(output_size, 2)
        if out_tuple is None:
            return None
        out_h, out_w = out_tuple
    if out_h <= 0 or out_w <= 0:
        return None

    kernel = _ntuple(params.get("kernel_size", 1), 2)
    dilation = _ntuple(params.get("dilation", 1), 2)
    padding = _ntuple(params.get("padding", 0), 2)
    stride = _ntuple(params.get("stride", 1), 2)
    if None in (kernel, dilation, padding, stride):
        return None
    kh, kw = kernel
    dh, dw = dilation
    ph, pw = padding
    sh, sw = stride

    if kh <= 0 or kw <= 0:
        return None
    patch_channels = kh * kw
    if c_kk % patch_channels != 0:
        return None
    out_channels = c_kk // patch_channels

    expected_h = math.floor((out_h + 2 * ph - dh * (kh - 1) - 1) / sh + 1)
    expected_w = math.floor((out_w + 2 * pw - dw * (kw - 1) - 1) / sw + 1)
    if expected_h <= 0 or expected_w <= 0:
        return None
    expected_l = expected_h * expected_w
    if l != expected_l:
        return None
    return [n, out_channels, out_h, out_w]


def _shape_glu(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
    """GLU halves the last dim."""
    if not inp:
        return None
    if inp[-1] % 2 != 0:
        return None
    return inp[:-1] + [inp[-1] // 2]


def _shape_loss(name: str):
    """Reduced losses are zero-dimensional; unreduced losses keep their sample axes."""
    def shape(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
        if not all_inputs:
            return None
        reduction = "none" if params.get("reduce") is False else params.get("reduction", "mean")
        if reduction in {"mean", "sum"} or reduction == "batchmean" and name == "KLDivLoss":
            return []
        if reduction != "none":
            return None
        if name in {"CrossEntropyLoss", "NLLLoss"}:
            return inp[:1] + inp[2:] if len(inp) > 1 else []
        if name == "CTCLoss":
            return [inp[1]] if len(inp) == 3 else []
        if name in {"TripletMarginLoss", "CosineEmbeddingLoss"}:
            return inp[:-1]
        if name in {"MSELoss", "L1Loss", "HuberLoss", "SmoothL1Loss", "KLDivLoss"} and len(all_inputs) > 1:
            return _broadcast_shapes(inp, all_inputs[1])
        return list(inp)
    return shape


def _broadcast_shapes(s1: List[int], s2: List[int]) -> Optional[List[int]]:
    """NumPy/torch broadcasting of two shapes (right-aligned). None if incompatible."""
    out = []
    for a, b in zip(reversed(s1), reversed(s2)):
        if a == b or a == 1 or b == 1:
            out.append(max(a, b))
        else:
            return None
    longer = s1 if len(s1) >= len(s2) else s2
    lead = longer[: abs(len(s1) - len(s2))]
    return list(lead) + list(reversed(out))


def _shape_matmul(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
    """torch.matmul semantics incl. 1-D promotion and batch broadcasting."""
    if len(all_inputs) < 2:
        return None
    a, b = list(all_inputs[0]), list(all_inputs[1])
    if not a or not b:
        return None
    if len(a) == 1 and len(b) == 1:  # (k) · (k) -> scalar
        return [] if a[0] == b[0] else None
    a_1d, b_1d = len(a) == 1, len(b) == 1
    aa = ([1] + a) if a_1d else a
    bb = (b + [1]) if b_1d else b
    if aa[-1] != bb[-2]:
        return None
    m, n = aa[-2], bb[-1]
    batch = _broadcast_shapes(aa[:-2], bb[:-2])
    if batch is None:
        return None
    out = batch + [m, n]
    if b_1d:
        out.pop(-1)
    if a_1d:
        out.pop(-1 if b_1d else -2)
    return out


def _shape_bmm(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
    """torch.bmm: [b,n,m] @ [b,m,p] -> [b,n,p]."""
    if len(all_inputs) < 2:
        return None
    a, b = all_inputs[0], all_inputs[1]
    if len(a) != 3 or len(b) != 3:
        return None
    if a[0] != b[0] or a[2] != b[1]:
        return None
    return [a[0], a[1], b[2]]


def _shape_reduce(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
    """sum/mean/max/min/argmax/argmin: drop `dim` (or keep as 1 with keepdim)."""
    if not inp:
        return None
    if params.get("dim", None) is None:  # full reduction -> scalar
        return []
    dim = params["dim"]
    if not isinstance(dim, int):
        return None
    rank = len(inp)
    if dim < 0:
        dim += rank
    if dim < 0 or dim >= rank:
        return None
    out = list(inp)
    if params.get("keepdim", False):
        out[dim] = 1
    else:
        out.pop(dim)
    return out


def _parse_int_list(value: Any) -> Optional[List[int]]:
    """A dim/shape tuple as the canvas stores it: a real list, or the comma string
    the Input node's shape uses. Returns None when it is neither."""
    if isinstance(value, (list, tuple)):
        if all(isinstance(v, int) for v in value):
            return [int(v) for v in value]
        return None
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace("(", "").replace(")", "").split(",")]
        parts = [p for p in parts if p != ""]
        if not parts:
            return None
        try:
            return [int(p) for p in parts]
        except ValueError:
            return None
    if isinstance(value, int):
        return [value]
    return None


def _shape_reshape(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
    """torch.reshape. Declines rather than guessing: without a `shape` param there
    is nothing to derive, and the passthrough it used to return was simply wrong
    (a reshape that reports its input shape is a lie the whole graph then trusts).
    One -1 is inferred from the element count, as torch does."""
    if not inp:
        return None
    target = _parse_int_list(params.get("shape", params.get("size", None)))
    if not target:
        return None
    if any(v == 0 for v in target):
        return None
    wild = [i for i, v in enumerate(target) if v < 0]
    if len(wild) > 1:
        return None
    total = 1
    for d in inp:
        total *= d
    if not wild:
        fixed = 1
        for v in target:
            fixed *= v
        return list(target) if fixed == total else None
    fixed = 1
    for v in target:
        if v > 0:
            fixed *= v
    if fixed == 0 or total % fixed != 0:
        return None
    out = list(target)
    out[wild[0]] = total // fixed
    return out


def _shape_declines_multi_output(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
    """torch.split / torch.chunk. These return a TUPLE, and the graph cannot carry
    that: a node has exactly one output_shape and edges have no output index (the
    document format's source_port is ignored by this engine and by codegen). So the
    honest answer is "I cannot know" — declining sets shape_error rather than
    reporting the input shape unchanged, which is what it used to do."""
    return None


def _shape_placeholder_recorded(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
    """`python` / `custom_code`: code the registry deliberately does not model.

    Nothing about an opaque statement or a user-defined nn.Module can be DERIVED
    from its parameters, and a wrong shape is worse than a declined one — it is
    trusted by every node downstream. So this computes nothing. It only reads
    back the shape the node already carries: `_out_shape` is what the trace
    measured, or what the class declared in its own source (task.md M6), and the
    node has it because something outside the registry knew. Without one, the
    honest answer is still None, and every caller then declines rather than
    passing the input through."""
    recorded = params.get("_out_shape")
    if not isinstance(recorded, str):
        return None
    try:
        dims = [int(part) for part in recorded.split("x") if part.strip()]
    except ValueError:
        return None
    return dims or None


def _shape_stack(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
    """torch.stack inserts a NEW axis of size len(inputs) at `dim` — it does not
    preserve the input shape. Every input must already agree."""
    if not all_inputs or not inp:
        return None
    first = list(all_inputs[0])
    for other in all_inputs[1:]:
        if list(other) != first:
            return None
    dim = params.get("dim", 0)
    if dim is None:
        dim = 0
    if not isinstance(dim, int):
        return None
    rank = len(first) + 1  # stack can insert at the new trailing position
    if dim < 0:
        dim += rank
    if dim < 0 or dim >= rank:
        return None
    out = list(first)
    out.insert(dim, len(all_inputs))
    return out


def _shape_where(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
    """torch.where(cond, a, b) broadcasts all of its operands together."""
    if not all_inputs:
        return None
    acc = list(all_inputs[0])
    for other in all_inputs[1:]:
        acc = _broadcast_shapes(acc, list(other))
        if acc is None:
            return None
    return acc


def _shape_narrow(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
    """torch.narrow(input, dim, start, length) — the slicing op. Declines on an
    out-of-range dim or a window that runs off the end, so an impossible slice
    reads as an error the user can act on rather than crashing at verify time."""
    if not inp:
        return None
    dim, start, length = params.get("dim"), params.get("start"), params.get("length")
    if not all(isinstance(v, int) for v in (dim, start, length)):
        return None
    rank = len(inp)
    if dim < 0:
        dim += rank
    if dim < 0 or dim >= rank:
        return None
    if start < 0 or length <= 0 or start + length > inp[dim]:
        return None
    out = list(inp)
    out[dim] = length
    return out


def _shape_squeeze(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
    """torch.squeeze: drop size-1 dims (all, or the given `dim`)."""
    if not inp:
        return None
    dim = params.get("dim", None)
    if dim is None:
        return [d for d in inp if d != 1]
    if not isinstance(dim, int):
        return None
    rank = len(inp)
    if dim < 0:
        dim += rank
    if dim < 0 or dim >= rank:
        return None
    if inp[dim] != 1:
        return list(inp)
    out = list(inp)
    out.pop(dim)
    return out


def _shape_unsqueeze(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
    """torch.unsqueeze: insert a size-1 dim at `dim`."""
    dim = params.get("dim", None)
    if inp is None or not isinstance(dim, int):
        return None
    rank = len(inp)
    if dim < 0:
        dim += rank + 1
    if dim < 0 or dim > rank:
        return None
    out = list(inp)
    out.insert(dim, 1)
    return out


def _shape_transpose(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
    """torch.transpose: swap the two dims in params['dims']."""
    # The Properties field is free text, so a hand-typed pair arrives as the comma
    # string the Input node's shape uses — same as `shape` on reshape above.
    dims = _parse_int_list(params.get("dims", None))
    if not inp or dims is None or len(dims) != 2:
        return None
    rank = len(inp)
    d0, d1 = dims
    if d0 < 0:
        d0 += rank
    if d1 < 0:
        d1 += rank
    if not (0 <= d0 < rank and 0 <= d1 < rank):
        return None
    out = list(inp)
    out[d0], out[d1] = out[d1], out[d0]
    return out


def _shape_permute(params: dict, inp: List[int], all_inputs: List[List[int]]) -> Optional[List[int]]:
    """torch.permute: reorder dims per params['dims'] (a permutation of range(rank))."""
    dims = _parse_int_list(params.get("dims", None))
    if not inp or dims is None:
        return None
    rank = len(inp)
    norm = []
    for d in dims:
        if d < 0:
            d += rank
        if d < 0 or d >= rank:
            return None
        norm.append(d)
    if sorted(norm) != list(range(rank)):
        return None
    return [inp[d] for d in norm]


# ──────────────────────────────────────────────────────────────────────────────
# Build functions
# All have signature: (params: dict) -> nn.Module
# ──────────────────────────────────────────────────────────────────────────────

def _build_torch_nn(class_name: str):
    """Return a build_fn that constructs `torch.nn.<class_name>(**params)`."""
    def _build(params: dict) -> "nn.Module":
        # Registry lookup and analytic/source-only projection must work without
        # a model framework. Import it only when an actual module is requested.
        import torch.nn as nn
        # Strip non-constructor keys
        clean = {k: v for k, v in params.items() if k != "name" and not k.startswith("_")}
        cls = getattr(nn, class_name)
        return cls(**clean)

    _build.__name__ = f"build_{class_name}"
    return _build


def _build_linear(params: dict) -> "Any":
    import torch.nn as nn
    in_f = params.get("in_features")
    out_f = params.get("out_features")
    if in_f is None or out_f is None:
        raise ValueError("Linear requires in_features and out_features")
    bias = params.get("bias", True)
    return nn.Linear(in_features=in_f, out_features=out_f, bias=bias)


def _build_embedding(params: dict) -> "Any":
    import torch.nn as nn
    n = params.get("num_embeddings")
    d = params.get("embedding_dim")
    if n is None or d is None:
        raise ValueError("Embedding requires num_embeddings and embedding_dim")
    return nn.Embedding(num_embeddings=n, embedding_dim=d)


def _build_layernorm(params: dict) -> "Any":
    import torch.nn as nn
    ns = params.get("normalized_shape")
    if ns is None:
        raise ValueError("LayerNorm requires normalized_shape")
    eps = params.get("eps", 1e-5)
    return nn.LayerNorm(normalized_shape=ns, eps=eps)


def _build_multihead_attention(params: dict) -> "Any":
    import torch.nn as nn
    embed_dim = params.get("embed_dim")
    num_heads = params.get("num_heads", params.get("nhead"))
    if embed_dim is None or num_heads is None:
        raise ValueError("MultiheadAttention requires embed_dim and num_heads")
    dropout = params.get("dropout", 0.0)
    return nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)


def _build_transformer_encoder_layer(params: dict) -> "Any":
    import torch.nn as nn
    d_model = params.get("d_model")
    nhead = params.get("nhead")
    if d_model is None or nhead is None:
        raise ValueError("TransformerEncoderLayer requires d_model and nhead")
    return nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=params.get("dim_feedforward", 2048),
        dropout=params.get("dropout", 0.1),
    )


def _build_transformer_decoder_layer(params: dict) -> "Any":
    import torch.nn as nn
    d_model = params.get("d_model")
    nhead = params.get("nhead")
    if d_model is None or nhead is None:
        raise ValueError("TransformerDecoderLayer requires d_model and nhead")
    return nn.TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=params.get("dim_feedforward", 2048),
        dropout=params.get("dropout", 0.1),
    )


def _build_groupnorm(params: dict) -> "Any":
    import torch.nn as nn
    num_groups = params.get("num_groups")
    num_channels = params.get("num_channels")
    if num_groups is None or num_channels is None:
        raise ValueError("GroupNorm requires num_groups and num_channels")
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=params.get("eps", 1e-5))


def _build_rnn_family(class_name: str):
    def _build(params: dict) -> "Any":
        import torch.nn as nn
        input_size = params.get("input_size")
        hidden_size = params.get("hidden_size")
        if input_size is None or hidden_size is None:
            raise ValueError(f"{class_name} requires input_size and hidden_size")
        cls = getattr(nn, class_name)
        return cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=params.get("num_layers", 1),
            bias=params.get("bias", True),
            batch_first=params.get("batch_first", False),
            dropout=params.get("dropout", 0.0),
            bidirectional=params.get("bidirectional", False),
        )
    _build.__name__ = f"build_{class_name}"
    return _build


def _no_build(params: dict) -> None:  # type: ignore[return]
    """Placeholder build function for custom/placeholder ops."""
    raise NotImplementedError("This op has no build_fn (placeholder op).")


# ──────────────────────────────────────────────────────────────────────────────
# OpSpec dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OpSpec:
    """Complete specification for one PyTorch operation."""

    canonical: str
    """Canonical class name, e.g. 'nn.Conv2d'."""

    aliases: Tuple[str, ...]
    """All accepted op strings (case-insensitive lookup on top of these)."""

    input_rank: Tuple[int, ...]
    """Accepted input ranks. Empty tuple means any rank."""

    arity: str
    """'one' | 'merge' | 'qkv' | 'loss' | 'any'."""

    in_dtype: str
    """'float' | 'int' | 'any'."""

    out_dtype: str
    """'float' | 'int' | 'same'."""

    defining_params: Tuple[str, ...]
    """Params that must be known for shape inference.  Used by lazy-fill logic."""

    param_schema: Dict[str, Any] = field(default_factory=dict, compare=False, hash=False)
    """Param constraints: range tuples, divisibility rules. e.g. {'p': (0.0, 1.0)}."""

    shape_fn: Callable = field(compare=False, hash=False, default=_shape_passthrough)
    """shape_fn(params, primary_input_shape, all_input_shapes) -> output_shape | None."""

    build_fn: Optional[Callable] = field(compare=False, hash=False, default=None)
    """build_fn(params) -> nn.Module.  None for custom/placeholder ops."""


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

_REGISTRY: Dict[str, OpSpec] = {}

# Placeholder ops are resolvable but carry NO shape/build contract, so they are
# kept in their own table: `spec_for` finds them, `all_specs()` — whose only
# consumer is the tier-0 contract suite asserting analytic == real torch forward
# for every op — does not. See the `python` / `custom_code` registrations at the
# bottom of this file.
_PLACEHOLDER_REGISTRY: Dict[str, OpSpec] = {}


def _register(*specs: OpSpec) -> None:
    for spec in specs:
        _REGISTRY[spec.canonical.lower()] = spec
        for alias in spec.aliases:
            _REGISTRY[alias.lower()] = spec


def _register_placeholder(*specs: OpSpec) -> None:
    for spec in specs:
        _PLACEHOLDER_REGISTRY[spec.canonical.lower()] = spec
        for alias in spec.aliases:
            _PLACEHOLDER_REGISTRY[alias.lower()] = spec


def _lookup_key(op: str) -> str:
    key = op.strip().lower()
    # Strip common prefixes so 'nn.Conv2d', 'torch.nn.Conv2d', 'Conv2d' all resolve.
    for prefix in ("torch.nn.", "nn.", "torch."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def spec_for(op: str) -> Optional[OpSpec]:
    """Alias-aware, case-insensitive lookup.  Returns None for unknown ops."""
    if not op:
        return None
    key = _lookup_key(op)
    spec = _REGISTRY.get(key)
    return spec if spec is not None else _PLACEHOLDER_REGISTRY.get(key)


def is_placeholder_op(op: str) -> bool:
    """True for `python` / `custom_code` — ids that name code we do not model."""
    return bool(op) and _lookup_key(op) in _PLACEHOLDER_REGISTRY


def all_specs() -> List[OpSpec]:
    """De-duplicated list of every registered OpSpec that carries a contract.

    Placeholder ops (`python`, `custom_code`) are excluded: they have no torch
    counterpart, so the tier-0 "analytic shape == real forward" contract cannot
    be stated for them at all. Use `spec_for` when you want them.
    """
    return list({id(s): s for s in _REGISTRY.values()}.values())


# ──────────────────────────────────────────────────────────────────────────────
# Op definitions  (alphabetical within family)
# ──────────────────────────────────────────────────────────────────────────────

# ── Activations (shape-preserving) ──────────────────────────────────────────
_ACTIVATION_OPS = [
    ("nn.CELU",        ("celu", "CELU"),                    _build_torch_nn("CELU")),
    ("nn.ELU",         ("elu", "ELU"),                      _build_torch_nn("ELU")),
    ("nn.GELU",        ("gelu", "GELU"),                    _build_torch_nn("GELU")),
    ("nn.GLU",         ("glu", "GLU"),                      _build_torch_nn("GLU")),
    ("nn.Hardshrink",  ("hardshrink", "Hardshrink"),        _build_torch_nn("Hardshrink")),
    ("nn.Hardsigmoid", ("hardsigmoid", "Hardsigmoid"),      _build_torch_nn("Hardsigmoid")),
    ("nn.Hardswish",   ("hardswish", "Hardswish"),          _build_torch_nn("Hardswish")),
    ("nn.Hardtanh",    ("hardtanh", "Hardtanh"),            _build_torch_nn("Hardtanh")),
    ("nn.Identity",    ("identity", "Identity"),            _build_torch_nn("Identity")),
    ("nn.LeakyReLU",   ("leakyrelu", "LeakyReLU"),          _build_torch_nn("LeakyReLU")),
    ("nn.LogSoftmax",  ("logsoftmax", "LogSoftmax"),        _build_torch_nn("LogSoftmax")),
    ("nn.Mish",        ("mish", "Mish"),                    _build_torch_nn("Mish")),
    ("nn.PReLU",       ("prelu", "PReLU"),                  _build_torch_nn("PReLU")),
    ("nn.ReLU",        ("relu", "ReLU"),                    _build_torch_nn("ReLU")),
    ("nn.ReLU6",       ("relu6", "ReLU6"),                  _build_torch_nn("ReLU6")),
    ("nn.RReLU",       ("rrelu", "RReLU"),                  _build_torch_nn("RReLU")),
    ("nn.SELU",        ("selu", "SELU"),                    _build_torch_nn("SELU")),
    ("nn.SiLU",        ("silu", "SiLU"),                    _build_torch_nn("SiLU")),
    ("nn.Sigmoid",     ("sigmoid", "Sigmoid"),              _build_torch_nn("Sigmoid")),
    ("nn.Softmax",     ("softmax", "Softmax"),              _build_torch_nn("Softmax")),
    ("nn.Softplus",    ("softplus", "Softplus"),            _build_torch_nn("Softplus")),
    ("nn.Softshrink",  ("softshrink", "Softshrink"),        _build_torch_nn("Softshrink")),
    ("nn.Softsign",    ("softsign", "Softsign"),            _build_torch_nn("Softsign")),
    ("nn.Tanh",        ("tanh", "Tanh"),                    _build_torch_nn("Tanh")),
    ("nn.Tanhshrink",  ("tanhshrink", "Tanhshrink"),        _build_torch_nn("Tanhshrink")),
    ("nn.Threshold",   ("threshold", "Threshold"),          _build_torch_nn("Threshold")),
]

for _canonical, _aliases, _build in _ACTIVATION_OPS:
    # GLU is special: it halves the last dim
    if _canonical == "nn.GLU":
        _sfn = _shape_glu
    else:
        _sfn = _shape_passthrough
    _register(OpSpec(
        canonical=_canonical,
        aliases=_aliases,
        input_rank=(),
        arity="one",
        in_dtype="float",
        out_dtype="same",
        defining_params=(),
        shape_fn=_sfn,
        build_fn=_build,
    ))

# ── Dropout (shape-preserving) ───────────────────────────────────────────────
for _name in ("Dropout", "Dropout1d", "Dropout2d", "Dropout3d", "AlphaDropout"):
    _register(OpSpec(
        canonical=f"nn.{_name}",
        aliases=(_name.lower(), _name),
        input_rank=(),
        arity="one",
        in_dtype="any",
        out_dtype="same",
        defining_params=(),
        param_schema={"p": (0.0, 1.0)},
        shape_fn=_shape_passthrough,
        build_fn=_build_torch_nn(_name),
    ))

# ── Linear ───────────────────────────────────────────────────────────────────
_register(OpSpec(
    canonical="nn.Linear",
    aliases=("linear", "Linear"),
    input_rank=(),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=("in_features", "out_features"),
    param_schema={},
    shape_fn=_shape_linear,
    build_fn=_build_linear,
))

_register(OpSpec(
    canonical="nn.Bilinear",
    aliases=("bilinear", "Bilinear"),
    input_rank=(),
    arity="merge",
    in_dtype="float",
    out_dtype="float",
    defining_params=("in1_features", "in2_features", "out_features"),
    param_schema={},
    shape_fn=lambda p, inp, _: (list(inp[:-1]) + [p["out_features"]]
                                if inp and isinstance(p.get("out_features"), int) else None),
    build_fn=_build_torch_nn("Bilinear"),
))

_register(OpSpec(
    canonical="nn.LazyLinear",
    aliases=("lazylinear", "LazyLinear"),
    input_rank=(),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=("out_features",),
    param_schema={},
    shape_fn=lambda p, inp, _: inp[:-1] + [p["out_features"]] if inp and isinstance(p.get("out_features"), int) else None,
    build_fn=_build_torch_nn("LazyLinear"),
))

# ── Convolutions ─────────────────────────────────────────────────────────────
_register(OpSpec(
    canonical="nn.Conv1d",
    aliases=("conv1d", "Conv1d"),
    input_rank=(3,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=("in_channels", "out_channels"),
    param_schema={"kernel_size": (1, None), "stride": (1, None), "groups|divides_both": ("in_channels", "out_channels")},
    shape_fn=_shape_conv_nd(1),
    build_fn=_build_torch_nn("Conv1d"),
))

_register(OpSpec(
    canonical="nn.Conv2d",
    aliases=("conv2d", "Conv2d"),
    input_rank=(4,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=("in_channels", "out_channels"),
    param_schema={"kernel_size": (1, None), "stride": (1, None), "groups|divides_both": ("in_channels", "out_channels")},
    shape_fn=_shape_conv_nd(2),
    build_fn=_build_torch_nn("Conv2d"),
))

_register(OpSpec(
    canonical="nn.Conv3d",
    aliases=("conv3d", "Conv3d"),
    input_rank=(5,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=("in_channels", "out_channels"),
    param_schema={"kernel_size": (1, None), "stride": (1, None), "groups|divides_both": ("in_channels", "out_channels")},
    shape_fn=_shape_conv_nd(3),
    build_fn=_build_torch_nn("Conv3d"),
))

_register(OpSpec(
    canonical="nn.ConvTranspose1d",
    aliases=("convtranspose1d", "ConvTranspose1d"),
    input_rank=(3,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=("in_channels", "out_channels"),
    param_schema={},
    shape_fn=_shape_conv_transpose_nd(1),
    build_fn=_build_torch_nn("ConvTranspose1d"),
))

_register(OpSpec(
    canonical="nn.ConvTranspose2d",
    aliases=("convtranspose2d", "ConvTranspose2d"),
    input_rank=(4,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=("in_channels", "out_channels"),
    param_schema={},
    shape_fn=_shape_conv_transpose_nd(2),
    build_fn=_build_torch_nn("ConvTranspose2d"),
))

_register(OpSpec(
    canonical="nn.ConvTranspose3d",
    aliases=("convtranspose3d", "ConvTranspose3d"),
    input_rank=(5,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=("in_channels", "out_channels"),
    param_schema={},
    shape_fn=_shape_conv_transpose_nd(3),
    build_fn=_build_torch_nn("ConvTranspose3d"),
))

# ── Pooling ───────────────────────────────────────────────────────────────────
_register(OpSpec(
    canonical="nn.MaxPool1d",
    aliases=("maxpool1d", "MaxPool1d"),
    input_rank=(3,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=(),
    param_schema={"kernel_size": (1, None)},
    shape_fn=_shape_pool_nd(1),
    build_fn=_build_torch_nn("MaxPool1d"),
))

_register(OpSpec(
    canonical="nn.MaxPool2d",
    aliases=("maxpool2d", "MaxPool2d"),
    input_rank=(4,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=(),
    param_schema={"kernel_size": (1, None)},
    shape_fn=_shape_pool_nd(2),
    build_fn=_build_torch_nn("MaxPool2d"),
))

_register(OpSpec(
    canonical="nn.MaxPool3d",
    aliases=("maxpool3d", "MaxPool3d"),
    input_rank=(5,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=(),
    param_schema={"kernel_size": (1, None)},
    shape_fn=_shape_pool_nd(3),
    build_fn=_build_torch_nn("MaxPool3d"),
))

_register(OpSpec(
    canonical="nn.AvgPool1d",
    aliases=("avgpool1d", "AvgPool1d"),
    input_rank=(3,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=(),
    param_schema={"kernel_size": (1, None)},
    shape_fn=_shape_pool_nd(1),
    build_fn=_build_torch_nn("AvgPool1d"),
))

_register(OpSpec(
    canonical="nn.AvgPool2d",
    aliases=("avgpool2d", "AvgPool2d"),
    input_rank=(4,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=(),
    param_schema={"kernel_size": (1, None)},
    shape_fn=_shape_pool_nd(2),
    build_fn=_build_torch_nn("AvgPool2d"),
))

_register(OpSpec(
    canonical="nn.AvgPool3d",
    aliases=("avgpool3d", "AvgPool3d"),
    input_rank=(5,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=(),
    param_schema={"kernel_size": (1, None)},
    shape_fn=_shape_pool_nd(3),
    build_fn=_build_torch_nn("AvgPool3d"),
))

_register(OpSpec(
    canonical="nn.AdaptiveAvgPool1d",
    aliases=("adaptiveavgpool1d", "AdaptiveAvgPool1d"),
    input_rank=(3,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=(),
    param_schema={},
    shape_fn=_shape_adaptive_pool_nd(1),
    build_fn=_build_torch_nn("AdaptiveAvgPool1d"),
))

_register(OpSpec(
    canonical="nn.AdaptiveAvgPool2d",
    aliases=("adaptiveavgpool2d", "AdaptiveAvgPool2d"),
    input_rank=(4,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=(),
    param_schema={},
    shape_fn=_shape_adaptive_pool_nd(2),
    build_fn=_build_torch_nn("AdaptiveAvgPool2d"),
))

_register(OpSpec(
    canonical="nn.AdaptiveAvgPool3d",
    aliases=("adaptiveavgpool3d", "AdaptiveAvgPool3d"),
    input_rank=(5,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=(),
    param_schema={},
    shape_fn=_shape_adaptive_pool_nd(3),
    build_fn=_build_torch_nn("AdaptiveAvgPool3d"),
))

_register(OpSpec(
    canonical="nn.AdaptiveMaxPool1d",
    aliases=("adaptivemaxpool1d", "AdaptiveMaxPool1d"),
    input_rank=(3,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=(),
    param_schema={},
    shape_fn=_shape_adaptive_pool_nd(1),
    build_fn=_build_torch_nn("AdaptiveMaxPool1d"),
))

_register(OpSpec(
    canonical="nn.AdaptiveMaxPool2d",
    aliases=("adaptivemaxpool2d", "AdaptiveMaxPool2d"),
    input_rank=(4,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=(),
    param_schema={},
    shape_fn=_shape_adaptive_pool_nd(2),
    build_fn=_build_torch_nn("AdaptiveMaxPool2d"),
))

_register(OpSpec(
    canonical="nn.AdaptiveMaxPool3d",
    aliases=("adaptivemaxpool3d", "AdaptiveMaxPool3d"),
    input_rank=(5,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=(),
    param_schema={},
    shape_fn=_shape_adaptive_pool_nd(3),
    build_fn=_build_torch_nn("AdaptiveMaxPool3d"),
))

_register(OpSpec(
    canonical="nn.LPPool1d",
    aliases=("lppool1d", "LPPool1d"),
    input_rank=(3,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=(),
    param_schema={"kernel_size": (1, None)},
    shape_fn=_shape_pool_nd(1),
    build_fn=_build_torch_nn("LPPool1d"),
))

_register(OpSpec(
    canonical="nn.LPPool2d",
    aliases=("lppool2d", "LPPool2d"),
    input_rank=(4,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=(),
    param_schema={"kernel_size": (1, None)},
    shape_fn=_shape_pool_nd(2),
    build_fn=_build_torch_nn("LPPool2d"),
))

_register(OpSpec(
    canonical="nn.FractionalMaxPool2d",
    aliases=("fractionalmaxpool2d", "FractionalMaxPool2d"),
    input_rank=(4,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=(),
    param_schema={},
    shape_fn=_shape_fractional_maxpool2d,
    build_fn=_build_torch_nn("FractionalMaxPool2d"),
))

# ── Normalization ─────────────────────────────────────────────────────────────
_register(OpSpec(
    canonical="nn.BatchNorm1d",
    aliases=("batchnorm1d", "BatchNorm1d"),
    input_rank=(2, 3),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=("num_features",),
    param_schema={},
    shape_fn=_shape_batchnorm_nd(),
    build_fn=_build_torch_nn("BatchNorm1d"),
))

_register(OpSpec(
    canonical="nn.BatchNorm2d",
    aliases=("batchnorm2d", "BatchNorm2d"),
    input_rank=(4,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=("num_features",),
    param_schema={},
    shape_fn=_shape_batchnorm_nd(min_rank=4),
    build_fn=_build_torch_nn("BatchNorm2d"),
))

_register(OpSpec(
    canonical="nn.BatchNorm3d",
    aliases=("batchnorm3d", "BatchNorm3d"),
    input_rank=(5,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=("num_features",),
    param_schema={},
    shape_fn=_shape_batchnorm_nd(min_rank=5),
    build_fn=_build_torch_nn("BatchNorm3d"),
))

_register(OpSpec(
    canonical="nn.GroupNorm",
    aliases=("groupnorm", "GroupNorm"),
    input_rank=(),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=("num_groups", "num_channels"),
    param_schema={"num_groups|divides": "num_channels"},
    shape_fn=_shape_groupnorm,
    build_fn=_build_groupnorm,
))

_register(OpSpec(
    canonical="nn.InstanceNorm1d",
    aliases=("instancenorm1d", "InstanceNorm1d"),
    input_rank=(2, 3),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=("num_features",),
    param_schema={},
    shape_fn=_shape_batchnorm_nd(),
    build_fn=_build_torch_nn("InstanceNorm1d"),
))

_register(OpSpec(
    canonical="nn.InstanceNorm2d",
    aliases=("instancenorm2d", "InstanceNorm2d"),
    input_rank=(4,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=("num_features",),
    param_schema={},
    shape_fn=_shape_batchnorm_nd(min_rank=4),
    build_fn=_build_torch_nn("InstanceNorm2d"),
))

_register(OpSpec(
    canonical="nn.InstanceNorm3d",
    aliases=("instancenorm3d", "InstanceNorm3d"),
    input_rank=(5,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=("num_features",),
    param_schema={},
    shape_fn=_shape_batchnorm_nd(min_rank=5),
    build_fn=_build_torch_nn("InstanceNorm3d"),
))

_register(OpSpec(
    canonical="nn.LayerNorm",
    aliases=("layernorm", "LayerNorm"),
    input_rank=(),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=("normalized_shape",),
    param_schema={},
    shape_fn=_shape_layernorm,
    build_fn=_build_layernorm,
))

_register(OpSpec(
    canonical="nn.RMSNorm",
    aliases=("rmsnorm", "RMSNorm"),
    input_rank=(),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=("normalized_shape",),
    param_schema={},
    shape_fn=_shape_layernorm,  # same semantics as LayerNorm
    build_fn=_build_torch_nn("RMSNorm"),
))

_register(OpSpec(
    canonical="nn.LocalResponseNorm",
    aliases=("localresponsenorm", "LocalResponseNorm"),
    input_rank=(),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=(),
    param_schema={},
    shape_fn=_shape_passthrough,
    build_fn=_build_torch_nn("LocalResponseNorm"),
))

# ── Recurrent ─────────────────────────────────────────────────────────────────
_register(OpSpec(
    canonical="nn.LSTM",
    aliases=("lstm", "LSTM"),
    input_rank=(2, 3),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=("input_size", "hidden_size"),
    param_schema={},
    shape_fn=_shape_rnn,
    build_fn=_build_rnn_family("LSTM"),
))

_register(OpSpec(
    canonical="nn.GRU",
    aliases=("gru", "GRU"),
    input_rank=(2, 3),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=("input_size", "hidden_size"),
    param_schema={},
    shape_fn=_shape_rnn,
    build_fn=_build_rnn_family("GRU"),
))

_register(OpSpec(
    canonical="nn.RNN",
    aliases=("rnn", "RNN"),
    input_rank=(2, 3),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=("input_size", "hidden_size"),
    param_schema={},
    shape_fn=_shape_rnn,
    build_fn=_build_rnn_family("RNN"),
))

_register(OpSpec(
    canonical="nn.RNNCell",
    aliases=("rnncell", "RNNCell"),
    input_rank=(2,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=("input_size", "hidden_size"),
    param_schema={},
    shape_fn=lambda p, inp, _: [inp[0], p["hidden_size"]] if len(inp) == 2 and isinstance(p.get("hidden_size"), int) else None,
    build_fn=_build_torch_nn("RNNCell"),
))

_register(OpSpec(
    canonical="nn.GRUCell",
    aliases=("grucell", "GRUCell"),
    input_rank=(2,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=("input_size", "hidden_size"),
    param_schema={},
    shape_fn=lambda p, inp, _: [inp[0], p["hidden_size"]] if len(inp) == 2 and isinstance(p.get("hidden_size"), int) else None,
    build_fn=_build_torch_nn("GRUCell"),
))

_register(OpSpec(
    canonical="nn.LSTMCell",
    aliases=("lstmcell", "LSTMCell"),
    input_rank=(2,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=("input_size", "hidden_size"),
    param_schema={},
    shape_fn=lambda p, inp, _: [inp[0], p["hidden_size"]] if len(inp) == 2 and isinstance(p.get("hidden_size"), int) else None,
    build_fn=_build_torch_nn("LSTMCell"),
))

# ── Embedding ─────────────────────────────────────────────────────────────────
_register(OpSpec(
    canonical="nn.Embedding",
    aliases=("embedding", "Embedding"),
    input_rank=(),
    arity="one",
    in_dtype="int",
    out_dtype="float",
    defining_params=("num_embeddings", "embedding_dim"),
    param_schema={},
    shape_fn=_shape_embedding,
    build_fn=_build_embedding,
))

_register(OpSpec(
    canonical="nn.EmbeddingBag",
    aliases=("embeddingbag", "EmbeddingBag"),
    input_rank=(),
    arity="one",
    in_dtype="int",
    out_dtype="float",
    defining_params=("num_embeddings", "embedding_dim"),
    param_schema={},
    shape_fn=lambda p, inp, _: [inp[0], p["embedding_dim"]] if inp and isinstance(p.get("embedding_dim"), int) else None,
    build_fn=_build_torch_nn("EmbeddingBag"),
))

# ── Attention / Transformer ──────────────────────────────────────────────────
_register(OpSpec(
    canonical="nn.MultiheadAttention",
    aliases=("multiheadattention", "MultiheadAttention"),
    input_rank=(2, 3),
    arity="qkv",
    in_dtype="float",
    out_dtype="float",
    defining_params=("embed_dim", "num_heads"),
    param_schema={"num_heads|divides": "embed_dim"},
    shape_fn=_shape_multihead_attention,
    build_fn=_build_multihead_attention,
))

_register(OpSpec(
    canonical="nn.TransformerEncoderLayer",
    aliases=("transformerencoderlayer", "TransformerEncoderLayer"),
    input_rank=(3,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=("d_model", "nhead"),
    param_schema={"nhead|divides": "d_model"},
    shape_fn=_shape_transformer_layer,
    build_fn=_build_transformer_encoder_layer,
))

_register(OpSpec(
    canonical="nn.TransformerDecoderLayer",
    aliases=("transformerdecoderlayer", "TransformerDecoderLayer"),
    input_rank=(3,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=("d_model", "nhead"),
    param_schema={"nhead|divides": "d_model"},
    shape_fn=_shape_transformer_layer,
    build_fn=_build_transformer_decoder_layer,
))

_register(OpSpec(
    canonical="nn.Transformer",
    aliases=("transformer", "Transformer"),
    input_rank=(3,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=("d_model", "nhead"),
    param_schema={"nhead|divides": "d_model"},
    shape_fn=_shape_transformer_layer,
    build_fn=_build_torch_nn("Transformer"),
))

# ── Tensor manipulation ───────────────────────────────────────────────────────
_register(OpSpec(
    canonical="nn.Flatten",
    aliases=("flatten", "Flatten"),
    input_rank=(),
    arity="one",
    in_dtype="any",
    out_dtype="same",
    defining_params=(),
    param_schema={},
    shape_fn=_shape_flatten,
    build_fn=_build_torch_nn("Flatten"),
))

_register(OpSpec(
    canonical="nn.Unflatten",
    aliases=("unflatten", "Unflatten"),
    input_rank=(),
    arity="one",
    in_dtype="any",
    out_dtype="same",
    defining_params=("dim", "unflattened_size"),
    param_schema={},
    shape_fn=_shape_unflatten,
    build_fn=_build_torch_nn("Unflatten"),
))

_register(OpSpec(
    canonical="nn.Upsample",
    aliases=("upsample", "Upsample"),
    input_rank=(3, 4, 5),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=(),
    param_schema={},
    shape_fn=_shape_upsample,
    build_fn=_build_torch_nn("Upsample"),
))

_register(OpSpec(
    canonical="nn.PixelShuffle",
    aliases=("pixelshuffle", "PixelShuffle"),
    input_rank=(4,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=("upscale_factor",),
    param_schema={},
    shape_fn=_shape_pixel_shuffle,
    build_fn=_build_torch_nn("PixelShuffle"),
))

_register(OpSpec(
    canonical="nn.PixelUnshuffle",
    aliases=("pixelunshuffle", "PixelUnshuffle"),
    input_rank=(4,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=("downscale_factor",),
    param_schema={},
    shape_fn=_shape_pixel_unshuffle,
    build_fn=_build_torch_nn("PixelUnshuffle"),
))

_register(OpSpec(
    canonical="nn.ChannelShuffle",
    aliases=("channelshuffle", "ChannelShuffle"),
    input_rank=(),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=(),
    param_schema={"groups": (1, None)},
    shape_fn=_shape_channel_shuffle,
    build_fn=_build_torch_nn("ChannelShuffle"),
))

_register(OpSpec(
    canonical="nn.Fold",
    aliases=("fold", "Fold"),
    input_rank=(3,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=("output_size", "kernel_size"),
    param_schema={},
    shape_fn=_shape_fold,
    build_fn=_build_torch_nn("Fold"),
))

_register(OpSpec(
    canonical="nn.Unfold",
    aliases=("unfold", "Unfold"),
    input_rank=(4,),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=("kernel_size",),
    param_schema={},
    shape_fn=_shape_unfold,
    build_fn=_build_torch_nn("Unfold"),
))

# Padding ops: output = input with each spatial dim grown by its per-side padding.
for _pad_name in (
    "ZeroPad2d",
    "ConstantPad1d", "ConstantPad2d", "ConstantPad3d",
    "ReflectionPad1d", "ReflectionPad2d", "ReflectionPad3d",
    "ReplicationPad1d", "ReplicationPad2d", "ReplicationPad3d",
):
    _pad_ndim = int(_pad_name[-2])  # "1d"/"2d"/"3d" → 1/2/3
    _register(OpSpec(
        canonical=f"nn.{_pad_name}",
        aliases=(_pad_name.lower(), _pad_name),
        input_rank=(),
        arity="one",
        in_dtype="any",
        out_dtype="same",
        defining_params=(),
        param_schema={},
        shape_fn=_shape_padnd(_pad_ndim),
        build_fn=_build_torch_nn(_pad_name),
    ))

# ── Losses ────────────────────────────────────────────────────────────────────
for _loss_name in (
    "CrossEntropyLoss",
    "MSELoss",
    "L1Loss",
    "BCELoss",
    "BCEWithLogitsLoss",
    "NLLLoss",
    "KLDivLoss",
    "SmoothL1Loss",
    "HuberLoss",
    "TripletMarginLoss",
    "CosineEmbeddingLoss",
    "CTCLoss",
):
    _register(OpSpec(
        canonical=f"nn.{_loss_name}",
        aliases=(_loss_name.lower(), _loss_name),
        input_rank=(),
        arity="loss",
        in_dtype="float",
        out_dtype="float",
        defining_params=(),
        param_schema={},
        shape_fn=_shape_loss(_loss_name),
        build_fn=_build_torch_nn(_loss_name),
    ))

# ── torch.* ops (merge/elementwise) ──────────────────────────────────────────
_register(OpSpec(
    canonical="torch.add",
    aliases=("add", "Add", "torch.add"),
    input_rank=(),
    arity="merge",
    in_dtype="any",
    out_dtype="same",
    defining_params=(),
    param_schema={},
    shape_fn=_shape_add,
    build_fn=None,
))

_register(OpSpec(
    canonical="torch.cat",
    aliases=("cat", "Concat", "torch.cat"),
    input_rank=(),
    arity="merge",
    in_dtype="any",
    out_dtype="same",
    defining_params=(),
    param_schema={},
    shape_fn=_shape_cat,
    build_fn=None,
))

# Shape-preserving binary ops
for _op_name, _aliases in [
    ("torch.mul",  ("mul", "Mul", "torch.mul")),
    ("torch.sub",  ("sub", "Sub", "torch.sub")),
    ("torch.div",  ("div", "Div", "torch.div")),
]:
    _register(OpSpec(
        canonical=_op_name,
        aliases=_aliases,
        input_rank=(),
        arity="merge",
        in_dtype="any",
        out_dtype="same",
        defining_params=(),
        param_schema={},
        shape_fn=_shape_add,  # elementwise: all inputs same shape → same output
        build_fn=None,
    ))

# Shape-preserving unary torch ops
for _op_name, _aliases in [
    ("torch.abs",    ("abs", "torch.abs")),
    ("torch.exp",    ("exp", "torch.exp")),
    ("torch.log",    ("log", "torch.log")),
    ("torch.sqrt",   ("sqrt", "torch.sqrt")),
    ("torch.pow",    ("pow", "torch.pow")),
    ("torch.clamp",  ("clamp", "clip", "torch.clamp", "torch.clip")),
    ("torch.round",  ("round", "torch.round")),
    ("torch.floor",  ("floor", "torch.floor")),
    ("torch.ceil",   ("ceil", "torch.ceil")),
    ("torch.sign",   ("sign", "torch.sign")),
    ("torch.sin",    ("sin", "torch.sin")),
    ("torch.cos",    ("cos", "torch.cos")),
    ("torch.tan",    ("tan", "torch.tan")),
    ("torch.asin",   ("asin", "torch.asin")),
    ("torch.acos",   ("acos", "torch.acos")),
    ("torch.atan",   ("atan", "torch.atan")),
    ("torch.sinh",   ("sinh", "torch.sinh")),
    ("torch.cosh",   ("cosh", "torch.cosh")),
    ("torch.tanh",   ("tanh_fn", "torch.tanh")),  # note: nn.Tanh is separate
]:
    _register(OpSpec(
        canonical=_op_name,
        aliases=_aliases,
        input_rank=(),
        arity="one",
        in_dtype="any",
        out_dtype="same",
        defining_params=(),
        param_schema={},
        shape_fn=_shape_passthrough,
        build_fn=None,
    ))

# matmul / bmm
_register(OpSpec(
    canonical="torch.matmul",
    aliases=("matmul", "torch.matmul"),
    input_rank=(),
    arity="merge",
    in_dtype="float",
    out_dtype="float",
    defining_params=(),
    param_schema={},
    shape_fn=_shape_matmul,
    build_fn=None,
))

_register(OpSpec(
    canonical="torch.bmm",
    aliases=("bmm", "torch.bmm"),
    input_rank=(3,),
    arity="merge",
    in_dtype="float",
    out_dtype="float",
    defining_params=(),
    param_schema={},
    shape_fn=_shape_bmm,
    build_fn=None,
))

# Slicing. torch.narrow(input, dim, start, length) rather than subscript syntax:
# it is an ordinary kwargs call, so codegen's generic torch.* tail emits it with no
# new branch, and emitting `x[:, a:b]` would need the tensor RANK at emission time,
# which _emit_op does not have.
_register(OpSpec(
    canonical="torch.narrow",
    aliases=("narrow", "slice", "torch.narrow"),
    input_rank=(),
    arity="one",
    in_dtype="any",
    out_dtype="same",
    defining_params=(),
    param_schema={},
    shape_fn=_shape_narrow,
    build_fn=None,
))

# F.interpolate. Same size/scale_factor/mode semantics as nn.Upsample, so it
# shares _shape_upsample rather than duplicating the arithmetic. NOTE for codegen:
# there is no `torch.interpolate` — it must be emitted as `F.interpolate`.
_register(OpSpec(
    canonical="interpolate",
    # spec_for() strips a leading "torch.nn." before lookup, so the fully
    # qualified name arrives as "functional.interpolate" — it needs its own alias.
    aliases=("interpolate", "F.interpolate", "functional.interpolate",
             "torch.nn.functional.interpolate"),
    input_rank=(3, 4, 5),
    arity="one",
    in_dtype="float",
    out_dtype="float",
    defining_params=(),
    param_schema={},
    shape_fn=_shape_upsample,
    build_fn=None,
))

# Multi-input ops that were mis-declared as arity="one" AND as shape-preserving.
# stack inserts an axis; where broadcasts its three operands. Both sat on
# _shape_passthrough, so both reported their first input's shape verbatim.
_register(OpSpec(
    canonical="torch.stack",
    aliases=("stack", "torch.stack"),
    input_rank=(),
    arity="merge",
    in_dtype="any",
    out_dtype="same",
    defining_params=(),
    param_schema={},
    shape_fn=_shape_stack,
    build_fn=None,
))

_register(OpSpec(
    canonical="torch.where",
    aliases=("where", "torch.where"),
    input_rank=(),
    arity="merge",
    in_dtype="any",
    out_dtype="same",
    defining_params=(),
    param_schema={},
    shape_fn=_shape_where,
    build_fn=None,
))

# Reductions (dim-aware: drop `dim`, keep as 1 with keepdim).
# torch.sum was the one op here that reported a shape it did not have: it sat on
# _shape_passthrough because the real reduction dropped the strict single-template
# sweep 612 -> 610 on message_passing_block ("Linear in_features mismatch: expected
# 64, got 1"). That was the passthrough masking a wrong aggregation AXIS in the
# template — an MPNN must sum over neighbours, not over features — not a cost of
# telling the truth. Template fixed in canonical_template_registry.py, sweep back at
# 612/612, and all six now carry the real reduction rule.
for _op_name, _aliases, _sfn, _out_dtype in [
    ("torch.sum",    ("sum", "torch.sum"),        _shape_reduce,      "same"),
    ("torch.mean",   ("mean", "torch.mean"),      _shape_reduce,      "same"),
    ("torch.max",    ("max", "torch.max"),        _shape_reduce,      "same"),
    ("torch.min",    ("min", "torch.min"),        _shape_reduce,      "same"),
    ("torch.argmax", ("argmax", "torch.argmax"),  _shape_reduce,      "int"),
    ("torch.argmin", ("argmin", "torch.argmin"),  _shape_reduce,      "int"),
]:
    _register(OpSpec(
        canonical=_op_name,
        aliases=_aliases,
        input_rank=(),
        arity="one",
        in_dtype="any",
        out_dtype=_out_dtype,
        defining_params=(),
        param_schema={},
        shape_fn=_sfn,
        build_fn=None,
    ))

# Movement ops, all deriving from params.
# chunk/split DECLINE: they return tuples, which a single output_shape per node
# cannot represent (see _shape_declines_multi_output). Reporting the input shape
# unchanged — what they did before — was a lie the rest of the graph then trusted.
# reshape now derives from its `shape` param and declines without one; the strict
# template sweep was re-measured at 612/612 after the change.
for _op_name, _aliases, _sfn in [
    ("torch.chunk",     ("chunk", "torch.chunk"),         _shape_declines_multi_output),
    ("torch.split",     ("split", "torch.split"),         _shape_declines_multi_output),
    ("torch.reshape",   ("reshape", "torch.reshape"),     _shape_reshape),
    ("torch.squeeze",   ("squeeze", "torch.squeeze"),     _shape_squeeze),
    ("torch.unsqueeze", ("unsqueeze", "torch.unsqueeze"), _shape_unsqueeze),
    ("torch.transpose", ("transpose", "torch.transpose"), _shape_transpose),
    ("torch.permute",   ("permute", "torch.permute"),     _shape_permute),
]:
    _register(OpSpec(
        canonical=_op_name,
        aliases=_aliases,
        input_rank=(),
        arity="one",
        in_dtype="any",
        out_dtype="same",
        defining_params=(),
        param_schema={},
        shape_fn=_sfn,
        build_fn=None,
    ))


# ── Placeholder ops (no torch counterpart) ───────────────────────────────────
# Two ids stand in for code the registry deliberately does NOT model:
#   `python`      — an opaque statement the AST parser could not recognise
#                   (x.transpose(1, 2), a bare call, anything inside a compound
#                   statement it walked but cannot type).
#   `custom_code` — an instance of a user-defined nn.Module.
# Both DECLINE a shape rather than guess one: a wrong shape propagates through
# the whole downstream graph, while a declined one just shows as unknown
# (docs/tensorviz-plan.md §11.7). Neither has a build_fn — nothing here can be
# constructed from parameters alone; the real object lives in the user's file.
# They are registered so `spec_for` resolves them (the palette-coverage gate),
# NOT so they gain a shape contract, which is why they are kept out of
# `all_specs()` — see `_PLACEHOLDER_REGISTRY` above.
_register_placeholder(OpSpec(
    canonical="python",
    aliases=("python",),
    input_rank=(),
    arity="any",
    in_dtype="any",
    out_dtype="same",
    defining_params=(),
    param_schema={},
    shape_fn=_shape_placeholder_recorded,
    build_fn=None,
))

_register_placeholder(OpSpec(
    canonical="custom_code",
    aliases=("custom_code",),
    input_rank=(),
    arity="any",
    in_dtype="any",
    out_dtype="same",
    defining_params=(),
    param_schema={},
    shape_fn=_shape_placeholder_recorded,
    build_fn=None,
))
