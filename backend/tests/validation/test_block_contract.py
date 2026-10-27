"""Tier-0 block contract (docs/tensorviz-plan.md §5.3, §9 Tier 0).

For every op in the registry, the ANALYTIC output shape (op_registry + shape_engine)
must equal a REAL torch forward pass on a canonical input. This is the moat, made
executable: analytic and torch agree, per block.

Ops whose registry rule cannot match torch are documented in KNOWN_GAPS and marked
xfail(strict=False) — the suite is GREEN today (136 of 138 ops locked in, 2 gaps),
and each gap auto-flips to XPASS the moment its rule is implemented. Losses carry
no shape contract (always scalar) and are skipped with a reason.

Sibling suite: test_op_shape_parity.py checks the same analytic==torch property
with hand-written cases, including the DECLINE cases (a rule returning None) that
this suite cannot express.

Run: KMP_DUPLICATE_LIB_OK=TRUE <viz-python> -m pytest tests/test_block_contract.py -q
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # must precede torch import

import pytest
import torch

from core.op_registry import all_specs
from core.layer_builder import build_layer, call_module
from core.shape_engine import infer
from models.architecture_models import ArchitectureEdge, ArchitectureNode


def torch_class(canonical: str) -> str:
    for p in ("torch.nn.", "nn.", "torch."):
        if canonical.startswith(p):
            return canonical[len(p):]
    return canonical


# ── op categories keyed by the bare (torch_class) name ─────────────────────────
REDUCTIONS = {"sum", "mean", "max", "min", "argmax", "argmin"}
MOVEMENT = {"reshape", "permute", "transpose", "unsqueeze", "squeeze", "stack", "chunk", "split"}
MATMUL = {"matmul", "bmm"}
MERGE = {"add", "sub", "mul", "div", "cat"}
TWO_INPUT = MERGE | MATMUL | {"Bilinear"}


def build_case(spec):
    """Return (input_shapes:list[list[int]], params:dict, dtype:str). Fail-loud on
    a registry op with no case so adding an op forces adding coverage here."""
    name = torch_class(spec.canonical)
    dtype = "long" if getattr(spec, "in_dtype", None) == "int" else "float"

    # --- convolutions ---
    if name in ("Conv1d", "ConvTranspose1d"):
        return [[1, 4, 16]], dict(in_channels=4, out_channels=8, kernel_size=3, padding=1), dtype
    if name in ("Conv2d", "ConvTranspose2d"):
        return [[1, 4, 8, 8]], dict(in_channels=4, out_channels=8, kernel_size=3, padding=1), dtype
    if name in ("Conv3d", "ConvTranspose3d"):
        return [[1, 4, 4, 8, 8]], dict(in_channels=4, out_channels=8, kernel_size=3, padding=1), dtype
    # --- pools ---
    if name in ("MaxPool1d", "AvgPool1d"):
        return [[1, 4, 16]], dict(kernel_size=2), dtype
    if name == "LPPool1d":
        return [[1, 4, 16]], dict(norm_type=2, kernel_size=2), dtype
    if name in ("MaxPool2d", "AvgPool2d"):
        return [[1, 4, 8, 8]], dict(kernel_size=2), dtype
    if name == "LPPool2d":
        return [[1, 4, 8, 8]], dict(norm_type=2, kernel_size=2), dtype
    if name == "FractionalMaxPool2d":
        return [[1, 4, 8, 8]], dict(kernel_size=2, output_ratio=0.5), dtype
    if name in ("MaxPool3d", "AvgPool3d"):
        return [[1, 4, 4, 8, 8]], dict(kernel_size=2), dtype
    if name in ("AdaptiveAvgPool1d", "AdaptiveMaxPool1d"):
        return [[1, 4, 16]], dict(output_size=4), dtype
    if name in ("AdaptiveAvgPool2d", "AdaptiveMaxPool2d"):
        return [[1, 4, 8, 8]], dict(output_size=(4, 4)), dtype
    if name in ("AdaptiveAvgPool3d", "AdaptiveMaxPool3d"):
        return [[1, 4, 4, 8, 8]], dict(output_size=(2, 4, 4)), dtype
    # --- norms ---
    if name == "BatchNorm1d":
        return [[4, 8]], dict(num_features=8), dtype
    if name == "InstanceNorm1d":
        return [[2, 8, 16]], dict(num_features=8), dtype
    if name in ("BatchNorm2d", "InstanceNorm2d"):
        return [[1, 8, 8, 8]], dict(num_features=8), dtype
    if name in ("BatchNorm3d", "InstanceNorm3d"):
        return [[1, 8, 4, 8, 8]], dict(num_features=8), dtype
    if name == "GroupNorm":
        return [[1, 8, 8, 8]], dict(num_groups=2, num_channels=8), dtype
    if name in ("LayerNorm", "RMSNorm"):
        return [[2, 10, 16]], dict(normalized_shape=[16]), dtype
    if name == "LocalResponseNorm":
        return [[1, 8, 8, 8]], dict(size=2), dtype
    # --- linear family ---
    if name == "Linear":
        return [[4, 16]], dict(in_features=16, out_features=32), dtype
    if name == "LazyLinear":
        return [[4, 16]], dict(out_features=32), dtype
    if name == "Bilinear":
        return [[4, 16], [4, 16]], dict(in1_features=16, in2_features=16, out_features=32), dtype
    # --- embedding (long inputs) ---
    if name in ("Embedding", "EmbeddingBag"):
        return [[4, 10]], dict(num_embeddings=100, embedding_dim=16), "long"
    # --- reshape / movement layers (nn.*) ---
    if name == "Flatten":
        return [[2, 3, 4, 4]], {}, dtype
    if name == "Unflatten":
        return [[2, 16]], dict(dim=1, unflattened_size=(4, 4)), dtype
    if name == "Unfold":
        return [[1, 4, 8, 8]], dict(kernel_size=3), dtype
    if name == "Fold":
        return [[1, 36, 36]], dict(output_size=(8, 8), kernel_size=3), dtype
    if name == "PixelShuffle":
        return [[1, 16, 4, 4]], dict(upscale_factor=2), dtype
    if name == "PixelUnshuffle":
        return [[1, 4, 8, 8]], dict(downscale_factor=2), dtype
    if name == "ChannelShuffle":
        return [[1, 8, 8, 8]], dict(groups=2), dtype
    if name == "GLU":
        return [[4, 16]], dict(dim=-1), dtype
    if name == "Upsample":
        return [[1, 4, 8, 8]], dict(scale_factor=2), dtype
    # --- recurrent ---
    if name in ("LSTM", "GRU", "RNN"):
        return [[2, 10, 16]], dict(input_size=16, hidden_size=24, batch_first=True), dtype
    if name in ("LSTMCell", "GRUCell", "RNNCell"):
        return [[4, 16]], dict(input_size=16, hidden_size=24), dtype
    # --- attention / transformer ---
    if name == "MultiheadAttention":
        return [[2, 10, 16]], dict(embed_dim=16, num_heads=4, batch_first=True), dtype
    if name in ("TransformerEncoderLayer", "TransformerDecoderLayer", "Transformer"):
        return [[2, 10, 16]], dict(d_model=16, nhead=4, batch_first=True), dtype
    # --- padding ---
    if name in ("ZeroPad2d", "ReflectionPad2d", "ReplicationPad2d", "ConstantPad2d"):
        p = dict(padding=1)
        if name == "ConstantPad2d":
            p["value"] = 0.0
        return [[1, 4, 8, 8]], p, dtype
    if name in ("ReflectionPad1d", "ReplicationPad1d", "ConstantPad1d"):
        p = dict(padding=1)
        if name == "ConstantPad1d":
            p["value"] = 0.0
        return [[1, 4, 16]], p, dtype
    if name in ("ReflectionPad3d", "ReplicationPad3d", "ConstantPad3d"):
        p = dict(padding=1)
        if name == "ConstantPad3d":
            p["value"] = 0.0
        return [[1, 4, 4, 8, 8]], p, dtype
    if name == "Threshold":
        return [[4, 16]], dict(threshold=0.0, value=0.0), dtype
    # --- functional torch.* families ---
    if name in MERGE:
        return ([[2, 4, 8], [2, 4, 8]], {"dim": 1} if name == "cat" else {}, dtype)
    if name in MATMUL:
        return [[2, 4, 8], [2, 8, 4]], {}, dtype
    if name in REDUCTIONS:
        return [[2, 4, 8]], dict(dim=1), dtype
    if name == "reshape":
        return [[2, 4, 8]], dict(shape=(2, 32)), dtype
    if name == "permute":
        return [[2, 4, 8]], dict(dims=(0, 2, 1)), dtype
    if name == "transpose":
        return [[2, 4, 8]], dict(dims=(1, 2)), dtype
    if name == "unsqueeze":
        return [[2, 4, 8]], dict(dim=1), dtype
    if name == "squeeze":
        return [[2, 1, 8]], {}, dtype
    if name == "narrow":
        return [[1, 8, 8]], dict(dim=1, start=2, length=4), dtype
    if name == "interpolate":
        return [[1, 4, 8, 8]], dict(scale_factor=2, mode="nearest"), dtype
    if name == "stack":
        # TWO inputs: stack is arity="merge" and its output gains an axis of
        # len(inputs), so a single-input case would assert the wrong truth.
        return [[2, 4, 8], [2, 4, 8]], dict(dim=1), dtype
    if name in ("chunk", "split"):
        return [[2, 4, 8]], dict(dim=1), dtype
    if name == "clamp":
        return [[2, 4, 8]], {}, dtype
    if name == "pow":
        return [[2, 4, 8]], {}, dtype
    if name == "where":
        return [[2, 4, 8]], {}, dtype
    # --- losses (skipped in _cases; case unused) ---
    if getattr(spec, "arity", None) == "loss":
        return [[4, 8]], {}, dtype
    # --- generic functional (unary, shape-preserving) ---
    if getattr(spec, "build_fn", None) is None:
        return [[2, 4, 8]], {}, dtype
    # --- generic layer (activations / dropout / identity) ---
    return [[4, 16]], {}, dtype


def _make(shape, dtype):
    if dtype == "long":
        return torch.randint(0, 100, tuple(shape))
    return torch.randn(*shape)


def analytic_shape(spec, input_shapes, params):
    nodes, edges, init = [], [], {}
    for i, s in enumerate(input_shapes):
        nodes.append(ArchitectureNode(node_id=f"in{i}", kind="input", op="input", params={}))
        edges.append(
            ArchitectureEdge(edge_id=f"e{i}", source_node_id=f"in{i}", source_port="y",
                             target_node_id="op", target_port="x")
        )
        init[f"in{i}"] = s
    nodes.append(ArchitectureNode(node_id="op", kind="layer", op=spec.canonical, params=params))
    return infer(nodes, edges, init)["op"].output_shape


def _functional_truth(name, xs, params):
    x = xs[0]
    if name == "cat":
        return torch.cat([xs[0], xs[1]], dim=params.get("dim", 1))
    if name in ("add", "sub", "mul", "div"):
        return {"add": torch.add, "sub": torch.sub, "mul": torch.mul, "div": torch.div}[name](xs[0], xs[1])
    if name in MATMUL:
        return getattr(torch, name)(xs[0], xs[1])
    if name in REDUCTIONS:
        d = params.get("dim", 1)
        if name in ("max", "min"):
            return getattr(torch, name)(x, dim=d).values
        return getattr(torch, name)(x, dim=d)
    if name == "reshape":
        return torch.reshape(x, tuple(params["shape"]))
    if name == "permute":
        return torch.permute(x, tuple(params["dims"]))
    if name == "transpose":
        return torch.transpose(x, *params["dims"])
    if name == "unsqueeze":
        return torch.unsqueeze(x, params["dim"])
    if name == "squeeze":
        return torch.squeeze(x)
    if name == "stack":
        return torch.stack([x, x], dim=params.get("dim", 0))
    if name == "chunk":
        return torch.chunk(x, 2, dim=params.get("dim", 1))[0]
    if name == "split":
        return torch.split(x, 2, dim=params.get("dim", 1))[0]
    if name == "narrow":
        return torch.narrow(x, params["dim"], params["start"], params["length"])
    if name == "interpolate":
        # There is no torch.interpolate; it lives on torch.nn.functional. Codegen
        # has to emit F.interpolate for exactly the same reason.
        return torch.nn.functional.interpolate(
            x, **{k: v for k, v in params.items() if k in ("size", "scale_factor", "mode")})
    if name == "clamp":
        return torch.clamp(x, min=0.0, max=1.0)
    if name == "pow":
        return torch.pow(torch.abs(x) + 0.5, 2)
    if name == "where":
        return torch.where(x > 0, x, x)
    return getattr(torch, name)(x)  # unary shape-preserving


def torch_truth(spec, input_shapes, params, dtype):
    name = torch_class(spec.canonical)
    xs = [_make(s, dtype) for s in input_shapes]
    with torch.no_grad():
        if name == "Bilinear":
            return list(build_layer("Bilinear", params).eval()(xs[0], xs[1]).shape)
        if getattr(spec, "build_fn", None) is not None:
            return list(call_module(build_layer(name, params).eval(), xs[0]).shape)
        return list(_functional_truth(name, xs, params).shape)


# ── Known registry placeholder shape rules (the remaining gaps). xfail(strict=False)
# so each auto-flips to XPASS when its rule is implemented (Steps B2/B3 etc.). ────
KNOWN_GAPS = {
    # chunk/split DECLINE (return None) rather than guess: they produce tuples, and a
    # node carries one output_shape with no output index on its edges, so the result is
    # genuinely unrepresentable. Not a placeholder — a deliberate refusal. They stay
    # listed because a declining rule cannot match torch's truth by definition.
    "torch.chunk", "torch.split",
    # NOTE: torch.reshape, torch.stack and torch.sum used to be here and are now FIXED.
    # reshape derives from its `shape` param (the ssd_head_block template that depended
    # on the placeholder now uses nn.Flatten), stack inserts its axis properly, and sum
    # carries the real reduction now that message_passing_block aggregates over the
    # node axis instead of the feature axis.
}


def _cases():
    out = []
    for spec in sorted(all_specs(), key=lambda s: s.canonical):
        marks = []
        if getattr(spec, "arity", None) == "loss":
            marks.append(pytest.mark.skip(reason="loss = scalar reduction; no shape contract"))
        elif spec.canonical in KNOWN_GAPS:
            marks.append(pytest.mark.xfail(strict=False,
                         reason=f"registry placeholder shape rule for {spec.canonical}"))
        out.append(pytest.param(spec, id=spec.canonical, marks=marks))
    return out


@pytest.mark.parametrize("spec", _cases())
def test_block_contract(spec):
    input_shapes, params, dtype = build_case(spec)
    analytic = analytic_shape(spec, input_shapes, params)
    truth = torch_truth(spec, input_shapes, params, dtype)
    assert analytic == truth, f"{spec.canonical}: analytic={analytic} torch={truth}"


def test_coverage_and_no_regression(capsys):
    """Independent registry walk: prints the tally and is the real regression gate."""
    ran = passed = 0
    unexpected_fail, unexpected_pass = [], []
    for spec in all_specs():
        if getattr(spec, "arity", None) == "loss":
            continue
        input_shapes, params, dtype = build_case(spec)  # raises if a new op lacks a case
        try:
            ok = analytic_shape(spec, input_shapes, params) == torch_truth(spec, input_shapes, params, dtype)
        except Exception:
            ok = False
        ran += 1
        passed += int(ok)
        if not ok and spec.canonical not in KNOWN_GAPS:
            unexpected_fail.append(spec.canonical)
        if ok and spec.canonical in KNOWN_GAPS:
            unexpected_pass.append(spec.canonical)

    with capsys.disabled():
        print(f"\n[Tier-0 block contract] {passed}/{ran} ops analytic==torch "
              f"({len(KNOWN_GAPS)} known gaps, losses skipped)")
        if unexpected_pass:
            print("FIXED (prune from KNOWN_GAPS): " + ", ".join(sorted(unexpected_pass)))

    assert not unexpected_fail, f"Non-gap ops regressed (analytic != torch): {unexpected_fail}"
    stale = KNOWN_GAPS - {s.canonical for s in all_specs()}
    assert not stale, f"KNOWN_GAPS references removed/misspelled ops: {stale}"
