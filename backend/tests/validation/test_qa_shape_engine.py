"""QA regression tests — Shape engine & op registry: analytic == torch forward.

Area: PRs #31, #35, #36, #39, #45.

The contract of ``core/op_registry.py`` shape functions (as consumed by
``core/shape_engine.infer``) is that the ANALYTIC output shape must match what
the corresponding ``torch.nn`` module / ``torch`` op actually produces on a
concrete forward pass.  Every parity test below builds the real torch module
(or runs the real torch op) as the oracle and compares it to the engine — so a
regression in the shape math makes the test fail rather than silently drift.

These tests deliberately target the last-2-days behaviors and gaps not already
covered by ``test_shape_engine.py`` / ``test_op_registry.py``:
  * odd kernel/stride/padding/dilation combinations for conv & pool
  * dim-aware reductions (mean/max) incl. keepdim, and the *intentional*
    torch.sum passthrough exception
  * matmul (1-D promotion + batch broadcasting) and bmm through the full engine
  * exact movement-op derivation (transpose/permute/squeeze/unsqueeze)
  * string padding modes ('same' / 'valid')
  * dtype propagation (Embedding int->float) and param-schema divisibility
  * a small end-to-end CNN graph vs a torch ``nn.Sequential``

Five confirmed bugs are pinned with ``xfail(strict=False)`` so the suite stays
green while documenting the exact regression each one represents.
"""



import pytest
import torch
import torch.nn as nn

from core.shape_engine import infer, NodeShapeResult
from core.op_registry import spec_for
from models.architecture_models import ArchitectureNode, ArchitectureEdge


# ── Helpers ──────────────────────────────────────────────────────────────────

def _node(node_id, op, kind="layer", params=None):
    return ArchitectureNode(node_id=node_id, op=op, kind=kind, params=params or {})


def _input_node(node_id="input"):
    return ArchitectureNode(node_id=node_id, op="input", kind="input", params={})


def _output_node(node_id="output"):
    return ArchitectureNode(node_id=node_id, op="output", kind="output", params={})


def _edge(src, tgt):
    return ArchitectureEdge(
        edge_id=f"{src}_{tgt}",
        source_node_id=src,
        source_port="y",
        target_node_id=tgt,
        target_port="x",
    )


def _run_single(op, params, in_shape) -> NodeShapeResult:
    """Run a single op through the full engine: input -> op -> output."""
    nodes = [_input_node(), _node("n", op, params=params), _output_node()]
    edges = [_edge("input", "n"), _edge("n", "output")]
    return infer(nodes, edges, {"input": in_shape})["n"]


def _run_merge(op, a_shape, b_shape, params=None) -> NodeShapeResult:
    """Run a 2-input merge op through the engine: (a, b) -> op -> output.

    Edge insertion order fixes the operand order (a first, b second)."""
    nodes = [_input_node("a"), _input_node("b"), _node("m", op, params=params), _output_node()]
    edges = [_edge("a", "m"), _edge("b", "m"), _edge("m", "output")]
    return infer(nodes, edges, {"a": a_shape, "b": b_shape})["m"]


def _torch_out_shape(module, in_shape):
    with torch.no_grad():
        return list(module(torch.randn(*in_shape)).shape)


def _assert_parity(op, params, in_shape, module):
    """Engine analytic shape must equal the real torch forward output shape."""
    res = _run_single(op, params, in_shape)
    torch_shape = _torch_out_shape(module, in_shape)
    assert res.output_shape == torch_shape, (
        f"{op} params={params} in={in_shape}: engine={res.output_shape} torch={torch_shape}"
    )
    assert res.shape_error is None


# ── Conv parity (odd kernel / stride / padding / dilation) ────────────────────

@pytest.mark.parametrize("k,s,p,d,in_shape", [
    (3, 1, 1, 1, [1, 3, 64, 64]),       # baseline "same-size"
    (5, 2, 2, 1, [1, 3, 31, 31]),       # odd input, strided, padded
    (3, 1, 0, 2, [1, 3, 20, 20]),       # dilation, valid
    (7, 3, 3, 1, [2, 3, 40, 40]),       # large kernel, stride 3
    (1, 1, 0, 1, [1, 3, 13, 17]),       # 1x1 conv on non-square
])
def test_conv2d_parity(k, s, p, d, in_shape):
    params = {"in_channels": 3, "out_channels": 8, "kernel_size": k,
              "stride": s, "padding": p, "dilation": d}
    module = nn.Conv2d(3, 8, kernel_size=k, stride=s, padding=p, dilation=d)
    _assert_parity("nn.Conv2d", params, in_shape, module)


def test_conv2d_asymmetric_kernel_stride_padding_parity():
    """Per-axis (tuple) kernel/stride/padding must zip positionally with H, W."""
    params = {"in_channels": 3, "out_channels": 8,
              "kernel_size": [3, 5], "stride": [2, 1], "padding": [1, 2]}
    module = nn.Conv2d(3, 8, kernel_size=(3, 5), stride=(2, 1), padding=(1, 2))
    _assert_parity("nn.Conv2d", params, [1, 3, 28, 32], module)


def test_conv1d_strided_parity():
    params = {"in_channels": 4, "out_channels": 8, "kernel_size": 3, "stride": 2, "padding": 1}
    module = nn.Conv1d(4, 8, 3, stride=2, padding=1)
    _assert_parity("nn.Conv1d", params, [2, 4, 50], module)


def test_conv_transpose2d_output_padding_parity():
    params = {"in_channels": 4, "out_channels": 8, "kernel_size": 3,
              "stride": 2, "padding": 1, "output_padding": 1}
    module = nn.ConvTranspose2d(4, 8, 3, stride=2, padding=1, output_padding=1)
    _assert_parity("nn.ConvTranspose2d", params, [1, 4, 16, 16], module)


def test_conv_transpose1d_parity():
    params = {"in_channels": 4, "out_channels": 8, "kernel_size": 3,
              "stride": 2, "padding": 1, "output_padding": 1}
    module = nn.ConvTranspose1d(4, 8, 3, stride=2, padding=1, output_padding=1)
    _assert_parity("nn.ConvTranspose1d", params, [2, 4, 20], module)


def test_conv2d_channel_mismatch_flags_error():
    """in_channels != input channel dim => no valid shape (engine must refuse)."""
    res = _run_single("nn.Conv2d",
                      {"in_channels": 5, "out_channels": 8, "kernel_size": 3, "padding": 1},
                      [1, 3, 16, 16])
    assert res.output_shape is None
    assert res.shape_error is not None


# ── Pool parity ───────────────────────────────────────────────────────────────

def test_maxpool2d_odd_kernel_stride_padding_parity():
    params = {"kernel_size": 3, "stride": 2, "padding": 1}
    module = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    _assert_parity("nn.MaxPool2d", params, [1, 4, 15, 15], module)


def test_maxpool2d_stride_defaults_to_kernel_parity():
    """When stride key is ABSENT, torch (and the engine) default stride=kernel_size."""
    params = {"kernel_size": 2}  # note: no stride key at all
    module = nn.MaxPool2d(kernel_size=2)  # torch defaults stride=kernel_size
    _assert_parity("nn.MaxPool2d", params, [1, 4, 8, 8], module)


def test_adaptive_avg_pool2d_parity():
    params = {"output_size": [1, 1]}
    module = nn.AdaptiveAvgPool2d((1, 1))
    _assert_parity("nn.AdaptiveAvgPool2d", params, [2, 64, 7, 7], module)


def test_adaptive_avg_pool2d_nonunit_parity():
    params = {"output_size": [4, 6]}
    module = nn.AdaptiveAvgPool2d((4, 6))
    _assert_parity("nn.AdaptiveAvgPool2d", params, [2, 32, 13, 17], module)


# ── Reductions (dim-aware) ───────────────────────────────────────────────────

def test_mean_reduce_dim_parity():
    """mean carries the real reduction rule: dropping `dim`."""
    x = torch.randn(2, 4, 8)
    out = spec_for("mean").shape_fn({"dim": 1}, [2, 4, 8], [[2, 4, 8]])
    assert out == list(x.mean(dim=1).shape) == [2, 8]


def test_mean_reduce_keepdim_parity():
    x = torch.randn(2, 4, 8)
    out = spec_for("mean").shape_fn({"dim": 1, "keepdim": True}, [2, 4, 8], [[2, 4, 8]])
    assert out == list(x.mean(dim=1, keepdim=True).shape) == [2, 1, 8]


def test_mean_full_reduction_is_scalar():
    x = torch.randn(2, 4, 8)
    out = spec_for("mean").shape_fn({}, [2, 4, 8], [[2, 4, 8]])
    assert out == list(x.mean().shape) == []


def test_max_reduce_neg_dim_keepdim_parity():
    x = torch.randn(2, 4, 8)
    vals, _ = x.max(dim=-1, keepdim=True)
    out = spec_for("max").shape_fn({"dim": -1, "keepdim": True}, [2, 4, 8], [[2, 4, 8]])
    assert out == list(vals.shape) == [2, 4, 1]


def test_sum_reduces_exactly_like_mean():
    """torch.sum used to be registered as passthrough while mean, max, min, argmax and
    argmin all reduced — one reduction reporting its input shape verbatim, which every
    downstream node then trusted. sum and mean take the same arguments and drop the same
    axis, so any divergence between these two lines is the bug coming back."""
    out = spec_for("sum").shape_fn({"dim": 1}, [2, 4, 8], [[2, 4, 8]])
    assert out == [2, 8], "torch.sum must reduce the dim it is given"
    assert spec_for("mean").shape_fn({"dim": 1}, [2, 4, 8], [[2, 4, 8]]) == out


# ── matmul / bmm through the engine ──────────────────────────────────────────

def test_matmul_batch_broadcast_via_engine():
    a, b = [3, 1, 4, 5], [1, 6, 5, 7]
    res = _run_merge("torch.matmul", a, b)
    torch_shape = list((torch.randn(*a) @ torch.randn(*b)).shape)
    assert res.output_shape == torch_shape == [3, 6, 4, 7]
    assert res.arity_warning is None  # matmul is a merge op, not single-input


def test_matmul_vector_promotion():
    """1-D operands are promoted then squeezed exactly like torch.matmul."""
    # (k) @ (k, n) -> (n)
    out = spec_for("matmul").shape_fn({}, [5], [[5], [5, 7]])
    assert out == list((torch.randn(5) @ torch.randn(5, 7)).shape) == [7]
    # (m, k) @ (k) -> (m)
    out = spec_for("matmul").shape_fn({}, [4, 5], [[4, 5], [5]])
    assert out == list((torch.randn(4, 5) @ torch.randn(5)).shape) == [4]


def test_matmul_inner_mismatch_returns_none():
    out = spec_for("matmul").shape_fn({}, [2, 3], [[2, 3], [4, 5]])
    assert out is None


def test_bmm_via_engine_parity():
    a, b = [8, 3, 4], [8, 4, 5]
    res = _run_merge("torch.bmm", a, b)
    torch_shape = list(torch.bmm(torch.randn(*a), torch.randn(*b)).shape)
    assert res.output_shape == torch_shape == [8, 3, 5]


def test_bmm_batch_mismatch_returns_none():
    out = spec_for("bmm").shape_fn({}, [8, 3, 4], [[8, 3, 4], [2, 4, 5]])
    assert out is None


# ── Movement ops (exact derivation) ──────────────────────────────────────────

def test_transpose_swaps_two_dims_parity():
    x = torch.randn(2, 3, 4)
    res = _run_single("torch.transpose", {"dims": [1, 2]}, [2, 3, 4])
    assert res.output_shape == list(x.transpose(1, 2).shape) == [2, 4, 3]


def test_transpose_negative_dim_parity():
    x = torch.randn(2, 3, 4)
    out = spec_for("transpose").shape_fn({"dims": [0, -1]}, [2, 3, 4], [[2, 3, 4]])
    assert out == list(x.transpose(0, -1).shape) == [4, 3, 2]


def test_permute_reorders_dims_parity():
    x = torch.randn(2, 3, 4)
    res = _run_single("torch.permute", {"dims": [2, 0, 1]}, [2, 3, 4])
    assert res.output_shape == list(x.permute(2, 0, 1).shape) == [4, 2, 3]


def test_permute_invalid_returns_none():
    out = spec_for("permute").shape_fn({"dims": [0, 0, 1]}, [2, 3, 4], [[2, 3, 4]])
    assert out is None


def test_squeeze_specific_dim_parity():
    x = torch.randn(2, 1, 4)
    # squeezing a size-1 dim removes it; squeezing a non-1 dim is a no-op (matches torch).
    assert spec_for("squeeze").shape_fn({"dim": 1}, [2, 1, 4], [[2, 1, 4]]) == list(x.squeeze(1).shape) == [2, 4]
    assert spec_for("squeeze").shape_fn({"dim": 0}, [2, 1, 4], [[2, 1, 4]]) == list(x.squeeze(0).shape) == [2, 1, 4]


def test_unsqueeze_parity():
    x = torch.randn(2, 4)
    res = _run_single("torch.unsqueeze", {"dim": -1}, [2, 4])
    assert res.output_shape == list(x.unsqueeze(-1).shape) == [2, 4, 1]


# ── String padding modes ('same' / 'valid') ──────────────────────────────────

def test_conv2d_padding_same_stride1_parity():
    params = {"in_channels": 3, "out_channels": 8, "kernel_size": 3, "padding": "same"}
    module = nn.Conv2d(3, 8, 3, padding="same")
    _assert_parity("nn.Conv2d", params, [1, 3, 15, 17], module)


def test_conv2d_padding_valid_nonsquare_parity():
    params = {"in_channels": 3, "out_channels": 8, "kernel_size": 3, "padding": "valid"}
    module = nn.Conv2d(3, 8, 3, padding="valid")
    _assert_parity("nn.Conv2d", params, [1, 3, 10, 14], module)


# ── dtype propagation & param-schema ─────────────────────────────────────────

def test_linear_on_3d_input_parity():
    params = {"in_features": 128, "out_features": 64}
    module = nn.Linear(128, 64)
    _assert_parity("nn.Linear", params, [2, 10, 128], module)


def test_embedding_shape_and_dtype():
    """Embedding: int indices in, float table out; adds embedding_dim as last axis."""
    res = _run_single("nn.Embedding", {"num_embeddings": 100, "embedding_dim": 16}, [4, 10])
    tokens = torch.randint(0, 100, (4, 10))
    torch_shape = list(nn.Embedding(100, 16)(tokens).shape)
    assert res.output_shape == torch_shape == [4, 10, 16]
    assert res.dtype == "float"


def test_groupnorm_divisibility_guard():
    """num_groups must divide num_channels — otherwise torch raises at build time,
    so the engine must refuse the shape AND surface a divisibility violation."""
    module = nn.GroupNorm(4, 16)
    _assert_parity("nn.GroupNorm", {"num_groups": 4, "num_channels": 16}, [2, 16, 8, 8], module)

    bad = _run_single("nn.GroupNorm", {"num_groups": 3, "num_channels": 16}, [2, 16, 8, 8])
    assert bad.output_shape is None
    assert bad.shape_error is not None
    assert any(v.get("error_type") == "param_divisibility" for v in bad.schema_violations)


# ── End-to-end: tiny CNN graph vs torch nn.Sequential ────────────────────────

def test_e2e_cnn_chain_matches_sequential():
    """A conv/pool/flatten/linear graph must propagate to the same final shape as
    the equivalent torch Sequential run forward — the whole point of the engine."""
    nodes = [
        _input_node(),
        _node("c1", "nn.Conv2d", params={"in_channels": 3, "out_channels": 8, "kernel_size": 3, "stride": 1, "padding": 1}),
        _node("p1", "nn.MaxPool2d", params={"kernel_size": 2, "stride": 2}),
        _node("c2", "nn.Conv2d", params={"in_channels": 8, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1}),
        _node("p2", "nn.MaxPool2d", params={"kernel_size": 2, "stride": 2}),
        _node("fl", "nn.Flatten", params={"start_dim": 1}),
        _node("fc", "nn.Linear", params={"in_features": 16 * 8 * 8, "out_features": 10}),
        _output_node(),
    ]
    edges = [
        _edge("input", "c1"), _edge("c1", "p1"), _edge("p1", "c2"), _edge("c2", "p2"),
        _edge("p2", "fl"), _edge("fl", "fc"), _edge("fc", "output"),
    ]
    res = infer(nodes, edges, {"input": [1, 3, 32, 32]})

    seq = nn.Sequential(
        nn.Conv2d(3, 8, 3, 1, 1), nn.MaxPool2d(2, 2),
        nn.Conv2d(8, 16, 3, 1, 1), nn.MaxPool2d(2, 2),
        nn.Flatten(1), nn.Linear(16 * 8 * 8, 10),
    )
    torch_shape = _torch_out_shape(seq, [1, 3, 32, 32])

    assert res["fc"].output_shape == torch_shape == [1, 10]
    assert res["output"].output_shape == torch_shape
    # Intermediate stages must also match the analytic expectation.
    assert res["p1"].output_shape == [1, 8, 16, 16]
    assert res["p2"].output_shape == [1, 16, 8, 8]
    assert res["fl"].output_shape == [1, 1024]


# ── Confirmed bugs (pinned green via xfail; assertions encode the CORRECT torch
#    behavior so each xpasses the day the bug is fixed) ─────────────────────────

def test_bug_pool_ceil_mode_ignored():
    # MaxPool2d(2,2,ceil_mode=True) on [1,4,7,7]: torch rounds up -> [1,4,4,4].
    r2 = _run_single("nn.MaxPool2d", {"kernel_size": 2, "stride": 2, "ceil_mode": True}, [1, 4, 7, 7])
    assert r2.output_shape == _torch_out_shape(nn.MaxPool2d(2, 2, ceil_mode=True), [1, 4, 7, 7]) == [1, 4, 4, 4]
    # MaxPool1d(3,2,ceil_mode=True) on [1,4,10]: torch -> [1,4,5].
    r1 = _run_single("nn.MaxPool1d", {"kernel_size": 3, "stride": 2, "ceil_mode": True}, [1, 4, 10])
    assert r1.output_shape == _torch_out_shape(nn.MaxPool1d(3, 2, ceil_mode=True), [1, 4, 10]) == [1, 4, 5]
    # AvgPool2d ceil_mode too.
    ra = _run_single("nn.AvgPool2d", {"kernel_size": 2, "stride": 2, "ceil_mode": True}, [1, 4, 7, 7])
    assert ra.output_shape == _torch_out_shape(nn.AvgPool2d(2, 2, ceil_mode=True), [1, 4, 7, 7]) == [1, 4, 4, 4]


def test_bug_pool_stride_none_false_error():
    # stride=None is a valid torch layer (defaults to kernel_size) and a common
    # serialization of an omitted stride.
    r = _run_single("nn.MaxPool2d", {"kernel_size": 2, "stride": None}, [1, 4, 8, 8])
    assert r.shape_error is None
    assert r.output_shape == _torch_out_shape(nn.MaxPool2d(2, stride=None), [1, 4, 8, 8]) == [1, 4, 4, 4]
    r3 = _run_single("nn.AvgPool3d", {"kernel_size": 2, "stride": None}, [1, 4, 8, 8, 8])
    assert r3.output_shape == _torch_out_shape(nn.AvgPool3d(2, stride=None), [1, 4, 8, 8, 8]) == [1, 4, 4, 4, 4]


def test_bug_conv_valid_nonuniform_dilation_square():
    # Square input hides the bug: both spatial dims resolve to dilation[0].
    params = {"in_channels": 3, "out_channels": 8, "kernel_size": 3, "padding": "valid", "dilation": [1, 2]}
    res = _run_single("nn.Conv2d", params, [1, 3, 12, 12])
    module = nn.Conv2d(3, 8, 3, padding="valid", dilation=(1, 2))
    assert res.output_shape == _torch_out_shape(module, [1, 3, 12, 12]) == [1, 8, 10, 8]


def test_bug_conv_same_strided_should_be_invalid():
    # torch raises ValueError for strided 'same', so the analytic contract should
    # flag the config (output None / shape_error) rather than fabricate a shape.
    res = _run_single("nn.Conv2d",
                      {"in_channels": 3, "out_channels": 8, "kernel_size": 3, "stride": 2, "padding": "same"},
                      [1, 3, 16, 16])
    with pytest.raises(ValueError):
        nn.Conv2d(3, 8, 3, stride=2, padding="same")
    assert res.output_shape is None
    assert res.shape_error is not None


def test_bug_argmax_dtype_should_be_int():
    res = _run_single("torch.argmax", {"dim": 1}, [2, 4, 8])
    x = torch.randn(2, 4, 8)
    idx = torch.argmax(x, dim=1)
    # Shape is already correct; the dtype is the regression.
    assert res.output_shape == list(idx.shape) == [2, 8]
    assert idx.dtype == torch.int64
    assert res.dtype == "int"
