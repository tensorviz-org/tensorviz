"""Tests for Step 3: lazy parameter resolution in core/shape_engine.py.

When a node's defining_params are missing/None, shape_engine should infer
them from the resolved input shape, fill them in before calling shape_fn,
and emit set_param repair suggestions.
"""



from core.shape_engine import infer
from models.architecture_models import ArchitectureNode, ArchitectureEdge


# ── Helpers ──────────────────────────────────────────────────────────────────

def _node(node_id, op, params=None, kind="layer"):
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


# ── Linear with missing in_features ──────────────────────────────────────────

def test_linear_missing_in_features_inferred():
    """Linear with only out_features should infer in_features from input shape."""
    nodes = [
        _input_node(),
        _node("fc", "nn.Linear", params={"out_features": 16}),
        _output_node(),
    ]
    edges = [_edge("input", "fc"), _edge("fc", "output")]
    initial = {"input": [4, 32]}
    res = infer(nodes, edges, initial)

    assert res["fc"].output_shape == [4, 16], "Linear should infer in_features=32"
    assert "in_features" in res["fc"].infilled_params
    assert res["fc"].infilled_params["in_features"] == 32


def test_linear_missing_in_features_emits_repair():
    """A set_param repair for in_features should be emitted."""
    nodes = [
        _input_node(),
        _node("fc", "nn.Linear", params={"out_features": 16}),
        _output_node(),
    ]
    edges = [_edge("input", "fc"), _edge("fc", "output")]
    initial = {"input": [4, 32]}
    res = infer(nodes, edges, initial)

    repairs = res["fc"].set_param_repairs
    assert any(r["param"] == "in_features" and r["value"] == 32 for r in repairs)
    assert all(r["op"] == "set_param" for r in repairs)


def test_linear_with_in_features_no_repair():
    """When in_features is already set, no repair should be emitted."""
    nodes = [
        _input_node(),
        _node("fc", "nn.Linear", params={"in_features": 32, "out_features": 16}),
        _output_node(),
    ]
    edges = [_edge("input", "fc"), _edge("fc", "output")]
    initial = {"input": [4, 32]}
    res = infer(nodes, edges, initial)

    assert res["fc"].output_shape == [4, 16]
    assert res["fc"].infilled_params == {}
    assert res["fc"].set_param_repairs == []


# ── Conv2d with missing in_channels ──────────────────────────────────────────

def test_conv2d_missing_in_channels_inferred():
    """Conv2d with only out_channels should infer in_channels from input[1]."""
    nodes = [
        _input_node(),
        _node("conv", "nn.Conv2d", params={"out_channels": 64, "kernel_size": 3}),
        _output_node(),
    ]
    edges = [_edge("input", "conv"), _edge("conv", "output")]
    initial = {"input": [1, 3, 32, 32]}
    res = infer(nodes, edges, initial)

    assert res["conv"].output_shape is not None, "Conv2d should resolve with inferred in_channels"
    assert res["conv"].infilled_params.get("in_channels") == 3


# ── LayerNorm with missing normalized_shape ───────────────────────────────────

def test_layernorm_missing_normalized_shape_inferred():
    """LayerNorm with no normalized_shape should infer it as [last_dim]."""
    nodes = [
        _input_node(),
        _node("ln", "nn.LayerNorm", params={}),
        _output_node(),
    ]
    edges = [_edge("input", "ln"), _edge("ln", "output")]
    initial = {"input": [2, 10, 128]}
    res = infer(nodes, edges, initial)

    assert res["ln"].output_shape == [2, 10, 128]
    assert res["ln"].infilled_params.get("normalized_shape") == [128]


def test_rmsnorm_missing_normalized_shape_inferred():
    """RMSNorm (nn.RMSNorm) with no normalized_shape should infer [last_dim]."""
    nodes = [
        _input_node(),
        _node("rms", "nn.RMSNorm", params={}),
        _output_node(),
    ]
    edges = [_edge("input", "rms"), _edge("rms", "output")]
    initial = {"input": [2, 10, 64]}
    res = infer(nodes, edges, initial)

    assert res["rms"].output_shape == [2, 10, 64]
    assert res["rms"].infilled_params.get("normalized_shape") == [64]


# ── BatchNorm with missing num_features ───────────────────────────────────────

def test_batchnorm1d_missing_num_features_inferred():
    """BatchNorm1d with no num_features should infer from input[1]."""
    nodes = [
        _input_node(),
        _node("bn", "nn.BatchNorm1d", params={}),
        _output_node(),
    ]
    edges = [_edge("input", "bn"), _edge("bn", "output")]
    initial = {"input": [8, 64]}
    res = infer(nodes, edges, initial)

    assert res["bn"].output_shape == [8, 64]
    assert res["bn"].infilled_params.get("num_features") == 64


# ── RNN / LSTM with missing input_size ───────────────────────────────────────

def test_lstm_missing_input_size_inferred():
    """LSTM with no input_size should infer from last dim of input."""
    nodes = [
        _input_node(),
        _node("lstm", "nn.LSTM", params={"hidden_size": 128}),
        _output_node(),
    ]
    edges = [_edge("input", "lstm"), _edge("lstm", "output")]
    initial = {"input": [1, 10, 32]}
    res = infer(nodes, edges, initial)

    assert res["lstm"].infilled_params.get("input_size") == 32
    # Shape should resolve: [batch, seq, hidden_size]
    assert res["lstm"].output_shape == [1, 10, 128]


def test_linear_with_wrong_in_features_still_errors():
    """When in_features is explicitly set to wrong value and no bridge is inserted,
    shape inference should fail (shape_fn returns None for mismatched in_features)."""
    nodes = [
        _input_node(),
        _node("fc", "nn.Linear", params={"in_features": 999, "out_features": 16}),
        _output_node(),
    ]
    edges = [_edge("input", "fc"), _edge("fc", "output")]
    initial = {"input": [4, 32]}
    res = infer(nodes, edges, initial)

    # shape_fn returns None because inp[-1]=32 != in_features=999
    assert res["fc"].output_shape is None, (
        "Linear with explicit wrong in_features should fail shape inference"
    )
    assert res["fc"].infilled_params == {}, "No lazy fill should occur when param is already set"
