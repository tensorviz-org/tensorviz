"""Tests for core/shape_engine.py — topological shape propagation engine."""



import pytest
from core.shape_engine import infer, NodeShapeResult
from models.architecture_models import ArchitectureNode, ArchitectureEdge


# ── Helpers ──────────────────────────────────────────────────────────────────

def _node(node_id, op="nn.ReLU", kind="layer", params=None):
    return ArchitectureNode(node_id=node_id, op=op, kind=kind, params=params or {})


def _input_node(node_id="input"):
    return ArchitectureNode(node_id=node_id, op="input", kind="input", params={})


def _output_node(node_id="output"):
    return ArchitectureNode(node_id=node_id, op="output", kind="output", params={})


def _edge(src, tgt, edge_id=None):
    return ArchitectureEdge(
        edge_id=edge_id or f"{src}_{tgt}",
        source_node_id=src,
        source_port="y",
        target_node_id=tgt,
        target_port="x",
    )


def _linear_node(node_id, in_features=32, out_features=16):
    return _node(node_id, op="nn.Linear", params={"in_features": in_features, "out_features": out_features})


# ── Basic topology ────────────────────────────────────────────────────────────

def test_single_relu_passthrough():
    """ReLU should forward input shape unchanged."""
    nodes = [_input_node(), _node("relu", "nn.ReLU"), _output_node()]
    edges = [_edge("input", "relu"), _edge("relu", "output")]
    initial = {"input": [1, 32]}
    res = infer(nodes, edges, initial)

    assert res["relu"].output_shape == [1, 32]
    assert res["relu"].cycle_member is False
    assert res["relu"].needs_fallback is False


def test_linear_shape():
    """Linear should produce [batch, out_features]."""
    nodes = [_input_node(), _linear_node("fc", 32, 16), _output_node()]
    edges = [_edge("input", "fc"), _edge("fc", "output")]
    initial = {"input": [4, 32]}
    res = infer(nodes, edges, initial)

    assert res["fc"].output_shape == [4, 16]


def test_output_node_receives_parent_shape():
    """Output node should have output_shape == its parent's shape."""
    nodes = [_input_node(), _output_node()]
    edges = [_edge("input", "output")]
    initial = {"input": [1, 64]}
    res = infer(nodes, edges, initial)

    assert res["output"].output_shape == [1, 64]


def test_chain_shapes_propagate():
    """Shapes flow through a linear chain of ops."""
    nodes = [
        _input_node(),
        _linear_node("fc1", 32, 64),
        _node("relu", "nn.ReLU"),
        _linear_node("fc2", 64, 10),
        _output_node(),
    ]
    edges = [
        _edge("input", "fc1"),
        _edge("fc1", "relu"),
        _edge("relu", "fc2"),
        _edge("fc2", "output"),
    ]
    initial = {"input": [8, 32]}
    res = infer(nodes, edges, initial)

    assert res["fc1"].output_shape == [8, 64]
    assert res["relu"].output_shape == [8, 64]
    assert res["fc2"].output_shape == [8, 10]


def test_no_parents_yields_empty_result():
    """A node with no incoming edges has no input_shapes."""
    nodes = [_node("orphan", "nn.ReLU")]
    edges = []
    initial = {}
    res = infer(nodes, edges, initial)

    assert res["orphan"].output_shape is None
    assert res["orphan"].input_shapes == []


# ── Cycle detection ───────────────────────────────────────────────────────────

def test_cycle_detection_two_nodes():
    """Two nodes forming a cycle should both be marked cycle_member=True."""
    nodes = [_node("a"), _node("b")]
    edges = [_edge("a", "b"), _edge("b", "a")]
    initial = {}
    res = infer(nodes, edges, initial)

    assert res["a"].cycle_member is True
    assert res["b"].cycle_member is True


def test_cycle_detection_three_node_cycle():
    """Three-node cycle — all three should be marked."""
    nodes = [_input_node(), _node("a"), _node("b"), _node("c")]
    edges = [
        _edge("input", "a"),
        _edge("a", "b"),
        _edge("b", "c"),
        _edge("c", "b"),  # b←c creates the cycle b→c→b
    ]
    initial = {"input": [1, 32]}
    res = infer(nodes, edges, initial)

    assert res["b"].cycle_member is True
    assert res["c"].cycle_member is True
    # 'a' is not in the cycle
    assert not res["a"].cycle_member


# ── Arity detection ───────────────────────────────────────────────────────────

def test_arity_warning_single_op_two_inputs():
    """A single-input op receiving two parents should set arity_warning."""
    nodes = [
        _input_node("inp1"),
        _input_node("inp2"),
        _node("relu", "nn.ReLU"),
        _output_node(),
    ]
    edges = [_edge("inp1", "relu"), _edge("inp2", "relu"), _edge("relu", "output")]
    initial = {"inp1": [1, 32], "inp2": [1, 32]}
    res = infer(nodes, edges, initial)

    assert res["relu"].arity_warning is not None
    # Shape is still propagated (from first parent)
    assert res["relu"].output_shape == [1, 32]


def test_no_arity_warning_merge_op():
    """Merge ops (add) with two inputs should NOT warn."""
    nodes = [
        _input_node("inp1"),
        _input_node("inp2"),
        _node("add", "torch.add", params={}),
        _output_node(),
    ]
    edges = [_edge("inp1", "add"), _edge("inp2", "add"), _edge("add", "output")]
    initial = {"inp1": [1, 32], "inp2": [1, 32]}
    res = infer(nodes, edges, initial)

    assert res["add"].arity_warning is None


# ── dtype propagation ─────────────────────────────────────────────────────────

def test_embedding_output_dtype_stays_float():
    """Embedding output is a float tensor (embedding_dim floats per token)."""
    nodes = [
        _input_node(),
        _node("emb", "nn.Embedding", params={"num_embeddings": 100, "embedding_dim": 16}),
        _output_node(),
    ]
    edges = [_edge("input", "emb"), _edge("emb", "output")]
    initial = {"input": [4, 10]}
    res = infer(nodes, edges, initial)

    assert res["emb"].output_shape == [4, 10, 16]
    # Dtype propagation — embedding output is float (float lookup table)
    assert res["emb"].dtype in ("float",)


def test_relu_preserves_float_dtype():
    nodes = [_input_node(), _node("relu", "nn.ReLU"), _output_node()]
    edges = [_edge("input", "relu"), _edge("relu", "output")]
    initial = {"input": [1, 64]}
    res = infer(nodes, edges, initial)
    assert res["relu"].dtype == "float"


# ── Passthrough ops ───────────────────────────────────────────────────────────

def test_passthrough_op_forwards_shape():
    """An op in the passthrough set should forward input shape even without a spec."""
    nodes = [_input_node(), _node("custom", "se_block"), _output_node()]
    edges = [_edge("input", "custom"), _edge("custom", "output")]
    initial = {"input": [1, 64, 28, 28]}
    res = infer(
        nodes,
        edges,
        initial,
        passthrough_ops=frozenset({"se_block"}),
    )

    assert res["custom"].output_shape == [1, 64, 28, 28]
    assert res["custom"].used_passthrough is True


def test_unknown_op_without_passthrough_sets_needs_fallback():
    """An op not in registry and not in passthrough should set needs_fallback."""
    nodes = [_input_node(), _node("mystery", "totally_unknown_op"), _output_node()]
    edges = [_edge("input", "mystery"), _edge("mystery", "output")]
    initial = {"input": [1, 32]}
    res = infer(nodes, edges, initial)

    assert res["mystery"].needs_fallback is True
    assert res["mystery"].output_shape is None


# ── Runtime fallback ──────────────────────────────────────────────────────────

def test_runtime_fallback_is_called_and_shape_propagates():
    """runtime_fallback_fn result should be used for downstream nodes."""
    fallback_called = []

    def _fallback(node, parent_shapes):
        fallback_called.append(node.node_id)
        return [p + 1 for p in parent_shapes[0]]  # e.g. [1,32] → [2,33]

    nodes = [
        _input_node(),
        _node("custom", "custom_op"),
        _linear_node("fc", in_features=33, out_features=10),
        _output_node(),
    ]
    edges = [
        _edge("input", "custom"),
        _edge("custom", "fc"),
        _edge("fc", "output"),
    ]
    initial = {"input": [1, 32]}
    res = infer(nodes, edges, initial, runtime_fallback_fn=_fallback)

    assert "custom" in fallback_called
    assert res["custom"].output_shape == [2, 33]
    assert res["fc"].output_shape == [2, 10]


# ── Conv2d shape ──────────────────────────────────────────────────────────────

def test_conv2d_shape():
    """Conv2d with known params should produce correct output shape."""
    nodes = [
        _input_node(),
        _node("conv", "nn.Conv2d", params={
            "in_channels": 3, "out_channels": 32,
            "kernel_size": 3, "stride": 1, "padding": 1,
        }),
        _output_node(),
    ]
    edges = [_edge("input", "conv"), _edge("conv", "output")]
    initial = {"input": [1, 3, 64, 64]}
    res = infer(nodes, edges, initial)

    assert res["conv"].output_shape == [1, 32, 64, 64]


def test_conv2d_wrong_rank_returns_none():
    """Conv2d with rank-2 input should produce None output (shape_fn fails)."""
    nodes = [
        _input_node(),
        _node("conv", "nn.Conv2d", params={
            "in_channels": 3, "out_channels": 32, "kernel_size": 3,
        }),
        _output_node(),
    ]
    edges = [_edge("input", "conv"), _edge("conv", "output")]
    initial = {"input": [1, 32]}
    res = infer(nodes, edges, initial)

    assert res["conv"].output_shape is None


# ── Flatten / Linear chain (common pattern) ───────────────────────────────────

def test_flatten_then_linear():
    nodes = [
        _input_node(),
        _node("flat", "nn.Flatten", params={"start_dim": 1}),
        _linear_node("fc", in_features=3 * 8 * 8, out_features=10),
        _output_node(),
    ]
    edges = [_edge("input", "flat"), _edge("flat", "fc"), _edge("fc", "output")]
    initial = {"input": [2, 3, 8, 8]}
    res = infer(nodes, edges, initial)

    assert res["flat"].output_shape == [2, 192]
    assert res["fc"].output_shape == [2, 10]


# ── Initial shapes not in result dict ────────────────────────────────────────

def test_input_nodes_not_in_result():
    """Input nodes are skipped — they provide shapes but shouldn't be in results."""
    nodes = [_input_node(), _node("relu", "nn.ReLU"), _output_node()]
    edges = [_edge("input", "relu"), _edge("relu", "output")]
    initial = {"input": [1, 8]}
    res = infer(nodes, edges, initial)

    assert "input" not in res
    assert "relu" in res
    assert "output" in res
