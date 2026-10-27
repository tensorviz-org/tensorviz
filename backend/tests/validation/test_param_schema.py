"""Tests for Step 4: param-range and semantic checks."""


from models.architecture_models import ArchitectureEdge, ArchitectureGraph, ArchitectureNode
from services.canonical_validation import CanonicalValidationService


def _build_graph(nodes, edges):
    return ArchitectureGraph(
        graph_id="g1",
        name="test",
        revision=1,
        inputs=[{"name": "x", "shape": [1, 3, 32, 32]}],
        nodes=nodes,
        edges=edges,
    )


def _inp_layer_out(op, params, input_shape):
    nodes = [
        ArchitectureNode(node_id="input", kind="input", op="Input",
                         params={"shape": list(input_shape)}, in_ports=[], out_ports=["y"]),
        ArchitectureNode(node_id="layer", kind="layer", op=op, params=dict(params)),
        ArchitectureNode(node_id="output", kind="output", op="Output",
                         params={}, in_ports=["x"], out_ports=[]),
    ]
    edges = [
        ArchitectureEdge(edge_id="e1", source_node_id="input", source_port="y",
                         target_node_id="layer", target_port="x"),
        ArchitectureEdge(edge_id="e2", source_node_id="layer", source_port="y",
                         target_node_id="output", target_port="x"),
    ]
    return nodes, edges


def validate(op, params, input_shape=(1, 32, 8, 8), strict=False):
    nodes, edges = _inp_layer_out(op, params, input_shape)
    return CanonicalValidationService().validate(_build_graph(nodes, edges), strict=strict)


def test_dropout_valid_p():
    assert validate("nn.Dropout", {"p": 0.5}, input_shape=(1, 32)).valid


def test_dropout_invalid_p_high():
    result = validate("nn.Dropout", {"p": 2.0}, input_shape=(1, 32))
    assert not result.valid
    nr = result.node_results["layer"]
    assert nr.suggested_repairs, "Expected repair for p=2.0"


def test_dropout_invalid_p_negative():
    assert not validate("nn.Dropout", {"p": -0.1}, input_shape=(1, 32)).valid


def test_dropout_boundary_p_zero():
    assert validate("nn.Dropout", {"p": 0.0}, input_shape=(1, 32)).valid


def test_dropout_boundary_p_one():
    assert validate("nn.Dropout", {"p": 1.0}, input_shape=(1, 32)).valid


def test_conv2d_valid_groups():
    result = validate("nn.Conv2d",
                      {"in_channels": 32, "out_channels": 32, "kernel_size": 3, "groups": 8},
                      input_shape=(1, 32, 8, 8))
    assert result.valid, result


def test_conv2d_invalid_groups():
    result = validate("nn.Conv2d",
                      {"in_channels": 32, "out_channels": 32, "kernel_size": 3, "groups": 5},
                      input_shape=(1, 32, 8, 8))
    assert not result.valid
    nr = result.node_results["layer"]
    assert nr.suggested_repairs, "Expected repair for groups=5"


def test_conv2d_depthwise_valid():
    result = validate("nn.Conv2d",
                      {"in_channels": 16, "out_channels": 16, "kernel_size": 3, "groups": 16},
                      input_shape=(1, 16, 8, 8))
    assert result.valid, result


def test_groupnorm_valid():
    result = validate("nn.GroupNorm", {"num_groups": 8, "num_channels": 32},
                      input_shape=(1, 32, 8, 8))
    assert result.valid, result


def test_groupnorm_invalid_groups():
    result = validate("nn.GroupNorm", {"num_groups": 7, "num_channels": 32},
                      input_shape=(1, 32, 8, 8))
    assert not result.valid
    nr = result.node_results["layer"]
    assert nr.suggested_repairs, "Expected repair for num_groups=7"


def test_mha_valid_nhead():
    result = validate("nn.MultiheadAttention", {"embed_dim": 64, "num_heads": 8},
                      input_shape=(10, 1, 64))
    assert result.valid, result


def test_mha_invalid_nhead():
    result = validate("nn.MultiheadAttention", {"embed_dim": 64, "num_heads": 7},
                      input_shape=(10, 1, 64))
    assert not result.valid
    nr = result.node_results["layer"]
    assert nr.suggested_repairs, "Expected repair for num_heads=7"


def test_conv2d_padding_same_preserves_spatial():
    result = validate("nn.Conv2d",
                      {"in_channels": 32, "out_channels": 16, "kernel_size": 3,
                       "padding": "same", "stride": 1},
                      input_shape=(1, 32, 8, 8))
    assert result.valid, result
    out_shape = result.node_results["output"].output_shape
    assert out_shape is not None
    assert out_shape[2] == 8, f"Expected H=8 with padding='same', got {out_shape[2]}"
    assert out_shape[3] == 8, f"Expected W=8 with padding='same', got {out_shape[3]}"


def test_conv2d_padding_valid_reduces_spatial():
    result = validate("nn.Conv2d",
                      {"in_channels": 32, "out_channels": 16, "kernel_size": 3,
                       "padding": "valid", "stride": 1},
                      input_shape=(1, 32, 8, 8))
    assert result.valid, result
    out_shape = result.node_results["output"].output_shape
    assert out_shape is not None
    assert out_shape[2] == 6, f"Expected H=6 with padding='valid', got {out_shape[2]}"


def test_duplicate_edge_detected():
    nodes = [
        ArchitectureNode(node_id="input", kind="input", op="Input",
                         params={"shape": [1, 32]}, in_ports=[], out_ports=["y"]),
        ArchitectureNode(node_id="lin", kind="layer", op="nn.Linear",
                         params={"in_features": 32, "out_features": 16}),
        ArchitectureNode(node_id="output", kind="output", op="Output",
                         params={}, in_ports=["x"], out_ports=[]),
    ]
    edges = [
        ArchitectureEdge(edge_id="e1", source_node_id="input", source_port="y",
                         target_node_id="lin", target_port="x"),
        ArchitectureEdge(edge_id="e2", source_node_id="input", source_port="y",
                         target_node_id="lin", target_port="x"),
        ArchitectureEdge(edge_id="e3", source_node_id="lin", source_port="y",
                         target_node_id="output", target_port="x"),
    ]
    result = CanonicalValidationService().validate(_build_graph(nodes, edges), strict=False)
    assert not result.valid, "Duplicate edge should make graph invalid"


def test_self_loop_detected():
    nodes = [
        ArchitectureNode(node_id="input", kind="input", op="Input",
                         params={"shape": [1, 32]}, in_ports=[], out_ports=["y"]),
        ArchitectureNode(node_id="lin", kind="layer", op="nn.Linear",
                         params={"in_features": 32, "out_features": 32}),
        ArchitectureNode(node_id="output", kind="output", op="Output",
                         params={}, in_ports=["x"], out_ports=[]),
    ]
    edges = [
        ArchitectureEdge(edge_id="e1", source_node_id="input", source_port="y",
                         target_node_id="lin", target_port="x"),
        ArchitectureEdge(edge_id="e2", source_node_id="lin", source_port="y",
                         target_node_id="lin", target_port="x"),
        ArchitectureEdge(edge_id="e3", source_node_id="lin", source_port="y",
                         target_node_id="output", target_port="x"),
    ]
    result = CanonicalValidationService().validate(_build_graph(nodes, edges), strict=False)
    assert not result.valid, "Self-loop should make graph invalid"


def test_valid_conv_strict_ok():
    result = validate("nn.Conv2d",
                      {"in_channels": 32, "out_channels": 64, "kernel_size": 3, "groups": 1},
                      input_shape=(1, 32, 8, 8), strict=True)
    assert result.valid, result
