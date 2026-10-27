"""Tests for canonical graph validation rules."""



from models.architecture_models import (  # noqa: E402
    ArchitectureEdge,
    ArchitectureGraph,
    ArchitectureNode,
)
from services.canonical_validation import CanonicalValidationService  # noqa: E402


def _build_graph(nodes, edges):
    return ArchitectureGraph(
        graph_id="g1",
        name="test_graph",
        revision=1,
        inputs=[{"name": "x", "shape": [1, 3, 32, 32]}],
        nodes=nodes,
        edges=edges,
    )


def test_validation_passes_for_simple_conv_pipeline():
    nodes = [
        ArchitectureNode(node_id="input", kind="input", op="Input", params={"shape": [1, 3, 32, 32]}, in_ports=[], out_ports=["y"]),
        ArchitectureNode(node_id="conv1", kind="layer", op="nn.Conv2d", params={"in_channels": 3, "out_channels": 8, "kernel_size": 3}),
        ArchitectureNode(node_id="pool1", kind="layer", op="nn.MaxPool2d", params={"kernel_size": 2}),
        ArchitectureNode(node_id="flatten", kind="layer", op="nn.Flatten", params={}),
        ArchitectureNode(node_id="fc1", kind="layer", op="nn.Linear", params={"in_features": 1800, "out_features": 10}),
        ArchitectureNode(node_id="output", kind="output", op="Output", params={}, in_ports=["x"], out_ports=[]),
    ]
    edges = [
        ArchitectureEdge(edge_id="e1", source_node_id="input", source_port="y", target_node_id="conv1", target_port="x"),
        ArchitectureEdge(edge_id="e2", source_node_id="conv1", source_port="y", target_node_id="pool1", target_port="x"),
        ArchitectureEdge(edge_id="e3", source_node_id="pool1", source_port="y", target_node_id="flatten", target_port="x"),
        ArchitectureEdge(edge_id="e4", source_node_id="flatten", source_port="y", target_node_id="fc1", target_port="x"),
        ArchitectureEdge(edge_id="e5", source_node_id="fc1", source_port="y", target_node_id="output", target_port="x"),
    ]

    result = CanonicalValidationService().validate(_build_graph(nodes, edges))
    assert result.valid is True
    assert result.node_results["fc1"].output_shape == [1, 10]


def test_validation_reports_linear_mismatch():
    nodes = [
        ArchitectureNode(node_id="input", kind="input", op="Input", params={"shape": [1, 8]}, in_ports=[], out_ports=["y"]),
        ArchitectureNode(node_id="fc1", kind="layer", op="nn.Linear", params={"in_features": 16, "out_features": 4}),
        ArchitectureNode(node_id="output", kind="output", op="Output", params={}, in_ports=["x"], out_ports=[]),
    ]
    edges = [
        ArchitectureEdge(edge_id="e1", source_node_id="input", source_port="y", target_node_id="fc1", target_port="x"),
        ArchitectureEdge(edge_id="e2", source_node_id="fc1", source_port="y", target_node_id="output", target_port="x"),
    ]

    result = CanonicalValidationService().validate(_build_graph(nodes, edges))
    assert result.valid is False
    assert result.node_results["fc1"].status == "error"
    assert "in_features mismatch" in (result.node_results["fc1"].message or "")


def test_validation_reports_add_shape_mismatch():
    nodes = [
        ArchitectureNode(node_id="input", kind="input", op="Input", params={"shape": [1, 8]}, in_ports=[], out_ports=["y"]),
        ArchitectureNode(node_id="a", kind="layer", op="nn.Linear", params={"in_features": 8, "out_features": 8}),
        ArchitectureNode(node_id="b", kind="layer", op="nn.Linear", params={"in_features": 8, "out_features": 16}),
        ArchitectureNode(node_id="add1", kind="op", op="torch.add", params={}),
        ArchitectureNode(node_id="output", kind="output", op="Output", params={}, in_ports=["x"], out_ports=[]),
    ]
    edges = [
        ArchitectureEdge(edge_id="e1", source_node_id="input", source_port="y", target_node_id="a", target_port="x"),
        ArchitectureEdge(edge_id="e2", source_node_id="input", source_port="y", target_node_id="b", target_port="x"),
        ArchitectureEdge(edge_id="e3", source_node_id="a", source_port="y", target_node_id="add1", target_port="x"),
        ArchitectureEdge(edge_id="e4", source_node_id="b", source_port="y", target_node_id="add1", target_port="x"),
        ArchitectureEdge(edge_id="e5", source_node_id="add1", source_port="y", target_node_id="output", target_port="x"),
    ]

    result = CanonicalValidationService().validate(_build_graph(nodes, edges))
    assert result.valid is False
    assert result.node_results["add1"].status == "error"


def test_validation_reports_unconnected_layer_node():
    nodes = [
        ArchitectureNode(node_id="input", kind="input", op="Input", params={"shape": [1, 3, 32, 32]}, in_ports=[], out_ports=["y"]),
        ArchitectureNode(node_id="conv1", kind="layer", op="nn.Conv2d", params={"in_channels": 3, "out_channels": 16, "kernel_size": 3, "padding": 1}),
        ArchitectureNode(node_id="output", kind="output", op="Output", params={}, in_ports=["x"], out_ports=[]),
    ]
    edges = []

    result = CanonicalValidationService().validate(_build_graph(nodes, edges))
    assert result.valid is False
    assert result.node_results["conv1"].status == "error"
    assert "no input connections" in (result.node_results["conv1"].message or "")
    assert result.node_results["output"].status == "error"


def test_validation_reports_unreachable_output():
    nodes = [
        ArchitectureNode(node_id="input", kind="input", op="Input", params={"shape": [1, 8]}, in_ports=[], out_ports=["y"]),
        ArchitectureNode(node_id="fc1", kind="layer", op="nn.Linear", params={"in_features": 8, "out_features": 4}),
        ArchitectureNode(node_id="output", kind="output", op="Output", params={}, in_ports=["x"], out_ports=[]),
    ]
    edges = [
        ArchitectureEdge(edge_id="e1", source_node_id="input", source_port="y", target_node_id="fc1", target_port="x"),
    ]

    result = CanonicalValidationService().validate(_build_graph(nodes, edges))
    assert result.valid is False
    assert result.node_results["output"].status == "error"
    assert any("Unreachable output nodes" in err for err in result.global_errors)


def test_validation_reports_rnn_rank_mismatch():
    nodes = [
        ArchitectureNode(node_id="input", kind="input", op="Input", params={"shape": [1, 3, 32, 32]}, in_ports=[], out_ports=["y"]),
        ArchitectureNode(node_id="lstm1", kind="layer", op="nn.LSTM", params={"input_size": 32, "hidden_size": 64, "batch_first": True}),
        ArchitectureNode(node_id="output", kind="output", op="Output", params={}, in_ports=["x"], out_ports=[]),
    ]
    edges = [
        ArchitectureEdge(edge_id="e1", source_node_id="input", source_port="y", target_node_id="lstm1", target_port="x"),
        ArchitectureEdge(edge_id="e2", source_node_id="lstm1", source_port="y", target_node_id="output", target_port="x"),
    ]

    result = CanonicalValidationService().validate(_build_graph(nodes, edges), strict=True)
    assert result.valid is False
    assert result.node_results["lstm1"].status == "error"
    assert "RNN-family" in (result.node_results["lstm1"].message or "")


def test_validation_allows_rank2_lstm_input_in_strict_mode():
    nodes = [
        ArchitectureNode(node_id="input", kind="input", op="Input", params={"shape": [1, 512]}, in_ports=[], out_ports=["y"]),
        ArchitectureNode(node_id="lstm1", kind="layer", op="nn.LSTM", params={"input_size": 512, "hidden_size": 128, "batch_first": True}),
        ArchitectureNode(node_id="output", kind="output", op="Output", params={}, in_ports=["x"], out_ports=[]),
    ]
    edges = [
        ArchitectureEdge(edge_id="e1", source_node_id="input", source_port="y", target_node_id="lstm1", target_port="x"),
        ArchitectureEdge(edge_id="e2", source_node_id="lstm1", source_port="y", target_node_id="output", target_port="x"),
    ]

    result = CanonicalValidationService().validate(_build_graph(nodes, edges), strict=True)
    assert result.valid is True
    assert result.node_results["lstm1"].status == "ok"
    assert result.node_results["lstm1"].output_shape == [1, 1, 128]


def test_validation_reports_multihead_attention_embed_mismatch():
    nodes = [
        ArchitectureNode(node_id="input", kind="input", op="Input", params={"shape": [4, 16, 256]}, in_ports=[], out_ports=["y"]),
        ArchitectureNode(node_id="attn1", kind="layer", op="nn.MultiheadAttention", params={"embed_dim": 512, "num_heads": 8}),
        ArchitectureNode(node_id="output", kind="output", op="Output", params={}, in_ports=["x"], out_ports=[]),
    ]
    edges = [
        ArchitectureEdge(edge_id="e1", source_node_id="input", source_port="y", target_node_id="attn1", target_port="x"),
        ArchitectureEdge(edge_id="e2", source_node_id="attn1", source_port="y", target_node_id="output", target_port="x"),
    ]

    result = CanonicalValidationService().validate(_build_graph(nodes, edges), strict=True)
    assert result.valid is False
    assert result.node_results["attn1"].status == "error"
    assert "MultiheadAttention" in (result.node_results["attn1"].message or "")


def test_validation_allows_rank2_multihead_attention_input():
    nodes = [
        ArchitectureNode(node_id="input", kind="input", op="Input", params={"shape": [4, 256]}, in_ports=[], out_ports=["y"]),
        ArchitectureNode(node_id="attn1", kind="layer", op="nn.MultiheadAttention", params={"embed_dim": 256, "num_heads": 8}),
        ArchitectureNode(node_id="output", kind="output", op="Output", params={}, in_ports=["x"], out_ports=[]),
    ]
    edges = [
        ArchitectureEdge(edge_id="e1", source_node_id="input", source_port="y", target_node_id="attn1", target_port="x"),
        ArchitectureEdge(edge_id="e2", source_node_id="attn1", source_port="y", target_node_id="output", target_port="x"),
    ]

    result = CanonicalValidationService().validate(_build_graph(nodes, edges), strict=True)
    assert result.valid is True
    assert result.node_results["attn1"].status == "ok"
    assert result.node_results["attn1"].output_shape == [4, 256]


def test_validation_reports_attention_branch_count_semantic_error():
    nodes = [
        ArchitectureNode(node_id="input", kind="input", op="Input", params={"shape": [2, 12, 128]}, in_ports=[], out_ports=["y"]),
        ArchitectureNode(node_id="q1", kind="layer", op="nn.Linear", params={"in_features": 128, "out_features": 128}),
        ArchitectureNode(node_id="k1", kind="layer", op="nn.Linear", params={"in_features": 128, "out_features": 128}),
        ArchitectureNode(node_id="attn1", kind="layer", op="nn.MultiheadAttention", params={"embed_dim": 128, "num_heads": 8}),
        ArchitectureNode(node_id="output", kind="output", op="Output", params={}, in_ports=["x"], out_ports=[]),
    ]
    edges = [
        ArchitectureEdge(edge_id="e1", source_node_id="input", source_port="y", target_node_id="q1", target_port="x"),
        ArchitectureEdge(edge_id="e2", source_node_id="input", source_port="y", target_node_id="k1", target_port="x"),
        ArchitectureEdge(edge_id="e3", source_node_id="q1", source_port="y", target_node_id="attn1", target_port="x"),
        ArchitectureEdge(edge_id="e4", source_node_id="k1", source_port="y", target_node_id="attn1", target_port="x"),
        ArchitectureEdge(edge_id="e5", source_node_id="attn1", source_port="y", target_node_id="output", target_port="x"),
    ]
    result = CanonicalValidationService().validate(_build_graph(nodes, edges))
    assert result.valid is False
    assert result.node_results["attn1"].status == "error"
    assert "either 1 input" in (result.node_results["attn1"].message or "")


def test_validation_enriches_merge_errors_with_branch_lineage():
    nodes = [
        ArchitectureNode(node_id="input", kind="input", op="Input", params={"shape": [1, 8]}, in_ports=[], out_ports=["y"]),
        ArchitectureNode(node_id="left", kind="layer", op="nn.Linear", params={"in_features": 8, "out_features": 8}),
        ArchitectureNode(node_id="right", kind="layer", op="nn.Linear", params={"in_features": 8, "out_features": 16}),
        ArchitectureNode(node_id="add1", kind="op", op="torch.add", params={}),
        ArchitectureNode(node_id="output", kind="output", op="Output", params={}, in_ports=["x"], out_ports=[]),
    ]
    edges = [
        ArchitectureEdge(edge_id="e1", source_node_id="input", source_port="y", target_node_id="left", target_port="x"),
        ArchitectureEdge(edge_id="e2", source_node_id="input", source_port="y", target_node_id="right", target_port="x"),
        ArchitectureEdge(edge_id="e3", source_node_id="left", source_port="y", target_node_id="add1", target_port="x"),
        ArchitectureEdge(edge_id="e4", source_node_id="right", source_port="y", target_node_id="add1", target_port="x"),
        ArchitectureEdge(edge_id="e5", source_node_id="add1", source_port="y", target_node_id="output", target_port="x"),
    ]
    result = CanonicalValidationService().validate(_build_graph(nodes, edges))
    assert result.valid is False
    assert "Branch lineage:" in (result.node_results["add1"].message or "")
    assert len(result.node_results["add1"].suggested_repairs) >= 1


def test_validation_reports_conv_to_rnn_bridge_issue_with_repairs():
    nodes = [
        ArchitectureNode(node_id="input", kind="input", op="Input", params={"shape": [1, 3, 32, 32]}, in_ports=[], out_ports=["y"]),
        ArchitectureNode(node_id="conv1", kind="layer", op="nn.Conv2d", params={"in_channels": 3, "out_channels": 16, "kernel_size": 3, "padding": 1}),
        ArchitectureNode(node_id="lstm1", kind="layer", op="nn.LSTM", params={"input_size": 16, "hidden_size": 32, "batch_first": True}),
        ArchitectureNode(node_id="output", kind="output", op="Output", params={}, in_ports=["x"], out_ports=[]),
    ]
    edges = [
        ArchitectureEdge(edge_id="e1", source_node_id="input", source_port="y", target_node_id="conv1", target_port="x"),
        ArchitectureEdge(edge_id="e2", source_node_id="conv1", source_port="y", target_node_id="lstm1", target_port="x"),
        ArchitectureEdge(edge_id="e3", source_node_id="lstm1", source_port="y", target_node_id="output", target_port="x"),
    ]
    result = CanonicalValidationService().validate(_build_graph(nodes, edges), strict=True)
    assert result.valid is False
    assert "Conv->RNN bridge is invalid" in (result.node_results["lstm1"].message or "")
    assert len(result.node_results["lstm1"].suggested_repairs) >= 1


def test_validation_caches_by_graph_revision_and_mode():
    nodes = [
        ArchitectureNode(node_id="input", kind="input", op="Input", params={"shape": [1, 3, 16, 16]}, in_ports=[], out_ports=["y"]),
        ArchitectureNode(node_id="conv1", kind="layer", op="nn.Conv2d", params={"in_channels": 3, "out_channels": 4, "kernel_size": 3, "padding": 1}),
        ArchitectureNode(node_id="output", kind="output", op="Output", params={}, in_ports=["x"], out_ports=[]),
    ]
    edges = [
        ArchitectureEdge(edge_id="e1", source_node_id="input", source_port="y", target_node_id="conv1", target_port="x"),
        ArchitectureEdge(edge_id="e2", source_node_id="conv1", source_port="y", target_node_id="output", target_port="x"),
    ]
    graph = _build_graph(nodes, edges)
    validator = CanonicalValidationService()

    # First validation populates cache.
    first = validator.validate(graph, strict=True)
    # Second validation should hit cache and return equivalent payload.
    second = validator.validate(graph, strict=True)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_validation_cache_separates_same_revision_with_different_semantics():
    nodes = [
        ArchitectureNode(
            node_id="input",
            kind="input",
            op="Input",
            params={"shape": [1, 8]},
            in_ports=[],
            out_ports=["y"],
        ),
        ArchitectureNode(
            node_id="fc1",
            kind="layer",
            op="nn.Linear",
            params={"in_features": 8, "out_features": 4},
        ),
        ArchitectureNode(
            node_id="output",
            kind="output",
            op="Output",
            params={},
            in_ports=["x"],
            out_ports=[],
        ),
    ]
    edges = [
        ArchitectureEdge(
            edge_id="e1",
            source_node_id="input",
            source_port="y",
            target_node_id="fc1",
            target_port="x",
        ),
        ArchitectureEdge(
            edge_id="e2",
            source_node_id="fc1",
            source_port="y",
            target_node_id="output",
            target_port="x",
        ),
    ]
    first_graph = _build_graph(nodes, edges)
    second_graph = first_graph.model_copy(deep=True)
    second_graph.nodes[1].params["out_features"] = 7
    validator = CanonicalValidationService()

    first = validator.validate(first_graph, strict=True)
    second = validator.validate(second_graph, strict=True)

    assert first.node_results["fc1"].output_shape == [1, 4]
    assert second.node_results["fc1"].output_shape == [1, 7]


def test_stage_validation_does_not_mutate_op_names_or_graph_content():
    nodes = [
        ArchitectureNode(
            node_id="input",
            kind="input",
            op="Input",
            params={"shape": [1, 8]},
            in_ports=[],
            out_ports=["y"],
        ),
        ArchitectureNode(
            node_id="fc1",
            kind="layer",
            op="nn.linear",
            params={"in_features": 8, "out_features": 4},
        ),
        ArchitectureNode(
            node_id="output",
            kind="output",
            op="Output",
            params={},
            in_ports=["x"],
            out_ports=[],
        ),
    ]
    edges = [
        ArchitectureEdge(
            edge_id="e1",
            source_node_id="input",
            source_port="y",
            target_node_id="fc1",
            target_port="x",
        ),
        ArchitectureEdge(
            edge_id="e2",
            source_node_id="fc1",
            source_port="y",
            target_node_id="output",
            target_port="x",
        ),
    ]
    graph = _build_graph(nodes, edges)
    before = graph.model_dump(mode="json")

    report = CanonicalValidationService().validate_stages(graph, strict=True)

    assert report.result.valid is True
    assert report.structural.valid is True
    assert report.shape.valid is True
    assert report.semantic.valid is True
    assert graph.model_dump(mode="json") == before
    assert graph.nodes[1].op == "nn.linear"


def test_validation_names_the_conv2d_channel_mismatch_rather_than_a_rank_problem():
    """A Conv2d fed a *4-D* tensor with the wrong channel count has no rank problem.

    Reporting "requires [N, C, H, W]" for this case sends the reader looking for a
    flatten or reshape that is not there: it is what made the combination-sweep entry
    blame ASPP's branch sizes. Conv2d was the only
    conv-family message that omitted the channel condition; it now reads like
    Linear's, and the numbers say which side is wrong.
    """
    nodes = [
        ArchitectureNode(node_id="input", kind="input", op="Input", params={"shape": [1, 15, 32, 32]}, in_ports=[], out_ports=["y"]),
        ArchitectureNode(node_id="conv1", kind="layer", op="nn.Conv2d", params={"in_channels": 3, "out_channels": 3, "kernel_size": 1}),
        ArchitectureNode(node_id="output", kind="output", op="Output", params={}, in_ports=["x"], out_ports=[]),
    ]
    edges = [
        ArchitectureEdge(edge_id="e1", source_node_id="input", source_port="y", target_node_id="conv1", target_port="x"),
        ArchitectureEdge(edge_id="e2", source_node_id="conv1", source_port="y", target_node_id="output", target_port="x"),
    ]

    result = CanonicalValidationService().validate(_build_graph(nodes, edges))
    conv = result.node_results["conv1"]
    assert result.valid is False
    assert conv.status == "error"
    assert conv.message == "Conv2d in_channels mismatch: expected 3, got 15"
    # The repair was already right; it must stay right.
    assert conv.suggested_repairs == [
        {"op": "set_param", "node_id": "conv1", "param": "in_channels", "value": 15}
    ]


def test_validation_still_reports_rank_when_a_conv2d_is_fed_a_flat_vector():
    """Counterweight: the rank message is the true one for a flattened input."""
    nodes = [
        ArchitectureNode(node_id="input", kind="input", op="Input", params={"shape": [1, 128]}, in_ports=[], out_ports=["y"]),
        ArchitectureNode(node_id="conv1", kind="layer", op="nn.Conv2d", params={"in_channels": 128, "out_channels": 16, "kernel_size": 3}),
        ArchitectureNode(node_id="output", kind="output", op="Output", params={}, in_ports=["x"], out_ports=[]),
    ]
    edges = [
        ArchitectureEdge(edge_id="e1", source_node_id="input", source_port="y", target_node_id="conv1", target_port="x"),
        ArchitectureEdge(edge_id="e2", source_node_id="conv1", source_port="y", target_node_id="output", target_port="x"),
    ]

    result = CanonicalValidationService().validate(_build_graph(nodes, edges))
    conv = result.node_results["conv1"]
    assert result.valid is False
    assert conv.message == "Conv2d requires [N, C, H, W] input shape"
