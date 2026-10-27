"""Tests for GraphDiffService — structural diff between two graphs."""



from models.architecture_models import (  # noqa: E402
    ArchitectureEdge,
    ArchitectureGraph,
    ArchitectureNode,
)
from services.graph_diff_service import GraphDiffService  # noqa: E402


def _g(graph_id: str, nodes, edges) -> ArchitectureGraph:
    return ArchitectureGraph(graph_id=graph_id, name=graph_id, revision=1, nodes=nodes, edges=edges)


def _node(node_id: str, op: str, **params) -> ArchitectureNode:
    return ArchitectureNode(node_id=node_id, kind="layer", op=op, params=params)


def _edge(edge_id: str, src: str, tgt: str) -> ArchitectureEdge:
    return ArchitectureEdge(edge_id=edge_id, source_node_id=src, source_port="y", target_node_id=tgt, target_port="x")


def test_identical_graphs_have_no_diff():
    g = _g("g", [_node("conv", "nn.Conv2d", in_channels=3, out_channels=8, kernel_size=3)], [])
    result = GraphDiffService().diff(g, g)
    assert result.summary["added_nodes"] == 0
    assert result.summary["removed_nodes"] == 0
    assert result.summary["modified_nodes"] == 0
    assert result.summary["unchanged_nodes"] == 1


def test_added_node_is_detected():
    g_a = _g("a", [_node("conv", "nn.Conv2d", in_channels=3, out_channels=8, kernel_size=3)], [])
    g_b = _g("b", [
        _node("conv", "nn.Conv2d", in_channels=3, out_channels=8, kernel_size=3),
        _node("relu", "nn.ReLU"),
    ], [_edge("e1", "conv", "relu")])
    result = GraphDiffService().diff(g_a, g_b)
    assert result.summary["added_nodes"] == 1
    assert result.summary["removed_nodes"] == 0
    added = [d for d in result.nodes if d.status == "added"]
    assert added[0].node_id_b == "relu"


def test_removed_node_is_detected():
    g_a = _g("a", [
        _node("conv", "nn.Conv2d", in_channels=3, out_channels=8, kernel_size=3),
        _node("relu", "nn.ReLU"),
    ], [_edge("e1", "conv", "relu")])
    g_b = _g("b", [_node("conv", "nn.Conv2d", in_channels=3, out_channels=8, kernel_size=3)], [])
    result = GraphDiffService().diff(g_a, g_b)
    assert result.summary["removed_nodes"] == 1
    removed = [d for d in result.nodes if d.status == "removed"]
    assert removed[0].node_id_a == "relu"


def test_modified_params_are_reported():
    g_a = _g("a", [_node("conv", "nn.Conv2d", in_channels=3, out_channels=8, kernel_size=3)], [])
    g_b = _g("b", [_node("conv", "nn.Conv2d", in_channels=3, out_channels=16, kernel_size=3)], [])
    result = GraphDiffService().diff(g_a, g_b)
    assert result.summary["modified_nodes"] == 1
    modified = [d for d in result.nodes if d.status == "modified"][0]
    assert "out_channels" in modified.param_changes
    assert modified.param_changes["out_channels"] == {"before": 8, "after": 16}


def test_renamed_node_with_same_structure_is_matched():
    """conv1 in graph A matches conv_a in graph B because they have the
    same op and identical params. Without this, every renamed layer
    would look like a delete+add pair."""
    g_a = _g("a", [_node("conv1", "nn.Conv2d", in_channels=3, out_channels=8, kernel_size=3)], [])
    g_b = _g("b", [_node("conv_a", "nn.Conv2d", in_channels=3, out_channels=8, kernel_size=3)], [])
    result = GraphDiffService().diff(g_a, g_b)
    assert result.summary["added_nodes"] == 0
    assert result.summary["removed_nodes"] == 0
    assert result.summary["unchanged_nodes"] == 1
    assert result.correspondence == {"conv1": "conv_a"}


def test_added_edge_is_detected():
    g_a = _g("a", [
        _node("conv", "nn.Conv2d", in_channels=3, out_channels=8, kernel_size=3),
        _node("relu", "nn.ReLU"),
    ], [])
    g_b = _g("b", [
        _node("conv", "nn.Conv2d", in_channels=3, out_channels=8, kernel_size=3),
        _node("relu", "nn.ReLU"),
    ], [_edge("e1", "conv", "relu")])
    result = GraphDiffService().diff(g_a, g_b)
    assert result.summary["added_edges"] == 1
    assert result.summary["removed_edges"] == 0


def test_explicit_correspondence_overrides_structural_match():
    """User-provided correspondence wins over structural matching."""
    g_a = _g("a", [
        _node("conv1", "nn.Conv2d", in_channels=3, out_channels=8, kernel_size=3),
        _node("conv2", "nn.Conv2d", in_channels=3, out_channels=8, kernel_size=3),
    ], [])
    g_b = _g("b", [
        _node("conv1", "nn.Conv2d", in_channels=3, out_channels=8, kernel_size=3),
        _node("conv2", "nn.Conv2d", in_channels=3, out_channels=8, kernel_size=3),
    ], [])
    # Without correspondence, conv1->conv1 and conv2->conv2 (default ID match).
    # Force a swap.
    correspondence = {"conv1": "conv2", "conv2": "conv1"}
    result = GraphDiffService().diff(g_a, g_b, correspondence=correspondence)
    assert result.correspondence["conv1"] == "conv2"
    assert result.correspondence["conv2"] == "conv1"


def test_changed_op_marks_node_modified():
    g_a = _g("a", [_node("act", "nn.ReLU")], [])
    g_b = _g("b", [_node("act", "nn.GELU")], [])
    result = GraphDiffService().diff(g_a, g_b)
    # Same node_id, different op — should be detected as modified.
    modified = [d for d in result.nodes if d.status == "modified"]
    assert len(modified) == 1
    assert modified[0].op == "nn.GELU"
