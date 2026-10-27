from models.architecture_models import ArchitectureEdge, ArchitectureGraph, ArchitectureNode
from services.canonical_validation import CanonicalValidationService


def _graph_with_unflatten(unflattened_size):
    nodes = [
        ArchitectureNode(node_id="input", kind="input", op="Input", params={"shape": [1, 12]}, in_ports=[], out_ports=["y"]),
        ArchitectureNode(
            node_id="unflatten",
            kind="layer",
            op="nn.Unflatten",
            params={"dim": 1, "unflattened_size": unflattened_size},
            in_ports=["x"],
            out_ports=["y"],
        ),
        ArchitectureNode(node_id="output", kind="output", op="Output", params={}, in_ports=["x"], out_ports=[]),
    ]
    edges = [
        ArchitectureEdge(
            edge_id="input:y->unflatten:x",
            source_node_id="input",
            source_port="y",
            target_node_id="unflatten",
            target_port="x",
        ),
        ArchitectureEdge(
            edge_id="unflatten:y->output:x",
            source_node_id="unflatten",
            source_port="y",
            target_node_id="output",
            target_port="x",
        ),
    ]
    return ArchitectureGraph(
        graph_id="graph_unflatten_test",
        name="unflatten_test",
        revision=0,
        inputs=[{"name": "x", "shape": [1, 12]}],
        nodes=nodes,
        edges=edges,
        metadata={},
    )


def test_unflatten_shape_inference_valid():
    validator = CanonicalValidationService()
    graph = _graph_with_unflatten([3, 4])
    result = validator.validate(graph=graph, strict=False)
    assert result.valid is True
    assert result.node_results["unflatten"].output_shape == [1, 3, 4]


def test_unflatten_shape_inference_invalid_size_product():
    validator = CanonicalValidationService()
    graph = _graph_with_unflatten([5, 5])
    result = validator.validate(graph=graph, strict=False)
    assert result.valid is False
    assert result.node_results["unflatten"].status == "error"
