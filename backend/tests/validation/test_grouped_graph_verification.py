"""Verification must not lose a graph's dataflow just because it is grouped.

Three independent defects, all hit by an imported model and all fixed here:

1. `_reconstruct_hierarchy` DROPPED every edge that crossed a module boundary
   (`root_edges` kept only edges whose endpoints were both non-children). An imported
   resnet18 therefore reported "Node avgpool has no input connections" — 11 such errors —
   while the same graph verified fine once the grouping was stripped.
2. A group node was recognised as a container only when its layerType.name was literally
   "Module", so an imported block card named "BasicBlock x2" fell through to the layer
   builder: "Unknown layer type 'BasicBlock x2'".
3. The analytic path had no merge-op handling at all, so a materialised residual died with
   "Unknown layer type 'Add'" even though the dry forward handled the same node — and the
   error REPORTER then crashed on its own, because a multi-input node's cached input is a
   list and it read `.shape` off it: "'list' object has no attribute 'shape'".
"""

from __future__ import annotations



import torch  # noqa: E402

from core.shape_inference import _apply_analytic_merge  # noqa: E402
from services.verification_service import _reconstruct_hierarchy  # noqa: E402


def _node(node_id, layer_id, name, parent=None, node_type="layerNode", params=None):
    data = {
        "layerType": {"id": layer_id, "name": name, "parameters": []},
        "parameters": params or {},
    }
    if parent:
        data["parentId"] = parent
    if node_type == "moduleNode":
        data["isGroup"] = True
    return {"id": node_id, "type": node_type, "position": {"x": 0, "y": 0}, "data": data}


def _edge(eid, source, target):
    return {"id": eid, "source": source, "target": target}


def test_a_flat_graph_whose_modules_carry_no_edges_is_verified_as_is():
    """An imported graph has child-level dataflow and display-only group cards.
    Reconstructing it replaced exact per-layer dataflow with a coarse box chain whose
    boxes then failed to build."""
    payload = {
        "nodes": [
            _node("stem", "conv2d", "Conv2d"),
            _node("block", "basicblock", "BasicBlock x2", node_type="moduleNode"),
            _node("block.c1", "conv2d", "Conv2d", parent="block"),
            _node("tail", "linear", "Linear"),
        ],
        "edges": [
            _edge("e0", "stem", "block.c1"),
            _edge("e1", "block.c1", "tail"),
        ],
    }

    out = _reconstruct_hierarchy(payload)

    assert out is payload or out == payload, "a complete flat graph must pass through"
    assert "root_nodes" not in out


def test_a_cross_boundary_edge_is_re_pointed_not_dropped():
    """When reconstruction DOES apply, an edge into or out of a module must survive as a
    boundary edge. Dropping it is what left the node after a block with no input."""
    payload = {
        "nodes": [
            _node("stem", "conv2d", "Conv2d"),
            _node("mod", "vit", "Module", node_type="moduleNode"),
            _node("mod.a", "linear", "Linear", parent="mod"),
            _node("mod.b", "relu", "ReLU", parent="mod"),
            _node("tail", "linear", "Linear"),
        ],
        # An edge terminating ON the module box makes this the client-flattened shape
        # that reconstruction exists for.
        "edges": [
            _edge("e0", "stem", "mod"),
            _edge("e1", "mod.a", "mod.b"),
            _edge("e2", "mod.b", "tail"),
        ],
    }

    out = _reconstruct_hierarchy(payload)

    assert "root_nodes" in out, "this payload SHOULD be reconstructed"
    boundary = {(e["source"], e["target"]) for e in out["root_edges"]}
    assert ("mod", "tail") in boundary, "the edge out of the module must survive, re-pointed"
    assert ("stem", "mod") in boundary
    # The purely-internal edge is carried by the module, not the root graph.
    assert ("mod.a", "mod.b") not in boundary
    assert any(
        (e["source"], e["target"]) == ("mod.a", "mod.b")
        for e in out["modules"]["mod"]["internal_edges"]
    )


def test_analytic_merge_matches_the_dry_forward():
    """The analytic path and the forward must agree about a merge, or a node accepted by
    one reports a phantom error from the other."""
    a, b = torch.zeros(1, 4, 8, 8), torch.zeros(1, 4, 8, 8)

    assert list(_apply_analytic_merge("add", [a, b], {}).shape) == [1, 4, 8, 8]
    assert list(_apply_analytic_merge("mul", [a, b], {}).shape) == [1, 4, 8, 8]
    # Channels concatenate: dim defaults to 1, which is what every feature-map concat
    # (DenseNet, U-Net) uses. Defaulting to -1 would silently change the shape.
    assert list(_apply_analytic_merge("cat", [a, b], {}).shape) == [1, 8, 8, 8]
    assert list(_apply_analytic_merge("cat", [a, b], {"dim": 2}).shape) == [1, 4, 16, 8]


def test_a_single_input_merge_is_an_identity():
    """A merge fed by one producer must pass it through rather than fail, so a graph that
    lost a branch still verifies."""
    a = torch.zeros(1, 4, 8, 8)

    assert list(_apply_analytic_merge("cat", [a], {}).shape) == [1, 4, 8, 8]
    assert list(_apply_analytic_merge("add", [a], {}).shape) == [1, 4, 8, 8]


def test_an_empty_merge_raises_rather_than_returning_none():
    """Returning None here would surface later as an unrelated AttributeError."""
    try:
        _apply_analytic_merge("add", [], {})
    except ValueError:
        return
    raise AssertionError("expected ValueError for a merge with no inputs")


def test_an_edge_free_module_frame_is_not_a_disconnected_layer():
    """The defect that silently destroyed grouping. A canvas group frame carries no
    semantic edges — the dataflow runs through its children — so strict validation
    reported "Node has no input connections" once per block: 7 errors on a raw depth-2
    resnet18. The AI repair loop then acted on them, wiring the frames into the trunk and
    finally DELETING all 7, so accepting the candidate dropped the model from 71 nodes to
    65 with no removal ever appearing in the plan.
    """
    from models.architecture_models import ArchitectureEdge, ArchitectureGraph, ArchitectureNode
    from services.canonical_validation import CanonicalValidationService

    graph = ArchitectureGraph(
        graph_id="g",
        name="g",
        revision=0,
        inputs=[{"name": "in", "shape": "1,4"}],
        nodes=[
            ArchitectureNode(node_id="in", kind="input", op="Input",
                             params={"shape": "1,4"}, in_ports=[], out_ports=["y"]),
            ArchitectureNode(node_id="fc", kind="layer", op="Linear",
                             params={"in_features": 4, "out_features": 4},
                             in_ports=["x"], out_ports=["y"]),
            # The frame: kind="module", no edges at all.
            ArchitectureNode(node_id="block", kind="module", op="BasicBlock x2",
                             params={}, in_ports=["x"], out_ports=["y"]),
            ArchitectureNode(node_id="out", kind="output", op="Output",
                             params={}, in_ports=["x"], out_ports=[]),
        ],
        edges=[
            ArchitectureEdge(edge_id="e0", source_node_id="in", source_port="y",
                             target_node_id="fc", target_port="x"),
            ArchitectureEdge(edge_id="e1", source_node_id="fc", source_port="y",
                             target_node_id="out", target_port="x"),
        ],
        modules=[],
        operations=[],
        metadata={},
    )

    result = CanonicalValidationService().validate(graph)
    errors = {
        node_id: r.message
        for node_id, r in (result.node_results or {}).items()
        if r.status == "error"
    }

    assert "block" not in errors, f"the group frame must not be flagged: {errors}"
    assert errors == {}, f"nothing else should be flagged either: {errors}"


def test_a_module_that_DOES_carry_edges_is_still_validated():
    """The exemption must be narrow: a module wired into the graph is an executable op and
    a missing input is a real defect. Only an EDGE-FREE module is a canvas frame."""
    from models.architecture_models import ArchitectureEdge, ArchitectureGraph, ArchitectureNode
    from services.canonical_validation import CanonicalValidationService

    graph = ArchitectureGraph(
        graph_id="g",
        name="g",
        revision=0,
        inputs=[{"name": "in", "shape": "1,4"}],
        nodes=[
            ArchitectureNode(node_id="in", kind="input", op="Input",
                             params={"shape": "1,4"}, in_ports=[], out_ports=["y"]),
            # Has an OUTGOING edge but no incoming one: a real disconnection.
            ArchitectureNode(node_id="block", kind="module", op="Block",
                             params={}, in_ports=["x"], out_ports=["y"]),
            ArchitectureNode(node_id="out", kind="output", op="Output",
                             params={}, in_ports=["x"], out_ports=[]),
        ],
        edges=[
            ArchitectureEdge(edge_id="e0", source_node_id="block", source_port="y",
                             target_node_id="out", target_port="x"),
        ],
        modules=[],
        operations=[],
        metadata={},
    )

    result = CanonicalValidationService().validate(graph)
    messages = {
        node_id: r.message
        for node_id, r in (result.node_results or {}).items()
        if r.status == "error"
    }

    assert messages.get("block") == "Node has no input connections"
