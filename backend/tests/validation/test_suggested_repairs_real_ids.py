"""Suggested repairs must never name a node id that is not in the graph.

``suggested_repairs`` is not a human hint. ``build_repair_prompt`` embeds it in the
AI repair prompt as "Suggested repair operations (apply or adapt these)", so a
placeholder id is an instruction to reference a node that does not exist. The
structural checks used to emit the literals ``"input"``, ``"last_layer"`` and
``"output"``; planners copied them verbatim and the turn died with
``ValueError: Source node 'input' not found`` — observed on gpt-5.5 and
gpt-5.6-terra.

The rule pinned here: emit a real id, or emit no repair at all. Inventing a
plausible-looking id is strictly worse than staying silent, because the message
already says what is wrong.
"""

from __future__ import annotations

from models.architecture_models import ArchitectureGraph
from services.canonical_validation import CanonicalValidationService


def _graph(nodes, edges=()):
    return ArchitectureGraph.model_validate({
        "graph_id": "g", "name": "n", "revision": 1,
        "nodes": [
            {"node_id": nid, "kind": kind, "op": op, "params": params,
             "in_ports": ["x"], "out_ports": ["y"]}
            for nid, kind, op, params in nodes
        ],
        "edges": [
            {"edge_id": f"e{i}", "source_node_id": s, "source_port": "y",
             "target_node_id": t, "target_port": "x"}
            for i, (s, t) in enumerate(edges)
        ],
    })


def _all_suggested_ids(result):
    """Every node id referenced by any suggested repair, across all nodes."""
    ids = []
    for node_result in (result.node_results or {}).values():
        for repair in (node_result.suggested_repairs or []):
            for field in ("source_node_id", "target_node_id", "node_id"):
                if repair.get(field):
                    ids.append(repair[field])
    return ids


def _validate(graph):
    return CanonicalValidationService().validate(graph)


def test_a_disconnected_layer_is_not_told_to_connect_from_a_nonexistent_input():
    """The exact live failure: source_node_id "input" when the node is "input_node"."""
    graph = _graph([
        ("input_node", "input", "Input", {"shape": "1,16"}),
        ("orphan", "layer", "ReLU", {}),
        ("output_node", "output", "Output", {}),
    ])
    result = _validate(graph)
    referenced = _all_suggested_ids(result)
    real = {n.node_id for n in graph.nodes}
    assert "input" not in referenced, "placeholder 'input' is back"
    assert "last_layer" not in referenced
    for node_id in referenced:
        assert node_id in real, f"suggested repair names a nonexistent node {node_id!r}"


def test_the_real_input_id_is_used_when_there_is_exactly_one():
    graph = _graph([
        ("input_node", "input", "Input", {"shape": "1,16"}),
        ("orphan", "layer", "ReLU", {}),
    ])
    result = _validate(graph)
    assert "input_node" in _all_suggested_ids(result), (
        "with one unambiguous input the repair should name it"
    )


def test_no_repair_is_invented_when_the_input_is_ambiguous():
    """Two inputs: nominating one arbitrarily would be a guess presented as advice."""
    graph = _graph([
        ("in_a", "input", "Input", {"shape": "1,16"}),
        ("in_b", "input", "Input", {"shape": "1,16"}),
        ("orphan", "layer", "ReLU", {}),
    ])
    result = _validate(graph)
    orphan = (result.node_results or {}).get("orphan")
    assert orphan is not None and orphan.status == "error", "the error must still be reported"
    assert not (orphan.suggested_repairs or []), "a guess was emitted for an ambiguous input"


def test_every_suggested_id_exists_on_a_realistic_broken_graph():
    """Sweep: a dangling tail plus an unconnected output, the shape that produced
    both "last_layer" and "output"."""
    graph = _graph(
        [
            ("input_node", "input", "Input", {"shape": "1,16"}),
            ("fc", "layer", "Linear", {"in_features": 16, "out_features": 8}),
            ("output_node", "output", "Output", {}),
        ],
        edges=[("input_node", "fc")],
    )
    result = _validate(graph)
    real = {n.node_id for n in graph.nodes}
    for node_id in _all_suggested_ids(result):
        assert node_id in real, f"nonexistent id {node_id!r} in suggested repairs"
