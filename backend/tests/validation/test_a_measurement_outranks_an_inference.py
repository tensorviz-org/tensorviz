"""A node the trace measured is not starved by anything upstream of it (#196).

Two floods in the validator mark nodes by inference from their neighbours: `unreached`
(the run stopped before this) and `blocked_by` (an opaque block in front of this has no
shape). Neither may overrule a MEASUREMENT. A node carrying an observed `_out_shape`
demonstrably ran and demonstrably has a shape, whatever is upstream.

The `unreached` flood had no such guard, and it cost F5-TTS 207 findings. Its projection
invents three ghost statements — `for _blk in self.dit_block:` loop bodies that the
worker's frame heuristic fails to credit, because a user-authored submodule's first op
runs inside the CHILD's forward rather than at the call site. Those three ghosts are
spliced into the middle of the traced chain, and the mark then flooded forward across 143
nodes that each carry an observed shape. Every one was reported "was not reached: the run
failed before it" — on a model whose trace error is None.

The opaque flood twenty lines below had always had the equivalent guard on its CAUSE
(`_recorded_output_shape(n) is None`). Now both defer to a measurement, on victims too.
"""


from api.routers.validate import ValidationRequest, validate_shapes  # noqa: E402

OBSERVED = {"_out_shape": "1x16x32", "_shape_provenance": "observed"}


def _node(node_id, node_type, op, params=None):
    return {
        "id": node_id,
        "type": node_type,
        "position": {"x": 0, "y": 0},
        "data": {
            "layerType": {"id": op, "name": op, "parameters": []},
            "parameters": params or {},
        },
    }


def _ghost_then_measured_chain(length=12):
    """input -> emb(measured) -> GHOST(unreached) -> N measured layers -> output."""
    nodes = [
        _node("input", "inputNode", "input", {"shape": "1,16"}),
        _node("emb", "layerNode", "embedding",
              {"num_embeddings": 256, "embedding_dim": 32, **OBSERVED}),
        _node("ghost", "layerNode", "python", {"_shape_provenance": "unreached"}),
    ]
    edges = [
        {"id": "e0", "source": "input", "target": "emb"},
        {"id": "e1", "source": "emb", "target": "ghost"},
    ]
    previous = "ghost"
    for index in range(length):
        node_id = f"blk{index}"
        nodes.append(_node(node_id, "layerNode", "relu", dict(OBSERVED)))
        edges.append({"id": f"e{index + 2}", "source": previous, "target": node_id})
        previous = node_id
    nodes.append(_node("output", "outputNode", "output", dict(OBSERVED)))
    edges.append({"id": "eout", "source": previous, "target": "output"})
    return {"nodes": nodes, "edges": edges}


def _run(graph):
    result = validate_shapes(ValidationRequest(**graph))
    return result if isinstance(result, dict) else result.dict()


def test_one_ghost_does_not_condemn_everything_measured_after_it():
    result = _run(_ghost_then_measured_chain())
    errors = result.get("errors") or []
    # The ghost itself is honestly unreached. Nothing else is.
    assert [e["nodeId"] for e in errors] == ["ghost"], errors


def test_a_measured_node_is_not_reported_unchecked_either():
    """The opaque flood must defer to a measurement on its VICTIMS, not only its cause."""
    result = _run(_ghost_then_measured_chain())
    blocked = [b for entry in (result.get("unchecked") or []) for b in entry["blocked"]]
    assert blocked == [], f"measured nodes reported as unchecked: {blocked}"


def test_the_flood_still_works_where_nothing_was_measured():
    """The guard must not switch the flood off — a genuinely unmeasured chain after a
    failure is still unreached, which is what the mark is for."""
    graph = _ghost_then_measured_chain(length=3)
    for node in graph["nodes"]:
        if node["id"].startswith("blk") or node["id"] == "output":
            node["data"]["parameters"] = {}
    flagged = {e["nodeId"] for e in (_run(graph).get("errors") or [])}
    assert "ghost" in flagged
    assert len(flagged) > 1, f"the unreached mark stopped propagating entirely: {flagged}"
