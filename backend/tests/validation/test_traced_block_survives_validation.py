"""A block the trace MEASURED must reach the validator (#196).

The parser cannot type a user class, so it becomes a module node carrying an
opaque op. `_build_canonical_graph` dropped every module node — correct while the
node carried nothing, and wrong the moment the trace worker started writing the
shape it measured onto that same node.

Dropping it deleted the one fact that could resolve everything downstream, and the
`included_ids` filter deleted its edges with it. So a model that had just executed
came back claiming its layers had no input connections. Tracing made validation
WORSE than not tracing, which is the opposite of what a measurement is for.
"""


from api.routers.validate import (  # noqa: E402
    ValidationRequest,
    _build_canonical_graph,
    _keep_module_node,
    validate_shapes,
)


def _block_node(node_id="blk", out_shape="1x16"):
    """The real request model, not a stand-in: `_keep_module_node` reads it."""
    request = ValidationRequest(**_graph("moduleNode", out_shape, is_group=True))
    node = next(n for n in request.nodes if n.id == "blk")
    node.id = node_id
    return node


def _graph(node_type, out_shape, *, is_group=False, op="custom_code"):
    """input -> BLOCK -> linear -> output, where BLOCK is the user's own class."""
    block_params = {"_out_shape": out_shape} if out_shape else {}
    block_data = {
        "layerType": {"id": op, "name": op, "parameters": []},
        "parameters": block_params,
    }
    if is_group:
        block_data["isGroup"] = True
    return {
        "nodes": [
            {"id": "in", "type": "inputNode", "position": {"x": 0, "y": 0},
             "data": {"layerType": {"id": "input", "name": "Input", "parameters": []},
                      "parameters": {"shape": "1x16"}}},
            {"id": "blk", "type": node_type, "position": {"x": 0, "y": 0}, "data": block_data},
            {"id": "lin", "type": "layerNode", "position": {"x": 0, "y": 0},
             "data": {"layerType": {"id": "linear", "name": "Linear", "parameters": []},
                      "parameters": {"in_features": 16, "out_features": 4}}},
            {"id": "out", "type": "outputNode", "position": {"x": 0, "y": 0},
             "data": {"layerType": {"id": "output", "name": "Output", "parameters": []},
                      "parameters": {}}},
        ],
        "edges": [
            {"id": "e1", "source": "in", "target": "blk"},
            {"id": "e2", "source": "blk", "target": "lin"},
            {"id": "e3", "source": "lin", "target": "out"},
        ],
    }


def _errors(graph):
    result = validate_shapes(ValidationRequest(**graph))
    result = result if isinstance(result, dict) else result.dict()
    return result.get("errors") or []


def test_a_traced_block_validates_clean():
    """The whole point: a measured block resolves its own output, so the chain holds."""
    assert _errors(_graph("moduleNode", "1x16", is_group=True)) == []


def test_a_traced_block_is_not_reported_as_disconnected():
    """The specific regression. Dropping the node took its edges, so `lin` was told
    it had no input connections — with a `connect` repair offering to re-make a wire
    the user already had. That error is one of the few the editor actually SHOWS,
    so this was a visible lie about a working model."""
    messages = " ".join(str(e.get("message", "")) for e in _errors(_graph("moduleNode", "1x16", is_group=True)))
    assert "no input connections" not in messages, messages
    assert "unreachable" not in messages.lower(), messages


def test_an_unmeasured_block_is_kept_but_reported_as_unchecked():
    """Without a measurement there is nothing to resolve with — but dropping the node
    was never the way to say so.

    Deleting it took its EDGES too, so an untraced modular model came back with "Node
    is unreachable from input" and a global unreachable-output error: false wiring
    errors about a graph that is wired correctly. The node stays either way; what
    differs is what is known about it. Without a measurement it is the opaque cause,
    and it must be reported as unchecked — never as clean, and never as broken.
    """
    result = validate_shapes(ValidationRequest(**_graph("moduleNode", None, is_group=True)))
    result = result if isinstance(result, dict) else result.dict()
    messages = " ".join(str(e.get("message", "")) for e in (result.get("errors") or []))
    assert "unreachable" not in messages.lower(), messages
    assert "no input connections" not in messages, messages
    assert result.get("unchecked"), "an unmeasured opaque block must not validate silently clean"
    assert result["unchecked"][0]["nodeId"] == "blk"


def test_a_keep_module_node_whose_op_the_registry_knows_is_not_special_cased():
    """`_out_shape` is only honoured for placeholder ops, so a module node wearing a
    real op id must keep the old drop behaviour rather than trade a cascade for an
    'unknown layer type' error."""
    errors = _errors(_graph("moduleNode", "1x16", is_group=True, op="transformerblock"))
    assert errors, "a non-placeholder op must not be kept on the strength of _out_shape"


def test_an_unwired_keep_module_node_is_still_dropped():
    """Keeping a node nothing connects to would introduce the dangling node the
    connectivity pass then reports as unreachable — trading one false error for another."""
    graph = _graph("moduleNode", "1x16", is_group=True)
    graph["edges"] = [{"id": "e3", "source": "lin", "target": "out"}]
    messages = " ".join(str(e.get("message", "")) for e in _errors(graph))
    assert "blk" not in messages


def test_a_partially_wired_opaque_boundary_keeps_its_real_edge_and_its_own_error():
    graph = _graph("moduleNode", None, is_group=True)
    graph["edges"] = [edge for edge in graph["edges"] if edge["id"] != "e1"]
    canonical = _build_canonical_graph(ValidationRequest(**graph))
    assert "blk" in {node.node_id for node in canonical.nodes}
    assert ("blk", "lin") in {(edge.source_node_id, edge.target_node_id) for edge in canonical.edges}
    errors = _errors(graph)
    assert any(e["nodeId"] == "blk" and "no input connections" in e["message"] for e in errors)
    assert not any(e["nodeId"] == "lin" and "no input connections" in e["message"] for e in errors)


def test_a_frame_whose_children_are_present_is_not_kept():
    """An edge-free group node is grouping, not computation.

    A traced projection sends the block AND its expansion — `encoder_stack` plus
    twelve `encoder_stack.encoder_stack__layer_N` children. There the block is a
    display FRAME and the children carry the computation, so keeping both would
    validate the same model twice. That is how a correctly wired BERT came back with
    "output: Node is unreachable from input".

    Asserted on the rule rather than through a whole graph: what matters is the
    decision, and a hand-built graph of orphan children would test the connectivity
    pass instead.
    """
    node = _block_node("encoder_stack")
    wired = {"encoder_stack"}
    assert _keep_module_node(node, wired, wired, set()) is True
    assert _keep_module_node(node, wired, wired, {"encoder_stack"}) is False
