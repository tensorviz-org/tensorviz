"""A module is a FRAME when its children carry the dataflow — not when a dotted id exists.

The first version of this rule asked "does some node's id start with mine plus a dot",
which is a string match, not a question about the graph. It misread every traced
`nn.MultiheadAttention`: at the worker's default depth an attention node materialises one
decoration child, `<attn>.out_proj`, that carries no edges at all. The attention itself
sits on the trunk with an edge in and an edge out, so dropping it deleted `norm1 -> attn`
and `attn -> norm2` and cut the chain in half — seven connectivity errors on a ViT block
that is wired correctly. 84 templates, 497 such errors, and not one of them true.

The corrected rule has three parts, and each of these tests holds one of them.
"""


from api.routers.validate import ValidationRequest, validate_shapes  # noqa: E402


def _node(node_id, node_type, op, params=None, group=False):
    data = {
        "layerType": {"id": op, "name": op, "parameters": []},
        "parameters": params or {},
    }
    if group:
        data["isGroup"] = True
    return {"id": node_id, "type": node_type, "position": {"x": 0, "y": 0}, "data": data}


def _errors(graph):
    result = validate_shapes(ValidationRequest(**graph))
    result = result if isinstance(result, dict) else result.dict()
    return [f"{e.get('nodeId')} | {e.get('message')}" for e in (result.get("errors") or [])]


def _attention_with_decoration_child():
    return {
        "nodes": [
            _node("input", "inputNode", "input", {"shape": "1,16,256"}),
            _node("norm1", "layerNode", "layernorm", {"normalized_shape": 256}),
            _node("attn", "moduleNode", "multiheadattention",
                  {"embed_dim": 256, "num_heads": 8}, group=True),
            # carries NO edges: this is what the prefix test mistook for proof of a frame
            _node("attn.out_proj", "layerNode", "linear",
                  {"in_features": 256, "out_features": 256}),
            _node("norm2", "layerNode", "layernorm", {"normalized_shape": 256}),
            _node("output", "outputNode", "output"),
        ],
        "edges": [
            {"id": "e1", "source": "input", "target": "norm1"},
            {"id": "e2", "source": "norm1", "target": "attn"},
            {"id": "e3", "source": "attn", "target": "norm2"},
            {"id": "e4", "source": "norm2", "target": "output"},
        ],
    }


def _real_frame():
    """The children carry the chain; the boundary edges name the FRAME."""
    return {
        "nodes": [
            _node("input", "inputNode", "input", {"shape": "1,3,8,8"}),
            _node("features", "moduleNode", "custom_code", {}, group=True),
            _node("features.c0", "layerNode", "relu"),
            _node("features.c1", "layerNode", "relu"),
            _node("output", "outputNode", "output"),
        ],
        "edges": [
            {"id": "e1", "source": "input", "target": "features"},
            {"id": "e2", "source": "features.c0", "target": "features.c1"},
            {"id": "e3", "source": "features", "target": "output"},
        ],
    }


def test_a_module_whose_children_carry_no_edge_is_the_computation():
    """Rule 1 + 3: keep the attention, drop its edge-free decoration child."""
    assert _errors(_attention_with_decoration_child()) == []


def test_a_real_frames_boundary_edges_are_rewired_not_deleted():
    """Rule 2: `input -> features` and `features -> output` name the frame, and the
    frame is correctly dropped — so they must land on its entry and exit children
    rather than being deleted, which orphaned `output`."""
    assert _errors(_real_frame()) == []


def test_a_boundary_edge_the_payload_already_states_at_child_level_is_not_duplicated():
    """A rewired edge that duplicates one the caller already sent is suppressed —
    but only a REWIRED one, so a duplicate the caller really sent is still theirs."""
    graph = _real_frame()
    graph["edges"].append({"id": "e4", "source": "input", "target": "features.c0"})
    assert _errors(graph) == []


def test_a_plain_layer_named_like_a_child_cannot_delete_its_namesake():
    """Only a module node may own children. A user naming a layer `block.norm` must
    not cause `block` — an ordinary layer on the trunk — to be dropped as a frame."""
    graph = {
        "nodes": [
            _node("input", "inputNode", "input", {"shape": "1,8"}),
            _node("block", "layerNode", "relu"),
            _node("block.norm", "layerNode", "relu"),
            _node("output", "outputNode", "output"),
        ],
        "edges": [
            {"id": "e1", "source": "input", "target": "block"},
            {"id": "e2", "source": "block", "target": "block.norm"},
            {"id": "e3", "source": "block.norm", "target": "output"},
        ],
    }
    assert _errors(graph) == []
