"""`/api/models/verify` BUILDS the model; a source is not a thing it can build.

The IDE's Problems panel and the canvas disagreed about the same file: the panel said
zero, the canvas said the model was broken. They run different engines. The panel asks
`/api/validate_shapes`, which analyses the graph and never constructs anything. The
canvas's Validate button asks `/api/models/verify`, which builds real `nn.Module`s and
runs a dry forward — and it called `.eval()` on every node it built:

    Model build failed: 'Parameter' object has no attribute 'eval'

`nn.Parameter` is a Tensor. It has no `.eval()` and no `__call__`, so the whole build
aborted and the user was told the model was broken while the extension, correctly, said
nothing was wrong.

Fourth engine, same misconception: a source PRODUCES a tensor and consumes nothing, so it
is seeded like an input node rather than constructed like a layer.
"""
from __future__ import annotations

from core.shape_inference import _SOURCE_OPS
from services.verification_service import _bare_source_op


def test_the_source_set_is_shared_not_re_spelled():
    """Three engines already agreed on this set; a fourth spelling is how they drift."""
    assert "parameter" in _SOURCE_OPS


def test_the_op_name_is_read_from_either_shape_the_graph_uses():
    class LayerType:
        name = "torch.nn.Parameter"
        id = "parameter"

    assert _bare_source_op({"name": "torch.nn.Parameter"}) == "parameter"
    assert _bare_source_op({"id": "Parameter"}) == "parameter"
    assert _bare_source_op(LayerType()) == "parameter"
    assert _bare_source_op({}) == ""
    assert _bare_source_op(None) == ""


def test_a_parameter_node_verifies_instead_of_aborting_the_build():
    """The end-to-end shape of the bug, without a network round trip."""
    from services.verification_service import verify_model

    graph = dict(
        nodes=[
            {"id": "input", "type": "inputNode", "position": {"x": 0, "y": 0},
             "data": {"layerType": {"id": "input", "name": "Input", "parameters": []},
                      "parameters": {"shape": "1,4"}}},
            {"id": "pos", "type": "layerNode", "position": {"x": 0, "y": 0},
             "data": {"layerType": {"id": "parameter", "name": "Parameter", "parameters": []},
                      "parameters": {"_out_shape": "1x4"}}},
            {"id": "add", "type": "layerNode", "position": {"x": 0, "y": 0},
             "data": {"layerType": {"id": "add", "name": "torch.add", "parameters": []},
                      "parameters": {}}},
            {"id": "output", "type": "outputNode", "position": {"x": 0, "y": 0},
             "data": {"layerType": {"id": "output", "name": "Output", "parameters": []},
                      "parameters": {}}},
        ],
        edges=[
            {"id": "e1", "source": "input", "target": "add"},
            {"id": "e2", "source": "pos", "target": "add"},
            {"id": "e3", "source": "add", "target": "output"},
        ],
    )
    result = verify_model(graph, [1, 4], run_forward=True)
    messages = " ".join(str(e.get("message", "")) for e in (result.get("errors") or []))
    assert "has no attribute 'eval'" not in messages, messages
    assert "Model build failed" not in messages, messages
