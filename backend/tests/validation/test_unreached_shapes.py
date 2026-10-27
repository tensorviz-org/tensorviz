"""A node the run never reached gets no invented shape (task.md M4/M8).

The trace marks every node after a failing statement `unreached`, and the static
shape engine used to compute a shape for it anyway: a GELU after a Linear that
raised showed (2, 20), confidently, derived from out_features alone. That put a
number on a card for a layer whose input never existed, while the header said the
run had failed. The engine now says what the trace said, and says it for the
nodes downstream too, instead of blaming their wiring.
"""

from __future__ import annotations



from core.shape_inference import infer_node_shapes_with_context  # noqa: E402
from models.graph_models import Graph  # noqa: E402


#: The engine keys on `layerType.name` ('Input', 'Linear', 'GELU'), not on the id.
_NAMES = {"input": "Input", "output": "Output", "linear": "Linear", "gelu": "GELU"}


def _node(node_id, op, params, ntype="layerNode"):
    return {
        "id": node_id,
        "type": ntype,
        "position": {"x": 0.0, "y": 0.0},
        "data": {"layerType": {"id": op, "name": _NAMES[op], "parameters": []}, "parameters": params},
    }


def _graph(act_params):
    return Graph(
        **{
            "nodes": [
                _node("input", "input", {"shape": "2,8"}, "inputNode"),
                _node("fc", "linear", {"in_features": 8, "out_features": 10}),
                _node("act", "gelu", act_params),
                _node("head", "linear", {"in_features": 10, "out_features": 4}),
                _node("output", "output", {}, "outputNode"),
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "fc"},
                {"id": "e2", "source": "fc", "target": "act"},
                {"id": "e3", "source": "act", "target": "head"},
                {"id": "e4", "source": "head", "target": "output"},
            ],
        }
    )


def _run(graph):
    result = infer_node_shapes_with_context(graph, [2, 8])
    shapes = result.get("shapes") if isinstance(result, dict) else result[0]
    errors = result.get("errors") if isinstance(result, dict) else result[1]
    return shapes, {e["nodeId"]: e for e in errors}


def test_an_unreached_node_gets_no_shape_and_says_why():
    shapes, errors = _run(_graph({"_shape_provenance": "unreached"}))

    assert "act" not in shapes, f"a shape was invented for a node the run never reached: {shapes.get('act')}"
    assert errors["act"]["error_type"] == "unreached"
    assert "run failed before it" in errors["act"]["message"]


def test_the_nodes_after_it_inherit_the_reason_not_a_wiring_complaint():
    # Downstream of an unreached node, "check that preceding layers are connected"
    # is a wrong diagnosis: the wiring is fine, the run stopped. Blaming
    # connectivity sends the user to fix an edge that is not broken.
    shapes, errors = _run(_graph({"_shape_provenance": "unreached"}))

    assert "head" not in shapes
    assert errors["head"]["error_type"] == "unreached", errors["head"]


def test_the_same_graph_without_the_mark_is_computed_normally():
    # The other half: the mark is the only thing that changes the answer, so a
    # healthy graph keeps every shape it had.
    shapes, errors = _run(_graph({}))

    assert shapes["act"] == [2, 10]
    assert shapes["head"] == [2, 4]
    assert "act" not in errors and "head" not in errors


def test_the_route_the_canvas_calls_gives_an_unreached_node_no_shape():
    """`/api/validate_shapes` is what the canvas asks 700 ms after any change, and
    it is backed by the canonical validator, not the engine above — so the fix has
    to hold there too, or the card keeps its invented (2, 20)."""
    from fastapi.testclient import TestClient

    import server_v2

    client = TestClient(server_v2.app)

    def node(node_id, op, params, ntype="layerNode"):
        return {
            "id": node_id,
            "type": ntype,
            "position": {"x": 0.0, "y": 0.0},
            "data": {"layerType": {"id": op, "name": _NAMES[op], "parameters": []}, "parameters": params},
        }

    body = {
        "nodes": [
            node("input", "input", {"shape": "2,8"}, "inputNode"),
            node("fc", "linear", {"in_features": 8, "out_features": 10}),
            node("act", "gelu", {"_shape_provenance": "unreached"}),
            node("head", "linear", {"in_features": 10, "out_features": 4}),
            node("output", "output", {}, "outputNode"),
        ],
        "edges": [
            {"id": "e1", "source": "input", "target": "fc"},
            {"id": "e2", "source": "fc", "target": "act"},
            {"id": "e3", "source": "act", "target": "head"},
            {"id": "e4", "source": "head", "target": "output"},
        ],
    }
    response = client.post("/api/validate_shapes", json=body)
    assert response.status_code == 200, response.text
    payload = response.json()

    shapes = payload["shapes"]
    assert shapes.get("fc") == [2, 10], "the part of the model that ran keeps its shape"
    for node_id in ("act", "head", "output"):
        assert not shapes.get(node_id), f"{node_id} was given a shape the run never produced: {shapes.get(node_id)}"

    by_node = {e["nodeId"]: e for e in payload["errors"]}
    assert "not reached" in by_node["act"]["message"]
    assert "not reached" in by_node["head"]["message"], "downstream must inherit the reason, not blame its wiring"
