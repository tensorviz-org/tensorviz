"""Regression tests for the two hierarchical-graph backend bugs.

BUG 1: verify_model unpacked infer_hierarchical_shapes (which returns
{"shapes", "errors"}) as a (shapes, output_shape) tuple. Unpacking a dict
yields its KEYS, so /api/models/verify answered with the literal strings
"shapes" and "errors" instead of shape data — the Shape Table had nothing real
to read whenever the dry forward did not overwrite stage-1 output.

BUG 2: /api/models/validate calls
ModelValidationService.validate_hierarchical_graph, which did not exist — every
hierarchical graph POSTed there raised AttributeError (HTTP 500).
"""



from models.graph_models import HierarchicalGraph  # noqa: E402
from services.verification_service import verify_model  # noqa: E402
from utils.model_validation import ModelValidationService  # noqa: E402

INPUT_SHAPE = [1, 3, 8, 8]


def _layer_type(type_id: str, name: str):
    return {"id": type_id, "name": name, "parameters": []}


def _hier_payload():
    """Input -> Module(Conv2d) -> Output, the smallest real hierarchical graph."""
    return {
        "root_nodes": [
            {"id": "input_node", "type": "inputNode", "position": {"x": 0, "y": 0},
             "data": {"layerType": _layer_type("input", "Input"),
                      "parameters": {"shape": "1,3,8,8"}}},
            {"id": "block", "type": "moduleNode", "position": {"x": 200, "y": 0},
             "data": {"layerType": _layer_type("module", "Module"), "parameters": {}}},
            {"id": "output_node", "type": "outputNode", "position": {"x": 400, "y": 0},
             "data": {"layerType": _layer_type("output", "Output"), "parameters": {}}},
        ],
        "root_edges": [
            {"id": "e1", "source": "input_node", "target": "block"},
            {"id": "e2", "source": "block", "target": "output_node"},
        ],
        "modules": {
            "block": {
                "module_id": "block",
                "parent_id": None,
                "internal_nodes": [
                    {"id": "conv1", "type": "layerNode", "position": {"x": 0, "y": 0},
                     "data": {"layerType": _layer_type("conv2d", "Conv2d"),
                              "parameters": {"in_channels": 3, "out_channels": 4,
                                             "kernel_size": 3, "padding": 1}}},
                ],
                "internal_edges": [],
                "input_node_id": "conv1",
                "output_node_id": "conv1",
            }
        },
        "node_hierarchy": {"conv1": "block"},
    }


def test_hierarchical_shape_inference_returns_real_shapes():
    """BUG 1: stage-1 output must be shape data, never the dict's keys.

    run_forward=False isolates shape inference — this is the data-plane
    validation path, where nothing downstream can paper over the bad unpack.
    """
    result = verify_model(_hier_payload(), INPUT_SHAPE, run_forward=False)

    shapes = result["shapes"]
    assert shapes != "shapes", "BUG 1: dict unpacked as a 2-tuple leaked its keys"
    assert result["output_shape"] != "errors", "BUG 1: output_shape leaked the 'errors' key"
    assert isinstance(shapes, dict), f"expected a node_id -> shape map, got {shapes!r}"
    assert shapes.get("input_node") == INPUT_SHAPE, f"no real input shape in {shapes!r}"
    assert isinstance(shapes.get("block"), list), f"no real module shape in {shapes!r}"
    # A structurally valid graph must not report inference errors: the module's
    # internals legitimately have no Input node of their own.
    assert result["errors"] == [], f"unexpected inference errors: {result['errors']}"
    assert result["ok"] is True, result["errors"]


def test_validate_hierarchical_graph_exists_and_reports_real_errors():
    """BUG 2: the method the /api/models/validate hierarchical branch calls."""
    service = ModelValidationService()

    valid = HierarchicalGraph(**_hier_payload())
    assert service.validate_hierarchical_graph(valid, INPUT_SHAPE) == [], (
        "a valid hierarchical graph must validate clean"
    )

    payload = _hier_payload()
    # Module entry point that is not one of its internal nodes, and a node
    # assigned to a module that does not exist — both make codegen/forward
    # impossible, so validation has to report them.
    payload["modules"]["block"]["input_node_id"] = "ghost"
    payload["node_hierarchy"]["stray"] = "no_such_module"

    errors = service.validate_hierarchical_graph(HierarchicalGraph(**payload), INPUT_SHAPE)
    messages = [e["message"] for e in errors]
    assert any("ghost" in m for m in messages), messages
    assert any("no_such_module" in m for m in messages), messages
    assert all(set(e) == {"nodeId", "message"} for e in errors), errors
