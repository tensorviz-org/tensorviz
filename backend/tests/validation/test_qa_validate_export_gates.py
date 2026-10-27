"""HTTP gates on the model plane: what validate blesses, and what export ships.

Every payload here is one the Ship-Risk audit sent live and got the wrong answer
for. The endpoints had no HTTP-level test at all, which is how a *validate: true
-> export -> RuntimeError* chain survived a green 1,900-test suite: each plane
trusted its input independently, and nothing exercised them through the router.

The two properties worth stating outright, because they are what "valid" and
"runnable export" have to mean:

  * a graph the product cannot build is not valid, and
  * nothing gets exported that the product cannot run.
"""

from typing import Any, Dict, List

import pytest
import torch
from fastapi.testclient import TestClient


import server_v2  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(server_v2.app)


def _node(node_id: str, node_type: str, layer_id: str, layer_name: str,
          parameters: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": node_id,
        "type": node_type,
        "position": {"x": 0.0, "y": 0.0},
        "data": {
            "layerType": {"id": layer_id, "name": layer_name, "parameters": []},
            "parameters": parameters,
        },
    }


def _edge(edge_id: str, source: str, target: str) -> Dict[str, str]:
    return {"id": edge_id, "source": source, "target": target}


def _runnable_graph() -> Dict[str, Any]:
    """A small CNN that really does run: 1x3x32x32 -> Conv2d -> Flatten -> Linear."""
    return {
        "nodes": [
            _node("i", "inputNode", "input", "Input", {"shape": "1,3,32,32"}),
            _node("c", "layerNode", "conv2d", "Conv2d",
                  {"in_channels": 3, "out_channels": 8, "kernel_size": 3, "padding": 1}),
            _node("f", "layerNode", "flatten", "Flatten", {}),
            _node("l", "layerNode", "linear", "Linear",
                  {"in_features": 8192, "out_features": 10}),
        ],
        "edges": [_edge("e1", "i", "c"), _edge("e2", "c", "f"), _edge("e3", "f", "l")],
    }


def _messages(errors: List[Dict[str, Any]]) -> str:
    return " | ".join(entry.get("message", "") for entry in errors)


# ── validate ────────────────────────────────────────────────────────────────

def test_an_unknown_layer_type_is_not_valid(client: TestClient) -> None:
    """`foobar_layer` names no operation, so nothing can build, infer or export it.

    It answered valid:true because the validator hard-coded checks for three
    layer ids and had no opinion about any other — including ids that exist
    nowhere.
    """
    response = client.post("/api/models/validate", json={
        "graph": {"nodes": [_node("n1", "layerNode", "foobar_layer", "FooBar", {})],
                  "edges": []},
    })

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    assert "foobar_layer" in _messages(body["errors"])


def test_source_code_the_validator_cannot_model_is_not_valid(client: TestClient) -> None:
    """A `python` node is real code, not an operation — so the graph is not blessed.

    `python` and `custom_code` are registered so the canvas can render a node for
    the statement the parser could not type, and so `spec_for` resolves it. That
    made `_names_an_operation` answer True and the whole graph come back
    valid:true with an empty error list — a green tick on a model nothing had
    checked, which is the exact failure this file exists to catch, just wearing a
    registered id instead of an unknown one.

    The refusal must also say WHERE the code is, because a node id like
    `python_1` is not something the user chose or recognises.
    """
    graph = _runnable_graph()
    graph["nodes"].append(_node("python_1", "layerNode", "python", "Python", {
        "_raw_call": "out.transpose(1, 2)",
        "_source_file": "models/resblock.py",
        "_lineno": 15,
    }))
    graph["edges"].append(_edge("e4", "l", "python_1"))

    response = client.post("/api/models/validate", json={"graph": graph})

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False, "a graph carrying unmodelled source is not a checked model"
    message = _messages(body["errors"])
    assert "python_1" in message
    assert "models/resblock.py:15" in message, "the refusal must point at the statement"


def test_parameter_sanity_does_not_depend_on_the_node_type_spelling(client: TestClient) -> None:
    """in_features=-5 is invalid however the node labels itself.

    The parameter checks ran only for nodes whose `type` read exactly
    "layerNode", so a payload carrying any other spelling skipped every one of
    them and came back valid:true with no warning.
    """
    response = client.post("/api/models/validate", json={
        "graph": {"nodes": [_node("n1", "default", "linear", "Linear",
                                  {"in_features": -5, "out_features": 4})],
                  "edges": []},
    })

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    assert "in_features" in _messages(body["errors"])


@pytest.mark.parametrize("layer_id", [
    "Linear", "LINEAR", "nn.Linear", "torch.nn.Linear", "  Linear  ", "torch.Linear",
    # Double-prefixed spellings. The registry accepts these because its keys ARE
    # the prefixed canonical names ("nn.linear"), so spec_for strips one prefix
    # off "nn.nn.Linear" and lands on a real spec. A hand-rolled one-shot regex in
    # the validator did not, which put the node back in the silent-skip hole this
    # whole test exists to close. Independent verification found it; it is pinned
    # here so the two normalisations can never drift apart again.
    "nn.nn.Linear", "torch.nn.nn.Linear", "NN.NN.LINEAR",
])
def test_parameter_sanity_does_not_depend_on_how_the_layer_id_is_spelled(
    client: TestClient, layer_id: str
) -> None:
    """The other half of the audit's silent skip: "or the layer id is capitalized".

    Unknown-id resolution goes through op_registry, which is alias-aware and
    case-insensitive, so `Linear` is not "unknown" — but the parameter checks
    compared against the bare lowercase literal, so a capitalized or prefixed id
    fell off the end of the chain and came back valid:true with no error at all.
    Every spelling the registry accepts has to reach the same checks.
    """
    response = client.post("/api/models/validate", json={
        "graph": {"nodes": [_node("n1", "layerNode", layer_id, "Linear",
                                  {"in_features": -5, "out_features": 4})],
                  "edges": []},
    })

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False, f"{layer_id} bypassed the parameter checks"
    assert "in_features" in _messages(body["errors"])


def test_a_canvas_with_no_layers_is_not_a_complete_model(client: TestClient) -> None:
    """"Nothing to generate" means no LAYER, not zero nodes.

    The empty-list guard left the symptom one step away: a canvas holding only an
    Input node — or only an Input and an Output — produced exactly the same
    `def forward(self, x): return x` stamped complete:true.
    """
    input_only = _node("i", "inputNode", "input", "Input", {"shape": "1,3,32,32"})
    output = _node("o", "outputNode", "output", "Output", {})

    for nodes in ([input_only], [input_only, output]):
        response = client.post("/api/models/generate-code", json={"graph": {"nodes": nodes, "edges": []}})
        assert response.status_code == 400, response.text
        assert "no layers" in response.json()["detail"]


def test_a_graph_with_no_input_node_says_which_shape_it_assumed(client: TestClient) -> None:
    """GO-19: the tick has to disclose the shape nobody chose.

    A lone Linear(512 -> 256) is structurally fine, so it stays valid — but
    validation silently measured it against [1, 3, 224, 224], and the endpoint
    hardcoded `"warnings": []`, so the user saw an unqualified green tick for a
    shape they never picked.
    """
    response = client.post("/api/models/validate", json={
        "graph": {"nodes": [_node("n1", "layerNode", "linear", "Linear",
                                  {"in_features": 512, "out_features": 256})],
                  "edges": []},
    })

    assert response.status_code == 200
    body = response.json()
    warnings = _messages(body["warnings"])
    assert "No Input node found" in warnings
    assert "[1, 3, 224, 224]" in warnings


def test_a_cycle_is_still_invalid(client: TestClient) -> None:
    """The one check that already worked. Tightening the others must not lose it."""
    response = client.post("/api/models/validate", json={
        "graph": {
            "nodes": [_node("a", "layerNode", "relu", "ReLU", {}),
                      _node("b", "layerNode", "relu", "ReLU", {})],
            "edges": [_edge("e1", "a", "b"), _edge("e2", "b", "a")],
        },
    })

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    assert "cycle" in _messages(body["errors"]).lower()


def test_a_runnable_graph_is_valid_with_no_warnings(client: TestClient) -> None:
    """The counterweight: tightening validation must not reject a working model."""
    response = client.post("/api/models/validate",
                           json={"graph": _runnable_graph(), "input_shape": [1, 3, 32, 32]})

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True, _messages(body["errors"])
    assert "No Input node found" not in _messages(body["warnings"])


def test_a_malformed_graph_report_does_not_quote_what_was_submitted(client: TestClient) -> None:
    """Say what is wrong, not what they sent.

    `str(pydantic.ValidationError)` embeds `input_value=…` for every failure, so a
    plain f-string echoed submitted request content straight back into the response
    body. It only ever reaches the requester — but this service deliberately
    redacts the messages a custom validator produces for exactly this reason, and
    one rule with an undocumented exception is how the first real leak happens.
    """
    response = client.post("/api/models/export-package", json={
        "graph": {
            "nodes": [{
                "id": "n1", "type": "layerNode", "position": {"x": 0.0, "y": 0.0},
                # layerType.parameters is missing, so this fails to parse — and the
                # value below must not come back in the message.
                "data": {"layerType": {"id": "linear", "name": "Linear"},
                         "parameters": {"api_key": "sk-do-not-echo-me"}},
            }],
            "edges": [],
        },
    })

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "sk-do-not-echo-me" not in detail
    assert "input_value" not in detail
    # Still actionable: the caller is told the field path and what is wrong with it.
    assert "layerType.parameters" in detail
    assert "missing" in detail


def test_a_malformed_graph_is_a_client_error_not_a_server_fault(client: TestClient) -> None:
    """{nodes: 'hello'} used to answer 500 "'str' object has no attribute 'get'".

    A client's malformed body reported as a server failure poisons error
    monitoring and leaks internals.
    """
    response = client.post("/api/models/validate", json={"graph": {"nodes": "hello", "edges": 42}})

    assert response.status_code == 400


# ── generate-code ───────────────────────────────────────────────────────────

def test_code_generation_refuses_a_graph_that_cannot_be_valid(client: TestClient) -> None:
    """Linear(in_features=-5) reported total_parameters:-16 next to complete:true.

    A summary claiming a model has minus sixteen parameters is not a display
    glitch, it is the graph being broken. Refusing is the only honest answer.
    """
    response = client.post("/api/models/generate-code", json={
        "graph": {"nodes": [_node("n1", "layerNode", "linear", "Linear",
                                  {"in_features": -5, "out_features": 4})],
                  "edges": []},
    })

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "in_features" in _messages(detail["errors"])


def test_an_empty_canvas_is_not_a_complete_model(client: TestClient) -> None:
    """{nodes: [], edges: []} returned 200 with `return x` stamped complete:true.

    `if not graph` never caught it: an empty-but-present dict is truthy.
    """
    response = client.post("/api/models/generate-code",
                           json={"graph": {"nodes": [], "edges": []}})

    assert response.status_code == 400
    assert "no layers" in response.json()["detail"]


def test_generated_metadata_carries_a_real_timestamp(client: TestClient) -> None:
    """Every response claimed it was generated on 2024-01-01T00:00:00Z."""
    response = client.post("/api/models/generate-code",
                           json={"graph": _runnable_graph(), "input_shape": [1, 3, 32, 32]})

    assert response.status_code == 200
    generated_at = response.json()["metadata"]["generated_at"]
    assert generated_at != "2024-01-01T00:00:00Z"
    assert generated_at.startswith("20") and generated_at.endswith("Z")


def test_a_comma_string_kernel_does_not_crash_the_summary(client: TestClient) -> None:
    """"3,3" is a form the canvas stores, and the summary multiplied its characters.

    `in_channels * out_channels * kernel_size[0]` took the character "3", so
    int*str repeated a string and str*str raised TypeError — a 500 on an ordinary
    layer.

    It is also the form codegen passed through verbatim, emitting
    `kernel_size='3,3'` — a file torch cannot build — so export refused a graph
    validation had already called fine. The generated kwarg has to be a tuple, and
    the graph has to export.
    """
    graph = _runnable_graph()
    graph["nodes"][1]["data"]["parameters"]["kernel_size"] = "3,3"

    response = client.post("/api/models/generate-code",
                           json={"graph": graph, "input_shape": [1, 3, 32, 32]})

    assert response.status_code == 200
    assert response.json()["summary"]["total_parameters"] > 0
    assert "kernel_size=(3, 3)" in response.json()["code"]

    export = client.post("/api/models/export-package",
                         json={"graph": graph, "input_shape": [1, 3, 32, 32]})

    assert export.status_code == 200, export.text


@pytest.mark.parametrize("kernel_size", [3, [3, 3], "3", "3,3"])
def test_every_stored_kernel_form_exports_and_generates_code_that_runs(
    client: TestClient, kernel_size: Any
) -> None:
    """The canvas stores one kernel field in four shapes; all four are one model.

    The editor writes `3`, the code importer writes `[3, 3]`, and a typed field
    arrives as the string the user typed — `"3"` or `"3,3"`. BOTH string forms
    reached nn.Conv2d as a str ("type must be tuple of ints, but got str"), so
    export refused them and shape inference reported a layer error, while validate
    called the same graph fine; `"3,3"` was additionally emitted verbatim into the
    generated file. Every plane is asserted, because pinning one is what let the
    planes disagree in the first place — and generating code is not the promise,
    generating code that RUNS is, so this execs it.
    """
    graph = _runnable_graph()
    graph["nodes"][1]["data"]["parameters"]["kernel_size"] = kernel_size

    validated = client.post("/api/models/validate",
                            json={"graph": graph, "input_shape": [1, 3, 32, 32]})
    assert validated.status_code == 200
    assert validated.json()["valid"], _messages(validated.json()["errors"])

    inferred = client.post("/api/models/infer-shapes",
                           json={"graph": graph, "input_shape": [1, 3, 32, 32]})
    assert inferred.status_code == 200
    assert inferred.json()["errors"] == []
    assert inferred.json()["shapes"]["c"] == [1, 8, 32, 32]

    export = client.post("/api/models/export-package",
                         json={"graph": graph, "input_shape": [1, 3, 32, 32]})
    assert export.status_code == 200, export.text

    response = client.post("/api/models/generate-code",
                           json={"graph": graph, "input_shape": [1, 3, 32, 32]})
    assert response.status_code == 200, response.text

    namespace: Dict[str, Any] = {}
    exec(compile(response.json()["code"], "<generated model.py>", "exec"), namespace)  # noqa: S102
    model = namespace["GeneratedModel"]()
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 32, 32))
    assert list(output.shape) == [1, 10]


# ── export-package ──────────────────────────────────────────────────────────

def test_export_refuses_a_model_that_cannot_run(client: TestClient) -> None:
    """The audit's exact chain: Linear(-5) exported a zip that died on first run.

    "RuntimeError: Trying to create tensor with negative dimension -5" is what a
    user got out of the product's flagship promise.
    """
    response = client.post("/api/models/export-package", json={
        "graph": {"nodes": [_node("n1", "layerNode", "linear", "Linear",
                                  {"in_features": -5, "out_features": 4})],
                  "edges": []},
    })

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")


def test_export_of_an_unknown_layer_is_a_client_error(client: TestClient) -> None:
    """Was a 500 carrying "layer operation 'FooBar' is not a supported operation"."""
    response = client.post("/api/models/export-package", json={
        "graph": {"nodes": [_node("n1", "layerNode", "foobar_layer", "FooBar", {})],
                  "edges": []},
    })

    assert response.status_code in (400, 422)


def test_a_runnable_model_still_exports_a_zip(client: TestClient) -> None:
    """The counterweight. A gate that refuses everything is not a fix."""
    response = client.post("/api/models/export-package",
                           json={"graph": _runnable_graph(), "input_shape": [1, 3, 32, 32]})

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    assert response.content[:2] == b"PK"


@pytest.mark.parametrize("layer_id,params,needle", [
    ("nn.nn.Conv2d", {"in_channels": 3, "out_channels": 16, "kernel_size": -3}, "kernel_size"),
    ("nn.nn.BatchNorm2d", {"num_features": 0}, "num_features"),
])
def test_double_prefixed_ids_reach_the_checks_for_every_gated_layer(
    client: TestClient, layer_id: str, params: dict, needle: str
) -> None:
    """Not just Linear — the same normalisation feeds Conv2d and BatchNorm2d."""
    response = client.post("/api/models/validate", json={
        "graph": {"nodes": [_node("n1", "layerNode", layer_id, layer_id, params)],
                  "edges": []},
    })

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False, f"{layer_id} bypassed the parameter checks"
    assert needle in _messages(body["errors"])


@pytest.mark.parametrize("layer_id", ["linear", "Linear", "nn.Linear", "nn.nn.Linear"])
def test_a_sound_model_still_validates_under_every_spelling(
    client: TestClient, layer_id: str
) -> None:
    """The counterweight: normalising harder must not start rejecting good models.

    Without this, the test above passes just as well for a validator that calls
    everything invalid.
    """
    response = client.post("/api/models/validate", json={
        "graph": {"nodes": [_node("n1", "layerNode", layer_id, "Linear",
                                  {"in_features": 10, "out_features": 5})],
                  "edges": []},
    })

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True, body["errors"]


@pytest.mark.parametrize("route", ["/api/models/export-onnx", "/api/models/verify"])
def test_a_layerless_canvas_is_refused_by_the_onnx_and_verify_planes_too(
    client: TestClient, route: str
) -> None:
    """Q-m3's gate has to hold on every plane that makes a claim about the model.

    The first pass gated generate-code, code/export and export-package. ONNX
    export has its own entry point and was still handing back ok:true with a
    downloadable .onnx containing a single Identity op; /verify was still
    answering ok:true, stage:"complete", layer_count:0. "Verified" is the
    strongest claim the product makes, and the builder gates Export Code on it.
    """
    response = client.post(route, json={"graph": {"nodes": [], "edges": []},
                                        "input_shape": [1, 10]})

    assert response.status_code == 400, response.text
    assert "no layers" in response.json()["detail"]


@pytest.mark.parametrize("route", ["/api/models/export-onnx", "/api/models/verify"])
def test_the_onnx_and_verify_planes_still_serve_a_real_model(
    client: TestClient, route: str
) -> None:
    """Counterweight to the gate above: a real one-layer model must still pass."""
    graph = {"nodes": [_node("in", "inputNode", "input", "Input", {}),
                       _node("n1", "layerNode", "linear", "Linear",
                             {"in_features": 10, "out_features": 5}),
                       _node("out", "outputNode", "output", "Output", {})],
             "edges": [{"id": "e1", "source": "in", "target": "n1"},
                       {"id": "e2", "source": "n1", "target": "out"}]}

    response = client.post(route, json={"graph": graph, "input_shape": [1, 10]})

    assert response.status_code == 200, response.text
    assert response.json()["ok"] is True, response.text
