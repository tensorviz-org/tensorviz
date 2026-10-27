"""The getting-started recipe must build a model that validates and runs.

The First-Run audit's second critical was that it could not: the tutorial set
exactly one value — `out_features=10` — and never mentioned the two that decide
whether the model works, the Input node's shape and the Linear layer's
`in_features`. On the app's defaults, Flatten emits 197,136 features against
Linear's default 512, so the failure was not occasional, it was arithmetic:
"mat1 and mat2 shapes cannot be multiplied" every time, for everyone who
followed the page exactly.

The rewritten recipe adds the Input node at 1,3,32,32 and sets
`in_features=3600`, with the arithmetic on the page: 32x32 through
Conv2d(3, 16, kernel_size=3) gives 30x30, MaxPool2d(2) halves that to 15x15, and
16*15*15 = 3600 — which is the number the page's own sample output always showed
and no step ever produced.

These tests hold the page and the product to that number. `test_the_old_recipe_…`
is the counterweight: it asserts that the value the tutorial USED to leave in
place really does fail, so a future change that made 512 work would tell us that
this test has stopped meaning anything.
"""

import html
import re
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient


import server_v2  # noqa: E402

GUIDE = Path(__file__).parent.resolve().parents[2] / "docs" / "getting-started.html"
INPUT_SHAPE = [1, 3, 32, 32]
FLATTENED = 16 * 15 * 15  # 3600


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


def _recipe(in_features: int) -> Dict[str, Any]:
    """The tutorial's chain, with in_features as the one thing under test."""
    nodes = [
        _node("i", "inputNode", "input", "Input", {"shape": "1,3,32,32"}),
        _node("c", "layerNode", "conv2d", "Conv2d",
              {"in_channels": 3, "out_channels": 16, "kernel_size": 3}),
        _node("r", "layerNode", "relu", "ReLU", {}),
        _node("p", "layerNode", "maxpool2d", "MaxPool2d", {"kernel_size": 2}),
        _node("f", "layerNode", "flatten", "Flatten", {}),
        _node("l", "layerNode", "linear", "Linear",
              {"in_features": in_features, "out_features": 10}),
    ]
    chain = ["i", "c", "r", "p", "f", "l"]
    edges = [
        {"id": f"e{index}", "source": source, "target": target}
        for index, (source, target) in enumerate(zip(chain, chain[1:]))
    ]
    return {"nodes": nodes, "edges": edges}


def test_the_recipe_validates(client: TestClient) -> None:
    response = client.post(
        "/api/models/validate",
        json={"graph": _recipe(FLATTENED), "input_shape": INPUT_SHAPE},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True, body["errors"]
    # And with an Input node present, no warning about an assumed shape.
    assert not any("No Input node" in w["message"] for w in body["warnings"]), body["warnings"]


def test_the_recipe_runs_a_real_forward_pass(client: TestClient) -> None:
    """Validation is structural; this is torch actually running the model."""
    response = client.post(
        "/api/models/verify",
        json={"graph": _recipe(FLATTENED), "input_shape": INPUT_SHAPE, "run_forward": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True, body.get("errors")
    assert body["output_shape"] == [1, 10], body["output_shape"]


def test_the_recipe_exports_a_package(client: TestClient) -> None:
    response = client.post(
        "/api/models/export-package",
        json={"graph": _recipe(FLATTENED), "input_shape": INPUT_SHAPE},
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    assert response.content[:2] == b"PK"


def test_the_old_recipe_still_fails_for_the_reason_the_audit_gave(client: TestClient) -> None:
    """The counterweight.

    512 is Linear's palette default and what the old tutorial left in place. If
    this ever starts passing, the three tests above have stopped meaning anything
    and the number in the guide needs revisiting rather than trusting.
    """
    response = client.post(
        "/api/models/verify",
        json={"graph": _recipe(512), "input_shape": INPUT_SHAPE, "run_forward": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert any("multiplied" in error["message"] for error in body["errors"]), body["errors"]


def test_the_current_source_example_opens_and_runs(client: TestClient) -> None:
    """Run the exact source readers copy, including its declared trace shape."""
    guide = GUIDE.read_text(encoding="utf-8")
    example = re.search(r'<section id="example">.*?<code>(.*?)</code>', guide, re.S)
    assert example, "the source example must remain in the installation guide"
    source = html.unescape(example.group(1))
    declared = re.search(r"# tensorviz: input=([0-9,]+)", source)
    assert declared, "the example must declare its input for manual Validate"
    shape = [int(value) for value in declared.group(1).split(',')]
    response = client.post("/api/ide/project", json={"file": "model.py", "cls": "Net", "source": source})
    assert response.status_code == 200, response.text
    assert response.json()["traced"] is False
    import torch
    namespace = {}
    exec(compile(source, "getting-started-example.py", "exec"), namespace)
    with torch.no_grad():
        output = namespace["Net"]()(torch.zeros(*shape))
    assert list(output.shape) == [2, 2]
