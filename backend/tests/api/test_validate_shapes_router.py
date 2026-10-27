"""Integration tests for /api/validate_shapes robustness and compatibility."""


from fastapi import FastAPI
from fastapi.testclient import TestClient


from api.routers import validate  # noqa: E402


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(validate.router, prefix="/api")
    return TestClient(app)


def test_validate_shapes_supports_nn_prefixed_layer_names():
    client = _client()
    payload = {
        "nodes": [
            {
                "id": "input",
                "type": "inputNode",
                "position": {"x": 0, "y": 0},
                "data": {"layerType": {"id": "input", "name": "Input"}, "parameters": {"shape": "1,3,32,32"}},
            },
            {
                "id": "conv1",
                "type": "layerNode",
                "position": {"x": 100, "y": 0},
                "data": {
                    "layerType": {"id": "conv2d", "name": "nn.Conv2d"},
                    "parameters": {"in_channels": 3, "out_channels": 8, "kernel_size": 3, "padding": 1},
                },
            },
            {
                "id": "relu1",
                "type": "layerNode",
                "position": {"x": 200, "y": 0},
                "data": {"layerType": {"id": "relu", "name": "nn.ReLU"}, "parameters": {}},
            },
            {
                "id": "output",
                "type": "outputNode",
                "position": {"x": 300, "y": 0},
                "data": {"layerType": {"id": "output", "name": "Output"}, "parameters": {}},
            },
        ],
        "edges": [
            {"id": "e1", "source": "input", "target": "conv1"},
            {"id": "e2", "source": "conv1", "target": "relu1"},
            {"id": "e3", "source": "relu1", "target": "output"},
        ],
        "input_shape": [1, 3, 32, 32],
    }
    response = client.post("/api/validate_shapes", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "valid"
    assert body["shapes"]["conv1"] == [1, 8, 32, 32]
    assert body["shapes"]["relu1"] == [1, 8, 32, 32]


def test_validate_shapes_ignores_module_nodes_without_layer_type():
    client = _client()
    payload = {
        "nodes": [
            {
                "id": "input",
                "type": "inputNode",
                "position": {"x": 0, "y": 0},
                "data": {"layerType": {"id": "input", "name": "Input"}, "parameters": {"shape": "1,3,32,32"}},
            },
            {
                "id": "conv1",
                "type": "layerNode",
                "position": {"x": 100, "y": 0},
                "data": {
                    "layerType": {"id": "conv2d", "name": "Conv2d"},
                    "parameters": {"in_channels": 3, "out_channels": 8, "kernel_size": 3, "padding": 1},
                },
            },
            {
                "id": "module:demo",
                "type": "moduleNode",
                "position": {"x": 160, "y": 0},
                "data": {"parameters": {}, "isGroup": True},
            },
            {
                "id": "output",
                "type": "outputNode",
                "position": {"x": 300, "y": 0},
                "data": {"layerType": {"id": "output", "name": "Output"}, "parameters": {}},
            },
        ],
        "edges": [
            {"id": "e1", "source": "input", "target": "conv1"},
            {"id": "e2", "source": "conv1", "target": "output"},
        ],
        "input_shape": [1, 3, 32, 32],
    }
    response = client.post("/api/validate_shapes", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "valid"
    assert "module:demo" not in body["shapes"]
    assert body["shapes"]["conv1"] == [1, 8, 32, 32]

