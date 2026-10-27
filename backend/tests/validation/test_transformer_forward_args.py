"""Regression tests: transformer/attention modules need more than one
positional forward() argument.

These guard the bug where shape verification called `layer(x)` with a single
tensor and crashed on modules like nn.TransformerDecoderLayer with
"forward() missing 1 required positional argument: 'memory'". The fix routes
every verification forward pass through core.layer_builder.call_module, which
synthesizes the extra positional args from the primary tensor.
"""


import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient


from api.routers import validate  # noqa: E402
from core.layer_builder import build_layer, call_module  # noqa: E402
from services.verification_service import verify_model  # noqa: E402


# ── call_module unit coverage ──────────────────────────────────────────────

def test_call_module_decoder_layer_supplies_memory():
    """The exact reported failure: TransformerDecoderLayer.forward(tgt, memory)."""
    layer = build_layer(
        "TransformerDecoderLayer",
        {"d_model": 768, "nhead": 12, "dim_feedforward": 3072, "dropout": 0.1, "batch_first": True},
    )
    out = call_module(layer, torch.randn(1, 10, 768))
    assert list(out.shape) == [1, 10, 768]


def test_call_module_multihead_attention_unwraps_tuple():
    """MultiheadAttention needs (q, k, v) and returns a tuple — must unwrap."""
    layer = build_layer("MultiheadAttention", {"embed_dim": 768, "num_heads": 12})
    out = call_module(layer, torch.randn(10, 1, 768))
    assert isinstance(out, torch.Tensor)
    assert list(out.shape) == [10, 1, 768]


def test_call_module_single_input_layer_unchanged():
    """Ordinary single-input layers must keep working through the helper."""
    layer = build_layer("Linear", {"in_features": 768, "out_features": 10})
    out = call_module(layer, torch.randn(1, 10, 768))
    assert list(out.shape) == [1, 10, 10]


# ── full verifier path ──────────────────────────────────────────────────────

def _layer_node(node_id, params):
    return {
        "id": node_id,
        "type": "layerNode",
        "position": {"x": 0, "y": 0},
        "data": {
            "layerType": {"id": "transformerdecoderlayer", "name": "TransformerDecoderLayer", "parameters": []},
            "parameters": params,
        },
    }


def _passthrough_node(node_id, node_type):
    return {
        "id": node_id,
        "type": node_type,
        "position": {"x": 0, "y": 0},
        "data": {"layerType": {"id": node_type, "name": node_type, "parameters": []}, "parameters": {}},
    }


def test_verify_model_decoder_stack_has_no_forward_error():
    """A stack of TransformerDecoderLayers (the vision-transformer graph in the
    bug report) verifies cleanly instead of raising a missing-'memory' error."""
    params = {"d_model": 768, "nhead": 12, "dim_feedforward": 3072, "dropout": 0.1, "batch_first": True}
    graph = {
        "nodes": [
            _passthrough_node("in", "inputNode"),
            _layer_node("d1", params),
            _layer_node("d2", params),
            _passthrough_node("out", "outputNode"),
        ],
        "edges": [
            {"id": "e1", "source": "in", "target": "d1"},
            {"id": "e2", "source": "d1", "target": "d2"},
            {"id": "e3", "source": "d2", "target": "out"},
        ],
    }
    result = verify_model(graph, input_shape=[1, 10, 768], run_forward=True)
    assert result["errors"] == []
    assert result["output_shape"] == [1, 10, 768]


# ── validate_shapes router path ──────────────────────────────────────────────

def test_validate_shapes_router_handles_decoder_layer():
    app = FastAPI()
    app.include_router(validate.router, prefix="/api")
    client = TestClient(app)
    payload = {
        "nodes": [
            {
                "id": "input",
                "type": "inputNode",
                "position": {"x": 0, "y": 0},
                "data": {"layerType": {"id": "input", "name": "Input"}, "parameters": {"shape": "1,10,768"}},
            },
            {
                "id": "dec",
                "type": "layerNode",
                "position": {"x": 100, "y": 0},
                "data": {
                    "layerType": {"id": "transformerdecoderlayer", "name": "nn.TransformerDecoderLayer"},
                    "parameters": {"d_model": 768, "nhead": 12, "dim_feedforward": 3072, "batch_first": True},
                },
            },
            {
                "id": "output",
                "type": "outputNode",
                "position": {"x": 200, "y": 0},
                "data": {"layerType": {"id": "output", "name": "Output"}, "parameters": {}},
            },
        ],
        "edges": [
            {"id": "e1", "source": "input", "target": "dec"},
            {"id": "e2", "source": "dec", "target": "output"},
        ],
        "input_shape": [1, 10, 768],
    }
    response = client.post("/api/validate_shapes", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["shapes"].get("dec") == [1, 10, 768]
