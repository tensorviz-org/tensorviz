"""F5-TTS is the worked example of modular codegen, so pin what it produces.

`server_v2.generate_code` cites this model by name: "factor repeated blocks into reusable
xN modules (e.g. F5-TTS -> DiTBlock x22 via nn.ModuleList)". If the export ever flattens
back to 290 inline layers, modular codegen has regressed to nothing and this is the test
that says so.
"""


import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import server_v2  # noqa: E402


@pytest.fixture(scope="module")
def generated():
    client = TestClient(server_v2.app)
    body = client.post("/api/templates/f5_tts/instantiate", json={"config": {}}).json()
    graph = body.get("graph", body)
    flat = {
        "nodes": graph.get("root_nodes") or graph["nodes"],
        "edges": graph.get("root_edges") or graph["edges"],
    }
    response = client.post(
        "/api/models/generate-code",
        json={"graph": flat, "input_shape": [1, 3, 224, 224], "class_name": "F5TTS"},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_the_repeated_dit_stack_is_one_module_list_of_22(generated):
    code = generated["code"]
    assert "nn.ModuleList([DitBlock() for _ in range(22)])" in code, code[:2000]


def test_the_export_is_complete(generated):
    # No library-private composite the generator had to leave as a stub.
    assert generated["complete"] is True
    assert generated["unsupported"] == []


def test_the_custom_ops_ship_their_implementations(generated):
    """grn / sinusoidal_pos_embedding are not torch.nn ops, so the file must define them.

    This is exactly what made the template look unbuildable to a filter that asked the op
    registry: these resolve in the GENERATOR, not the registry.
    """
    code = generated["code"]
    for cls in ("class GRN(nn.Module)", "class SinusoidalPosEmbedding(nn.Module)",
                "class RotaryEmbedding(nn.Module)", "class AdaLayerNorm(nn.Module)"):
        assert cls in code, cls


def test_the_generated_file_imports_and_builds(generated, tmp_path):
    import importlib.util

    path = tmp_path / "f5.py"
    path.write_text(generated["code"])
    spec = importlib.util.spec_from_file_location("f5_generated", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # the promise is runnable PyTorch
    net = module.F5TTS()
    assert len(net.dit_block) == 22
