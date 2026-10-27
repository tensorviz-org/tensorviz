"""F13: real tensors -> normalization -> source projection -> browser fixture.

The frontend consumes the same checked fixture, so shape/count/byte transport
cannot pass on two independently invented frontend and backend examples.
"""
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from models.graph_models import CodeParseRequest
from services.ide.projection import project
from services.ingest.service import _introspect_and_trace
from services.ingest.trace import trace_model
from utils.code_parser import CodeParserService

SOURCE = '''import torch.nn as nn
class GoodNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 2)
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
'''
FIXTURE = Path(__file__).parent.resolve().parents[2] / "canvas/src/test/fixtures/metricProjections.json"


def goodnet_projection(batch, dtype):
    namespace = {}
    exec(SOURCE, namespace)
    model = namespace["GoodNet"]().to(dtype=dtype)
    graph, diagnostics, error = _introspect_and_trace(
        "GoodNet", model, 3, ((torch.ones(batch, 4, dtype=dtype),), {}), {},
        tier="real", strict_recipe=True, recipe_provenance="explicit",
    )
    assert error is None
    parsed = CodeParserService().parse_pytorch_code(CodeParseRequest(code=SOURCE, file="goodnet.py"))
    ast = next(item["graph"] for item in parsed["graphs"] if item["name"] == "GoodNet")
    merged = project({"nodes": [node.model_dump() for node in ast.nodes],
                      "edges": [edge.model_dump() for edge in ast.edges]},
                     {"protocol_version": 1, "result": {"graph": graph, "diagnostics": diagnostics.model_dump()},
                      "spans": {}, "error": None},
                     cls="GoodNet", rel_file="goodnet.py", source=SOURCE)
    document = merged.document.model_dump()
    # Source locations depend on the installed Torch package. No source metadata
    # contributes to this contract fixture; production portable privacy is tested in JS.
    for node in document["semantic"]["nodes"]:
        for key in ("_source_file", "_source_line", "_decl_file", "_decl_line"):
            node["params"].pop(key, None)
    return document


def fixture_cases():
    return {name: goodnet_projection(batch, dtype) for name, batch, dtype in (
        ("batch2", 2, torch.float32), ("batch3", 3, torch.float32),
        ("double", 3, torch.float64), ("empty", 0, torch.float32),
    )}


@pytest.mark.parametrize("batch,dtype,size", [(2, torch.float32, 4), (3, torch.float32, 4),
                                            (3, torch.float64, 8), (0, torch.float32, 4)])
def test_exact_measurements_survive_real_trace_and_source_projection(batch, dtype, size):
    document = goodnet_projection(batch, dtype)
    data = document["view"]["layout"]["node_data"]
    assert [data[node]["parameterCount"] for node in ("fc1", "relu", "fc2")] == [20, 0, 10]
    for node, width in (("fc1", 4), ("relu", 4), ("fc2", 2), ("output", 2)):
        assert data[node]["outputTensor"] == {
            "shape": [batch, width], "dtype": str(dtype).removeprefix("torch."), "logicalBytes": batch * width * size,
        }


def test_browser_fixture_is_real_backend_projection():
    assert json.loads(FIXTURE.read_text()) == fixture_cases()


@pytest.mark.parametrize("value", [torch.tensor(1., dtype=torch.float64), torch.empty(0, 3, dtype=torch.int16),
                                  torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.bfloat16)])
def test_scalar_empty_and_non_float32_outputs(value):
    result = trace_model(nn.Identity(), (value,), tier="real")
    assert result.io[""]["output_tensor"] == {
        "shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch."),
        "logicalBytes": value.numel() * value.element_size(),
    }
    graph, _, error = _introspect_and_trace("ScalarOrEmpty", nn.Sequential(nn.Identity()), 2,
                                           ((value,), {}), {}, tier="real", strict_recipe=True)
    assert error is None
    output = next(node["data"] for node in graph["nodes"] if node["id"] == "output_node")
    assert output["outputTensor"] == result.io[""]["output_tensor"]
    assert output["parameters"]["_shape_provenance"] == "observed"
    if value.ndim == 0:
        assert output["parameters"]["_out_shape"] == []


@pytest.mark.parametrize("output", [None, (), (torch.ones(2),), {"a": torch.ones(2), "b": torch.ones(3)}])
def test_non_tensor_and_container_outputs_do_not_claim_a_complete_tensor(output):
    class Returns(nn.Module):
        def forward(self, x):
            return output
    result = trace_model(Returns(), (torch.ones(1),), tier="real")
    assert "output_tensor" not in result.io[""]


@pytest.mark.parametrize("fail", [False, True])
def test_repeated_or_failed_call_cannot_reuse_an_earlier_measurement(fail):
    class Layer(nn.Module):
        def forward(self, x):
            if fail and x.shape[0] == 3:
                raise ValueError("second call failed")
            return x
    class Reuses(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = Layer()
            self.unreached = nn.Identity()
        def forward(self, x):
            self.layer(x[:2])
            return self.layer(x)
    result = trace_model(Reuses(), (torch.ones(3, 4),), tier="real")
    assert "output_tensor" not in result.io["layer"]
    assert "unreached" not in result.io
    if fail:
        assert result.error
        assert result.io["layer"]["out"] is None
        assert "output_tensor" not in result.io[""]


if __name__ == "__main__":
    # Intentional regeneration, checked by pytest on every run.
    FIXTURE.write_text(json.dumps(fixture_cases(), indent=2) + "\n")
