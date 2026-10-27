"""A captured boundary is not a captured implementation, or an empty module."""
from pathlib import Path

import torch.nn as nn

from services.ingest.document import to_document
from services.ingest.introspect import build_tree
from services.ingest.normalise import build_graph


class CustomBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8)

    def forward(self, x):
        raise AssertionError("normalization must not execute custom Python")


def captured_graph(depth):
    model = nn.Module()
    model.encoder = nn.Module()
    model.encoder.layers = nn.ModuleList([CustomBlock(), CustomBlock()])
    intro = build_tree(model, depth)
    io = {path: {"in": [1, 10, 8], "out": [1, 10, 8]}
          for path in ("encoder", "encoder.layers.0")}
    return build_graph("Net", intro, io, {}, set(), set(io), [1, 10, 8],
                       roots=[str(Path(__file__).parent)])[0]


def test_depth_limit_retains_measured_repeat_boundary_without_inventing_contents():
    graph = captured_graph(2)
    block = next(n for n in graph["nodes"] if n["id"] == "encoder.layers.0")
    assert block["type"] == "moduleNode"
    assert block["data"]["moduleCapture"] == {"status": "depth-limited", "maxDepth": 2}
    assert block["data"]["moduleLayerCount"] == 0  # captured count, never total layers
    assert block["data"]["repeatCount"] == 2
    assert block["data"]["parentId"] == "encoder"
    params = block["data"]["parameters"]
    assert params["_out_shape"] == "1x10x8"
    assert params["_shape_provenance"] == "observed"
    assert params["_class"] == "CustomBlock"
    assert params["_import_path"] == "test_ingest_capture_bounds"
    assert not any(n["id"].startswith("encoder.layers.0.") for n in graph["nodes"])
    assert not any(".proj" in str(e) for e in graph["edges"])
    doc = to_document("Net", graph)
    assert doc.view.layout["node_data"][block["id"]]["moduleCapture"] == block["data"]["moduleCapture"]


def test_deeper_capture_and_true_leaves_are_not_depth_limited():
    graph = captured_graph(3)
    block = next(n for n in graph["nodes"] if n["id"] == "encoder.layers.0")
    leaf = next(n for n in graph["nodes"] if n["id"] == "encoder.layers.0.proj")
    assert block["data"]["moduleLayerCount"] == 1
    assert "moduleCapture" not in block["data"]
    assert "moduleCapture" not in leaf["data"]
    assert leaf["data"]["parameters"]["_shape_provenance"] == "unreached"
    assert block["data"]["repeatCount"] == 2
