"""A layer named `input` or `output` is a layer, not the graph's terminal.

`self.output = nn.Linear(hidden, classes)` and `self.input = nn.Embedding(...)`
are ordinary PyTorch. The parser used to give its two synthetic terminals the
hardcoded ids "input"/"output" — the same namespace a node id is drawn from,
since a node id IS the attribute that declared the layer. So the user's layer
and the terminal became one node: the canvas keys shapes by id, the layer was
dropped from the drawing, and the toolbar went on counting it ("7 layers" over
six cards). Nothing said so — no error, no diagnostic, no placeholder.

The graph is the product's claim to be the model's ground truth, so a layer
that silently vanishes from it is the worst class of defect available here.

What is asserted:

  * ids are unique in every parsed graph, which is the invariant that was broken;
  * a layer named `output`/`input` still appears, with its own op;
  * the terminals still exist, and still bound the chain;
  * the ordinary case is untouched — no collision means the ids stay "input"
    and "output", so nothing downstream that reads them changes;
  * the shipped example that triggers it (examples/graph_ui/branches.py) draws
    every layer it declares.

Run: KMP_DUPLICATE_LIB_OK=TRUE <viz-python> -m pytest tests/test_code_parser_terminal_collision.py -q
"""

from __future__ import annotations

import os

import pytest


from models.graph_models import CodeParseRequest  # noqa: E402
from utils.code_parser import CodeParserService  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", ".."))


def _graphs(code: str):
    result = CodeParserService().parse_pytorch_code(CodeParseRequest(code=code))
    assert result.get("graphs"), result
    return {entry["name"]: entry for entry in result["graphs"]}


def _graph(code: str, name: str):
    entry = _graphs(code)[name]
    assert entry["graph"] is not None, entry.get("error")
    return entry["graph"]


def _ids(graph):
    return [node.id for node in graph.nodes]


def _terminal(graph, node_type: str) -> str:
    matches = [node.id for node in graph.nodes if node.type == node_type]
    assert len(matches) == 1, f"expected one {node_type}, got {matches}"
    return matches[0]


COLLIDING = """
import torch.nn as nn


class NamedOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 16)
        self.output = nn.Sigmoid()

    def forward(self, x):
        return self.output(self.fc(x))


class NamedInput(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(32, 32)
        self.fc = nn.Linear(32, 16)

    def forward(self, x):
        return self.fc(self.input(x))


class NamedNeither(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 16)
        self.final = nn.Sigmoid()

    def forward(self, x):
        return self.final(self.fc(x))
"""


@pytest.mark.parametrize("cls", ["NamedOutput", "NamedInput", "NamedNeither"])
def test_node_ids_are_unique(cls: str) -> None:
    # The invariant, stated plainly. Two nodes with one id is data loss whatever
    # the cause, and every consumer downstream — shapes, spans, codegen — keys
    # off the id, so none of them can tell the two apart afterwards.
    ids = _ids(_graph(COLLIDING, cls))
    assert len(ids) == len(set(ids)), f"{cls} has duplicate node ids: {ids}"


def test_a_layer_named_output_is_drawn_and_wired() -> None:
    graph = _graph(COLLIDING, "NamedOutput")
    exit_id = _terminal(graph, "outputNode")

    sigmoid = next((node for node in graph.nodes if node.id == "output"), None)
    assert sigmoid is not None, f"the nn.Sigmoid is missing from {_ids(graph)}"
    assert sigmoid.type == "layerNode"
    # The terminal yields the name, because a layer id is the attribute that
    # declared it and renaming that breaks reveal-in-editor and every edit.
    assert exit_id != "output"

    pairs = {(edge.source, edge.target) for edge in graph.edges}
    assert ("fc", "output") in pairs, pairs
    assert ("output", exit_id) in pairs, pairs


def test_a_layer_named_input_is_drawn_and_wired() -> None:
    graph = _graph(COLLIDING, "NamedInput")
    entry_id = _terminal(graph, "inputNode")

    linear = next((node for node in graph.nodes if node.id == "input"), None)
    assert linear is not None, f"the nn.Linear is missing from {_ids(graph)}"
    assert linear.type == "layerNode"
    assert entry_id != "input"

    pairs = {(edge.source, edge.target) for edge in graph.edges}
    assert (entry_id, "input") in pairs, pairs
    assert ("input", "fc") in pairs, pairs


def test_the_ordinary_case_keeps_the_ordinary_ids() -> None:
    # No collision, no rename. Consumers that read the literal "input"/"output"
    # — crossfile bridging, code_apply's boundaries, the shape engine — see
    # exactly what they saw before, so this fix costs them nothing.
    graph = _graph(COLLIDING, "NamedNeither")
    assert _terminal(graph, "inputNode") == "input"
    assert _terminal(graph, "outputNode") == "output"
    assert _ids(graph) == ["input", "fc", "final", "output"]


def test_the_shipped_example_draws_every_layer_it_declares() -> None:
    # examples/graph_ui/branches.py declares `self.output = nn.Sigmoid()`, so it
    # reproduced this on a clean install: six cards drawn under a toolbar that
    # said seven. It ships, so a user can hit it in under a minute.
    path = os.path.join(REPO, "examples", "graph_ui", "branches.py")
    if not os.path.exists(path):
        pytest.skip(f"{path} is not in this checkout")
    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    assert "self.output" in source, "the example no longer triggers the collision"

    for name, entry in _graphs(source).items():
        graph = entry["graph"]
        if graph is None:
            continue
        ids = _ids(graph)
        assert len(ids) == len(set(ids)), f"{name} has duplicate node ids: {ids}"
        if "self.output" in source and name == "ParallelBranches":
            assert "output" in ids, f"the declared Sigmoid is missing from {ids}"
