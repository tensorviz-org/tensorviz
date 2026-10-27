"""Every parsed node says where in the source it came from.

The code->graph half of the IDE pivot (task.md M1, §6.1) is only useful if a node
can be anchored back to the statement that declares it: "click a node -> go to the
line" and "apply this canvas edit as a one-line source patch" are both impossible
without a span. Before M1 the parser recorded no positions at all (`grep lineno`
-> 0 hits) and computed `raw_call` only to throw it away.

The contract asserted here, for every node that is not a terminal:

  * ``_lineno``/``_end_lineno``/``_col_offset``/``_end_col_offset`` locate a REAL
    AST statement in the source that was parsed — not an approximate line;
  * ``_raw_call`` is ``ast.unparse`` of an expression inside that statement;
  * ``_source_file`` is exactly what the request said the file was.

Run: KMP_DUPLICATE_LIB_OK=TRUE <viz-python> -m pytest tests/test_code_parser_spans.py -q
"""

from __future__ import annotations

import ast


import pytest  # noqa: E402

from models.graph_models import CodeParseRequest  # noqa: E402
from utils.code_parser import CodeParserService  # noqa: E402


SOURCE_FILE = "models/fixtures.py"

# ── the four fixtures: declared layers, functional ops, a merge, control flow ──
MLP = """import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
"""

CNN = """import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = x.flatten(1)
        return self.fc(x)
"""

RESIDUAL = """import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 4)
        self.l2 = nn.Linear(4, 4)

    def forward(self, x):
        h = self.l1(x)
        return x + self.l2(h)
"""

OPAQUE = """import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        x = self.fc(x)
        if self.training:
            x = x.transpose(1, 2)
        return x
"""

FIXTURES = {"MLP": MLP, "CNN": CNN, "RESIDUAL": RESIDUAL, "OPAQUE": OPAQUE}


def _graph(code: str, input_shape):
    result = CodeParserService().parse_pytorch_code(
        CodeParseRequest(code=code, input_shape=input_shape, file=SOURCE_FILE)
    )
    assert result.get("graphs"), result
    graph = result["graphs"][0]["graph"]
    assert graph is not None, result["graphs"][0].get("error")
    return graph


def _body_nodes(graph):
    """Every node except the two synthetic terminals, which describe the tensor
    entering and leaving the model rather than a statement in the file."""
    return [n for n in graph.nodes if n.data.layerType.id not in ("input", "output")]


def _statement_at_span(code: str, parameters) -> ast.AST:
    """The AST node the recorded span points at, or None when it points at
    nothing — which is the failure this test exists to catch."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if (
            getattr(node, "lineno", None) == parameters["_lineno"]
            and getattr(node, "end_lineno", None) == parameters["_end_lineno"]
            and getattr(node, "col_offset", None) == parameters["_col_offset"]
            and getattr(node, "end_col_offset", None) == parameters["_end_col_offset"]
        ):
            return node
    return None


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_every_body_node_carries_a_usable_span(name):
    code = FIXTURES[name]
    shape = [2, 3, 8, 8] if name == "CNN" else [2, 4, 4]
    graph = _graph(code, shape)
    nodes = _body_nodes(graph)
    assert nodes, f"{name} parsed to terminals only"

    for node in nodes:
        parameters = node.data.parameters
        assert parameters.get("_source_file") == SOURCE_FILE, node.id

        lineno = parameters.get("_lineno")
        end_lineno = parameters.get("_end_lineno")
        assert isinstance(lineno, int) and isinstance(end_lineno, int), node.id
        assert lineno <= end_lineno, node.id
        assert isinstance(parameters.get("_col_offset"), int), node.id
        assert isinstance(parameters.get("_end_col_offset"), int), node.id

        statement = _statement_at_span(code, parameters)
        assert statement is not None, (
            f"{node.id}: span {lineno}:{parameters['_col_offset']}-"
            f"{end_lineno}:{parameters['_end_col_offset']} matches no AST node"
        )

        raw_call = parameters.get("_raw_call")
        assert isinstance(raw_call, str) and raw_call, node.id
        unparsed = {ast.unparse(inner) for inner in ast.walk(statement)}
        assert raw_call in unparsed, (
            f"{node.id}: _raw_call {raw_call!r} is not an expression of the "
            f"statement its span names"
        )


def test_a_declared_layer_points_at_its_assignment_and_its_constructor_call():
    graph = _graph(MLP, [2, 8])
    fc1 = next(n for n in graph.nodes if n.id == "fc1")
    # `self.fc1 = nn.Linear(8, 16)` is line 7 of MLP (1-indexed).
    assert fc1.data.parameters["_lineno"] == 7
    assert fc1.data.parameters["_end_lineno"] == 7
    assert fc1.data.parameters["_raw_call"] == "nn.Linear(8, 16)"
    # The span addresses the whole statement — what a source edit must replace —
    # while the raw call is the expression the parameters were read from.
    assert MLP.splitlines()[6].strip() == "self.fc1 = nn.Linear(8, 16)"


def test_an_op_written_in_forward_points_at_its_forward_statement():
    graph = _graph(CNN, [2, 3, 8, 8])
    flatten = next(n for n in graph.nodes if n.data.layerType.name == "Flatten")
    assert flatten.data.parameters["_raw_call"] == "x.flatten(1)"
    assert CNN.splitlines()[flatten.data.parameters["_lineno"] - 1].strip() == "x = x.flatten(1)"


def test_a_node_inside_a_compound_statement_is_marked_as_control_flow():
    """`if self.training:` means the transpose does not always run. A node that
    hides that is worse than no node: the graph would claim a shape change the
    model may never perform."""
    graph = _graph(OPAQUE, [2, 4, 4])
    opaque = next(n for n in graph.nodes if n.data.layerType.id == "python")
    assert opaque.data.parameters["_control_flow"] is True
    assert opaque.data.parameters["_raw_call"] == "x.transpose(1, 2)"

    # The layer declared in __init__ runs unconditionally and says so by omission.
    fc = next(n for n in graph.nodes if n.id == "fc")
    assert fc.data.parameters.get("_control_flow") is None


def test_no_file_means_no_source_file_key_rather_than_a_null():
    """The web paste box does not know a filename. Writing None would make every
    consumer test for it; the honest form is to say nothing."""
    result = CodeParserService().parse_pytorch_code(
        CodeParseRequest(code=MLP, input_shape=[2, 8])
    )
    fc1 = next(n for n in result["graphs"][0]["graph"].nodes if n.id == "fc1")
    assert "_source_file" not in fc1.data.parameters
    assert isinstance(fc1.data.parameters["_lineno"], int)
