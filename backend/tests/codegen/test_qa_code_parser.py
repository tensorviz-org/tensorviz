"""Regression suite for the Code -> canvas parser (PRs #46, #60).

The parser (utils/code_parser.CodeParserService) turns hand-written PyTorch
``nn.Module`` source into a canvas Graph; the promise the PRs make is that the
graph then round-trips back through CodeGenerationService.generate_flat_code
into runnable code that reproduces the original topology. These tests exercise
the four model families the feature targets (MLP, Sequential, CNN with
functional ops, RNN), assert node ops/order + resolved constructor params, and
drive the full parse -> export -> exec round-trip so a regression that silently
drops a layer, mis-sizes a Linear, or scrambles the wiring FAILS here.

Two confirmed bugs are pinned with xfail(strict=False): they keep the suite
GREEN today and auto-flip to XPASS the moment the parser is fixed.

Run:
  KMP_DUPLICATE_LIB_OK=TRUE <viz-python> -m pytest tests/test_qa_code_parser.py -q
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # must precede torch import
os.environ.setdefault("AUTH_DISABLED", "true")


import pytest
import torch

from models.graph_models import CodeParseRequest  # noqa: E402
from utils.code_parser import CodeParserService  # noqa: E402
from services.code_generation import CodeGenerationService  # noqa: E402
from core.graph_input_safety import UnsafeGraphInput  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────────
def _parse(code: str, input_shape):
    """Full parser result dict ({success, graphs, imports, ...})."""
    return CodeParserService().parse_pytorch_code(
        CodeParseRequest(code=code, input_shape=input_shape)
    )


def _graph(code: str, input_shape):
    """The first parsed pydantic Graph (nodes are Node objects)."""
    res = _parse(code, input_shape)
    assert res.get("graphs"), res
    graph = res["graphs"][0]["graph"]
    assert graph is not None, res["graphs"][0].get("error")
    return graph


def _names(graph):
    return [n.data.layerType.name for n in graph.nodes]


def _node(graph, node_id):
    return next(n for n in graph.nodes if n.id == node_id)


def _params(node):
    """A node's SEMANTIC parameters. `_`-prefixed keys are provenance (source
    file, span, raw call) that every plane strips before torch or codegen sees
    them, so they are not part of what these tests are about."""
    return {k: v for k, v in node.data.parameters.items() if not k.startswith("_")}


def _edges(graph):
    return [(e.source, e.target) for e in graph.edges]


def _run_flat(graph, input_shape, class_name):
    """parse-result Graph -> generate_flat_code -> exec -> forward a dummy
    tensor. Returns the output tensor's shape as a tuple. Uses eval() so the
    BatchNorm/Dropout stats path (and batch-size-1 forwards) don't blow up."""
    model_py = CodeGenerationService().generate_flat_code(graph, input_shape, class_name)
    ns: dict = {}
    exec(compile(model_py, "<generated model.py>", "exec"), ns)  # noqa: S102 - code under test
    model = ns[class_name]()
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(*input_shape))
    assert torch.is_tensor(out), f"exported model returned {type(out).__name__}"
    return tuple(out.shape)


# ── model fixtures (source strings) ──────────────────────────────────────
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

SEQ = """import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )
    def forward(self, x):
        return self.net(x)
"""

CNN = """import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(4096, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = x.flatten(1)
        return self.fc(x)
"""

RNN = """import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(10, 20, batch_first=True)
        self.fc = nn.Linear(20, 5)
    def forward(self, x):
        x, _ = self.lstm(x)
        return self.fc(x)
"""

# forward whose FIRST statement is a functional/method op (before any self.X).
LEADING_FUNC = """import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(12, 4)
    def forward(self, x):
        x = x.flatten(1)
        return self.fc(x)
"""


# ── core: node ops / order / resolved params ─────────────────────────────
def test_mlp_positional_args_resolve_to_real_param_names():
    """PR #60: positional constructor args are mapped to their real signature
    names, so nn.Linear(8, 16) yields in_features/out_features (not arg_0/arg_1)
    and the re-exported layer is a valid nn.Linear. The full chain must also
    round-trip and run."""
    graph = _graph(MLP, [1, 8])
    assert _names(graph) == ["Input", "Linear", "ReLU", "Linear", "Output"]
    assert _params(_node(graph, "fc1")) == {"in_features": 8, "out_features": 16}
    assert _params(_node(graph, "fc2")) == {"in_features": 16, "out_features": 4}
    assert _edges(graph) == [
        ("input", "fc1"), ("fc1", "act"), ("act", "fc2"), ("fc2", "output"),
    ]
    assert _run_flat(graph, [1, 8], "MlpModel") == (1, 4)


def test_sequential_children_expand_and_wire_in_series():
    """PR #46: an nn.Sequential is expanded into one node per child (net_0,
    net_1, net_2) wired in series, instead of collapsing to a single opaque
    node. Losing the expansion would drop the ReLU and mis-order the Linears."""
    graph = _graph(SEQ, [1, 8])
    assert _names(graph) == ["Input", "Linear", "ReLU", "Linear", "Output"]
    child_ids = [n.id for n in graph.nodes if n.id.startswith("net_")]
    assert child_ids == ["net_0", "net_1", "net_2"]
    edges = _edges(graph)
    assert ("net_0", "net_1") in edges and ("net_1", "net_2") in edges
    assert ("input", "net_0") in edges and ("net_2", "output") in edges
    assert _run_flat(graph, [1, 8], "SeqModel") == (1, 4)


def test_cnn_functional_ops_roundtrip_to_runnable_code():
    """PR #46/#60 headline: a CNN whose forward mixes module calls with
    functional/method ops (F.relu, x.flatten(1)) parses into a graph that
    exports to runnable code reproducing the classifier's (batch, 10) output.
    The Flatten MUST land between the pool and the Linear or the Linear
    mis-sizes; conv/bn params must resolve to real names (no arg_i)."""
    graph = _graph(CNN, [1, 3, 32, 32])
    names = _names(graph)
    # Flatten synthesized between pool and fc; F.relu synthesized as ReLU.
    assert "ReLU" in names and "Flatten" in names
    assert names.index("MaxPool2d") < names.index("Flatten") < names.index("Linear")
    assert _params(_node(graph, "conv1")) == {
        "in_channels": 3, "out_channels": 16, "kernel_size": 3, "padding": 1,
    }
    assert _params(_node(graph, "bn1")) == {"num_features": 16}
    # The Flatten node collapses spatial dims (start_dim=1), keeping the batch.
    flat = next(n for n in graph.nodes if n.data.layerType.name == "Flatten")
    assert _params(flat) == {"start_dim": 1}

    model_py = CodeGenerationService().generate_flat_code(graph, [1, 3, 32, 32], "CnnModel")
    assert "arg_0" not in model_py, "positional args leaked as arg_i into export"
    assert _run_flat(graph, [1, 3, 32, 32], "CnnModel") == (1, 10)


def test_rnn_parses_lstm_node_in_order():
    """An RNN-family model parses to an LSTM node threaded input->lstm->fc->
    output. (The LSTM's positional-arg naming is separately pinned as a known
    bug below; here we only assert the ops/order, which is correct today.)"""
    graph = _graph(RNN, [2, 7, 10])
    assert _names(graph) == ["Input", "LSTM", "Linear", "Output"]
    assert _edges(graph) == [("input", "lstm"), ("lstm", "fc"), ("fc", "output")]


# ── statements the parser cannot type: kept, not dropped ─────────────────
# These two used to pin the OPPOSITE behaviour — the parser dropped an
# unrecognised statement silently and the tests asserted the resulting chain was
# "clean". It was not clean, it was wrong: the graph claimed the Linear fed the
# output when a transpose sat between them, so a canvas edit or a re-export
# rebuilt a model the file never described. An op we cannot type is now an opaque
# `python` node carrying its own source text (task.md M1, D5).
def test_transpose_becomes_an_opaque_python_node():
    """x.transpose(...) is not a recognized functional op. It stays in the graph
    as an opaque `python` node between the Linear and the output, carrying the
    statement's source text, rather than vanishing."""
    code = """import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
    def forward(self, x):
        x = self.fc(x)
        x = x.transpose(1, 2)
        return x
"""
    graph = _graph(code, [2, 4, 4])
    assert _names(graph) == ["Input", "Linear", "Python", "Output"]
    opaque = next(n for n in graph.nodes if n.data.layerType.id == "python")
    assert opaque.data.parameters["_raw_call"] == "x.transpose(1, 2)"
    assert _edges(graph) == [("input", "fc"), ("fc", opaque.id), (opaque.id, "output")]


def test_view_with_literal_shape_preserves_reshape_instead_of_guessing_flatten():
    """A literal view retains its shape and tensor edges; only the existing
    `view(_, -1)` idiom lowers to Flatten."""
    code = """import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 4)
        return x
"""
    graph = _graph(code, [2, 4])
    assert "Flatten" not in _names(graph)
    assert _names(graph) == ["Input", "Linear", "torch.reshape", "Output"]
    reshape = next(n for n in graph.nodes if n.data.layerType.name == "torch.reshape")
    assert _params(reshape) == {"shape": [-1, 4]}
    assert _edges(graph) == [("input", "fc"), ("fc", reshape.id), (reshape.id, "output")]
    assert _run_flat(graph, [2, 4], "ViewModel") == (2, 4)


def test_view_flatten_idiom_recovered_with_nonliteral_batch():
    """The canonical `x.view(x.size(0), -1)` flatten-before-Linear idiom is
    recovered as nn.Flatten(start_dim=1) even though the batch arg is a
    non-literal expression, and the result round-trips to runnable code."""
    code = """import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.fc = nn.Linear(256, 10)
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
"""
    graph = _graph(code, [1, 3, 8, 8])
    assert _names(graph) == ["Input", "Conv2d", "Flatten", "Linear", "Output"]
    flat = next(n for n in graph.nodes if n.data.layerType.name == "Flatten")
    assert _params(flat) == {"start_dim": 1}
    assert _run_flat(graph, [1, 3, 8, 8], "ViewModel") == (1, 10)


def test_permute_preserves_literal_dims_and_source_only_symbolic_dims():
    """Both permutations stay visible; symbolic dimensions are retained as
    source expressions and cannot be exported as guessed executable dimensions."""
    literal = """import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
    def forward(self, x):
        x = self.fc(x)
        x = x.permute(0, 2, 1)
        return x
"""
    graph = _graph(literal, [2, 3, 4])
    perm = next(n for n in graph.nodes if n.data.layerType.name == "torch.permute")
    assert _params(perm) == {"dims": [0, 2, 1]}

    nonliteral = """import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
    def forward(self, x):
        x = self.fc(x)
        d = 1
        x = x.permute(0, d, 2)
        return x
"""
    graph2 = _graph(nonliteral, [2, 3, 4])
    assert _names(graph2) == ["Input", "Linear", "torch.permute", "Output"]
    perm2 = next(n for n in graph2.nodes if n.data.layerType.name == "torch.permute")
    assert _params(perm2) == {}
    assert perm2.data.parameters["_source_arguments"] == {"dims": "(0, d, 2)"}
    assert _edges(graph2) == [("input", "fc"), ("fc", perm2.id), (perm2.id, "output")]
    with pytest.raises(UnsafeGraphInput, match="symbolic source arguments"):
        _run_flat(graph2, [2, 3, 4], "SymbolicPermute")


def test_code_without_model_class_fails_gracefully():
    """Source with no nn.Module subclass returns success=False with an error
    message and no graphs, rather than raising or fabricating a graph."""
    res = _parse("import torch\nx = torch.randn(2, 3)\n", [2, 3])
    assert res["success"] is False
    assert res["graphs"] == []
    assert "model" in res["error"].lower()


# ── HTTP integration (e2e via /api/code/parse) ───────────────────────────
@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from server_v2 import app
    return TestClient(app)


def test_http_parse_returns_graph(client):
    """POST /api/code/parse parses an MLP end-to-end over HTTP and returns the
    canvas graph payload (auth disabled in the test env)."""
    r = client.post("/api/code/parse", json={"code": MLP, "input_shape": [1, 8]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    nodes = body["graphs"][0]["graph"]["nodes"]
    assert [n["data"]["layerType"]["name"] for n in nodes] == [
        "Input", "Linear", "ReLU", "Linear", "Output",
    ]


def test_http_parse_syntax_error_is_400(client):
    """Malformed source is a client error: the endpoint surfaces HTTP 400 rather
    than a 200 success=false (which would read as 'parsed')."""
    r = client.post("/api/code/parse", json={"code": "class Net(:", "input_shape": [1, 8]})
    assert r.status_code == 400, r.text


# ── formerly-confirmed bugs, now FIXED (regression tests) ────────────────
def test_bug_rnn_positional_args_named_and_roundtrips():
    graph = _graph(RNN, [2, 7, 10])
    lstm = _node(graph, "lstm")
    assert lstm.data.parameters.get("input_size") == 10
    assert lstm.data.parameters.get("hidden_size") == 20
    assert lstm.data.parameters.get("batch_first") is True
    assert "arg_0" not in lstm.data.parameters
    # LSTM(batch_first) [2,7,10] -> seq [2,7,20] -> Linear(20,5) -> [2,7,5]
    assert _run_flat(graph, [2, 7, 10], "RnnModel") == (2, 7, 5)


def test_bug_leading_functional_op_ordering_and_roundtrip():
    graph = _graph(LEADING_FUNC, [2, 3, 4])
    assert _names(graph) == ["Input", "Flatten", "Linear", "Output"]
    edges = _edges(graph)
    assert ("input", "flatten_0") in edges
    assert ("flatten_0", "fc") in edges
    assert ("fc", "output") in edges
    # fc must receive exactly one input (not the input node AND the flatten).
    assert sum(1 for _s, t in edges if t == "fc") == 1
    # flatten(1): [2,3,4] -> [2,12] -> Linear(12,4) -> [2,4]
    assert _run_flat(graph, [2, 3, 4], "LeadModel") == (2, 4)


def test_bug_http_rnn_positional_args_named(client):
    r = client.post("/api/code/parse", json={"code": RNN, "input_shape": [2, 7, 10]})
    assert r.status_code == 200, r.text
    nodes = r.json()["graphs"][0]["graph"]["nodes"]
    lstm = next(n for n in nodes if n["id"] == "lstm")
    assert lstm["data"]["parameters"].get("input_size") == 10
    assert lstm["data"]["parameters"].get("hidden_size") == 20
