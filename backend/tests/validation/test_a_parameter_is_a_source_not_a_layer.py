"""`nn.Parameter` produces a tensor and consumes nothing.

Modelling it as a layer — something that takes an input and returns one — was wrong in
four independent places at once, and a freshly opened demo showed the result as its only
problem: `self.pos` hanging off the canvas with "Node has no input connections", on a
model that runs correctly.

    self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))     # line 14
    x = self.embed(tokens) + self.pos[:, : tokens.shape[1]]       # line 24

Line 24 states the connection plainly. It was not drawn because:

  1. `resolve` mapped `self.<attr>` to a node only when it was CALLED. A bare read fell
     through to `resolve(self)`, which is nothing — so the parameter had no out-edge.
  2. `visit_binop` therefore saw one source instead of two, took its "`x * 2` scales one
     tensor" branch, and emitted no `add` — so the join drew nothing either.
  3. the connectivity rule asked a source for an input connection, and
  4. the tracer can never enter a parameter (it has no frame), so the node is marked
     `unreached` on every SUCCESSFUL run, and that was reported as "the run failed
     before it".

Each is tested here, because fixing them one at a time just moved the error along — which
canonical_validation's own comment had already warned about for the frame case.
"""
from __future__ import annotations

from core.shape_inference import _SOURCE_OPS, _bare_op
from services.canonical_validation import _SOURCE_OPS as VALIDATION_SOURCE_OPS
from models.graph_models import CodeParseRequest
from utils.code_parser import CodeParserService

SOURCE = '''
import torch
import torch.nn as nn


# tensorviz: input=1,10
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=16, embedding_dim=32)
        self.pos = nn.Parameter(torch.zeros(1, 12, 32))
        self.head = nn.Linear(in_features=32, out_features=16)

    def forward(self, tokens):
        x = self.embed(tokens) + self.pos[:, : tokens.shape[1]]
        return self.head(x)
'''


def _graph():
    result = CodeParserService().parse_pytorch_code(CodeParseRequest(code=SOURCE))
    assert result.get("success"), result
    return result["graphs"][0]["graph"]


def _nodes(graph):
    return graph["nodes"] if isinstance(graph, dict) else graph.nodes


def _edge_pairs(graph):
    edges = graph["edges"] if isinstance(graph, dict) else graph.edges
    out = []
    for e in edges:
        if isinstance(e, dict):
            out.append((str(e.get("source")), str(e.get("target"))))
        else:
            out.append((str(e.source), str(e.target)))
    return out


def _node_ids(graph):
    ids = []
    for n in _nodes(graph):
        ids.append(str(n.get("id") if isinstance(n, dict) else n.id))
    return ids


def _edges(graph):
    out = {}
    for src, tgt in _edge_pairs(graph):
        out.setdefault(src, []).append(tgt)
    return out


def test_a_parameter_read_in_forward_is_wired_to_what_uses_it():
    """The reported symptom: the card hung off the canvas with no edge at all."""
    graph = _graph()
    edges = _edges(graph)
    assert edges.get("pos"), "nn.Parameter has no outgoing edge — it is dangling"


def test_the_addition_that_joins_them_becomes_a_node():
    """`embed(tokens) + pos[...]` joins two tensors, so something must merge them."""
    graph = _graph()
    adds = [i for i in _node_ids(graph) if "add" in i.lower()]
    assert adds, "the `+` on line 24 produced no node, so the join is invisible"
    add_id = adds[0]
    feeding = {s for s, t in _edge_pairs(graph) if t == add_id}
    assert "pos" in feeding, "the parameter does not reach the add"
    assert "embed" in feeding, "the embedding does not reach the add"


def test_calling_a_layer_does_not_wire_it_to_itself():
    """The trap in fixing the above: `self.embed` is the callee, not a tensor flowing in."""
    graph = _graph()
    assert not [p for p in _edge_pairs(graph) if p[0] == p[1]], (
        "a layer was connected to itself — the callee was resolved as its own source"
    )


def test_both_engines_agree_on_what_counts_as_a_source():
    """Two exemption lists that disagree would reintroduce the bug on one path only."""
    assert "parameter" in _SOURCE_OPS
    assert "parameter" in VALIDATION_SOURCE_OPS


def test_the_bare_op_helper_strips_the_module_path():
    assert _bare_op("torch.nn.Parameter") == "parameter"
    assert _bare_op("Parameter") == "parameter"
    assert _bare_op(None) == ""
