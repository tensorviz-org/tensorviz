"""forward() is read as dataflow, not linearised into one chain.

Before M1 the parser appended every self.X call it saw to a single chain and
concatenated the two operands of a BinOp, so `x + self.conv(x)` produced no add
node and no skip edge: a residual network imported as a straight stack. Branches,
merges and anything the parser could not type were silently dropped, and the
graph said the model was something it is not.

What is asserted here (task.md M1):

  * a residual `x + self.l2(h)` is an ``add`` node with TWO in-edges, one of them
    the input tensor itself — the skip connection;
  * ``torch.cat([a, b], 1)`` is a ``cat`` node fed by both branches, carrying the
    axis it joins on;
  * an unrecognised statement is an opaque ``python`` node in the topology, not a
    hole in it;
  * compound statement bodies are walked, and what is inside them is marked.

Run: KMP_DUPLICATE_LIB_OK=TRUE <viz-python> -m pytest tests/test_code_parser_dataflow.py -q
"""

from __future__ import annotations



from models.graph_models import CodeParseRequest  # noqa: E402
from utils.code_parser import CodeParserService  # noqa: E402


def _graph(code: str, input_shape):
    result = CodeParserService().parse_pytorch_code(
        CodeParseRequest(code=code, input_shape=input_shape)
    )
    assert result.get("graphs"), result
    graph = result["graphs"][0]["graph"]
    assert graph is not None, result["graphs"][0].get("error")
    return graph


def _by_op(graph, op_id):
    return [n for n in graph.nodes if n.data.layerType.id == op_id]


def _sources_of(graph, node_id):
    return {e.source for e in graph.edges if e.target == node_id}


def _op_topology(graph):
    """Edges rewritten in terms of layer-type ids, so a test can state the shape
    of the graph without pinning generated node ids."""
    op = {n.id: n.data.layerType.id for n in graph.nodes}
    return [(op[e.source], op[e.target]) for e in graph.edges]


TWO_LAYERS = """import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 4)
        self.l2 = nn.Linear(4, 4)
    def forward(self, x):
"""


def test_a_residual_add_is_one_merge_node_fed_by_the_skip_and_the_branch():
    code = TWO_LAYERS + """        h = self.l1(x)
        return x + self.l2(h)
"""
    graph = _graph(code, [2, 4])

    adds = _by_op(graph, "add")
    assert len(adds) == 1, [n.id for n in graph.nodes]
    add = adds[0]
    sources = _sources_of(graph, add.id)
    # Two in-edges: the tensor that skipped the branch, and the branch's output.
    assert sources == {"input", "l2"}
    assert add.data.layerType.name == "torch.add"
    # The residual is what leaves the model.
    assert (add.id, "output") in [(e.source, e.target) for e in graph.edges]


def test_two_branches_off_one_tensor_meet_at_a_cat_carrying_its_axis():
    code = TWO_LAYERS + """        a = self.l1(x)
        b = self.l2(x)
        return torch.cat([a, b], 1)
"""
    graph = _graph(code, [2, 4])

    cats = _by_op(graph, "cat")
    assert len(cats) == 1, [n.id for n in graph.nodes]
    cat = cats[0]
    assert _sources_of(graph, cat.id) == {"l1", "l2"}
    assert cat.data.parameters["dim"] == 1
    # Both branches read the SAME input tensor; that is the fan-out the old
    # linearising walk turned into l1 -> l2.
    assert _sources_of(graph, "l1") == {"input"}
    assert _sources_of(graph, "l2") == {"input"}


def test_an_unrecognised_statement_is_an_opaque_node_in_the_chain():
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

    opaques = _by_op(graph, "python")
    assert len(opaques) == 1
    assert opaques[0].data.parameters["_raw_call"] == "x.transpose(1, 2)"
    assert _op_topology(graph) == [
        ("input", "linear"), ("linear", "python"), ("python", "output"),
    ]


def test_a_nested_unrecognised_call_is_an_operand_not_a_statement():
    """`x.size(0)` inside a view is not an op the model performs — it reads a
    dimension. Turning every nested call into a node would fill the canvas with
    nodes no tensor flows through."""
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
    assert _by_op(graph, "python") == []
    assert _op_topology(graph) == [
        ("input", "conv2d"), ("conv2d", "flatten"), ("flatten", "linear"),
        ("linear", "output"),
    ]


def test_a_loop_body_is_walked_and_what_it_contains_is_marked():
    """The old walk skipped compound statements outright, so a model whose whole
    forward ran inside a loop parsed to an empty chain."""
    code = """import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
    def forward(self, x):
        for _ in range(3):
            x = self.fc(x)
        return x
"""
    graph = _graph(code, [2, 4])
    assert _op_topology(graph) == [("input", "linear"), ("linear", "output")]


def test_a_branch_body_is_walked_and_its_nodes_say_they_are_conditional():
    code = """import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(4, 4)
        self.out = nn.Linear(4, 4)
    def forward(self, x):
        if x.sum() > 0:
            x = self.gate(x)
        return self.out(x)
"""
    graph = _graph(code, [2, 4])
    # The conditional layer is wired, not dropped: input -> gate -> out.
    assert _op_topology(graph) == [
        ("input", "linear"), ("linear", "linear"), ("linear", "output"),
    ]
    assert _sources_of(graph, "gate") == {"input"}
    # Known limit of the static half: the walk takes the branch's binding rather
    # than joining it with the one that skipped the branch, so `out` reads `gate`
    # only. A join would give a single-input Linear two in-edges, which shape
    # inference would read as a merge — a different lie. What the conditionality
    # IS recorded on is the node the branch creates (`_control_flow`, asserted in
    # test_code_parser_spans.py); the executed trace settles it in M4.
    assert _sources_of(graph, "out") == {"gate"}
    # `x.sum() > 0` asks a question about the tensor; it does not transform one,
    # so it is not a node.
    assert _by_op(graph, "python") == []


def test_an_augmented_assignment_joins_the_old_value_with_the_new_one():
    code = TWO_LAYERS + """        x = self.l1(x)
        x += self.l2(x)
        return x
"""
    graph = _graph(code, [2, 4])
    adds = _by_op(graph, "add")
    assert len(adds) == 1
    assert _sources_of(graph, adds[0].id) == {"l1", "l2"}


def test_scaling_one_tensor_preserves_its_literal_operand():
    """Review must not erase arithmetic just because it has one tensor input."""
    code = """import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
    def forward(self, x):
        return self.fc(x) * 2
"""
    graph = _graph(code, [2, 4])
    mul = _by_op(graph, "mul")
    assert len(mul) == 1
    assert mul[0].data.parameters["other"] == 2
    assert mul[0].data.parameters["_scalar_side"] == "right"
    assert _op_topology(graph) == [("input", "linear"), ("linear", "mul"), ("mul", "output")]


def test_a_user_class_used_as_a_submodule_links_to_its_own_subgraph():
    """`self.blk = MyBlock()` used to emit two unrelated graphs and lose the fact
    that one holds the other (the C5 TODO). The reference is now a `custom_code`
    node carrying the class name and that class's static subgraph."""
    code = """import torch.nn as nn


class MyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Linear(4, 4)

    def forward(self, x):
        return self.c(x)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.blk = MyBlock()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(self.blk(x))
"""
    result = CodeParserService().parse_pytorch_code(
        CodeParseRequest(code=code, input_shape=[2, 4], file="models/blocks.py")
    )
    top = next(g for g in result["graphs"] if g["is_top_level"])
    assert top["name"] == "Net"
    blk = next(n for n in top["graph"].nodes if n.id == "blk")
    assert blk.data.layerType.id == "custom_code"
    assert blk.data.parameters["_class"] == "MyBlock"
    assert blk.data.parameters["_source_file"] == "models/blocks.py"
    inner = blk.data.internalGraph
    assert inner is not None
    assert [n["data"]["layerType"]["id"] for n in inner["nodes"]] == [
        "input", "linear", "output",
    ]
