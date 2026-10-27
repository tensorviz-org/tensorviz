"""CODE -> GRAPH: what the user's file makes them see on the canvas.

Everything here goes over HTTP. The user's mental model is simple and the product
either honours it or it does not: the layers I wrote are on the canvas, the arrows
follow my forward pass, and nothing I declared silently disappears.
"""
from __future__ import annotations

import pytest

# --- sources -----------------------------------------------------------------
# Written out in full rather than generated: when one of these fails, the person
# reading the failure needs to see the exact model a user would have typed.

THREE_LAYERS_ONE_UNUSED = '''import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)
        self.unused = nn.Linear(3, 3)

    def forward(self, x):
        h = self.fc1(x)
        h = self.act(h)
        return self.fc2(h)
'''

DECLARED_BACKWARDS = '''import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Linear(8, 8)
        self.a = nn.Linear(4, 8)

    def forward(self, x):
        return self.b(self.a(x))
'''

SEQUENTIAL = '''import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.block(x)
'''

MODULELIST_LOOP = '''import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
'''

TWO_CLASSES = '''import torch.nn as nn


class Alpha(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha_fc = nn.Linear(2, 2)

    def forward(self, x):
        return self.alpha_fc(x)


class Beta(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta_fc = nn.Linear(3, 3)

    def forward(self, x):
        return self.beta_fc(x)
'''

BRANCHING = '''import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)

    def forward(self, x, flag: bool = True):
        if flag:
            return self.a(x)
        return self.b(x)
'''

RESIDUAL = '''import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)

    def forward(self, x):
        h = self.a(x)
        return self.b(h) + h
'''

NESTED_SUBMODULE = '''import torch.nn as nn


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = nn.Linear(8, 8)

    def forward(self, x):
        return self.inner(x)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block()
        self.head = nn.Linear(8, 2)

    def forward(self, x):
        return self.head(self.block(x))
'''

FUNCTIONAL = '''import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8)

    def forward(self, x):
        return F.relu(self.a(x))
'''

WEIGHT_TIED = '''import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8)

    def forward(self, x):
        return self.a(self.a(x))
'''

HIDDEN_FROM_CTOR = '''import torch.nn as nn


class Net(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, x):
        return self.fc2(self.fc1(x))
'''

UNCLOSED_PAREN = '''import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(8, 8
    def forward(self, x):
        return self.l1(x)
'''


# --- helpers -----------------------------------------------------------------


def graph_of(api, source, cls, file):
    response = api.parse(source, cls, file)
    assert response.status_code == 200, response.text
    return response.json()


def node_ids(document):
    return [node["node_id"] for node in document["semantic"]["nodes"]]


def node_by_id(document, node_id):
    for node in document["semantic"]["nodes"]:
        if node["node_id"] == node_id:
            return node
    raise AssertionError(f"{node_id} is not on the canvas: {node_ids(document)}")


def edge_pairs(document):
    return [
        (edge["source_node_id"], edge["target_node_id"])
        for edge in document["semantic"]["edges"]
    ]


# --- what the user declared shows up ----------------------------------------


def test_every_layer_in_init_is_on_the_canvas_under_its_attribute_name(api):
    """The user clicks a node to edit `self.fc1`; the id has to be `fc1` or the
    canvas is talking about a layer they cannot find in their file."""
    document = graph_of(api, THREE_LAYERS_ONE_UNUSED, "Net", "c2g_names.py")["document"]

    assert node_ids(document) == ["input", "fc1", "act", "fc2", "unused", "output"]
    assert node_by_id(document, "fc1")["op"] == "Linear"
    assert node_by_id(document, "act")["op"] == "ReLU"
    assert node_by_id(document, "fc1")["params"]["in_features"] == 4
    assert node_by_id(document, "fc1")["params"]["out_features"] == 8


def test_a_layer_never_called_in_forward_still_appears_disconnected(api):
    """Dead layers are how you notice you wired something wrong. If the projector
    dropped `self.unused`, the canvas would agree with the mistake instead of
    showing it."""
    document = graph_of(api, THREE_LAYERS_ONE_UNUSED, "Net", "c2g_unused.py")["document"]

    assert "unused" in node_ids(document)
    touching_unused = [pair for pair in edge_pairs(document) if "unused" in pair]
    assert touching_unused == []


def test_edges_follow_the_forward_pass_not_the_declaration_order(api):
    """`self.b` is declared first and runs second. Drawing declaration order would
    show the user a model that is not the one they will train."""
    document = graph_of(api, DECLARED_BACKWARDS, "Net", "c2g_order.py")["document"]

    assert edge_pairs(document) == [("input", "a"), ("a", "b"), ("b", "output")]


def test_a_branch_in_forward_shows_both_paths(api):
    """An `if` picks one path at runtime, but the user is reading their code, not a
    run: both layers are reachable and both must be drawn as reachable."""
    document = graph_of(api, BRANCHING, "Net", "c2g_branch.py")["document"]

    assert set(edge_pairs(document)) == {
        ("input", "a"),
        ("input", "b"),
        ("a", "output"),
        ("b", "output"),
    }


def test_a_residual_add_shows_the_skip_as_a_second_arrow(api):
    """`self.b(h) + h` is the one line where the picture beats the code. The add
    needs both of its inputs, or the skip connection is invisible."""
    document = graph_of(api, RESIDUAL, "Net", "c2g_residual.py")["document"]

    add_ids = [n["node_id"] for n in document["semantic"]["nodes"] if n["op"] == "torch.add"]
    assert len(add_ids) == 1
    add = add_ids[0]
    into_add = sorted(source for source, target in edge_pairs(document) if target == add)
    assert into_add == ["a", "b"]


def test_a_functional_call_in_forward_becomes_its_own_node(api):
    """`F.relu` is a layer to the reader even though it is not in `__init__`."""
    document = graph_of(api, FUNCTIONAL, "Net", "c2g_functional.py")["document"]

    relu = [n for n in document["semantic"]["nodes"] if n["op"] == "ReLU"]
    assert len(relu) == 1
    assert (relu[0]["node_id"], "output") in edge_pairs(document)


def test_a_custom_submodule_is_one_node_that_names_its_class(api):
    """`self.block = Block()` is one box on the canvas, and the box has to say
    which class it is or the user cannot open it."""
    document = graph_of(api, NESTED_SUBMODULE, "Net", "c2g_nested.py")["document"]

    assert node_ids(document) == ["input", "block", "head", "output"]
    assert node_by_id(document, "block")["params"]["_class"] == "Block"
    assert edge_pairs(document) == [("input", "block"), ("block", "head"), ("head", "output")]


def test_a_layer_applied_twice_is_still_one_layer(api):
    """Weight tying (`self.a(self.a(x))`) is one set of weights. Two boxes would
    tell the user they have twice the parameters they do."""
    document = graph_of(api, WEIGHT_TIED, "Net", "c2g_tied.py")["document"]

    assert node_ids(document).count("a") == 1


# --- containers --------------------------------------------------------------


def test_sequential_members_are_each_their_own_node(api):
    """A user reads `nn.Sequential` as three layers, so three boxes wired in the
    order they were written."""
    document = graph_of(api, SEQUENTIAL, "Net", "c2g_sequential.py")["document"]

    ops = [n["op"] for n in document["semantic"]["nodes"]]
    assert ops == ["Input", "Linear", "ReLU", "Linear", "Output"]

    ids = node_ids(document)
    first, relu, second = ids[1], ids[2], ids[3]
    assert (first, relu) in edge_pairs(document)
    assert (relu, second) in edge_pairs(document)
    assert ("input", first) in edge_pairs(document)
    assert (second, "output") in edge_pairs(document)


def test_a_for_loop_over_a_modulelist_still_yields_a_node(api):
    """Control flow the projector cannot unroll must not swallow the forward pass:
    the canvas still has to carry input to output through something."""
    document = graph_of(api, MODULELIST_LOOP, "Net", "c2g_loop.py")["document"]

    body = [n for n in document["semantic"]["nodes"] if n["params"].get("_control_flow")]
    assert len(body) == 1
    assert ("input", body[0]["node_id"]) in edge_pairs(document)
    assert (body[0]["node_id"], "output") in edge_pairs(document)


def test_the_layers_inside_a_modulelist_are_visible(api):
    """The user declared two `nn.Linear`. Every other declaration in `__init__`
    becomes something they can see and click; these two are the exception."""
    document = graph_of(api, MODULELIST_LOOP, "Net", "c2g_ml_members.py")["document"]

    linears = [n for n in document["semantic"]["nodes"] if n["op"] == "Linear"]
    # `_modules`, not `modules`: the same bag is handed to a torch constructor with
    # `name` and every `_`-prefixed key removed, so the container's own argument name
    # was the one key that reached `nn.ModuleList(...)` — and the canvas answered a
    # correct model with `build error: dict is not a Module subclass`.
    listed = node_by_id(document, "layers")["params"]["_modules"]
    assert len(linears) == 2 or len(listed) == 2


@pytest.mark.xfail(
    strict=True,
    reason="FINDING: an nn.ModuleList the forward loop iterates is drawn with no "
    "edges, i.e. exactly like a layer that is never called",
)
def test_a_modulelist_the_forward_uses_is_not_drawn_as_dead(api):
    """This product uses "no arrows" to mean "you declared this and never used it"
    (see the unused-layer test). Using the same picture for a ModuleList the
    forward pass iterates tells the user a straight-out falsehood about their
    model."""
    document = graph_of(api, MODULELIST_LOOP, "Net", "c2g_ml_edges.py")["document"]

    assert [pair for pair in edge_pairs(document) if "layers" in pair] != []


# --- picking the class -------------------------------------------------------


@pytest.mark.parametrize(
    "cls, expected",
    [("Alpha", "alpha_fc"), ("Beta", "beta_fc")],
)
def test_cls_picks_the_class_the_user_asked_for(api, cls, expected):
    """Research files hold several models. Opening `Beta` and being shown `Alpha`
    would be worse than an error."""
    document = graph_of(api, TWO_CLASSES, cls, f"c2g_pick_{cls}.py")["document"]

    assert node_ids(document) == ["input", expected, "output"]
    assert edge_pairs(document) == [("input", expected), (expected, "output")]


def test_no_cls_falls_back_to_the_first_model_in_the_file(api):
    """Opening a file without naming a class must still draw something."""
    parsed = graph_of(api, TWO_CLASSES, None, "c2g_pick_default.py")

    assert parsed["cls"] == "Alpha"
    assert node_ids(parsed["document"]) == ["input", "alpha_fc", "output"]


def test_asking_for_a_class_that_is_not_there_says_so_by_name(api):
    """A stale class name in the extension must produce an error the user can act
    on, not an empty canvas."""
    response = api.parse(TWO_CLASSES, "Gamma", "c2g_missing_cls.py")

    assert response.status_code == 404
    assert response.json()["detail"] == "Gamma is not in c2g_missing_cls.py"


# --- failures the user has to be told about ---------------------------------


def test_a_syntax_error_reports_the_line_the_user_must_fix(api):
    """Mid-edit the file is broken most of the time. The error has to point at the
    line, because that is the whole difference between a hint and a red screen."""
    response = api.parse(UNCLOSED_PAREN, "Net", "c2g_syntax.py")

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["error"] == "syntax_error"
    assert detail["line"] == 7
    assert "never closed" in detail["message"]


@pytest.mark.parametrize(
    "source",
    ["def f(x):\n    return x\n", "class Plain:\n    pass\n", ""],
    ids=["function-only", "plain-class", "empty-file"],
)
def test_a_file_with_no_model_is_refused_not_drawn_empty(api, source):
    """Opening the wrong file should say so rather than hand back a blank canvas
    the user will read as "my model has no layers"."""
    response = api.parse(source, None, "c2g_nomodel.py")

    assert response.status_code == 422
    assert response.json()["detail"] == "No PyTorch model classes found in the code"


# --- shapes are a reading of the code, never a measurement -------------------


def test_parse_never_claims_to_have_measured_a_shape(api):
    """Nothing has been run. Every shape on the canvas is inferred from source, and
    the document must say so, or the user will trust a number that was guessed."""
    parsed = graph_of(api, THREE_LAYERS_ONE_UNUSED, "Net", "c2g_static.py")

    assert parsed["traced"] is False
    provenance = {
        node["node_id"]: node["params"].get("_shape_provenance")
        for node in parsed["document"]["semantic"]["nodes"]
    }
    assert set(provenance.values()) == {"static"}


def test_the_input_shape_matches_the_first_layer_it_feeds(api):
    """`nn.Linear(4, 8)` first means the input box says 4 features. A number that
    disagreed with the layer next to it is worse than no number."""
    parsed = graph_of(api, THREE_LAYERS_ONE_UNUSED, "Net", "c2g_inshape.py")

    assert parsed["document"]["semantic"]["inputs"][0]["shape"] == "1,4"
    assert node_by_id(parsed["document"], "input")["params"]["shape"] == "1,4"


def test_the_input_shape_agrees_with_a_width_taken_from_a_constructor_default(api):
    """The projector already resolved `hidden=16` into `in_features: 16` on the
    layer. The input box beside it still says a 224x224 image, so the canvas shows
    an impossible model."""
    parsed = graph_of(api, HIDDEN_FROM_CTOR, "Net", "c2g_ctorarg.py")

    assert node_by_id(parsed["document"], "fc1")["params"]["in_features"] == 16
    assert parsed["document"]["semantic"]["inputs"][0]["shape"] == "1,16"


# --- the canvas has to be able to draw it ------------------------------------


@pytest.mark.parametrize(
    "source, cls",
    [
        (THREE_LAYERS_ONE_UNUSED, "Net"),
        (SEQUENTIAL, "Net"),
        (MODULELIST_LOOP, "Net"),
        (BRANCHING, "Net"),
        (RESIDUAL, "Net"),
        (NESTED_SUBMODULE, "Net"),
    ],
    ids=["plain", "sequential", "modulelist", "branching", "residual", "nested"],
)
def test_every_node_has_a_position_to_be_drawn_at(api, source, cls):
    """A node with no entry in `view.node_positions` cannot be rendered, so from
    the user's seat it does not exist — the exact failure the disconnected-layer
    guarantee above is meant to prevent."""
    document = graph_of(api, source, cls, "c2g_positions.py")["document"]

    positions = document["view"]["node_positions"]
    assert set(positions) == set(node_ids(document))
    for node_id, point in positions.items():
        assert isinstance(point["x"], (int, float)), node_id
        assert isinstance(point["y"], (int, float)), node_id


def test_two_nodes_never_share_a_position(api):
    """Stacked boxes look like one box: the user would count their layers wrong."""
    document = graph_of(api, THREE_LAYERS_ONE_UNUSED, "Net", "c2g_no_overlap.py")["document"]

    points = [(p["x"], p["y"]) for p in document["view"]["node_positions"].values()]
    assert len(set(points)) == len(points)


def test_parsing_the_same_file_twice_gives_the_same_graph(api):
    """Saving a file re-projects it. A graph that shuffled between two identical
    parses would move the user's canvas under their cursor for no reason."""
    first = graph_of(api, THREE_LAYERS_ONE_UNUSED, "Net", "c2g_stable.py")
    second = graph_of(api, THREE_LAYERS_ONE_UNUSED, "Net", "c2g_stable.py")

    assert first["source_digest"] == second["source_digest"]
    assert first["document"]["semantic"] == second["document"]["semantic"]
    assert first["document"]["view"]["node_positions"] == second["document"]["view"]["node_positions"]
    assert first["spanmap"] == second["spanmap"]


def test_the_spanmap_points_at_the_line_each_layer_was_declared_on(api):
    """Clicking a node jumps to its declaration. A span off by a line sends the
    user to the wrong layer, which is how a graph edit lands in the wrong place."""
    parsed = graph_of(api, THREE_LAYERS_ONE_UNUSED, "Net", "c2g_spans.py")

    spans = parsed["spanmap"]
    assert spans["fc1"]["start"][0] == 7
    assert spans["act"]["start"][0] == 8
    assert spans["fc2"]["start"][0] == 9
    assert spans["unused"]["start"][0] == 10
    assert spans["fc1"]["file"] == "c2g_spans.py"

    source_lines = THREE_LAYERS_ONE_UNUSED.splitlines()
    line = source_lines[spans["fc1"]["start"][0] - 1]
    assert line[spans["fc1"]["start"][1] : spans["fc1"]["end"][1]] == "self.fc1 = nn.Linear(4, 8)"
