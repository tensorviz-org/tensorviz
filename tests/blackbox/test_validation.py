"""Validation and trace merge, over HTTP only.

The question these tests answer is the one a user asks when they hit "Trace":
*did the picture I am looking at get better, and did it lose anything?* The
merge is the only place in the product where two sources of truth meet — the
file the user is typing in, and what their interpreter actually did — so the
failure modes that matter are silent ones: a card that quietly disappears
because the run took the other branch, a shape still presented as measured when
nothing ran, a partial trace answered with a 500 instead of the half-graph it
did produce.

Nothing here imports the product. The trace envelope below is written out as
literal JSON because that is exactly what the extension holds: it reads the
worker's stdout (task.md §6.3 — ``{protocol_version, result, spans, error,
worker}``) out of ``.tensorviz/cache`` and posts it back. Building it by calling
the worker would make these tests depend on torch, on the user's interpreter and
on the very code they are meant to check from the outside.
"""
from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional

import pytest

# The worker reports absolute paths from the user's machine; the editor talks in
# workspace-relative ones. Keeping them different here is the point: a merge that
# only worked when the two strings matched would break on every real workspace.
WORKSPACE = "/Users/someone/project"

# Where an nn.Linear's class is defined. A span pointing here is a span that would
# navigate the user out of their own project, which is why the parser's declaration
# must win it back.
TORCH_LINEAR = "/opt/miniconda/lib/python3.11/site-packages/torch/nn/modules/linear.py"


BRANCHED = '''import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.l1 = nn.Linear(width, width)
        self.skip = nn.Linear(width, width)
        self.l2 = nn.Linear(width, width)

    def forward(self, x, use_skip=False):
        h = self.l1(x)
        if use_skip:
            h = self.skip(h)
        return self.l2(h)
'''

# The same model with a functional call in forward. The parser sees one opaque
# `python` statement; the tracer sees a relu op. They have to end up as one card.
FUNCTIONAL = '''import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.l1 = nn.Linear(width, width)
        self.l2 = nn.Linear(width, width)

    def forward(self, x):
        h = torch.relu(self.l1(x))
        return self.l2(h)
'''


def _linear_node(
    node_id: str,
    *,
    in_shape: str,
    out_shape: str,
    source_line: int,
) -> Dict[str, Any]:
    """One executed ``nn.Linear``, shaped the way the worker shapes it."""
    return {
        "id": node_id,
        "type": "layerNode",
        "position": {"x": 0.0, "y": 0.0},
        "data": {
            "layerType": {"id": "linear", "name": "Linear", "parameters": []},
            "parameters": {
                "in_features": 8,
                "out_features": 8,
                "_module_path": node_id,
                "_class": "Linear",
                "_params_M": 0.0001,
                "_subtitle": "Linear",
                "_source_file": TORCH_LINEAR,
                "_source_line": source_line,
                "_in_shape": in_shape,
                "_out_shape": out_shape,
                "_shape_provenance": "observed",
            },
            "parentId": None,
            "depth": 1,
            "expanded": False,
            "isGroup": False,
            "moduleType": "builtin",
            "displayName": node_id,
            "moduleInstanceId": node_id,
            "moduleLayerCount": 0,
        },
    }


def _materialised_node(
    node_id: str, op: str, *, decl_file: str, decl_line: int, in_shape: str, out_shape: str
) -> Dict[str, Any]:
    """An op the tracer caught that has no module behind it — a functional relu.

    Its source is the line in the user's own forward that ran it, which is what
    lets the merge recognise the parser's opaque statement as the same card.
    """
    return {
        "id": node_id,
        "type": "layerNode",
        "position": {"x": 0.0, "y": 0.0},
        "data": {
            "layerType": {"id": op, "name": op.capitalize(), "parameters": []},
            "parameters": {
                "_module_path": node_id,
                "_class": op.capitalize(),
                "_params_M": 0.0,
                "_subtitle": op.capitalize(),
                "_decl_file": f"{WORKSPACE}/{decl_file}",
                "_decl_line": decl_line,
                "_in_shape": in_shape,
                "_out_shape": out_shape,
                "_shape_provenance": "observed",
            },
            "parentId": None,
            "depth": 1,
            "expanded": False,
            "isGroup": False,
            "moduleType": "builtin",
            "displayName": node_id,
            "moduleInstanceId": node_id,
            "moduleLayerCount": 0,
        },
    }


def _boundaries(entry_shape: str) -> List[Dict[str, Any]]:
    return [
        {
            "id": "input_node",
            "type": "inputNode",
            "position": {"x": 0.0, "y": 0.0},
            "data": {
                "layerType": {"id": "input", "name": "Input", "parameters": []},
                "parameters": {"name": "Input", "shape": entry_shape, "_subtitle": "Input"},
            },
        },
        {
            "id": "output_node",
            "type": "outputNode",
            "position": {"x": 600.0, "y": 0.0},
            "data": {
                "layerType": {"id": "output", "name": "Output", "parameters": []},
                "parameters": {"name": "Output", "_subtitle": "Output"},
            },
        },
    ]


def _edges(pairs: List[tuple]) -> List[Dict[str, Any]]:
    return [{"id": f"e{i}", "source": s, "target": t} for i, (s, t) in enumerate(pairs)]


def _envelope(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    spans: Dict[str, Dict[str, Any]],
    *,
    error: Optional[Dict[str, Any]] = None,
    materialised: tuple = (),
) -> Dict[str, Any]:
    """The §6.3 envelope, verbatim: five keys and a protocol version."""
    return {
        "protocol_version": 1,
        "result": {
            "graph": {"nodes": nodes, "edges": edges},
            "diagnostics": {"materialised_ops": list(materialised)},
        },
        "spans": spans,
        "error": error,
        "worker": {"python": "3.11.9", "torch": "2.4.1"},
    }


def _class_span(line: int) -> Dict[str, Any]:
    return {
        "file": TORCH_LINEAR,
        "start": [line, 0],
        "end": [line, 0],
        "kind": "class",
        "digest": "0f1e2d3c4b5a",
    }


def _frame_span(file: str, line: int) -> Dict[str, Any]:
    return {
        "file": f"{WORKSPACE}/{file}",
        "start": [line, 0],
        "end": [line, 0],
        "kind": "frame",
        "digest": "9a8b7c6d5e4f",
    }


def branched_trace(**kwargs) -> Dict[str, Any]:
    """A run of BRANCHED with ``use_skip=False``: l1 and l2 ran, skip did not."""
    nodes = _boundaries("2,8") + [
        _linear_node("l1", in_shape="2,8", out_shape="2,8", source_line=48),
        _linear_node("l2", in_shape="2,8", out_shape="2,8", source_line=48),
    ]
    edges = _edges([("input_node", "l1"), ("l1", "l2"), ("l2", "output_node")])
    spans = {"l1": _class_span(48), "l2": _class_span(48)}
    return _envelope(nodes, edges, spans, **kwargs)


def functional_trace(rel_file: str) -> Dict[str, Any]:
    """A run of FUNCTIONAL: the relu the parser could only see as a statement.

    The worker reports the frame it caught the op in as an ABSOLUTE path in the
    user's checkout, while the editor asked about a workspace-relative one — so
    the file here is deliberately the long form of ``rel_file``.
    """
    nodes = _boundaries("2,8") + [
        _linear_node("l1", in_shape="2,8", out_shape="2,8", source_line=48),
        _materialised_node(
            "relu1", "relu", decl_file=rel_file, decl_line=12, in_shape="2,8", out_shape="2,8"
        ),
        _linear_node("l2", in_shape="2,8", out_shape="2,8", source_line=48),
    ]
    edges = _edges(
        [("input_node", "l1"), ("l1", "relu1"), ("relu1", "l2"), ("l2", "output_node")]
    )
    spans = {
        "l1": _class_span(48),
        "relu1": _frame_span(rel_file, 12),
        "l2": _class_span(48),
    }
    return _envelope(nodes, edges, spans, materialised=("relu1",))


def nodes_by_id(document: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {node["node_id"]: node for node in document["semantic"]["nodes"]}


def edge_pairs(document: Dict[str, Any]) -> set:
    return {
        (edge["source_node_id"], edge["target_node_id"])
        for edge in document["semantic"]["edges"]
    }


def provenance(node: Dict[str, Any]) -> Optional[str]:
    return node["params"].get("_shape_provenance")


# --------------------------------------------------------------------------- #
# /project with no trace


def test_project_without_a_trace_answers_exactly_what_parse_answers(api):
    """The canvas is drawn from whichever call the extension happened to make.

    If /project drifted from /parse by so much as a node id, the picture would
    change when the user pressed Trace and the trace failed — and they would read
    that as the trace having done something.
    """
    parsed = api.parse(BRANCHED, "Net", file="validation_parse_parity.py")
    assert parsed.status_code == 200, parsed.text
    projected = api.project(BRANCHED, "Net", file="validation_parse_parity.py")
    assert projected.status_code == 200, projected.text

    left, right = parsed.json(), projected.json()
    assert right["traced"] is False
    assert left["traced"] is False
    assert set(nodes_by_id(left["document"])) == {"input", "l1", "skip", "l2", "output"}
    assert right["document"]["semantic"]["nodes"] == left["document"]["semantic"]["nodes"]
    assert right["document"]["semantic"]["edges"] == left["document"]["semantic"]["edges"]
    assert right["spanmap"] == left["spanmap"]
    assert right["source_digest"] == left["source_digest"]
    assert right["graph_id"] == left["graph_id"]


def test_an_untraced_projection_never_claims_a_shape_was_measured(api):
    """Every shape on an untraced canvas is a guess read off the source, and the
    document has to say so — the badge that tells the user "stale" is this field."""
    response = api.project(BRANCHED, "Net", file="validation_static_badge.py")
    assert response.status_code == 200, response.text
    document = response.json()["document"]
    by_id = nodes_by_id(document)
    assert set(by_id) == {"input", "l1", "skip", "l2", "output"}
    assert {provenance(node) for node in by_id.values()} == {"static"}


# --------------------------------------------------------------------------- #
# /project with a trace


def test_a_trace_flips_traced_and_puts_measured_shapes_on_the_cards(api):
    """The whole point of running the model: the numbers on the cards stop being
    the parser's guess and become what the tensors actually were."""
    response = api.project(
        BRANCHED, "Net", file="validation_merge_shapes.py", trace_result=branched_trace()
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["traced"] is True

    by_id = nodes_by_id(body["document"])
    assert provenance(by_id["l1"]) == "observed"
    assert by_id["l1"]["params"]["_out_shape"] == "2,8"
    assert by_id["l1"]["params"]["_in_shape"] == "2,8"
    assert provenance(by_id["l2"]) == "observed"


def test_a_measured_shape_replaces_the_static_one_it_contradicts(api):
    """A measurement that disagrees with the declaration is the interesting case —
    it is how a user finds out their model is not doing what they wrote. The
    measurement must win and must not be smoothed back towards the source."""
    envelope = branched_trace()
    for node in envelope["result"]["graph"]["nodes"]:
        if node["id"] == "l2":
            # The declaration says 8 wide; the run says 16. Only one of them ran.
            node["data"]["parameters"]["_out_shape"] = "2,16"
    response = api.project(
        BRANCHED, "Net", file="validation_shape_conflict.py", trace_result=envelope
    )
    assert response.status_code == 200, response.text
    by_id = nodes_by_id(response.json()["document"])
    assert by_id["l2"]["params"]["_out_shape"] == "2,16"
    assert by_id["l2"]["params"]["out_features"] == 8
    assert provenance(by_id["l2"]) == "observed"


def test_a_layer_the_run_never_reached_stays_on_the_canvas(api):
    """`use_skip=False` means the tracer never saw `self.skip`. It is still in the
    user's file, so deleting the card would answer "that layer does not exist" to
    someone looking straight at it."""
    response = api.project(
        BRANCHED, "Net", file="validation_unreached.py", trace_result=branched_trace()
    )
    assert response.status_code == 200, response.text
    by_id = nodes_by_id(response.json()["document"])
    assert "skip" in by_id
    assert provenance(by_id["skip"]) == "unreached"
    assert by_id["skip"]["params"]["_raw_call"] == "nn.Linear(width, width)"


def test_an_unreached_layer_keeps_the_wiring_the_parser_read(api):
    """A dimmed card floating with no edges reads as a bug in the model. The parser
    read `h = self.skip(h)` between l1 and l2 and that is where it belongs."""
    response = api.project(
        BRANCHED, "Net", file="validation_unreached_edges.py", trace_result=branched_trace()
    )
    assert response.status_code == 200, response.text
    pairs = edge_pairs(response.json()["document"])
    assert ("l1", "skip") in pairs
    assert ("skip", "l2") in pairs
    # And the path that actually ran is still drawn beside it.
    assert ("l1", "l2") in pairs


def test_every_card_after_a_merge_has_somewhere_to_be_drawn(api):
    """A node with no entry in view.node_positions is a node the canvas cannot
    place — the dimmed ones are the ones at risk, because no trace positioned them."""
    response = api.project(
        BRANCHED, "Net", file="validation_positions.py", trace_result=branched_trace()
    )
    assert response.status_code == 200, response.text
    document = response.json()["document"]
    assert set(nodes_by_id(document)) == {"input", "l1", "skip", "l2", "output"}
    assert set(document["view"]["node_positions"]) == set(nodes_by_id(document))


def test_the_declaration_in_the_users_file_beats_the_span_inside_torch(api):
    """Clicking an nn.Linear card must land on the line the user wrote it on, not
    on torch/nn/modules/linear.py, which is where the worker's span points."""
    response = api.project(
        BRANCHED, "Net", file="validation_span_wins.py", trace_result=branched_trace()
    )
    assert response.status_code == 200, response.text
    span = response.json()["spanmap"]["l1"]
    assert span["file"] == "validation_span_wins.py"
    assert span["start"] == [8, 8]
    assert span["kind"] == "decl"


def test_a_statement_that_ran_under_a_materialised_op_is_not_drawn_twice(api):
    """`torch.relu(...)` is one opaque statement to the parser and a real op to the
    tracer. Drawing both would put a ghost card beside every functional call."""
    response = api.project(
        FUNCTIONAL,
        "Net",
        file="validation_materialised.py",
        trace_result=functional_trace("validation_materialised.py"),
    )
    assert response.status_code == 200, response.text
    by_id = nodes_by_id(response.json()["document"])
    assert "relu@Net#0" in by_id
    assert "python@Net#0" not in by_id
    assert provenance(by_id["relu@Net#0"]) == "observed"


# --------------------------------------------------------------------------- #
# a run that died half way


def test_a_trace_that_raised_partway_keeps_the_layers_it_never_reached(api):
    """Worker exit 4: the forward blew up, and `result` is the half-graph it built
    before it did. That half is the most useful picture the user will get all day —
    it shows exactly how far the tensors got — so the layers past the failure have
    to stay on the canvas rather than being dropped with the run."""
    partial = _envelope(
        _boundaries("2,8") + [_linear_node("l1", in_shape="2,8", out_shape="2,8", source_line=48)],
        _edges([("input_node", "l1")]),
        {"l1": _class_span(48)},
        error={
            "type": "RuntimeError",
            "message": "mat1 and mat2 shapes cannot be multiplied (2x8 and 16x8)",
            "frames": [{"file": f"{WORKSPACE}/models/net.py", "line": 16, "name": "forward"}],
            "node_id": "l2",
        },
    )
    response = api.project(
        BRANCHED, "Net", file="validation_partial_trace.py", trace_result=partial
    )
    assert response.status_code == 200, response.text
    body = response.json()
    by_id = nodes_by_id(body["document"])
    assert provenance(by_id["l1"]) == "observed"
    # Everything the run never got to is still there, and honestly labelled.
    assert provenance(by_id["l2"]) == "unreached"
    assert provenance(by_id["skip"]) == "unreached"
    assert body["traced"] is True


def test_the_failure_that_stopped_the_run_reaches_the_canvas_as_data(api):
    """A partial trace without its reason is indistinguishable from a model whose
    second half is genuinely dead code. The worker knows which it was — it captured
    the RuntimeError and the line — and that has to survive the projection."""
    message = "mat1 and mat2 shapes cannot be multiplied (2x8 and 16x8)"
    partial = _envelope(
        _boundaries("2,8") + [_linear_node("l1", in_shape="2,8", out_shape="2,8", source_line=48)],
        _edges([("input_node", "l1")]),
        {"l1": _class_span(48)},
        error={
            "type": "RuntimeError",
            "message": message,
            "frames": [{"file": f"{WORKSPACE}/models/net.py", "line": 16, "name": "forward"}],
            "node_id": "l2",
        },
    )
    response = api.project(
        BRANCHED, "Net", file="validation_error_surfaced.py", trace_result=partial
    )
    assert response.status_code == 200, response.text
    assert message in response.text


# --------------------------------------------------------------------------- #
# envelopes this backend must refuse rather than half-read


@pytest.mark.parametrize(
    "envelope, why",
    [
        ({"protocol_version": 99, "result": {"graph": {"nodes": [{"id": "l1"}]}}}, "wrong build"),
        ({"protocol_version": 1, "result": {"graph": {"nodes": []}}}, "nothing ran"),
        ({"protocol_version": 1, "result": {}}, "no graph"),
        ({"protocol_version": 1}, "no result"),
        ({}, "not an envelope at all"),
    ],
)
def test_an_envelope_this_backend_cannot_read_is_refused_loudly(api, envelope, why):
    """Accepting one of these and quietly projecting the static tier would answer
    `traced: false` while the user watches a notification that says a trace ran —
    the worst possible outcome, because the numbers would look measured."""
    response = api.project(
        BRANCHED, "Net", file="validation_refuse.py", trace_result=envelope
    )
    assert response.status_code == 400, f"{why}: {response.status_code} {response.text}"
    assert response.json()["detail"]["error"] == "trace_rejected"


def test_a_trace_result_that_is_not_an_object_is_a_client_error_not_a_crash(api):
    """A truncated cache file deserialises to anything at all. The backend must
    reject the request, not fall over inside the merge."""
    for junk in ("not-json-at-all", 7, [1, 2, 3]):
        response = api.project(
            BRANCHED, "Net", file="validation_junk_envelope.py", trace_result=junk
        )
        assert response.status_code == 422, f"{junk!r}: {response.status_code}"


def test_a_broken_buffer_is_a_syntax_diagnostic_even_with_a_trace_attached(api):
    """The user types, the buffer breaks, and the extension still has yesterday's
    trace in its cache. The answer must be the diagnostic that underlines the typo,
    not a projection of a file that no longer parses."""
    broken = BRANCHED.replace("def forward(self, x, use_skip=False):", "def forward(self x):")
    response = api.project(
        broken, "Net", file="validation_broken_buffer.py", trace_result=branched_trace()
    )
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert detail["error"] == "syntax_error"
    assert detail["line"] == 12


def test_a_class_the_buffer_does_not_define_is_refused_before_any_merge(api):
    """The extension asks for the class under the cursor. If the user renamed it,
    projecting some other class in the file would silently show the wrong model."""
    response = api.project(
        BRANCHED, "Renamed", file="validation_missing_class.py", trace_result=branched_trace()
    )
    assert response.status_code == 404, response.text
    assert "Renamed" in str(response.json()["detail"])


def test_a_file_with_no_nn_module_is_refused_rather_than_drawn_empty(api):
    """An empty canvas and "this file has no model in it" look the same to the
    backend and completely different to the user."""
    response = api.project(
        "x = 1\n", None, file="validation_no_model.py", trace_result=branched_trace()
    )
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == "No PyTorch model classes found in the code"


def test_projecting_a_buffer_does_not_run_a_line_of_it(tmp_path, api):
    """The buffer under the cursor is half-written by definition, and this process
    is a server. If /project imported it, opening a file with a `shutil.rmtree` at
    module scope would run it — which is why execution lives in the worker."""
    sentinel = tmp_path / "the_backend_executed_the_buffer"
    source = (
        "import pathlib\n"
        "import torch.nn as nn\n"
        "\n"
        f"pathlib.Path({str(sentinel)!r}).write_text('executed')\n"
        "\n"
        "\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.l1 = nn.Linear(8, 8)\n"
        "\n"
        "    def forward(self, x):\n"
        "        return self.l1(x)\n"
    )
    response = api.project(source, "Net", file="validation_no_execution.py")
    assert response.status_code == 200, response.text
    assert not sentinel.exists()


def test_the_same_buffer_and_trace_project_to_the_same_document_twice(api):
    """The extension re-projects on every save and diffs the result to decide what
    to redraw. A merge that reordered its own output would repaint the whole canvas
    — and move every card the user had positioned — on a keystroke that changed
    nothing."""
    envelope = branched_trace()
    first = api.project(
        BRANCHED, "Net", file="validation_stable.py", trace_result=copy.deepcopy(envelope)
    )
    second = api.project(
        BRANCHED, "Net", file="validation_stable.py", trace_result=copy.deepcopy(envelope)
    )
    assert first.status_code == 200 and second.status_code == 200
    assert first.json()["document"]["semantic"]["nodes"] == second.json()["document"]["semantic"]["nodes"]
    assert first.json()["document"]["semantic"]["edges"] == second.json()["document"]["semantic"]["edges"]
    assert first.json()["spanmap"] == second.json()["spanmap"]


def test_health_reports_the_interpreter_a_trace_would_have_to_agree_with(api):
    """A merge is only meaningful if the backend and the worker are the same build;
    /health is what the extension checks before it offers to trace at all."""
    response = api.health()
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "healthy"
    assert re.fullmatch(r"\d+\.\d+(\..*)?", body["environment"]["torch"]), body["environment"]
    assert re.fullmatch(r"3\.\d+(\..*)?", body["environment"]["python"]), body["environment"]


# --------------------------------------------------------------------------- #
# a repeated block: one declaration, many executed instances

REPEATED = '''import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, width=8, depth=2):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.head = nn.Linear(width, width)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x)
'''


def repeated_trace(rel_file: str) -> Dict[str, Any]:
    """A run of REPEATED: the loop unrolled into blocks.0 and blocks.1.

    Each instance carries the forward line it was invoked from, which is what the
    worker's frame capture records and the only thing that ties two executed
    Linears back to the one `for` statement the parser saw.
    """
    instances = []
    for index in (0, 1):
        node = _linear_node(
            f"blocks.{index}", in_shape="2,8", out_shape="2,8", source_line=48
        )
        node["data"]["parameters"]["_decl_file"] = f"{WORKSPACE}/{rel_file}"
        node["data"]["parameters"]["_decl_line"] = 13
        instances.append(node)
    nodes = (
        _boundaries("2,8")
        + instances
        + [_linear_node("head", in_shape="2,8", out_shape="2,8", source_line=48)]
    )
    edges = _edges(
        [
            ("input_node", "blocks.0"),
            ("blocks.0", "blocks.1"),
            ("blocks.1", "head"),
            ("head", "output_node"),
        ]
    )
    spans = {
        "blocks.0": _frame_span(rel_file, 13),
        "blocks.1": _frame_span(rel_file, 13),
        "head": _class_span(48),
    }
    return _envelope(nodes, edges, spans)


def test_a_declaration_the_trace_unrolled_is_not_also_drawn_as_a_ghost(api):
    """`self.blocks` is one line in the file and two cards on the traced canvas.
    Drawing the declaration a third time — dimmed, as if it never ran — would tell
    the user their ModuleList was skipped while its instances sit right beside it."""
    response = api.project(
        REPEATED,
        "Net",
        file="validation_repeat.py",
        trace_result=repeated_trace("validation_repeat.py"),
    )
    assert response.status_code == 200, response.text
    by_id = nodes_by_id(response.json()["document"])
    assert provenance(by_id["blocks.0"]) == "observed"
    assert provenance(by_id["blocks.1"]) == "observed"
    assert "blocks" not in by_id
    # `for block in self.blocks:` is what ran them; it is not a separate card.
    assert "python@Net#0" not in by_id


def test_an_unrolled_instance_points_back_at_the_line_that_declared_it(api):
    """"Reveal in editor" on blocks.0 has to land on the ModuleList the user wrote,
    not on torch's linear.py, which is the only file the tracer knows about."""
    response = api.project(
        REPEATED,
        "Net",
        file="validation_repeat_span.py",
        trace_result=repeated_trace("validation_repeat_span.py"),
    )
    assert response.status_code == 200, response.text
    span = response.json()["spanmap"]["blocks.0"]
    assert span["file"] == "validation_repeat_span.py"
    assert span["start"] == [8, 8]
    assert span["kind"] == "decl"


# --------------------------------------------------------------------------- #
# the model that could not even be built


def _construct_failure() -> Dict[str, Any]:
    """Worker exit 3: the class raised in ``__init__``, so there is no graph."""
    return {
        "protocol_version": 1,
        "result": None,
        "spans": {},
        "error": {
            "type": "TypeError",
            "message": "Net.__init__() missing 1 required positional argument: 'width'",
            "frames": [{"file": f"{WORKSPACE}/net.py", "line": 6, "name": "__init__"}],
            "node_id": None,
        },
        "worker": {"python": "3.11.9"},
    }


def test_a_model_that_could_not_be_constructed_is_a_4xx_not_a_crash(api):
    """The commonest trace failure there is — a model whose constructor needs an
    argument nobody passed. The backend has to answer the client, not fall over,
    and it must not claim a trace happened."""
    response = api.project(
        BRANCHED, "Net", file="validation_ctor_failed.py", trace_result=_construct_failure()
    )
    assert response.status_code == 400, response.text
    assert response.json()["detail"]["error"] == "trace_rejected"


def test_the_reason_the_model_would_not_build_reaches_the_user(api):
    """"Trace failed" is not actionable. "Net.__init__() missing 1 required positional
    argument: 'width'" is: it names the argument the user has to supply."""
    response = api.project(
        BRANCHED, "Net", file="validation_ctor_reason.py", trace_result=_construct_failure()
    )
    assert response.status_code == 400, response.text
    assert "missing 1 required positional argument" in response.text
