"""Graph edits as minimal source patches, and the edits that must be refused
(task.md M5).

Two claims are under test and they carry equal weight.

The first is *minimal*: changing one parameter changes one line. That is the
whole reason this plane is LibCST and not a regeneration of the file, so it is
asserted on the diff itself — one ``-`` line and one ``+`` line — and not on
"the result contains 256", which a rewrite of the whole module would also
satisfy.

The second is *honest*: an operation the source cannot express safely comes back
as a refusal naming a reason and a range, and the buffer is left alone. A canvas
that silently did nothing, or silently did something else, would be worse than
one that cannot be edited at all.

Every ``range`` a test sees is also replayed the way the extension replays it
(``_replay``), so a passing assertion about ``result.source`` is also an
assertion that the ranges handed to ``WorkspaceEdit`` reconstruct it.

No module-scope torch import and no model is ever instantiated (task.md §8.3);
every fixture class here is eight units wide.
"""

from __future__ import annotations

import ast
import difflib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


from api.routers import ide  # noqa: E402
from auth import CurrentUser, get_current_user  # noqa: E402
from models.graph_models import CodeParseRequest  # noqa: E402
from services.ide import code_apply, projection  # noqa: E402
from services.project_store import ProjectStore  # noqa: E402
from utils.code_parser import CodeParserService  # noqa: E402

FIXTURE_REPO = Path(__file__).parent.resolve().parent / "fixtures" / "ide_repo"
MODELS = FIXTURE_REPO / "models"

MLP = '''import torch.nn as nn


class Net(nn.Module):
    """Three layers and a comment, so a rewrite of the file is visible."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
'''

_PARSER = CodeParserService()


# ── helpers ──────────────────────────────────────────────────────────────────


def _project(source: str, cls: str, file: str = "models/net.py") -> projection.Projection:
    parsed = _PARSER.parse_pytorch_code(CodeParseRequest(code=source, file=file))
    entry = next(
        e for e in parsed["graphs"] if e.get("graph") is not None and e["name"] == cls
    )
    graph = entry["graph"]
    return projection.project(
        {
            "nodes": [node.model_dump() for node in graph.nodes],
            "edges": [edge.model_dump() for edge in graph.edges],
        },
        None,
        cls=cls,
        rel_file=file,
        source=source,
    )


def _apply(
    source: str, ops: Sequence[Dict[str, Any]], cls: str = "Net", file: str = "models/net.py"
) -> code_apply.ApplyResult:
    spanmap = _project(source, cls, file).spanmap
    result = code_apply.apply_operations(source, spanmap, ops, cls=cls, file=file)
    # Every result is checked against its own ranges: the extension never sees
    # `result.source`, it sees the edits, so a test that trusted the former would
    # pass while the product wrote garbage.
    assert _replay(source, result.edits) == result.source
    return result


def _offset(source: str, line: int, column: int) -> int:
    """A 1-based line / 0-based column as a character offset, the way VS Code
    resolves a Position against a document."""
    lines = source.splitlines(keepends=True)
    if line - 1 >= len(lines):
        return len(source)
    return sum(len(text) for text in lines[: line - 1]) + column


def _replay(source: str, edits: Sequence[Dict[str, Any]]) -> str:
    """Apply edits the way ``WorkspaceEdit`` does: against the original offsets,
    which is why they go on back to front."""
    out = source
    for edit in sorted(edits, key=lambda e: e["range"]["start"], reverse=True):
        start = _offset(source, *edit["range"]["start"])
        end = _offset(source, *edit["range"]["end"])
        out = out[:start] + edit["newText"] + out[end:]
    return out


def _diff_lines(before: str, after: str) -> Dict[str, List[str]]:
    diff = list(difflib.unified_diff(before.splitlines(True), after.splitlines(True), n=0))
    return {
        "-": [line for line in diff if line.startswith("-") and not line.startswith("---")],
        "+": [line for line in diff if line.startswith("+") and not line.startswith("+++")],
    }


def _refusal(result: code_apply.ApplyResult, op_id: str) -> Dict[str, Any]:
    found = [entry for entry in result.refused if entry["op_id"] == op_id]
    assert found, f"{op_id} was not refused; refusals were {result.refused}"
    return found[0]


def _graph(source: str, cls: str = "Net", file: str = "models/net.py"):
    semantic = _project(source, cls, file).document.semantic
    return (
        {(node.node_id, node.op) for node in semantic.nodes},
        {(edge.source_node_id, edge.target_node_id) for edge in semantic.edges},
    )


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    ide._store = ProjectStore(str(tmp_path / "ide.sqlite3"))
    app = FastAPI()
    app.include_router(ide.router)
    app.dependency_overrides[get_current_user] = lambda: CurrentUser(uid="local-dev-user")
    return TestClient(app)


# ── set_param: the minimality claim ──────────────────────────────────────────


def test_set_param_changes_exactly_one_line():
    """The acceptance criterion this whole module exists for.

    ``ast.unparse`` would have produced a correct file and a diff touching every
    line of it — including the docstring, which it would have re-quoted. One
    changed line is what makes a canvas edit reviewable in a source-control diff.
    """
    result = _apply(
        MLP,
        [{"op": "set_param", "op_id": "o1", "node_id": "fc1", "param": "out_features", "value": 256}],
    )
    assert result.refused == []
    assert len(result.edits) == 1, result.edits

    diff = _diff_lines(MLP, result.source)
    assert len(diff["-"]) == 1, diff
    assert len(diff["+"]) == 1, diff
    assert diff["+"][0].strip() == "+        self.fc1 = nn.Linear(784, 256)"
    # And nothing else in the file moved.
    assert result.source.count('"""') == MLP.count('"""')


def test_set_param_names_a_positional_argument_the_way_the_parser_read_it():
    """``out_features`` is the second positional argument of ``nn.Linear``. The
    node the canvas is showing calls it that because ``code_parser`` mapped it
    through ``inspect.signature``; if this module used a different order it would
    rewrite ``in_features`` and the graph would be silently wrong."""
    assert code_apply.positional_param_names("Linear")[:2] == ["in_features", "out_features"]
    result = _apply(
        MLP,
        [{"op": "set_param", "op_id": "o1", "node_id": "fc1", "param": "in_features", "value": 512}],
    )
    assert "nn.Linear(512, 128)" in result.source


def test_set_param_accepts_the_text_a_parameter_field_holds():
    """The inspector keeps every value as text, so ``256`` arrives as ``"256"``.
    Writing ``nn.Linear(784, '256')`` would be a bug the user meets at run time."""
    result = _apply(
        MLP,
        [{"op": "set_param", "op_id": "o1", "node_id": "fc1", "param": "out_features", "value": "256"}],
    )
    assert "nn.Linear(784, 256)" in result.source
    assert "'256'" not in result.source


def test_set_param_writes_a_keyword_the_author_never_wrote():
    result = _apply(
        MLP,
        [{"op": "set_param", "op_id": "o1", "node_id": "fc1", "param": "bias", "value": False}],
    )
    assert "nn.Linear(784, 128, bias=False)" in result.source


# ── add_layer + connect ──────────────────────────────────────────────────────


def test_add_layer_and_connect_insert_two_lines_and_leave_valid_python():
    result = _apply(
        MLP,
        [
            {"op": "add_layer", "op_id": "a", "node_id": "drop", "layer_op": "Dropout", "params": {"p": 0.1}},
            {"op": "connect", "op_id": "c", "source_node_id": "fc1", "target_node_id": "drop"},
        ],
    )
    assert result.refused == []
    diff = _diff_lines(MLP, result.source)
    assert diff["-"] == [], diff
    assert len(diff["+"]) == 2, diff
    ast.parse(result.source)

    nodes, edges = _graph(result.source)
    assert ("drop", "Dropout") in nodes
    assert ("fc1", "drop") in edges and ("drop", "act") in edges
    assert ("fc1", "act") not in edges


def test_a_layer_added_and_never_connected_is_refused_not_dropped():
    """An ``add_layer`` on its own names no place in the file. Writing it at the
    end of ``__init__`` and leaving it out of ``forward`` would put a layer in
    the model that never runs — the canvas would show it, the graph would not."""
    result = _apply(
        MLP,
        [{"op": "add_layer", "op_id": "a", "node_id": "drop", "layer_op": "Dropout", "params": {"p": 0.1}}],
    )
    assert result.edits == []
    assert _refusal(result, "drop")["reason"] == code_apply.UNSUPPORTED_OPERATION


def test_a_layer_anchored_on_the_input_boundary_runs_first():
    result = _apply(
        MLP,
        [
            {"op": "add_layer", "op_id": "a", "node_id": "pre", "layer_op": "Dropout", "params": {"p": 0.2}},
            {"op": "connect", "op_id": "c", "source_node_id": "input", "target_node_id": "pre"},
        ],
    )
    assert result.refused == []
    _, edges = _graph(result.source)
    assert ("input", "pre") in edges and ("pre", "fc1") in edges


def test_an_insertion_on_a_nested_call_changes_one_line_instead_of_adding_one():
    """``return self.act(self.fc(x))`` binds nothing, so there is no statement to
    insert after. Wrapping the producer's call in place is still one statement and
    still minimal — and it is the only form that is correct here."""
    source = (MODELS / "blocks.py").read_text(encoding="utf-8")
    result = _apply(
        source,
        [
            {"op": "add_layer", "op_id": "a", "node_id": "drop", "layer_op": "Dropout", "params": {"p": 0.1}},
            {"op": "connect", "op_id": "c", "source_node_id": "fc", "target_node_id": "drop"},
        ],
        cls="MyBlock",
        file="models/blocks.py",
    )
    assert result.refused == []
    assert "return self.act(self.drop(self.fc(x)))" in result.source
    _, edges = _graph(result.source, cls="MyBlock", file="models/blocks.py")
    assert ("fc", "drop") in edges and ("drop", "act") in edges


def test_the_compiled_insert_between_form_is_what_this_understands():
    """``semantic_actions.compile_plan_items`` expands an ``insert_between``
    action into ``remove_connection`` + ``add_layer`` + two ``connect``s. That
    expansion needs an ``ArchitectureGraph``, which this endpoint deliberately
    does not receive, so the compiled form is the contract — and the
    ``remove_connection`` in it must not come back as a refusal."""
    result = _apply(
        MLP,
        [
            {"op": "remove_connection", "op_id": "rc", "source_node_id": "fc1", "target_node_id": "act"},
            {"op": "add_layer", "op_id": "al", "node_id": "drop", "layer_op": "Dropout", "params": {"p": 0.1}},
            {"op": "connect", "op_id": "c1", "source_node_id": "fc1", "target_node_id": "drop"},
            {"op": "connect", "op_id": "c2", "source_node_id": "drop", "target_node_id": "act"},
        ],
    )
    assert result.refused == []
    _, edges = _graph(result.source)
    assert ("fc1", "drop") in edges and ("drop", "act") in edges


def test_removing_an_edge_on_its_own_is_refused():
    result = _apply(
        MLP,
        [{"op": "remove_connection", "op_id": "rc", "source_node_id": "fc1", "target_node_id": "act"}],
    )
    assert result.edits == []
    assert _refusal(result, "rc")["reason"] == code_apply.UNSUPPORTED_OPERATION


# ── remove / replace / rename ────────────────────────────────────────────────


def test_remove_layer_leaves_the_graph_the_deletion_meant():
    result = _apply(MLP, [{"op": "remove_layer", "op_id": "r", "node_id": "fc1"}])
    assert result.refused == []
    nodes, edges = _graph(result.source)
    assert nodes == {("input", "Input"), ("act", "ReLU"), ("fc2", "Linear"), ("output", "Output")}
    assert edges == {("input", "act"), ("act", "fc2"), ("fc2", "output")}
    # Both of fc1's lines went, and nothing else did.
    diff = _diff_lines(MLP, result.source)
    assert len(diff["-"]) == 2 and diff["+"] == [], diff


def test_remove_layer_in_the_middle_of_a_nested_call():
    source = (MODELS / "blocks.py").read_text(encoding="utf-8")
    result = _apply(
        source,
        [{"op": "remove_layer", "op_id": "r", "node_id": "fc"}],
        cls="MyBlock",
        file="models/blocks.py",
    )
    assert "return self.act(x)" in result.source
    assert "self.fc = " not in result.source


def test_replace_layer_rewrites_the_constructor_only():
    result = _apply(
        MLP,
        [{"op": "replace_layer", "op_id": "p", "node_id": "act", "new_layer_op": "gelu", "params": {}}],
    )
    assert result.refused == []
    assert len(result.edits) == 1
    assert "self.act = nn.GELU()" in result.source


def test_rename_layer_moves_the_node_and_nothing_else():
    """The D6 identity rule made visible: renaming ``self.fc1`` is one node
    removed and one added, with every other node and its position untouched."""
    before_nodes, _ = _graph(MLP)
    result = _apply(
        MLP, [{"op": "rename_layer", "op_id": "n", "node_id": "fc1", "new_node_id": "proj"}]
    )
    assert result.refused == []
    after_nodes, after_edges = _graph(result.source)
    assert before_nodes - after_nodes == {("fc1", "Linear")}
    assert after_nodes - before_nodes == {("proj", "Linear")}
    assert ("proj", "act") in after_edges


# ── the refusals ─────────────────────────────────────────────────────────────


def test_an_op_inside_a_for_loop_is_refused_as_a_control_flow_region():
    """``for block in self.blocks: h = block(h)`` is one card on the canvas and
    ``depth`` statements at run time. There is no single statement whose edit
    means what the user meant."""
    source = (MODELS / "resid.py").read_text(encoding="utf-8")
    spanmap = _project(source, "Net", "models/resid.py").spanmap
    assert "python@Net#0" in spanmap, spanmap
    result = _apply(
        source,
        [{"op": "set_param", "op_id": "o1", "node_id": "python@Net#0", "param": "dim", "value": 1}],
        file="models/resid.py",
    )
    assert result.edits == []
    assert _refusal(result, "o1")["reason"] == code_apply.CONTROL_FLOW_REGION


def test_a_layer_called_inside_a_branch_cannot_be_removed():
    source = (MODELS / "gate.py").read_text(encoding="utf-8")
    result = _apply(
        source,
        [{"op": "remove_layer", "op_id": "r", "node_id": "up"}],
        cls="Gate",
        file="models/gate.py",
    )
    assert result.edits == []
    assert _refusal(result, "r")["reason"] == code_apply.CONTROL_FLOW_REGION


def test_a_parameter_written_as_an_expression_is_revealed_not_overwritten():
    """``nn.Linear(cfg.width, cfg.hidden)`` says the width comes from a config.
    Writing ``32`` there would delete that decision without telling anyone, so the
    refusal carries the range of the expression instead and the IDE reveals it."""
    source = (MODELS / "cfg.py").read_text(encoding="utf-8")
    result = _apply(
        source,
        [{"op": "set_param", "op_id": "o1", "node_id": "proj", "param": "out_features", "value": 32}],
        cls="Configured",
        file="models/cfg.py",
    )
    assert result.edits == []
    refusal = _refusal(result, "o1")
    assert refusal["reason"] == code_apply.NON_LITERAL_VALUE
    span = refusal["span"]
    line = source.splitlines()[span["start"][0] - 1]
    assert line[span["start"][1] : span["end"][1]] == "cfg.hidden"


def test_a_node_with_no_declaration_is_refused_as_dynamic_construction():
    """``nn.ModuleList([... for _ in range(depth)])`` builds its children in a
    comprehension. The trace tier names them ``blocks.0``; no ``self.blocks.0 =``
    statement exists, so there is nothing to patch."""
    source = (MODELS / "resid.py").read_text(encoding="utf-8")
    result = _apply(
        source,
        [{"op": "set_param", "op_id": "o1", "node_id": "blocks.0", "param": "out_features", "value": 16}],
        file="models/resid.py",
    )
    assert result.edits == []
    assert _refusal(result, "o1")["reason"] == code_apply.DYNAMIC_CONSTRUCTION


def test_a_refused_batch_still_applies_the_operations_that_can_be_made():
    """A refusal is per-operation. Dropping the whole batch would make one
    impossible edit cost the user the four that were fine."""
    source = (MODELS / "resid.py").read_text(encoding="utf-8")
    result = _apply(
        source,
        [
            {"op": "rename_layer", "op_id": "ok", "node_id": "l1", "new_node_id": "proj"},
            {"op": "set_param", "op_id": "no", "node_id": "python@Net#0", "param": "dim", "value": 1},
        ],
        file="models/resid.py",
    )
    assert result.edits, "the applicable operation was dropped with the refused one"
    assert "self.proj = nn.Linear(width, width)" in result.source
    assert [entry["op_id"] for entry in result.refused] == ["no"]


def test_an_unknown_operation_is_refused_rather_than_ignored():
    result = _apply(MLP, [{"op": "instantiate_template", "op_id": "t", "template_id": "vit"}])
    assert result.edits == []
    assert _refusal(result, "t")["reason"] == code_apply.UNSUPPORTED_OPERATION


# ── the HTTP contract ────────────────────────────────────────────────────────


def _apply_body(source: str, ops: Sequence[Dict[str, Any]], digest: Optional[str] = None):
    return {
        "file": "models/net.py",
        "cls": "Net",
        "source": source,
        "source_digest": digest or projection.digest_of(source),
        "spanmap": _project(source, "Net").spanmap,
        "ops": list(ops),
    }


def test_apply_endpoint_returns_edits_and_a_fresh_spanmap(client: TestClient):
    response = client.post(
        "/api/ide/apply",
        json=_apply_body(
            MLP,
            [
                {"op": "add_layer", "op_id": "a", "node_id": "drop", "layer_op": "Dropout", "params": {"p": 0.1}},
                {"op": "connect", "op_id": "c", "source_node_id": "fc1", "target_node_id": "drop"},
            ],
        ),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["edits"]) == 2
    assert body["refused"] == []
    # The spans below the insertion have moved, and the answer says so — without
    # this the client's next edit would be applied to lines that shifted.
    assert "drop" in body["spanmap"]
    assert body["spanmap"]["fc2"]["start"][0] > _project(MLP, "Net").spanmap["fc2"]["start"][0]
    assert body["source_digest"] != projection.digest_of(MLP)


def test_a_stale_spanmap_is_a_409_and_nothing_is_edited(client: TestClient):
    """The client holds spans computed from the projection it is showing. If the
    buffer has moved on, every one of them points at the wrong line — there is no
    partial application worth making, so this fails before anything is planned."""
    response = client.post(
        "/api/ide/apply",
        json=_apply_body(
            MLP,
            [{"op": "set_param", "op_id": "o1", "node_id": "fc1", "param": "out_features", "value": 256}],
            digest=projection.digest_of("# something else\n"),
        ),
    )
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "spanmap_stale"


def test_the_apply_endpoint_writes_no_file(client: TestClient, tmp_path: Path):
    """task.md §8.5: FastAPI's only workspace write is ``state.sqlite3``. The
    edits go back to the extension, which owns every ``WorkspaceEdit``."""
    target = tmp_path / "net.py"
    target.write_text(MLP, encoding="utf-8")
    body = _apply_body(
        MLP, [{"op": "set_param", "op_id": "o1", "node_id": "fc1", "param": "out_features", "value": 256}]
    )
    body["file"] = str(target)
    response = client.post("/api/ide/apply", json=body)
    assert response.status_code == 200, response.text
    assert target.read_text(encoding="utf-8") == MLP


def test_a_buffer_that_does_not_parse_is_a_400(client: TestClient):
    broken = MLP.replace("def forward(self, x):", "def forward(self, x)")
    body = _apply_body(MLP, [{"op": "remove_layer", "op_id": "r", "node_id": "fc1"}])
    body["source"] = broken
    body["source_digest"] = projection.digest_of(broken)
    response = client.post("/api/ide/apply", json=body)
    assert response.status_code == 400, response.text
    assert response.json()["detail"]["error"] == "syntax_error"


def test_a_sequential_child_cannot_be_deleted_by_deleting_its_statement():
    """Every child of `nn.Sequential(...)` lives inside one `self.net = ...`
    statement. Dropping that statement to delete one card would delete the whole
    block — so it is refused, and the card stays where the file put it."""
    source = '''import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))

    def forward(self, x):
        return self.net(x)
'''
    spanmap = _project(source, "Net").spanmap
    # The projection names a Sequential child after the op it is, not after the
    # attribute holding the block — `net` is the only attribute here.
    child = "linear@Net#0"
    assert spanmap[child]["kind"] == "call"
    result = _apply(source, [{"op": "remove_layer", "op_id": "r", "node_id": child}])
    assert result.edits == []
    assert _refusal(result, "r")["reason"] == code_apply.UNSUPPORTED_OPERATION
    # But its parameters are still editable, because the child call is its own span.
    edited = _apply(
        source,
        [{"op": "set_param", "op_id": "o", "node_id": child, "param": "out_features", "value": 32}],
    )
    assert len(edited.edits) == 1
    assert "nn.Linear(8, 32)" in edited.source


def test_a_rename_onto_an_existing_attribute_is_refused():
    result = _apply(
        MLP, [{"op": "rename_layer", "op_id": "n", "node_id": "fc1", "new_node_id": "fc2"}]
    )
    assert result.edits == []
    assert _refusal(result, "n")["reason"] == code_apply.UNSUPPORTED_OPERATION


@pytest.mark.parametrize("anchor,source,expected_edges", [
    ("fc1", MLP, {("fc1", "extra1"), ("extra1", "extra2"), ("extra2", "act")}),
    ("input", MLP, {("input", "extra1"), ("extra1", "extra2"), ("extra2", "fc1")}),
    ("fc", "from torch import nn\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(8, 4)\n    def forward(self, x):\n        return self.fc(x)\n", {("fc", "extra1"), ("extra1", "extra2"), ("extra2", "output")}),
], ids=["assigned", "input", "nested-return"])
def test_consecutive_new_layers_follow_each_other_in_source(anchor, source, expected_edges):
    width = {"fc1": 128, "input": 784, "fc": 4}[anchor]
    ops = [
        {"op": "add_layer", "node_id": "extra1", "layer_op": "Linear", "params": {"in_features": width, "out_features": width}},
        {"op": "connect", "source_node_id": anchor, "target_node_id": "extra1"},
        {"op": "add_layer", "node_id": "extra2", "layer_op": "Linear", "params": {"in_features": width, "out_features": width}},
        {"op": "connect", "source_node_id": "extra1", "target_node_id": "extra2"},
    ]
    result = _apply(source, ops)
    assert result.refused == []
    _, edges = _graph(result.source)
    assert expected_edges <= edges
    assert result.source.count('self.extra1 = nn.Linear(') == 1
    assert result.source.count('self.extra2 = nn.Linear(') == 1
    ast.parse(result.source)


def test_new_layer_cannot_anchor_to_an_unplaced_new_layer():
    result = _apply(MLP, [
        {"op": "add_layer", "node_id": "extra1", "layer_op": "ReLU"},
        {"op": "add_layer", "node_id": "extra2", "layer_op": "ReLU"},
        {"op": "connect", "source_node_id": "extra1", "target_node_id": "extra2"},
    ])
    assert result.refused
    assert result.source == MLP
