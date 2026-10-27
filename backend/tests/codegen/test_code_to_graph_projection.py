"""The code→graph direction: the user edits Python and the canvas follows (task.md D6/M3).

``test_projection_identity.py`` pins the *identity* rules — which ids survive an
edit — and ``test_ide_router.py`` pins the HTTP shape. Neither says what the
canvas actually ends up looking like after the edits people spend their day
making: adding a layer, forgetting to call one, deleting one, renaming one,
retyping a constructor argument, and breaking the file halfway through all of
that. Those are asserted here as a diff of *cards and wires*, because that is
what the user sees.

This is the STATIC tier throughout — ``projection.project(..., trace=None)``,
i.e. exactly what ``POST /api/ide/parse`` answers while the user is typing. The
extension only re-runs the trace on save (``extension/src/projection.ts:304``),
so the static tier is the only thing that can put a card on the canvas mid-edit;
the traced half of the same seam lives in ``test_ingest_worker.py``. Going
through ``services.ide.projection`` rather than HTTP is deliberate: a failure
here should name the rule, not the route.

No module-scope ``import torch`` and nothing here is executed (task.md §8.3):
every source below is read, never run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pytest


from models.graph_models import CodeParseRequest  # noqa: E402
from services.ide import projection  # noqa: E402
from utils.code_parser import CodeParserService  # noqa: E402

RESID = Path(__file__).parent.resolve().parent / "fixtures" / "ide_repo" / "models" / "resid.py"


def _project(source: str, cls: str = "Net", rel_file: str = "models/resid.py"):
    parsed = CodeParserService().parse_pytorch_code(
        CodeParseRequest(code=source, file=rel_file)
    )
    entry = next(e for e in parsed["graphs"] if e["name"] == cls)
    graph = entry["graph"]
    return projection.project(
        {
            "nodes": [n.model_dump() for n in graph.nodes],
            "edges": [e.model_dump() for e in graph.edges],
        },
        cls=cls,
        rel_file=rel_file,
        source=source,
    )


def _ids(result) -> Set[str]:
    return {n.node_id for n in result.document.semantic.nodes}


def _kinds(result) -> Dict[str, str]:
    return {n.node_id: n.kind for n in result.document.semantic.nodes}


def _edges(result) -> List[Tuple[str, str]]:
    return sorted(
        (e.source_node_id, e.target_node_id) for e in result.document.semantic.edges
    )


def _params(result, node_id: str) -> Dict[str, Any]:
    return next(n.params for n in result.document.semantic.nodes if n.node_id == node_id)


def _touching(result, node_id: str) -> List[Tuple[str, str]]:
    return [pair for pair in _edges(result) if node_id in pair]


def _line_of(source: str, needle: str) -> int:
    """The 1-based line the edit was written on, so no span assertion is a magic
    number that goes stale the next time the fixture gains a docstring line."""
    for index, line in enumerate(source.splitlines(), start=1):
        if needle in line:
            return index
    raise AssertionError(f"{needle!r} is not in the source under test")


@pytest.fixture()
def source() -> str:
    return RESID.read_text(encoding="utf-8")


# ── adding ────────────────────────────────────────────────────────────────────

def test_a_layer_added_to_forward_is_one_new_card_wired_where_it_was_written(source: str):
    """Declaring a layer and calling it must draw it *in the pipeline*.

    If the new card arrived unwired — or the wire it interrupts were left in
    place — the canvas would show the user a model that is not the one on
    screen in their editor: either a card floating beside the graph, or a
    phantom ``blocks → l2`` shortcut past a layer their tensors really pass
    through. The whole promise of the pivot is that the picture is the file.
    """
    before = _project(source)
    edited = source.replace(
        "        self.l2 = nn.Linear(width, width)",
        "        self.l2 = nn.Linear(width, width)\n        self.head = nn.Linear(width, width)",
    ).replace(
        "        return x + self.l2(h)",
        "        h = self.head(h)\n        return x + self.l2(h)",
    )
    after = _project(edited)

    # Exactly one card appeared and nothing else changed identity.
    assert _ids(after) - _ids(before) == {"head"}
    assert _ids(before) - _ids(after) == set()

    # It was spliced into the wire it was written between, not hung off the side.
    assert ("python@Net#0", "head") in _edges(after)
    assert ("head", "l2") in _edges(after)
    assert ("python@Net#0", "l2") in _edges(before)
    assert ("python@Net#0", "l2") not in _edges(after)

    # And "reveal in editor" on the new card lands on the line that declared it.
    span = after.spanmap["head"]
    assert span["kind"] == "decl"
    assert span["start"][0] == _line_of(edited, "self.head = nn.Linear")


def test_a_submodule_never_called_in_forward_is_still_a_card_with_no_wires(source: str):
    """Half-finished work must be visible, not silently dropped.

    Declaring ``self.extra`` and not yet calling it is the normal middle of an
    edit. If the parser answered with no card, the user would read that as "my
    line didn't register" and start debugging the tool instead of their model.
    The disconnected card is the honest answer: the layer exists, and nothing
    flows through it yet.
    """
    before = _project(source)
    edited = source.replace(
        "        self.l2 = nn.Linear(width, width)",
        "        self.l2 = nn.Linear(width, width)\n        self.extra = nn.Linear(width, width)",
    )
    after = _project(edited)

    assert _ids(after) - _ids(before) == {"extra"}
    assert _touching(after, "extra") == [], "an uncalled layer must not invent wiring"
    assert _edges(after) == _edges(before), "declaring a layer must not re-route the model"

    span = after.spanmap["extra"]
    assert span["kind"] == "decl"
    assert span["start"][0] == _line_of(edited, "self.extra = nn.Linear")


def test_an_annotated_submodule_declaration_is_a_card_like_any_other(source: str):
    """The spelling of the declaration must not decide whether the layer exists.

    Was ``xfail`` until the parser was taught ``ast.AnnAssign``. It walked only
    ``ast.Assign``, so an ``ast.AnnAssign`` — the type-annotated spelling — is
    invisible to it. A user who annotates their attributes gets no card at all
    while typing, and after a save the trace does produce one but with a span
    pointing into ``torch/nn/modules/linear.py``, so "reveal in editor" jumps
    out of their project entirely.

    Note the disagreement this test names: ``projection.init_attributes``
    already treats the annotated attribute as a declared module, so the two
    halves of the same tier hold different beliefs about whether it exists.
    """
    edited = source.replace(
        "        self.l2 = nn.Linear(width, width)",
        "        self.l2 = nn.Linear(width, width)\n"
        "        self.extra: nn.Module = nn.Linear(width, width)",
    )
    # The projection layer already believes the attribute is a declared module…
    assert "extra" in projection.init_attributes(edited, "Net")

    # …but the parser it consumes never emits a node for it.
    after = _project(edited)
    assert "extra" in _ids(after)
    assert _touching(after, "extra") == []
    assert after.spanmap["extra"]["kind"] == "decl"
    assert after.spanmap["extra"]["start"][0] == _line_of(edited, "self.extra: nn.Module")


# ── removing and renaming ─────────────────────────────────────────────────────

def test_deleting_a_layer_removes_that_card_and_leaves_no_wire_pointing_at_it(source: str):
    """A deleted layer must take its wires with it and close the gap behind it.

    An edge left pointing at a card that no longer exists is not cosmetic: the
    canvas resolves edges by node id, validation counts it as a dangling tail,
    and codegen writes a call to an attribute the file no longer declares. The
    user deleted one line and got a model that cannot be exported.
    """
    before = _project(source)
    edited = source.replace("        self.l2 = nn.Linear(width, width)\n", "").replace(
        "        return x + self.l2(h)", "        return x + h"
    )
    after = _project(edited)

    assert _ids(before) - _ids(after) == {"l2"}
    assert _ids(after) - _ids(before) == set(), "deleting a layer must not invent one"
    assert _touching(after, "l2") == []
    assert "l2" not in after.spanmap, "a card that is gone must not keep a source span"

    # The two neighbours of the deleted card are now wired to each other.
    assert ("python@Net#0", "add@Net#0") in _edges(after)


def test_renaming_an_attribute_moves_every_wire_onto_the_new_card(source: str):
    """A rename is one card out, one card in — and the *same* position in the graph.

    ``test_projection_identity.py`` pins the id diff; what matters to the user is
    that the replacement lands where the old card was. A rename that dropped the
    wiring would turn a working model into a disconnected one on a keystroke that
    changed nothing about what the code computes.
    """
    before = _project(source)
    edited = source.replace("self.l1", "self.proj")
    assert edited.count("self.proj") == 2, "rename must touch the declaration and the call"
    after = _project(edited)

    assert _ids(before) - _ids(after) == {"l1"}
    assert _ids(after) - _ids(before) == {"proj"}
    assert _touching(after, "l1") == []

    # The new id inherits the old one's place in the dataflow, wire for wire.
    assert _touching(before, "l1"), "the renamed layer must have been wired to begin with"
    assert [
        tuple("proj" if end == "l1" else end for end in pair)
        for pair in _touching(before, "l1")
    ] == _touching(after, "proj")


def test_changing_a_constructor_argument_keeps_the_card_and_moves_its_parameters(source: str):
    """Retyping ``8`` as ``16`` must edit the card, not replace it.

    Node ids key the saved layout, the debugger's overlays and any AI turn in
    flight. If widening a layer minted a new id, every parameter tweak would
    throw away the position the user dragged that card to — and the shapes shown
    downstream would still be the old ones, because nothing told the canvas the
    statement changed.
    """
    before = _project(source)
    edited = source.replace(
        "        self.l1 = nn.Linear(width, width)",
        "        self.l1 = nn.Linear(width, 16)",
    )
    after = _project(edited)

    # Same graph, structurally: nothing moved and nothing was re-wired.
    assert _ids(after) == _ids(before)
    assert _edges(after) == _edges(before)

    # The parameters the canvas renders followed the source.
    assert _params(before, "l1")["out_features"] == 8
    assert _params(after, "l1")["out_features"] == 16
    assert _params(after, "l1")["_raw_call"] == "nn.Linear(width, 16)"

    # The digest is what tells a stale consumer this statement means something
    # new; only the edited card's may move.
    assert after.spanmap["l1"]["digest"] != before.spanmap["l1"]["digest"]
    assert {k: v for k, v in after.spanmap.items() if k != "l1"} == {
        k: v for k, v in before.spanmap.items() if k != "l1"
    }


# ── containers ────────────────────────────────────────────────────────────────

def test_the_module_list_is_one_card_and_the_loop_that_walks_it_is_another(source: str):
    """``self.blocks = nn.ModuleList(...)`` is one line, so it gets one card.

    Two things would hurt the user if this stopped holding. If the declaration
    produced no card, the ``self.blocks`` line would have nothing on the canvas
    to click, and an edit aimed at it would have no span to write through. If it
    produced one card per member, a twelve-block model would bury its topology
    under twelve identical boxes the source never names — and the AST tier
    cannot know there are twelve without running the code.

    The loop body is deliberately a *separate* opaque card: ``block`` is a loop
    variable, not a ``self.<attr>``, so the parser carries its source text rather
    than pretending to have typed it. The traced tier's version of this is in
    ``test_ingest_worker.py``.
    """
    result = _project(source)
    ids = _ids(result)

    assert "blocks" in ids
    assert [i for i in ids if i.startswith("blocks.")] == [], "the list is not expanded"
    assert _params(result, "blocks")["_raw_call"].startswith("nn.ModuleList(")

    # The declaration carries no wiring; the statement that walks the list does.
    assert _touching(result, "blocks") == []
    assert _params(result, "python@Net#0")["_raw_call"] == "block(h)"
    assert ("l1", "python@Net#0") in _edges(result)
    assert ("python@Net#0", "l2") in _edges(result)

    assert result.spanmap["blocks"]["kind"] == "decl"
    assert result.spanmap["blocks"]["start"][0] == _line_of(source, "self.blocks = nn.ModuleList")


def test_the_member_list_is_carried_on_a_key_no_constructor_will_see(source: str):
    """The one card standing for the list has to say what is in it — and must not
    say it in a way that gets passed to ``nn.ModuleList``.

    ``params`` serves two audiences: it is what the card draws, and it is what
    ``core/layer_builder.py::build_layer`` forwards to a torch constructor after
    dropping ``name`` and every ``_``-prefixed key. The members were recorded under
    ``modules`` — ``nn.ModuleList``'s own first argument — so they were the ONE key
    that survived that filter, and opening a correct model drew a red panel reading
    ``ModuleList — build error: dict is not a Module subclass``. The underscore is
    this codebase's word for "not a constructor argument"; ``cardParams.ts`` reads
    the prefixed key to draw "2 × Linear".
    """
    params = _params(_project(source), "blocks")

    assert "modules" not in params
    assert [m["op"] for m in params["_modules"]] == ["Linear", "Linear"]


def test_the_container_builds_from_the_parameters_this_tier_gave_it(source: str):
    """The same assertion from the other end: the exact call the verify path makes,
    on the exact bag this projection produced. torch is imported inside the test so
    the file's no-torch-at-module-scope rule still holds."""
    from core.layer_builder import build_layer

    build_layer("ModuleList", dict(_params(_project(source), "blocks")))


def test_the_module_list_is_a_container_frame_and_not_an_executable_layer(source: str):
    """A Python container is grouping, not a layer, and the graph must say so.

    ``kind`` is the switch the rest of the backend reads: an edge-free
    ``kind='module'`` node is a canvas frame — dropped from the codegen payload
    (``verification_service.py``) and exempt from the wiring check
    (``canonical_validation.py``). Previously ``blocks`` was ``kind='layer'`` and
    had no incident edges, so the shape engine reported it as
    ``disconnected_node`` with "Connect this node to a preceding layer", and the
    AI repair loop has already once acted on exactly that advice by deleting the
    blocks (the resnet18 incident the two files above record). Meanwhile
    ``spec_for('modulelist')`` is ``None``, so if anything ever took the advice
    and wired it, export would fail the way ``nn.BasicBlock`` did.
    """
    result = _project(source)
    assert _kinds(result)["blocks"] == "module"
    # A frame, not a step of the pipeline: it must stay edge-free to qualify.
    assert _touching(result, "blocks") == []


def test_a_local_custom_class_named_modulelist_is_not_a_framework_container(source):
    source += '''
class ModuleList(nn.Module):
    def forward(self, x):
        return x.relu()
'''
    source = source.replace("nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])", "ModuleList()")
    result = _project(source)
    assert _kinds(result)["blocks"] == "layer"
    assert _params(result, "blocks")["_class"] == "ModuleList"
    assert _params(result, "blocks")["_import_path"] == "models.resid"


@pytest.mark.parametrize("members", [
    "[nn.Linear(width, width), nn.ReLU()]",
    "[nn.Linear(width, width) for _ in range(depth)]",
    "[]",
    "unknown_members",
])
def test_static_container_kind_does_not_require_known_or_materialized_members(source, members):
    source = source.replace("[nn.Linear(width, width) for _ in range(depth)]", members)
    result = _project(source)
    assert _kinds(result)["blocks"] == "module"
    assert not any(node.startswith("blocks.") for node in _ids(result))
    assert _touching(result, "blocks") == []
    assert _params(result, "blocks")["_shape_provenance"] == "static"
    assert result.spanmap["blocks"]["kind"] == "decl"
    data = result.document.view.layout["node_data"]["blocks"]
    assert data["moduleMembers"] == []
    assert data["moduleLayerCount"] == 0
    assert data["expanded"] is False
    if members == "unknown_members":
        assert "_modules" not in _params(result, "blocks")


def test_container_consumers_exclude_only_the_frame_and_keep_real_disconnected_errors(source):
    from core.shape_inference import infer_node_shapes
    from services.canonical_validation import CanonicalValidationService
    from services.graph_document import graph_document_v2
    from services.verification_service import architecture_graph_to_runtime_payload

    source = source.replace("        self.l2 =", "        self.unused = nn.ReLU()\n        self.l2 =")
    projected = _project(source).document
    parsed = CodeParserService().parse_pytorch_code(CodeParseRequest(code=source, file="models/resid.py"))
    raw = next(e["graph"] for e in parsed["graphs"] if e["name"] == "Net")
    flat_validation = infer_node_shapes(raw, [2, 8])
    assert not any(error["nodeId"] == "blocks" for error in flat_validation["errors"])
    assert "blocks" not in flat_validation["shapes"]
    legacy = graph_document_v2(raw.model_dump(), graph_id="legacy", name="Net", revision=0)
    for document in (projected, legacy):
        graph = document.semantic
        before = graph.model_dump()
        assert next(n for n in graph.nodes if n.node_id == "blocks").kind == "module"
        validation = CanonicalValidationService().validate(graph, strict=True)
        assert validation.node_results["blocks"].status != "error"
        assert not validation.node_results["blocks"].suggested_repairs
        assert validation.node_results["blocks"].output_shape is None
        assert validation.node_results["unused"].status == "error"
        assert "no input connections" in validation.node_results["unused"].message
        payload = architecture_graph_to_runtime_payload(graph)
        assert {n["id"] for n in payload["nodes"]} == {n.node_id for n in graph.nodes} - {"blocks"}
        assert [(e["source"], e["target"]) for e in payload["edges"]] == [
            (e.source_node_id, e.target_node_id) for e in graph.edges
        ]
        assert graph.model_dump() == before


def test_a_wired_container_is_not_silently_removed_from_the_runtime_payload(source):
    from api.routers.validate import ValidationRequest, _build_canonical_graph, validate_shapes
    from core.shape_inference import infer_node_shapes
    from services.verification_service import architecture_graph_to_runtime_payload, architecture_graph_runtime_support

    # Calling a ModuleList itself is unsupported, but its written call must not
    # turn into invented member wiring or disappear as if it were a free frame.
    source = source.replace("        for block in self.blocks:\n            h = block(h)", "        h = self.blocks(h)")
    result = _project(source)
    assert _kinds(result)["blocks"] == "module"
    assert _touching(result, "blocks") == [("blocks", "l2"), ("l1", "blocks")]
    payload = architecture_graph_to_runtime_payload(result.document.semantic)
    assert "blocks" in {n["id"] for n in payload["nodes"]}
    assert sorted((e["source"], e["target"]) for e in payload["edges"]) == _edges(result)
    supported, reasons = architecture_graph_runtime_support(result.document.semantic)
    assert not supported
    assert any("blocks: semantic module" in reason for reason in reasons)
    parsed = CodeParserService().parse_pytorch_code(CodeParseRequest(code=source, file="models/resid.py"))
    raw = next(e["graph"] for e in parsed["graphs"] if e["name"] == "Net")
    flat_validation = infer_node_shapes(raw, [2, 8])
    assert "blocks" not in flat_validation["shapes"]
    assert any(error["nodeId"] == "blocks" for error in flat_validation["errors"])
    request = ValidationRequest(**raw.model_dump())
    canonical = _build_canonical_graph(request)
    assert [(e.source_node_id, e.target_node_id) for e in canonical.edges] == [
        (e.source, e.target) for e in raw.edges
    ]
    validation = validate_shapes(request)
    assert "blocks" not in validation["shapes"]
    assert any(item["nodeId"] == "blocks" for item in validation["unchecked"])


# ── the buffer that stops parsing ─────────────────────────────────────────────

def test_a_half_typed_buffer_is_refused_by_line_and_the_graph_comes_back_after(source: str):
    """Mid-edit the file is usually invalid, and that must be a *named* refusal.

    The extension turns this into a diagnostic on the reported line and re-posts
    the last good projection dimmed (``extension/src/projection.ts:728``). Two
    things make that possible and both are asserted here: the parse must raise
    rather than return a half-built graph — a partial document would quietly
    delete every card below the broken line — and it must carry the line number,
    or the squiggle lands on the wrong statement. The recovery half matters just
    as much: an unclosed paren must leave nothing behind, so the graph the user
    gets back when they finish typing is the graph they had.
    """
    before = _ids(_project(source))

    broken = source.replace("        h = self.l1(x)", "        h = self.l1(x")
    assert broken != source
    with pytest.raises(SyntaxError) as caught:
        _project(broken)
    assert caught.value.lineno == _line_of(broken, "h = self.l1(x")

    assert _ids(_project(source)) == before
