"""One experiment, projected into the document the canvas already knows how to draw.

The research graph stays the source of truth; this only *renders* it, into the same
`GraphDocumentV2` the per-class canvas consumes. That is the whole reason there is one
canvas rather than two: a project card and a layer card are the same kind of shape at
different depths, so the project level and the model level share one surface, one
selection model and one navigation path.

Two mappings carry it:

* every non-model research node becomes a flat card carrying `custom_code` with
  `params["_kind"]` naming the research kind. That is exactly how the existing canvas
  already draws opaque user code, so **no new shape type is needed** — `LayerShapeUtil`
  renders these unchanged;
* every model node becomes a **module frame**, collapsed by default, whose children are
  that class's layers from the Phase 2 parser, namespaced under the frame id so two
  models in one experiment cannot collide on `l1` or `input`.

**Only expanded frames project their contents.** That is both the interaction and the
scale mechanism: a collapsed frame contributes exactly one shape, so a five-hundred-file
project is an experiment rail and a dozen cards until the user opens something.

Static only. Nothing here imports or runs the user's model; a frame's layers come from
the same AST parser `/api/ide/parse` uses.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field

from research.validatable import verdict_for
from research.contracts import (
    CODE_STRUCTURE_RELATIONSHIPS,
    DATAFLOW_RELATIONSHIPS,
    ResearchGraph,
    ResearchNode,
)

#: Top-level spacing. Wider than the per-class canvas's, because a project card
#: carries a file path and a kind badge, and because edges have to bend around an
#: expanded frame rather than through it.
#: Kinds whose BODY is worth drawing inside a frame. A trainer and an evaluator are
#: where the dataflow actually is; a configuration or a dataset is a thing, not a
#: sequence, and opening one would draw a frame around a single box.
FRAMED_FUNCTION_KINDS = {"trainer", "evaluator"}

STRIDE_X = 420.0
STRIDE_Y = 260.0

#: Where a frame's children start, relative to the frame's own origin.
CHILD_INSET_X = 24.0
CHILD_INSET_Y = 64.0
CHILD_PITCH_Y = 96.0


class DocumentNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: str
    position: Dict[str, float]
    data: Dict[str, Any]
    parentNode: Optional[str] = None


class DocumentEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source: str
    target: str
    label: str = ""
    data: Dict[str, Any] = Field(default_factory=dict)


class ExperimentDocument(BaseModel):
    """What the page draws, plus the spanmap that makes every card clickable."""

    model_config = ConfigDict(extra="forbid")

    graph_id: str
    experiment_id: str
    name: str
    nodes: List[DocumentNode]
    edges: List[DocumentEdge]
    #: node id → `{file, start, end, digest}`, in the shape the extension's existing
    #: navigation already reads, so click-to-source needs no second format.
    spanmap: Dict[str, Dict[str, Any]]
    #: Frame ids currently open. Echoed back so the page and host agree.
    expanded: List[str]


def _rank(graph: ResearchGraph) -> Dict[str, int]:
    """Longest-path layering over dataflow, so the pipeline reads left to right.

    Structure edges (`contains`, `imports`) are excluded deliberately: they say where
    code lives, not how data moves, and including them would rank every definition
    under its file and collapse the pipeline into a column.
    """
    flow = [e for e in graph.edges if e.relationship in DATAFLOW_RELATIONSHIPS]
    incoming: Dict[str, List[str]] = {n.id: [] for n in graph.nodes}
    for edge in flow:
        if edge.to_node_id in incoming:
            incoming[edge.to_node_id].append(edge.from_node_id)

    rank: Dict[str, int] = {}

    def depth(node_id: str, seen: Set[str]) -> int:
        if node_id in rank:
            return rank[node_id]
        if node_id in seen:
            # A cycle in dataflow is unusual but not impossible; cut rather than
            # recurse, so one odd graph cannot hang the projection.
            return 0
        seen.add(node_id)
        parents = incoming.get(node_id) or []
        value = 0 if not parents else 1 + max(depth(p, seen) for p in parents)
        seen.discard(node_id)
        rank[node_id] = value
        return value

    for node in graph.nodes:
        depth(node.id, set())
    return rank


def _edge_label(edge: Any) -> str:
    """What the edge says, in the words the source used.

    `builds` is true and says nothing a reader could not have guessed. The line it
    came from says where the thing lives, what holds it and how many there are:

        h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

    so the edge reads `transformer.h ×config.n_layer`. The canvas is not a
    replacement for the code — it is a way to see what the code already says, and
    an edge a reader can map back to a line is the difference between the two.

    The count is the expression AS WRITTEN. `config.n_layer` is 12 only once a
    config exists, and drawing 12 would claim a number nothing here ran to find.
    """
    attr = edge.params.get("attr") if edge.params else ""
    if not attr:
        return edge.relationship
    count = edge.params.get("count") or ""
    return f"{attr} \u00d7{count}" if count else attr


def _readable(value: Any) -> Any:
    """A value a card can print on one line.

    Numbers, strings and booleans pass through — the canvas renders them itself.
    A list becomes its members joined, because `['nn.Module']` drawn raw is
    punctuation the reader has to parse. Anything still structured becomes its own
    repr rather than "[object Object]", which is what JavaScript prints for an
    object handed to a string context and tells the reader nothing at all.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return repr(value)


def _card_data(
    node: ResearchNode,
    entries: Set[str],
    terminals: Set[str],
    configurations: Optional[Sequence[Dict[str, Any]]] = None,
    held: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The card's rendered payload.

    `_kind`, `_provenance` and `_reason` are what let the canvas draw a guess
    differently from a fact and say *why* on hover — the only way a user can tell a
    rule fired wrongly, and therefore the only way they know to correct it.
    """
    # The `_`-prefixed keys go inside `parameters`, not beside it. That is the one
    # per-node channel that survives every plane — the card model is handed
    # `data.parameters` and nothing else, so a key placed beside it reaches the
    # canvas and then vanishes. `utils/code_parser.py::_provenance` states the same
    # rule for the per-class tier; this is the project tier obeying it.
    parameters: Dict[str, Any] = {}
    for key, value in node.params.items():
        if key.startswith("_"):
            continue
        if key == "settings" and isinstance(value, dict):
            # The hyperparameters themselves, as rows (step 5.2). The card renders
            # one line per parameter, so a dict handed over whole arrived as the
            # literal text "[object Object]" — a card that says a configuration
            # declares seven settings and then refuses to name any of them.
            # Capped, because a config with eighty fields is a file to open rather
            # than a card to read, and the reason line already gives the count.
            for name, setting in list(value.items())[:12]:
                parameters[name] = _readable(setting)
            continue
        readable = _readable(value)
        # An empty value is not a row. `bases: ''` drew a label with nothing beside
        # it on every configuration card, which reads as a fact the graph failed to
        # find rather than one it does not have.
        if readable == "" or readable is None:
            continue
        parameters[key] = readable
    # Whether this card's Validate can be pressed, and what it needs if not
    # (step 5.9). Decided here rather than on the canvas because it is a reading of
    # the index, and computed for every node so the answer is on the card BEFORE
    # the button — a refusal that arrives after the press has wasted the press.
    parameters["_validatable"] = verdict_for(
        node.kind, node.label, node.params, configurations=configurations
    ).as_dict()
    # Whether this card could OPEN — a property of the node, not of the frame it is
    # not currently wearing. Inside `parameters` like every other underscore key,
    # because the canvas hands the card model `data.parameters` and nothing else:
    # placed beside it, this reached the canvas and then vanished, and the Show
    # contents row it gates would never once have been drawn.
    if node.kind == "model" or node.kind in FRAMED_FUNCTION_KINDS:
        parameters["_expandable"] = True
    # What the containment arrow used to say — the attribute holding this, and how
    # many there are. See `nested_label` in build_experiment_document.
    if held:
        parameters.update(held)
    parameters["_kind"] = node.kind
    parameters["_provenance"] = node.provenance
    parameters["_producer"] = node.producer
    parameters["_label"] = node.label
    reason = node.params.get("_reason")
    if reason:
        parameters["_reason"] = reason
    if node.source:
        parameters["_source_file"] = node.source.file
        parameters["_lineno"] = node.source.start[0]
    if node.id in entries:
        parameters["_entry"] = True
    if node.id in terminals:
        parameters["_terminal"] = True

    return {
        "label": node.label,
        "layerType": {"id": "custom_code", "name": node.kind},
        "parameters": parameters,
    }


def build_experiment_document(
    graph: ResearchGraph,
    *,
    name: str = "",
    expanded: Sequence[str] = (),
    layers_for: Optional[Any] = None,
    steps_for: Optional[Any] = None,
    show_structure: bool = False,
) -> ExperimentDocument:
    """Project one experiment.

    `layers_for(node)` supplies an expanded model frame's children — the Phase 2
    parser, injected rather than imported, so this module stays pure and testable
    without torch. `steps_for(node)` does the same for a function frame, from
    `research.frames`. Both are called **only** for frames in `expanded`, which is
    what keeps a large project cheap.
    """
    open_frames = set(expanded)
    # The configurations this experiment declares, so a module asking for a `config`
    # can be told which one to name (step 5.9). Gathered once: the answer is the
    # same for every card, and re-deriving it per node would walk the graph n times.
    configurations = [
        {"label": node.label, "settings": node.params.get("settings")}
        for node in graph.nodes
        if node.kind == "configuration"
    ]
    # ── composition is containment, not an arrow ────────────────────────────────
    #
    # `TinyLM --builds--> TransformerBlock --builds--> MultiHeadAttention` was drawn
    # as three sibling cards joined by two long dotted lines, and on a real project
    # that reads as scatter: the parts of one model sit as far from it as anything
    # else in the repository, and the only thing saying they belong to it is a line
    # crossing the canvas. A model made of other models is a CONTAINER.
    #
    # Only when there is exactly one builder. A class several models build — a shared
    # `Linear`, a `LayerNorm` used everywhere — belongs to all of them and to none, so
    # it stays at the top level where it can be seen from each. Nesting it under
    # whichever happened to be first would be a claim the code does not make.
    builders: Dict[str, List[str]] = {}
    model_ids = {node.id for node in graph.nodes if node.kind == "model"}
    for edge in graph.edges:
        if edge.from_node_id in model_ids and edge.to_node_id in model_ids:
            builders.setdefault(edge.to_node_id, []).append(edge.from_node_id)
    nested_parent: Dict[str, str] = {
        child: parents[0] for child, parents in builders.items() if len(set(parents)) == 1
    }
    # What the arrow said, kept. `GPT --transformer.h ×config.n_layer--> Block` is
    # not only "GPT builds Block": it says which attribute holds it and how many
    # there are, which is most of what a reader wants from a composition. Dropping
    # the line without carrying that would lose it, so it goes on the nested card.
    nested_label: Dict[str, Dict[str, Any]] = {}
    for edge in graph.edges:
        if nested_parent.get(edge.to_node_id) != edge.from_node_id:
            continue
        held: Dict[str, Any] = {}
        attribute = _edge_label(edge)
        if attribute and attribute != "builds":
            held["_held_as"] = attribute
        if held:
            nested_label[edge.to_node_id] = held
    # A cycle (A builds B builds A) would nest each inside the other and neither
    # would ever be drawn. Broken by dropping the link that closes it.
    for child in list(nested_parent):
        seen_chain = {child}
        walker = nested_parent.get(child)
        while walker is not None:
            if walker in seen_chain:
                nested_parent.pop(child, None)
                break
            seen_chain.add(walker)
            walker = nested_parent.get(walker)

    entries = set(graph.entry_node_ids())
    terminals = set(graph.terminal_node_ids())
    rank = _rank(graph)

    nodes: List[DocumentNode] = []
    edges: List[DocumentEdge] = []
    spanmap: Dict[str, Dict[str, Any]] = {}

    # Files are structure, not pipeline. They appear only behind the toggle, so the
    # default view is the dataflow a researcher came to see rather than a directory
    # listing with arrows.
    visible = [n for n in graph.nodes if show_structure or n.kind != "file"]

    # A nested model is drawn only when every frame above it is open — the same rule
    # the canvas already applies to a module's children, and what keeps a collapsed
    # project cheap: a closed `TinyLM` contributes one shape, not four.
    def _reachable(node_id: str) -> bool:
        parent = nested_parent.get(node_id)
        while parent is not None:
            if parent not in open_frames:
                return False
            parent = nested_parent.get(parent)
        return True

    # Parents before children, always. The canvas assembles render nodes in the
    # order it receives them and a child whose parent has not been created yet is
    # not parented at all — an open `TinyLM` said "9 layers" and drew an empty box,
    # because `TransformerBlock` arrived one position ahead of the frame it belongs
    # to. Graph order put it there; nothing else was wrong.
    def _depth(node_id: str) -> int:
        depth = 0
        parent = nested_parent.get(node_id)
        while parent is not None and depth < 64:
            depth += 1
            parent = nested_parent.get(parent)
        return depth

    drawn = sorted(
        (node for node in visible if _reachable(node.id)),
        key=lambda node: _depth(node.id),
    )
    # Where each nested child sits inside its parent, counted per parent so two
    # models do not stack their children on one another.
    nested_row: Dict[str, int] = {}

    per_rank: Dict[int, int] = {}
    for node in drawn:
        parent_id = nested_parent.get(node.id)
        if parent_id is None:
            column = rank.get(node.id, 0)
            row = per_rank.get(column, 0)
            per_rank[column] = row + 1
        else:
            column = 0
            row = nested_row.get(parent_id, 0)
            nested_row[parent_id] = row + 1

        is_model = node.kind == "model"
        # A function frame opens the same way a model does (step 5.3). The card for
        # `LanguageModelTrainer` says "calls .backward() and an optimizer .step()",
        # which is true and is the whole of what the project graph knows about the
        # one place training actually happens — `inputs -> logits -> loss ->
        # backward -> step` is inside, and invisible while a function is one box.
        is_function = node.kind in FRAMED_FUNCTION_KINDS
        is_open = (is_model or is_function) and node.id in open_frames
        # A FRAME only when it is actually open. `moduleNode` is the per-class
        # canvas's frame: it renders "N layers", "Collapsed block view" and an
        # Expand Block button, and it draws none of the project card — so every
        # model node came out as an anonymous box titled "module", with its class
        # name, the rule that classified it and its file:line all discarded. A
        # collapsed model is a card, and the card already says those things.
        nodes.append(
            DocumentNode(
                id=node.id,
                type="moduleNode" if is_open else "layerNode",
                position=(
                    {"x": CHILD_INSET_X, "y": CHILD_INSET_Y + row * CHILD_PITCH_Y}
                    if parent_id is not None
                    else {"x": column * STRIDE_X, "y": row * STRIDE_Y}
                ),
                parentNode=parent_id,
                data={
                    **_card_data(
                        node, entries, terminals, configurations,
                        held=nested_label.get(node.id),
                    ),
                    **({"expanded": True, "isGroup": True} if is_open else {}),
                },
            )
        )
        if node.source:
            spanmap[node.id] = {
                "file": node.source.file,
                "start": list(node.source.start),
                "end": list(node.source.end),
                "kind": node.kind,
                "digest": node.source.digest or "",
            }

        children: List[Dict[str, Any]] = []
        if is_open and is_model and layers_for is not None:
            children = list(layers_for(node) or [])
        elif is_open and is_function and steps_for is not None:
            children = list(steps_for(node) or [])
        if is_open:
            # `moduleNode` is the per-class canvas's frame and reads its own three
            # fields — not `label`, and not the card payload above. Left unset, a
            # correctly-built frame with six children rendered as a card titled
            # "module" saying "0 layers", with a Collapse button that did nothing
            # because it is gated on `moduleInstanceId`. The document was right and
            # every visible part of it was wrong.
            frame = nodes[-1]
            frame.data["moduleTemplateId"] = node.label
            frame.data["moduleInstanceId"] = node.id
            frame.data["moduleLayerCount"] = len(children) + sum(
                1 for other in visible if nested_parent.get(other.id) == node.id
            )

        if children:
            for index, layer in enumerate(children):
                # Namespaced under the frame, so two models in one experiment cannot
                # collide on `l1` — and so a frame id stays resolvable to its owner.
                child_id = f"{node.id}/{layer['id']}"
                nodes.append(
                    DocumentNode(
                        id=child_id,
                        type="layerNode",
                        position={
                            "x": CHILD_INSET_X,
                            "y": CHILD_INSET_Y + index * CHILD_PITCH_Y,
                        },
                        data={
                            "label": layer.get("label") or layer["id"],
                            "layerType": {
                                "id": layer.get("op") or "custom_code",
                                "name": layer.get("type") or "layer",
                            },
                            # Plain layer parameters and nothing else. A frame's child
                            # is an ordinary layer card — Embedding, Linear — titled and
                            # coloured from the catalogue by its `op`. Putting `_kind`
                            # in here sends it down the RESEARCH-card branch instead,
                            # which titles a card `_label or kind`: every child came out
                            # "component" in a custom-code box. (The old `_kind` beside
                            # `parameters` reached nothing and did no harm; it is
                            # simply gone.)
                            "parameters": layer.get("params") or {},
                        },
                        parentNode=node.id,
                    )
                )
                # Inside a function frame the EDGES are the point: a step that reads
                # a name an earlier step wrote is downstream of it, and that chain
                # (`logits -> loss -> backward`) is the whole reason to open a
                # trainer. A model frame's layers carry no `reads` and draw none.
                for source_step in layer.get("reads") or []:
                    from_id = f"{node.id}/{source_step}"
                    edges.append(
                        DocumentEdge(
                            id=f"{from_id}->{child_id}",
                            source=from_id,
                            target=child_id,
                            label="",
                            data={
                                # `feeds`, so the layout straightens it and the
                                # canvas draws it lit: inside a function body this
                                # IS the dataflow.
                                "_relationship": "feeds",
                                "_provenance": "static",
                                "_producer": "torch-ecosystem",
                            },
                        )
                    )
                if layer.get("line") and node.source:
                    spanmap[child_id] = {
                        "file": node.source.file,
                        "start": [layer["line"], 0],
                        "end": [layer.get("end_line") or layer["line"], 0],
                        "kind": "component",
                        "digest": "",
                    }

    present = {n.id for n in nodes}
    for edge in graph.edges:
        # Containment replaces the arrow. `TinyLM --builds--> TransformerBlock` is
        # now the fact that one is drawn inside the other, and keeping the line as
        # well would draw an edge from a frame to its own contents.
        if nested_parent.get(edge.to_node_id) == edge.from_node_id:
            continue
        # Only code arrangement hides. `runs_on` stays, or the configuration card is
        # drawn with no edge at all, which reads as a bug rather than an omission.
        if not show_structure and edge.relationship in CODE_STRUCTURE_RELATIONSHIPS:
            continue
        if edge.from_node_id not in present or edge.to_node_id not in present:
            continue
        edges.append(
            DocumentEdge(
                id=edge.id,
                source=edge.from_node_id,
                target=edge.to_node_id,
                label=_edge_label(edge),
                data={
                    # The label is what a reader sees and may be the attribute the
                    # source named; the RELATIONSHIP is the fact, and it has to
                    # survive that. Without it nothing downstream could tell a
                    # `builds` from a `trains` once the label became `transformer.h`.
                    "_relationship": edge.relationship,
                    "_provenance": edge.provenance,
                    "_producer": edge.producer,
                    **{f"_{key}": value for key, value in edge.params.items()},
                },
            )
        )

    return ExperimentDocument(
        graph_id=graph.graph_id,
        experiment_id=graph.experiment_id,
        name=name or graph.experiment_id,
        nodes=nodes,
        edges=edges,
        spanmap=spanmap,
        expanded=sorted(open_frames),
    )
