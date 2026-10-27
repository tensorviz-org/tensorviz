"""Projecting one experiment into the canvas document.

The rule under test throughout: **only expanded frames project their contents.** That
is what makes a five-hundred-file project cheap to draw, and it is the thing a naive
"render everything" implementation would quietly break.
"""


import pytest


from research.contracts import ResearchEdge, ResearchGraph, ResearchNode, SourceRef  # noqa: E402
from research.document import build_experiment_document  # noqa: E402


def node(node_id, kind, label=None, *, file=None, provenance="static", reason=None):
    return ResearchNode(
        id=node_id,
        graph_id="g",
        experiment_id="e",
        kind=kind,
        label=label or node_id,
        provenance=provenance,
        params={"_reason": reason} if reason else {},
        source=SourceRef(file=file, start=(3, 0), end=(9, 0), digest="abc") if file else None,
    )


def edge(a, b, relationship="feeds", provenance="static"):
    return ResearchEdge(
        id=f"{a}->{b}:{relationship}",
        graph_id="g",
        experiment_id="e",
        from_node_id=a,
        to_node_id=b,
        relationship=relationship,
        provenance=provenance,
    )


@pytest.fixture
def pipeline():
    return ResearchGraph(
        graph_id="g",
        experiment_id="e",
        nodes=[
            node("f", "file", "train.py", file="train.py"),
            node("cfg", "configuration", "load_config", file="train.py"),
            node("data", "dataset", "make_loader", file="data/toy.py"),
            node("net", "model", "Net", file="models/net.py"),
            node("tr", "trainer", "train", file="train.py"),
        ],
        edges=[
            edge("f", "net", "contains"),
            edge("data", "net", "feeds"),
            edge("net", "tr", "trains"),
            edge("cfg", "tr", "runs_on", provenance="inferred"),
        ],
    ).validate_graph()


def ids(document):
    return {n.id for n in document.nodes}


# ── lazy expansion ──────────────────────────────────────────────────────────────


def layers_for(_node):
    return [
        {"id": "fc1", "type": "Linear", "op": "linear", "params": {"out_features": 8}, "line": 6},
        {"id": "fc2", "type": "Linear", "op": "linear", "params": {"out_features": 10}, "line": 7},
    ]


def test_a_collapsed_frame_contributes_exactly_one_shape(pipeline):
    document = build_experiment_document(pipeline, layers_for=layers_for)
    model_shapes = [n for n in document.nodes if n.id == "net" or n.id.startswith("net/")]
    assert len(model_shapes) == 1
    # No `expanded` flag at all when closed, because a closed model is not a frame
    # that happens to be shut — it is a card. `expanded: False` on a card is a
    # statement about a frame that is not there.
    assert "expanded" not in model_shapes[0].data
    assert model_shapes[0].data["parameters"]["_expandable"] is True


def test_an_expanded_frame_brings_its_children(pipeline):
    document = build_experiment_document(pipeline, expanded=["net"], layers_for=layers_for)
    assert {"net/fc1", "net/fc2"} <= ids(document)
    assert all(
        n.parentNode == "net" for n in document.nodes if n.id.startswith("net/")
    )


def test_children_are_namespaced_so_two_models_cannot_collide():
    """Both classes declare `fc1`; unprefixed they would be one node."""
    graph = ResearchGraph(
        graph_id="g",
        experiment_id="e",
        nodes=[node("a", "model", "A", file="a.py"), node("b", "model", "B", file="b.py")],
    ).validate_graph()

    document = build_experiment_document(graph, expanded=["a", "b"], layers_for=layers_for)
    assert {"a/fc1", "b/fc1"} <= ids(document)


def test_layers_are_never_requested_for_a_collapsed_frame(pipeline):
    """The scale mechanism: parsing a class the user cannot see is wasted work."""
    calls = []

    def spy(model):
        calls.append(model.id)
        return layers_for(model)

    build_experiment_document(pipeline, expanded=[], layers_for=spy)
    assert calls == []

    build_experiment_document(pipeline, expanded=["net"], layers_for=spy)
    assert calls == ["net"]


def test_expansion_is_echoed_back(pipeline):
    document = build_experiment_document(pipeline, expanded=["net"], layers_for=layers_for)
    assert document.expanded == ["net"]


# ── what the canvas draws ───────────────────────────────────────────────────────


def test_a_model_is_a_frame_only_while_it_is_open(pipeline):
    """`moduleNode` is the per-class canvas's frame: it renders "N layers",
    "Collapsed block view" and an Expand Block button, and none of the project
    card. A model drawn as one while closed came out as an anonymous box titled
    "module", with its class name, the rule that classified it and its file:line
    all discarded — which is what nanoGPT looked like on screen.
    """
    document = build_experiment_document(pipeline)
    types = {n.id: n.type for n in document.nodes}
    assert types["net"] == "layerNode", "a closed model is a card"
    assert types["data"] == "layerNode"
    assert types["tr"] == "layerNode"

    opened = build_experiment_document(pipeline, expanded=["net"], layers_for=layers_for)
    assert {n.id: n.type for n in opened.nodes}["net"] == "moduleNode"


def test_no_new_shape_type_is_needed_for_research_kinds(pipeline):
    """Research cards ride the existing `custom_code` placeholder."""
    document = build_experiment_document(pipeline)
    card = next(n for n in document.nodes if n.id == "data")
    assert card.data["layerType"]["id"] == "custom_code"
    assert card.data["parameters"]["_kind"] == "dataset"


def test_the_card_model_can_reach_every_underscore_key(pipeline):
    """The canvas hands `data.parameters` to the card model and nothing else.

    A key placed beside `parameters` reaches the canvas and then vanishes, which is
    how every research card came to render as an unlabelled "Custom Code" box. This
    asserts the channel, not just the values.
    """
    document = build_experiment_document(pipeline)
    for card in document.nodes:
        beside = {key for key in card.data if key.startswith("_")}
        assert not beside, f"{card.id} puts {beside} where the card model cannot see it"
    # `_reason` is present only where a rule fired; the rest are always carried.
    card = next(n for n in document.nodes if n.id == "data")
    assert {"_kind", "_provenance", "_producer", "_label"} <= set(card.data["parameters"])


def test_structure_is_hidden_by_default_and_available_on_request(pipeline):
    """A researcher came for the dataflow, not a directory listing with arrows."""
    plain = build_experiment_document(pipeline)
    assert "f" not in ids(plain)
    assert all(e.label != "contains" for e in plain.edges)

    structured = build_experiment_document(pipeline, show_structure=True)
    assert "f" in ids(structured)
    assert any(e.label == "contains" for e in structured.edges)


def test_provenance_and_reason_reach_the_card(pipeline):
    graph = ResearchGraph(
        graph_id="g",
        experiment_id="e",
        nodes=[node("d", "dataset", "raw", provenance="declared", reason="declared by tensorviz.node()")],
    ).validate_graph()
    document = build_experiment_document(graph)
    parameters = document.nodes[0].data["parameters"]
    assert parameters["_provenance"] == "declared"
    assert parameters["_reason"] == "declared by tensorviz.node()"


def test_an_inferred_edge_says_so(pipeline):
    document = build_experiment_document(pipeline)
    inferred = next(e for e in document.edges if e.source == "cfg")
    assert inferred.data["_provenance"] == "inferred"


def test_dataflow_ranks_left_to_right_and_structure_does_not_distort_it(pipeline):
    document = build_experiment_document(pipeline)
    x = {n.id: n.position["x"] for n in document.nodes}
    assert x["data"] < x["net"] < x["tr"], "the pipeline must read left to right"


def test_entry_and_terminal_are_marked(pipeline):
    document = build_experiment_document(pipeline)
    by_id = {n.id: n for n in document.nodes}
    assert by_id["data"].data["parameters"].get("_entry") is True
    assert by_id["tr"].data["parameters"].get("_terminal") is True


# ── navigation ──────────────────────────────────────────────────────────────────


def test_every_card_with_a_source_is_clickable(pipeline):
    document = build_experiment_document(pipeline)
    for card in document.nodes:
        assert card.id in document.spanmap, f"{card.id} has no span to click through to"
        span = document.spanmap[card.id]
        assert span["file"] and span["start"] and span["end"]


def test_an_expanded_child_resolves_to_its_owning_file(pipeline):
    document = build_experiment_document(pipeline, expanded=["net"], layers_for=layers_for)
    assert document.spanmap["net/fc1"]["file"] == "models/net.py"


def test_an_edge_whose_endpoint_is_hidden_is_dropped_not_dangling(pipeline):
    """A `contains` edge from a hidden file node must not survive into the document."""
    document = build_experiment_document(pipeline)
    present = ids(document)
    for drawn in document.edges:
        assert drawn.source in present and drawn.target in present


def test_an_open_frame_fills_the_fields_the_frame_itself_reads(pipeline):
    """The document was right and every visible part of it was wrong.

    `moduleNode` is the per-class canvas's frame component. It titles itself from
    `data.moduleTemplateId`, counts from `data.moduleLayerCount`, and gates its
    Collapse button on `data.moduleInstanceId` — none of which is `data.label` or
    the card payload beside it. So a correctly-built frame carrying six children
    rendered as a box titled "module", claiming "0 layers", with a Collapse button
    that returned on its first line.
    """
    document = build_experiment_document(pipeline, expanded=["net"], layers_for=layers_for)
    frame = next(n for n in document.nodes if n.id == "net")
    assert frame.type == "moduleNode"
    assert frame.data["moduleTemplateId"] == "Net"
    assert frame.data["moduleInstanceId"] == "net"
    assert frame.data["moduleLayerCount"] == 2
    assert sum(1 for n in document.nodes if n.parentNode == "net") == 2


def test_a_collapsed_model_claims_no_frame_fields(pipeline):
    """A collapsed model is a card, and a card must not carry a frame's furniture —
    the count would go stale the moment the frame is closed."""
    document = build_experiment_document(pipeline, expanded=[], layers_for=layers_for)
    card = next(n for n in document.nodes if n.id == "net")
    assert card.type == "layerNode"
    assert "moduleLayerCount" not in card.data
    assert "moduleInstanceId" not in card.data


def test_a_trainer_opens_into_its_loop(pipeline):
    """Step 5.3. A function frame draws its steps and, unlike a model frame, the
    EDGES between them: `logits -> loss -> backward` is the whole reason to open a
    trainer."""
    document = build_experiment_document(
        pipeline,
        expanded=["tr"],
        steps_for=lambda _n: [
            {"id": "s1", "label": "self.model → logits", "reads": []},
            {"id": "s2", "label": "F.cross_entropy → loss", "reads": ["s1"]},
            {"id": "s3", "label": "loss.backward", "reads": ["s2"]},
        ],
    )
    assert {"tr/s1", "tr/s2", "tr/s3"} <= ids(document)
    inner = [e for e in document.edges if e.source.startswith("tr/")]
    assert {(e.source, e.target) for e in inner} == {("tr/s1", "tr/s2"), ("tr/s2", "tr/s3")}
    # Drawn as dataflow, so the layout straightens it and the canvas lights it.
    assert all(e.data["_relationship"] == "feeds" for e in inner)


def test_a_model_frame_draws_the_edges_its_parser_read_and_none_it_did_not(pipeline):
    """The parser's edges are the forward pass — `token -> add -> norm -> head` —
    and they were read and then dropped, so an open frame laid nine children in a
    row with nothing joining them; dagre, given no edges, put every one at rank 0
    side by side. They travel as `reads`, the form a function frame already uses,
    so one emitter serves both kinds of frame. A child that reads nothing gets
    nothing: inventing an edge would assert an order the parser did not read."""
    wired = lambda _n: [
        {"id": "fc1", "type": "Linear", "op": "linear", "reads": []},
        {"id": "act", "type": "ReLU", "op": "relu", "reads": ["fc1"]},
        {"id": "aside", "type": "Dropout", "op": "dropout", "reads": []},
    ]
    document = build_experiment_document(pipeline, expanded=["net"], layers_for=wired)
    inner = {(e.source, e.target) for e in document.edges if e.source.startswith("net/")}
    assert inner == {("net/fc1", "net/act")}
    assert all(e.data["_relationship"] == "feeds" for e in document.edges if e.source.startswith("net/"))


def test_a_frame_child_stays_an_ordinary_layer_card(pipeline):
    """No `_kind` in a child's parameters. That key sends a card down the
    research-card branch, which titles it `_label or kind` — and every Embedding
    and Linear inside a frame came out "component" in a custom-code box."""
    document = build_experiment_document(pipeline, expanded=["net"], layers_for=layers_for)
    for child in (n for n in document.nodes if n.parentNode == "net"):
        assert "_kind" not in child.data["parameters"]
        assert child.data["label"]


def test_only_kinds_with_a_body_worth_drawing_are_expandable(pipeline):
    """A configuration is a thing, not a sequence. Opening one would draw a frame
    around a single box."""
    document = build_experiment_document(pipeline)
    expandable = {
        n.data["label"] for n in document.nodes if n.data["parameters"].get("_expandable")
    }
    assert "train" in expandable and "Net" in expandable
    assert "load_config" not in expandable and "make_loader" not in expandable


# ── composition is containment, not an arrow ────────────────────────────────────


def composed():
    """`TinyLM` builds `Block`, which builds `Attention` and `FeedForward`."""
    return ResearchGraph(
        graph_id="g",
        experiment_id="e",
        nodes=[
            node("lm", "model", "TinyLM"),
            node("blk", "model", "Block"),
            node("att", "model", "Attention"),
            node("ff", "model", "FeedForward"),
        ],
        edges=[
            edge("lm", "blk", "builds"),
            edge("blk", "att", "builds"),
            edge("blk", "ff", "builds"),
        ],
    ).validate_graph()


def test_a_model_made_of_models_draws_as_one_card_until_opened():
    """Three sibling cards joined by two long dotted lines is scatter: the parts of
    one model sat as far from it as anything else in the repository, and the only
    thing saying they belonged to it was a line crossing the canvas."""
    document = build_experiment_document(composed())
    assert ids(document) == {"lm"}


def test_opening_a_model_reveals_the_models_it_is_made_of():
    document = build_experiment_document(composed(), expanded=["lm"])
    assert ids(document) == {"lm", "blk"}
    assert next(n for n in document.nodes if n.id == "blk").parentNode == "lm"
    # Still shut itself, so its own parts stay hidden one level down.
    assert "att" not in ids(document)


def test_the_nesting_goes_as_deep_as_the_user_opens_it():
    document = build_experiment_document(composed(), expanded=["lm", "blk"])
    assert ids(document) == {"lm", "blk", "att", "ff"}
    assert {n.parentNode for n in document.nodes if n.id in {"att", "ff"}} == {"blk"}


def test_containment_replaces_the_arrow():
    """Keeping the line as well would draw an edge from a frame to its own
    contents."""
    document = build_experiment_document(composed(), expanded=["lm", "blk"])
    assert not [e for e in document.edges if (e.source, e.target) == ("lm", "blk")]


def test_a_class_several_models_build_stays_at_the_top():
    """A shared `Linear` belongs to all of them and to none. Nesting it under
    whichever happened to be first would be a claim the code does not make."""
    graph = ResearchGraph(
        graph_id="g",
        experiment_id="e",
        nodes=[node("a", "model", "A"), node("b", "model", "B"), node("shared", "model", "Linear")],
        edges=[edge("a", "shared", "builds"), edge("b", "shared", "builds")],
    ).validate_graph()
    document = build_experiment_document(graph)
    assert next(n for n in document.nodes if n.id == "shared").parentNode is None


def test_a_cycle_does_not_hide_both_of_its_models():
    """A builds B builds A would nest each inside the other, and neither would ever
    be drawn."""
    graph = ResearchGraph(
        graph_id="g",
        experiment_id="e",
        nodes=[node("a", "model", "A"), node("b", "model", "B")],
        edges=[edge("a", "b", "builds"), edge("b", "a", "builds")],
    ).validate_graph()
    assert ids(build_experiment_document(graph))


def test_a_frame_counts_the_models_inside_it():
    document = build_experiment_document(composed(), expanded=["lm"])
    assert next(n for n in document.nodes if n.id == "lm").data["moduleLayerCount"] == 1


def test_a_parent_is_always_emitted_before_its_children():
    """The canvas assembles render nodes in the order it receives them, so a child
    whose parent has not been created yet is not parented at all.

    An open `TinyLM` said "9 layers" and drew an empty box: `TransformerBlock` was
    emitted one position AHEAD of the frame it belongs to, because that is the order
    the graph happened to hold them in. Nothing else was wrong — the count was
    right, the children were there, and none of them rendered.
    """
    document = build_experiment_document(composed(), expanded=["lm", "blk"])
    position = {node.id: index for index, node in enumerate(document.nodes)}
    for node in document.nodes:
        if node.parentNode:
            assert position[node.parentNode] < position[node.id], (
                f"{node.id} is drawn before its parent {node.parentNode}"
            )


def test_a_frame_and_its_parsed_layers_keep_that_order_too():
    """The same rule for the other kind of child: a model's layers come from the
    parser rather than the graph, and are appended after the frame."""
    document = build_experiment_document(composed(), expanded=["lm", "blk"], layers_for=layers_for)
    position = {node.id: index for index, node in enumerate(document.nodes)}
    for node in document.nodes:
        if node.parentNode:
            assert position[node.parentNode] < position[node.id]
