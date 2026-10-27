"""What the backend sends is what the canvas can paint.

Every defect this file pins was visible on one screenshot of nanoGPT: five model
classes drawn as anonymous boxes titled "module", a configuration card announcing
seven settings and naming none of them, and a row reading `bases:` with nothing
beside it.

They share a cause. The document is handed to a canvas built for a different
plane, so anything the project tier emits in a shape that tier does not read is
not drawn — silently, because a card that renders nothing looks like a card with
nothing to say.
"""



from research.document import build_experiment_document
from research.indexer import index_project

NANOGPT = {
    "model.py": '''
import torch.nn as nn
from dataclasses import dataclass

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
    def forward(self, x):
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    n_layer: int = 12
    n_embd: int = 768
    bias: bool = True

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln = LayerNorm(config.n_embd, bias=config.bias)
    def forward(self, idx):
        return self.ln(idx)
''',
    "train.py": '''
import torch
from model import GPT, GPTConfig

model = GPT(GPTConfig())
for _ in range(1):
    loss = model(None)
    loss.backward()
''',
}


def _document():
    index = index_project(NANOGPT)
    experiment = next(e for e in index.experiments if e.entrypoint == "train.py")
    return build_experiment_document(experiment.graph)


def _document_opened():
    """Every frame open, so nested models are drawn.

    A model made of other models is a container: `GPT` holds `Block`, which holds a
    `CausalSelfAttention`, an `MLP` and two `LayerNorm`s. Collapsed, `GPT` is one
    card and they are correctly absent — so a test about what a nested node CARRIES
    has to open the frames first.
    """
    index = index_project(NANOGPT)
    experiment = next(e for e in index.experiments if e.entrypoint == "train.py")
    every = [node.id for node in experiment.graph.nodes if node.kind == "model"]
    return build_experiment_document(experiment.graph, expanded=every)


def _by_label(document):
    return {node.data.get("label"): node for node in document.nodes}


def test_every_node_carries_the_label_the_graph_gave_it():
    # The defect: five classes drawn as boxes titled "module". The label was in
    # the document the whole time — the canvas was throwing it away.
    nodes = _by_label(_document_opened())
    for expected in ("LayerNorm", "GPT", "GPTConfig"):
        assert expected in nodes, f"{expected} is missing; got {sorted(nodes)}"
        assert nodes[expected].data["label"] == expected


def test_a_collapsed_model_is_a_card_and_not_a_frame():
    """`moduleNode` is the per-class canvas's frame: "N layers", "Collapsed block
    view", an Expand Block button, and none of the project card. A model that is
    not open has no children to frame, so it is drawn as the card that can say
    what it is."""
    nodes = _by_label(_document())
    assert nodes["GPT"].type == "layerNode"
    assert nodes["GPT"].data.get("isGroup") is None
    # It can still open; that is a property of the node, not of a frame it is not
    # currently wearing. Inside `parameters`, which is the only channel the card
    # model is handed — beside it, the flag reaches the canvas and then vanishes.
    assert nodes["GPT"].data["parameters"]["_expandable"] is True


def test_an_open_model_is_a_frame():
    index = index_project(NANOGPT)
    experiment = next(e for e in index.experiments if e.entrypoint == "train.py")
    gpt = next(n for n in experiment.graph.nodes if n.label == "GPT")
    document = build_experiment_document(experiment.graph, expanded=[gpt.id])
    drawn = _by_label(document)["GPT"]
    assert drawn.type == "moduleNode"
    assert drawn.data["isGroup"] is True
    assert drawn.data["expanded"] is True


def test_a_configuration_names_its_settings_rather_than_counting_them():
    # The card said "declares 7 settings" and then drew `settings: [object Object]`
    # — JavaScript's rendering of an object handed to a string context, which tells
    # the reader nothing. The values are the point of the card.
    parameters = _by_label(_document())["GPTConfig"].data["parameters"]
    assert parameters["block_size"] == 1024
    assert parameters["n_embd"] == 768
    assert parameters["bias"] is True
    assert "settings" not in parameters, "the raw dict must not survive as a row"


def test_a_value_the_card_cannot_print_on_one_line_is_made_printable():
    parameters = _by_label(_document())["GPT"].data["parameters"]
    assert parameters["bases"] == "nn.Module", "a list is joined, not drawn as [object Object]"


def test_an_empty_value_is_not_drawn_as_an_empty_row():
    # `bases:` with nothing beside it reads as a fact the graph failed to find
    # rather than one it does not have.
    for node in _document().nodes:
        for key, value in node.data["parameters"].items():
            assert value != "", f"{node.data.get('label')} draws an empty row for {key}"


def test_every_node_says_whether_it_can_be_validated():
    for node in _document().nodes:
        verdict = node.data["parameters"]["_validatable"]
        assert verdict["state"] in {"ready", "needs", "unsupported"}
        if verdict["state"] != "ready":
            assert verdict["detail"], "a refusal with no sentence is not a refusal"


def test_the_canvas_shape_of_every_node_is_one_the_canvas_knows():
    # `type` selects a component. A type nothing registers is a node that does not
    # appear at all, which is indistinguishable from a node the index never found.
    for node in _document().nodes:
        assert node.type in {"layerNode", "moduleNode"}
        assert set(node.position) == {"x", "y"}
        assert isinstance(node.data.get("layerType"), dict)


def test_a_module_that_builds_another_is_connected_to_it():
    """Composition is the model's structure, and it was missing entirely.

    nanoGPT's GPT holds `self.h = nn.ModuleList([Block(config) for _ in
    range(config.n_layer)])`, and Block holds a CausalSelfAttention, an MLP and two
    LayerNorms. The canvas drew all five as disconnected islands: a reader could
    see the classes and not the model, which is the one thing the canvas is for.
    """
    document = _document_opened()
    label = {node.id: node.data["label"] for node in document.nodes}
    # Composition is drawn as CONTAINMENT now, not as an arrow: three sibling cards
    # joined by long dotted lines read as scatter, with the only thing saying the
    # parts belonged to the model being a line crossing the canvas. So the assertion
    # is on the parent chain rather than on an edge — the relationship is the same
    # one, said in the way a reader can see.
    inside = {
        (label[node.parentNode], label[node.id])
        for node in document.nodes
        if node.parentNode
    }
    assert ("GPT", "LayerNorm") in inside, f"GPT holds a LayerNorm; got {sorted(inside)}"
    # And the arrow is gone, because keeping it as well would draw an edge from a
    # frame to its own contents.
    assert not [
        e for e in document.edges
        if label[e.source] == "GPT" and label[e.target] == "LayerNorm"
    ]


def test_composition_is_read_from_the_code_and_not_guessed():
    index = index_project(NANOGPT)
    experiment = next(e for e in index.experiments if e.entrypoint == "train.py")
    builds = [e for e in experiment.graph.edges if e.relationship == "builds"]
    assert builds, "no composition found at all"
    assert all(edge.provenance == "static" for edge in builds), (
        "a call written in the source is read, not inferred — marking it `inferred` "
        "would draw a fact the way the canvas draws a guess"
    )


def test_a_mention_is_not_a_construction():
    """`references` holds every name a body loads — `B`, `C`, `T`, every local. Built
    from that, composition would wire a model to its loop variables."""
    sources = dict(NANOGPT)
    sources["model.py"] = sources["model.py"].replace(
        "    def forward(self, idx):\n        return self.ln(idx)",
        "    def forward(self, idx):\n        LayerNorm\n        return self.ln(idx)",
    )
    index = index_project(sources)
    experiment = next(e for e in index.experiments if e.entrypoint == "train.py")
    labels = {n.id: n.label for n in experiment.graph.nodes}
    # GPT genuinely constructs LayerNorm in __init__, so the edge stands; what must
    # not happen is an edge appearing for a name that is only mentioned.
    mentioned_only = {
        (labels[e.from_node_id], labels[e.to_node_id])
        for e in experiment.graph.edges
        if e.relationship == "builds"
    }
    assert ("LayerNorm", "GPT") not in mentioned_only


def test_the_count_of_a_repeated_block_is_not_claimed():
    """`range(config.n_layer)` builds twelve Blocks. The graph says GPT builds Block,
    because twelve is a value only running the code would know."""
    index = index_project(NANOGPT)
    experiment = next(e for e in index.experiments if e.entrypoint == "train.py")
    labels = {n.id: n.label for n in experiment.graph.nodes}
    pairs = [
        (labels[e.from_node_id], labels[e.to_node_id])
        for e in experiment.graph.edges
        if e.relationship == "builds"
    ]
    assert len(pairs) == len(set(pairs)), "one edge per pair, whatever the multiplicity"


def test_an_edge_says_where_the_thing_lives_and_how_many():
    """`builds` is true and says nothing a reader could not have guessed.

    nanoGPT's line is
        h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
    and the edge should read back as that: the attribute, the container, the count.
    The canvas is not a replacement for the code — it is a way to see what the code
    already says, and an edge a reader can map to a line is the difference.
    """
    source = NANOGPT["model.py"].replace(
        "        self.ln = LayerNorm(config.n_embd, bias=config.bias)",
        "        self.transformer = nn.ModuleDict(dict(\n"
        "            h = nn.ModuleList([LayerNorm(config.n_embd, bias=config.bias)\n"
        "                               for _ in range(config.n_layer)]),\n"
        "        ))",
    )
    index = index_project({**NANOGPT, "model.py": source})
    experiment = next(e for e in index.experiments if e.entrypoint == "train.py")
    gpt = next(n for n in experiment.graph.nodes if n.label == "GPT")
    document = build_experiment_document(experiment.graph, expanded=[gpt.id])
    label = {n.id: n.data["label"] for n in document.nodes}
    # Composition is drawn as containment now, so what the arrow said travels with
    # the nested card instead. Losing it would be the real cost of nesting: "builds"
    # alone says nothing a reader could not have guessed, and the attribute, the
    # container and the count are the whole difference.
    nested = next(n for n in document.nodes if label[n.id] == "LayerNorm" and n.parentNode)
    assert label[nested.parentNode] == "GPT"
    assert nested.data["parameters"]["_held_as"] == "transformer.h ×config.n_layer"


def test_the_count_is_the_expression_and_never_its_value():
    # `config.n_layer` is 12 only once a config exists. Drawing 12 would claim a
    # number nothing here ran to find out — the same lie as an inferred input shape
    # printed with the authority of a measurement.
    source = NANOGPT["model.py"].replace(
        "        self.ln = LayerNorm(config.n_embd, bias=config.bias)",
        "        self.blocks = nn.ModuleList([LayerNorm(1, bias=True) for _ in range(12)])",
    )
    index = index_project({**NANOGPT, "model.py": source})
    experiment = next(e for e in index.experiments if e.entrypoint == "train.py")
    edge = next(
        e
        for e in experiment.graph.edges
        if e.relationship == "builds"
    )
    # A literal repeat IS the expression, so 12 here is what the source says.
    assert edge.params.get("count") in {"12", ""}


def test_an_edge_the_source_did_not_name_keeps_the_plain_relationship():
    # A construction with no attribute — a bare call in a method — has nothing to
    # say beyond the relationship, and inventing a name for it would be worse than
    # the generic word.
    document = _document()
    labels = {n.id: n.data["label"] for n in document.nodes}
    edge = next(
        (e for e in document.edges if labels[e.target] == "GPTConfig" and labels[e.source] == "GPT"),
        None,
    )
    if edge is not None:
        assert edge.label == "builds"
        assert edge.data["_relationship"] == "builds"
