"""The graph as Mermaid.

Taken from the DOCUMENT rather than the graph, deliberately: the document is what
the canvas draws, so an export read from anywhere else would be a second opinion
about what is on screen.

Most of what is pinned here is Mermaid's syntax being less forgiving than it looks.
Every one of these produces a diagram that renders as *something*, just not the
graph — which is the worst failure an export can have, because it looks like it
worked.
"""


import pytest


from research.contracts import ResearchEdge, ResearchGraph, ResearchNode  # noqa: E402
from research.document import build_experiment_document  # noqa: E402
from research.export import to_mermaid  # noqa: E402


def node(node_id, kind, label=None):
    return ResearchNode(
        id=node_id,
        graph_id="g",
        experiment_id="e",
        kind=kind,
        label=label or node_id,
        definition_id=f"f.py::{node_id}",
        provenance="static",
    )


def edge(a, b, relationship="feeds"):
    return ResearchEdge(
        id=f"{a}->{b}:{relationship}",
        graph_id="g",
        experiment_id="e",
        from_node_id=a,
        to_node_id=b,
        relationship=relationship,
        provenance="static",
    )


@pytest.fixture
def pipeline():
    return ResearchGraph(
        graph_id="g",
        experiment_id="e",
        nodes=[node("data", "dataset"), node("net", "model"), node("tr", "trainer")],
        edges=[edge("data", "net"), edge("net", "tr", "trains"), edge("data", "tr", "builds")],
    ).validate_graph()


def test_dataflow_is_solid_and_everything_else_is_dotted(pipeline):
    """The same distinction the canvas makes with a dashed, dimmed line. `builds` is
    true of nearly every pair of definitions and says nothing about sequence."""
    out = to_mermaid(build_experiment_document(pipeline))
    assert "-->|feeds|" in out
    assert "-.->|builds|" in out


def test_the_label_goes_after_the_arrow_not_inside_it():
    """`A -->|label| B`. Splicing the label between the arrow's own characters gives
    `--|feeds|>`, which Mermaid does not parse — it renders as nothing at all, and
    the first version of this exporter did exactly that."""
    graph = ResearchGraph(
        graph_id="g",
        experiment_id="e",
        nodes=[node("a", "dataset"), node("b", "model")],
        edges=[edge("a", "b")],
    ).validate_graph()
    out = to_mermaid(build_experiment_document(graph))
    assert "--|" not in out
    assert "-->|feeds|" in out


def test_a_node_called_end_does_not_truncate_the_diagram():
    """`end` closes a subgraph. Used as an id it ends the diagram there, and every
    node after it silently disappears."""
    graph = ResearchGraph(
        graph_id="g",
        experiment_id="e",
        nodes=[node("end", "dataset"), node("after", "model")],
        edges=[edge("end", "after")],
    ).validate_graph()
    out = to_mermaid(build_experiment_document(graph))
    lines = [line.strip() for line in out.splitlines()]
    assert not any(line.startswith("end[") or line == "end" for line in lines)
    assert any("after" in line for line in lines)


def test_two_ids_that_normalise_alike_stay_apart():
    """`a.b` and `a_b` both sanitise to `a_b`. Merged, two nodes become one box and
    the edges of both arrive at it."""
    graph = ResearchGraph(
        graph_id="g",
        experiment_id="e",
        nodes=[node("a.b", "dataset"), node("a_b", "model")],
        edges=[edge("a.b", "a_b")],
    ).validate_graph()
    out = to_mermaid(build_experiment_document(graph))
    identifiers = [line.strip().split("[")[0] for line in out.splitlines() if "[" in line and "-" not in line.split("[")[0]]
    assert len(set(identifiers)) == len(identifiers)


def test_an_expanded_frame_becomes_a_subgraph():
    """The drill-down is what makes a model worth exporting; flattened into
    namespaced ids it is a list, not a diagram."""
    graph = ResearchGraph(
        graph_id="g",
        experiment_id="e",
        nodes=[node("net", "model")],
        edges=[],
    ).validate_graph()
    document = build_experiment_document(
        graph,
        expanded=["net"],
        layers_for=lambda _n: [{"id": "fc1", "type": "Linear", "op": "linear"}],
    )
    out = to_mermaid(document)
    assert "subgraph" in out
    assert "fc1" in out
    assert out.count("end") >= 1


def test_a_quote_in_a_label_does_not_end_the_label(pipeline):
    """A label is double-quoted, so a double quote inside it closes the box early
    and the rest of the line becomes syntax."""
    graph = ResearchGraph(
        graph_id="g",
        experiment_id="e",
        nodes=[node("a", "dataset", label='say "hi" [now]')],
        edges=[],
    ).validate_graph()
    out = to_mermaid(build_experiment_document(graph))
    assert '"say' not in out.replace('["say', "")
    assert "'hi'" in out
    assert "(now)" in out


def test_an_edge_to_a_node_the_document_hides_is_dropped(pipeline):
    """The document hides `file` nodes behind the structure toggle. An edge naming
    one would reference an id that has no box, and Mermaid invents an empty node
    for it — a graph with a phantom in it."""
    graph = ResearchGraph(
        graph_id="g",
        experiment_id="e",
        nodes=[node("f", "file", "train.py"), node("net", "model")],
        edges=[edge("f", "net", "contains")],
    ).validate_graph()
    out = to_mermaid(build_experiment_document(graph, show_structure=False))
    assert "train.py" not in out
    assert "-.->" not in out
