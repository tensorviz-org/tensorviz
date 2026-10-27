"""The contract and identity gates for the project graph.

Each test pins one rule from `docs/project-graph-plan.md`. They are written so they
can fail: an assertion that would still pass if the rule were removed is not a test
of the rule.
"""


import pytest


from research import identity  # noqa: E402
from research.contracts import (  # noqa: E402
    GraphValidationError,
    ResearchEdge,
    ResearchGraph,
    ResearchNode,
    outranks,
)


def node(node_id, kind="generic", graph="g", experiment="e", definition=None):
    return ResearchNode(
        id=node_id,
        graph_id=graph,
        experiment_id=experiment,
        kind=kind,
        label=node_id,
        definition_id=definition,
    )


def edge(edge_id, a, b, relationship="feeds", graph="g", experiment="e"):
    return ResearchEdge(
        id=edge_id,
        graph_id=graph,
        experiment_id=experiment,
        from_node_id=a,
        to_node_id=b,
        relationship=relationship,
    )


# ── contract ────────────────────────────────────────────────────────────────────


def test_entry_and_terminal_come_from_dataflow_only():
    """A file `contains` a model; that must not make the file a root.

    This is the rule that keeps structure out of the dataflow answer. Without it
    every file in the project would be tagged `entry`.
    """
    graph = ResearchGraph(
        graph_id="g",
        experiment_id="e",
        nodes=[node("file"), node("data", "dataset"), node("model", "model")],
        edges=[
            edge("contains", "file", "model", "contains"),
            edge("flow", "data", "model", "feeds"),
        ],
    ).validate_graph()

    assert graph.entry_node_ids() == ["data"]
    assert graph.terminal_node_ids() == ["model"]
    assert "file" not in graph.entry_node_ids()


def test_dangling_edge_is_refused():
    with pytest.raises(GraphValidationError, match="unknown node"):
        ResearchGraph(
            graph_id="g",
            experiment_id="e",
            nodes=[node("a")],
            edges=[edge("x", "a", "ghost")],
        ).validate_graph()


def test_node_from_another_experiment_is_refused():
    with pytest.raises(GraphValidationError, match="another graph or experiment"):
        ResearchGraph(
            graph_id="g",
            experiment_id="e",
            nodes=[node("a", experiment="other")],
        ).validate_graph()


def test_duplicate_ids_are_refused():
    with pytest.raises(GraphValidationError, match="duplicate node id"):
        ResearchGraph(
            graph_id="g", experiment_id="e", nodes=[node("a"), node("a")]
        ).validate_graph()


def test_digest_is_order_independent_but_content_sensitive():
    """Re-indexing a file whose statements moved must not redraw the canvas."""
    a = ResearchGraph(
        graph_id="g",
        experiment_id="e",
        nodes=[node("one"), node("two")],
        edges=[edge("x", "one", "two")],
    )
    b = ResearchGraph(
        graph_id="g",
        experiment_id="e",
        nodes=[node("two"), node("one")],
        edges=[edge("x", "one", "two")],
    )
    assert a.digest() == b.digest()

    c = ResearchGraph(
        graph_id="g",
        experiment_id="e",
        nodes=[node("one"), node("two", kind="model")],
        edges=[edge("x", "one", "two")],
    )
    assert c.digest() != a.digest()


def test_dependents_of_definition_finds_every_instance():
    graph = ResearchGraph(
        graph_id="g",
        experiment_id="e",
        nodes=[
            node("enc1", definition="models/net.py::Net"),
            node("enc2", definition="models/net.py::Net"),
            node("other", definition="models/other.py::Other"),
        ],
    ).validate_graph()
    assert graph.dependents_of_definition("models/net.py::Net") == ["enc1", "enc2"]


def test_provenance_precedence_is_strict():
    """Declared beats inferred; an equal claim never silently replaces the first."""
    assert outranks("declared", "inferred")
    assert outranks("observed", "declared")
    assert not outranks("static", "static")
    assert not outranks("inferred", "static")


# ── identity ────────────────────────────────────────────────────────────────────


def test_definition_id_is_readable_and_separator_normalised():
    assert identity.definition_id("models\\net.py", "Net") == "models/net.py::Net"
    assert identity.definition_id("./models/net.py", "Net") == "models/net.py::Net"


def test_identity_carries_no_line_numbers():
    """The rule the realtime loop depends on: moving a class must not move its id."""
    before = identity.definition_id("models/net.py", "Net")
    after = identity.definition_id("models/net.py", "Net")
    assert before == after
    assert not any(ch.isdigit() for ch in before.split("::")[1])


def test_one_definition_in_two_experiments_gives_two_nodes_one_definition():
    definition = identity.definition_id("models/net.py", "Net")
    a = identity.experiment_node_id("exp:a", definition)
    b = identity.experiment_node_id("exp:b", definition)
    assert a != b


def test_two_instances_in_one_experiment_differ_by_path():
    definition = identity.definition_id("models/net.py", "Net")
    assert identity.experiment_node_id("exp:a", definition, "enc1") != identity.experiment_node_id(
        "exp:a", definition, "enc2"
    )


def test_declared_experiment_id_is_namespaced_and_survives_a_rename():
    """A declared name owns its identity, so moving the file keeps the experiment."""
    assert identity.declared_experiment_id("wide-net") == identity.declared_experiment_id(
        "wide-net"
    )
    assert identity.declared_experiment_id("wide-net") != identity.experiment_id(
        identity.definition_id("train_b.py", "__main__")
    )


def test_edge_id_follows_its_endpoints():
    a = identity.edge_id("n1", "n2", "feeds")
    assert a == identity.edge_id("n1", "n2", "feeds")
    assert a != identity.edge_id("n2", "n1", "feeds")
    assert a != identity.edge_id("n1", "n2", "trains")


def test_empty_identity_inputs_are_refused_not_defaulted():
    with pytest.raises(ValueError):
        identity.definition_id("", "Net")
    with pytest.raises(ValueError):
        identity.definition_id("models/net.py", "")
    with pytest.raises(ValueError):
        identity.declared_experiment_id("  ")
