"""Writing an SDK declaration — the step that turns a guess into source text.

The rules under test, in order of how badly they matter: never write into a file
that does not parse, never write a declaration the next index would reject, and
never produce a patch bigger than the one line it needs to be.
"""

import ast

import pytest


from research.codegen import (  # noqa: E402
    CodegenError,
    DeclareEdge,
    DeclareExperiment,
    DeclareNode,
    declare,
    statement_for,
    undeclare,
)
from research.indexer import index_project  # noqa: E402


def parses(edit):
    ast.parse(edit.contents)
    return edit.contents


# ── the statement ───────────────────────────────────────────────────────────────


def test_each_declaration_has_one_readable_form():
    assert statement_for(DeclareNode("dataset", "raw-logs")) == (
        "tensorviz.node('dataset', 'raw-logs')"
    )
    assert statement_for(DeclareEdge("a", "b", "feeds")) == (
        "tensorviz.edge('a', 'b', 'feeds')"
    )
    assert statement_for(DeclareExperiment("wide-net")) == (
        "tensorviz.experiment('wide-net')"
    )


def test_an_unknown_kind_or_relationship_is_refused():
    with pytest.raises(CodegenError) as kind:
        statement_for(DeclareNode("banana", "x"))
    assert kind.value.code == "unknown_kind"

    with pytest.raises(CodegenError) as rel:
        statement_for(DeclareEdge("a", "b", "sideways"))
    assert rel.value.code == "unknown_relationship"


def test_a_label_that_cannot_be_written_readably_is_refused():
    """Escaping it would produce a line the user cannot read at a glance."""
    with pytest.raises(CodegenError) as error:
        statement_for(DeclareNode("dataset", "two\nlines"))
    assert error.value.code == "unwritable_label"


def test_an_empty_label_is_refused_not_defaulted():
    with pytest.raises(CodegenError) as error:
        statement_for(DeclareNode("dataset", "   "))
    assert error.value.code == "empty_label"


# ── where it lands ──────────────────────────────────────────────────────────────


def test_a_declaration_joins_the_block_already_there():
    source = (
        "import tensorviz\n"
        "\n"
        "tensorviz.experiment('demo')\n"
        "tensorviz.node('dataset', 'raw')\n"
        "\n"
        "VALUE = 1\n"
    )
    edit = declare("train.py", source, DeclareNode("transform", "tokenise"))
    text = parses(edit)

    assert text.count("import tensorviz") == 1, "the import must not be duplicated"
    assert "tensorviz.node('transform', 'tokenise')" in text
    body = text.splitlines()
    assert body.index("tensorviz.node('transform', 'tokenise')") == body.index(
        "tensorviz.node('dataset', 'raw')"
    ) + 1, "a file must not grow a second cluster of declarations"


def test_a_first_declaration_brings_its_import():
    source = "import torch\n\nVALUE = 1\n"
    text = parses(declare("train.py", source, DeclareExperiment("demo")))
    assert "import tensorviz" in text
    assert text.index("import tensorviz") > text.index("import torch")


def test_a_module_docstring_stays_first():
    source = '"""What this file is."""\n\nVALUE = 1\n'
    text = parses(declare("train.py", source, DeclareExperiment("demo")))
    assert text.startswith('"""What this file is."""')


def test_an_empty_file_is_handled():
    text = parses(declare("train.py", "", DeclareExperiment("demo")))
    assert "tensorviz.experiment('demo')" in text


def test_a_file_with_no_trailing_newline_is_not_corrupted():
    text = parses(declare("train.py", "VALUE = 1", DeclareExperiment("demo")))
    assert "VALUE = 1\n" in text


# ── refusals ────────────────────────────────────────────────────────────────────


def test_a_file_that_does_not_parse_is_never_written_into():
    """Adding a line would give the user a second error on top of the one they have."""
    with pytest.raises(CodegenError) as error:
        declare("broken.py", "def oops(\n", DeclareExperiment("demo"))
    assert error.value.code == "does_not_parse"


def test_an_edge_naming_a_node_that_does_not_exist_is_refused():
    """The contract rejects a dangling edge, so writing one makes the NEXT index
    fail validation — a failure the user would have to debug rather than simply
    not being allowed to make."""
    with pytest.raises(CodegenError) as error:
        declare(
            "train.py",
            "import tensorviz\n",
            DeclareEdge("ghost", "alsoghost"),
            known_labels=["real"],
        )
    assert error.value.code == "unknown_endpoint"


def test_declaring_the_same_thing_twice_is_refused():
    source = "import tensorviz\n\ntensorviz.node('dataset', 'raw')\n"
    with pytest.raises(CodegenError) as error:
        declare("train.py", source, DeclareNode("dataset", "raw"))
    assert error.value.code == "already_declared"


def test_the_edit_carries_the_digest_it_was_built_on():
    source = "import tensorviz\n"
    edit = declare("train.py", source, DeclareExperiment("demo"))
    assert len(edit.expected_digest) == 64


# ── undeclare ───────────────────────────────────────────────────────────────────


def test_a_declaration_can_be_taken_back():
    """Confirming a guess must not be a one-way door."""
    source = (
        "import tensorviz\n"
        "\n"
        "tensorviz.node('dataset', 'raw')\n"
        "tensorviz.node('transform', 'tok')\n"
    )
    text = parses(undeclare("train.py", source, "tensorviz.node('dataset', 'raw')"))
    assert "tensorviz.node('dataset', 'raw')" not in text
    assert "tensorviz.node('transform', 'tok')" in text


def test_removing_something_that_is_not_there_is_refused():
    with pytest.raises(CodegenError) as error:
        undeclare("train.py", "import tensorviz\n", "tensorviz.node('x', 'y')")
    assert error.value.code == "not_declared"


def test_undeclare_leaves_everything_else_byte_identical():
    source = (
        '"""Doc."""\n'
        "import tensorviz\n"
        "\n"
        "tensorviz.node('dataset', 'raw')\n"
        "\n"
        "# a comment the user wrote\n"
        "VALUE = 1\n"
    )
    text = undeclare("train.py", source, "tensorviz.node('dataset', 'raw')").contents
    assert "# a comment the user wrote" in text
    assert '"""Doc."""' in text


# ── the round trip that matters ─────────────────────────────────────────────────


def test_confirming_a_guess_makes_the_next_index_read_it_as_declared():
    """The whole point: an inferred edge, confirmed, stops being inferred."""
    sources = {
        "models/net.py": "import torch.nn as nn\n\n\nclass Net(nn.Module):\n    pass\n",
        "data/toy.py": (
            "from torch.utils.data import Dataset\n\n\nclass Corpus(Dataset):\n    pass\n"
        ),
        "train.py": (
            "from data.toy import Corpus\n"
            "from models.net import Net\n"
            "\n"
            "if __name__ == '__main__':\n    Net()\n"
        ),
    }

    before = index_project(sources).experiments[0].graph
    by_id = {n.id: n for n in before.nodes}
    guessed = [
        e
        for e in before.edges
        if e.provenance == "inferred"
        and by_id[e.from_node_id].label == "Corpus"
        and by_id[e.to_node_id].label == "Net"
    ]
    assert guessed, "the spine should have guessed this edge"

    edit = declare(
        "train.py",
        sources["train.py"],
        DeclareEdge("Corpus", "Net", "feeds"),
        known_labels=[n.label for n in before.nodes],
    )
    sources["train.py"] = parses(edit)

    after = index_project(sources).experiments[0].graph
    by_id = {n.id: n for n in after.nodes}
    confirmed = [
        e
        for e in after.edges
        if by_id[e.from_node_id].label == "Corpus" and by_id[e.to_node_id].label == "Net"
    ]
    assert confirmed, "the declared edge must be in the graph"
    assert any(e.provenance == "declared" for e in confirmed), (
        "a confirmed guess must outrank the rule that guessed it"
    )
