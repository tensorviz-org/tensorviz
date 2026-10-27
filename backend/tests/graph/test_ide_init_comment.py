"""``# tensorviz: init=`` — the line that says how to BUILD the model.

``# tensorviz: input=`` says what the model is fed. It has never been enough on its
own: a class whose ``__init__`` needs an argument cannot be constructed at all, so the
worker exits 3 and the user sees nothing actionable (ide-plan.html §15 item 3). That is
the single most common first-trace failure on a real repository, because real models
take a config.

This file pins the companion declaration and, like ``test_declared_input_shape``, the
fact that the two tiers which read it cannot drift apart. The parser tier draws the
graph; the ingest tier runs inside the worker and builds the object. If they disagree
about what counts as a declaration, the graph that is drawn and the graph that is run
disagree — which is the bug the duplication exists to make impossible.
"""

from __future__ import annotations

import textwrap

import pytest


from utils.code_parser import ConfigCall, declared_init  # noqa: E402

FIXTURE = textwrap.dedent(
    '''
    import torch.nn as nn


    # tensorviz: init=Net(hidden=8, depth=2)
    class Net(nn.Module):
        def __init__(self, hidden, depth=1):
            super().__init__()
            self.l1 = nn.Linear(hidden, hidden)

        def forward(self, x):
            return self.l1(x)
    '''
)


def test_a_declaration_becomes_constructor_keyword_arguments():
    found = declared_init(FIXTURE, "Net")
    assert found is not None
    assert found.overrides == {"hidden": 8, "depth": 2}
    assert found.error is None


def test_a_file_that_declares_nothing_declares_nothing():
    # None, not an empty dict: "the user said nothing" and "the user said build it
    # with no arguments" are different, and only the first may fall through to the
    # ordinary no-argument construction.
    assert declared_init(FIXTURE.replace("# tensorviz: init=Net(hidden=8, depth=2)", "")) is None
    assert declared_init("") is None


def test_positional_arguments_are_refused_and_say_why():
    """The refusal is the feature.

    Silently accepting ``Net(8, 2)`` would mean this file guessing a parameter order
    it cannot see. Silently IGNORING it would mean the user edits a line that does
    nothing. So it is read, refused, and the reason is carried to the editor.
    """
    found = declared_init("# tensorviz: init=Net(8, 2)", "Net")
    assert found is not None
    assert found.overrides == {}
    assert found.error is not None
    assert "Name the arguments" in found.error


def test_a_constructor_call_is_carried_rather_than_refused():
    """The commonest real ``__init__`` takes a config object, not a number.

    ``Net(cfg=8)`` is not what such a user means and there is no literal that is, so
    before this the line they were told to write — ``# tensorviz: init=Net(cfg=)`` —
    had nothing they could legally put in it, and anything they did put in it was
    refused as "not a literal". The call is carried as a value: THIS tier still does
    not evaluate it (it promises not to import the user's module), the worker does.
    """
    found = declared_init("# tensorviz: init=Net(cfg=ModelConfig())", "Net")
    assert found is not None
    assert found.error is None
    assert found.overrides == {"cfg": ConfigCall("ModelConfig", {})}


def test_a_carried_call_keeps_its_own_literal_arguments():
    """A config class with a required field is answered inside the same line."""
    found = declared_init("# tensorviz: init=Net(cfg=ModelConfig(width=8), depth=2)", "Net")
    assert found.error is None
    assert found.overrides == {"cfg": ConfigCall("ModelConfig", {"width": 8}), "depth": 2}


@pytest.mark.parametrize(
    "declaration",
    [
        # A dotted path would have to be resolved through an import this tier cannot
        # follow; the user imports the name instead.
        "# tensorviz: init=Net(cfg=config.ModelConfig())",
        # One call, not a tree of them. The second is where a general expression
        # evaluator starts, and tensorViz is not going to have one.
        "# tensorviz: init=Net(cfg=ModelConfig(inner=Other()))",
        "# tensorviz: init=Net(cfg=ModelConfig(8))",
        # Still not a call, still not a literal: a bare name is a lookup in a
        # namespace neither tier has.
        "# tensorviz: init=Net(cfg=DEFAULT_CONFIG)",
        "# tensorviz: init=Net(cfg=width * 2)",
    ],
)
def test_anything_richer_than_one_call_is_refused_rather_than_evaluated(declaration):
    found = declared_init(declaration, "Net")
    assert found is not None
    assert found.error is not None
    assert found.overrides == {}


def test_a_line_that_is_not_a_call_is_refused():
    found = declared_init("# tensorviz: init=Net", "Net")
    assert found is not None
    assert found.error is not None


def test_the_declaration_for_another_class_is_not_this_class_s_declaration():
    """A file with three models carries three lines.

    Reading the wrong one builds the wrong model — with arguments that may well be
    valid for it, so the failure would be silent and the graph simply wrong.
    """
    two = "# tensorviz: init=Encoder(width=4)\n# tensorviz: init=Decoder(width=16)\n"
    assert declared_init(two, "Encoder").overrides == {"width": 4}
    assert declared_init(two, "Decoder").overrides == {"width": 16}
    assert declared_init(two, "Net") is None


def test_literal_containers_survive_the_round_trip():
    found = declared_init("# tensorviz: init=Net(dims=[4, 8], name='a', flag=True)", "Net")
    assert found.overrides == {"dims": [4, 8], "name": "a", "flag": True}
    assert found.error is None


# ───────────────────────────────────────────────────────── the drift guard ─────────
#: Every case both readers must answer identically. Pattern equality alone is not
#: enough: the two also have to agree on what a declaration MEANS, and that is where
#: a copied implementation actually drifts.
DRIFT_CASES = [
    ("# tensorviz: init=Net(hidden=8)", "Net"),
    ("# tensorviz: init=Net(hidden=8, depth=2)", "Net"),
    ("# tensorviz: init=Net(8)", "Net"),
    ("# tensorviz: init=Net(cfg=load())", "Net"),
    ("# tensorviz: init=Net(cfg=ModelConfig())", "Net"),
    ("# tensorviz: init=Net(cfg=ModelConfig(width=8), depth=2)", "Net"),
    ("# tensorviz: init=Net(cfg=config.ModelConfig())", "Net"),
    ("# tensorviz: init=Net(cfg=ModelConfig(inner=Other()))", "Net"),
    ("# tensorviz: init=Net(cfg=DEFAULT_CONFIG)", "Net"),
    ("# tensorviz: init=Net", "Net"),
    ("# tensorviz: init=", "Net"),
    ("#tensorviz:init=Net(hidden=8)", "Net"),
    ("# tensorviz: init=Other(hidden=8)", "Net"),
    ("", "Net"),
    ("# tensorviz: init=Net(dims=[1, 2], flag=False)", "Net"),
]


def test_both_tiers_spell_the_declaration_the_same_way():
    """The two patterns are duplicated on purpose, and may not drift.

    Same reason as ``test_declared_input_shape``: the worker runs with the USER's
    repository first on ``sys.path``, so importing ``utils.code_parser`` — and through
    it ``models.graph_models`` — lets any workspace with a ``models/`` package of its
    own shadow the backend's and break the worker's import outright.
    """
    from services.ingest import service

    from utils.code_parser import INIT_COMMENT

    assert service._INIT_COMMENT.pattern == INIT_COMMENT.pattern


def test_both_tiers_read_every_declaration_the_same_way():
    from services.ingest import service

    for source, cls in DRIFT_CASES:
        mine = declared_init(source, cls)
        theirs = service.declared_init(source, cls)
        if mine is None:
            assert theirs is None, source
            continue
        assert theirs is not None, source
        assert (mine.cls, mine.overrides) == (theirs[0], theirs[1]), source
        # The messages are written once each; what may not differ is WHETHER the
        # declaration was refused.
        assert (mine.error is None) == (theirs[2] is None), source
