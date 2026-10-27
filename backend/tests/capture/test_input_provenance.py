"""Whether anybody chose the tensor the shapes were measured on.

A trace is only as true as its input. When a file says ``# tensorviz: input=2,8``
the run measures the model the user meant; when it says nothing,
``services/ingest/inputs.py`` seeds a tensor off the first shape-bearing layer,
or failing that brute-forces up to eighty candidate tuples until one survives a
forward pass. Both produce ``tier="real"``, the same node count, the same number
of shaped nodes and ``input_candidates_tried == 1`` — so the card that reads
"In 2x8" and the card that reads "In 1x8" were equally confident, and only one of
them was describing the model the user opened.

Measured before the fix: stripping the declaration off the ``resid`` fixture left
byte-identical diagnostics except ``example_input``, which went from ``[[2, 8]]``
to ``[[1, 8]]``. Across the example corpus, 11 of 45 classes traced on an input
nobody had asked for, with no warning anywhere.

So the run has to say where its input came from. These pin that it does, and that
the two cases cannot be confused for each other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pytest


from services.ingest.service import ingest_source  # noqa: E402
from services.ingest.spec import IngestDiagnostics, ModelRef  # noqa: E402

FIXTURE = Path(__file__).parent.resolve().parent / "fixtures" / "ide_repo" / "models" / "resid.py"
DECLARATION = "# tensorviz: input=2,8"


def _run(path: Path, **kwargs: Any) -> Tuple[IngestDiagnostics, Dict[str, Any]]:
    """``(diagnostics, the Input node's params)`` for one ingest of ``path``."""
    result = ingest_source(ModelRef(file=str(path), cls="Net", depth=2), **kwargs)
    node = next(n for n in result.graph["nodes"] if n["id"] == "input_node")
    return result.diagnostics, node["data"]["parameters"]


@pytest.fixture
def stripped(tmp_path: Path) -> Path:
    """The same class with its declaration removed, and nothing else changed.

    A copy rather than an edit of the fixture: `resid.py` is the shared subject of
    the worker and projection suites, and the whole point here is that the two
    files differ in exactly one line.
    """
    source = FIXTURE.read_text(encoding="utf-8")
    assert DECLARATION in source, "the fixture must declare an input for this to mean anything"
    target = tmp_path / "resid.py"
    target.write_text(source.replace(DECLARATION, ""), encoding="utf-8")
    return target


def test_a_declared_input_is_reported_as_declared():
    diagnostics, _ = _run(FIXTURE)
    assert diagnostics.example_input == [[2, 8]]
    assert diagnostics.input_provenance == "declared"


def test_an_input_read_off_the_first_layer_says_so(stripped: Path):
    # `Linear(8, 8)` states its own width, so the seed is a good guess — and still
    # a guess. `randn(1, 8)` is a batch of one nobody asked for.
    diagnostics, _ = _run(stripped)
    assert diagnostics.example_input == [[1, 8]]
    assert diagnostics.input_provenance == "seed-from-first-layer"


def test_a_caller_supplied_input_outranks_the_file_and_is_labelled_as_its_own():
    # `--input` is the flag the QA driver and the Validate path pass. It is not the
    # file's declaration and must not be reported as one: the file still says 2,8.
    diagnostics, _ = _run(FIXTURE, example_input="3,8")
    assert diagnostics.example_input == [[3, 8]]
    assert diagnostics.input_provenance == "explicit"


def test_the_two_runs_are_told_apart_by_something_other_than_the_numbers(stripped: Path):
    """The defect, stated as a test.

    Every field an operator would compare is equal across the two runs — tier,
    shaped nodes, candidates tried. The only thing that moved is the tensor, which
    is precisely the number being asked to be trusted. Without a provenance the
    guess is indistinguishable from the statement, so this asserts the pair
    differs, not merely that each value is right.
    """
    declared, _ = _run(FIXTURE)
    guessed, _ = _run(stripped)

    assert declared.tier == guessed.tier == "real"
    assert declared.nodes_with_observed_shapes == guessed.nodes_with_observed_shapes
    assert (
        declared.timings_ms["input_candidates_tried"]
        == guessed.timings_ms["input_candidates_tried"]
        == 1
    )
    assert declared.example_input != guessed.example_input
    assert declared.input_provenance != guessed.input_provenance


def test_the_input_card_carries_the_same_answer_the_diagnostics_do(stripped: Path):
    """The diagnostics are an operator's view; the card is the user's.

    ``_input_provenance`` rides on the Input node's params under the leading
    underscore every projection-attached fact uses (``canvas``'s
    ``cardParams.ts`` hides those from the drawn parameter list), because that bag
    is the only per-node channel that survives the projection merge into the
    document the canvas reads.
    """
    declared_diagnostics, declared_params = _run(FIXTURE)
    guessed_diagnostics, guessed_params = _run(stripped)

    assert declared_params["_input_provenance"] == declared_diagnostics.input_provenance
    assert guessed_params["_input_provenance"] == guessed_diagnostics.input_provenance
    # And the shape it qualifies is still right there next to it.
    assert declared_params["shape"] == "2,8"
    assert guessed_params["shape"] == "1,8"
