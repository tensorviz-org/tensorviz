"""The same model, traced twice, must draw the same graph.

Three independent draws from the global RNG reach a ``tier='real'`` trace, and two
of them change the *picture*, not just the numbers:

1. **the input.** With no ``# tensorviz: input=`` declaration the shape search seeds
   a ``torch.randn`` tensor off the first shape-bearing layer, so ``if x.sum() > 0``
   is a coin toss. Measured on an undeclared branch before the fix: 10 runs of one
   process gave ``unreached=[]`` six times and ``unreached=['up']`` four times — two
   different graphs, 3 edges against 4.
2. **model construction.** ``repo.instantiate`` runs the user's own ``__init__``, and
   ``nn.Linear``/``nn.Conv2d`` draw their weights from the same global stream. This
   one survives a declared input: measured on ``probe_gate`` below, whose input is
   pinned to ``torch.full((2, 8), -1.0)``, 40 runs still split 24 / 16.
3. the forward — already handled, and not re-tested here.
   ``trace.trace_model`` puts a plain forward in ``eval()`` (services/ingest/trace.py),
   so Dropout is a no-op and the only stochastic path left is the deliberate
   ``--kind train_step`` exception.

Both of the first two have to be pinned at once. ``probe_gate`` is the case that
proves it: seeding only the input leaves it flipping exactly as measured above, so a
fix that converts ``inputs.py`` to a local ``torch.Generator`` would pass
``test_an_undeclared_branch_draws_the_same_graph_every_time`` and fail this one.

The runs share a process on purpose. That is how the IDE's own suite calls
``ingest_source``, it is the case a bare ``torch.manual_seed`` would poison for 1600
other tests, and ``test_an_ingest_leaves_the_callers_rng_untouched`` is what pins
that the fix does not.

No module-scope ``import torch`` (task.md §8.3); the fixture classes are eight units
wide. ``RUNS`` is 12 because the flip is not certain per run: at the measured 24/16
split, twelve identical runs by luck is a 0.2% event, and the undeclared case is
nearer 50/50 than that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest


from services.ingest.service import ingest_source  # noqa: E402
from services.ingest.spec import ModelRef  # noqa: E402

RUNS = 12


def _drawing(path: Path, cls: str) -> Tuple[Any, ...]:
    """What the canvas would draw, as a comparable key.

    Deliberately not the whole payload: this asserts the *graph* is reproducible,
    and node order or a timing in milliseconds failing to match would be a different
    (and untrue) claim. Which modules were dimmed, which edges were drawn and how
    many cards carry a measured shape are the three things an unpinned RNG was
    observed to change.
    """
    result = ingest_source(ModelRef(file=str(path), cls=cls))
    diagnostics = result.diagnostics
    return (
        tuple(diagnostics.unreached_modules),
        tuple(sorted((e["source"], e["target"]) for e in result.graph["edges"])),
        diagnostics.nodes_with_observed_shapes,
        tuple(tuple(shape) for shape in diagnostics.example_input),
    )


def _write(tmp_path: Path, name: str, lines: List[str]) -> Path:
    module = tmp_path / name
    module.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return module


@pytest.fixture()
def undeclared_gate(tmp_path: Path) -> Path:
    """A branch on the input's own values, with nothing declaring that input.

    Written here rather than added to ``tests/fixtures/ide_repo`` for the reason
    ``test_worker_capture`` gives for the same move: that fixture's contract is
    exactly four classes in ``models/``, and its own ``gate.py`` *does* declare an
    input, which is precisely the randomness this test needs left in.
    """
    return _write(tmp_path, "undeclared_gate.py", [
        "import torch.nn as nn",
        "",
        "",
        "class Gate(nn.Module):",
        "    def __init__(self, width=8):",
        "        super().__init__()",
        "        self.up = nn.Linear(width, width)",
        "        self.down = nn.Linear(width, width)",
        "",
        "    def forward(self, x):",
        "        if x.sum() > 0:",
        "            x = self.up(x)",
        "        return self.down(x)",
    ])


@pytest.fixture()
def probe_gate(tmp_path: Path) -> Path:
    """A branch on a WEIGHT, with the input pinned to a constant.

    ``torch.full`` rather than the ``2,8`` shorthand, and that distinction is the
    whole fixture: the shorthand pins the shape and fills it with ``torch.randn``,
    so it would leave source (1) live and this test would no longer isolate source
    (2). With the expression form the only thing left moving is ``probe``'s weights.
    """
    return _write(tmp_path, "probe_gate.py", [
        "import torch.nn as nn",
        "",
        "",
        "class ProbeGate(nn.Module):",
        "    def __init__(self, width=8):",
        "        super().__init__()",
        "        self.probe = nn.Linear(width, width)",
        "        self.up = nn.Linear(width, width)",
        "        self.down = nn.Linear(width, width)",
        "",
        "    def forward(self, x):",
        "        if self.probe(x).sum() > 0:",
        "            x = self.up(x)",
        "        return self.down(x)",
        "",
        "",
        "# tensorviz: input=torch.full((2, 8), -1.0)",
    ])


def test_an_undeclared_branch_draws_the_same_graph_every_time(undeclared_gate: Path) -> None:
    """Source (1): the searched input's VALUES, not just its shape."""
    drawings = {_drawing(undeclared_gate, "Gate") for _ in range(RUNS)}
    assert len(drawings) == 1, (
        f"{RUNS} traces of one unchanged file drew {len(drawings)} different graphs: "
        f"{sorted(drawings)}"
    )


def test_a_branch_on_a_weight_draws_the_same_graph_every_time(probe_gate: Path) -> None:
    """Source (2): construction. The input here is already pinned by the file.

    This is the test that fails if someone later "fixes" reproducibility by handing
    ``inputs.py`` a local ``torch.Generator``: ``nn.Linear.__init__`` takes no
    ``generator`` argument, so nothing local can reach the user's own ``__init__``.
    """
    first = _drawing(probe_gate, "ProbeGate")
    assert first[3] == ((2, 8),), "the declaration must be the input, or this proves nothing"
    drawings = {first} | {_drawing(probe_gate, "ProbeGate") for _ in range(RUNS - 1)}
    assert len(drawings) == 1, (
        f"{RUNS} traces with a declared input drew {len(drawings)} different graphs, "
        f"so construction is still unpinned: {sorted(drawings)}"
    )


def test_an_ingest_leaves_the_callers_rng_untouched(undeclared_gate: Path) -> None:
    """The in-process constraint: seeding the trace must not seed the caller.

    ``ingest_source`` is called in the same process as 1600 other tests and by the
    FastAPI server (api/routers/ingest.py), so a bare ``torch.manual_seed`` would
    make every later caller's stream depend on whether a graph had been opened.
    """
    import torch  # noqa: PLC0415 - kept out of module scope (task.md §8.3)

    torch.manual_seed(4242)
    before = torch.get_rng_state().clone()
    _drawing(undeclared_gate, "Gate")
    assert torch.equal(before, torch.get_rng_state()), (
        "the ingest advanced the caller's CPU generator"
    )


def test_the_run_reports_the_seed_it_used(undeclared_gate: Path) -> None:
    """A stable graph the user cannot account for is a graph they cannot trust.

    Seed 0 picks one side of every data-dependent branch and says nothing about it;
    the reported value is what makes "which graph did I get" answerable at all, and
    the hint that the answer came from an RNG rather than from the model.
    """
    from services.ingest.service import _TRACE_SEED  # noqa: PLC0415

    result = ingest_source(ModelRef(file=str(undeclared_gate), cls="Gate"))
    assert result.diagnostics.seed == _TRACE_SEED


def test_the_seed_survives_a_diagnostics_round_trip(undeclared_gate: Path) -> None:
    """Pydantic drops what it was not told about, which is how ``series`` was lost
    once already (services/ingest/spec.py). The worker's JSON is the only way the
    extension ever sees a diagnostic, so an undeclared field is a field that does
    not exist in production."""
    from services.ingest.spec import IngestDiagnostics  # noqa: PLC0415

    result = ingest_source(ModelRef(file=str(undeclared_gate), cls="Gate"))
    payload: Dict[str, Any] = result.diagnostics.model_dump()
    assert IngestDiagnostics(**payload).seed == result.diagnostics.seed
