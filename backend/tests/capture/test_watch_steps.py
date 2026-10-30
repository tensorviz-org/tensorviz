"""Watch: ``--steps N`` for ``--kind train_step`` (task.md P4).

What P4 claims of the worker half, and therefore what has to be true here:

* a watch is N runs of the user's OWN training step in one fresh process, so the
  series is a real curve — the loss moves because each step updated the weights the
  next one measured, not because a number was extrapolated;
* it answers the question no single step can: "which module went NaN at step 340".
  So the first NaN has to be reported at the step it actually happened and on the
  module it actually happened in, not on every card downstream of it;
* it is **additive under** ``result``. The five envelope keys and
  ``protocol_version`` 1 are the stdout contract (task.md §6.3) and a reader written
  against protocol 1 must see exactly what it saw before — same keys, same values,
  same types — and simply never look at ``result.diagnostics.series``;
* the payload is bounded on both axes, N ≤ 200 and per-node statistics only to the
  depth the canvas draws, because 200 steps × every leaf of a 300-module model is a
  JSON object nobody can carry across a pipe;
* and it stays JSON. ``json.dumps`` writes ``NaN``/``Infinity`` for those floats,
  which ``JSON.parse`` refuses — an overflowing gradient must reach the reader as a
  count, never as a float, or the whole envelope is lost at the exact moment it
  became interesting.

The worker is exercised as a **subprocess**, which is how it runs; no module-scope
``import torch`` (task.md §8.3) and every fixture tensor is eight units wide.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest


BACKEND = Path(__file__).parent.resolve().parent.parent
FIXTURE_REPO = BACKEND / "tests" / "fixtures" / "ide_repo"
MODELS = FIXTURE_REPO / "models"
SCRIPTS = FIXTURE_REPO / "scripts"

#: The interpreter under test is the one running pytest: it has torch, which is the
#: only thing the worker needs from a user environment.
WORKER_PYTHON = sys.executable

#: The learning rate the worker gives the optimiser it hands the user's function
#: (`worker.TRAIN_STEP_LR`). The NaN test replays a watch by hand to find out which
#: step SHOULD be the first bad one, and a replay at a different step size would be
#: an experiment about a different training run.
TRAIN_STEP_LR = 0.01


def run_worker(*arguments: str) -> Tuple[int, str, str]:
    """``(exit code, raw stdout, stderr)`` for one worker invocation.

    Raw, not parsed: two of the claims below are about the BYTES on stdout (one JSON
    object; no ``NaN`` token in it), and a helper that parsed for its caller would
    have already survived the thing being tested.
    """
    completed = subprocess.run(
        [WORKER_PYTHON, "-m", "services.ingest.worker", *arguments],
        cwd=str(BACKEND),
        capture_output=True,
        text=True,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
        timeout=600,
    )
    assert completed.stdout.strip(), f"worker printed nothing; stderr={completed.stderr[-2000:]}"
    return completed.returncode, completed.stdout, completed.stderr


def strict_json(raw: str) -> Dict[str, Any]:
    """Parse the way the extension does — refusing what JSON does not have.

    ``json.loads`` accepts ``NaN``, ``Infinity`` and ``-Infinity`` by default;
    ``JSON.parse`` does not. Every envelope in this file goes through here so that a
    payload the extension could not read fails in Python too.
    """

    def refuse(constant: str) -> None:
        raise AssertionError(f"the envelope carries {constant}, which is not JSON")

    return json.loads(raw, parse_constant=refuse)


def watch(*arguments: str) -> Dict[str, Any]:
    code, raw, stderr = run_worker(*arguments)
    envelope = strict_json(raw)
    assert code == 0, f"exit {code}: {envelope.get('error')} / {stderr[-2000:]}"
    return envelope


def series_of(envelope: Dict[str, Any]) -> List[Dict[str, Any]]:
    return envelope["result"]["diagnostics"]["series"]


@pytest.fixture(scope="module")
def five_steps() -> Dict[str, Any]:
    """Five steps of the fixture repository's own ``train_step`` on its own net."""
    return watch(
        "--kind", "train_step",
        "--file", str(MODELS / "resid.py"),
        "--cls", "Net",
        "--callable", f"{SCRIPTS / 'train_step.py'}:train_step",
        "--steps", "5",
        "--json",
    )


@pytest.fixture(scope="module")
def one_step() -> Dict[str, Any]:
    """The same run without ``--steps`` — the envelope protocol 1 already had."""
    return watch(
        "--kind", "train_step",
        "--file", str(MODELS / "resid.py"),
        "--cls", "Net",
        "--callable", f"{SCRIPTS / 'train_step.py'}:train_step",
        "--json",
    )


# ── the series ────────────────────────────────────────────────────────────────

def test_watching_five_steps_reports_one_entry_for_each_step_that_ran(
    five_steps: Dict[str, Any],
) -> None:
    """Five steps, numbered 1..5, each with the cards it measured.

    The numbering is the whole point of the entry: "NaN at step 340" is a sentence
    about a step index, and an index the reader has to infer from a list position is
    one that silently shifts the day a step is dropped.
    """
    entries = series_of(five_steps)
    assert [entry["step"] for entry in entries] == [1, 2, 3, 4, 5]
    for entry in entries:
        assert isinstance(entry["loss"], float), entry
        assert set(entry) == {"step", "loss", "per_node"}
        # l1, blocks.0 and l2 are the fixture net's parameterised cards. blocks.1 is
        # NOT among them and must not be: the ModuleList collapsed to one card in the
        # graph, and the series is keyed by the cards the canvas actually draws.
        assert set(entry["per_node"]) == {"l1", "blocks.0", "l2"}, entry["per_node"]
        for node_id, stats in entry["per_node"].items():
            assert set(stats) == {"grad_norm", "absmax", "nan"}, node_id
            assert stats["grad_norm"] > 0.0, f"{node_id} trained on a zero gradient"
            assert stats["nan"] == 0, node_id


def test_the_first_entry_is_the_step_the_graph_and_the_loss_already_describe(
    five_steps: Dict[str, Any],
) -> None:
    """Step 1 is the traced step, not a sixth one run beside it.

    If it were its own run, the graph, the spans and `_grad_stats` would describe a
    step that is not in the series, and the card the user clicks would be showing
    numbers from a different invocation than the strip beside it.
    """
    diagnostics = five_steps["result"]["diagnostics"]
    assert series_of(five_steps)[0]["loss"] == diagnostics["loss"]


def test_the_loss_moves_across_the_series_because_every_step_really_ran(
    five_steps: Dict[str, Any],
) -> None:
    """Strictly decreasing on this fixture: each step updated the weights the next
    one measured.

    A watch that ran the user's function once and repeated its answer would produce
    five identical losses and pass every other assertion in this file. This is the
    one that would fail.
    """
    losses = [entry["loss"] for entry in series_of(five_steps)]
    assert losses == sorted(losses, reverse=True), losses
    assert len(set(losses)) == 5, losses


# ── the question a watch exists to answer ─────────────────────────────────────

def _diverging_repo(tmp_path: Path) -> Tuple[Path, Path]:
    """A model that genuinely overflows, and a training step that drives it there.

    Not a stub that writes a NaN: `hot` grows by 1e20 every step until its output
    exceeds float32, and a softmax over an infinity is NaN — which is the real
    mechanism behind "block.3.attn.out went NaN at step 340". `calm` is on a parallel
    branch under a `sum`, so its gradient is the input and stays finite however bad
    `hot` gets: that is what makes "on ONE node" a claim with teeth.
    """
    model = tmp_path / "diverge.py"
    model.write_text(
        "import torch\n"
        "import torch.nn as nn\n"
        "\n"
        "\n"
        "class Diverge(nn.Module):\n"
        "    def __init__(self, width=8):\n"
        "        super().__init__()\n"
        "        self.hot = nn.Linear(width, width)\n"
        "        self.calm = nn.Linear(width, width)\n"
        "\n"
        "    def forward(self, x):\n"
        "        return torch.softmax(self.hot(x), dim=-1) + self.calm(x)\n"
        "\n"
        "\n"
        "# tensorviz: input=2,8\n",
        encoding="utf-8",
    )
    step = tmp_path / "step.py"
    step.write_text(
        "import torch\n"
        "\n"
        "\n"
        "def train_step(model, batch, optimizer):\n"
        "    optimizer.zero_grad()\n"
        "    loss = model(batch).sum()\n"
        "    loss.backward()\n"
        "    optimizer.step()\n"
        "    with torch.no_grad():\n"
        "        model.hot.weight.mul_(1e20)\n"
        "    return loss\n",
        encoding="utf-8",
    )
    return model, step


def _replayed_first_nan_step(model_file: Path, step_file: Path, steps: int) -> int:
    """Run that fixture by hand and say which step first has a NaN gradient.

    An independent reference for the assertion below, so the test is not the worker
    marking its own homework: the same two files, the same optimiser and step size,
    a loop written here. If the worker's series disagrees with this, one of them is
    lying about when the model died.
    """
    import torch

    module = _import_from(model_file, "diverge_fixture")
    step_module = _import_from(step_file, "diverge_step_fixture")
    torch.manual_seed(0)
    net = module.Diverge()
    batch = torch.randn(2, 8)
    optimizer = torch.optim.SGD(net.parameters(), lr=TRAIN_STEP_LR)
    for index in range(1, steps + 1):
        step_module.train_step(net, batch, optimizer)
        grad = net.hot.weight.grad
        if grad is not None and bool(torch.isnan(grad).any()):
            return index
    raise AssertionError("the diverging fixture stopped diverging; it measures nothing")


def _import_from(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_model_that_overflows_names_the_step_it_happened_at_and_the_one_card_it_happened_in(
    tmp_path: Path,
) -> None:
    """"NaN first observed at step N in <card>" has to be true on both halves.

    The step, because a run whose first NaN is reported one step late sends the user
    to the wrong place in their own curve; and the card, because a NaN that painted
    every module amber would name nothing at all.
    """
    model_file, step_file = _diverging_repo(tmp_path)
    expected_step = _replayed_first_nan_step(model_file, step_file, steps=5)

    envelope = watch(
        "--kind", "train_step",
        "--file", str(model_file),
        "--cls", "Diverge",
        "--callable", f"{step_file}:train_step",
        "--steps", "5",
        "--json",
    )
    entries = series_of(envelope)
    poisoned = {
        entry["step"]: sorted(
            node for node, stats in entry["per_node"].items() if stats["nan"] > 0
        )
        for entry in entries
    }
    first_bad = min((step for step, nodes in poisoned.items() if nodes), default=None)
    assert first_bad == expected_step, poisoned
    assert poisoned[first_bad] == ["hot"], poisoned
    for step in range(1, first_bad):
        assert poisoned[step] == [], poisoned
    # And the healthy branch keeps reporting real numbers after the crash, which is
    # what tells the user the failure is local rather than "the run is gone".
    for entry in entries:
        assert entry["per_node"]["calm"]["grad_norm"] > 0.0, entry


def test_a_nan_gradient_still_leaves_an_envelope_a_strict_reader_can_parse(
    tmp_path: Path,
) -> None:
    """No `NaN`/`Infinity` token anywhere on stdout, however bad the run went.

    `json.dumps(float("nan"))` writes `NaN`, `JSON.parse` throws on it, and the
    extension parses the envelope once with no recovery path — so the run that most
    needs to be read is the one a naive implementation loses whole.
    """
    model_file, step_file = _diverging_repo(tmp_path)
    code, raw, stderr = run_worker(
        "--kind", "train_step",
        "--file", str(model_file),
        "--cls", "Diverge",
        "--callable", f"{step_file}:train_step",
        "--steps", "4",
        "--json",
    )
    assert code == 0, stderr[-2000:]
    envelope = strict_json(raw)  # raises if NaN or Infinity is in there
    losses = [entry["loss"] for entry in series_of(envelope)]
    # A loss that is not a real number is reported as absent rather than as a float
    # JSON cannot express. The per-node counts are what names the failure anyway.
    assert None in losses, losses
    assert all(loss is None or isinstance(loss, float) for loss in losses), losses


# ── the bounds ────────────────────────────────────────────────────────────────

def test_per_node_statistics_stop_at_the_depth_the_canvas_draws(tmp_path: Path) -> None:
    """A card three levels down is in the graph and not in the series.

    The graph is drawn once; the series is drawn N times. Letting `--depth` widen
    both is how 200 steps of a deep model becomes a payload nobody can carry, so the
    watch depth is fixed independently of it.
    """
    model = tmp_path / "deep.py"
    model.write_text(
        "import torch.nn as nn\n"
        "\n"
        "\n"
        "class Inner(nn.Module):\n"
        "    def __init__(self, width=8):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(width, width)\n"
        "\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n"
        "\n"
        "\n"
        "class Mid(nn.Module):\n"
        "    def __init__(self, width=8):\n"
        "        super().__init__()\n"
        "        self.inner = Inner(width)\n"
        "\n"
        "    def forward(self, x):\n"
        "        return self.inner(x)\n"
        "\n"
        "\n"
        "# tensorviz: input=2,8\n"
        "class Deep(nn.Module):\n"
        "    def __init__(self, width=8):\n"
        "        super().__init__()\n"
        "        self.mid = Mid(width)\n"
        "\n"
        "    def forward(self, x):\n"
        "        return self.mid(x)\n",
        encoding="utf-8",
    )
    step = tmp_path / "step.py"
    step.write_text(
        "import torch\n"
        "\n"
        "\n"
        "def train_step(model, batch, optimizer):\n"
        "    optimizer.zero_grad()\n"
        "    loss = torch.nn.functional.mse_loss(model(batch), batch)\n"
        "    loss.backward()\n"
        "    optimizer.step()\n"
        "    return loss\n",
        encoding="utf-8",
    )
    envelope = watch(
        "--kind", "train_step",
        "--file", str(model),
        "--cls", "Deep",
        "--callable", f"{step}:train_step",
        "--steps", "2",
        "--depth", "4",
        "--json",
    )
    drawn = {
        str(node["id"]): (node.get("data") or {}).get("depth")
        for node in envelope["result"]["graph"]["nodes"]
    }
    assert drawn.get("mid.inner.fc") == 3, drawn
    for entry in series_of(envelope):
        assert set(entry["per_node"]) == {"mid", "mid.inner"}, entry["per_node"]


def test_more_steps_than_the_bound_are_refused_before_any_user_code_runs() -> None:
    """201 steps is a usage error, not a slow answer.

    Refused by argument checking, so nothing is imported, nothing is trained and the
    message names the bound — a caller who asked for 500 and silently got 200 would
    read a curve that stops as a run that stopped.
    """
    code, raw, stderr = run_worker(
        "--kind", "train_step",
        "--file", str(MODELS / "resid.py"),
        "--cls", "Net",
        "--callable", f"{SCRIPTS / 'train_step.py'}:train_step",
        "--steps", "201",
        "--json",
    )
    envelope = strict_json(raw)
    assert code == 2, raw[:400]
    assert envelope["result"] is None
    assert envelope["error"]["type"] == "UsageError"
    assert "200" in envelope["error"]["message"]
    assert "200" in stderr


def test_the_batch_check_reads_the_declaration_of_the_class_it_runs(tmp_path: Path) -> None:
    """Only ``Second`` declares a batch, inside its own body.

    The check before the run read the file for no class in particular, found no
    declaration that every class shared, and refused a run whose class had one. It
    now reads the traced class's declaration, which is the batch the run is fed.
    """
    model = tmp_path / "pair.py"
    model.write_text(
        "import torch.nn as nn\n"
        "\n"
        "\n"
        "class First(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(8, 8)\n"
        "\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n"
        "\n"
        "\n"
        "class Second(nn.Module):\n"
        "    # tensorviz: input=3,16\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(16, 16)\n"
        "\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n",
        encoding="utf-8",
    )
    envelope = watch(
        "--kind", "train_step",
        "--file", str(model),
        "--cls", "Second",
        "--callable", f"{SCRIPTS / 'train_step.py'}:train_step",
        "--json",
    )
    assert envelope["error"] is None


def test_steps_is_refused_for_a_kind_that_only_ever_runs_once() -> None:
    """`--kind model --steps 5` is a question about a training loop asked of a
    forward pass; answering it with one forward and no series would read as "this
    model does not change".
    """
    code, raw, _stderr = run_worker(
        "--file", str(MODELS / "resid.py"), "--cls", "Net", "--steps", "5", "--json"
    )
    envelope = strict_json(raw)
    assert code == 2, raw[:400]
    assert "train_step" in envelope["error"]["message"]


# ── the protocol ──────────────────────────────────────────────────────────────

def test_a_watch_changes_nothing_a_protocol_1_reader_reads(
    five_steps: Dict[str, Any], one_step: Dict[str, Any]
) -> None:
    """The five keys, version 1, and the same result shape as a run without --steps.

    This is the claim that lets `series` exist at all: it is additive UNDER `result`,
    so every key a protocol-1 reader knows is still there, still spelled the same and
    still holding the same kind of value.

    `series` is now declared on `IngestDiagnostics`, so it is present on EVERY run and
    empty on the ones that did not ask for steps. That is a deliberate change from
    "absent unless watched", and it is the stronger of the two properties: the field
    has to be on the model or a record read back from `.tensorviz/runs/` loses the
    curve silently (pydantic drops what it was not told about), and a key that is
    always there with one type is a key no reader has to branch on. An empty list is
    also the true answer — a single step has no curve.
    """
    for envelope in (five_steps, one_step):
        assert set(envelope) == {"protocol_version", "result", "spans", "error", "worker"}
        assert envelope["protocol_version"] == 1
        assert envelope["error"] is None
        assert envelope["worker"]["torch"]

    assert set(five_steps["result"]) == set(one_step["result"])
    watched = five_steps["result"]["diagnostics"]
    plain = one_step["result"]["diagnostics"]
    assert set(watched) == set(plain), "watching a run must not add a key to diagnostics"
    assert plain["series"] == [], "a run that asked for no steps measured no series"
    assert len(watched["series"]) == 5
    for key in plain:
        assert type(watched[key]) is type(plain[key]), key

    # And the parts a protocol-1 reader actually draws are untouched by watching.
    def shape(envelope: Dict[str, Any]) -> Tuple[List[str], List[Tuple[str, str]]]:
        graph = envelope["result"]["graph"]
        return (
            sorted(str(node["id"]) for node in graph["nodes"]),
            sorted((edge["source"], edge["target"]) for edge in graph["edges"]),
        )

    assert shape(five_steps) == shape(one_step)
    assert sorted(five_steps["spans"]) == sorted(one_step["spans"])


def test_the_series_is_the_only_thing_a_watch_adds_to_a_node(
    five_steps: Dict[str, Any], one_step: Dict[str, Any]
) -> None:
    """No card grew a field. The per-step numbers live in one place, under
    `diagnostics`, so a reader that walks the graph sees exactly the node payload it
    saw before and a 200-step watch does not multiply the size of every card.
    """

    def keys_by_node(envelope: Dict[str, Any]) -> Dict[str, List[str]]:
        return {
            str(node["id"]): sorted((node.get("data") or {}).get("parameters") or {})
            for node in envelope["result"]["graph"]["nodes"]
        }

    assert keys_by_node(five_steps) == keys_by_node(one_step)


def test_a_watch_survives_being_read_back_as_a_run_record(
    five_steps: Dict[str, Any]
) -> None:
    """The curve has to still be there after the round-trip the Runs view does.

    The extension writes a worker envelope into ``.tensorviz/runs/<window>.jsonl`` and
    this process reads it back as a :class:`TraceRecord`. Pydantic ignores fields it
    was not told about, so a ``diagnostics`` model without ``series`` would validate
    the record happily, report no error, and hand back a run whose sparkline had
    quietly disappeared — visible only to a user, on reload, with nothing in any log.

    Asserted through the model rather than on the field's declaration so it fails if
    ``series`` is ever dropped, renamed, or given a type that refuses what the worker
    actually writes (floats, ``None`` losses, integer NaN counts).
    """
    from services.ingest.spec import TraceRecord

    result = five_steps["result"]
    series = result["diagnostics"]["series"]
    assert len(series) == 5, "fixture precondition: the worker emitted a five-step series"

    record = TraceRecord(
        kind="train_step",
        target="train.py::Net",
        graph=result["graph"],
        spans=five_steps["spans"],
        diagnostics=result["diagnostics"],
        run_id="run-1",
        started_at="2026-09-06T00:00:00Z",
    )
    reloaded = TraceRecord.model_validate(json.loads(record.model_dump_json()))

    assert reloaded.diagnostics.series == series
