"""Runtime capture: what the numbers were, and where a forward went wrong (task.md M8).

Shapes say a model is wired correctly. They say nothing about whether it is
*working*: a layer whose activations have collapsed to zero, or gone to NaN, has
exactly the shape it had yesterday. M8 measures the numbers, and the two claims
that have to hold for those measurements to be worth anything are:

* **measuring must not change the model.** The capture hooks run inside the same
  ``TorchDispatchMode`` that reconstructs the graph, so a reduction taken in the
  wrong place is recorded as part of the user's forward and rewrites their edges.
  This is not hypothetical — it is what ``register_full_backward_hook`` did here
  (nine aten ops became sixteen, five phantom ``flatten`` nodes appeared and the
  residual ``add`` was lost), and it is why the gradients come off tensor hooks;
* **a NaN must survive being reported.** ``tensor.mean()`` over a tensor holding
  one NaN is NaN, which serialises to ``null`` and deletes the two numbers a
  reader needs at the exact moment they matter. The reductions skip non-finite
  elements and count them separately instead.

Plus the third M8 claim, which is about failure rather than numbers: a forward
that raises names the line it raised on and the node that owns it.

The worker runs as a **subprocess** here, as it does in the product — no
module-scope ``import torch`` (task.md §8.3), and every fixture is eight units
wide with a batch of two.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest


from models.graph_models import CodeParseRequest  # noqa: E402
from services.ide import projection  # noqa: E402
from utils.code_parser import CodeParserService  # noqa: E402

BACKEND = Path(__file__).parent.resolve().parent.parent
FIXTURE_REPO = BACKEND / "tests" / "fixtures" / "ide_repo"
MODELS = FIXTURE_REPO / "models"

#: The interpreter under test is the one running pytest: it has torch, which is the
#: only thing the worker needs from a user environment.
WORKER_PYTHON = sys.executable


def run_worker(*arguments: str) -> Tuple[int, Dict[str, Any], str]:
    """``(exit code, envelope, stderr)`` for one worker invocation."""
    completed = subprocess.run(
        [WORKER_PYTHON, "-m", "services.ingest.worker", *arguments],
        cwd=str(BACKEND),
        capture_output=True,
        text=True,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
        timeout=300,
    )
    assert completed.stdout.strip(), f"worker printed nothing; stderr={completed.stderr[-2000:]}"
    return completed.returncode, json.loads(completed.stdout), completed.stderr


def trace_of(file: Path, cls: str, *arguments: str) -> Dict[str, Any]:
    code, envelope, stderr = run_worker(
        "--file", str(file), "--cls", cls, "--json", *arguments
    )
    assert code == 0, f"exit {code}: {envelope.get('error')} / {stderr[-2000:]}"
    return envelope


def params_of(envelope: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(node["id"]): ((node.get("data") or {}).get("parameters") or {})
        for node in envelope["result"]["graph"]["nodes"]
    }


def module_nodes(envelope: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Nodes an ``nn.Module`` produced, which are the only ones a hook can measure.

    A materialised node — the residual ``add`` — has a ``_module_path`` but no
    module behind it, so it carries no statistics. That is an answer, not a gap.
    """
    materialised = set(envelope["result"]["diagnostics"]["materialised_ops"])
    return {
        node_id: params
        for node_id, params in params_of(envelope).items()
        if params.get("_module_path") and node_id not in materialised
    }


def reached(params: Dict[str, Any]) -> bool:
    return params.get("_shape_provenance") == "observed"


@pytest.fixture()
def raising_module(tmp_path: Path) -> Tuple[Path, int]:
    """A net whose second layer raises, and the line number of the ``raise``.

    Written here rather than added to ``tests/fixtures/ide_repo`` because it must
    never be importable by the workspace scan: that fixture's contract is exactly
    four classes in ``models/`` (task.md M3).
    """
    module = tmp_path / "boom.py"
    lines = [
        "import torch.nn as nn",
        "",
        "",
        "class Raiser(nn.Module):",
        "    def forward(self, x):",
        "        raise RuntimeError('the activations are not finite')",
        "",
        "",
        "class Boom(nn.Module):",
        "    def __init__(self, width=8):",
        "        super().__init__()",
        "        self.fc1 = nn.Linear(width, width)",
        "        self.fc2 = Raiser()",
        "",
        "    def forward(self, x):",
        "        h = self.fc1(x)",
        "        return self.fc2(h)",
    ]
    module.write_text("\n".join(lines) + "\n", encoding="utf-8")
    raise_line = next(i for i, line in enumerate(lines, start=1) if "raise RuntimeError" in line)
    return module, raise_line


@pytest.fixture()
def nan_module(tmp_path: Path) -> Path:
    """A forward that divides by zero, so ``fc2`` is fed NaN and nothing else is."""
    module = tmp_path / "nanny.py"
    module.write_text(
        "import torch\n"
        "import torch.nn as nn\n"
        "\n"
        "\n"
        "class Nanny(nn.Module):\n"
        "    def __init__(self, width=8):\n"
        "        super().__init__()\n"
        "        self.fc1 = nn.Linear(width, width)\n"
        "        self.fc2 = nn.Linear(width, width)\n"
        "\n"
        "    def forward(self, x):\n"
        "        h = self.fc1(x)\n"
        "        return self.fc2(h / torch.zeros_like(h))\n",
        encoding="utf-8",
    )
    return module


# ── activations (task.md M8) ──────────────────────────────────────────────────

def test_every_reached_module_reports_finite_activation_statistics():
    """The M8 acceptance, stated as the thing a reader relies on.

    ``mean`` and ``std`` are what say "this layer is alive"; a null, an infinity
    or a NaN in either would be indistinguishable from "we did not look".
    """
    envelope = trace_of(MODELS / "resid.py", "Net", "--input", "2,8", "--capture", "activations")
    measured = {
        node_id: params
        for node_id, params in module_nodes(envelope).items()
        if reached(params)
    }
    assert measured, "no module node was reached; the fixture stopped tracing"
    for node_id, params in measured.items():
        stats = params.get("_act_stats")
        assert stats, f"{node_id} carries no _act_stats"
        assert set(stats) == {"mean", "std", "absmax", "nan", "inf", "dtype", "device"}
        for key in ("mean", "std", "absmax"):
            value = stats[key]
            assert isinstance(value, float), f"{node_id}.{key} is {value!r}"
            assert value == value and abs(value) != float("inf"), f"{node_id}.{key} = {value}"
        assert stats["dtype"] == "float32" and stats["device"] == "cpu"


def test_nothing_is_measured_unless_it_was_asked_for():
    """Opt-in is a safety property, not a preference: the capture holds real
    activations alive in the user's process for the length of a forward."""
    envelope = trace_of(MODELS / "resid.py", "Net", "--input", "2,8")
    assert not any("_act_stats" in params for params in params_of(envelope).values())
    assert not any("_grad_stats" in params for params in params_of(envelope).values())


def test_measuring_the_model_does_not_change_the_model():
    """The graph a capture produces must be the graph without one, op for op.

    This is the regression guard for the mistake M8 actually made: hooks that
    reduce (or wrap) tensors inside the tracer's dispatch mode have their own
    work recorded as part of the user's forward. ``register_full_backward_hook``
    took this fixture from 9 aten ops to 16 and invented five ``flatten`` nodes.
    A debugging feature that silently rewrites the graph being debugged is worse
    than no debugging feature.
    """
    plain = trace_of(MODELS / "resid.py", "Net", "--input", "2,8")
    measured = trace_of(
        MODELS / "resid.py", "Net", "--input", "2,8",
        "--capture", "activations", "--capture", "grads", "--loss", "out.sum()",
    )

    def structure(envelope: Dict[str, Any]) -> Dict[str, Any]:
        diagnostics = envelope["result"]["diagnostics"]
        graph = envelope["result"]["graph"]
        return {
            "aten_ops": diagnostics["aten_ops"],
            "frames_captured": diagnostics["frames_captured"],
            "materialised": diagnostics["materialised_ops"],
            "nodes": sorted(str(node["id"]) for node in graph["nodes"]),
            "edges": sorted(
                (str(edge["source"]), str(edge["target"])) for edge in graph["edges"]
            ),
        }

    assert structure(measured) == structure(plain)


def test_a_nan_is_counted_rather_than_swallowing_the_statistics(nan_module: Path):
    """The whole reason to look at activations at all.

    ``fc2`` is fed nothing but NaN, so its own output is entirely NaN. The count
    is the alarm; ``mean`` and ``std`` stay real numbers because a reader who
    opens a card to find `null` learns less than one who finds a zero next to
    "16 NaN".
    """
    envelope = trace_of(nan_module, "Nanny", "--input", "2,8", "--capture", "activations")
    params = params_of(envelope)
    clean, poisoned = params["fc1"]["_act_stats"], params["fc2"]["_act_stats"]
    assert clean["nan"] == 0 and clean["inf"] == 0
    assert poisoned["nan"] > 0
    for key in ("mean", "std", "absmax"):
        value = poisoned[key]
        assert value == value and abs(value) != float("inf"), f"{key} = {value}"


# ── gradients (task.md M8) ────────────────────────────────────────────────────

def test_every_parameterised_node_reports_gradient_statistics():
    """With a loss the user supplied, and only with one.

    A gradient is the answer to "is this layer learning", and it is meaningless
    without saying what is being minimised — which is why ``--loss`` has no
    default and the run below names it.
    """
    envelope = trace_of(
        MODELS / "resid.py", "Net", "--input", "2,8",
        "--capture", "grads", "--loss", "out.sum()",
    )
    parameterised = {
        node_id: params
        for node_id, params in module_nodes(envelope).items()
        if reached(params) and float(params.get("_params_M") or 0) > 0
    }
    assert parameterised, "the fixture has no parameterised layers"
    for node_id, params in parameterised.items():
        stats = params.get("_grad_stats")
        assert stats, f"{node_id} carries no _grad_stats"
        for key in ("mean", "std", "absmax"):
            value = stats[key]
            assert isinstance(value, float) and value == value, f"{node_id}.{key} = {value}"
    # `out = x + self.l2(h)`, so d(out.sum())/d(l2's output) is 1 everywhere. The
    # exact number is what proves these are gradients and not another copy of the
    # activations — a mix-up no "is it finite" assertion could catch.
    assert parameterised["l2"]["_grad_stats"]["mean"] == pytest.approx(1.0)
    assert parameterised["l2"]["_grad_stats"]["std"] == pytest.approx(0.0)


def test_a_loss_expression_that_fails_is_reported_and_does_not_lose_the_graph():
    """Fail loud, but only about the half that failed (CLAUDE.md rule 12).

    The forward succeeded, so the structure and the shapes are real and the user
    keeps them. What they must not get is a graph with no gradients on it and no
    reason given, which reads as "this model has none".
    """
    envelope = trace_of(
        MODELS / "resid.py", "Net", "--input", "2,8",
        "--capture", "grads", "--loss", "out.no_such_method()",
    )
    warnings = envelope["result"]["diagnostics"]["warnings"]
    assert any("no_such_method" in warning for warning in warnings), warnings
    assert not any("_grad_stats" in params for params in params_of(envelope).values())
    assert params_of(envelope)["l1"]["_out_shape"] == "2x8"


@pytest.mark.parametrize(
    "arguments, expected",
    [
        (("--capture", "activations", "--tier", "fake"), "--tier real"),
        (("--capture", "grads"), "--capture grads needs --loss"),
        (("--loss", "out.sum()"), "--loss does nothing"),
    ],
)
def test_a_capture_that_cannot_be_honoured_is_refused_rather_than_half_run(
    arguments: Tuple[str, ...], expected: str
):
    """Exit 2 with the reason, for every flag combination this cannot perform.

    Statistics of a fake tensor are statistics of nothing, and a gradient needs a
    loss. Running anyway with the impossible half dropped would answer a question
    the user did not ask.
    """
    code, envelope, stderr = run_worker(
        "--file", str(MODELS / "resid.py"), "--cls", "Net", "--input", "2,8",
        "--json", *arguments,
    )
    assert code == 2
    assert envelope["result"] is None
    assert expected in envelope["error"]["message"]
    assert expected in stderr


# ── error -> node (task.md M8) ────────────────────────────────────────────────

def test_a_forward_that_raises_names_the_line_and_the_node(
    raising_module: Tuple[Path, int]
):
    """The M8 acceptance: ``error.node_id == 'fc2'`` and ``error.line`` is the raise.

    Both halves are needed and they are different questions. The line is where an
    editor underlines, and it is inside the module that failed — deeper than the
    ``self.fc2(h)`` that called it. The node is which card goes red, and it is
    the module the run entered and never left.
    """
    module, raise_line = raising_module
    code, envelope, _stderr = run_worker(
        "--file", str(module), "--cls", "Boom", "--input", "2,8", "--json"
    )
    assert code == 4
    error = envelope["error"]
    assert error["node_id"] == "fc2"
    assert error["line"] == raise_line
    assert error["file"] == str(module)
    # The graph up to the failure is still the useful half.
    assert params_of(envelope)["fc1"]["_shape_provenance"] == "observed"


def test_the_error_line_is_the_users_own_frame_and_not_torchs():
    """A shape mismatch raises several frames deep inside torch. The frame the
    editor can act on is the last one in the user's own file, and picking any
    other lands the diagnostic in site-packages."""
    module = MODELS / "resid.py"
    code, envelope, _stderr = run_worker(
        "--file", str(module), "--cls", "Net", "--input", "2,9", "--json"
    )
    assert code == 4
    error = envelope["error"]
    assert error["file"] == str(module)
    assert "site-packages" not in str(error["file"])
    assert any("site-packages" in frame["file"] for frame in error["frames"]), (
        "the raw frames must still carry the whole traceback"
    )


# ── the merge (task.md §5) ────────────────────────────────────────────────────

def test_the_statistics_survive_the_merge_onto_the_projected_document():
    """The canvas reads ``GraphDocumentV2``, not the worker envelope.

    Between the two sit the identity rewrite and the AST merge, both of which
    rebuild every node's parameter dict. The card cannot go amber for a NaN it
    never received, so the crossing is worth one test of its own.
    """
    envelope = trace_of(
        MODELS / "resid.py", "Net", "--input", "2,8",
        "--capture", "activations", "--capture", "grads", "--loss", "out.sum()",
    )
    rel = "models/resid.py"
    source = (MODELS / "resid.py").read_text(encoding="utf-8")
    parsed = CodeParserService().parse_pytorch_code(CodeParseRequest(code=source, file=rel))
    entry = next(e for e in parsed["graphs"] if e["name"] == "Net")
    result = projection.project(
        {
            "nodes": [n.model_dump() for n in entry["graph"].nodes],
            "edges": [e.model_dump() for e in entry["graph"].edges],
        },
        envelope,
        cls="Net",
        rel_file=rel,
        source=source,
    )
    measured = {
        node.node_id: node.params
        for node in result.document.semantic.nodes
        if node.params.get("_act_stats")
    }
    assert set(measured) >= {"l1", "l2"}
    assert all(isinstance(params["_act_stats"]["mean"], float) for params in measured.values())
    assert measured["l2"]["_grad_stats"]["mean"] == pytest.approx(1.0)
