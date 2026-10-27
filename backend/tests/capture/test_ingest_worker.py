"""The trace worker and the executed projection it feeds (task.md M4).

What M4 claims, and therefore what has to be true here:

* the worker is a **process contract**, not a Python API — one JSON object on
  stdout and an exit code that says what happened, so the extension can act
  without parsing prose;
* ``tier='real'`` genuinely runs real tensors, which is the only reason
  ``if x.sum() > 0`` traces at all (``torch.export`` refuses it; that refusal is
  why D3 chose eager tracing);
* every node can say where in the user's source it came from — a module by the
  file its class was written in, a functional op by the ``forward()`` line the
  tracer caught it on;
* the merge lets the trace win the structure while the parse keeps its spans and
  every layer the run never reached, because a graph that quietly loses the layer
  in the untaken branch is worse than one that shows it dimmed.

The worker is exercised as a **subprocess** throughout, which is how it actually
runs. That is not ceremony: half of what it promises (nothing written to disk, the
parent's ``sys.path`` untouched, an exit code) is only observable from outside it.
No module-scope ``import torch`` (task.md §8.3); the fixture classes are eight
units wide.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def trace_of(name: str, cls: str, *arguments: str) -> Dict[str, Any]:
    code, envelope, stderr = run_worker(
        "--file", str(MODELS / f"{name}.py"), "--cls", cls, "--json", *arguments
    )
    assert code == 0, f"exit {code}: {envelope.get('error')} / {stderr[-2000:]}"
    return envelope


def nodes_of(envelope: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(node["id"]): ((node.get("data") or {}).get("parameters") or {})
        for node in envelope["result"]["graph"]["nodes"]
    }


def project(name: str, cls: str, envelope: Optional[Dict[str, Any]]) -> projection.Projection:
    """The merged projection for a fixture class, as ``/api/ide/project`` computes it."""
    rel = f"models/{name}.py"
    source = (MODELS / f"{name}.py").read_text(encoding="utf-8")
    parsed = CodeParserService().parse_pytorch_code(
        CodeParseRequest(code=source, file=rel)
    )
    entry = next(e for e in parsed["graphs"] if e["name"] == cls)
    graph = entry["graph"]
    return projection.project(
        {
            "nodes": [n.model_dump() for n in graph.nodes],
            "edges": [e.model_dump() for e in graph.edges],
        },
        envelope,
        cls=cls,
        rel_file=rel,
        source=source,
    )


# ── the process contract (task.md §6.3) ───────────────────────────────────────

def test_the_worker_prints_exactly_the_envelope_the_protocol_fixes():
    """The extension reads this with `JSON.parse` and no recovery path.

    So the keys are the contract, not a convenience: a worker that answered with a
    bare result would be indistinguishable, to the caller, from one whose stdout had
    been corrupted by a `print` in the user's own module scope.
    """
    envelope = trace_of("resid", "Net", "--input", "2,8")
    assert set(envelope) == {"protocol_version", "result", "spans", "error", "worker"}
    assert envelope["protocol_version"] == 1
    assert envelope["error"] is None
    assert envelope["worker"]["torch"]
    assert envelope["worker"]["python"] == ".".join(str(p) for p in sys.version_info[:3])
    assert isinstance(envelope["worker"]["elapsed_ms"], (int, float))


def test_the_backend_merges_the_same_protocol_version_the_worker_prints():
    """Two constants, one contract. ``services/ide/projection.py`` declares its own
    rather than importing the worker's, because importing it pulls in torch on the
    hot path of every keystroke — so this is what keeps them from drifting."""
    from services.ingest import worker

    assert worker.PROTOCOL_VERSION == projection.WORKER_PROTOCOL_VERSION


def test_a_class_that_needs_an_argument_is_refused_by_name_and_not_guessed_at():
    """Exit 3, and the message says which class and what it wanted.

    The repository importer has a five-rung ladder for guessing constructor
    arguments; the IDE deliberately does not use it. The user clicked one `class`
    line, and building it from somebody else's config object would put a model on
    the canvas that their code never constructs.
    """
    code, envelope, _stderr = run_worker(
        "--file", str(MODELS / "cfg.py"), "--cls", "Configured", "--json"
    )
    assert code == 3
    assert envelope["result"] is None
    assert "Configured" in envelope["error"]["message"]
    assert "cfg" in envelope["error"]["message"]


def test_bad_arguments_exit_two_with_an_envelope_rather_than_bare_stderr():
    code, envelope, stderr = run_worker("--cls", "Net")
    assert code == 2
    assert envelope["error"]["type"] == "UsageError"
    assert "--file" in stderr


def test_a_forward_that_raises_still_yields_the_graph_up_to_that_point(tmp_path: Path):
    """Exit 4 with a partial result — the half a shape mismatch produced is exactly
    what tells the user where it went wrong, and a worker that answered with nothing
    would leave the canvas blank at the moment it is most needed."""
    module = tmp_path / "broken.py"
    module.write_text(
        "import torch.nn as nn\n"
        "\n"
        "\n"
        "class Broken(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc1 = nn.Linear(8, 8)\n"
        "        self.fc2 = nn.Linear(4, 4)\n"
        "\n"
        "    def forward(self, x):\n"
        "        h = self.fc1(x)\n"
        "        return self.fc2(h)\n",
        encoding="utf-8",
    )
    code, envelope, _stderr = run_worker(
        "--file", str(module), "--cls", "Broken", "--input", "2,8", "--json"
    )
    assert code == 4
    assert envelope["error"]["node_id"] == "fc2"
    # The line the editor underlines. The message ("mat1 and mat2 shapes cannot be
    # multiplied") names no line at all, which is why the frames are carried.
    raising = [f for f in envelope["error"]["frames"] if f["file"] == str(module)]
    assert raising and raising[-1]["line"] == 12

    params = nodes_of(envelope)
    assert params["fc1"]["_shape_provenance"] == "observed"
    assert params["fc2"]["_shape_provenance"] == "unreached"


# ── the real tier (task.md D3) ────────────────────────────────────────────────

def test_the_real_tier_takes_a_branch_a_fake_tensor_cannot_evaluate():
    """`if x.sum() > 0` is the case torch.export refuses and the reason this repo
    traces eagerly. It is also the case that proves the two tiers stopped being the
    byte-identical branches they were before M4: fake fails on it, real does not.
    """
    envelope = trace_of("gate", "Gate", "--tier", "real", "--input", "torch.full((2, 8), -1.0)")
    assert envelope["result"]["diagnostics"]["tier"] == "real"
    assert envelope["error"] is None

    params = nodes_of(envelope)
    assert params["down"]["_shape_provenance"] == "observed"
    # The sum of a tensor of -1 is negative, so `up` never runs. It must still be a
    # node, and it must say it was not reached rather than carry an invented shape.
    assert params["up"]["_shape_provenance"] == "unreached"
    assert "_out_shape" not in params["up"]


def test_the_fake_tier_still_refuses_data_dependent_control_flow():
    code, envelope, _stderr = run_worker(
        "--file", str(MODELS / "gate.py"), "--cls", "Gate",
        "--tier", "fake", "--input", "2,8", "--json",
    )
    assert code == 4
    assert "DataDependent" in envelope["error"]["message"]


# ── source provenance (task.md §6.1) ──────────────────────────────────────────

def test_every_module_node_knows_the_file_and_line_of_its_class():
    envelope = trace_of("resid", "Net", "--input", "2,8")
    modules = {
        node_id: params
        for node_id, params in nodes_of(envelope).items()
        if params.get("_module_path") and node_id not in envelope["result"]["diagnostics"]["materialised_ops"]
    }
    assert modules
    for node_id, params in modules.items():
        assert params["_source_file"].endswith(".py"), node_id
        assert isinstance(params["_source_line"], int), node_id
        # And where it RAN is the fixture file the graph was opened from — which is
        # the half the editor needs, since an nn.Linear's class lives in torch.
        assert params["_decl_file"].endswith(os.path.join("models", "resid.py")), node_id
        assert isinstance(params["_decl_line"], int), node_id


def test_a_users_own_class_reports_the_users_own_file(tmp_path: Path):
    """The nn.Linear case above cannot show this: torch's classes legitimately live
    in torch. A block the user wrote in a second file of their own project must
    report THAT file, or "reveal this card in the editor" opens site-packages."""
    (tmp_path / "block.py").write_text(
        "import torch.nn as nn\n"
        "\n"
        "\n"
        "class Block(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(8, 8)\n"
        "\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n",
        encoding="utf-8",
    )
    net = tmp_path / "net.py"
    net.write_text(
        "import torch.nn as nn\n"
        "\n"
        "from block import Block\n"
        "\n"
        "\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.b = Block()\n"
        "\n"
        "    def forward(self, x):\n"
        "        return self.b(x)\n",
        encoding="utf-8",
    )
    code, envelope, stderr = run_worker(
        "--file", str(net), "--cls", "Net", "--input", "2,8", "--depth", "1", "--json"
    )
    assert code == 0, stderr[-2000:]
    params = nodes_of(envelope)["b"]
    assert params["_class"] == "Block"
    assert params["_source_file"] == str(tmp_path / "block.py")
    assert params["_source_line"] == 4


def test_the_residual_add_points_at_the_line_that_wrote_it():
    """A materialised node has no nn.Module behind it, so the captured Python frame
    is the ONLY thing that can point it at a line — and pointing a residual at the
    wrong line is worse than not drawing it, because the reader will believe it."""
    import importlib.util
    import inspect

    spec = importlib.util.spec_from_file_location("_m4_resid", MODELS / "resid.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    lines, first = inspect.getsourcelines(module.Net.forward)
    expected = first + next(
        index for index, line in enumerate(lines) if "return x + self.l2(h)" in line
    )

    envelope = trace_of("resid", "Net", "--input", "2,8")
    materialised = envelope["result"]["diagnostics"]["materialised_ops"]
    assert len(materialised) == 1, materialised
    params = nodes_of(envelope)[materialised[0]]
    assert params["_class"] == "Add"
    assert params["_source_line"] == expected
    assert params["_source_file"].endswith(os.path.join("models", "resid.py"))


def test_every_recorded_op_carried_a_frame_from_the_users_code():
    """`frames_captured == aten_ops` is the claim "no op is anonymous" stated as a
    number. A capture that quietly stopped finding frames would still produce a
    graph, just one where nothing can be revealed in the editor."""
    envelope = trace_of("resid", "Net", "--input", "2,8")
    diagnostics = envelope["result"]["diagnostics"]
    assert diagnostics["aten_ops"] > 0
    assert diagnostics["frames_captured"] == diagnostics["aten_ops"]


def test_the_file_can_name_its_own_example_input():
    """`# tensorviz: input=` in the user's own source (task.md M4).

    The candidate search knows images and token ids, so a net whose first layer is
    Linear(8, 8) is unguessable and would trace to a shapeless hierarchy. The
    comment is the per-file escape hatch, and it lives in the file precisely so it
    survives being opened by someone who did not write the flag.
    """
    code, envelope, stderr = run_worker(
        "--file", str(MODELS / "resid.py"), "--cls", "Net", "--json"
    )
    assert code == 0, stderr[-2000:]
    assert envelope["result"]["diagnostics"]["tier"] == "real"
    assert envelope["result"]["diagnostics"]["example_input"] == [[2, 8]]
    assert nodes_of(envelope)["l1"]["_out_shape"] == "2x8"


def test_frame_capture_can_be_turned_off_without_changing_the_graph():
    """The benchmark's escape hatch (scripts/bench_frame_capture.py). Turning it off
    must cost only the source links — a different graph would make the flag a
    correctness switch, which nobody could reason about."""
    with_frames = trace_of("resid", "Net", "--input", "2,8")
    without = trace_of("resid", "Net", "--input", "2,8", "--no-frames")
    assert without["result"]["diagnostics"]["frames_captured"] == 0
    assert without["spans"] == {} or all(
        span["kind"] == "class" for span in without["spans"].values()
    )
    assert sorted(nodes_of(without)) == sorted(nodes_of(with_frames))


# ── isolation (task.md §8.5) ──────────────────────────────────────────────────

def test_a_worker_run_leaves_the_parent_processs_sys_path_alone():
    """Two halves of the same promise.

    The subprocess is trivially isolated, and that is the point of running it as
    one: the repository importer prepends ``sys.path`` and never unwinds it. The
    in-process call is the half that can actually fail, and it is checked because
    the backend suite imports this module into a process it shares with 1600 other
    tests.
    """
    from services.ingest.service import ingest_source
    from services.ingest.spec import ModelRef

    before = list(sys.path)
    trace_of("resid", "Net", "--input", "2,8")
    assert sys.path == before

    ingest_source(
        ModelRef(file=str(MODELS / "resid.py"), cls="Net", depth=2), example_input="2,8"
    )
    assert sys.path == before


def test_the_worker_writes_nothing_to_disk():
    """task.md §5: the extension persists the worker's stdout; the worker itself
    owns no file. A worker that cached its own output would be writing into a
    workspace it was handed by a code lens."""
    before = {
        str(path): path.stat().st_mtime_ns
        for path in FIXTURE_REPO.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    trace_of("resid", "Net", "--input", "2,8")
    after = {
        str(path): path.stat().st_mtime_ns
        for path in FIXTURE_REPO.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    assert after == before
    assert not (FIXTURE_REPO / ".tensorviz").exists()


# ── the merge (task.md §5) ────────────────────────────────────────────────────

def test_the_merged_projection_reports_observed_shapes_and_says_it_is_traced():
    result = project("resid", "Net", trace_of("resid", "Net", "--input", "2,8"))
    assert result.traced is True
    shaped = {
        node.node_id: node.params.get("_out_shape")
        for node in result.document.semantic.nodes
        if node.params.get("_shape_provenance") == "observed"
    }
    assert shaped and all(shape for shape in shaped.values())
    assert not any(
        node.params.get("_shape_provenance") == "static"
        for node in result.document.semantic.nodes
    )


def test_the_merge_carries_where_the_traced_input_came_from_onto_the_input_card():
    """The shapes on every card are a measurement of ONE tensor, and the merge is
    the last hop before a user reads them.

    `--input 3,8` and the file's own `# tensorviz: input=2,8` arrive at the worker
    as the same kind of thing — a tuple of tensors — and used to leave it that way,
    so a shape search's own guess reached the canvas looking exactly like a shape
    the user had written down. `_input_provenance` rides the Input node's params
    across the merge because that bag is the only per-node channel that survives it
    (task.md §3.3), and because the card that draws the shape reads the node rather
    than the diagnostics.
    """
    declared = project("resid", "Net", trace_of("resid", "Net"))
    explicit = project("resid", "Net", trace_of("resid", "Net", "--input", "3,8"))

    def input_params(result):
        return next(n for n in result.document.semantic.nodes if n.node_id == "input").params

    assert input_params(declared)["_input_provenance"] == "declared"
    assert input_params(declared)["shape"] == "2,8"
    assert input_params(explicit)["_input_provenance"] == "explicit"
    assert input_params(explicit)["shape"] == "3,8"


def test_the_trace_replaces_the_opaque_loop_statement_with_what_actually_ran():
    """`for block in self.blocks: h = block(h)` is a `python` node to the parser —
    `block` is a loop variable, not a `self.<attr>`, so there is nothing to type. The
    trace saw two Linear invocations on that line. The executed nodes replace the
    placeholder rather than being drawn beside it."""
    static = project("resid", "Net", None)
    traced = project("resid", "Net", trace_of("resid", "Net", "--input", "2,8"))

    static_ids = {node.node_id for node in static.document.semantic.nodes}
    traced_ids = {node.node_id for node in traced.document.semantic.nodes}
    assert "python@Net#0" in static_ids
    assert "python@Net#0" not in traced_ids
    assert "blocks.0" in traced_ids
    # The declaration the parser saw is not lost either — it is on the canvas under
    # the more specific name the trace gave its instance.
    assert "blocks" not in traced_ids


def test_the_untaken_branch_survives_the_merge_dimmed():
    """The rule that keeps the merge honest: a trace must never delete a layer the
    user can see in their own file. `up` did not run, so it carries no shape — but
    removing it would answer "that layer does not exist" about a layer they wrote."""
    envelope = trace_of("gate", "Gate", "--tier", "real", "--input", "torch.full((2, 8), -1.0)")
    result = project("gate", "Gate", envelope)
    by_id = {node.node_id: node for node in result.document.semantic.nodes}
    assert by_id["up"].params["_shape_provenance"] == "unreached"
    assert by_id["down"].params["_shape_provenance"] == "observed"
    # And it keeps the wiring the parser read, not the "input -> up -> output"
    # placeholder the importer gives an unreached module.
    edges = {(e.source_node_id, e.target_node_id) for e in result.document.semantic.edges}
    assert ("up", "down") in edges
    assert ("up", "output") not in edges


def test_the_merge_keeps_the_declaration_spans_the_parser_computed():
    """Navigation is the parser's job even after a trace: an nn.Linear's class span
    points into torch, and clicking a card must not leave the user's project."""
    result = project("resid", "Net", trace_of("resid", "Net", "--input", "2,8"))
    assert result.spanmap["l1"]["kind"] == "decl"
    assert result.spanmap["l1"]["file"] == "models/resid.py"
    # blocks.0 is the trace's name for the parser's `blocks`; it inherits that
    # declaration's span rather than pointing at torch/nn/modules/linear.py.
    assert result.spanmap["blocks.0"]["file"] == "models/resid.py"
    assert result.spanmap["blocks.0"]["kind"] == "decl"


def test_the_merge_keeps_the_raw_call_text_only_the_parser_can_know():
    result = project("resid", "Net", trace_of("resid", "Net", "--input", "2,8"))
    by_id = {node.node_id: node for node in result.document.semantic.nodes}
    assert by_id["add@Net#0"].params["_raw_call"] == "x + self.l2(h)"
    assert by_id["l1"].params["_raw_call"] == "nn.Linear(width, width)"


def test_a_trace_from_another_build_is_refused_rather_than_half_understood():
    envelope = trace_of("resid", "Net", "--input", "2,8")
    envelope["protocol_version"] = 99
    with pytest.raises(projection.TraceRejected) as raised:
        project("resid", "Net", envelope)
    assert "99" in str(raised.value)


def test_an_empty_trace_is_refused_rather_than_reported_as_an_executed_graph():
    with pytest.raises(projection.TraceRejected):
        project("resid", "Net", {"protocol_version": 1, "result": {"graph": {"nodes": []}}})


def test_the_router_accepts_a_trace_result_and_answers_traced():
    """The wiring, end to end: `POST /api/ide/project {trace_result}`. Until M4 the
    same call answered 501."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.routers import ide
    from auth import CurrentUser, get_current_user

    app = FastAPI()
    app.include_router(ide.router)
    app.dependency_overrides[get_current_user] = lambda: CurrentUser(uid="local-dev-user")
    client = TestClient(app)

    body = {
        "file": "models/resid.py",
        "source": (MODELS / "resid.py").read_text(encoding="utf-8"),
        "cls": "Net",
        "trace_result": trace_of("resid", "Net", "--input", "2,8"),
    }
    response = client.post("/api/ide/project", json=body)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["traced"] is True
    assert payload["spanmap"]["l1"]["file"] == "models/resid.py"

    body["trace_result"] = {**body["trace_result"], "protocol_version": 7}
    refused = client.post("/api/ide/project", json=body)
    assert refused.status_code == 400
    assert refused.json()["detail"]["error"] == "trace_rejected"


def test_the_worker_writes_nothing_to_disk_beside_the_stdout_envelope(tmp_path):
    """"This process writes nothing to disk" is the worker module docstring's own
    promise, and it includes the bytecode cache Python drops beside the user's file
    at import: a trace must not leave a ``__pycache__`` in a workspace it was only
    handed by a command. Reproduces the packaged clean-profile smoke failure where
    a first trace added ``__pycache__/model.cpython-312.pyc`` to the fixture."""
    target = tmp_path / "model.py"
    target.write_text(
        "import torch.nn as nn\n"
        "\n"
        "\n"
        "class Tiny(nn.Module):\n"
        "    def __init__(self, width=8):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(width, width)\n"
        "\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n"
        "\n"
        "\n"
        "# tensorviz: input=2,8\n",
        encoding="utf-8",
    )
    code, envelope, stderr = run_worker("--file", str(target), "--cls", "Tiny", "--json")
    assert code == 0, f"exit {code}: {envelope.get('error')} / {stderr[-2000:]}"
    assert [entry.name for entry in tmp_path.iterdir()] == ["model.py"]
