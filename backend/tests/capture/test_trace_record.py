"""The traced-program IR, the two new worker kinds and the run list (task.md M9).

What M9 claims, and therefore what has to be true here:

* a **record** is a first-class thing, not a model graph with extra keys: it says
  which kind of program ran, when, and which invocation it was, and it survives a
  round trip through JSON because that is how it reaches disk and comes back;
* ``--kind data`` describes a pipeline the only way a pipeline can be described —
  by the LINES of the user's own ``__getitem__``, with the collate as the merge
  that turns N samples into a batch;
* ``--kind train_step`` runs the user's own training step, so the loss is theirs
  and the gradients are a real optimiser step's, while the GRAPH stays the model's:
  a backward is measured, not drawn;
* ``GET /api/ide/runs`` reads what the extension wrote and nothing else. The
  backend never writes a run record, and a line it cannot read is counted rather
  than dropped.

The worker is exercised as a **subprocess**, which is how it runs; no module-scope
``import torch`` (task.md §8.3) and every fixture tensor is eight units wide.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


from api.routers import ide  # noqa: E402
from auth import CurrentUser, get_current_user  # noqa: E402

BACKEND = Path(__file__).parent.resolve().parent.parent
FIXTURE_REPO = BACKEND / "tests" / "fixtures" / "ide_repo"
MODELS = FIXTURE_REPO / "models"
SCRIPTS = FIXTURE_REPO / "scripts"

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


def params_of(envelope: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(node["id"]): ((node.get("data") or {}).get("parameters") or {})
        for node in envelope["result"]["graph"]["nodes"]
    }


@pytest.fixture(scope="module")
def data_run() -> Dict[str, Any]:
    code, envelope, stderr = run_worker(
        "--kind", "data",
        "--file", str(SCRIPTS / "data.py"),
        "--callable", "make_loader",
        "--json",
    )
    assert code == 0, f"exit {code}: {envelope.get('error')} / {stderr[-2000:]}"
    return envelope


@pytest.fixture(scope="module")
def train_step_run() -> Dict[str, Any]:
    code, envelope, stderr = run_worker(
        "--kind", "train_step",
        "--file", str(MODELS / "resid.py"),
        "--cls", "Net",
        "--callable", f"{SCRIPTS / 'train_step.py'}:train_step",
        "--json",
    )
    assert code == 0, f"exit {code}: {envelope.get('error')} / {stderr[-2000:]}"
    return envelope


# ── the IR (task.md M9, spec.TraceRecord) ─────────────────────────────────────

def test_a_record_survives_the_round_trip_it_is_written_and_read_through(
    data_run: Dict[str, Any],
) -> None:
    """A record's whole life is `model_dump` -> a line of JSON -> `model_validate`.

    The extension serialises one into `.tensorviz/runs/<id>.jsonl` and this process
    reads it back days later. A field that does not survive that trip is a field that
    silently becomes its default on the way home — which is how a `train_step` run
    would come back looking like a `model` one.
    """
    from services.ingest.spec import IngestResult, TraceRecord

    result = IngestResult.model_validate(data_run["result"])
    record = TraceRecord.of(
        kind="data",
        target=f"{SCRIPTS / 'data.py'}::make_loader",
        result=result,
        spans=data_run["spans"],
        run_id="run-0001",
        started_at="2026-09-05T12:00:00Z",
    )

    again = TraceRecord.model_validate(json.loads(json.dumps(record.model_dump(mode="json"))))

    assert again == record
    assert again.kind == "data"
    assert again.run_id == "run-0001"
    assert again.started_at == "2026-09-05T12:00:00Z"
    assert again.protocol_version == 1
    # Wrapping, not copying: the result's graph and diagnostics are the record's.
    assert again.graph == result.graph
    assert again.diagnostics.tier == result.diagnostics.tier
    assert again.spans == data_run["spans"]


def test_a_kind_no_view_renders_is_refused_rather_than_stored(data_run: Dict[str, Any]) -> None:
    """`eval` is task.md §10 material until something draws one.

    A `Literal` that accepts a kind nothing renders is a promise: the record would
    be written, listed, clicked, and open nothing.
    """
    from pydantic import ValidationError

    from services.ingest.spec import IngestResult, TraceRecord

    with pytest.raises(ValidationError):
        TraceRecord.of(
            kind="eval",  # type: ignore[arg-type]
            target="x::y",
            result=IngestResult.model_validate(data_run["result"]),
            run_id="r",
            started_at="2026-09-05T12:00:00Z",
        )


# ── --kind data (task.md M9) ──────────────────────────────────────────────────

def test_the_data_pipeline_is_the_lines_of_the_users_own_getitem(
    data_run: Dict[str, Any],
) -> None:
    """Three operation nodes at least, each pointing at a line of data.py.

    A data pipeline has no `nn.Module` hierarchy, so the only provenance its tensor
    ops have is the line they ran on — and a node with no line is a card the reader
    cannot get from the canvas back to their code.
    """
    data_py = str(SCRIPTS / "data.py")
    source_lines = len((SCRIPTS / "data.py").read_text(encoding="utf-8").splitlines())

    ops = {
        node_id: params
        for node_id, params in params_of(data_run).items()
        if params.get("_source_line")
    }
    assert len(ops) >= 3, f"only {len(ops)} operation nodes: {sorted(ops)}"
    for node_id, params in ops.items():
        assert params["_source_file"] == data_py, node_id
        assert 1 <= params["_source_line"] <= source_lines, node_id
        # The spanmap is what "reveal in editor" reads; a node with a line and no
        # span is a card that points nowhere.
        assert data_run["spans"][node_id]["file"] == data_py


def test_the_getitem_ops_hang_off_a_frame_that_computes_nothing(
    data_run: Dict[str, Any],
) -> None:
    """The function is a display frame, not a step of the pipeline.

    An edge-free `isGroup` node is grouping; giving it edges would say the function
    computed something its lines did not.
    """
    graph = data_run["result"]["graph"]
    frames = [n for n in graph["nodes"] if (n.get("data") or {}).get("isGroup")]
    assert len(frames) == 1, [n["id"] for n in frames]
    frame_id = frames[0]["id"]
    assert "__getitem__" in frame_id

    children = [
        n["id"] for n in graph["nodes"] if (n.get("data") or {}).get("parentId") == frame_id
    ]
    assert len(children) >= 3
    touching = [e for e in graph["edges"] if frame_id in (e["source"], e["target"])]
    assert touching == [], f"the display frame carries edges: {touching}"


def test_collation_is_a_merge_and_not_another_line(data_run: Dict[str, Any]) -> None:
    """Where N samples become one batch is the pipeline's most important node.

    It cannot be found by counting producers the way a residual is: sample 2 runs the
    same lines as sample 1, so one line is one producer however many samples pass
    through it. It is found by being a `stack`/`cat` that ran OUTSIDE the user's code
    on tensors the user's lines produced.
    """
    graph = data_run["result"]["graph"]
    merges = data_run["result"]["diagnostics"]["materialised_ops"]
    assert len(merges) == 1, merges
    collate = merges[0]

    inbound = [e["source"] for e in graph["edges"] if e["target"] == collate]
    assert inbound, "the collate consumes nothing"
    # It is fed by a line of the user's file, and it is the last thing before the exit.
    assert all(params_of(data_run)[src].get("_source_line") for src in inbound)
    assert any(
        e["source"] == collate and e["target"] == "output_node" for e in graph["edges"]
    )


def test_data_refuses_a_class_instead_of_quietly_tracing_the_loader(tmp_path: Path) -> None:
    """A flag that would be ignored is a flag that reads to the user as an answer."""
    code, envelope, stderr = run_worker(
        "--kind", "data",
        "--file", str(SCRIPTS / "data.py"),
        "--callable", "make_loader",
        "--cls", "Net",
        "--json",
    )
    assert code == 2
    assert envelope["error"]["type"] == "UsageError"
    assert "--cls" in envelope["error"]["message"]
    assert envelope["result"] is None


# ── --kind train_step (task.md M9) ────────────────────────────────────────────

def test_a_training_step_reports_the_loss_its_own_function_returned(
    train_step_run: Dict[str, Any],
) -> None:
    """A float, and the user's — not a synthesised `out.sum()`.

    `train_step` computes `mse_loss(model(batch), batch)`; anything we invented in
    its place would be a number about a different objective, printed beside their code.
    """
    loss = train_step_run["result"]["diagnostics"]["loss"]
    assert isinstance(loss, float), repr(loss)
    assert loss == pytest.approx(loss)  # a real number, not a nan
    assert loss > 0.0


def test_every_parameterised_node_carries_the_steps_gradients(
    train_step_run: Dict[str, Any],
) -> None:
    """`_grad_stats` on every node with weights the step actually ran.

    They come from the backward the user's own function performed, which is the
    difference M9 is about: M8 had to be handed a loss expression to differentiate,
    a training step already has one.
    """
    parameterised = {
        node_id: params
        for node_id, params in params_of(train_step_run).items()
        if (params.get("_params_M") or 0) > 0
    }
    assert len(parameterised) >= 3, sorted(parameterised)
    for node_id, params in parameterised.items():
        assert params.get("_shape_provenance") == "observed", node_id
        stats = params.get("_grad_stats")
        assert stats, f"{node_id} has weights and no gradients"
        assert set(stats) >= {"mean", "std", "absmax", "nan", "inf"}, node_id
        assert stats["nan"] == 0 and stats["inf"] == 0, node_id


def test_the_backward_is_measured_and_not_drawn(train_step_run: Dict[str, Any]) -> None:
    """The graph of a training step is the graph of the model it trains.

    Gradient accumulation is a chain of `add`s with two known producers each — the
    exact shape the materialiser turns a residual into. Recorded, the fixture net
    grew eight `add` cards that exist in no version of that model.
    """
    code, forward_only, stderr = run_worker(
        "--file", str(MODELS / "resid.py"), "--cls", "Net", "--json"
    )
    assert code == 0, stderr[-2000:]

    def shape(envelope: Dict[str, Any]) -> Tuple[List[str], List[Tuple[str, str]]]:
        graph = envelope["result"]["graph"]
        return (
            sorted(str(n["id"]) for n in graph["nodes"]),
            sorted((e["source"], e["target"]) for e in graph["edges"]),
        )

    assert shape(train_step_run) == shape(forward_only)
    assert (
        train_step_run["result"]["diagnostics"]["aten_ops"]
        == forward_only["result"]["diagnostics"]["aten_ops"]
    )


def test_a_training_step_without_a_batch_is_refused_before_it_runs(tmp_path: Path) -> None:
    """No batch means the input search would run — and each attempt is a real step.

    `_introspect_and_trace` retries candidate input shapes when it has no recipe.
    For a forward that is free; for a training step every attempt takes an optimiser
    step, so the second attempt would measure weights the first one already moved.
    Every fixture in `models/` carries a `# tensorviz: input=` comment, so the model
    without one has to be written here.
    """
    model = tmp_path / "unguessable.py"
    model.write_text(
        "import torch.nn as nn\n"
        "\n"
        "\n"
        "class Tiny(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.l = nn.Linear(8, 8)\n"
        "\n"
        "    def forward(self, x):\n"
        "        return self.l(x)\n",
        encoding="utf-8",
    )

    code, envelope, _stderr = run_worker(
        "--kind", "train_step",
        "--file", str(model),
        "--cls", "Tiny",
        "--callable", f"{SCRIPTS / 'train_step.py'}:train_step",
        "--json",
    )

    assert code == 2
    assert envelope["result"] is None
    assert "--input" in envelope["error"]["message"]


def test_a_missing_callable_is_named_rather_than_guessed_at() -> None:
    code, envelope, _stderr = run_worker(
        "--kind", "data", "--file", str(SCRIPTS / "data.py"), "--json"
    )
    assert code == 2
    assert "--callable" in envelope["error"]["message"]


def test_the_new_kinds_print_the_same_five_keys_the_protocol_fixes(
    data_run: Dict[str, Any], train_step_run: Dict[str, Any]
) -> None:
    """A second kind of trace must not be a second stdout contract (task.md §6.3).

    The extension reads every envelope with one `JSON.parse` and no recovery path,
    and the record it writes to `.tensorviz/runs/` is composed on its side from
    these five keys plus the kind it asked for.
    """
    for envelope in (data_run, train_step_run):
        assert set(envelope) == {"protocol_version", "result", "spans", "error", "worker"}
        assert envelope["protocol_version"] == 1
        assert envelope["error"] is None
        assert envelope["worker"]["torch"]


# ── GET /api/ide/runs (task.md M9, §6.5) ──────────────────────────────────────

@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(ide.router)
    app.dependency_overrides[get_current_user] = lambda: CurrentUser(uid="local-dev-user")
    return TestClient(app)


def _write_run(workspace: Path, file_stem: str, records: List[Dict[str, Any]]) -> Path:
    """Do what the extension does: append one JSON line per finished worker run."""
    runs = workspace / ".tensorviz" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / f"{file_stem}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def _record(**overrides: Any) -> Dict[str, Any]:
    record = {
        "kind": "train_step",
        "target": "models/resid.py::Net",
        "graph": {"nodes": [], "edges": []},
        "spans": {},
        "diagnostics": {"tier": "real"},
        "run_id": "run-0001",
        "started_at": "2026-09-05T12:00:00Z",
        "protocol_version": 1,
    }
    record.update(overrides)
    return record


def test_one_persisted_run_is_one_listed_record(client: TestClient, tmp_path: Path) -> None:
    """The acceptance sentence, as a test: one `.jsonl`, one record.

    The arrow only points one way (task.md §6.5). The extension appends the line;
    this endpoint reads it. Nothing here creates `.tensorviz/`.
    """
    _write_run(tmp_path, "session-a", [_record()])

    response = client.get("/api/ide/runs", params={"root": str(tmp_path)})

    assert response.status_code == 200, response.text
    body = response.json()
    assert len(list((tmp_path / ".tensorviz" / "runs").glob("*.jsonl"))) == 1
    assert len(body["records"]) == 1
    assert body["records"][0]["kind"] == "train_step"
    assert body["records"][0]["run_id"] == "run-0001"
    assert body["unreadable"] == 0


def test_a_workspace_that_has_never_been_traced_lists_nothing_and_is_not_an_error(
    client: TestClient, tmp_path: Path
) -> None:
    """The editor asks this on every window open, including the first one."""
    response = client.get("/api/ide/runs", params={"root": str(tmp_path)})

    assert response.status_code == 200
    assert response.json()["records"] == []
    assert not (tmp_path / ".tensorviz").exists(), "reading a run list created a directory"


def test_records_come_back_newest_first_across_files(
    client: TestClient, tmp_path: Path
) -> None:
    """Two windows, two session files, one list — in the order a Runs view shows."""
    _write_run(tmp_path, "session-a", [_record(run_id="a", started_at="2026-09-05T09:00:00Z")])
    _write_run(
        tmp_path,
        "session-b",
        [
            _record(run_id="b", kind="data", started_at="2026-09-05T11:00:00Z"),
            _record(run_id="c", kind="model", started_at="2026-09-05T10:00:00Z"),
        ],
    )

    body = client.get("/api/ide/runs", params={"root": str(tmp_path)}).json()

    assert [r["run_id"] for r in body["records"]] == ["b", "c", "a"]
    assert [r["kind"] for r in body["records"]] == ["data", "model", "train_step"]


def test_a_line_this_build_cannot_read_is_counted_and_not_dropped(
    client: TestClient, tmp_path: Path
) -> None:
    """A run from another build is the thing the user needs told about.

    Skipping it silently makes the list shorter than the directory, which reads as
    "that run never happened" rather than "this build cannot read it".
    """
    _write_run(tmp_path, "session-a", [_record()])
    with (tmp_path / ".tensorviz" / "runs" / "session-a.jsonl").open("a", encoding="utf-8") as h:
        h.write('{"kind": "eval", "target": "x", "run_id": "z", "started_at": "later"}\n')
        h.write("not json at all\n")

    body = client.get("/api/ide/runs", params={"root": str(tmp_path)}).json()

    assert len(body["records"]) == 1
    assert body["unreadable"] == 2
