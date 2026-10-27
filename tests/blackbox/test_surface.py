"""The surface a client meets before it ever draws a graph.

Everything here is what the extension calls on window open, on save, and when a
user mistypes: `/health` to decide whether this backend can do the job at all,
`/workspace/classes` to put a CodeLens over each model, `/runs` to list what has
been traced, and the input ceilings that stand between a malformed client and a
backend that falls over.

The recurring assertion is a boundary one: a bad request must come back as a 4xx
that names what was wrong. A 500 is not a smaller failure — the extension shows
"tensorViz backend error" and the user has no idea their class name was a typo.
"""
from __future__ import annotations

import json
import os
import re

import pytest
import requests


SOURCE = (
    "import torch.nn as nn\n"
    "\n"
    "\n"
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.l1 = nn.Linear(8, 8)\n"
    "\n"
    "    def forward(self, x):\n"
    "        return self.l1(x)\n"
)


@pytest.fixture(scope="module")
def applyable(api, projected):
    """`(ops -> response)` bound to one projection of SOURCE.

    Every ceiling and every malformed-op test needs a request that is valid
    except for the one thing it is testing, and the spanmap and digest have to be
    the product's own or the request fails at the 409 before reaching the part
    under test.
    """
    file = "surface_applyable.py"
    _document, spanmap, digest = projected(SOURCE, "Net", file)

    def _apply(ops):
        return api.apply(SOURCE, "Net", ops, spanmap, digest, file)

    return _apply


def _workspace(tmp_path):
    """A folder shaped like a user's repo: a package of models, a script, no torch run."""
    (tmp_path / "models").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "models" / "__init__.py").write_text("")
    (tmp_path / "models" / "blocks.py").write_text(
        "import torch.nn as nn\n"
        "\n"
        "\n"
        "class Block(nn.Module):\n"
        "    def __init__(self, width):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(width, width)\n"
        "\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n"
    )
    (tmp_path / "models" / "deep.py").write_text(
        "from .blocks import Block\n"
        "\n"
        "\n"
        "class Deep(Block):\n"
        "    def forward(self, x):\n"
        "        return super().forward(x)\n"
    )
    (tmp_path / "scripts" / "train.py").write_text("def main():\n    return 1\n")
    return str(tmp_path)


# ── /health ───────────────────────────────────────────────────────────────────


def test_health_says_it_is_healthy_and_names_which_backend_answered(api):
    """The extension refuses to talk to a stranger on a port it did not start.

    `status` gates the connection and `backend` is how it tells this process from
    whatever else is listening on 8000 — both are contract, not decoration.
    """
    body = api.health().json()
    assert body["status"] == "healthy"
    assert body["backend"] == "canvas-backend-v2"


def test_health_names_the_torch_and_python_a_trace_would_run_in(api):
    """A trace runs in *this* interpreter's torch, and shapes differ between them.

    When an observed shape disagrees with the user's expectation, this is the
    line in the bug report that settles it, so it has to be a real version and
    not null.
    """
    environment = api.health().json()["environment"]
    assert re.fullmatch(r"\d+\.\d+\.\d+.*", environment["torch"] or ""), environment
    assert re.fullmatch(r"3\.\d+\.\d+", environment["python"] or ""), environment
    assert environment["device"] in ("cpu", "cuda", "mps")


def test_health_declares_its_capabilities_so_a_client_need_not_probe_with_404s(api):
    """A client decides whether to enforce a token or fetch a UI from this origin.

    Inferring either from a 404 means a user sees a failure that was never a
    failure, so both keys must be present and boolean — not merely truthy.
    """
    capabilities = api.health().json()["capabilities"]
    assert set(capabilities) == {"ide", "static"}
    assert isinstance(capabilities["ide"], bool)
    assert isinstance(capabilities["static"], bool)


def test_health_lists_the_features_this_build_advertises(api):
    body = api.health().json()
    assert body["features"] == [
        "hierarchical_models",
        "template_system",
        "code_generation",
    ]


# ── /api/ide/workspace/classes ────────────────────────────────────────────────


def test_workspace_classes_finds_every_model_without_importing_the_workspace(
    base_url, tmp_path
):
    """One CodeLens per model, found by AST — importing a repo to list it would run it.

    `Deep` is the test that matters: it inherits from another file's block, so a
    scanner that only matched the literal text `nn.Module` would miss it and the
    user's own subclass would silently have no CodeLens.
    """
    root = _workspace(tmp_path)
    response = requests.post(
        f"{base_url}/api/ide/workspace/classes", json={"root": root}, timeout=60
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert [entry["name"] for entry in body["classes"]] == ["Block", "Deep"]

    block = body["classes"][0]
    assert block["file"] == "models/blocks.py"
    assert block["line"] == 4
    assert block["bases"] == ["nn.Module"]
    assert block["ctor_required"] == 1
    # The palette writes this import verbatim when it adds a workspace block to
    # another file; a name alone cannot produce an import statement.
    assert block["import_path"] == "models.blocks"


def test_workspace_classes_answers_empty_for_a_folder_that_holds_no_models(
    base_url, tmp_path
):
    """A scripts folder is not an error, it just has nothing to put a CodeLens on.

    `file_count` still counts what was read, so a client can tell "scanned, found
    none" from "scanned nothing".
    """
    root = _workspace(tmp_path)
    response = requests.post(
        f"{base_url}/api/ide/workspace/classes",
        json={"root": os.path.join(root, "scripts")},
        timeout=60,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["classes"] == []
    assert body["file_count"] == 1


def test_workspace_classes_says_404_when_the_root_is_not_a_directory(base_url, tmp_path):
    """A user points at a file, or at a folder that moved. That is theirs to fix, and
    a 500 would tell them the backend broke instead."""
    missing = str(tmp_path / "not-here")
    response = requests.post(
        f"{base_url}/api/ide/workspace/classes",
        json={"root": missing},
        timeout=60,
    )
    assert response.status_code == 404
    assert response.json()["detail"] == f"{missing} is not a directory"


def test_workspace_classes_rejects_a_request_with_no_root(base_url):
    response = requests.post(
        f"{base_url}/api/ide/workspace/classes", json={}, timeout=60
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "root"]


# ── /api/ide/runs ─────────────────────────────────────────────────────────────


def test_runs_is_an_empty_list_for_a_workspace_nobody_has_traced(base_url, tmp_path):
    """The editor asks this on every window open, long before any run exists.

    Answering 404 there would put an error banner in front of every new user.
    """
    response = requests.get(
        f"{base_url}/api/ide/runs", params={"root": str(tmp_path)}, timeout=60
    )
    assert response.status_code == 200, response.text
    assert response.json()["records"] == []
    assert response.json()["unreadable"] == 0


def test_runs_lists_records_newest_first_and_counts_the_ones_it_could_not_read(
    base_url, tmp_path
):
    """The last thing the user ran is the first thing they need to see.

    And a line written by an older build is COUNTED, not dropped: a list that is
    quietly shorter than the directory is how a user loses a run they remember
    making.
    """
    runs = tmp_path / ".tensorviz" / "runs"
    runs.mkdir(parents=True)
    record = {
        "kind": "model",
        "target": "models/net.py::Net",
        "graph": {"nodes": [], "edges": []},
        "spans": {},
        "diagnostics": {"tier": "real"},
        "run_id": "older",
        "started_at": "2026-01-01T00:00:00Z",
    }
    newer = {**record, "run_id": "newer", "started_at": "2026-02-02T00:00:00Z"}
    runs.joinpath("a.jsonl").write_text(
        json.dumps(record)
        + "\n"
        + json.dumps(newer)
        + "\n"
        + json.dumps({"kind": "model"})  # an older build's shorter record
        + "\n"
        + "this is not json\n"
    )

    response = requests.get(
        f"{base_url}/api/ide/runs", params={"root": str(tmp_path)}, timeout=60
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert [entry["run_id"] for entry in body["records"]] == ["newer", "older"]
    assert body["unreadable"] == 2


def test_runs_reads_only_the_tensorviz_runs_directory(base_url, tmp_path):
    """This process reads `runs/` and nothing else in the workspace.

    A checkpoint or a dataset sitting beside it must not become a "run", because
    the direction of that arrow is the whole safety story of the IDE plane.
    """
    runs = tmp_path / ".tensorviz" / "runs"
    runs.mkdir(parents=True)
    runs.joinpath("notes.txt").write_text("not a run record\n")
    tmp_path.joinpath("train.log").write_text("also not a run record\n")

    body = requests.get(
        f"{base_url}/api/ide/runs", params={"root": str(tmp_path)}, timeout=60
    ).json()
    assert body["records"] == []
    assert body["unreadable"] == 0


def test_runs_rejects_a_request_with_no_root(base_url):
    response = requests.get(f"{base_url}/api/ide/runs", timeout=60)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["query", "root"]


# ── input ceilings: a malformed client must not be able to break the backend ──


def test_a_source_larger_than_the_ceiling_is_rejected_before_it_is_parsed(api):
    """One megabyte is the ceiling. Past it the answer is a 422 naming the field,
    not an unbounded allocation inside an AST walk."""
    response = api.parse("#" * 1_000_001, None, "surface_oversized.py")
    assert response.status_code == 422
    detail = response.json()["detail"][0]
    assert detail["loc"] == ["body", "source"]
    assert detail["type"] == "string_too_long"


@pytest.mark.parametrize(
    "body, missing",
    [
        ({"file": "surface_missing.py"}, ["body", "source"]),
        ({"source": SOURCE}, ["body", "file"]),
    ],
)
def test_parse_names_the_field_a_request_left_out(base_url, body, missing):
    """"Something went wrong" is unactionable; the field name is the whole fix."""
    response = requests.post(f"{base_url}/api/ide/parse", json=body, timeout=60)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == missing


def test_apply_names_the_field_a_request_left_out(base_url):
    response = requests.post(
        f"{base_url}/api/ide/apply",
        json={
            "file": "surface_missing_digest.py",
            "cls": "Net",
            "source": SOURCE,
            "spanmap": {},
            "ops": [{"op": "remove_layer", "op_id": "o1", "node_id": "l1"}],
        },
        timeout=60,
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "source_digest"]


def test_a_class_that_is_not_in_the_file_is_a_4xx_naming_it(api):
    """A user types a class name into a CodeLens command, or renames the class and
    the editor asks for the old one. Both are the request's fault, and the message
    has to say which name was not found."""
    response = api.parse(SOURCE, "Nope", "surface_absent_class.py")
    assert 400 <= response.status_code < 500, response.text
    assert response.status_code == 404
    assert response.json()["detail"] == "Nope is not in surface_absent_class.py"


def test_more_ops_than_the_batch_ceiling_are_rejected_whole(applyable):
    """201 operations is a client bug, not a user gesture. Rejecting the batch
    beats applying 200 edits and leaving the file half-changed."""
    ops = [
        {"op": "set_param", "op_id": f"o{i}", "node_id": "l1", "param": "out_features", "value": 9}
        for i in range(201)
    ]
    response = applyable(ops)
    assert response.status_code == 422
    detail = response.json()["detail"][0]
    assert detail["loc"] == ["body", "ops"]
    assert detail["type"] == "too_long"


def test_an_empty_op_batch_is_rejected_rather_than_answered_with_no_edits(applyable):
    """An empty batch means the client lost the user's gesture on the way here.

    Answering 200 with no edits would let that bug look like a successful edit
    that simply changed nothing.
    """
    response = applyable([])
    assert response.status_code == 422
    detail = response.json()["detail"][0]
    assert detail["loc"] == ["body", "ops"]
    assert detail["type"] == "too_short"


def test_malformed_json_is_a_422_not_a_500(base_url):
    """A truncated body is what a dropped connection looks like from here."""
    response = requests.post(
        f"{base_url}/api/ide/parse",
        data="{not json",
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "json_invalid"


# ── unknown operations: refused, never crashed ────────────────────────────────


def test_an_unknown_op_kind_is_refused_with_a_reason(applyable):
    """A newer canvas can send an op this backend has never heard of.

    That has to come back as a refusal the canvas can show on the node, because a
    500 makes a forward-compatible client look broken.
    """
    response = applyable([{"op": "teleport_layer", "op_id": "o1", "node_id": "l1"}])
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["edits"] == []
    assert len(body["refused"]) == 1
    refusal = body["refused"][0]
    assert refusal["op_id"] == "o1"
    assert refusal["reason"] == "unsupported_operation"
    assert "teleport_layer" in refusal["message"]


def test_an_op_with_no_kind_at_all_is_refused_with_a_reason(applyable):
    response = applyable([{"op_id": "o1", "node_id": "l1"}])
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["edits"] == []
    assert body["refused"][0]["reason"] == "unsupported_operation"


@pytest.mark.parametrize("op", ["hello", None, 7])
def test_an_op_that_is_not_an_object_is_rejected_by_schema(base_url, projected, op):
    """A client that sends a bare string where an operation belongs gets told so."""
    _document, spanmap, digest = projected(SOURCE, "Net", "surface_scalar_op.py")
    response = requests.post(
        f"{base_url}/api/ide/apply",
        json={
            "file": "surface_scalar_op.py",
            "cls": "Net",
            "source": SOURCE,
            "source_digest": digest,
            "spanmap": spanmap,
            "ops": [op],
        },
        timeout=60,
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "ops", "0"]


def test_apply_writes_nothing_to_the_file_it_was_told_about(api, projected, tmp_path):
    """The extension owns every write, so the change lands in the user's undo stack.

    If this process ever started writing the path it is handed, an edit the user
    had not confirmed would already be on their disk.
    """
    path = tmp_path / "surface_untouched.py"
    path.write_text(SOURCE)
    before = path.read_text()

    _document, spanmap, digest = projected(SOURCE, "Net", str(path))
    response = api.apply(
        SOURCE,
        "Net",
        [{"op": "set_param", "op_id": "o1", "node_id": "l1", "param": "out_features", "value": 32}],
        spanmap,
        digest,
        str(path),
    )
    assert response.status_code == 200, response.text
    assert [edit["newText"] for edit in response.json()["edits"]] == [
        "        self.l1 = nn.Linear(8, 32)\n"
    ], "nothing was proposed, so this proves nothing"
    assert path.read_text() == before


# ── findings: 4xx belongs here and a 500 arrives instead ──────────────────────


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(
            {"op": "set_param", "op_id": "o1", "node_id": "l1", "value": 9},
            id="set_param-without-param",
        ),
        pytest.param(
            {"op": "set_param", "op_id": "o1", "param": "out_features", "value": 9},
            id="set_param-without-node_id",
        ),
        pytest.param(
            {"op": "remove_layer", "op_id": "o1"},
            id="remove_layer-without-node_id",
        ),
        pytest.param(
            {"op": "rename_layer", "op_id": "o1", "node_id": "l1"},
            id="rename_layer-without-new_node_id",
        ),
        pytest.param(
            {"op": "replace_layer", "op_id": "o1", "node_id": "l1"},
            id="replace_layer-without-a-replacement",
        ),
        pytest.param(
            {"op": "add_layer", "op_id": "o1"},
            id="add_layer-without-a-layer",
        ),
    ],
)
def test_an_op_missing_a_required_field_is_refused_not_a_500(applyable, op):
    """A field the client forgot is the client's bug, and `refused` is where the
    canvas already looks for it.

    A 500 instead means the user sees "tensorViz backend error" with no node
    named and no reason, and the batch's other operations are lost with it —
    which is exactly the failure `refused` exists to prevent. `connect` with no
    endpoints already does the right thing, so this is inconsistency, not policy.
    """
    response = applyable([op])
    assert response.status_code == 200, response.text
    body = response.json()
    assert [item["op_id"] for item in body["refused"]] == ["o1"]
    assert body["edits"] == []


def test_set_param_with_no_value_refuses_rather_than_writing_none(applyable):
    """A dropped `value` must never become a literal `None` in the model.

    Writing it would leave `nn.Linear(8, None)` — source that still parses, so
    nothing catches it until the layer is constructed and the user is reading a
    traceback about their own file. Silently writing a wrong value is worse than
    the 500 the sibling ops give: this one reaches disk.
    """
    response = applyable(
        [{"op": "set_param", "op_id": "o1", "node_id": "l1", "param": "out_features"}]
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert [item["op_id"] for item in body["refused"]] == ["o1"]
    assert body["edits"] == [], "no value means no edit"


def test_ctor_required_accounts_for_an_inherited_constructor(base_url, tmp_path):
    """`ctor_required == 0` is the promise that "Open as Graph" will just work.

    `Deep` defines no `__init__` and inherits one that demands `width`, so the
    scan answers 0, the CodeLens offers the one-click path, and the trace worker
    dies constructing it — after the user has already agreed to run their code.
    """
    root = _workspace(tmp_path)
    body = requests.post(
        f"{base_url}/api/ide/workspace/classes", json={"root": root}, timeout=60
    ).json()
    deep = next(entry for entry in body["classes"] if entry["name"] == "Deep")
    assert deep["ctor_required"] == 1
