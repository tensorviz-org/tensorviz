"""`POST /api/ide/project/index` — admission, the body handshake, and scale.

The point of these tests is the thing the multi-file endpoint cannot do: index a
repository larger than 256 files without a 422, and do it on a keystroke without
re-uploading the tree.
"""

import hashlib
import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


from api.routers import ide  # noqa: E402
from auth import CurrentUser, get_current_user  # noqa: E402


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture
def client():
    ide._source_cache.clear()
    app = FastAPI()
    app.include_router(ide.router)
    app.dependency_overrides[get_current_user] = lambda: CurrentUser(uid="local-dev-user")
    with TestClient(app) as test_client:
        yield test_client
    ide._source_cache.clear()


def index_call(client, sources, *, bodies=None, truncated=False, root="/w"):
    """Manifest for every file; bodies only for those named in `bodies`."""
    send = sources if bodies is None else {k: sources[k] for k in bodies}
    return client.post(
        "/api/ide/project/index",
        json={
            "protocol_version": 1,
            "root": root,
            "manifest": [{"file": f, "digest": digest(s)} for f, s in sources.items()],
            "bodies": [{"file": f, "source": s} for f, s in send.items()],
            "truncated": truncated,
        },
    )


TINY = {
    "models/net.py": "import torch.nn as nn\n\n\nclass Net(nn.Module):\n    pass\n",
    "train.py": "from models.net import Net\n\nif __name__ == '__main__':\n    Net()\n",
}


def test_indexes_a_small_project(client):
    response = index_call(client, TINY)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["indexed"] == 2
    assert [e["entrypoint"] for e in body["index"]["experiments"]] == ["train.py"]


def test_a_repo_over_the_multifile_cap_indexes_without_a_422(client):
    """The reason this endpoint exists. `MAX_SOURCES = 256` would refuse this."""
    from services.ide.multifile import MAX_SOURCES

    sources = {f"pkg/mod_{i:04d}.py": f"VALUE_{i} = {i}\n" for i in range(MAX_SOURCES + 200)}
    sources["train.py"] = "if __name__ == '__main__':\n    pass\n"

    response = index_call(client, sources)
    assert response.status_code == 200, response.text
    assert response.json()["indexed"] == len(sources)


def test_second_index_of_an_unchanged_workspace_sends_no_bodies(client):
    """The handshake that makes a per-keystroke index affordable."""
    assert index_call(client, TINY).status_code == 200

    response = index_call(client, TINY, bodies=[])
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok", "an unchanged workspace must resolve from cache alone"
    assert body["indexed"] == 2


def test_a_manifest_naming_unknown_digests_asks_for_them(client):
    response = index_call(client, TINY, bodies=[])
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "need_bodies"
    assert sorted(body["missing"]) == ["models/net.py", "train.py"]


def test_only_the_changed_file_needs_a_body(client):
    assert index_call(client, TINY).status_code == 200

    edited = dict(TINY)
    edited["train.py"] = "from models.net import Net\n\nif __name__ == '__main__':\n    Net()\n# x\n"

    response = index_call(client, edited, bodies=["train.py"])
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_a_body_that_does_not_match_its_manifest_digest_is_refused(client):
    response = client.post(
        "/api/ide/project/index",
        json={
            "protocol_version": 1,
            "root": "/w",
            "manifest": [{"file": "a.py", "digest": digest("X = 1\n")}],
            "bodies": [{"file": "a.py", "source": "X = 2\n"}],
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "digest_mismatch"


def test_absolute_and_traversal_paths_are_refused(client):
    for bad in ("/etc/passwd", "../outside.py", "a\\b.py"):
        response = client.post(
            "/api/ide/project/index",
            json={
                "protocol_version": 1,
                "root": "/w",
                "manifest": [{"file": bad, "digest": digest("X = 1\n")}],
                "bodies": [],
            },
        )
        assert response.status_code == 400, f"{bad} should be refused"
        assert response.json()["detail"]["error"] in {"invalid_path", "duplicate_path"}


def test_protocol_version_mismatch_is_refused(client):
    response = client.post(
        "/api/ide/project/index",
        json={"protocol_version": 99, "root": "/w", "manifest": [], "bodies": []},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "protocol_version_mismatch"


def test_truncation_reaches_the_index(client):
    response = index_call(client, TINY, truncated=True)
    assert response.json()["index"]["truncated"] is True


def test_oversized_body_is_refused(client):
    from research.admission import MAX_BODY_BYTES

    big = "X = 1\n" + ("# pad\n" * ((MAX_BODY_BYTES // 6) + 10))
    response = client.post(
        "/api/ide/project/index",
        json={
            "protocol_version": 1,
            "root": "/w",
            "manifest": [{"file": "big.py", "digest": digest(big)}],
            "bodies": [{"file": "big.py", "source": big}],
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "source_too_large"


# ── /api/ide/project/document ───────────────────────────────────────────────────


COMPOSED = {
    "models/net.py": (
        "import torch.nn as nn\n"
        "\n"
        "\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc1 = nn.Linear(784, 8)\n"
        "        self.fc2 = nn.Linear(8, 10)\n"
        "\n"
        "    def forward(self, x):\n"
        "        return self.fc2(self.fc1(x))\n"
    ),
    "train.py": "from models.net import Net\n\nif __name__ == '__main__':\n    Net()\n",
}


def document_call(client, sources, experiment_id, expanded=(), bodies=None):
    send = sources if bodies is None else {k: sources[k] for k in bodies}
    return client.post(
        "/api/ide/project/document",
        json={
            "protocol_version": 1,
            "root": "/w",
            "manifest": [{"file": f, "digest": digest(s)} for f, s in sources.items()],
            "bodies": [{"file": f, "source": s} for f, s in send.items()],
            "experiment_id": experiment_id,
            "expanded": list(expanded),
        },
    )


def experiment_id_of(client, sources):
    body = index_call(client, sources).json()
    return body["index"]["experiments"][0]["experiment_id"]


def test_document_draws_the_requested_experiment(client):
    exp = experiment_id_of(client, COMPOSED)
    response = document_call(client, COMPOSED, exp)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    labels = {n["data"]["label"] for n in body["document"]["nodes"]}
    assert "Net" in labels


def test_a_collapsed_frame_brings_no_children(client):
    exp = experiment_id_of(client, COMPOSED)
    body = document_call(client, COMPOSED, exp).json()
    assert not [n for n in body["document"]["nodes"] if "/" in n["id"]]


def test_expanding_a_frame_brings_its_layers(client):
    exp = experiment_id_of(client, COMPOSED)
    collapsed = document_call(client, COMPOSED, exp).json()
    frame = next(n for n in collapsed["document"]["nodes"] if n["data"]["label"] == "Net")

    expanded = document_call(client, COMPOSED, exp, expanded=[frame["id"]]).json()
    children = [n for n in expanded["document"]["nodes"] if n.get("parentNode") == frame["id"]]
    assert {c["data"]["label"] for c in children} >= {"fc1", "fc2"}
    assert expanded["expanded"] == [frame["id"]]


def test_every_drawn_card_is_clickable(client):
    exp = experiment_id_of(client, COMPOSED)
    body = document_call(client, COMPOSED, exp).json()
    for node in body["document"]["nodes"]:
        assert node["id"] in body["spanmap"], f"{node['data']['label']} has nowhere to click to"


def test_an_unknown_experiment_is_404_not_an_empty_document(client):
    """"Gone" and "empty" are different answers and the page acts on them differently."""
    response = document_call(client, COMPOSED, "exp:ghost")
    assert response.status_code == 404
    assert response.json()["detail"]["error"] == "unknown_experiment"


def test_the_document_endpoint_uses_the_same_body_handshake(client):
    exp = experiment_id_of(client, COMPOSED)
    response = document_call(client, COMPOSED, exp, bodies=[])
    assert response.status_code == 200
    assert response.json()["status"] == "ok", "the cache filled by /index must serve /document"


# ── /api/ide/project/declare ────────────────────────────────────────────────────


DECLARABLE = "import tensorviz\n\nVALUE = 1\n"


def declare_call(client, source, kind, args, known=(), stale=False):
    return client.post(
        "/api/ide/project/declare",
        json={
            "file": "train.py",
            "source": source,
            "source_digest": digest("something else") if stale else digest(source),
            "kind": kind,
            "args": list(args),
            "known_labels": list(known),
        },
    )


def test_declaring_a_node_returns_a_patch_not_a_write(client):
    response = declare_call(client, DECLARABLE, "node", ["dataset", "raw-logs"])
    assert response.status_code == 200
    body = response.json()
    assert "tensorviz.node('dataset', 'raw-logs')" in body["contents"]
    assert body["statement"] == "tensorviz.node('dataset', 'raw-logs')"
    assert len(body["expected_digest"]) == 64


def test_declaring_an_edge_confirms_a_guess(client):
    response = declare_call(
        client, DECLARABLE, "edge", ["Corpus", "Net", "feeds"], known=["Corpus", "Net"]
    )
    assert response.status_code == 200
    assert "tensorviz.edge('Corpus', 'Net', 'feeds')" in response.json()["contents"]


def test_an_edge_naming_an_absent_node_is_refused(client):
    response = declare_call(client, DECLARABLE, "edge", ["ghost", "Net"], known=["Net"])
    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "unknown_endpoint"


def test_an_unknown_kind_is_refused_with_a_reason(client):
    response = declare_call(client, DECLARABLE, "node", ["banana", "x"])
    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "unknown_kind"


def test_a_file_that_changed_since_the_proposal_is_refused(client):
    """The same staleness rule /apply enforces: never patch a buffer we did not read."""
    response = declare_call(client, DECLARABLE, "node", ["dataset", "raw"], stale=True)
    assert response.status_code == 409
    assert response.json()["detail"]["error"] == "source_stale"


def test_a_file_that_does_not_parse_is_never_written_into(client):
    source = "def oops(\n"
    response = declare_call(client, source, "node", ["dataset", "raw"])
    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "does_not_parse"


def test_a_malformed_declaration_is_refused(client):
    response = declare_call(client, DECLARABLE, "edge", ["only-one"])
    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "malformed_declaration"


# ── /api/ide/project/compare ────────────────────────────────────────────────────


TWO_RUNS = {
    "models/net.py": "import torch.nn as nn\n\n\nclass Net(nn.Module):\n    pass\n",
    "cfg.py": (
        "from dataclasses import dataclass\n\n\n@dataclass\nclass RunConfig:\n"
        "    n_layer: int = 12\n"
    ),
    "train_a.py": (
        "from cfg import RunConfig\nfrom models.net import Net\n\n"
        "if __name__ == '__main__':\n    Net()\n"
    ),
    "train_b.py": (
        "from cfg import RunConfig\nfrom models.net import Net\n\n"
        "def evaluate(model):\n    return 0\n\n"
        "if __name__ == '__main__':\n    Net()\n"
    ),
}


def compare_call(client, sources, left, right):
    return client.post(
        "/api/ide/project/compare",
        json={
            "protocol_version": 1,
            "root": "/w",
            "manifest": [{"file": f, "digest": digest(s)} for f, s in sources.items()],
            "bodies": [{"file": f, "source": s} for f, s in sources.items()],
            "left": left,
            "right": right,
        },
    )


def ids_of(client, sources):
    body = index_call(client, sources).json()
    return {e["entrypoint"]: e["experiment_id"] for e in body["index"]["experiments"]}


def test_compare_reports_what_only_one_side_has(client):
    ids = ids_of(client, TWO_RUNS)
    response = compare_call(client, TWO_RUNS, ids["train_a.py"], ids["train_b.py"])
    assert response.status_code == 200
    diff = response.json()["diff"]
    assert any(n["label"] == "evaluate" and n["side"] == "right" for n in diff["nodes"])
    assert "models/net.py::Net" in diff["shared"]


def test_comparing_an_experiment_with_itself_is_refused(client):
    ids = ids_of(client, TWO_RUNS)
    response = compare_call(client, TWO_RUNS, ids["train_a.py"], ids["train_a.py"])
    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "same_experiment"


def test_comparing_an_unknown_experiment_is_404(client):
    ids = ids_of(client, TWO_RUNS)
    response = compare_call(client, TWO_RUNS, ids["train_a.py"], "exp:ghost")
    assert response.status_code == 404
    assert response.json()["detail"]["error"] == "unknown_experiment"


def test_layers_for_carries_the_forward_pass_and_names_every_child():
    """`examples/tensorviz-research/models/tiny_lm.py`, the file that showed the
    defect: nine children in a row with no edges, and a ModuleList child with no
    title because the parser answers `label: None` for it."""
    import os
    from types import SimpleNamespace
    from api.routers.ide import _layers_for

    here = os.path.dirname(os.path.dirname(__file__))
    path = os.path.normpath(os.path.join(here, "..", "..", "examples", "tensorviz-research", "models", "tiny_lm.py"))
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    node = SimpleNamespace(label="TinyLM", source=SimpleNamespace(file="models/tiny_lm.py"))
    layers = _layers_for(node, {"models/tiny_lm.py": source})
    by_id = {layer["id"]: layer for layer in layers}

    assert "input" not in by_id and "output" not in by_id
    assert by_id["blocks"]["label"] == "blocks"
    assert all(layer["label"] for layer in layers)
    # The forward pass, as `reads`: something downstream of the embeddings, and
    # `head` last of all — and no read names a child that is not there.
    assert by_id["head"]["reads"], "head reads nothing: the edges were dropped again"
    for layer in layers:
        for upstream in layer["reads"]:
            assert upstream in by_id
