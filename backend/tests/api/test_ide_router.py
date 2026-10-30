"""HTTP contract for ``/api/ide/*`` (task.md M3).

What these assert, in the order the milestone claims them:
  * a folder of ``.py`` files becomes a class index without importing anything;
  * a buffer becomes a document whose every node says its shapes are static and
    whose spans point back at real lines;
  * the same buffer parsed twice lands on one derived graph id with a revision
    that goes up, which is the anchor an AI turn's ``base_revision`` needs (D7).

No module-scope torch import and no real model here (task.md §8.3): every
fixture class is eight units wide and none of them is ever instantiated.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


from api.routers import ide  # noqa: E402
from auth import CurrentUser, get_current_user  # noqa: E402
from services.project_store import ProjectStore  # noqa: E402

FIXTURE_REPO = Path(__file__).parent.resolve().parent / "fixtures" / "ide_repo"
MODELS = FIXTURE_REPO / "models"


@pytest.fixture(autouse=True)
def _clear_scan_roots(monkeypatch):
    """Ambient workspace identity must not turn fixture scans into 403s."""
    monkeypatch.delenv("TENSORVIZ_WORKSPACE_DIR", raising=False)
    monkeypatch.delenv("TENSORVIZ_ALLOWED_ROOTS", raising=False)
    monkeypatch.delenv("USER_HISTORY_DB_PATH", raising=False)
    monkeypatch.delenv("TENSORVIZ_LOCAL_TOKEN", raising=False)


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    ide._store = ProjectStore(str(tmp_path / "ide.sqlite3"))
    app = FastAPI()
    app.include_router(ide.router)
    app.dependency_overrides[get_current_user] = lambda: CurrentUser(uid="local-dev-user")
    return TestClient(app)


def _source(name: str) -> str:
    return (MODELS / f"{name}.py").read_text(encoding="utf-8")


def _parse(client: TestClient, name: str, source: str | None = None, cls: str | None = None):
    body = {"file": f"models/{name}.py", "source": source if source is not None else _source(name)}
    if cls:
        body["cls"] = cls
    response = client.post("/api/ide/parse", json=body)
    assert response.status_code == 200, response.text
    return response.json()


# ── workspace index ───────────────────────────────────────────────────────────

def test_workspace_classes_finds_exactly_the_four_fixture_modules(client: TestClient):
    """task.md §7 pins the fixture at four ``nn.Module`` classes in ``models/``.

    ``scripts/`` holds none on purpose, and ``cfg.py``'s ``Cfg`` dataclass is not
    one, so a scan that counted every class would answer 5 and a scan that
    followed imports would answer more.
    """
    response = client.post("/api/ide/workspace/classes", json={"root": str(MODELS)})
    assert response.status_code == 200, response.text
    classes = response.json()["classes"]
    assert {c["name"] for c in classes} == {"Net", "Gate", "MyBlock", "Configured"}
    assert len(classes) == 4

    by_name = {c["name"]: c for c in classes}
    assert by_name["MyBlock"]["file"] == "blocks.py"
    assert by_name["MyBlock"]["line"] == 5
    assert by_name["MyBlock"]["bases"] == ["nn.Module"]
    # Every fixture ctor defaults its width; only Configured demands a config.
    assert by_name["Net"]["ctor_required"] == 0
    assert by_name["Configured"]["ctor_required"] == 1


def test_workspace_classes_imports_nothing(client: TestClient):
    """The endpoint reads source with ``ast``; it must not import the workspace.

    An editor calls this on every open and save. Importing to answer would run a
    half-written training script in the backend process (task.md §8.5).

    The first call is made before the snapshot on purpose: it warms this
    router's own lazy ``services.ingest.repo_scan`` import, which is the
    backend's module and not the workspace's. What the assertion then covers is
    every module the *scan itself* could add, and it must add none.
    """
    client.post("/api/ide/workspace/classes", json={"root": str(MODELS)})
    before = set(sys.modules)
    response = client.post("/api/ide/workspace/classes", json={"root": str(MODELS)})
    assert response.status_code == 200
    assert set(sys.modules) == before


def test_workspace_classes_404s_on_a_path_that_is_not_a_directory(client: TestClient):
    response = client.post("/api/ide/workspace/classes", json={"root": str(MODELS / "resid.py")})
    assert response.status_code == 404


def test_workspace_classes_lists_a_repeated_class_name_once_per_file(
    client: TestClient, tmp_path: Path
):
    """Two files may declare the same class name, and each needs its own row.

    ``scan_repo``'s index is keyed by class name, so it holds only the file scanned
    last. The listing walks ``class_occurrences`` instead. This matters because
    ``codelens.ts`` matches rows to the open document by path: reading the name-keyed
    index dropped every copy but one, and "Open as Graph" then went missing on the
    others while appearing normally on identically-shaped neighbours.
    """
    for folder in ("alpha", "beta"):
        directory = tmp_path / folder
        directory.mkdir()
        (directory / "tiny.py").write_text(
            "import torch.nn as nn\n\n\nclass Tiny(nn.Module):\n    pass\n",
            encoding="utf-8",
        )

    response = client.post("/api/ide/workspace/classes", json={"root": str(tmp_path)})
    assert response.status_code == 200, response.text

    tiny = [c for c in response.json()["classes"] if c["name"] == "Tiny"]
    assert [c["file"] for c in tiny] == ["alpha/tiny.py", "beta/tiny.py"]


def test_workspace_classes_follows_imports_without_matching_unrelated_names(client: TestClient, tmp_path: Path):
    sources = {
        "base.py": "from torch import nn\nclass Block(nn.Module): pass\n",
        "child.py": "from base import Block as Parent\nclass Child(Parent): pass\n",
        "helper.py": "class Block: pass\nclass Helper(Block): pass\n",
    }
    for file, source in sources.items():
        (tmp_path / file).write_text(source)
    response = client.post("/api/ide/workspace/classes", json={"root": str(tmp_path)})
    assert response.status_code == 200, response.text
    assert {(c["file"], c["name"]) for c in response.json()["classes"]} == {
        ("base.py", "Block"), ("child.py", "Child")}


# ── buffer classes ────────────────────────────────────────────────────────────

def _buffer_classes(client: TestClient, source: str, file: str = "scratch.py"):
    response = client.post("/api/ide/buffer-classes", json={"file": file, "source": source})
    assert response.status_code == 200, response.text
    return response.json()["classes"]


def test_buffer_classes_lists_every_module_in_the_buffer_with_its_span(client: TestClient):
    """The answer carries 1-based spans, which is the whole point: the extension
    resolves the innermost class under the cursor without parsing anything itself."""
    classes = _buffer_classes(
        client,
        "import torch.nn as nn\n"
        "\n"
        "\n"
        "class First(nn.Module):\n"
        "    def forward(self, x):\n"
        "        return x\n"
        "\n"
        "\n"
        "class Second(nn.Module):\n"
        "    pass\n",
    )
    assert [c["name"] for c in classes] == ["First", "Second"]
    assert (classes[0]["line"], classes[0]["end_line"]) == (4, 6)
    assert (classes[1]["line"], classes[1]["end_line"]) == (9, 10)


def test_buffer_classes_closes_over_a_buffer_local_base(client: TestClient):
    """A subclass that names a sibling class — not ``Module`` — is still a model.

    If the buffer scan used a weaker rule than the workspace index, the picker
    could offer ``Child`` from one surface and refuse it on the other.
    """
    classes = _buffer_classes(
        client,
        "import torch.nn as nn\n"
        "\n"
        "class Base(nn.Module):\n"
        "    pass\n"
        "\n"
        "class Child(Base):\n"
        "    pass\n",
    )
    assert [c["name"] for c in classes] == ["Base", "Child"]


def test_module_discovery_follows_optional_import_fallback_aliases(client, tmp_path):
    """DINOv2's fused block inherits through an optional xformers fallback alias."""
    source = (
        'from torch import nn\n'
        'raise RuntimeError("discovery must not execute the model")\n'
        'class SwiGLUFFN(nn.Module):\n'
        '    pass\n'
        'try:\n'
        '    if XFORMERS_ENABLED:\n'
        '        from xformers.ops import SwiGLU\n'
        '    else:\n'
        '        raise ImportError\n'
        'except ImportError:\n'
        '    SwiGLU = SwiGLUFFN\n'
        'class SwiGLUFFNFused(SwiGLU):\n'
        '    pass\n'
        'class Child(SwiGLUFFNFused):\n'
        '    pass\n'
        'class SwiGLUFFNAligned(nn.Module):\n'
        '    pass\n'
    )
    (tmp_path / 'swiglu_ffn.py').write_text(source)
    expected = {'SwiGLUFFN', 'SwiGLUFFNFused', 'Child', 'SwiGLUFFNAligned'}
    assert {c['name'] for c in _buffer_classes(client, source)} == expected
    response = client.post('/api/ide/workspace/classes', json={'root': str(tmp_path)})
    assert response.status_code == 200, response.text
    classes = response.json()['classes']
    assert {c['name'] for c in classes} == expected
    fused = next(c for c in classes if c['name'] == 'SwiGLUFFNFused')
    assert (fused['file'], fused['line'], fused['bases']) == ('swiglu_ffn.py', 12, ['SwiGLU'])


def test_external_model_base_reaches_buffer_and_workspace_codelens_indexes(client, tmp_path):
    source = ('from transformers import PreTrainedModel\n'
              'raise RuntimeError("indexing must not load weights or execute source")\n'
              'class SmartTurnMixin: pass\n'
              'class AudioModel(SmartTurnMixin, PreTrainedModel): pass\n')
    (tmp_path / 'model.py').write_text(source)
    assert [(c['name'], c['line']) for c in _buffer_classes(client, source)] == [('AudioModel', 4)]
    response = client.post('/api/ide/workspace/classes', json={'root': str(tmp_path)})
    assert response.status_code == 200, response.text
    assert [(c['file'], c['name'], c['line']) for c in response.json()['classes']] == [('model.py', 'AudioModel', 4)]


def test_module_aliases_are_file_local_and_do_not_leak_from_scopes(client, tmp_path):
    source = (
        'class Base(nn.Module): pass\n'
        'Alias = Base\n'
        'Chained: type = Alias\n'
        'class Child(Chained): pass\n'
        'Alias = object\n'
        'class Helper(Alias): pass\n'
        'def configure():\n'
        '    LocalAlias = Base\n'
        'class Config:\n'
        '    ScopedAlias = Base\n'
        'class LocalHelper(LocalAlias): pass\n'
        'class ScopedHelper(ScopedAlias): pass\n'
        'CycleA = CycleB\n'
        'CycleB = CycleA\n'
        'class Cyclic(CycleA): pass\n'
        'class Plain: pass\n'
        'Other = Plain\n'
        'class NonModel(Other): pass\n'
    )
    (tmp_path / 'model.py').write_text(source)
    (tmp_path / 'other.py').write_text('class Unrelated(Chained): pass\n')
    assert {c['name'] for c in _buffer_classes(client, source)} == {'Base', 'Child'}
    response = client.post('/api/ide/workspace/classes', json={'root': str(tmp_path)})
    assert response.status_code == 200, response.text
    assert {c['name'] for c in response.json()['classes']} == {'Base', 'Child'}


def test_buffer_classes_reads_the_buffer_as_it_stands_not_the_disk(client: TestClient):
    """The unsaved-buffer contract: ``source`` wins, and the endpoint must lie
    about nothing — ``resid.py`` on disk declares only ``Net``."""
    classes = _buffer_classes(
        client,
        _source("resid") + "\n\nclass Unsaved(nn.Module):\n    pass\n",
        file="models/resid.py",
    )
    assert [c["name"] for c in classes] == ["Net", "Unsaved"]


def test_buffer_classes_with_no_module_is_an_empty_answer_not_an_error(client: TestClient):
    assert _buffer_classes(client, "class Config:\n    pass\n") == []


@pytest.mark.parametrize("source", ["return 1\n", "break\n", "await operation()\n", "nonlocal missing\n"])
def test_buffer_classes_optional_compilation_rejects_non_executable_syntax(client: TestClient, source: str):
    # These parse to an AST but cannot become an executable Python module.
    assert _buffer_classes(client, source) == []
    response = client.post("/api/ide/buffer-classes", json={
        "file": "proposal.py", "source": source, "check_compilation": True,
    })
    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "syntax_error"
    assert response.json()["detail"]["line"] == 1


def test_buffer_classes_compilation_never_executes_proposed_source(client: TestClient, tmp_path: Path):
    marker = tmp_path / "must-not-exist"
    source = f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\nclass Config:\n    width = 16\n"
    response = client.post("/api/ide/buffer-classes", json={
        "file": "proposal.py", "source": source, "check_compilation": True,
    })
    assert response.status_code == 200, response.text
    assert response.json() == {"classes": []}
    assert not marker.exists()


def test_buffer_classes_reports_a_syntax_error_in_the_parse_endpoints_shape(client: TestClient):
    response = client.post(
        "/api/ide/buffer-classes",
        json={"file": "scratch.py", "source": "class Net(nn.Module)\n    pass\n"},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["error"] == "syntax_error"
    assert detail["line"] == 1


# ── parse ─────────────────────────────────────────────────────────────────────

def test_fully_qualified_module_from_the_picker_opens_without_execution(client):
    source = ('import torch\n'
              'raise RuntimeError("static opening must never execute this")\n'
              'class Net(torch.nn.Module):\n'
              '    def forward(self, x):\n'
              '        y = x + 1\n'
              '        return y\n')
    classes = _buffer_classes(client, source)
    assert [item['name'] for item in classes] == ['Net']
    payload = _parse(client, 'qualified', source, cls=classes[0]['name'])
    assert payload['cls'] == 'Net' and payload['traced'] is False
    baseline = _parse(client, 'qualified', source.replace('torch.nn.Module', 'nn.Module'), cls='Net')
    assert [n['node_id'] for n in payload['document']['semantic']['nodes']] == [
        n['node_id'] for n in baseline['document']['semantic']['nodes']]


def test_parse_returns_a_static_document_with_spans_for_every_non_boundary_node(client: TestClient):
    payload = _parse(client, "resid")
    assert payload["cls"] == "Net"
    assert payload["traced"] is False

    nodes = payload["document"]["semantic"]["nodes"]
    assert len(nodes) == 7
    # Every node, boundaries included: nothing in this document was executed.
    for node in nodes:
        assert node["params"]["_shape_provenance"] == "static", node["node_id"]

    spanmap = payload["spanmap"]
    boundary = {"input", "output"}
    covered = {n["node_id"] for n in nodes if n["node_id"] not in boundary}
    assert set(spanmap) == covered
    for node_id, span in spanmap.items():
        assert span["file"] == "models/resid.py"
        assert span["start"][0] >= 1 and span["end"][0] >= span["start"][0]
        assert span["kind"] in {"decl", "call"}
        assert span["digest"], node_id


def test_parse_names_module_nodes_by_attribute_and_ops_by_fn_owner_ordinal(client: TestClient):
    """D6, the two identity rules, on one graph."""
    payload = _parse(client, "resid")
    ids = {n["node_id"] for n in payload["document"]["semantic"]["nodes"]}
    assert {"l1", "l2", "blocks"} <= ids           # attribute paths, verbatim
    assert {"add@Net#0", "python@Net#0"} <= ids    # fn@owner#ordinal
    assert not [i for i in ids if i.endswith("_0") or i.endswith("_1")]


def test_parse_reports_a_syntax_error_with_the_line_the_editor_must_underline(client: TestClient):
    response = client.post(
        "/api/ide/parse",
        json={"file": "models/resid.py", "source": "class Net(nn.Module)\n    pass\n"},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["error"] == "syntax_error"
    assert detail["line"] == 1


def test_parse_can_be_asked_for_a_class_by_name_and_404s_for_one_that_is_absent(client: TestClient):
    payload = _parse(client, "cfg", cls="Configured")
    assert payload["cls"] == "Configured"

    response = client.post(
        "/api/ide/parse", json={"file": "models/cfg.py", "source": _source("cfg"), "cls": "Nope"}
    )
    assert response.status_code == 404


# ── the CAS anchor (task.md D7 / M7) ──────────────────────────────────────────

def test_two_parses_of_one_buffer_share_a_graph_id_and_bump_the_revision(client: TestClient):
    first = _parse(client, "resid")
    second = _parse(client, "resid")

    assert first["graph_id"] == second["graph_id"]
    assert first["graph_id"].startswith("ide:")
    assert first["graph_id"].endswith("#Net")
    assert second["revision"] == first["revision"] + 1


def test_the_projection_is_readable_from_the_store_at_the_revision_returned(client: TestClient):
    """Without this row an AI turn cannot be created at all: ``AiTurnRecord``
    validates ``committed_revision == base_revision + 1`` against the store."""
    payload = _parse(client, "resid")
    stored = ide._store.get_graph("local-dev-user", payload["graph_id"])
    assert stored is not None
    assert stored["current_revision"] == payload["revision"]
    assert stored["name"] == "Net"


def test_a_different_file_with_the_same_class_name_gets_a_different_anchor(client: TestClient):
    """The id hashes the path, not just the class: two ``Net``s must not collide."""
    a = _parse(client, "resid")
    b = client.post(
        "/api/ide/parse", json={"file": "other/resid.py", "source": _source("resid")}
    ).json()
    assert a["graph_id"] != b["graph_id"]


# ── /project ──────────────────────────────────────────────────────────────────

def test_project_without_a_trace_matches_parse(client: TestClient):
    parsed = _parse(client, "gate")
    response = client.post(
        "/api/ide/project", json={"file": "models/gate.py", "source": _source("gate")}
    )
    assert response.status_code == 200, response.text
    projected = response.json()
    assert projected["graph_id"] == parsed["graph_id"]
    assert projected["document"]["semantic"]["nodes"] == parsed["document"]["semantic"]["nodes"]


def test_project_refuses_a_trace_it_cannot_read_rather_than_ignoring_it(client: TestClient):
    """M4 made ``/project`` accept a real trace, so what must stay refused is one this
    backend cannot read. Accepting and dropping it would return a document that claims
    to be executed when nothing ran (CLAUDE.md rule 12)."""
    response = client.post(
        "/api/ide/project",
        json={
            "file": "models/gate.py",
            "source": _source("gate"),
            "trace_result": {"protocol_version": 99, "result": {"graph": {"nodes": []}}},
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "trace_rejected"


@pytest.mark.parametrize('edited', [False, True])
def test_graph_setup_reopens_only_unchanged_generated_config(client, edited):
    import hashlib
    import tomllib
    from services.ide.graph_setup_profiles import prepare_profile
    from services.ide.multifile import MultifileProjectionRequest
    source = '''raise RuntimeError("discovery must not execute workspace source")
from torch import nn
from dataclasses import dataclass
@dataclass
class Config:
    width: int
class Net(nn.Module):
    def __init__(self, config: Config): pass
    def forward(self, x): return x
'''
    digest = hashlib.sha256(source.encode()).hexdigest()
    body = dict(protocol_version=1, root=dict(file='model.py', cls='Net', source_digest=digest),
                sources=[dict(file='model.py', source=source, digest=digest)])
    profile = dict(kind='model', model='model:Net', mode='eval', device='cpu', grad_enabled=False,
                   seed=23, init=dict(args=[], kwargs={}), inputs=dict(args=[1], kwargs={}))
    prepared = prepare_profile(MultifileProjectionRequest.model_validate(body), '', profile,
                               config_object=dict(index=0, parameter='config', values={'width': 12}))
    helper = prepared['factory']
    text = helper['source'] + ('\n# user change\n' if edited else '')
    body['sources'].append(dict(file=helper['file'], source=text, digest=hashlib.sha256(text.encode()).hexdigest()))
    saved = tomllib.loads(prepared['config'])['profiles'][prepared['profile']]
    response = client.post('/api/ide/graph-setup/discover', json={**body, 'profile': saved})
    assert response.status_code == 200, response.text
    restored = response.json()['restoredDraft']
    if edited:
        assert restored is None
    else:
        assert restored['configObject']['values'] == {'width': 12}
        assert restored['profile']['inputs'] == saved['inputs']
        assert restored['profile']['seed'] == 23


def test_explicit_graph_picker_can_request_unverified_external_subclasses(client, tmp_path):
    marker = tmp_path / 'must-not-import'
    source = f'''from pathlib import Path
Path({str(marker)!r}).touch()
from transformers import BertModel
class TextEncoder(BertModel): pass
class Settings: pass
class Plain(object): pass
'''
    assert _buffer_classes(client, source) == [], 'normal indexes must not certify unresolved inheritance'
    response = client.post('/api/ide/buffer-classes', json={
        'file': 'library.py', 'source': source, 'include_unresolved_bases': True})
    assert response.status_code == 200
    assert response.json()['classes'] == [dict(name='TextEncoder', line=4, end_line=4,
                                               bases=['BertModel'], unverified=True)]
    assert not marker.exists()


def test_unverified_picker_does_not_offer_nested_class_as_a_top_level_target(client):
    source = 'class Container:\n    class Nested(External): pass\n'
    response = client.post('/api/ide/buffer-classes', json={
        'file': 'library.py', 'source': source, 'include_unresolved_bases': True})
    assert response.status_code == 200 and response.json()['classes'] == []


def test_unverified_picker_keeps_the_top_level_span_when_nested_names_repeat(client):
    source = 'class TextEncoder(External): pass\nclass Container:\n    class TextEncoder(Other): pass\n'
    response = client.post('/api/ide/buffer-classes', json={
        'file': 'library.py', 'source': source, 'include_unresolved_bases': True})
    assert response.status_code == 200
    assert response.json()['classes'] == [dict(name='TextEncoder', line=1, end_line=1,
                                               bases=['External'], unverified=True)]


def test_project_rejects_parent_traversal(client: TestClient):
    for file in ("../secret.py", "models/../../etc/passwd.py", "..\\secret.py"):
        response = client.post("/api/ide/project", json={"file": file, "source": "x = 1\n"})
        assert response.status_code == 400, file
        assert response.json()["detail"]["error"] == "path_traversal"


def test_project_accepts_a_workspace_relative_file(client: TestClient):
    response = client.post(
        "/api/ide/project",
        json={"file": "models/gate.py", "source": _source("gate"), "cls": "Gate"},
    )
    assert response.status_code == 200, response.text


def test_workspace_classes_rejects_a_root_outside_the_workspace(client, tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "evil.py").write_text(
        "import torch.nn as nn\nclass Evil(nn.Module):\n    pass\n", encoding="utf-8"
    )
    monkeypatch.setenv("TENSORVIZ_WORKSPACE_DIR", str(workspace))
    refused = client.post("/api/ide/workspace/classes", json={"root": str(outside)})
    assert refused.status_code == 403
    nested = workspace / "pkg"
    nested.mkdir()
    (nested / "ok.py").write_text(
        "import torch.nn as nn\nclass Ok(nn.Module):\n    pass\n", encoding="utf-8"
    )
    allowed = client.post("/api/ide/workspace/classes", json={"root": str(nested)})
    assert allowed.status_code == 200, allowed.text
    assert {row["name"] for row in allowed.json()["classes"]} == {"Ok"}


def test_workspace_classes_rejects_a_symlink_that_leaves_the_workspace(client, tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "evil.py").write_text(
        "import torch.nn as nn\nclass Evil(nn.Module):\n    pass\n", encoding="utf-8"
    )
    (workspace / "link").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("TENSORVIZ_WORKSPACE_DIR", str(workspace))
    response = client.post("/api/ide/workspace/classes", json={"root": str(workspace / "link")})
    assert response.status_code == 403


def test_allowed_roots_cover_a_second_workspace_folder(client, tmp_path, monkeypatch):
    primary = tmp_path / "primary"
    extra = tmp_path / "extra"
    primary.mkdir()
    extra.mkdir()
    (extra / "m.py").write_text(
        "import torch.nn as nn\nclass Extra(nn.Module):\n    pass\n", encoding="utf-8"
    )
    monkeypatch.setenv("TENSORVIZ_WORKSPACE_DIR", str(primary))
    monkeypatch.setenv("TENSORVIZ_ALLOWED_ROOTS", os.pathsep.join((str(primary), str(extra))))
    response = client.post("/api/ide/workspace/classes", json={"root": str(extra)})
    assert response.status_code == 200, response.text
    assert {row["name"] for row in response.json()["classes"]} == {"Extra"}


def test_token_mode_without_a_workspace_refuses_every_root(client, tmp_path, monkeypatch):
    monkeypatch.setenv("TENSORVIZ_LOCAL_TOKEN", "launch-token-with-no-folder")
    response = client.post("/api/ide/workspace/classes", json={"root": str(tmp_path)})
    assert response.status_code == 403
