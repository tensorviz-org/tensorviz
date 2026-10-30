"""Baseline contract + characterization for ``POST /api/ide/multifile/project``.

Step 1 of "P0 Reliable Multi-File Model Graphs" (CURRENT_TASK.md) freezes the
versioned request/response, the digests, the diagnostic codes, the path rules and
the budgets — and characterizes what the projection does today, because the
shared contract is the thing every later step has to respect:

  * the parse is still the single-buffer one: the graph equals ``/api/ide/parse``
    for the root buffer, every span points back at the root file, the dependency
    set is the root alone, and imported classes stay opaque;
  * what step 1 DOES change: every import that stays opaque is surfaced as a
    structured UNRESOLVED_IMPORT diagnostic, so the limitation is visible rather
    than silent, and the projection digest tracks the whole supplied source set
    so a dirty dependency is observable before any resolution exists;
  * the request's security rules are NOT baseline behaviour: absolute paths,
    traversal, backslashes, duplicates, bad digests, a missing root and oversized
    payloads are refused deterministically, now and after the resolver lands.

Step 3 LANDED: the projection now resolves cross-file. The step-1 baseline
characterisation tests (document == single-buffer parse, spans root-only,
dependencies root-only, every imported class opaque) were replaced here by the
assertions this module's own docstring pre-announced:

  * absolute-from, aliased, relative and package re-export bindings all resolve,
    and the duplicate blocks.py/heads.py `Block` never contaminate each other;
  * `self.enc1`/`self.enc2` are distinct grouped modules sharing the canonical
    models.blocks::Block definition, expanded to qualified children (enc1.fc);
  * dependencies list every consulted file with its digest; spans are cross-file;
  * the CycleA/CycleB pair terminates at one opaque CYCLE_BOUNDARY card, and
    DynamicStage's computed block stays opaque with UNRESOLVED_DYNAMIC;
  * a trusted trace enriches the same ids without deleting unresolved nodes.
"""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import ide
from auth import CurrentUser, get_current_user
from services.ide import multifile
from services.project_store import ProjectStore

FIXTURE_REPO = Path(__file__).parent.resolve().parent / "fixtures" / "ide_multifile_repo"
FIXTURE_FILES = [
    "models/__init__.py",
    "models/blocks.py",
    "models/cycle_a.py",
    "models/cycle_b.py",
    "models/dynamic.py",
    "models/enc.py",
    "models/heads.py",
    "models/mixer.py",
    "models/net.py",
]
ROOT = "models/net.py"


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    ide._store = ProjectStore(str(tmp_path / "ide.sqlite3"))
    app = FastAPI()
    app.include_router(ide.router)
    app.dependency_overrides[get_current_user] = lambda: CurrentUser(uid="local-dev-user")
    return TestClient(app)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _entry(path: str, source: str | None = None) -> dict:
    text = (FIXTURE_REPO / path).read_text(encoding="utf-8") if source is None else source
    return {"file": path, "source": text, "digest": _sha(text)}


def _payload(
    overrides: dict | None = None,
    *,
    root_file: str = ROOT,
    cls: str = "Net",
    version: int = multifile.IDE_PROJECTION_PROTOCOL_VERSION,
    truncated: bool = False,
) -> dict:
    overrides = overrides or {}
    sources = [_entry(path, overrides.get(path)) for path in FIXTURE_FILES]
    root_entry = next((entry for entry in sources if entry["file"] == root_file), None)
    root_digest = root_entry["digest"] if root_entry is not None else _sha("missing")
    return {
        "protocol_version": version,
        "root": {"file": root_file, "cls": cls, "document_version": 1, "source_digest": root_digest},
        "sources": sources,
        "truncated": truncated,
    }


def _project(client: TestClient, payload: dict | None = None):
    response = client.post("/api/ide/multifile/project", json=payload or _payload())
    assert response.status_code == 200, response.text
    return response.json()


def test_modulelist_stays_a_static_container_after_framework_opacity(client):
    """Current browser projection must keep the container and its warning,
    without treating declared member descriptions as captured child nodes."""
    from models.architecture_models import GraphDocumentV2
    from services.canonical_validation import CanonicalValidationService
    from services.verification_service import architecture_graph_to_runtime_payload

    source = (Path(__file__).parent.parent / "fixtures/ide_repo/models/resid.py").read_text()
    payload = {"protocol_version": multifile.IDE_PROJECTION_PROTOCOL_VERSION,
               "root": {"file": "models/resid.py", "cls": "Net", "document_version": 1, "source_digest": _sha(source)},
               "sources": [_entry("models/resid.py", source)]}
    result = _project(client, payload)
    document = GraphDocumentV2.model_validate(result["document"])
    block = next(n for n in document.semantic.nodes if n.node_id == "blocks")
    assert block.kind == "module"
    assert block.op == "custom_code"
    assert block.params["_class"] == "ModuleList"
    assert block.params["_opaque"] is True
    assert block.params["_shape_provenance"] == "static"
    assert [m["op"] for m in block.params["_modules"]] == ["Linear", "Linear"]
    assert block.metadata["layerType"]["id"] == "custom_code"
    assert any(d["node"] == "blocks" for d in result["diagnostics"])
    assert result["spanmap"]["blocks"]["file"] == "models/resid.py"
    assert result["spanmap"]["blocks"]["kind"] == "decl"
    assert not any(n.node_id.startswith("blocks.") for n in document.semantic.nodes)
    pairs = [(e.source_node_id, e.target_node_id) for e in document.semantic.edges]
    assert not any("blocks" in pair for pair in pairs)
    assert ("l1", "python@Net#0") in pairs and ("python@Net#0", "l2") in pairs
    view = document.view.layout
    assert view["node_canvas"]["blocks"]["type"] == "moduleNode"
    assert view["node_data"]["blocks"]["moduleTemplateId"] == "ModuleList"
    assert view["node_data"]["blocks"]["moduleMembers"] == []
    assert view["node_data"]["blocks"]["expanded"] is False
    validation = CanonicalValidationService().validate(document.semantic, strict=True)
    assert validation.node_results["blocks"].status != "error"
    assert not validation.node_results["blocks"].suggested_repairs
    assert validation.node_results["blocks"].output_shape is None
    assert validation.node_results["python@Net#0"].status == "error"
    assert "not shape-checked" in validation.node_results["python@Net#0"].message
    assert "blocks" not in {n["id"] for n in architecture_graph_to_runtime_payload(document.semantic)["nodes"]}


def test_residual_role_survives_nested_input_boundary_projection(client: TestClient):
    root = '''import torch.nn as nn
from models.blocks import Block
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block()
    def forward(self, x):
        return self.block(x)
'''
    block = '''import torch.nn as nn
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(8, 16)
        self.down = nn.Linear(16, 8)
    def forward(self, x):
        return self.down(self.up(x)) + x
'''
    data = _project(client, _payload({ROOT: root, 'models/blocks.py': block}))
    residuals = [e for e in data['document']['semantic']['edges'] if e['metadata'].get('role') == 'residual']
    assert len(residuals) == 1
    assert residuals[0]['source_node_id'] == 'input'
    assert residuals[0]['target_node_id'].startswith('add@block#')


def _reject(client: TestClient, payload: dict, status: int, code: str):
    response = client.post("/api/ide/multifile/project", json=payload)
    assert response.status_code == status, response.text
    if status == 400:
        assert response.json()["detail"]["error"] == code
    return response


# ── fixture integrity ────────────────────────────────────────────────────────


def test_fixture_files_exist_and_parse():
    for path in FIXTURE_FILES:
        full = FIXTURE_REPO / path
        assert full.is_file(), f"missing fixture file {path}"
        ast.parse(full.read_text(encoding="utf-8"))

    # And only these: a stray file in the package would silently change the
    # source set every test builds.
    on_disk = sorted(
        str(p.relative_to(FIXTURE_REPO)) for p in FIXTURE_REPO.rglob("*.py")
        if "__pycache__" not in p.parts
    )
    assert on_disk == sorted(FIXTURE_FILES)


def test_fixture_has_the_two_duplicate_blocks():
    names = {}
    for path in ("models/blocks.py", "models/heads.py"):
        tree = ast.parse((FIXTURE_REPO / path).read_text(encoding="utf-8"))
        names[path] = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    assert names["models/blocks.py"] == ["Block"]
    assert names["models/heads.py"] == ["Block", "WideBlock"]


def test_fixture_cycle_is_mutual_and_never_importable_first():
    a = (FIXTURE_REPO / "models/cycle_a.py").read_text(encoding="utf-8")
    b = (FIXTURE_REPO / "models/cycle_b.py").read_text(encoding="utf-8")
    assert "from .cycle_b import CycleB" in a
    assert "from .cycle_a import CycleA" in b


def test_fixture_dynamic_module_is_computed():
    text = (FIXTURE_REPO / "models/dynamic.py").read_text(encoding="utf-8")
    assert "importlib.import_module(" in text
    assert "getattr(module, " in text


def test_fixture_root_uses_every_supported_form():
    tree = ast.parse((FIXTURE_REPO / ROOT).read_text(encoding="utf-8"))
    from_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    ]
    absolute = [n for n in from_imports if n.level == 0]
    relative = [n for n in from_imports if n.level == 1]
    assert {n.module for n in absolute} == {"models", "models.heads"}
    assert {n.module for n in relative} == {"cycle_a", "dynamic", "enc", "mixer"}
    aliased = [
        alias
        for node in from_imports
        for alias in node.names
        if alias.asname
    ]
    # Only HeadBlock is aliased on this side: CoreBlock's alias lives in
    # models/__init__.py (that aliasing is what makes it a re-export). The
    # root just imports the name the package exported.
    assert {a.asname for a in aliased} == {"HeadBlock"}
    unaliased = [alias.name for node in from_imports for alias in node.names if not alias.asname]
    assert "CoreBlock" in unaliased

    init = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "Net"
    )
    text = ast.dump(init)
    for repeated in ("CoreBlock",):
        assert text.count(repeated) == 2, f"{repeated} must be instantiated twice"


# ── request validation (the permanent security rules) ──────────────────────────


def test_protocol_version_must_match(client: TestClient):
    _reject(client, _payload(version=2), 400, "protocol_version_mismatch")


@pytest.mark.parametrize("bad", ["/abs/net.py", "//host/share/net.py", "~/net.py", "C:/work/net.py"])
def test_absolute_paths_refused(client: TestClient, bad: str):
    payload = _payload()
    payload["sources"][0] = _entry("models/net.py") | {"file": bad}
    _reject(client, payload, 400, "absolute_path")


@pytest.mark.parametrize("bad", ["models/../net.py", "models/./net.py", "models//net.py", "a/b/../../net.py"])
def test_traversal_and_unnormalised_paths_refused(client: TestClient, bad: str):
    payload = _payload()
    payload["sources"][0] = _entry("models/net.py") | {"file": bad}
    expected = "traversal" if (".." in bad or "/./" in bad) else "not_normalised_path"
    _reject(client, payload, 400, expected)


def test_backslash_paths_refused(client: TestClient):
    payload = _payload()
    payload["sources"][0] = _entry("models/net.py") | {"file": "models\\net.py"}
    _reject(client, payload, 400, "not_normalised_path")


def test_non_python_entries_refused(client: TestClient):
    payload = _payload()
    payload["sources"][0] = _entry("models/net.py") | {"file": "models/notes.txt"}
    _reject(client, payload, 400, "not_python")


def test_duplicate_paths_refused_whether_identical_or_conflicting(client: TestClient):
    payload = _payload()
    copy = dict(payload["sources"][0])
    payload["sources"].append(copy)
    _reject(client, payload, 400, "duplicate_path")

    payload = _payload()
    rival = _entry("models/blocks.py", "# different bytes\n")
    payload["sources"].append(rival)
    _reject(client, payload, 400, "duplicate_path")


def test_entry_digest_must_match_its_source(client: TestClient):
    payload = _payload()
    payload["sources"][1] = dict(payload["sources"][1]) | {"digest": "z" * 64}
    _reject(client, payload, 400, "digest_malformed")

    payload = _payload()
    payload["sources"][1] = dict(payload["sources"][1]) | {"digest": _sha("not this text")}
    _reject(client, payload, 400, "digest_mismatch")


def test_root_must_be_in_the_source_set_with_matching_digest(client: TestClient):
    payload = _payload()
    payload["root"]["file"] = "models/missing.py"
    # A root absent from the set cannot have a matching supplied digest either,
    # so the root check is what fails.
    _reject(client, payload, 400, "root_not_in_sources")

    payload = _payload()
    payload["root"]["source_digest"] = _sha("some other buffer")
    _reject(client, payload, 400, "root_digest_mismatch")


def test_source_count_ceiling(client: TestClient):
    payload = _payload()
    payload["sources"] = payload["sources"] + [
        _entry(f"extra/file_{i}.py", "# pad\n") for i in range(multifile.MAX_SOURCES)
    ]
    # len limit is structural (pydantic), so this one is a 422, not a coded 400.
    response = client.post("/api/ide/multifile/project", json=payload)
    assert response.status_code == 422


def test_total_byte_ceiling_is_a_coded_refusal(client: TestClient):
    chunk = "#" * multifile.MAX_SOURCE_BYTES
    sources = []
    index = 0
    total = 0
    while total <= multifile.MAX_TOTAL_SOURCE_BYTES:
        total += len(chunk.encode("utf-8"))
        sources.append(_entry(f"big/file_{index}.py", chunk))
        index += 1
    payload = _payload()
    payload["sources"] = sources + payload["sources"]
    _reject(client, payload, 400, multifile.RESOURCE_LIMIT)


def test_per_source_ceiling_counts_utf8_bytes_not_characters(client: TestClient):
    payload = _payload()
    # Pydantic's string max_length counts characters. The contract ceiling is
    # bytes, so a smaller number of multibyte characters must still be refused.
    oversized = "é" * (multifile.MAX_SOURCE_BYTES // 2 + 1)
    payload["sources"][0] = _entry("models/__init__.py", oversized)
    _reject(client, payload, 400, multifile.RESOURCE_LIMIT)


# ── baseline characterization: single-buffer truth, visibly labelled ───────────


def test_multifile_is_a_strict_superset_of_the_single_buffer_parse(client: TestClient):
    """Every node and span the single-buffer tier produced still exists, at the
    same identity — the cross-file tier REPLACES the opaque placeholders with
    expanded modules and adds the children the single-buffer tier could not see.
    """
    payload = _payload()
    single = client.post(
        "/api/ide/parse",
        json={"file": ROOT, "source": payload["sources"][-1]["source"], "cls": "Net"},
    )
    assert single.status_code == 200, single.text
    single = single.json()
    data = _project(client, payload)

    assert data["graph_id"] == single["graph_id"]
    assert data["source_digest"] == single["source_digest"]
    assert data["traced"] == single["traced"] is False
    assert data["protocol_version"] == multifile.IDE_PROJECTION_PROTOCOL_VERSION

    single_ids = {n["node_id"] for n in single["document"]["semantic"]["nodes"]}
    multi_ids = {n["node_id"] for n in data["document"]["semantic"]["nodes"]}
    assert single_ids <= multi_ids, single_ids - multi_ids
    assert {"enc1.fc", "enc1.act", "mix.inner.fc"} <= multi_ids

    stripped = {
        node_id: {k: v for k, v in span.items() if k not in {"file_digest", "focus"}}
        for node_id, span in data["spanmap"].items()
    }
    for node_id, span in single["spanmap"].items():
        assert stripped[node_id] == span, node_id
        # Workspace focus enriches the same source location without moving it.
        assert data["spanmap"][node_id]["focus"]["node"] == [[span["start"][0], span["end"][0]]]


def test_projection_digest_tracks_the_whole_source_set(client: TestClient):
    """`source_digest` covers the root; `projection_digest` covers everything.

    The second one is what lets a host notice "a dependency I supplied is now
    different" even while the baseline resolver ignores the dependency's
    contents — the digest moves, the graph does not (yet).
    """
    data = _project(client)
    assert data["source_digest"] == _sha((FIXTURE_REPO / ROOT).read_text(encoding="utf-8"))
    assert data["projection_digest"] != data["source_digest"]
    assert data["projection_digest"] == _project(client)["projection_digest"]

    dirty = _entry("models/blocks.py")[ "source" ] + "\n# unsaved edit\n"
    edited = _project(client, _payload(overrides={"models/blocks.py": dirty}))
    assert edited["source_digest"] == data["source_digest"]
    assert edited["projection_digest"] != data["projection_digest"]
    # Drawing is unchanged; capture-context evidence tracks the dependency bytes.
    left = data["document"]["semantic"]["metadata"]["reviewClass"]
    right = edited["document"]["semantic"]["metadata"]["reviewClass"]
    assert left["structureDigest"] != right["structureDigest"]
    assert left["dependencyDigests"]["models/blocks.py"] == _sha(_entry("models/blocks.py")["source"])
    assert right["dependencyDigests"]["models/blocks.py"] == _sha(dirty)
    assert {key: value for key, value in left["dependencyDigests"].items() if key != "models/blocks.py"} == {
        key: value for key, value in right["dependencyDigests"].items() if key != "models/blocks.py"}
    left.pop("structureDigest")
    right.pop("structureDigest")
    left.pop("dependencyDigests")
    right.pop("dependencyDigests")
    assert edited["document"] == data["document"]


def test_every_span_points_at_its_own_file_with_that_files_digest(client: TestClient):
    data = _project(client)
    digests = {d["file"]: d["digest"] for d in data["dependencies"]}
    assert data["spanmap"], "the fixture root must produce spans"
    for node_id, span in data["spanmap"].items():
        assert span["file"] in digests, (node_id, span["file"])
        assert span["file_digest"] == digests[span["file"]], node_id
    # The marquee case: enc1.fc's declaration lives in models/blocks.py, at the
    # exact line of `self.fc = nn.Linear(width, width)`.
    blocks_text = (FIXTURE_REPO / "models" / "blocks.py").read_text(encoding="utf-8")
    decl_line = blocks_text[: blocks_text.index("self.fc = ")].count("\n") + 1
    span = data["spanmap"]["enc1.fc"]
    assert span["file"] == "models/blocks.py"
    assert span["start"][0] == decl_line
    assert span["kind"] == "decl"
    # Instance declaration stays in the root file.
    assert data["spanmap"]["enc1"]["file"] == ROOT


def test_dependency_spans_count_utf16_columns_against_their_own_file(client: TestClient):
    # Python's AST counts UTF-8 bytes; the editor counts UTF-16 code units. Each
    # span converts against the text of the file it names, not the root's.
    root = (
        "from torch import nn\nfrom blocks import Block\nclass Net(nn.Module):\n"
        "    def __init__(self):\n        super().__init__()\n        self.block = Block()\n"
        "    def forward(self, x):\n        return self.block(x)\n"
    )
    blocks = (
        "from torch import nn\nclass Block(nn.Module):\n    def __init__(self, d_μ=4):\n"
        "        super().__init__()\n        self.seq = nn.Sequential(nn.Linear(d_μ, 4), nn.ReLU())\n"
        "    def forward(self, x):\n        return self.seq(x)\n"
    )
    sources = [{"file": file, "source": text, "digest": _sha(text)} for file, text in (("model.py", root), ("blocks.py", blocks))]
    data = _project(client, {
        "protocol_version": multifile.IDE_PROJECTION_PROTOCOL_VERSION,
        "root": {"file": "model.py", "cls": "Net", "source_digest": sources[0]["digest"]},
        "sources": sources,
    })
    line = blocks.split("\n")[4]
    relu = next(span for key, span in data["spanmap"].items() if key.startswith("relu@"))
    assert relu["file"] == "blocks.py"
    assert (relu["start"], relu["end"]) == ([5, line.index("nn.ReLU")], [5, line.rindex(")")])


def test_dependency_set_covers_every_file_the_projection_consumed(client: TestClient):
    data = _project(client)
    by_file = {d["file"]: d["digest"] for d in data["dependencies"]}
    assert set(by_file) == set(FIXTURE_FILES), sorted(set(FIXTURE_FILES) - set(by_file))
    # Every dependency digest is the digest of what was actually supplied.
    for filename in FIXTURE_FILES:
        text = (FIXTURE_REPO / filename).read_text(encoding="utf-8")
        assert by_file[filename] == _sha(text)
    # __init__.py is consulted as the re-export hop for CoreBlock even though it
    # defines no layers: editing it must retrigger a projection.
    dirty_init = _entry("models/__init__.py")["source"] + "\n"
    edited = _project(client, _payload(overrides={"models/__init__.py": dirty_init}))
    assert edited["projection_digest"] != data["projection_digest"]


def test_local_wildcard_projection_expands_children_and_keeps_export_dependencies(client):
    source = """from torch import nn
from blocks import *
from unused import *
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block()
    def forward(self, x):
        return self.block(x)
"""
    blocks = """from torch import nn
raise RuntimeError('static opening must not execute this')
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)
    def forward(self, x):
        return self.fc(x)
"""
    def request(block_source):
        return {"protocol_version": multifile.IDE_PROJECTION_PROTOCOL_VERSION,
                "root": {"file": "model.py", "cls": "Net", "document_version": 1, "source_digest": _sha(source)},
                "sources": [_entry('model.py', source), _entry('blocks.py', block_source),
                            _entry('unused.py', '__all__ = []\n')]}
    data = _project(client, request(blocks))
    nodes = {node['node_id']: node for node in data['document']['semantic']['nodes']}
    assert {'block', 'block.fc'} <= set(nodes)
    assert nodes['block']['params']['_defined_in'] == 'blocks.py'
    assert data['spanmap']['block.fc']['file'] == 'blocks.py'
    assert {item['file'] for item in data['dependencies']} == {'model.py', 'blocks.py', 'unused.py'}
    assert not any(d['code'] == multifile.UNSUPPORTED_IMPORT_FORM for d in data['diagnostics'])
    changed = _project(client, request(blocks + '\n__all__ = []\n'))
    assert changed['projection_digest'] != data['projection_digest']
    assert 'block.fc' not in {node['node_id'] for node in changed['document']['semantic']['nodes']}
    assert any(d['code'] == multifile.UNRESOLVED_IMPORT for d in changed['diagnostics'])


def test_imported_classes_resolve_into_grouped_modules(client: TestClient):
    """The six supported import forms all expand into real module groups; aliases
    and re-exports land on their canonical definitions despite the duplicate
    Block names; and what cannot resolve stays opaque WITH a reason.
    """
    data = _project(client)
    nodes = {n["node_id"]: n for n in data["document"]["semantic"]["nodes"]}
    view_data = data["document"]["view"]["layout"]["node_data"]

    for wrapper in ("enc1", "enc2", "head", "wide", "mix", "stack", "loop", "dyn"):
        assert nodes[wrapper]["kind"] == "module", wrapper
        assert nodes[wrapper]["op"] == "custom_code", wrapper
        meta = view_data[wrapper]
        assert meta["isGroup"] is True
        assert meta["expanded"] is False
        assert meta["parentId"] is None
        assert meta["depth"] == 0
        assert isinstance(meta["moduleMembers"], list)

    # The duplicate-name proof: enc1 is models.blocks::Block, head is
    # models.heads::Block — neither leaked into the other.
    assert nodes["enc1"]["params"]["_import_path"] == "models.blocks"
    assert nodes["enc1"]["params"]["_defined_in"] == "models/blocks.py"
    assert nodes["head"]["params"]["_import_path"] == "models.heads"
    assert nodes["head"]["params"]["_defined_in"] == "models/heads.py"

    # Repeated instances of one class are distinct subtrees.
    assert nodes["enc1"]["params"]["_class"] == "Block"
    assert view_data["enc1"]["moduleTemplateId"] == "models.blocks::Block"
    assert view_data["enc2"]["moduleTemplateId"] == "models.blocks::Block"
    assert view_data["enc1"]["moduleMembers"] == ["enc1.fc", "enc1.act"]
    assert view_data["enc2"]["moduleMembers"] == ["enc2.fc", "enc2.act"]
    assert {"enc1.fc", "enc1.act", "enc2.fc", "enc2.act"} <= set(nodes)
    assert view_data["enc1.fc"]["parentId"] == "enc1"
    assert view_data["enc1.fc"]["depth"] == 1

    # Nested expansion: Mixer holds a Block one level down.
    assert nodes["mix.inner"]["kind"] == "module"
    assert view_data["mix.inner"]["parentId"] == "mix"
    assert view_data["mix.inner.fc"]["parentId"] == "mix.inner"
    assert view_data["mix.inner.fc"]["depth"] == 2

    # The cycle terminates at exactly one opaque card, marked with the reason.
    assert nodes["loop"]["params"]["_import_path"] == "models.cycle_a"
    assert nodes["loop.inner"]["params"]["_import_path"] == "models.cycle_b"
    boundary = nodes["loop.inner.inner"]
    assert boundary["params"]["_opaque"] is True
    assert boundary["params"]["_class"] == "CycleA"

    # The computed class is visible and opaque, not guessed.
    computed = nodes["dyn.inner"]
    assert computed["params"]["_opaque"] is True
    assert computed["params"]["_class"] == "block_cls"

    codes = [d["code"] for d in data["diagnostics"]]
    assert codes.count("CYCLE_BOUNDARY") == 1
    assert "UNRESOLVED_DYNAMIC" in codes
    for diagnostic in data["diagnostics"]:
        assert diagnostic["code"] in multifile.DIAGNOSTIC_CODES
        assert diagnostic["severity"] in ("info", "warning", "error")
    cycle = next(d for d in data["diagnostics"] if d["code"] == "CYCLE_BOUNDARY")
    assert cycle["node"] == "loop.inner.inner"
    assert cycle["file"] == "models/cycle_b.py"
    dynamic = next(
        d for d in data["diagnostics"] if d.get("node") == "dyn.inner"
    )
    assert dynamic["code"] == "UNRESOLVED_DYNAMIC"
    assert dynamic["file"] == "models/dynamic.py"
    # Resolved classes earn NO unresolved diagnostics: nothing is double-reported.
    unresolved = [d for d in data["diagnostics"] if d["code"] == "UNRESOLVED_IMPORT"]
    assert unresolved == []


def test_expanded_graph_edges_bridge_through_modules(client: TestClient):
    """Wrapper edges (collapsed view) and deep leaf bridges (expanded view) coexist."""
    data = _project(client)
    pairs = {
        (e["source_node_id"], e["target_node_id"])
        for e in data["document"]["semantic"]["edges"]
    }
    # wrapper level
    assert ("inp", "enc1") in pairs
    assert ("enc1", "enc2") in pairs
    assert ("dyn", "out") in pairs
    # leaf level
    assert ("enc1.fc", "enc1.act") in pairs
    assert ("inp", "enc1.fc") in pairs
    assert ("enc1.act", "enc2.fc") in pairs
    assert ("enc2.act", "head.proj") in pairs
    assert ("head.proj", "wide.proj") in pairs
    assert ("wide.proj", "wide.narrow") in pairs
    assert ("wide.narrow", "mix.fc") in pairs
    assert ("mix.fc", "mix.act") in pairs
    assert ("mix.act", "mix.inner.fc") in pairs
    # opaque boundary nodes sit in the chain at their level
    assert ("loop.inner.inner", "dyn.inner") in pairs


def test_same_file_custom_classes_are_not_reported_unresolved(client: TestClient):
    """The diagnostic is for what the projection COULD NOT follow, not for every
    custom class: classes defined in the root file itself resolve today and must
    not be flagged."""
    source = (FIXTURE_REPO / ROOT).read_text(encoding="utf-8") + (
        "\n\nclass LocalBlock(nn.Module):\n"
        "    def __init__(self, width: int = 8):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(width, width)\n\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n"
    )
    payload = _payload(overrides={ROOT: source}, cls="Net")
    data = _project(client, payload)
    for diagnostic in data["diagnostics"]:
        assert "LocalBlock" not in diagnostic["detail"]


def test_truncated_source_sets_are_visible(client: TestClient):
    data = _project(client, _payload(truncated=True))
    assert data["truncated"] is True
    truncated = [d for d in data["diagnostics"] if d["code"] == "SOURCE_SET_TRUNCATED"]
    assert len(truncated) == 1
    assert truncated[0]["severity"] == "warning"

    complete = _project(client, _payload(truncated=False))
    assert complete["truncated"] is False
    assert not [d for d in complete["diagnostics"] if d["code"] == "SOURCE_SET_TRUNCATED"]


def test_the_call_imports_nothing_and_writes_nothing(client: TestClient):
    """The whole point of the feature: static projection without side effects.

    No module may appear in sys.modules (that would mean user code ran), and the
    fixture directory must be byte-identical afterwards — no caches, no
    __pycache__, no state files.
    """
    fixture_snapshot = {
        str(p.relative_to(FIXTURE_REPO)): p.read_bytes()
        for p in FIXTURE_REPO.rglob("*")
        if p.is_file()
    }
    before_modules = set(sys.modules)
    data = _project(client)
    assert data["cls"] == "Net"
    assert set(sys.modules) == before_modules
    after = {
        str(p.relative_to(FIXTURE_REPO)): p.read_bytes()
        for p in FIXTURE_REPO.rglob("*")
        if p.is_file()
    }
    assert after == fixture_snapshot
    assert not any("__pycache__" in name for name in after)


# ── stable identities: ids name instances, not import spellings or whitespace ────


def _ids(data: dict) -> set:
    return {node["node_id"] for node in data["document"]["semantic"]["nodes"]}


def _node(data: dict, node_id: str) -> dict:
    return next(
        node
        for node in data["document"]["semantic"]["nodes"]
        if node["node_id"] == node_id
    )


def _node_params(data: dict, node_id: str) -> dict:
    return _node(data, node_id)["params"]


def _plain_spans(data: dict) -> dict:
    return {
        node_id: {k: v for k, v in span.items() if k != "file_digest"}
        for node_id, span in data["spanmap"].items()
    }


def test_alias_only_rename_keeps_ids(client: TestClient):
    """Identities describe module instances, not import spellings (the product
    contract): renaming ONLY the alias the root binds `CoreBlock` to must move no
    node, no span, and no template link — even though the raw call text changes."""
    renamed_root = (
        (FIXTURE_REPO / ROOT)
        .read_text(encoding="utf-8")
        .replace(
            "from models import CoreBlock",
            "from models import CoreBlock as RenamedBlock",
        )
        .replace("= CoreBlock(width)", "= RenamedBlock(width)")
    )
    assert "RenamedBlock" in renamed_root and "= CoreBlock(" not in renamed_root

    before = _project(client)
    after = _project(client, _payload(overrides={ROOT: renamed_root}))

    assert _ids(after) == _ids(before)
    assert set(after["spanmap"]) == set(before["spanmap"])
    assert after["graph_id"] == before["graph_id"]

    # The spelling changed exactly where it should and nowhere it shouldn't.
    assert _node_params(after, "enc1")["_raw_call"] == "RenamedBlock(width)"
    assert _node_params(after, "enc1")["_class"] == "Block"
    assert _node_params(after, "enc1")["_import_path"] == "models.blocks"
    view_data = after["document"]["view"]["layout"]["node_data"]
    assert view_data["enc1"]["moduleTemplateId"] == "models.blocks::Block"


def test_unrelated_root_edit_preserves_identity(client: TestClient):
    """Noise must not scramble the canvas: a comment appended to the root is
    observed (both digests change — staleness is honest) but the graph is
    byte-identical, because there is nothing new to draw."""
    before = _project(client)
    commented = (FIXTURE_REPO / ROOT).read_text(encoding="utf-8") + "\n# note to self\n"
    after = _project(client, _payload(overrides={ROOT: commented}))

    assert after["source_digest"] != before["source_digest"]
    assert after["projection_digest"] != before["projection_digest"]
    assert after["document"] == before["document"]
    # Spans are identical apart from each root-file entry's own digest.
    assert _plain_spans(after) == _plain_spans(before)


def test_dependency_body_edit_moves_children(client: TestClient):
    """A real edit inside a dependency rewrites the children of EVERY instance of
    the edited class: `Block`'s `.fc` becoming `.lin` moves `enc1.fc`, `enc2.fc`
    AND `mix.inner.fc` together, because the identities name the declarations."""
    edited_blocks = (
        (FIXTURE_REPO / "models/blocks.py")
        .read_text(encoding="utf-8")
        .replace("self.fc = nn.Linear", "self.lin = nn.Linear")
        .replace("self.act(self.fc(x))", "self.act(self.lin(x))")
    )
    assert "self.fc" not in edited_blocks

    data = _project(client, _payload(overrides={"models/blocks.py": edited_blocks}))
    ids = _ids(data)
    view_data = data["document"]["view"]["layout"]["node_data"]

    assert "enc1.fc" not in ids and "enc2.fc" not in ids and "mix.inner.fc" not in ids
    assert "mix.fc" not in ids
    assert {"enc1.lin", "enc2.lin", "mix.inner.lin", "mix.lin"} <= ids
    assert view_data["enc1"]["moduleMembers"] == ["enc1.lin", "enc1.act"]
    assert view_data["mix"]["moduleMembers"] == ["mix.lin", "mix.act", "mix.inner"]
    assert view_data["mix.inner"]["moduleMembers"] == ["mix.inner.lin", "mix.inner.act"]

    # The moved child snaps to the renamed declaration, same file, same line.
    decl_line = edited_blocks[: edited_blocks.index("self.lin = ")].count("\n") + 1
    span = data["spanmap"]["enc1.lin"]
    assert span["file"] == "models/blocks.py"
    assert span["start"][0] == decl_line
    assert span["file_digest"] == _sha(edited_blocks)


# ── single local inheritance: base members, overrides, flows, provenance ──────


def test_single_local_inheritance_composes_members_flows_and_spans(client: TestClient):
    """`class Sub(Base)` with one workspace-local nn.Module base projects the
    members the subclass actually runs with: the base's ``self.x`` declarations
    when ``super().__init__`` is followed (or no own ``__init__`` exists), the
    subclass's own declarations, and a ``super().forward`` call wired INTO the
    base flow rather than passing its argument through untouched."""
    data = _project(client)
    ids = _ids(data)
    view_data = data["document"]["view"]["layout"]["node_data"]

    # wide: heads.py WideBlock(Block) — inherited proj runs before own narrow.
    assert view_data["wide"]["moduleMembers"] == ["wide.proj", "wide.narrow"]
    assert view_data["wide.proj"]["parentId"] == "wide"

    # stack: enc.py Encoder(blocks.Block) — inherited fc/act plus the residual
    # add that `super().forward(x) + x` creates.
    residual = next(n for n in ids if n.startswith("add@stack#"))
    assert view_data["stack"]["moduleMembers"] == ["stack.fc", "stack.act", residual]

    # mix: mixer.py Mixer(Block) — inherited fc/act plus its own nested inner.
    assert view_data["mix"]["moduleMembers"] == ["mix.fc", "mix.act", "mix.inner"]
    assert view_data["mix.inner"]["moduleMembers"] == ["mix.inner.fc", "mix.inner.act"]

    # Every span names the file the statement lives in, whatever class inherited it.
    blocks_text = (FIXTURE_REPO / "models/blocks.py").read_text(encoding="utf-8")
    heads_text = (FIXTURE_REPO / "models/heads.py").read_text(encoding="utf-8")
    mixer_text = (FIXTURE_REPO / "models/mixer.py").read_text(encoding="utf-8")
    fc_line = blocks_text[: blocks_text.index("self.fc = ")].count("\n") + 1
    proj_line = heads_text[: heads_text.index("self.proj = ")].count("\n") + 1
    narrow_line = heads_text[: heads_text.index("self.narrow = ")].count("\n") + 1
    inner_line = mixer_text[: mixer_text.index("self.inner = ")].count("\n") + 1

    for node_id in ("stack.fc", "mix.fc"):
        span = data["spanmap"][node_id]
        assert span["file"] == "models/blocks.py", node_id
        assert span["start"][0] == fc_line, node_id
        assert span["kind"] == "decl", node_id
    assert data["spanmap"]["wide.proj"]["file"] == "models/heads.py"
    assert data["spanmap"]["wide.proj"]["start"][0] == proj_line
    assert data["spanmap"]["wide.narrow"]["file"] == "models/heads.py"
    assert data["spanmap"]["wide.narrow"]["start"][0] == narrow_line
    assert data["spanmap"]["mix.inner"]["file"] == "models/mixer.py"
    assert data["spanmap"]["mix.inner"]["start"][0] == inner_line
    # The residual op belongs to the subclass's own forward.
    assert data["spanmap"][residual]["file"] == "models/enc.py"
    assert data["spanmap"][residual]["kind"] == "call"

    # Deep (leaf) edges cross the inherited flows, not the wrapper shells.
    pairs = {
        (e["source_node_id"], e["target_node_id"])
        for e in data["document"]["semantic"]["edges"]
    }
    assert ("head.proj", "wide.proj") in pairs
    assert ("wide.narrow", "mix.fc") in pairs
    assert ("mix.act", "mix.inner.fc") in pairs
    assert ("mix.inner.act", "stack.fc") in pairs
    assert ("stack.fc", "stack.act") in pairs
    assert ("stack.act", residual) in pairs
    assert ("mix.inner.act", residual) in pairs  # the residual's other input is x
    # Wrapper-order edges still exist for the collapsed view.
    assert ("wide", "mix") in pairs and ("mix", "stack") in pairs
    # Composing inheritance added no diagnostics to the clean fixture.
    assert [d for d in data["diagnostics"] if d["code"] == "UNRESOLVED_IMPORT"] == []


def test_unsupported_inheritance_is_coded_not_composed(client: TestClient):
    """Multiple nn.Module bases and an __init__ that never calls
    ``super().__init__`` both stay the subclass's own members plus a coded
    diagnostic — never a guessed merge."""
    source = (
        "import torch.nn as nn\n"
        "\n\n"
        "class BaseA(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.a = nn.Linear(8, 8)\n\n"
        "    def forward(self, x):\n"
        "        return self.a(x)\n\n\n"
        "class BaseB(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.b = nn.Linear(8, 8)\n\n"
        "    def forward(self, x):\n"
        "        return self.b(x)\n\n\n"
        "class Twin(BaseA, BaseB):\n"
        "    def __init__(self):\n"
        "        BaseA.__init__(self)\n"
        "        self.t = nn.Linear(8, 8)\n\n"
        "    def forward(self, x):\n"
        "        return self.t(self.a(x))\n\n\n"
        "class Quiet(BaseA):\n"
        "    def __init__(self):\n"
        "        self.q = nn.Linear(8, 8)\n\n"
        "    def forward(self, x):\n"
        "        return self.q(x)\n\n\n"
        "class Mini(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.twin = Twin()\n"
        "        self.quiet = Quiet()\n\n"
        "    def forward(self, x):\n"
        "        return self.quiet(self.twin(x))\n"
        "# tensorviz: input=8\n"
    )
    data = _project(client, _payload(overrides={ROOT: source}, cls="Mini"))
    ids = _ids(data)
    view_data = data["document"]["view"]["layout"]["node_data"]

    assert view_data["twin"]["moduleMembers"] == ["twin.t"]
    assert view_data["quiet"]["moduleMembers"] == ["quiet.q"]
    assert "twin.a" not in ids and "quiet.a" not in ids

    unsupported = [d for d in data["diagnostics"] if d["code"] == "UNSUPPORTED_INHERITANCE"]
    assert len(unsupported) == 2, unsupported
    assert all(d["severity"] == "warning" for d in unsupported)
    assert all(d["file"] == ROOT for d in unsupported)


def test_workspace_class_named_like_a_registry_op_still_expands(client: TestClient):
    """`class Sub(Block)` lowercases to the layer id ``sub`` — a REAL op in the
    registry. The registry short-circuit must answer only AFTER the module index
    has failed to ground the callee; otherwise every workspace class whose name
    collides with an op (Sub, Add, Cat) is silently treated as a framework op:
    kept as a leaf, never expanded, never diagnosed."""
    heads = (
        "import torch.nn as nn\n\n\n"
        "class Block(nn.Module):\n"
        "    def __init__(self, width: int = 8):\n"
        "        super().__init__()\n"
        "        self.proj = nn.Linear(width, width)\n\n"
        "    def forward(self, x):\n"
        "        return self.proj(x)\n\n\n"
        "class Sub(Block):\n"
        "    def __init__(self, width: int = 8):\n"
        "        super().__init__(width)\n\n"
        "    def forward(self, x):\n"
        "        return super().forward(x)\n"
    )
    net = (
        "import torch.nn as nn\n\n"
        "from .heads import Sub\n\n\n"
        "class Net(nn.Module):\n"
        "    def __init__(self, width: int = 8):\n"
        "        super().__init__()\n"
        "        self.sub = Sub(width)\n"
        "        self.out = nn.Linear(width, width)\n\n"
        "    def forward(self, x):\n"
        "        return self.out(self.sub(x))\n\n\n"
        "# tensorviz: input=8\n"
    )
    data = _project(client, _payload(overrides={"models/heads.py": heads, ROOT: net}))
    ids = _ids(data)
    view_data = data["document"]["view"]["layout"]["node_data"]

    assert view_data["sub"]["isGroup"] is True
    assert view_data["sub"]["moduleMembers"] == ["sub.proj"]
    assert "sub.proj" in ids
    assert data["spanmap"]["sub.proj"]["file"] == "models/heads.py"
    assert [
        d for d in data["diagnostics"] if d.get("node") == "sub"
    ] == [], "a resolvable class must earn no diagnostic, op-name collision or not"


def test_inherited_and_own_forward_ops_never_share_an_id(client: TestClient):
    """Both the base's forward and the replacing subclass forward call
    ``torch.relu``: per-class identity mapping gives BOTH statements the same
    class-level id (``python@<Class>#0``), and qualification must then keep them
    distinct — folding them into one ``python@wide#0`` would silently merge two
    statements and converge their edges."""
    heads = (
        "import torch\n"
        "import torch.nn as nn\n\n\n"
        "class Block(nn.Module):\n"
        "    def __init__(self, width: int = 8):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(width, width)\n\n"
        "    def forward(self, x):\n"
        "        return torch.relu(self.fc(x))\n\n\n"
        "class WideBlock(Block):\n"
        "    def __init__(self, width: int = 8):\n"
        "        super().__init__(width)\n"
        "        self.narrow = nn.Linear(width, width)\n\n"
        "    def forward(self, x):\n"
        "        return torch.relu(self.narrow(x))\n"
    )
    net = (
        "import torch.nn as nn\n\n"
        "from .heads import WideBlock\n\n\n"
        "class Net(nn.Module):\n"
        "    def __init__(self, width: int = 8):\n"
        "        super().__init__()\n"
        "        self.wide = WideBlock(width)\n\n"
        "    def forward(self, x):\n"
        "        return self.wide(x)\n\n\n"
        "# tensorviz: input=8\n"
    )
    data = _project(client, _payload(overrides={"models/heads.py": heads, ROOT: net}))
    view_data = data["document"]["view"]["layout"]["node_data"]

    members = view_data["wide"]["moduleMembers"]
    assert [m for m in members if not m.startswith("python@")] == ["wide.fc", "wide.narrow"]
    fn_children = sorted(
        n for n in _ids(data) if n.startswith("python@wide#") or n.startswith("python@wide@")
    )
    assert len(fn_children) == 2, fn_children
    assert len(set(fn_children)) == 2, fn_children

    pairs = {
        (e["source_node_id"], e["target_node_id"])
        for e in data["document"]["semantic"]["edges"]
    }
    fc_sinks = {t for s, t in pairs if s == "wide.fc"}
    narrow_sinks = {t for s, t in pairs if s == "wide.narrow"}
    assert fc_sinks and narrow_sinks and not (fc_sinks & narrow_sinks)

    # The two statements keep their own provenance: different lines, same file.
    spans = [data["spanmap"][n] for n in fn_children]
    assert all(s["file"] == "models/heads.py" for s in spans)
    assert spans[0]["start"] != spans[1]["start"]


def test_conditional_and_unsupported_super_init_refuse_composition(client: TestClient):
    """Composition rests on the base's ``__init__`` PROVABLY running — which
    only an unconditional top-level ``super().__init__()`` statement proves. An
    ``if width > 4: super().__init__()`` and a ``super(str, width).__init__()``
    both leave the base's ``self.`` assignments unprovable, so both refuse with
    a coded diagnostic rather than projecting members the object may never have."""
    heads = (
        "import torch.nn as nn\n\n\n"
        "class Block(nn.Module):\n"
        "    def __init__(self, width: int = 8):\n"
        "        super().__init__()\n"
        "        self.proj = nn.Linear(width, width)\n\n"
        "    def forward(self, x):\n"
        "        return self.proj(x)\n\n\n"
        "class WideBlock(Block):\n"
        "    def __init__(self, width: int = 8):\n"
        "        if width > 4:\n"
        "            super().__init__(width)\n"
        "        self.narrow = nn.Linear(width, width)\n\n"
        "    def forward(self, x):\n"
        "        return self.narrow(x)\n\n\n"
        "class OddBlock(Block):\n"
        "    def __init__(self, width: int = 8):\n"
        "        super(str, width).__init__()\n"
        "        self.side = nn.Linear(width, width)\n\n"
        "    def forward(self, x):\n"
        "        return self.side(x)\n"
    )
    net = (
        "import torch.nn as nn\n\n"
        "from .heads import WideBlock, OddBlock\n\n\n"
        "class Net(nn.Module):\n"
        "    def __init__(self, width: int = 8):\n"
        "        super().__init__()\n"
        "        self.wide = WideBlock(width)\n"
        "        self.odd = OddBlock(width)\n\n"
        "    def forward(self, x):\n"
        "        return self.odd(self.wide(x))\n\n\n"
        "# tensorviz: input=8\n"
    )
    data = _project(client, _payload(overrides={"models/heads.py": heads, ROOT: net}))
    ids = _ids(data)
    view_data = data["document"]["view"]["layout"]["node_data"]

    assert view_data["wide"]["moduleMembers"] == ["wide.narrow"]
    assert view_data["odd"]["moduleMembers"] == ["odd.side"]
    assert "wide.proj" not in ids and "odd.proj" not in ids

    unsupported = [
        d for d in data["diagnostics"]
        if d["code"] == "UNSUPPORTED_INHERITANCE" and d["file"] == "models/heads.py"
    ]
    assert len(unsupported) == 2, unsupported
    assert all("unconditional" in d["detail"] for d in unsupported)


def test_explicit_super_must_name_the_current_class(client: TestClient):
    source = (
        "import torch.nn as nn\n\n"
        "class Base(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.base = nn.Linear(8, 8)\n\n"
        "    def forward(self, x):\n"
        "        return self.base(x)\n\n"
        "class Other:\n"
        "    pass\n\n"
        "class Wrong(Base):\n"
        "    def __init__(self):\n"
        "        super(Other, self).__init__()\n"
        "        self.own = nn.Linear(8, 8)\n\n"
        "    def forward(self, x):\n"
        "        return self.own(x)\n\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.wrong = Wrong()\n\n"
        "    def forward(self, x):\n"
        "        return self.wrong(x)\n"
    )
    data = _project(client, _payload(overrides={ROOT: source}))
    members = data["document"]["view"]["layout"]["node_data"]["wrong"]["moduleMembers"]
    assert members == ["wrong.own"]
    assert "wrong.base" not in _ids(data)
    assert any(
        diagnostic["code"] == "UNSUPPORTED_INHERITANCE"
        and "unconditional" in diagnostic["detail"]
        for diagnostic in data["diagnostics"]
    )


def test_subclass_forward_calling_inherited_member_wires_it_in(client: TestClient):
    """``self.narrow(self.fc(x))`` in the subclass's forward where ``fc`` is
    INHERITED: the base's ``__init__`` provably assigned it, so the edge
    ``wide.fc -> wide.narrow`` must exist — and a non-module mixin base rides
    along with a diagnostic that says composition PROCEEDED for the nn.Module
    base, never the false "keeps its own members only"."""
    heads = (
        "import torch.nn as nn\n\n\n"
        "class Block(nn.Module):\n"
        "    def __init__(self, width: int = 8):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(width, width)\n\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n\n\n"
        "class Mixin:\n"
        "    pass\n\n\n"
        "class WideBlock(Block, Mixin):\n"
        "    def __init__(self, width: int = 8):\n"
        "        super().__init__(width)\n"
        "        self.narrow = nn.Linear(width, width)\n\n"
        "    def forward(self, x):\n"
        "        return self.narrow(self.fc(x))\n"
    )
    net = (
        "import torch.nn as nn\n\n"
        "from .heads import WideBlock\n\n\n"
        "class Net(nn.Module):\n"
        "    def __init__(self, width: int = 8):\n"
        "        super().__init__()\n"
        "        self.wide = WideBlock(width)\n\n"
        "    def forward(self, x):\n"
        "        return self.wide(x)\n\n\n"
        "# tensorviz: input=8\n"
    )
    data = _project(client, _payload(overrides={"models/heads.py": heads, ROOT: net}))
    ids = _ids(data)
    view_data = data["document"]["view"]["layout"]["node_data"]

    assert view_data["wide"]["moduleMembers"] == ["wide.fc", "wide.narrow"]
    assert {"wide.fc", "wide.narrow"} <= ids
    pairs = {
        (e["source_node_id"], e["target_node_id"])
        for e in data["document"]["semantic"]["edges"]
    }
    assert ("wide.fc", "wide.narrow") in pairs

    mixin = [
        d for d in data["diagnostics"]
        if d["code"] == "UNRESOLVED_IMPORT" and "Mixin" in d["detail"]
    ]
    assert len(mixin) == 1, mixin
    assert "composes only its nn.Module base's members" in mixin[0]["detail"]
    assert "keeps its own members only" not in mixin[0]["detail"]


# ── trace alignment: execution enriches the SAME nodes, never replaces them ──────


def _trace_node(node_id: str, op: str, *, module: bool = False) -> dict:
    data = {
        "layerType": {"id": op, "name": op},
        "parameters": {"_shape_provenance": "observed", "in_features": 8, "out_features": 8},
    }
    if module:
        data["isGroup"] = True
    return {"id": node_id, "type": "moduleNode" if module else "layerNode", "data": data}


def _trace_envelope() -> dict:
    """A partial trace of the fixture model, the way the worker reports one after
    the run stopped early: observed modules are named by their qualified
    attribute path — the same ids the static projection computed — and the tail
    from `wide` onwards is simply absent.
    """
    nodes = [
        {
            "id": "input_node",
            "type": "inputNode",
            "data": {
                "layerType": {"id": "input", "name": "Input"},
                "parameters": {"shape": [1, 8]},
            },
        },
        _trace_node("inp", "Linear"),
        _trace_node("enc1", "Block", module=True),
        _trace_node("enc1.fc", "Linear"),
        _trace_node("enc1.act", "GELU"),
        _trace_node("enc2", "Block", module=True),
        _trace_node("enc2.fc", "Linear"),
        _trace_node("enc2.act", "GELU"),
        _trace_node("head", "Block", module=True),
        _trace_node("head.proj", "Linear"),
    ]
    chain = [
        ("input_node", "inp"),
        ("inp", "enc1.fc"),
        ("enc1.fc", "enc1.act"),
        ("enc1.act", "enc2.fc"),
        ("enc2.fc", "enc2.act"),
        ("enc2.act", "head.proj"),
    ]
    edges = [
        {"id": f"trace_edge_{index}", "source": source, "target": target}
        for index, (source, target) in enumerate(chain)
    ]
    return {
        "protocol_version": 1,
        "result": {
            "graph": {"nodes": nodes, "edges": edges},
            "diagnostics": {},
        },
        "spans": {},
        "error": None,
    }


def test_a_trusted_trace_enriches_the_same_nodes_and_keeps_the_unresolved(client: TestClient):
    """Acceptance: "a trusted trace enriches the same nodes". The worker names
    executed modules by the SAME qualified instance path the static projection
    uses, so tracing enriches those nodes with observed facts while the parser
    supplies the source facts execution cannot see — and every node the run never
    reached stays on the canvas, dimmed, instead of being deleted."""
    static = _project(client)
    assert static["traced"] is False

    observed = {"inp", "enc1", "enc1.fc", "enc1.act", "enc2", "enc2.fc", "enc2.act", "head", "head.proj"}
    assert observed <= _ids(static), "the trace may only name nodes the static graph already owns"

    payload = _payload()
    payload["trace_result"] = _trace_envelope()
    data = _project(client, payload)

    assert data["traced"] is True
    assert data["trace_error"] is None
    assert observed <= _ids(data)

    # Same node, both tiers: observed shapes from the trace, declaration facts
    # (line in the OWNING file, raw call text) from the parse.
    fc = _node_params(data, "enc1.fc")
    assert fc["_shape_provenance"] == "observed"
    assert fc["in_features"] == 8
    blocks_text = (FIXTURE_REPO / "models" / "blocks.py").read_text(encoding="utf-8")
    decl_line = blocks_text[: blocks_text.index("self.fc = ")].count("\n") + 1
    assert fc["_lineno"] == decl_line
    assert fc["_raw_call"] == "nn.Linear(width, width)"

    span = data["spanmap"]["enc1.fc"]
    assert span["file"] == "models/blocks.py"
    assert span["kind"] == "decl"

    # The untraced tail survives, dimmed — including every deliberately opaque card.
    for survivor in ("wide", "mix", "mix.inner.fc", "stack", "loop", "loop.inner", "dyn", "out"):
        assert _node_params(data, survivor)["_shape_provenance"] == "unreached", survivor
    dyn_inner = _node_params(data, "dyn.inner")
    assert dyn_inner["_shape_provenance"] == "unreached"
    assert dyn_inner["_opaque"] is True

    # The traced head hands over to the static tail with real edges, not ghosts —
    # through wide's INHERITED proj, the same first hop the untraced graph draws.
    pairs = {
        (edge["source_node_id"], edge["target_node_id"])
        for edge in data["document"]["semantic"]["edges"]
    }
    assert ("head.proj", "wide.proj") in pairs
    assert ("loop.inner.inner", "dyn.inner") in pairs
    assert ("dyn.fc", "out") in pairs


def test_trace_error_frames_resolve_into_dependency_files(client: TestClient):
    """When the run raises, the surfaced frame is the innermost WORKSPACE frame —
    and with several files in play that can be a dependency's line, not the root's."""
    blocks_text = (FIXTURE_REPO / "models" / "blocks.py").read_text(encoding="utf-8")
    forward_line = blocks_text[: blocks_text.index("self.act(self.fc(x))")].count("\n") + 1

    envelope = _trace_envelope()
    envelope["error"] = {
        "type": "RuntimeError",
        "message": "mat1 and mat2 shapes cannot be multiplied",
        "node_id": "enc1.fc",
        "frames": [
            {"file": "/opt/venv/lib/python3.12/site-packages/torch/nn/modules/linear.py", "line": 116},
            {"file": "/work/fixture/models/net.py", "line": 40},
            {"file": "/work/fixture/models/blocks.py", "line": forward_line},
        ],
    }
    payload = _payload()
    payload["trace_result"] = envelope
    data = _project(client, payload)

    assert data["traced"] is True
    error = data["trace_error"]
    assert error["type"] == "RuntimeError"
    assert error["node_id"] == "enc1.fc"
    assert error["file"].endswith("models/blocks.py")
    assert error["line"] == forward_line


# ── root failure is a request failure; dependency failure is a diagnosis ─────────


def test_root_syntax_error_is_400_located(client: TestClient):
    """The root is the thing being projected: if IT does not parse there is no
    graph at all, so the refusal is a 400 that says where the syntax broke."""
    broken_root = (
        "class Net(nn.Module):\n"
        "    def __init__(self, width: int = 8):\n"
        "        self.inp = nn.Linear((width, width)\n"
    )
    response = client.post(
        "/api/ide/multifile/project", json=_payload(overrides={ROOT: broken_root})
    )
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert detail["error"] == "syntax_error"
    assert isinstance(detail["line"], int) and detail["line"] >= 1


def test_root_class_missing_is_404(client: TestClient):
    response = client.post("/api/ide/multifile/project", json=_payload(cls="MissingNet"))
    assert response.status_code == 404, response.text
    assert "MissingNet" in response.json()["detail"]


def test_dependency_syntax_error_isolated(client: TestClient):
    """One broken dependency is a diagnosis, not a request failure: the response
    still renders, names the broken file AND the nodes it owns, and keeps every
    resolvable sibling expanded."""
    broken_heads = (
        "import torch.nn as nn\n\n\n"
        "class Block(nn.Module):\n"
        "    def __init__(self, width: int = 8)\n"
    )
    data = _project(client, _payload(overrides={"models/heads.py": broken_heads}))
    ids = _ids(data)
    view_data = data["document"]["view"]["layout"]["node_data"]

    # Both classes that live behind the broken file stay opaque and say why.
    syntax = [d for d in data["diagnostics"] if d["code"] == "DEPENDENCY_SYNTAX_ERROR"]
    assert syntax, "a broken dependency must surface a DEPENDENCY_SYNTAX_ERROR"
    assert all(d["severity"] == "error" for d in syntax)
    assert any(d["file"] == "models/heads.py" for d in syntax)
    assert {d.get("node") for d in syntax if d.get("node")} == {"head", "wide"}
    for stalled in ("head", "wide"):
        assert _node(data, stalled)["params"]["_opaque"] is True
        assert not view_data[stalled].get("isGroup")
    assert "head.proj" not in ids and "wide.narrow" not in ids

    # ...while the parts that do not touch models/heads.py are untouched.
    assert view_data["enc1"]["isGroup"] is True
    assert {"enc1.fc", "enc1.act", "mix.inner.fc"} <= ids

    # The broken file is still a dependency: it was supplied, read, and its
    # digest is what the projection was computed against.
    by_file = {entry["file"]: entry["digest"] for entry in data["dependencies"]}
    assert by_file["models/heads.py"] == _sha(broken_heads)


def test_unrelated_missing_dotted_base_cannot_break_valid_root(client: TestClient):
    baseline = _project(client)
    payload = _payload()
    payload["sources"].extend([
        _entry("net.py", "import blocks\nclass Net(blocks.Missing):\n    pass\n"),
        _entry("blocks.py", "class Present:\n    pass\n"),
    ])
    data = _project(client, payload)
    # The workspace-wide inheritance scan must not let an unrelated broken
    # model erase the valid root's graph, provenance, or diagnostics.
    before = baseline["document"]["semantic"]["metadata"]["reviewClass"].pop("dependencyDigests")
    after = data["document"]["semantic"]["metadata"]["reviewClass"].pop("dependencyDigests")
    assert after == {**before, **{entry["file"]: entry["digest"] for entry in payload["sources"][-2:]}}
    assert data["document"]["semantic"] == baseline["document"]["semantic"]
    assert data["spanmap"] == baseline["spanmap"]
    assert data["diagnostics"] == baseline["diagnostics"]
    assert data["dependencies"] == baseline["dependencies"]


@pytest.mark.parametrize(
    "sources, code, severity",
    [
        pytest.param(
            {"blocks.py": "class Present: ...\n"},
            multifile.UNRESOLVED_IMPORT, "warning", id="missing-class",
        ),
        pytest.param(
            {"blocks.py": "from left import Missing\nfrom right import Missing\n"},
            multifile.AMBIGUOUS_BINDING, "warning", id="ambiguous-binding",
        ),
        pytest.param(
            {"blocks.py": "from other import Missing\n", "other.py": "from blocks import Missing\n"},
            multifile.CYCLE_BOUNDARY, "warning", id="reexport-cycle",
        ),
        pytest.param(
            {"blocks.py": "class Missing(:\n"},
            multifile.DEPENDENCY_SYNTAX_ERROR, "error", id="dependency-syntax",
        ),
    ],
)
def test_dotted_base_failure_preserves_own_members_and_diagnostic(
    client: TestClient, sources: dict, code: str, severity: str
):
    source = (
        "import torch.nn as nn\nimport blocks\n"
        "class Net(blocks.Missing, nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(8, 8)\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n"
    )
    payload = _payload(overrides={ROOT: source})
    payload["sources"] = [_entry(file, text) for file, text in {ROOT: source, **sources}.items()]
    data = _project(client, payload)
    # Unsupported inheritance must neither erase the class's own layer nor
    # fabricate inherited members. The refusal names the failing base.
    assert _ids(data) == {"input", "fc", "output"}
    assert _node(data, "fc")["op"] == "Linear"
    diagnostic = next(d for d in data["diagnostics"] if "base `blocks.Missing`" in d["detail"])
    assert diagnostic["code"] == code
    assert diagnostic["severity"] == severity
    assert diagnostic["file"] == ROOT
    assert diagnostic["line"] == 3
    assert "blocks" in diagnostic["detail"]
    assert {entry["file"]: entry["digest"] for entry in data["dependencies"]} == {
        file: _sha(text) for file, text in {ROOT: source, **sources}.items()
    }


def test_a_framework_class_without_a_registry_spec_is_not_reported_as_unavailable(
    client: TestClient,
):
    """`nn.TransformerEncoder` is torch itself, not a package the user forgot to supply.

    It has no op-registry spec — nothing infers a shape THROUGH it — so the static
    tier keeps it as one opaque card. That is a limit of static projection, not a
    fault in the user's workspace, and the card must not carry a warning that reads
    as "your torch.nn is unavailable". This is the same misreading the `_SOURCE_OPS`
    branch was added to stop for `nn.Parameter`.
    """
    source = '''import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 32)
        layer = nn.TransformerEncoderLayer(32, 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(32, 16)

    def forward(self, tokens):
        return self.head(self.encoder(self.embed(tokens)))
'''
    body = _project(
        client,
        _payload({"models/net.py": source}, root_file="models/net.py", cls="Net"),
    )
    framework = [
        entry
        for entry in body["diagnostics"]
        if entry.get("node") == "encoder"
    ]
    assert framework, "the opaque framework class must still be explained, not silent"
    entry = framework[0]

    # The resolver already classifies torch.nn as informational; the projection used
    # to discard that and re-stamp every opaque card as a warning.
    assert entry["severity"] == "info", entry

    # And it must not describe the framework as something missing from the workspace.
    assert "external to the workspace" not in entry["detail"], entry["detail"]
    assert "torch" in entry["detail"]

    # Registry ops are still silent: only the class with no spec is explained.
    assert {e.get("node") for e in body["diagnostics"]} & {"embed", "head"} == set()


def _tiny_graph(io):
    """One real Linear model through the same builder the tracer uses."""
    import torch.nn as nn
    from services.ingest.introspect import build_tree
    from services.ingest.normalise import build_graph

    model = nn.Sequential()
    model.add_module("l1", nn.Linear(8, 4))
    intro = build_tree(model, 3)
    entry = (io.get("") or {}).get("in")
    return build_graph("Net", intro, io, {"l1": 1}, set(), {"l1"}, entry)[0]


def test_output_terminal_reports_the_measured_output_shape():
    """The Output card answers "what does this model return", and could not.

    It was built with no shape field at all, so however the run went it stayed blank
    while every real layer carried an observed shape. The tracer records the root module
    under the empty name, so the model's own return shape was already collected and
    simply never read.
    """
    graph = _tiny_graph({"": {"in": [2, 8], "out": [2, 4]}, "l1": {"in": [2, 8], "out": [2, 4]}})
    params = next(n for n in graph["nodes"] if n["id"] == "output_node")["data"]["parameters"]
    assert params["shape"] == "2,4", "the card's own field, spelled like the Input terminal's"
    assert params["_out_shape"] == "2x4", "the layer spelling the evidence and overlay paths read"
    assert params["_shape_provenance"] == "observed"


def test_output_terminal_stays_blank_when_nothing_measured_the_model():
    """A static graph must not invent a return shape."""
    graph = _tiny_graph({"l1": {"in": None, "out": None}})
    params = next(n for n in graph["nodes"] if n["id"] == "output_node")["data"]["parameters"]
    assert "shape" not in params
    assert "_out_shape" not in params


@pytest.mark.parametrize("observed", [True, False])
def test_traced_framework_warning_distinguishes_static_definition_from_boundary_evidence(client, observed):
    """F3: a bounded Transformer trace must not invite an empty expansion or another Validate."""
    import torch.nn as nn
    from services.ingest.introspect import build_tree
    from services.ingest.normalise import build_graph

    source = '''import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(8, 2, batch_first=True), 2)
    def forward(self, x):
        return self.encoder(x)
'''
    model = nn.Module()
    model.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(8, 2, batch_first=True), 2)
    io = {p: {"in": [1, 10, 8], "out": [1, 10, 8]}
          for p in ("encoder", "encoder.layers.0")} if observed else {}
    graph = build_graph("Net", build_tree(model, 2), io, {}, set(), set(io), [1, 10, 8])[0]
    payload = _payload({ROOT: source})
    static = _project(client, payload)
    static_diagnostic = next(d for d in static["diagnostics"] if d.get("node") == "encoder")
    payload["trace_result"] = {"protocol_version": 1, "result": {"graph": graph, "diagnostics": {}},
                               "spans": {}, "error": None}
    traced = _project(client, payload)
    diagnostic = next(d for d in traced["diagnostics"] if d.get("node") == "encoder")
    assert diagnostic["code"] == static_diagnostic["code"] == "EXTERNAL_PACKAGE"
    assert diagnostic["severity"] == "info"
    assert "Run Validate" not in diagnostic["detail"]
    assert ("Boundary shapes were measured" in diagnostic["detail"]) is observed
    if not observed:
        assert diagnostic == static_diagnostic
    layout = traced["document"]["view"]["layout"]["node_data"]
    block = layout["encoder.layers.0"]
    assert block["moduleCapture"] == {"status": "depth-limited", "maxDepth": 2}
    assert block["moduleLayerCount"] == 0
    assert block["repeatCount"] == 2
    assert block["parentId"] == "encoder"
    assert traced["spanmap"]["encoder"] == static["spanmap"]["encoder"]
    assert not any(n["node_id"].startswith("encoder.layers.0.") for n in traced["document"]["semantic"]["nodes"])
