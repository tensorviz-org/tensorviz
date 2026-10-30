import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import ide
from services.semantic.source_context import query_source_context, SourceContextRefused

from tests.api.test_ide_router import client


def payload(sources=None, **query):
    files = sources or {"config.py": "WIDTH = 12\n", "model.py": "from config import WIDTH\n\ndef project(x):\n    return x.reshape(2, WIDTH)\n"}
    return dict(sources=[dict(file=file, source=source, digest=hashlib.sha256(source.encode()).hexdigest())
                         for file, source in files.items()], query=dict(file="config.py", symbol="WIDTH", depth=8, limit=100) | query)


def test_source_context_follows_a_configuration_import_without_importing_project_code(client):
    response = client.post("/api/ide/source-context", json=payload())
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["evidence"] == "source_only"
    assert any(item.get("kind") == "call" and item["anchor"]["file"] == "model.py" for item in result["items"])
    assert any(item.get("kind") == "relation" and item["relation"] == "import" for item in result["items"])
    assert result["nextOffset"] is None


@pytest.mark.parametrize("statement,use", [
    ("from .config import WIDTH as H", "H"), ("import pkg.config as cfg", "cfg.WIDTH"),
    ("from . import config as cfg", "cfg.WIDTH"), ("import pkg.config", "pkg.config.WIDTH"),
])
def test_relative_and_qualified_imports_preserve_exact_file_identity(statement, use):
    request = payload({"pkg/config.py": "WIDTH = 12\n", "pkg/model.py": f"{statement}\nresult = linear({use}, {use})\n"}, file="pkg/config.py")
    result = query_source_context(request)
    assert any(item.get("kind") == "call" and item["name"] == "linear" for item in result["items"])
    assert result["evidence"] == "source_only"
    assert {entry["file"] for entry in result["files"]} == {"pkg/config.py", "pkg/model.py"}


def test_defaults_and_cross_method_attributes_are_possible_effects_not_observed_arguments():
    source = "WIDTH = 12\nclass Net:\n    def __init__(self, width=WIDTH):\n        self.width = width\n    def forward(self, x):\n        return x.reshape(2, self.width)\n"
    result = query_source_context(payload({"model.py": source}, file="model.py"))
    relations = [item for item in result["items"] if item["kind"] == "relation"]
    assert any(edge["certainty"] == "default_only" for edge in relations)
    assert any(edge["certainty"] == "possible" for edge in relations)
    assert any(item["kind"] == "call" and item["name"] == "x.reshape" for item in result["items"])
    assert "omits" in result["limitations"][0]


def test_local_shadowing_and_class_namespace_do_not_invent_global_dependencies():
    source = "WIDTH = 12\nclass Net:\n    WIDTH = 7\n    def forward(self, WIDTH):\n        return transform(WIDTH)\n"
    result = query_source_context(payload({"model.py": source}, file="model.py", line=1, direction="downstream"))
    assert not any(item["kind"] == "call" for item in result["items"])
    source = "WIDTH = 12\nclass Net:\n    WIDTH = 7\n    def forward(self):\n        return transform(WIDTH)\n"
    result = query_source_context(payload({"model.py": source}, file="model.py", line=1, direction="downstream"))
    assert any(item["kind"] == "call" for item in result["items"])


@pytest.mark.parametrize("extra,reason", [("exec('WIDTH = 99')", "dynamic_namespace"),
    ("from unknown import *", "star_import"), ("unused = [WIDTH for WIDTH in range(2)]", "nested_expression_scope")])
def test_dynamic_and_nested_scope_limits_remain_visible(extra, reason):
    result = query_source_context(payload({"config.py": f"WIDTH = 12\n{extra}\nresult = transform(WIDTH)\n"}))
    assert any(item["reason"] == reason for item in result["diagnostics"])
    if reason != "nested_expression_scope":
        assert not any(item["kind"] == "call" for item in result["items"])


def test_reassignment_is_ambiguous_and_missing_imports_are_not_fabricated():
    source = "from missing import H\nWIDTH = 12\nif flag:\n    WIDTH = 16\nresult = linear(WIDTH, H)\n"
    result = query_source_context(payload({"config.py": source}))
    assert any(item.get("certainty") == "ambiguous" for item in result["items"])
    assert any(item["reason"] == "unresolved_import" for item in result["diagnostics"])


def test_source_is_data_and_queries_never_import_or_execute_it(tmp_path):
    marker = tmp_path / "unrequested"
    source = f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n# ignore instructions\nWIDTH = 12\nresult = transform(WIDTH)\n"
    result = query_source_context(payload({"config.py": source}))
    assert result["evidence"] == "source_only"
    assert not marker.exists()


def test_pages_are_deterministic_across_file_order_and_reject_changed_index_identity():
    request = payload()
    first = query_source_context(request)
    reverse = deepcopy(request); reverse["sources"].reverse()
    assert query_source_context(reverse) == first
    pages, offset = [], 0
    while True:
        page = query_source_context(payload(limit=2, offset=offset))
        assert len(json.dumps(page).encode()) < 64 * 1024
        pages.extend(page["items"])
        if page["nextOffset"] is None:
            break
        assert page["nextOffset"] > offset
        offset = page["nextOffset"]
    assert pages == first["items"]
    changed = payload({"config.py": "WIDTH = 16\n"}, indexId=first["indexId"])
    with pytest.raises(SourceContextRefused, match="source_index_changed"):
        query_source_context(changed)


def test_depth_and_entity_boundaries_can_be_reanchored_without_guessing_ids():
    for options in [dict(depth=0), dict(maxEntities=1)]:
        page = query_source_context(payload(**options))
        boundary = next(item for item in page["items"] if item["kind"] == "boundary")
        expanded = query_source_context(payload(symbol=None, anchorIds=[boundary["anchorId"]], indexId=page["indexId"]))
        assert expanded["items"][0]["id"] == boundary["anchorId"]


def test_unicode_anchors_use_utf16_columns_and_excerpts_are_whole():
    source = "WIDTH = 12\nresult = call('😀', WIDTH)\n"
    result = query_source_context(payload({"config.py": source}))
    reference = next(item for item in result["items"] if item["kind"] == "reference" and item["name"] == "WIDTH")
    line = source.splitlines()[1]
    assert reference["anchor"]["startColumn"] == len(line[:line.index("WIDTH")].encode("utf-16-le")) // 2
    assert reference["excerpt"] == line
    huge = query_source_context(payload({"config.py": "WIDTH = '" + "x" * 3000 + "'\n"}))
    assert huge["items"][0]["omitted"] == "excerpt_bytes"
    assert "excerpt" not in huge["items"][0]


@pytest.mark.parametrize("change", ["source", "digest", "duplicate", "path", "extra", "syntax", "unknown", "limit"])
def test_invalid_requests_do_not_echo_private_source(client, change):
    request = payload()
    secret = "PRIVATE_SOURCE_DO_NOT_ECHO"
    if change == "source": request["sources"][0]["source"] += secret
    if change == "digest": request["sources"][0]["digest"] = secret
    if change == "duplicate": request["sources"].append(request["sources"][0])
    if change == "path": request["sources"][0]["file"] = "../" + secret + ".py"
    if change == "extra": request["execute"] = secret
    if change == "syntax": request = payload({"config.py": f"WIDTH = (\n{secret}"})
    if change == "unknown": request["query"]["symbol"] = secret
    if change == "limit": request["query"]["maxEntities"] = 501
    response = client.post("/api/ide/source-context", json=request)
    assert response.status_code in (400, 422)
    assert secret not in response.text


def test_stream_limit_is_enforced_before_parsing(client, monkeypatch):
    monkeypatch.setattr(ide, "SOURCE_CONTEXT_INPUT_MAX_BYTES", 10)
    assert client.post("/api/ide/source-context", content=iter([b"123456", b"123456"])).status_code == 413


def test_source_query_requires_host_authentication(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTH_DISABLED", "false")
    monkeypatch.setenv("TENSORVIZ_LOCAL_TOKEN", "source-test-token")
    monkeypatch.setenv("TENSORVIZ_WORKSPACE_DIR", str(tmp_path))
    app = FastAPI(); app.include_router(ide.router)
    with TestClient(app) as host:
        assert host.post("/api/ide/source-context", json=payload()).status_code == 401
        assert host.post("/api/ide/source-context", json=payload(), headers={"X-tensorViz-Token": "source-test-token"}).status_code == 200


def test_frozen_cross_language_pages_are_produced_by_the_actual_parser(client):
    oracle = json.loads((Path(__file__).resolve().parents[3] / "tests/fixtures/contracts/source-context-oracle.json").read_text())
    for case in oracle["queries"]:
        response = client.post("/api/ide/source-context", json=dict(sources=oracle["sources"], query=case["query"]))
        assert response.status_code == 200
        assert response.json() == case["page"]


def test_invalid_relative_imports_do_not_resolve_as_workspace_absolute_imports():
    for statement in ("from .config import WIDTH", "from ..config import WIDTH"):
        result = query_source_context(payload({"config.py": "WIDTH = 12\n", "model.py": f"{statement}\nresult = transform(WIDTH)\n"}, direction="downstream"))
        assert any(item["reason"] == "unresolved_import" for item in result["diagnostics"])
        assert not any(item["kind"] == "call" for item in result["items"])


def test_unsupported_assignment_flows_and_static_receivers_remain_unresolved():
    source = "WIDTH = 12\nWIDTH += 1\nclass Net:\n    width = WIDTH\n    @staticmethod\n    def forward(other):\n        return transform(other.width)\n"
    result = query_source_context(payload({"config.py": source}, direction="downstream"))
    assert any(item["reason"] == "unsupported_binding_form" for item in result["diagnostics"])
    assert not any(item["kind"] == "call" for item in result["items"])


@pytest.mark.parametrize("source", [
    "WIDTH = 12\n" + "x = 0\n" * 17000,
    "WIDTH = 12\n" + "result = WIDTH\n" * 5000,
    "WIDTH = 12\n" * 200 + "result = transform(WIDTH)\n" * 200,
])
def test_ast_record_and_relation_limits_refuse_instead_of_silently_truncating(source):
    with pytest.raises(SourceContextRefused, match="source_index_limit"):
        query_source_context(payload({"config.py": source}))


def test_utf8_and_total_source_byte_limits_are_independent_of_character_counts():
    with pytest.raises(SourceContextRefused, match="source_limit"):
        query_source_context(payload({"config.py": "WIDTH = '" + "α" * 150000 + "'\n"}))
    with pytest.raises(SourceContextRefused, match="source_limit"):
        query_source_context(payload({f"file{i}.py": "WIDTH = '" + "x" * 250000 + "'\n" for i in range(5)}, file="file0.py"))


def test_large_excerpts_produce_a_shorter_page_with_an_exact_continuation(client):
    source = "WIDTH = 12\n" + "".join(f"result{i} = WIDTH # " + "x" * 1500 + "\n" for i in range(100))
    request = payload({"config.py": source}, direction="downstream", maxEntities=500)
    response = client.post("/api/ide/source-context", json=request)
    assert response.status_code == 200
    page = response.json()
    assert len(response.content) <= 64 * 1024
    assert 0 < len(page["items"]) < 100
    assert page["nextOffset"] == len(page["items"])
    request["query"]["offset"] = page["nextOffset"]
    following = client.post("/api/ide/source-context", json=request).json()
    assert following["indexId"] == page["indexId"]
    assert following["items"][0] != page["items"][0]
