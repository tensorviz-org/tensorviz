"""Reuse source analysis only after revalidating the entire supplied inventory."""
import ast
from copy import deepcopy
from unittest.mock import patch

import pytest

from services.semantic import source_context as source
from tests.source_review.test_source_context import payload, client


@pytest.fixture(autouse=True)
def clean_cache():
    cache = getattr(source, "_SOURCE_CACHE", None)
    if cache:
        cache.clear()
    yield
    if cache:
        cache.clear()


def test_reuses_exact_inventory_across_pages_and_file_order():
    request = payload()
    with patch.object(source.ast, "parse", wraps=ast.parse) as parse:
        first = source.query_source_context(request)
        reverse = deepcopy(request)
        reverse["sources"].reverse()
        assert source.query_source_context(reverse) == first
        following = deepcopy(request)
        following["query"].update(offset=1, limit=2)
        assert source.query_source_context(following)["items"] == first["items"][1:3]
        assert parse.call_count == 2, "Paging must not repeatedly parse unchanged source."


def test_changed_configuration_reuses_only_unchanged_file_parsing():
    original = payload()
    before = source.query_source_context(original)
    changed = payload({"config.py": "WIDTH = 16\n", "model.py": original["sources"][1]["source"]})
    with patch.object(source.ast, "parse", wraps=ast.parse) as parse:
        after = source.query_source_context(changed)
        assert parse.call_count == 1, "An edit must reparse its file and rebuild relationships, while reusing unchanged ASTs."
    assert after["indexId"] != before["indexId"]
    assert any(item.get("literalSource") == "16" for item in after["items"])
    assert any(item.get("kind") == "call" and item["anchor"]["file"] == "model.py" for item in after["items"])
    changed["query"]["indexId"] = before["indexId"]
    with pytest.raises(source.SourceContextRefused, match="source_index_changed"):
        source.query_source_context(changed)


def test_added_removed_and_renamed_dependencies_rebuild_resolution():
    model = "from config import WIDTH\nresult = transform(WIDTH)\n"
    missing = payload({"model.py": model}, file="model.py")
    before = source.query_source_context(missing)
    assert any(item["reason"] == "unresolved_import" for item in before["diagnostics"])
    supplied = payload({"config.py": "WIDTH = 12\n", "model.py": model})
    resolved = source.query_source_context(supplied)
    assert any(item.get("relation") == "import" for item in resolved["items"])
    assert source.query_source_context(missing) == before
    renamed = payload({"renamed.py": "WIDTH = 12\n", "model.py": model}, file="renamed.py")
    with patch.object(source.ast, "parse", wraps=ast.parse) as parse:
        page = source.query_source_context(renamed)
        assert parse.call_count == 1, "A new path needs separate AST identities even when its text matches."
    assert page["indexId"] != resolved["indexId"]
    assert not any(item.get("kind") == "call" for item in page["items"])
    assert any(item["reason"] == "unresolved_import" for item in page["diagnostics"])


def test_dynamic_dependencies_stay_unresolved_after_neighbor_changes():
    dynamic = "WIDTH = 12\nexec('WIDTH = 99')\nresult = transform(WIDTH)\n"
    for neighbor in ["VALUE = 1\n", "VALUE = 2\n"]:
        request = payload({"config.py": dynamic, "other.py": neighbor})
        cached = source.query_source_context(request)
        cold_request = source.SourceContextRequest.model_validate(request)
        cold = source.SourceContextIndex(cold_request.sources).query(cold_request.query)
        assert cached == cold
        assert any(item["reason"] == "dynamic_namespace" for item in cached["diagnostics"])
        assert not any(item.get("kind") == "call" for item in cached["items"])


def test_identical_text_in_different_files_keeps_separate_scopes_and_edges():
    text = "WIDTH = 12\nresult = transform(WIDTH)\n"
    request = payload({"left.py": text, "right.py": text}, file="left.py", direction="downstream")
    parsed = source.SourceContextRequest.model_validate(request)
    expected = source.SourceContextIndex(parsed.sources).query(parsed.query)
    assert source.query_source_context(request) == expected
    assert any(item.get("kind") == "call" and item["anchor"]["file"] == "left.py" for item in expected["items"])


@pytest.mark.parametrize("mutation", ["text", "digest", "duplicate", "path", "too_large"])
def test_warm_cache_never_bypasses_source_validation(mutation):
    request = payload()
    expected = source.query_source_context(request)
    bad = deepcopy(request)
    if mutation == "text":
        bad["sources"][0]["source"] = "WIDTH = 999\n"
    elif mutation == "digest":
        bad["sources"][0]["digest"] = "0" * 64
    elif mutation == "duplicate":
        bad["sources"].append(bad["sources"][0])
    elif mutation == "path":
        bad["sources"][0]["file"] = "../config.py"
    else:
        bad["sources"][0]["source"] = "#" + "x" * 262144
    with pytest.raises(ValueError):
        source.query_source_context(bad)
    assert source.query_source_context(request) == expected


def test_results_and_caller_owned_models_cannot_mutate_cached_facts():
    request = source.SourceContextRequest.model_validate(payload())
    first = source.query_source_context(request)
    expected = deepcopy(first)
    first["items"][0]["anchor"]["file"] = "forged.py"
    first["diagnostics"].append({"reason": "forged"})
    request.sources[0].source = "WIDTH = 999\n"
    assert source.query_source_context(payload()) == expected


def test_schema_version_changes_cannot_reuse_prior_indexes(monkeypatch):
    first = source.query_source_context(payload())
    monkeypatch.setattr(source, "VERSION", source.VERSION + 1)
    with patch.object(source.ast, "parse", wraps=ast.parse) as parse:
        following = source.query_source_context(payload())
        assert parse.call_count == 2
    assert following["indexId"] != first["indexId"]
    assert following["version"] == first["version"] + 1


def test_expiry_releases_indexes_and_trees_before_the_next_lookup(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(source, "monotonic", lambda: clock[0])
    first = source.query_source_context(payload())
    clock[0] = source._CACHE_SECONDS + 1
    with patch.object(source.ast, "parse", wraps=ast.parse) as parse:
        assert source.query_source_context(payload()) == first
        assert parse.call_count == 2, "Expired source must be reconstructed even when its content is identical."


@pytest.mark.parametrize("budget", ["_CACHE_BYTES", "_CACHE_NODES"])
def test_retention_budgets_never_change_query_results_or_keep_oversized_entries(monkeypatch, budget):
    expected = source.query_source_context(payload())
    source._SOURCE_CACHE.clear()
    monkeypatch.setattr(source, budget, 1)
    for _ in range(2):
        with patch.object(source.ast, "parse", wraps=ast.parse) as parse:
            assert source.query_source_context(payload()) == expected
            assert parse.call_count == 2
        assert not source._SOURCE_CACHE.parsed and not source._SOURCE_CACHE.indexes


def test_lru_eviction_keeps_only_two_indexes_and_thirty_two_parsed_files():
    for width in range(40):
        source.query_source_context(payload({"config.py": f"WIDTH = {width}\n"}))
    assert len(source._SOURCE_CACHE.indexes) == 2
    assert len(source._SOURCE_CACHE.parsed) == 32
    with patch.object(source.ast, "parse", wraps=ast.parse) as parse:
        source.query_source_context(payload({"config.py": "WIDTH = 0\n"}))
        assert parse.call_count == 1


def test_reduced_analysis_limits_refuse_a_previous_warm_inventory(monkeypatch):
    source.query_source_context(payload())
    monkeypatch.setattr(source, "MAX_RECORDS", 1)
    with pytest.raises(source.SourceContextRefused, match="source_index_limit"):
        source.query_source_context(payload())


def test_failed_syntax_is_not_retained_as_a_valid_parse():
    request = payload({"config.py": "WIDTH = (\n"})
    with patch.object(source.ast, "parse", wraps=ast.parse) as parse:
        for _ in range(2):
            with pytest.raises(SyntaxError):
                source.query_source_context(request)
        assert parse.call_count == 2
    assert not source._SOURCE_CACHE.parsed and not source._SOURCE_CACHE.indexes


def test_http_pages_are_byte_identical_with_warm_indexes(client):
    request = payload()
    with patch.object(source.ast, "parse", wraps=ast.parse) as parse:
        first = client.post("/api/ide/source-context", json=request)
        warm = client.post("/api/ide/source-context", json=request)
        assert first.status_code == warm.status_code == 200
        assert first.content == warm.content
        assert first.headers["cache-control"] == "no-store"
        assert parse.call_count == 2
