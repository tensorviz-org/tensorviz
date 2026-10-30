"""Conditional source arithmetic must not turn Python names into observed facts."""
import hashlib

import pytest

from services.semantic.source_context import query_source_context, SourceContextRefused
from tests.api.test_ide_router import client


def request(sources=None, **query):
    sources = sources or {"config.py": "WIDTH = 12\nHEADS = 3\n", "model.py":
        "from config import WIDTH, HEADS\ninner = WIDTH * 4\na = linear(WIDTH, inner)\nb = reshape(WIDTH // HEADS, WIDTH % HEADS)\n"}
    return dict(sources=[dict(file=file, source=source, digest=hashlib.sha256(source.encode()).hexdigest())
                         for file, source in sources.items()], query=dict(file="config.py", symbol="WIDTH",
                         direction="downstream", depth=8, limit=100, assumeValue=16) | query)


def calls(page):
    return {item["name"]: item["configurationChange"] for item in page["items"]
            if item["kind"] == "call" and "configurationChange" in item}


def test_imported_width_change_calculates_arithmetic_without_claiming_execution():
    page = query_source_context(request())
    assert page["changeAnalysis"]["before"] == 12
    assert page["changeAnalysis"]["after"] == 16
    assert page["changeAnalysis"]["evidence"] == "conditional_source_arithmetic"
    assert page["evidence"] == "source_only"
    found = calls(page)
    assert [(arg["before"], arg["after"]) for arg in found["linear"]["arguments"]] == [(12, 16), (48, 64)]
    assert [(arg["before"], arg["after"]) for arg in found["reshape"]["arguments"]] == [(4, 5), (0, 1)]
    assert all(arg["status"] == "derived" for call in found.values() for arg in call["arguments"])


def test_argument_leaves_include_unchanged_and_unknown_dimensions_without_executing_them():
    source = "WIDTH = 12\nresult = reshape(x, (x.shape[0], WIDTH // 3, -1), other=compute())\n"
    change = calls(query_source_context(request({"config.py": source})))["reshape"]
    assert change["omittedArgumentValues"] == 0
    assert [(row["path"], row["status"], row["before"], row["after"]) for row in change["argumentValues"]] == [
        (["args", 0], "unknown", None, None),
        (["args", 1, "items", 0], "unknown", None, None),
        (["args", 1, "items", 1], "derived", 4, 5),
        (["args", 1, "items", 2], "derived", -1, -1),
        (["kwargs", "other"], "unknown", None, None),
    ]
    assert change["arguments"][0]["status"] == "unknown", "Existing whole-tuple scalar semantics must not change."


def test_argument_leaf_bound_never_presents_a_truncated_tuple_as_complete():
    source = "WIDTH = 12\nresult = f((WIDTH," + ",".join(["1"] * 39) + "))\n"
    change = calls(query_source_context(request({"config.py": source})))["f"]
    assert len(change["argumentValues"]) == 32 and change["omittedArgumentValues"] == 8
    assert change["argumentValues"][0]["after"] == 16


@pytest.mark.parametrize("expression", ["WIDTH / 3", "compute(WIDTH)", "[WIDTH]", "WIDTH ** 100000", "WIDTH // 0", "WIDTH * 9007199254740991"])
def test_unsupported_or_unbounded_arithmetic_remains_unknown(expression):
    result = calls(query_source_context(request({"config.py": f"WIDTH = 12\nresult = consume({expression})\n"})))["consume"]
    assert result["arguments"][0]["status"] == "unknown"
    assert result["arguments"][0]["after"] is None


@pytest.mark.parametrize("source", [
    "WIDTH = compute()\nresult = f(WIDTH)\n",
    "if flag:\n    WIDTH = 12\nresult = f(WIDTH)\n",
    "WIDTH = 12\nWIDTH = 13\nresult = f(WIDTH)\n",
    "WIDTH = True\nresult = f(WIDTH)\n",
    "WIDTH = 12.0\nresult = f(WIDTH)\n",
    "WIDTH = 12\nexec('WIDTH = 99')\nresult = f(WIDTH)\n",
])
def test_a_change_requires_one_direct_unambiguous_integer_definition(source):
    with pytest.raises(SourceContextRefused, match="configuration_change_unavailable"):
        query_source_context(request({"config.py": source}))


def test_unknown_defaults_attributes_and_conditional_dependencies_stay_unknown():
    source = """WIDTH = 12
if flag:
    HIDDEN = WIDTH
class Model:
    def __init__(self, width=WIDTH):
        self.width = width
    def forward(self, x):
        return reshape(x, self.width, HIDDEN)
"""
    result = calls(query_source_context(request({"config.py": source})))["reshape"]
    assert len(result["arguments"]) == 2
    assert all(arg["status"] == "unknown" for arg in result["arguments"])


def test_shadowed_names_do_not_inherit_the_outer_configuration_change():
    source = "WIDTH = 12\ndef local(WIDTH):\n    return shadow(WIDTH)\nresult = outer(WIDTH)\n"
    result = calls(query_source_context(request({"config.py": source}, line=1)))
    assert "shadow" not in result
    assert result["outer"]["arguments"][0]["after"] == 16


def test_pages_preserve_effects_and_changes_do_not_mutate_a_cached_index():
    original = request()
    full = query_source_context(original)
    items, offset = [], 0
    while True:
        page = query_source_context(request(limit=2, offset=offset))
        assert page["changeAnalysis"] == full["changeAnalysis"]
        items.extend(page["items"])
        if page["nextOffset"] is None: break
        offset = page["nextOffset"]
    assert items == full["items"]
    assert calls(query_source_context(request(assumeValue=24)))["linear"]["arguments"][0]["after"] == 24
    assert query_source_context(original) == full
    plain = request(); plain["query"].pop("assumeValue")
    clean = query_source_context(plain)
    assert "changeAnalysis" not in clean
    assert all("configurationChange" not in item for item in clean["items"])
    changed = request({"config.py": "WIDTH = 13\n"}, indexId=full["indexId"])
    with pytest.raises(SourceContextRefused, match="source_index_changed"):
        query_source_context(changed)


def test_query_never_executes_source_or_its_changed_expression(tmp_path):
    marker = tmp_path / "executed"
    source = f"from pathlib import Path\nWIDTH = 12\nPath({str(marker)!r}).write_text(str(WIDTH))\n"
    page = query_source_context(request({"config.py": source}))
    assert page["changeAnalysis"]["after"] == 16
    assert not marker.exists()


@pytest.mark.parametrize("expression,expected", [("-WIDTH", -16), ("+WIDTH", 16), ("5 - WIDTH", -11), ("-WIDTH // 3", -6), ("-WIDTH % 3", 2)])
def test_integer_rules_match_python_sign_and_floor_semantics(expression, expected):
    result = calls(query_source_context(request({"config.py": f"WIDTH = 12\nresult = f({expression})\n"})))["f"]
    assert result["arguments"][0]["after"] == expected


def test_argument_limits_and_expansion_are_explicit():
    source = "WIDTH = 12\nresult = f(" + ", ".join(["WIDTH"] * 20) + ", **options)\n"
    effect = calls(query_source_context(request({"config.py": source})))["f"]
    assert len(effect["arguments"]) == 16
    assert effect["omittedArguments"] == 4
    assert effect["hasExpansion"] is True


def test_changed_definition_and_arithmetic_are_checked_through_actual_host_route(client):
    good = client.post("/api/ide/source-context", json=request())
    assert good.status_code == 200
    assert calls(good.json())["linear"]["arguments"][1]["after"] == 64
    bad = client.post("/api/ide/source-context", json=request({"config.py": "WIDTH = get_secret()\n"}))
    assert bad.status_code == 422
    assert bad.json()["detail"]["reason"] == "configuration_change_unavailable"
    assert "get_secret" not in bad.text


def test_frozen_change_pages_match_the_actual_parser():
    import json
    from pathlib import Path
    oracle = json.loads((Path(__file__).resolve().parents[3] / "tests/fixtures/contracts/source-change-oracle.json").read_text())
    for case in oracle["queries"]:
        assert query_source_context(dict(sources=oracle["sources"], query=case["query"])) == case["page"]
    shapes = json.loads((Path(__file__).resolve().parents[3] / "tests/fixtures/contracts/source-shape-oracle.json").read_text())
    for fixture in shapes["cases"]:
        for case in fixture["queries"]:
            assert query_source_context(dict(sources=fixture["sources"], query=case["query"])) == case["page"]


def test_large_expression_analysis_is_bounded_and_page_independent():
    expression = "1"
    for _ in range(12): expression = f"({expression}+{expression})"
    sources = {"config.py": f"WIDTH = 12\nresult = consume(WIDTH + {expression})\n"}
    full = query_source_context(request(sources))
    effect = calls(full)["consume"]["arguments"][0]
    assert effect["status"] == "unknown" and effect["reason"] == "analysis_limit"
    items, offset = [], 0
    while True:
        page = query_source_context(request(sources, limit=2, offset=offset))
        items.extend(page["items"])
        if page["nextOffset"] is None: break
        offset = page["nextOffset"]
    assert items == full["items"]
