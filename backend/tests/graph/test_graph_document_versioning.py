"""Tests for document schema versioning and the migration seam.

The behaviour these pin down: a document format has to be able to CHANGE. Before this,
``schemaVersion`` was ``Literal[2]``, so pydantic rejected every other value and a version 3
document could not be constructed at all — and because the reader only special-cased
``== 2``, anything else fell through to the legacy ModelGraph converter and came back as a
document with zero semantic nodes. An unreadable document became a silently empty model,
which every downstream consumer then trusted.
"""

from __future__ import annotations

import pytest

from models.architecture_models import (
    CURRENT_SCHEMA_VERSION,
    GraphDocumentV2,
)
from services import graph_document
from services.graph_document import (
    DocumentVersionError,
    graph_document_v2,
    migrate_document,
)


def _v2_document(version: int = CURRENT_SCHEMA_VERSION) -> dict:
    return {
        "schemaVersion": version,
        "semantic": {
            "graph_id": "g1",
            "name": "m",
            "revision": 0,
            "inputs": [{"name": "input_node", "shape": "1,16"}],
            "nodes": [
                {"node_id": "input_node", "kind": "input", "op": "Input",
                 "params": {"shape": "1,16"}, "in_ports": [], "out_ports": ["y"], "metadata": {}},
                {"node_id": "fc", "kind": "layer", "op": "Linear",
                 "params": {"in_features": 16, "out_features": 8},
                 "in_ports": ["x"], "out_ports": ["y"], "metadata": {}},
            ],
            "edges": [{"edge_id": "e1", "source_node_id": "input_node", "source_port": "y",
                       "target_node_id": "fc", "target_port": "x", "metadata": {}}],
            "modules": [], "operations": [], "metadata": {},
        },
        "view": {"graph_id": "g1", "node_positions": {}, "collapsed_modules": [],
                 "selection": [], "layout": {}},
    }


def _legacy_model_graph() -> dict:
    return {
        "nodes": [
            {"id": "input_node", "type": "inputNode", "position": {"x": 0, "y": 0},
             "data": {"layerType": {"id": "input", "name": "Input", "parameters": []},
                      "parameters": {"shape": "1,16"}}},
            {"id": "fc", "type": "layerNode", "position": {"x": 300, "y": 0},
             "data": {"layerType": {"id": "linear", "name": "Linear", "parameters": []},
                      "parameters": {"in_features": 16, "out_features": 8}}},
        ],
        "edges": [{"id": "e1", "source": "input_node", "target": "fc"}],
    }


# ── the change that makes a format change possible ───────────────────────────
def test_a_future_schema_version_is_constructible_at_all():
    """As Literal[2] this raised a ValidationError, so there was no way to author the next
    format version even in a test — the type made the format permanently frozen."""
    document = GraphDocumentV2.model_validate(_v2_document(version=CURRENT_SCHEMA_VERSION + 1))
    assert document.schemaVersion == CURRENT_SCHEMA_VERSION + 1


def test_a_newer_document_is_refused_instead_of_silently_read_as_empty():
    """THE bug. A version the reader does not know used to fall through to the legacy
    ModelGraph converter, which looks for `nodes`/`edges` at the top level, finds neither in
    a versioned document, and returns a perfectly valid document containing NOTHING. Every
    consumer downstream (ai/orchestrator builds its known-node set from semantic.nodes) then
    treats the model as empty and rejects operations against it.
    """
    future = _v2_document(version=CURRENT_SCHEMA_VERSION + 1)

    # what the old code path would have produced, for contrast
    silently_empty = graph_document._legacy_model_graph_to_v2(
        future, graph_id="g1", name="m", revision=0
    )
    assert silently_empty.semantic.nodes == [], "guard assumption: legacy conversion loses everything"

    with pytest.raises(DocumentVersionError) as excinfo:
        graph_document_v2(future, graph_id="g1", name="m", revision=0)
    assert str(CURRENT_SCHEMA_VERSION + 1) in str(excinfo.value)


# ── behaviour that must not regress ──────────────────────────────────────────
def test_current_version_documents_still_load_and_are_restamped():
    document = graph_document_v2(_v2_document(), graph_id="g9", name="renamed", revision=7)
    assert document.schemaVersion == CURRENT_SCHEMA_VERSION
    assert [n.node_id for n in document.semantic.nodes] == ["input_node", "fc"]
    assert document.semantic.graph_id == "g9"
    assert document.semantic.name == "renamed"
    assert document.semantic.revision == 7
    assert document.view.graph_id == "g9"


def test_a_legacy_model_graph_with_no_version_still_converts():
    """Unversioned payloads are a different case from an unknown version and must keep
    working — this is what every pre-V2 stored revision looks like."""
    document = graph_document_v2(_legacy_model_graph(), graph_id="g1", name="m", revision=0)
    assert document.schemaVersion == CURRENT_SCHEMA_VERSION
    assert len(document.semantic.nodes) == 2
    assert document.semantic.metadata["migration"]["sourceSchema"] == "ModelGraph"


# ── the migration mechanism ──────────────────────────────────────────────────
def test_the_migration_chain_upgrades_an_older_document(monkeypatch):
    """The registry is empty today (version 2 is the only format ever written), so exercise
    the mechanism with a temporary step. This is what proves the seam works BEFORE there is
    real user data that depends on it."""
    calls = []

    def one_to_two(raw: dict) -> dict:
        calls.append(raw["schemaVersion"])
        out = dict(raw)
        out["schemaVersion"] = 2
        out["semantic"] = dict(out["semantic"])
        out["semantic"]["name"] = "upgraded"
        return out

    monkeypatch.setitem(graph_document.SCHEMA_MIGRATIONS, 1, one_to_two)

    migrated = migrate_document(_v2_document(version=1))
    assert calls == [1], "the v1 -> v2 step must have run exactly once"
    assert migrated["schemaVersion"] == 2
    assert migrated["semantic"]["name"] == "upgraded"


def test_a_missing_migration_step_fails_loudly(monkeypatch):
    """A gap in the chain must be an error, not a document that quietly keeps its old
    shape and then fails validation somewhere far away."""
    monkeypatch.setattr(graph_document, "CURRENT_SCHEMA_VERSION", 3)
    with pytest.raises(DocumentVersionError, match="no migration registered"):
        migrate_document(_v2_document(version=2))


def test_a_migration_that_does_not_advance_the_version_is_rejected(monkeypatch):
    """Guards against an infinite loop: a step that forgets to stamp the new version would
    otherwise be retried forever."""
    monkeypatch.setattr(graph_document, "CURRENT_SCHEMA_VERSION", 3)
    monkeypatch.setitem(graph_document.SCHEMA_MIGRATIONS, 2, lambda raw: dict(raw))
    with pytest.raises(DocumentVersionError, match="must set schemaVersion"):
        migrate_document(_v2_document(version=2))


def test_a_non_integer_version_is_rejected():
    with pytest.raises(DocumentVersionError, match="must be an int"):
        migrate_document({"schemaVersion": "2"})
