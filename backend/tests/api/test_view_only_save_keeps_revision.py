"""Moving a card must not consume a graph revision.

A revision numbers a change to the architecture. Charging one for a view-only save
had a sharp consequence: AI staleness is a revision comparison
(``ai/orchestrator.py`` and the commit gate), so nudging a single node while a
candidate was under review invalidated it, and Accept failed with
``canvas changed after the candidate was reviewed`` — the reviewed work discarded.
Reproduced in a browser by dragging one card with a candidate pending.

Fixing it at the commit gate was rejected: ``canonical_semantic_bytes`` includes
``revision`` and the store stamps it, so retargeting the CAS would trip
``PersistedCandidateMismatch``. That gate is deliberate. The revision bump itself
was the defect.
"""

from __future__ import annotations

import copy
import json

import pytest

from services.project_store import GraphRevisionConflict, ProjectStore

OWNER = "test-owner"


def _document():
    return {
        "schemaVersion": 2,
        "semantic": {
            "graph_id": "g1", "name": "n", "revision": 1,
            "nodes": [
                {"node_id": "input_node", "kind": "input", "op": "Input",
                 "params": {"shape": "1,16"}, "in_ports": [], "out_ports": ["y"]},
                {"node_id": "fc", "kind": "layer", "op": "Linear",
                 "params": {"in_features": 16, "out_features": 8},
                 "in_ports": ["x"], "out_ports": ["y"]},
            ],
            "edges": [{"edge_id": "e1", "source_node_id": "input_node",
                       "source_port": "y", "target_node_id": "fc", "target_port": "x"}],
        },
        "view": {
            "graph_id": "g1",
            "node_positions": {"input_node": {"x": 0, "y": 0}, "fc": {"x": 200, "y": 0}},
        },
    }


@pytest.fixture()
def store(tmp_path):
    s = ProjectStore(db_path=str(tmp_path / "t.sqlite3"))
    project = s.create_project(OWNER, "p")
    graph = s.create_graph(OWNER, project["id"], "g", _document())
    return s, project["id"], graph["id"], graph["current_revision"]


def _semantic_of(store, graph_id, project_id):
    row = store.get_graph(OWNER, graph_id, project_id=project_id)
    return json.loads(row["canonical_graph"])["semantic"]


# ── the fix ──────────────────────────────────────────────────────────────────
def test_a_position_only_save_keeps_the_revision(store):
    s, pid, gid, rev = store
    doc = _document()
    doc["view"]["node_positions"]["fc"] = {"x": 200, "y": 140}   # the drag

    saved = s.update_graph_view_if_revision(gid, rev, doc, owner_uid=OWNER, project_id=pid)

    assert saved["current_revision"] == rev, "a card move consumed a revision"


def test_the_moved_position_is_actually_persisted(store):
    """Keeping the revision must not mean dropping the user's edit."""
    s, pid, gid, rev = store
    doc = _document()
    doc["view"]["node_positions"]["fc"] = {"x": 999, "y": 140}

    s.update_graph_view_if_revision(gid, rev, doc, owner_uid=OWNER, project_id=pid)

    row = s.get_graph(OWNER, gid, project_id=pid)
    view = json.loads(row["canonical_graph"])["view"]
    assert view["node_positions"]["fc"]["x"] == 999


def test_the_stamped_semantic_revision_stays_consistent_with_the_store(store):
    """If the document said revision N+1 while the store said N, later reads and the
    AI commit gate (which hashes revision) would disagree."""
    s, pid, gid, rev = store
    doc = _document()
    doc["view"]["node_positions"]["fc"] = {"x": 5, "y": 5}

    saved = s.update_graph_view_if_revision(gid, rev, doc, owner_uid=OWNER, project_id=pid)

    assert _semantic_of(s, gid, pid)["revision"] == saved["current_revision"] == rev


def test_no_version_row_is_appended(store):
    """View fiddling used to fill the history with entries carrying an empty
    op_summary — two such rows appeared in the live repro."""
    s, pid, gid, rev = store
    before = len(s.list_graph_versions(OWNER, gid, project_id=pid) or [])
    doc = _document()
    doc["view"]["node_positions"]["fc"] = {"x": 1, "y": 1}

    s.update_graph_view_if_revision(gid, rev, doc, owner_uid=OWNER, project_id=pid)

    assert len(s.list_graph_versions(OWNER, gid, project_id=pid) or []) == before


# ── guards ───────────────────────────────────────────────────────────────────
def test_the_revision_cas_is_still_enforced(store):
    """Dropping the bump must not drop concurrency safety."""
    s, pid, gid, rev = store
    with pytest.raises(GraphRevisionConflict):
        s.update_graph_view_if_revision(gid, rev + 5, _document(), owner_uid=OWNER, project_id=pid)


def test_a_foreign_owner_cannot_write(store):
    s, pid, gid, rev = store
    assert s.update_graph_view_if_revision(
        gid, rev, _document(), owner_uid="someone-else", project_id=pid
    ) is None


def test_a_semantic_change_through_this_path_would_be_invisible(store):
    """Documents the contract: this method does NOT check semantics, so the caller
    must. If a future caller skips that check, a real edit would silently land
    without a revision — this test exists so that contract is written down.
    """
    s, pid, gid, rev = store
    doc = copy.deepcopy(_document())
    doc["semantic"]["nodes"].append(
        {"node_id": "relu", "kind": "layer", "op": "ReLU", "params": {},
         "in_ports": ["x"], "out_ports": ["y"]}
    )
    saved = s.update_graph_view_if_revision(gid, rev, doc, owner_uid=OWNER, project_id=pid)
    assert saved["current_revision"] == rev
    assert len(_semantic_of(s, gid, pid)["nodes"]) == 3, (
        "the method persisted what it was given; callers must gate on semantics"
    )
