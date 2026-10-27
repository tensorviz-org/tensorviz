"""Version-snapshot retention must be bounded in BYTES, not only in count.

Snapshots are full documents with no delta encoding, so a count-only cap
(``keep_last``) is unbounded in bytes: an imported model (a Stable Diffusion
UNet is ~340 nodes, SD3 ~756) is megabytes per snapshot, so a few dozen edit
commits would mint hundreds of megabytes inside a single graph. These tests
pin the byte budget and the most-recent-N floor that stop that.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict


from services.project_store import (
    VERSION_BYTES_BUDGET,
    VERSION_KEEP_FLOOR,
    VERSION_KEEP_LAST,
    ProjectStore,
)


def _imported_model(node_count: int, param_bytes: int, label: str) -> Dict[str, Any]:
    """A ModelGraph payload shaped like an import: nodes carrying op params."""
    return {
        "nodes": [
            {
                "id": f"n{i}",
                "type": "op",
                "position": {"x": i * 10, "y": 0},
                "data": {
                    "label": f"{label}-node-{i}",
                    "op": "linear",
                    "params": {"in_features": 4096, "trace": "t" * param_bytes},
                    "shape": [1, 4096],
                },
            }
            for i in range(node_count)
        ],
        "edges": [
            {"id": f"e{i}", "source": f"n{i}", "target": f"n{i + 1}"}
            for i in range(node_count - 1)
        ],
    }


def _store(tmp_path: Path) -> tuple[ProjectStore, Path]:
    db_path = tmp_path / "retention.sqlite3"
    return ProjectStore(str(db_path)), db_path


def _retained(db_path: Path, graph_id: str) -> tuple[int, int]:
    """(row count, total retained bytes) for one graph's snapshots."""
    conn = sqlite3.connect(str(db_path))
    try:
        count, total = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(LENGTH(CAST(canonical_graph AS BLOB))), 0) "
            "FROM app_graph_versions WHERE graph_id = ?",
            (graph_id,),
        ).fetchone()
    finally:
        conn.close()
    return int(count), int(total)


def _commit_many(
    store: ProjectStore,
    graph: Dict[str, Any],
    commits: int,
    node_count: int,
    param_bytes: int,
) -> Dict[str, Any]:
    revision = graph["current_revision"]
    for index in range(1, commits + 1):
        graph = store.commit_graph_if_revision(
            graph["id"],
            revision,
            _imported_model(node_count, param_bytes, f"r{index}"),
            "user",
            f"edit-{index}",
            owner_uid="user-a",
        )
        assert graph is not None
        revision = graph["current_revision"]
    return graph


def test_many_small_revisions_still_retain_keep_last(tmp_path: Path):
    """The byte budget must not shrink normal history: small documents keep
    behaving exactly as before (count-capped at ``VERSION_KEEP_LAST``)."""
    store, db_path = _store(tmp_path)
    project = store.create_project("user-a", "P")
    graph = store.create_graph("user-a", project["id"], "G", _imported_model(3, 0, "r0"))
    assert graph is not None

    graph = _commit_many(store, graph, VERSION_KEEP_LAST + 20, 3, 0)

    count, total = _retained(db_path, graph["id"])
    assert count == VERSION_KEEP_LAST
    assert total <= VERSION_BYTES_BUDGET


def test_large_document_revisions_are_pruned_to_the_byte_budget(tmp_path: Path):
    """A multi-MB imported model must not blow up storage: retention prunes
    oldest snapshots so the graph's retained bytes stay inside the budget,
    which a count-only cap of 100 full snapshots would never do."""
    store, db_path = _store(tmp_path)
    project = store.create_project("user-a", "P")
    commits = 15
    graph = store.create_graph("user-a", project["id"], "G", _imported_model(340, 1200, "r0"))
    assert graph is not None

    _, one_snapshot = _retained(db_path, graph["id"])
    assert one_snapshot > 512 * 1024, "fixture must be a genuinely large document"

    graph = _commit_many(store, graph, commits, 340, 1200)

    count, total = _retained(db_path, graph["id"])
    assert total <= VERSION_BYTES_BUDGET
    # Count-only retention would have kept all 16 revisions (~16 MB).
    assert count < commits + 1
    assert count >= VERSION_KEEP_FLOOR


def test_most_recent_floor_survives_a_document_larger_than_the_budget(tmp_path: Path):
    """A single document can exceed the whole budget (SD3 at full depth). It
    must still be undoable: the newest ``VERSION_KEEP_FLOOR`` revisions are
    kept, so history can never prune itself to zero."""
    store, db_path = _store(tmp_path)
    project = store.create_project("user-a", "P")
    graph = store.create_graph("user-a", project["id"], "G", _imported_model(340, 14000, "r0"))
    assert graph is not None

    _, one_snapshot = _retained(db_path, graph["id"])
    assert one_snapshot > VERSION_BYTES_BUDGET, "fixture must exceed the whole budget"

    graph = _commit_many(store, graph, VERSION_KEEP_FLOOR + 4, 340, 14000)

    count, _ = _retained(db_path, graph["id"])
    assert count == VERSION_KEEP_FLOOR

    versions = store.list_graph_versions("user-a", graph["id"])
    assert versions is not None
    latest = graph["current_revision"]
    assert [v["revision"] for v in versions] == [
        latest - VERSION_KEEP_FLOOR + 1 + i for i in range(VERSION_KEEP_FLOOR)
    ]
    assert store.get_graph_version("user-a", graph["id"], latest) is not None
