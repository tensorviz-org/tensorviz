from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


from services.project_store import GraphRevisionConflict, ProjectStore


def _v2(label: str = "base"):
    return {
        "schemaVersion": 2,
        "semantic": {
            "graph_id": "client-value",
            "name": label,
            "nodes": [],
            "edges": [],
            "operations": [{"op": "note", "label": label}],
        },
        "view": {"graph_id": "client-value", "layout": {"label": label}},
    }


def _created_graph(store: ProjectStore):
    project = store.create_project("user-a", "P")
    graph = store.create_graph("user-a", project["id"], "G", _v2())
    assert graph is not None
    return project, graph


def test_new_graph_is_v2_and_exposes_initial_revision(tmp_path: Path):
    store = ProjectStore(str(tmp_path / "graphs.sqlite3"))
    _, graph = _created_graph(store)

    document = json.loads(graph["canonical_graph"])
    assert graph["current_revision"] == 1
    assert document["schemaVersion"] == 2
    assert document["semantic"]["graph_id"] == graph["id"]
    assert document["semantic"]["revision"] == 1


def test_idempotent_create_race_returns_one_graph_and_one_revision(tmp_path: Path):
    store = ProjectStore(str(tmp_path / "graphs.sqlite3"))
    project = store.create_project("user-a", "P")

    def create(label: str):
        return store.create_graph(
            "user-a",
            project["id"],
            label,
            _v2(label),
            author="agent",
            op_summary=label,
            idempotency_key="create-attempt-1",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        created = list(executor.map(create, ("first", "retry")))

    assert created[0] is not None and created[1] is not None
    assert created[0]["id"] == created[1]["id"]
    graphs = store.list_graphs("user-a", project["id"])
    assert [graph["id"] for graph in graphs] == [created[0]["id"]]
    versions = store.list_graph_versions("user-a", created[0]["id"])
    assert versions is not None
    assert [version["revision"] for version in versions] == [1]


def test_unkeyed_creates_remain_distinct(tmp_path: Path):
    store = ProjectStore(str(tmp_path / "graphs.sqlite3"))
    project = store.create_project("user-a", "P")

    first = store.create_graph("user-a", project["id"], "First", _v2("first"))
    second = store.create_graph("user-a", project["id"], "Second", _v2("second"))

    assert first is not None and second is not None
    assert first["id"] != second["id"]


def test_same_revision_race_commits_once_and_conflicts_once(tmp_path: Path):
    store = ProjectStore(str(tmp_path / "graphs.sqlite3"))
    project, graph = _created_graph(store)

    def commit(label: str):
        try:
            result = store.commit_graph_if_revision(
                graph["id"],
                1,
                _v2(label),
                "agent" if label == "agent" else "user",
                label,
                owner_uid="user-a",
                project_id=project["id"],
            )
            return ("ok", result)
        except GraphRevisionConflict as exc:
            return ("conflict", exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(commit, ("user", "agent")))

    assert [status for status, _ in results].count("ok") == 1
    assert [status for status, _ in results].count("conflict") == 1
    conflict = next(value for status, value in results if status == "conflict")
    assert conflict.graph_id == graph["id"]
    assert conflict.expected_revision == 1
    assert conflict.current_revision == 2

    versions = store.list_graph_versions(
        "user-a", graph["id"], project_id=project["id"]
    )
    assert versions is not None
    assert [version["revision"] for version in versions] == [1, 2]
    assert store.get_graph("user-a", graph["id"])["current_revision"] == 2


def test_user_and_agent_commits_add_exactly_one_revision_each(tmp_path: Path):
    store = ProjectStore(str(tmp_path / "graphs.sqlite3"))
    project, graph = _created_graph(store)

    user_commit = store.commit_graph_if_revision(
        graph["id"],
        1,
        _v2("user edit"),
        "user",
        "user edit",
        owner_uid="user-a",
        project_id=project["id"],
    )
    agent_commit = store.commit_graph_if_revision(
        graph["id"],
        2,
        _v2("AI batch"),
        "agent",
        "AI: 4 operations",
        owner_uid="user-a",
        project_id=project["id"],
    )

    assert user_commit["current_revision"] == 2
    assert agent_commit["current_revision"] == 3
    versions = store.list_graph_versions("user-a", graph["id"])
    assert [(v["revision"], v["author"]) for v in versions] == [
        (1, "user"),
        (2, "user"),
        (3, "agent"),
    ]


def test_legacy_bytes_survive_first_v2_commit(tmp_path: Path):
    db = tmp_path / "graphs.sqlite3"
    store = ProjectStore(str(db))
    project = store.create_project("user-a", "P")
    legacy_bytes = '{  "nodes" : [{"id":"n1","position":{"x":1,"y":2},"data":{}}], "edges": [] }'
    graph = store.create_graph("user-a", project["id"], "Legacy", legacy_bytes)

    committed = store.commit_graph_if_revision(
        graph["id"],
        1,
        _v2("migrated"),
        "user",
        "edit legacy graph",
        owner_uid="user-a",
        project_id=project["id"],
    )
    assert committed["current_revision"] == 2

    with sqlite3.connect(str(db)) as conn:
        row = conn.execute(
            "SELECT canonical_graph, legacy_payload FROM app_graphs WHERE id = ?",
            (graph["id"],),
        ).fetchone()
    assert row[1] == legacy_bytes
    document = json.loads(row[0])
    assert document["schemaVersion"] == 2
    assert "migration" not in document["semantic"]["metadata"]
    assert document["semantic"]["operations"] == [{"label": "migrated", "op": "note"}]


def test_existing_database_backfills_current_revision(tmp_path: Path):
    db = tmp_path / "old.sqlite3"
    with sqlite3.connect(str(db)) as conn:
        conn.executescript(
            """
            CREATE TABLE app_projects (
                id TEXT PRIMARY KEY, owner_uid TEXT NOT NULL, name TEXT NOT NULL,
                created_at TEXT, updated_at TEXT, deleted_at TEXT
            );
            CREATE TABLE app_graphs (
                id TEXT PRIMARY KEY, project_id TEXT NOT NULL, owner_uid TEXT NOT NULL,
                name TEXT NOT NULL, canonical_graph TEXT NOT NULL DEFAULT '{}',
                created_at TEXT, updated_at TEXT, deleted_at TEXT
            );
            CREATE TABLE app_graph_versions (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, graph_id TEXT NOT NULL,
                revision INTEGER NOT NULL, canonical_graph TEXT NOT NULL,
                author TEXT NOT NULL, op_summary TEXT, created_at TEXT
            );
            INSERT INTO app_projects VALUES ('p1', 'user-a', 'P', 't', 't', NULL);
            INSERT INTO app_graphs VALUES ('g1', 'p1', 'user-a', 'G', '{"nodes":[]}', 't', 't', NULL);
            INSERT INTO app_graph_versions(graph_id, revision, canonical_graph, author, created_at)
            VALUES ('g1', 2, '{"nodes":[]}', 'user', 't');
            INSERT INTO app_graph_versions(graph_id, revision, canonical_graph, author, created_at)
            VALUES ('g1', 5, '{"nodes":[]}', 'agent', 't');
            """
        )

    store = ProjectStore(str(db))
    graph = store.get_graph("user-a", "g1", project_id="p1")

    assert graph["current_revision"] == 5
    with sqlite3.connect(str(db)) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(app_graphs)")]
    assert "current_revision" in columns
    assert "legacy_payload" in columns
    assert "create_idempotency_key" in columns
    with sqlite3.connect(str(db)) as conn:
        indexes = [row[1] for row in conn.execute("PRAGMA index_list(app_graphs)")]
    assert "idx_app_graphs_create_idempotency" in indexes
