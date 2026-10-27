"""Where a file-and-class projection is anchored for the AI harness (task.md D7).

The ``.py`` is the truth; this record is a cache. It exists for one reason: the
AI-turn contracts validate ``committed_revision == base_revision + 1``
(``ai/context/contracts.py``), so a turn cannot even be created against a graph
the store has never heard of. Every projection is therefore upserted under a
derived id — ``ide:<sha256(rel_file)[:12]>#<Class>`` — and its revision is
bumped, so M7 has something to point ``base_revision`` at.

Why this is not simply ``ProjectStore.create_graph``: that method mints a uuid
for the id, and the whole point here is that the id is *derived*, so the same
file re-parsed in a second window updates one record instead of minting a rival.
The initial row is therefore written through ProjectStore's own connection and
version helpers rather than duplicating its schema, and every subsequent write
goes through the public revision-CAS method.
"""

from __future__ import annotations

from typing import Any

from services.project_store import GraphRevisionConflict, ProjectStore, _utc_now

#: One project holds every IDE projection for an owner. Named, not derived: it
#: shows up in the web product's project list, and "tensorViz IDE" is what the
#: rows in it came from.
IDE_PROJECT_NAME = "tensorViz IDE"


def project_id(store: ProjectStore, owner_uid: str) -> str:
    """The id of the one project every IDE projection is anchored to.

    Public because the agent loop needs it and cannot derive it: an agent session
    is created against a project id, the extension is the side that creates the
    session, and nothing else in the IDE surface ever mentions one. Creating it on
    demand is the same behaviour projections already rely on, so asking for the id
    before the first projection is not a special case.
    """

    return _project_id(store, owner_uid)


def _project_id(store: ProjectStore, owner_uid: str) -> str:
    for project in store.list_projects(owner_uid):
        if project.get("name") == IDE_PROJECT_NAME:
            return str(project["id"])
    return str(store.create_project(owner_uid, IDE_PROJECT_NAME)["id"])


def upsert_projection(
    store: ProjectStore,
    *,
    owner_uid: str,
    graph_id: str,
    name: str,
    document: Any,
    op_summary: str = "ide:projection",
) -> int:
    """Write the projection at ``graph_id`` and return its new revision.

    Monotonic by construction: a first write lands at 1 and every later one is
    ``current + 1``, whether or not the document changed. An unchanged buffer
    still bumps because the revision numbers *the projection*, not the source —
    two parses of the same buffer are two facts the harness can distinguish.
    """
    existing = store.get_graph(owner_uid, graph_id)
    if existing is None:
        return _insert_first(store, owner_uid=owner_uid, graph_id=graph_id, name=name, document=document)

    try:
        committed = store.commit_graph_if_revision(
            graph_id,
            int(existing["current_revision"]),
            document,
            "user",
            op_summary,
            owner_uid=owner_uid,
        )
    except GraphRevisionConflict:
        # A second editor window upserted between the read and the write. The
        # projection is derived from the buffer we hold, so re-reading the
        # revision and writing again is correct — there is nothing to merge.
        current = store.get_graph(owner_uid, graph_id)
        if current is None:
            return _insert_first(
                store, owner_uid=owner_uid, graph_id=graph_id, name=name, document=document
            )
        committed = store.commit_graph_if_revision(
            graph_id,
            int(current["current_revision"]),
            document,
            "user",
            op_summary,
            owner_uid=owner_uid,
        )
    if committed is None:
        raise RuntimeError(f"projection anchor {graph_id} vanished mid-upsert")
    return int(committed["current_revision"])


def _insert_first(
    store: ProjectStore,
    *,
    owner_uid: str,
    graph_id: str,
    name: str,
    document: Any,
) -> int:
    project_id = _project_id(store, owner_uid)
    canonical = store._document_json(document, graph_id=graph_id, name=name, revision=1)
    now = _utc_now()
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        # Another window may have created it while we were resolving the project.
        row = conn.execute(
            "SELECT current_revision FROM app_graphs WHERE id = ? AND owner_uid = ? AND deleted_at IS NULL",
            (graph_id, owner_uid),
        ).fetchone()
        if row is not None:
            return int(row["current_revision"])
        conn.execute(
            """
            INSERT INTO app_graphs(
                id, project_id, owner_uid, name, canonical_graph,
                current_revision, legacy_payload, create_idempotency_key,
                created_at, updated_at, deleted_at
            )
            VALUES (?, ?, ?, ?, ?, 1, NULL, NULL, ?, ?, NULL)
            """,
            (graph_id, project_id, owner_uid, name, canonical, now, now),
        )
        store._insert_version(
            conn,
            graph_id=graph_id,
            revision=1,
            canonical_graph=canonical,
            author="user",
            op_summary="ide:projection",
        )
    return 1
