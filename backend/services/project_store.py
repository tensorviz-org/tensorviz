"""Owner-scoped Projects/Graphs store (Phase 1.2 §3).

Server-owned projects and graphs with per-graph version history. Mirrors the
SQLite open-per-call pattern of :class:`UserHistoryStore` and shares the same
DB file (``USER_HISTORY_DB_PATH``). Every method takes ``owner_uid`` and filters
``owner_uid = ?`` AND ``deleted_at IS NULL`` so a lookup for another uid's row
returns ``None`` (the router turns that into a 404). Schema is Postgres-portable
with TEXT timestamps and soft deletes.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from services.graph_document import dump_graph_document, graph_document_v2


def _utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


#: Version snapshots are FULL documents (no delta encoding), so a count-only cap
#: is unbounded in bytes: an imported model (an SD UNet is ~340 nodes, SD3 ~756)
#: is several MB per snapshot, and a few dozen commits would mint hundreds of MB
#: inside one graph. Retention is therefore count-aware *and* size-aware.
VERSION_KEEP_LAST = 100
#: Total retained snapshot bytes allowed per graph.
VERSION_BYTES_BUDGET = 8 * 1024 * 1024
#: Newest revisions kept regardless of budget, so a single huge document can
#: never prune its own history to zero.
VERSION_KEEP_FLOOR = 3


class GraphRevisionConflict(RuntimeError):
    """Raised when a graph commit was planned from a stale revision."""

    def __init__(self, graph_id: str, expected_revision: int, current_revision: int) -> None:
        super().__init__(f"graph changed at revision {current_revision}")
        self.graph_id = graph_id
        self.expected_revision = expected_revision
        self.current_revision = current_revision


class ProjectStore:
    """SQLite-backed store for owner-scoped projects, graphs, and versions."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path:
            resolved = Path(db_path)
        else:
            base = Path(__file__).resolve().parents[1]
            resolved = Path(
                os.getenv(
                    "USER_HISTORY_DB_PATH",
                    str(base / ".runtime" / "user-history.sqlite3"),
                )
            )
        self._db_path = resolved
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self._db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS app_projects (
                    id TEXT PRIMARY KEY,
                    owner_uid TEXT NOT NULL,
                    name TEXT NOT NULL,
                    instructions TEXT NOT NULL DEFAULT '',
                    created_at TEXT,
                    updated_at TEXT,
                    deleted_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_app_projects_owner
                ON app_projects(owner_uid);

                CREATE TABLE IF NOT EXISTS app_graphs (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    owner_uid TEXT NOT NULL,
                    name TEXT NOT NULL,
                    canonical_graph TEXT NOT NULL DEFAULT '{}',
                    current_revision INTEGER NOT NULL DEFAULT 0,
                    legacy_payload TEXT,
                    create_idempotency_key TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    deleted_at TEXT,
                    FOREIGN KEY (project_id) REFERENCES app_projects(id)
                );

                CREATE INDEX IF NOT EXISTS idx_app_graphs_project_owner
                ON app_graphs(project_id, owner_uid);

                CREATE TABLE IF NOT EXISTS app_graph_versions (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    graph_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    canonical_graph TEXT NOT NULL,
                    author TEXT NOT NULL CHECK (author IN ('user', 'agent')),
                    op_summary TEXT,
                    created_at TEXT,
                    FOREIGN KEY (graph_id) REFERENCES app_graphs(id)
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_app_graph_versions_graph
                ON app_graph_versions(graph_id, revision);

                CREATE TABLE IF NOT EXISTS app_user_templates (
                    id TEXT PRIMARY KEY,
                    owner_uid TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    canonical_graph TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT,
                    updated_at TEXT,
                    deleted_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_app_user_templates_owner
                ON app_user_templates(owner_uid);
                """
            )
            graph_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(app_graphs)").fetchall()
            }
            project_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(app_projects)").fetchall()
            }
            if "instructions" not in project_columns:
                conn.execute(
                    "ALTER TABLE app_projects ADD COLUMN instructions TEXT NOT NULL DEFAULT ''"
                )
            if "current_revision" not in graph_columns:
                conn.execute(
                    "ALTER TABLE app_graphs ADD COLUMN current_revision INTEGER NOT NULL DEFAULT 0"
                )
            if "legacy_payload" not in graph_columns:
                conn.execute("ALTER TABLE app_graphs ADD COLUMN legacy_payload TEXT")
            if "create_idempotency_key" not in graph_columns:
                conn.execute("ALTER TABLE app_graphs ADD COLUMN create_idempotency_key TEXT")
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_app_graphs_create_idempotency
                ON app_graphs(owner_uid, project_id, create_idempotency_key)
                WHERE create_idempotency_key IS NOT NULL AND deleted_at IS NULL
                """
            )
            conn.execute(
                """
                UPDATE app_graphs
                SET current_revision = COALESCE(
                    (SELECT MAX(revision) FROM app_graph_versions WHERE graph_id = app_graphs.id),
                    0
                )
                WHERE current_revision = 0
                """
            )

    # ── serializers ───────────────────────────────────────────────────────

    @staticmethod
    def _project_row(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "owner_uid": row["owner_uid"],
            "name": row["name"],
            "instructions": row["instructions"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _graph_row(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "project_id": row["project_id"],
            "owner_uid": row["owner_uid"],
            "name": row["name"],
            "canonical_graph": row["canonical_graph"],
            "current_revision": row["current_revision"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _graph_scope(
        graph_id: str, owner_uid: str, project_id: Optional[str]
    ) -> tuple[str, List[Any]]:
        """WHERE fragment + params selecting an owned, live graph, optionally scoped
        to ``project_id`` (a graph must belong to the project named in the path)."""
        sql = "id = ? AND owner_uid = ? AND deleted_at IS NULL"
        params: List[Any] = [graph_id, owner_uid]
        if project_id is not None:
            sql += " AND project_id = ?"
            params.append(project_id)
        return sql, params

    def _insert_version(
        self,
        conn: sqlite3.Connection,
        *,
        graph_id: str,
        revision: int,
        canonical_graph: str,
        author: str,
        op_summary: Optional[str],
        keep_last: int = VERSION_KEEP_LAST,
    ) -> None:
        """Insert a known revision within ``conn`` and prune old snapshots."""
        if author not in ("user", "agent"):
            raise ValueError("author must be 'user' or 'agent'")
        now = _utc_now()
        conn.execute(
            """
            INSERT INTO app_graph_versions(graph_id, revision, canonical_graph, author, op_summary, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (graph_id, revision, canonical_graph, author, op_summary, now),
        )
        conn.execute(
            """
            DELETE FROM app_graph_versions
            WHERE graph_id = ?
              AND revision <= (
                  SELECT MAX(revision) - ? FROM app_graph_versions WHERE graph_id = ?
              )
            """,
            (graph_id, keep_last, graph_id),
        )
        self._prune_version_bytes(conn, graph_id=graph_id)

    @staticmethod
    def _prune_version_bytes(conn: sqlite3.Connection, *, graph_id: str) -> None:
        """Drop oldest snapshots until the graph's retained bytes fit
        ``VERSION_BYTES_BUDGET``, always keeping the ``VERSION_KEEP_FLOOR``
        newest revisions even when one document exceeds the whole budget."""
        rows = conn.execute(
            """
            SELECT revision, LENGTH(CAST(canonical_graph AS BLOB)) AS bytes
            FROM app_graph_versions WHERE graph_id = ?
            ORDER BY revision DESC
            """,
            (graph_id,),
        ).fetchall()
        retained = 0
        for kept, row in enumerate(rows, start=1):
            retained += int(row["bytes"])
            if kept > VERSION_KEEP_FLOOR and retained > VERSION_BYTES_BUDGET:
                # This revision and everything older than it is over budget.
                conn.execute(
                    "DELETE FROM app_graph_versions WHERE graph_id = ? AND revision <= ?",
                    (graph_id, row["revision"]),
                )
                return

    @staticmethod
    def _document_json(
        document: Any,
        *,
        graph_id: str,
        name: str,
        revision: int,
    ) -> str:
        if isinstance(document, str):
            document = json.loads(document) if document else {}
        normalized = graph_document_v2(
            document,
            graph_id=graph_id,
            name=name,
            revision=revision,
        )
        return json.dumps(
            dump_graph_document(normalized),
            separators=(",", ":"),
            sort_keys=True,
        )

    # ── projects ──────────────────────────────────────────────────────────

    def create_project(
        self, owner_uid: str, name: str, instructions: str = ""
    ) -> Dict[str, Any]:
        now = _utc_now()
        project_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO app_projects(
                    id, owner_uid, name, instructions, created_at, updated_at, deleted_at
                )
                VALUES (?, ?, ?, ?, ?, ?, NULL)
                """,
                (project_id, owner_uid, name, instructions, now, now),
            )
            row = conn.execute("SELECT * FROM app_projects WHERE id = ?", (project_id,)).fetchone()
            return self._project_row(row)

    def list_projects(self, owner_uid: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM app_projects WHERE owner_uid = ? AND deleted_at IS NULL ORDER BY created_at ASC",
                (owner_uid,),
            ).fetchall()
        return [self._project_row(r) for r in rows]

    def get_project(self, owner_uid: str, project_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM app_projects WHERE id = ? AND owner_uid = ? AND deleted_at IS NULL",
                (project_id, owner_uid),
            ).fetchone()
        return self._project_row(row) if row else None

    def update_project(
        self,
        owner_uid: str,
        project_id: str,
        *,
        name: Optional[str] = None,
        instructions: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        now = _utc_now()
        with self._connect() as conn:
            assignments = ["updated_at = ?"]
            values: List[Any] = [now]
            if name is not None:
                assignments.append("name = ?")
                values.append(name)
            if instructions is not None:
                assignments.append("instructions = ?")
                values.append(instructions)
            values.extend([project_id, owner_uid])
            cur = conn.execute(
                f"UPDATE app_projects SET {', '.join(assignments)} "
                "WHERE id = ? AND owner_uid = ? AND deleted_at IS NULL",
                values,
            )
            if cur.rowcount == 0:
                return None
            row = conn.execute("SELECT * FROM app_projects WHERE id = ?", (project_id,)).fetchone()
            return self._project_row(row)

    def delete_project(self, owner_uid: str, project_id: str) -> bool:
        """Soft-delete the project and its graphs. Returns False if not owned/missing."""
        now = _utc_now()
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE app_projects SET deleted_at = ?, updated_at = ? WHERE id = ? AND owner_uid = ? AND deleted_at IS NULL",
                (now, now, project_id, owner_uid),
            )
            if cur.rowcount == 0:
                return False
            conn.execute(
                "UPDATE app_graphs SET deleted_at = ?, updated_at = ? WHERE project_id = ? AND owner_uid = ? AND deleted_at IS NULL",
                (now, now, project_id, owner_uid),
            )
            return True

    # ── graphs ────────────────────────────────────────────────────────────

    def create_graph(
        self,
        owner_uid: str,
        project_id: str,
        name: str,
        document: Any = None,
        author: str = "user",
        op_summary: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Create a graph under an owned project; append the initial version.

        A non-null idempotency key is scoped to the owner and project. Replays
        return the original live graph without appending another revision.
        """
        if author not in ("user", "agent"):
            raise ValueError("author must be 'user' or 'agent'")
        now = _utc_now()
        graph_id = uuid.uuid4().hex
        with self._connect() as conn:
            if idempotency_key is not None:
                # Serialize lookup + insert so a lost create response can be
                # retried concurrently without minting a second graph.
                conn.execute("BEGIN IMMEDIATE")
            project = conn.execute(
                "SELECT id FROM app_projects WHERE id = ? AND owner_uid = ? AND deleted_at IS NULL",
                (project_id, owner_uid),
            ).fetchone()
            if project is None:
                return None
            if idempotency_key is not None:
                existing = conn.execute(
                    """
                    SELECT * FROM app_graphs
                    WHERE owner_uid = ? AND project_id = ?
                      AND create_idempotency_key = ? AND deleted_at IS NULL
                    """,
                    (owner_uid, project_id, idempotency_key),
                ).fetchone()
                if existing is not None:
                    return self._graph_row(existing)
            canonical_graph = self._document_json(
                document if document is not None else {
                    "schemaVersion": 2,
                    "semantic": {"graph_id": graph_id, "name": name},
                    "view": {"graph_id": graph_id},
                },
                graph_id=graph_id,
                name=name,
                revision=1,
            )
            legacy_payload = None
            if document is not None:
                if isinstance(document, str):
                    source_text = document
                    try:
                        source_object = json.loads(document) if document else {}
                    except (TypeError, ValueError):
                        source_object = {}
                else:
                    source_object = document
                    source_text = json.dumps(document, separators=(",", ":"))
                if not isinstance(source_object, dict) or source_object.get("schemaVersion") != 2:
                    legacy_payload = source_text
            conn.execute(
                """
                INSERT INTO app_graphs(
                    id, project_id, owner_uid, name, canonical_graph,
                    current_revision, legacy_payload, create_idempotency_key,
                    created_at, updated_at, deleted_at
                )
                VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?, NULL)
                """,
                (
                    graph_id,
                    project_id,
                    owner_uid,
                    name,
                    canonical_graph,
                    legacy_payload,
                    idempotency_key,
                    now,
                    now,
                ),
            )
            self._insert_version(
                conn,
                graph_id=graph_id,
                revision=1,
                canonical_graph=canonical_graph,
                author=author,
                op_summary=op_summary or "create",
            )
            row = conn.execute("SELECT * FROM app_graphs WHERE id = ?", (graph_id,)).fetchone()
            return self._graph_row(row)

    def list_graphs(self, owner_uid: str, project_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM app_graphs WHERE project_id = ? AND owner_uid = ? AND deleted_at IS NULL ORDER BY created_at ASC",
                (project_id, owner_uid),
            ).fetchall()
        return [self._graph_row(r) for r in rows]

    def get_graph(
        self, owner_uid: str, graph_id: str, project_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        where, params = self._graph_scope(graph_id, owner_uid, project_id)
        with self._connect() as conn:
            row = conn.execute(f"SELECT * FROM app_graphs WHERE {where}", params).fetchone()
        return self._graph_row(row) if row else None

    def commit_graph_if_revision(
        self,
        graph_id: str,
        expected_revision: int,
        document: Any,
        author: str,
        op_summary: Optional[str],
        *,
        owner_uid: str,
        project_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Atomically replace the saved document when its revision still matches."""
        if author not in ("user", "agent"):
            raise ValueError("author must be 'user' or 'agent'")
        if expected_revision < 0:
            raise ValueError("expected_revision must be non-negative")

        now = _utc_now()
        where, params = self._graph_scope(graph_id, owner_uid, project_id)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(f"SELECT * FROM app_graphs WHERE {where}", params).fetchone()
            if row is None:
                return None

            current_revision = int(row["current_revision"])
            if current_revision != expected_revision:
                raise GraphRevisionConflict(graph_id, expected_revision, current_revision)

            next_revision = current_revision + 1
            next_name = name if name is not None else row["name"]
            legacy_payload = row["legacy_payload"]
            if legacy_payload is None:
                try:
                    current_document = json.loads(row["canonical_graph"])
                except (TypeError, ValueError):
                    current_document = {}
                if not isinstance(current_document, dict) or current_document.get("schemaVersion") != 2:
                    legacy_payload = row["canonical_graph"]
            canonical_graph = self._document_json(
                document,
                graph_id=graph_id,
                name=next_name,
                revision=next_revision,
            )
            updated = conn.execute(
                """
                UPDATE app_graphs
                SET name = ?, canonical_graph = ?, current_revision = ?,
                    legacy_payload = COALESCE(legacy_payload, ?), updated_at = ?
                WHERE id = ? AND current_revision = ?
                """,
                (
                    next_name,
                    canonical_graph,
                    next_revision,
                    legacy_payload,
                    now,
                    graph_id,
                    expected_revision,
                ),
            )
            if updated.rowcount != 1:
                actual = conn.execute(
                    "SELECT current_revision FROM app_graphs WHERE id = ?", (graph_id,)
                ).fetchone()
                raise GraphRevisionConflict(
                    graph_id, expected_revision, int(actual["current_revision"])
                )

            self._insert_version(
                conn,
                graph_id=graph_id,
                revision=next_revision,
                canonical_graph=canonical_graph,
                author=author,
                op_summary=op_summary,
            )
            committed = conn.execute(
                "SELECT * FROM app_graphs WHERE id = ?", (graph_id,)
            ).fetchone()
            return self._graph_row(committed)

    def update_graph_view_if_revision(
        self,
        graph_id: str,
        expected_revision: int,
        document: Any,
        *,
        owner_uid: str,
        project_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Persist a document whose SEMANTIC plane is unchanged, keeping the revision.

        A revision numbers a change to the architecture. Dragging a card or panning
        the camera is not one, and charging it a revision had a sharp consequence:
        it invalidated any AI candidate under review, because staleness is a
        revision comparison. Accepting the candidate then failed with
        ``canvas changed after the candidate was reviewed`` and the reviewed work
        was discarded — reproduced by nudging one node while a candidate was
        pending.

        Callers MUST have already established that the semantic bytes match; this
        method only enforces the revision CAS, and stamps the document with the
        SAME revision so ``semantic.revision`` stays consistent with the store.
        No version row is appended, so view fiddling no longer fills the history
        with entries carrying an empty ``op_summary``.
        """
        if expected_revision < 0:
            raise ValueError("expected_revision must be non-negative")
        now = _utc_now()
        where, params = self._graph_scope(graph_id, owner_uid, project_id)
        with self._connect() as conn:
            row = conn.execute(f"SELECT * FROM app_graphs WHERE {where}", params).fetchone()
            if row is None:
                return None
            current_revision = int(row["current_revision"])
            if current_revision != expected_revision:
                raise GraphRevisionConflict(graph_id, expected_revision, current_revision)
            next_name = name if name is not None else row["name"]
            canonical_graph = self._document_json(
                document,
                graph_id=graph_id,
                name=next_name,
                revision=current_revision,
            )
            updated = conn.execute(
                """
                UPDATE app_graphs
                SET name = ?, canonical_graph = ?, updated_at = ?
                WHERE id = ? AND current_revision = ?
                """,
                (next_name, canonical_graph, now, graph_id, expected_revision),
            )
            if updated.rowcount != 1:
                actual = conn.execute(
                    "SELECT current_revision FROM app_graphs WHERE id = ?", (graph_id,)
                ).fetchone()
                raise GraphRevisionConflict(
                    graph_id, expected_revision, int(actual["current_revision"])
                )
            committed = conn.execute(
                "SELECT * FROM app_graphs WHERE id = ?", (graph_id,)
            ).fetchone()
            return self._graph_row(committed)

    def delete_graph(
        self, owner_uid: str, graph_id: str, project_id: Optional[str] = None
    ) -> bool:
        now = _utc_now()
        where, params = self._graph_scope(graph_id, owner_uid, project_id)
        with self._connect() as conn:
            cur = conn.execute(
                f"UPDATE app_graphs SET deleted_at = ?, updated_at = ? WHERE {where}",
                [now, now, *params],
            )
            return cur.rowcount > 0

    # ── user templates (personal reusable "starter files") ────────────────

    @staticmethod
    def _user_template_row(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "owner_uid": row["owner_uid"],
            "name": row["name"],
            "description": row["description"],
            "canonical_graph": row["canonical_graph"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def create_user_template(
        self,
        owner_uid: str,
        name: str,
        canonical_graph: str = "{}",
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save the current graph as a reusable personal template."""
        now = _utc_now()
        template_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO app_user_templates(id, owner_uid, name, description, canonical_graph, created_at, updated_at, deleted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (template_id, owner_uid, name, description, canonical_graph, now, now),
            )
            row = conn.execute("SELECT * FROM app_user_templates WHERE id = ?", (template_id,)).fetchone()
            return self._user_template_row(row)

    def list_user_templates(self, owner_uid: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM app_user_templates WHERE owner_uid = ? AND deleted_at IS NULL ORDER BY created_at DESC, rowid DESC",
                (owner_uid,),
            ).fetchall()
        return [self._user_template_row(r) for r in rows]

    def get_user_template(self, owner_uid: str, template_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM app_user_templates WHERE id = ? AND owner_uid = ? AND deleted_at IS NULL",
                (template_id, owner_uid),
            ).fetchone()
        return self._user_template_row(row) if row else None

    def delete_user_template(self, owner_uid: str, template_id: str) -> bool:
        now = _utc_now()
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE app_user_templates SET deleted_at = ?, updated_at = ? WHERE id = ? AND owner_uid = ? AND deleted_at IS NULL",
                (now, now, template_id, owner_uid),
            )
            return cur.rowcount > 0

    # ── versions ──────────────────────────────────────────────────────────

    def list_graph_versions(
        self, owner_uid: str, graph_id: str, project_id: Optional[str] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Version metadata (newest last). Returns None if the graph isn't owned."""
        where, params = self._graph_scope(graph_id, owner_uid, project_id)
        with self._connect() as conn:
            graph = conn.execute(f"SELECT id FROM app_graphs WHERE {where}", params).fetchone()
            if graph is None:
                return None
            rows = conn.execute(
                "SELECT revision, author, op_summary, created_at FROM app_graph_versions "
                "WHERE graph_id = ? ORDER BY revision ASC",
                (graph_id,),
            ).fetchall()
        return [
            {
                "revision": r["revision"],
                "author": r["author"],
                "op_summary": r["op_summary"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def get_graph_version(
        self, owner_uid: str, graph_id: str, revision: int
    ) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            graph = conn.execute(
                "SELECT id FROM app_graphs WHERE id = ? AND owner_uid = ? AND deleted_at IS NULL",
                (graph_id, owner_uid),
            ).fetchone()
            if graph is None:
                return None
            row = conn.execute(
                "SELECT revision, canonical_graph, author, op_summary, created_at "
                "FROM app_graph_versions WHERE graph_id = ? AND revision = ?",
                (graph_id, revision),
            ).fetchone()
        if row is None:
            return None
        return {
            "revision": row["revision"],
            "canonical_graph": row["canonical_graph"],
            "author": row["author"],
            "op_summary": row["op_summary"],
            "created_at": row["created_at"],
        }

    def restore_graph_version(
        self,
        owner_uid: str,
        graph_id: str,
        revision: int,
        expected_revision: int,
        project_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Restore a historical payload using the same revision guard as commits."""
        where, params = self._graph_scope(graph_id, owner_uid, project_id)
        with self._connect() as conn:
            graph = conn.execute(f"SELECT * FROM app_graphs WHERE {where}", params).fetchone()
            if graph is None:
                return None
            version = conn.execute(
                "SELECT canonical_graph FROM app_graph_versions WHERE graph_id = ? AND revision = ?",
                (graph_id, revision),
            ).fetchone()
            if version is None:
                return None
            document = version["canonical_graph"]
        return self.commit_graph_if_revision(
            graph_id,
            expected_revision,
            document,
            "user",
            f"restore r{revision}",
            owner_uid=owner_uid,
            project_id=project_id,
        )
