"""Persistent per-user history store for chat + canvas events.

This service is intentionally append-only for events and keeps schemas
compatible with a future Postgres migration.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple


def _utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


class UserHistoryStore:
    """SQLite-backed history store (Phase 1 foundation).

    Notes:
    - Uses append-only `history_events`.
    - Keeps tenant dimensions: workspace/project/session/user.
    - Includes bootstrap metadata tables and reserved snapshot/memory tables.
    """

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
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS workspaces (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    owner_user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (owner_user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS workspace_members (
                    workspace_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (workspace_id, user_id),
                    FOREIGN KEY (workspace_id) REFERENCES workspaces(id),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (workspace_id) REFERENCES workspaces(id),
                    FOREIGN KEY (created_by) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (project_id) REFERENCES projects(id),
                    FOREIGN KEY (created_by) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS history_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT UNIQUE NOT NULL,
                    workspace_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_version INTEGER NOT NULL DEFAULT 1,
                    payload TEXT NOT NULL,
                    idempotency_key TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (workspace_id) REFERENCES workspaces(id),
                    FOREIGN KEY (project_id) REFERENCES projects(id),
                    FOREIGN KEY (session_id) REFERENCES sessions(id),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );

                CREATE INDEX IF NOT EXISTS idx_history_scope_time
                ON history_events(workspace_id, project_id, session_id, sequence);

                CREATE INDEX IF NOT EXISTS idx_history_event_type
                ON history_events(event_type);

                CREATE TABLE IF NOT EXISTS canvas_snapshots (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    graph_state TEXT NOT NULL,
                    base_event_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(session_id, revision)
                );

                CREATE TABLE IF NOT EXISTS memory_records (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    project_id TEXT,
                    user_id TEXT,
                    scope TEXT NOT NULL,
                    status TEXT NOT NULL,
                    text TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '[]',
                    source_event_ids TEXT NOT NULL DEFAULT '[]',
                    confidence REAL,
                    created_by_agent TEXT,
                    approved_by TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT
                );

                """
            )

    def _ensure_context(
        self,
        *,
        user_id: str,
        workspace_id: str,
        project_id: str,
        session_id: str,
        session_kind: str = "mixed",
        session_title: Optional[str] = None,
    ) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO users(id, email, created_at) VALUES (?, ?, ?)",
                (user_id, None, now),
            )
            conn.execute(
                "INSERT OR IGNORE INTO workspaces(id, name, owner_user_id, created_at) VALUES (?, ?, ?, ?)",
                (workspace_id, f"workspace_{workspace_id[:8]}", user_id, now),
            )
            conn.execute(
                "INSERT OR IGNORE INTO workspace_members(workspace_id, user_id, role, created_at) VALUES (?, ?, ?, ?)",
                (workspace_id, user_id, "owner", now),
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO projects(id, workspace_id, name, created_by, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    workspace_id,
                    f"project_{project_id[:8]}",
                    user_id,
                    now,
                    now,
                ),
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO sessions(id, project_id, kind, title, created_by, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    project_id,
                    session_kind,
                    session_title or f"session_{session_id[:8]}",
                    user_id,
                    now,
                    now,
                ),
            )

    def workspace_exists(self, workspace_id: str) -> bool:
        """True if the workspace has ever been written to (read-only)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM workspaces WHERE id = ?", (workspace_id,)
            ).fetchone()
            return row is not None

    def user_can_access_workspace(self, workspace_id: str, user_id: str) -> bool:
        """True if user_id owns the workspace or is a member (read-only).

        The first writer of a workspace becomes its owner + a member (see
        _ensure_context), so this is the authorization boundary for history.
        """
        with self._connect() as conn:
            if conn.execute(
                "SELECT 1 FROM workspaces WHERE id = ? AND owner_user_id = ?",
                (workspace_id, user_id),
            ).fetchone() is not None:
                return True
            return conn.execute(
                "SELECT 1 FROM workspace_members WHERE workspace_id = ? AND user_id = ?",
                (workspace_id, user_id),
            ).fetchone() is not None

    @staticmethod
    def _to_event_payload(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "sequence": row["sequence"],
            "id": row["id"],
            "workspace_id": row["workspace_id"],
            "project_id": row["project_id"],
            "session_id": row["session_id"],
            "user_id": row["user_id"],
            "event_type": row["event_type"],
            "event_version": row["event_version"],
            "payload": json.loads(row["payload"]),
            "idempotency_key": row["idempotency_key"],
            "created_at": row["created_at"],
        }

    def append_event(
        self,
        *,
        workspace_id: str,
        project_id: str,
        session_id: str,
        user_id: str,
        event_type: str,
        payload: Dict[str, Any],
        idempotency_key: str,
        event_version: int = 1,
        session_kind: str = "mixed",
        session_title: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], bool]:
        self._ensure_context(
            user_id=user_id,
            workspace_id=workspace_id,
            project_id=project_id,
            session_id=session_id,
            session_kind=session_kind,
            session_title=session_title,
        )

        now = _utc_now()
        event_id = str(uuid.uuid4())
        encoded_payload = json.dumps(payload, separators=(",", ":"))

        with self._connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO history_events(
                        id, workspace_id, project_id, session_id, user_id,
                        event_type, event_version, payload, idempotency_key, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        workspace_id,
                        project_id,
                        session_id,
                        user_id,
                        event_type,
                        event_version,
                        encoded_payload,
                        idempotency_key,
                        now,
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM history_events WHERE id = ?",
                    (event_id,),
                ).fetchone()
                return self._to_event_payload(row), False
            except sqlite3.IntegrityError:
                row = conn.execute(
                    "SELECT * FROM history_events WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                if row is None:
                    raise
                return self._to_event_payload(row), True

    def list_events(
        self,
        *,
        workspace_id: str,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        event_type_prefix: Optional[str] = None,
        after_sequence: Optional[int] = None,
        limit: int = 50,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        clamped_limit = max(1, min(limit, 500))
        clauses = ["workspace_id = ?"]
        params: List[Any] = [workspace_id]

        if project_id:
            clauses.append("project_id = ?")
            params.append(project_id)
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if event_type_prefix:
            clauses.append("event_type LIKE ?")
            params.append(f"{event_type_prefix}%")
        if after_sequence is not None:
            clauses.append("sequence > ?")
            params.append(after_sequence)

        where_clause = " AND ".join(clauses)
        query = f"""
            SELECT * FROM history_events
            WHERE {where_clause}
            ORDER BY sequence ASC
            LIMIT ?
        """
        params.append(clamped_limit + 1)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        has_more = len(rows) > clamped_limit
        effective_rows = rows[:clamped_limit]
        items = [self._to_event_payload(row) for row in effective_rows]
        next_cursor = str(effective_rows[-1]["sequence"]) if has_more and effective_rows else None
        return items, next_cursor

    @staticmethod
    def _to_snapshot_payload(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "workspace_id": row["workspace_id"],
            "project_id": row["project_id"],
            "session_id": row["session_id"],
            "revision": row["revision"],
            "graph_state": json.loads(row["graph_state"]),
            "base_event_id": row["base_event_id"],
            "created_at": row["created_at"],
        }

    def create_canvas_snapshot(
        self,
        *,
        workspace_id: str,
        project_id: str,
        session_id: str,
        user_id: str,
        revision: int,
        graph_state: Dict[str, Any],
        base_event_id: str,
        session_kind: str = "canvas",
    ) -> Dict[str, Any]:
        self._ensure_context(
            user_id=user_id,
            workspace_id=workspace_id,
            project_id=project_id,
            session_id=session_id,
            session_kind=session_kind,
            session_title=None,
        )
        snapshot_id = str(uuid.uuid4())
        created_at = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO canvas_snapshots(
                    id, workspace_id, project_id, session_id, revision,
                    graph_state, base_event_id, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    workspace_id,
                    project_id,
                    session_id,
                    revision,
                    json.dumps(graph_state, separators=(",", ":")),
                    base_event_id,
                    created_at,
                ),
            )
            row = conn.execute(
                "SELECT * FROM canvas_snapshots WHERE id = ?",
                (snapshot_id,),
            ).fetchone()
            return self._to_snapshot_payload(row)

    def replay_session(
        self,
        *,
        workspace_id: str,
        project_id: str,
        session_id: str,
        to_sequence: Optional[int] = None,
        limit: int = 5000,
    ) -> Dict[str, Any]:
        max_limit = max(1, min(limit, 10000))
        with self._connect() as conn:
            snapshot_row = conn.execute(
                """
                SELECT * FROM canvas_snapshots
                WHERE workspace_id = ? AND project_id = ? AND session_id = ?
                ORDER BY revision DESC, created_at DESC
                LIMIT 1
                """,
                (workspace_id, project_id, session_id),
            ).fetchone()

            snapshot = self._to_snapshot_payload(snapshot_row) if snapshot_row else None
            start_sequence = 0
            if snapshot and snapshot["base_event_id"]:
                base_row = conn.execute(
                    "SELECT sequence FROM history_events WHERE id = ?",
                    (snapshot["base_event_id"],),
                ).fetchone()
                if base_row:
                    start_sequence = int(base_row["sequence"])

            clauses = [
                "workspace_id = ?",
                "project_id = ?",
                "session_id = ?",
                "sequence > ?",
            ]
            params: List[Any] = [workspace_id, project_id, session_id, start_sequence]
            if to_sequence is not None:
                clauses.append("sequence <= ?")
                params.append(to_sequence)

            where_clause = " AND ".join(clauses)
            rows = conn.execute(
                f"""
                SELECT * FROM history_events
                WHERE {where_clause}
                ORDER BY sequence ASC
                LIMIT ?
                """,
                [*params, max_limit],
            ).fetchall()

        events = [self._to_event_payload(row) for row in rows]
        replay_to_sequence = events[-1]["sequence"] if events else start_sequence
        return {
            "snapshot": snapshot,
            "events": events,
            "count": len(events),
            "replay_to_sequence": replay_to_sequence,
        }
