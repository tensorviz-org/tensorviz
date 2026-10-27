"""Process-local storage for transient agent candidate graphs.

Candidate sessions are owner-scoped but deliberately never durable. A graph
becomes saved project state only through ``ProjectStore``'s revision-guarded V2
commit path.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from models.architecture_models import ArchitectureGraph, GraphSession

# Matches auth.DEV_UID so AUTH_DISABLED sessions have a stable owner.
DEFAULT_OWNER_UID = "local-dev-user"


class GraphSessionStore:
    """In-memory, owner-scoped candidate session store."""

    def __init__(self):
        self._sessions: Dict[str, GraphSession] = {}
        self._undo_stacks: Dict[str, List[ArchitectureGraph]] = {}
        self._redo_stacks: Dict[str, List[ArchitectureGraph]] = {}
        self._owners: Dict[str, str] = {}

    # ── public API ─────────────────────────────────────────────────────────

    def create_session(
        self,
        graph: ArchitectureGraph,
        session_id: Optional[str] = None,
        owner_uid: Optional[str] = None,
    ) -> GraphSession:
        sid = session_id or f"session_{uuid.uuid4().hex}"
        session = GraphSession(session_id=sid, graph=graph)
        self._sessions[sid] = session
        self._undo_stacks[sid] = []
        self._redo_stacks[sid] = []
        self._owners[sid] = owner_uid or DEFAULT_OWNER_UID
        return session

    def get_session(
        self, session_id: str, requester_uid: Optional[str] = None
    ) -> Optional[GraphSession]:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if requester_uid is not None and self._owners.get(session_id) not in (None, requester_uid):
            # Owned by someone else → behave as not-found (no existence leak).
            return None
        return session

    def owner_of(self, session_id: str) -> Optional[str]:
        return self._owners.get(session_id)

    def save_session(self, session: GraphSession) -> GraphSession:
        session.updated_at = datetime.utcnow()
        self._sessions[session.session_id] = session
        return session

    def push_undo_snapshot(self, session_id: str, graph_snapshot: ArchitectureGraph) -> bool:
        if self._sessions.get(session_id) is None:
            return False
        self._undo_stacks.setdefault(session_id, []).append(graph_snapshot.model_copy(deep=True))
        self._redo_stacks[session_id] = []
        return True

    def undo(self, session_id: str) -> Optional[GraphSession]:
        session = self._sessions.get(session_id)
        if not session:
            return None
        undo_stack = self._undo_stacks.setdefault(session_id, [])
        if not undo_stack:
            return None
        self._redo_stacks.setdefault(session_id, []).append(session.graph.model_copy(deep=True))
        session.graph = undo_stack.pop()
        session.updated_at = datetime.utcnow()
        return session

    def redo(self, session_id: str) -> Optional[GraphSession]:
        session = self._sessions.get(session_id)
        if not session:
            return None
        redo_stack = self._redo_stacks.setdefault(session_id, [])
        if not redo_stack:
            return None
        self._undo_stacks.setdefault(session_id, []).append(session.graph.model_copy(deep=True))
        session.graph = redo_stack.pop()
        session.updated_at = datetime.utcnow()
        return session

    def history_state(self, session_id: str) -> Optional[dict]:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        return {
            "undo_count": len(self._undo_stacks.get(session_id, [])),
            "redo_count": len(self._redo_stacks.get(session_id, [])),
            "event_count": len(session.operation_history),
        }

    def append_event(self, session_id: str, event: dict) -> Optional[GraphSession]:
        session = self._sessions.get(session_id)
        if not session:
            return None
        session.operation_history.append(event)
        session.updated_at = datetime.utcnow()
        return session
