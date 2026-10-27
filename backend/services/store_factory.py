"""The one place a :class:`ProjectStore` is constructed (task.md M9).

Seven call sites — five routers, the AI orchestrator and the domain executor —
used to each write ``ProjectStore()`` and so each bind their own copy of the
"where does the project database live" decision. That decision is now moving
(per-workspace identity in M9, per-workspace *storage* after it), and a decision
that lives in seven files cannot move at all.

This module is deliberately thin: it is a seam, not a layer. It adds no caching
and no behaviour — ``get_project_store()`` today constructs exactly what those
call sites constructed, resolving ``USER_HISTORY_DB_PATH`` per call the way
:class:`ProjectStore` always did — so switching the sites is provably a no-op and
the next change to store selection is a change to one function.
"""

from __future__ import annotations

from typing import Optional

from services.project_store import ProjectStore


def get_project_store(db_path: Optional[str] = None) -> ProjectStore:
    """Return the project store this process should use.

    ``db_path`` overrides the location for callers that already know it (tests,
    and anything that has resolved a workspace itself); with it omitted the store
    resolves ``USER_HISTORY_DB_PATH`` at call time.
    """
    return ProjectStore(db_path)
