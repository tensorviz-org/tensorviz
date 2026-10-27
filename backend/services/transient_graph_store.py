"""Process-local candidate graphs used only while planning agent changes.

These sessions are deliberately not durable. ProjectStore is the sole saved
graph authority; a candidate must be materialized as GraphDocumentV2 and
committed there before it becomes project state.
"""

from services.graph_session_store import GraphSessionStore


transient_graph_store = GraphSessionStore()
