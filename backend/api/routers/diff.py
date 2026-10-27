"""Graph diff router. Compares two ArchitectureGraphs and returns the
structural delta (added / removed / modified nodes and edges)."""

from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth import CurrentUser, get_current_user
from services.transient_graph_store import transient_graph_store
from models.architecture_models import ArchitectureGraph
from services.graph_diff_service import GraphDiffResult, GraphDiffService


router = APIRouter(prefix="/api/diff", tags=["diff"])

_diff_service = GraphDiffService()
# Candidate session lookups share agent_ops' process-local registry. Saved
# project graphs never resolve through this endpoint; ProjectStore remains the
# only durable authority.
_session_store = transient_graph_store


class GraphRef(BaseModel):
    """Either a saved session reference or an inline graph payload."""

    session_id: Optional[str] = None
    graph: Optional[ArchitectureGraph] = None


class DiffRequest(BaseModel):
    a: GraphRef
    b: GraphRef
    correspondence: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional explicit node_id_a -> node_id_b map to anchor matching.",
    )


def _resolve(ref: GraphRef, label: str, requester_uid: str) -> ArchitectureGraph:
    if ref.graph is not None:
        return ref.graph
    if ref.session_id:
        session = _session_store.get_session(ref.session_id, requester_uid=requester_uid)
        if session is None:
            raise HTTPException(status_code=404, detail=f"{label}: session '{ref.session_id}' not found")
        return session.graph
    raise HTTPException(status_code=400, detail=f"{label}: provide either session_id or graph")


@router.post("/compare", response_model=GraphDiffResult)
def compare_graphs(request: DiffRequest, user: CurrentUser = Depends(get_current_user)) -> GraphDiffResult:
    graph_a = _resolve(request.a, "a", user.uid)
    graph_b = _resolve(request.b, "b", user.uid)
    return _diff_service.diff(graph_a, graph_b, correspondence=request.correspondence)
