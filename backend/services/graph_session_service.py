"""Helpers for creating and managing transient candidate graph sessions."""

import uuid
from typing import Dict, List, Optional

from models.architecture_models import ArchitectureGraph, ArchitectureNode
from services.graph_session_store import GraphSessionStore


class GraphSessionService:
    """High-level candidate-session helpers used by agent endpoints."""

    def __init__(self, store: GraphSessionStore):
        self.store = store

    def create_graph(
        self,
        name: str,
        input_shape: Optional[List[int]] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> ArchitectureGraph:
        shape = input_shape or [1, 3, 224, 224]
        graph_id = f"graph_{uuid.uuid4().hex}"
        input_node = ArchitectureNode(
            node_id="input",
            kind="input",
            op="Input",
            params={"shape": shape},
            in_ports=[],
            out_ports=["y"],
        )
        output_node = ArchitectureNode(
            node_id="output",
            kind="output",
            op="Output",
            params={},
            in_ports=["x"],
            out_ports=[],
        )
        return ArchitectureGraph(
            graph_id=graph_id,
            name=name,
            revision=0,
            inputs=[{"name": "x", "shape": shape}],
            nodes=[input_node, output_node],
            metadata=metadata or {},
        )

    def create_session(
        self,
        name: str,
        input_shape: Optional[List[int]] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        owner_uid: Optional[str] = None,
    ):
        graph = self.create_graph(name=name, input_shape=input_shape, metadata=metadata)
        return self.store.create_session(graph=graph, session_id=session_id, owner_uid=owner_uid)

    def record_operation_event(self, session_id: str, event: dict):
        return self.store.append_event(session_id=session_id, event=event)
