"""Canonical architecture and agent-operation models.

This module intentionally grows in small, reviewable commits.
"""

from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class ArchitectureNode(BaseModel):
    """Semantic architecture node independent from UI representation."""

    node_id: str = Field(min_length=1)
    kind: Literal["input", "output", "layer", "module", "op"]
    op: str = Field(min_length=1)
    params: Dict[str, Any] = Field(default_factory=dict)
    in_ports: List[str] = Field(default_factory=lambda: ["x"])
    out_ports: List[str] = Field(default_factory=lambda: ["y"])
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ArchitectureEdge(BaseModel):
    """Typed connection between source and target node ports."""

    edge_id: str = Field(min_length=1)
    source_node_id: str = Field(min_length=1)
    source_port: str = Field(min_length=1)
    target_node_id: str = Field(min_length=1)
    target_port: str = Field(min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ArchitectureModule(BaseModel):
    """Reusable hierarchical module definition."""

    module_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    input_ports: List[str] = Field(default_factory=lambda: ["x"])
    output_ports: List[str] = Field(default_factory=lambda: ["y"])
    nodes: List[ArchitectureNode] = Field(default_factory=list)
    edges: List[ArchitectureEdge] = Field(default_factory=list)
    params: Dict[str, Any] = Field(default_factory=dict)


class ArchitectureGraph(BaseModel):
    """Canonical server-owned architecture representation."""

    graph_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    revision: int = Field(ge=0, default=0)
    inputs: List[Dict[str, Any]] = Field(default_factory=list)
    nodes: List[ArchitectureNode] = Field(default_factory=list)
    edges: List[ArchitectureEdge] = Field(default_factory=list)
    modules: List[ArchitectureModule] = Field(default_factory=list)
    operations: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphView(BaseModel):
    """UI projection model for layout and editor state."""

    graph_id: str = Field(min_length=1)
    node_positions: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    collapsed_modules: List[str] = Field(default_factory=list)
    selection: List[str] = Field(default_factory=list)
    layout: Dict[str, Any] = Field(default_factory=dict)


# The document format's current version, and the oldest one that can still be read and
# upgraded. Bump CURRENT_SCHEMA_VERSION and register an upgrade step in
# services/graph_document.py::SCHEMA_MIGRATIONS in the same change.
CURRENT_SCHEMA_VERSION = 2
MIN_SUPPORTED_SCHEMA_VERSION = 1


class GraphDocumentV2(BaseModel):
    """Versioned project document containing model meaning and editor view.

    ``schemaVersion`` is a plain int, not ``Literal[2]``. As a Literal, pydantic rejected
    every value except 2, so a version 3 document could not be constructed at all — there
    was no way to ship a format change, and no way for a reader to even recognise a newer
    document and say so. Worse, ``graph_document_v2`` only special-cased ``== 2``, so
    anything else fell through to the legacy ModelGraph converter and came back as a
    document with ZERO semantic nodes rather than an error.

    Reading is therefore version-aware (see ``services/graph_document.py``): older
    documents are migrated forward, and newer ones are refused with a clear message
    instead of being silently misread. The class name is kept for import stability.
    """

    schemaVersion: int = Field(default=CURRENT_SCHEMA_VERSION, ge=1)
    semantic: ArchitectureGraph
    view: GraphView


class NodeValidationResult(BaseModel):
    """Validation outcome for a single architecture node."""

    status: Literal["ok", "warning", "error"]
    message: Optional[str] = None
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    suggested_repairs: List[Dict[str, Any]] = Field(default_factory=list)
    #: The opaque node that made this one uncheckable, if any.
    #:
    #: Set on every node downstream of a placeholder whose shape nobody has
    #: measured. It is what lets the answer collapse from "N layers are broken" to
    #: one statement about the block that is actually unknown — the blame belongs
    #: on the node nothing is known about, not on the correct layers after it.
    #: `None` on the cause itself, and on every node that was genuinely checked.
    blocked_by: Optional[str] = None


class ValidationResult(BaseModel):
    """Validation summary and per-node diagnostics."""

    valid: bool
    graph_id: str = Field(min_length=1)
    revision: int = Field(ge=0)
    global_errors: List[str] = Field(default_factory=list)
    node_results: Dict[str, NodeValidationResult] = Field(default_factory=dict)


class GraphOperation(BaseModel):
    """Base operation envelope for graph mutation requests."""

    op: str = Field(min_length=1)
    op_id: Optional[str] = None


class CreateModelOperation(GraphOperation):
    """Initialize a new canonical graph instance."""

    op: Literal["create_model"] = "create_model"
    name: str = Field(min_length=1)
    input_shape: List[int] = Field(default_factory=lambda: [1, 3, 224, 224])


class AddLayerOperation(GraphOperation):
    """Add a semantic layer node to the graph."""

    op: Literal["add_layer"] = "add_layer"
    node_id: str = Field(min_length=1)
    layer_op: str = Field(min_length=1)
    params: Dict[str, Any] = Field(default_factory=dict)
    in_ports: List[str] = Field(default_factory=lambda: ["x"])
    out_ports: List[str] = Field(default_factory=lambda: ["y"])


class RemoveLayerOperation(GraphOperation):
    """Remove a node and (optionally) all incident edges."""

    op: Literal["remove_layer"] = "remove_layer"
    node_id: str = Field(min_length=1)
    remove_incident_edges: bool = True


class ConnectOperation(GraphOperation):
    """Connect a source port to a target port."""

    op: Literal["connect"] = "connect"
    source_node_id: str = Field(min_length=1)
    source_port: str = Field(min_length=1)
    target_node_id: str = Field(min_length=1)
    target_port: str = Field(min_length=1)


class RemoveConnectionOperation(GraphOperation):
    """Remove one or more graph connections."""

    op: Literal["remove_connection"] = "remove_connection"
    edge_id: Optional[str] = None
    source_node_id: Optional[str] = None
    source_port: Optional[str] = None
    target_node_id: Optional[str] = None
    target_port: Optional[str] = None


class SetParamOperation(GraphOperation):
    """Update one parameter on a target node."""

    op: Literal["set_param"] = "set_param"
    node_id: str = Field(min_length=1)
    param: str = Field(min_length=1)
    value: Any


class InstantiateTemplateOperation(GraphOperation):
    """Instantiate a named template into the active graph."""

    op: Literal["instantiate_template"] = "instantiate_template"
    template_id: str = Field(min_length=1)
    instance_id: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)


class ValidateOperation(GraphOperation):
    """Request a validation pass on the current graph revision."""

    op: Literal["validate"] = "validate"


class ReplaceLayerOperation(GraphOperation):
    """Replace an existing node with a new one, preserving its connections."""

    op: Literal["replace_layer"] = "replace_layer"
    node_id: str = Field(min_length=1)
    new_layer_op: str = Field(min_length=1)
    new_node_id: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)


GraphOperationUnion = Annotated[
    Union[
        CreateModelOperation,
        AddLayerOperation,
        RemoveLayerOperation,
        ReplaceLayerOperation,
        ConnectOperation,
        RemoveConnectionOperation,
        SetParamOperation,
        InstantiateTemplateOperation,
        ValidateOperation,
    ],
    Field(discriminator="op"),
]


class AgentGraphPlan(BaseModel):
    """Structured output payload for operation-based graph planning."""

    intent: str = Field(min_length=1)
    ops: List[GraphOperationUnion] = Field(min_length=1)
    notes: Optional[str] = None


class GraphSession(BaseModel):
    """Transient agent candidate state; never a saved project authority."""

    session_id: str = Field(min_length=1)
    graph: ArchitectureGraph
    view: Optional[GraphView] = None
    operation_history: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
