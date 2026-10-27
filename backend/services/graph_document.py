"""Conversion and validation for the versioned project graph document."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Mapping

from models.architecture_models import (
    CURRENT_SCHEMA_VERSION,
    MIN_SUPPORTED_SCHEMA_VERSION,
    ArchitectureEdge,
    ArchitectureGraph,
    ArchitectureNode,
    GraphDocumentV2,
    GraphView,
)


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _node_kind(node: Mapping[str, Any], data: Mapping[str, Any]) -> str:
    layer = _as_dict(data.get("layerType"))
    candidate = str(layer.get("id") or data.get("type") or node.get("type") or "").lower()
    if candidate == "input":
        return "input"
    if candidate == "output":
        return "output"
    if node.get("type") == "moduleNode" or candidate in {"group", "module", "modulenode"} or data.get("isGroup"):
        return "module"
    return "layer"


def _legacy_model_graph_to_v2(
    payload: Mapping[str, Any],
    *,
    graph_id: str,
    name: str,
    revision: int,
) -> GraphDocumentV2:
    raw_nodes = payload.get("nodes") if isinstance(payload.get("nodes"), list) else []
    raw_edges = payload.get("edges") if isinstance(payload.get("edges"), list) else []

    nodes = []
    positions: Dict[str, Dict[str, float]] = {}
    collapsed_modules = []
    selection = []
    inputs = []

    for index, raw in enumerate(raw_nodes):
        node = _as_dict(raw)
        node_id = str(node.get("id") or f"node-{index}")
        data = _as_dict(node.get("data"))
        layer = _as_dict(data.get("layerType"))
        kind = _node_kind(node, data)
        op = str(layer.get("name") or layer.get("id") or node.get("type") or "Unknown")
        params = _as_dict(data.get("parameters"))

        position = _as_dict(node.get("position"))
        positions[node_id] = {
            "x": float(position.get("x") or 0),
            "y": float(position.get("y") or 0),
        }
        if bool(node.get("selected")) or bool(data.get("isSelected")):
            selection.append(node_id)
        if (kind == "module" and data.get("expanded") is not True) or bool(data.get("collapsed")):
            collapsed_modules.append(node_id)

        if kind == "input":
            shape = params.get("shape") or data.get("inputShape")
            inputs.append({"name": node_id, "shape": shape})

        canvas = deepcopy(node)
        canvas.pop("id", None)
        canvas.pop("position", None)
        nodes.append(
            ArchitectureNode(
                node_id=node_id,
                kind=kind,
                op=op,
                params=params,
                in_ports=[] if kind == "input" else ["x"],
                out_ports=[] if kind == "output" else ["y"],
                metadata={"canvas": canvas},
            )
        )

    edges = []
    for index, raw in enumerate(raw_edges):
        edge = _as_dict(raw)
        edge_id = str(edge.get("id") or f"edge-{index}")
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if not source or not target:
            continue
        canvas = deepcopy(edge)
        for key in ("id", "source", "target", "sourceHandle", "targetHandle"):
            canvas.pop(key, None)
        edges.append(
            ArchitectureEdge(
                edge_id=edge_id,
                source_node_id=source,
                source_port=str(edge.get("sourceHandle") or "y"),
                target_node_id=target,
                target_port=str(edge.get("targetHandle") or "x"),
                metadata={"canvas": canvas},
            )
        )

    semantic = ArchitectureGraph(
        graph_id=graph_id,
        name=name,
        revision=revision,
        inputs=inputs,
        nodes=nodes,
        edges=edges,
        operations=[],
        metadata={
            "migration": {
                "sourceSchema": "ModelGraph",
                "legacyPayload": deepcopy(dict(payload)),
            }
        },
    )
    view = GraphView(
        graph_id=graph_id,
        node_positions=positions,
        collapsed_modules=collapsed_modules,
        selection=selection,
        layout={"strategy": "manual"},
    )
    return GraphDocumentV2(semantic=semantic, view=view)


class DocumentVersionError(ValueError):
    """A stored document's schemaVersion cannot be read by this build.

    Raised rather than guessed at. The alternative — treating an unrecognised version as a
    legacy ModelGraph — silently produced a document with zero semantic nodes, which every
    downstream consumer then read as "this model is empty".
    """


# Upgrade steps keyed by the version they read, each returning the next version's shape.
# Empty today because version 2 is the only format that has ever been written; the registry
# exists so the first format change is an addition here plus a bump of
# CURRENT_SCHEMA_VERSION, rather than a rewrite of this function.
SCHEMA_MIGRATIONS: Dict[int, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}


def migrate_document(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Bring a versioned document up to CURRENT_SCHEMA_VERSION.

    Walks the migration chain one version at a time so each step only has to know how to
    get from N to N+1. A document from the future is refused: this build cannot know what
    changed, and reading it optimistically would corrupt it on the next save.
    """
    out = dict(raw)
    version = out.get("schemaVersion")
    if not isinstance(version, int):
        raise DocumentVersionError(f"schemaVersion must be an int, got {version!r}")

    if version > CURRENT_SCHEMA_VERSION:
        raise DocumentVersionError(
            f"document schemaVersion {version} was written by a newer build of canvas; "
            f"this one supports up to {CURRENT_SCHEMA_VERSION}. Upgrade rather than opening it, "
            "so the newer format is not downgraded on save."
        )
    if version < MIN_SUPPORTED_SCHEMA_VERSION:
        raise DocumentVersionError(
            f"document schemaVersion {version} is older than the oldest supported version "
            f"({MIN_SUPPORTED_SCHEMA_VERSION})"
        )

    while version < CURRENT_SCHEMA_VERSION:
        step = SCHEMA_MIGRATIONS.get(version)
        if step is None:
            raise DocumentVersionError(
                f"no migration registered from schemaVersion {version} to {version + 1}; "
                "add one to SCHEMA_MIGRATIONS"
            )
        out = dict(step(out))
        next_version = out.get("schemaVersion")
        if next_version != version + 1:
            raise DocumentVersionError(
                f"migration from schemaVersion {version} must set schemaVersion "
                f"{version + 1}, produced {next_version!r}"
            )
        version = next_version
    return out


def graph_document_v2(
    payload: Any,
    *,
    graph_id: str,
    name: str,
    revision: int,
) -> GraphDocumentV2:
    """Return a validated current-version document, converting or migrating on read.

    Three input shapes are handled distinctly, because conflating them is how an unreadable
    document turns into a silently empty one:
      * a versioned document at the current version — validated as-is;
      * a versioned document at an older version — migrated forward, then validated;
      * anything with no schemaVersion — a legacy ModelGraph, converted.

    Legacy conversion is intentionally non-destructive: the original ModelGraph is carried
    in migration metadata for an identical first reopen, while the raw stored revision
    remains untouched for database-level rollback.
    """

    raw = _as_dict(payload)
    if "schemaVersion" in raw:
        document = GraphDocumentV2.model_validate(migrate_document(raw))
        document.semantic.graph_id = graph_id
        document.semantic.name = name
        document.semantic.revision = revision
        document.view.graph_id = graph_id
        return document
    return _legacy_model_graph_to_v2(
        raw,
        graph_id=graph_id,
        name=name,
        revision=revision,
    )


def dump_graph_document(document: GraphDocumentV2) -> Dict[str, Any]:
    return document.model_dump(mode="json")


def model_graph_from_document(document: GraphDocumentV2) -> Dict[str, Any]:
    """Project a V2 document back to the canvas boundary.

    A converted legacy record uses its exact migration payload. Native V2
    documents are reconstructed from semantic node/edge metadata plus GraphView.
    """

    migration = _as_dict(document.semantic.metadata.get("migration"))
    legacy_payload = migration.get("legacyPayload")
    if isinstance(legacy_payload, Mapping):
        return deepcopy(dict(legacy_payload))

    selected = set(document.view.selection)
    collapsed = set(document.view.collapsed_modules)
    nodes = []
    for semantic_node in document.semantic.nodes:
        canvas = deepcopy(_as_dict(semantic_node.metadata.get("canvas")))
        data = _as_dict(canvas.get("data"))
        data["parameters"] = deepcopy(semantic_node.params)
        if semantic_node.node_id in selected:
            canvas["selected"] = True
            data["isSelected"] = True
        if semantic_node.node_id in collapsed:
            data["collapsed"] = True
        canvas.update(
            {
                "id": semantic_node.node_id,
                "position": deepcopy(
                    document.view.node_positions.get(
                        semantic_node.node_id, {"x": 0.0, "y": 0.0}
                    )
                ),
                "data": data,
            }
        )
        nodes.append(canvas)

    edges = []
    for semantic_edge in document.semantic.edges:
        canvas = deepcopy(_as_dict(semantic_edge.metadata.get("canvas")))
        canvas.update(
            {
                "id": semantic_edge.edge_id,
                "source": semantic_edge.source_node_id,
                "target": semantic_edge.target_node_id,
            }
        )
        if semantic_edge.source_port != "y":
            canvas["sourceHandle"] = semantic_edge.source_port
        if semantic_edge.target_port != "x":
            canvas["targetHandle"] = semantic_edge.target_port
        edges.append(canvas)
    return {"nodes": nodes, "edges": edges}
