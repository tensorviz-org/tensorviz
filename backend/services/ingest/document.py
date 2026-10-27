"""Encode a flat ``ModelGraph`` as the ``GraphDocumentV2`` shape the CLIENT reads.

There are two different V2 encodings in this system and only one of them round-trips
canvas state:

* ``services/graph_document.py::_legacy_model_graph_to_v2`` (the backend's migration path)
  nests each node's canvas fields under ``semantic.nodes[].metadata.canvas`` and keeps a
  ``metadata.migration.legacyPayload`` copy.
* ``canvas/src/services/graphDocumentAdapter.ts`` (what the app itself writes and
  reads) puts the node type in ``view.layout.node_canvas[id].type`` and the per-node data
  in ``view.layout.node_data[id]``, and rebuilds ``data`` as
  ``{...semantic.nodes[].metadata, ...view.layout.node_data[id], layerType: <derived from
  op>, parameters: node.params, expanded: !collapsed_modules.has(id)}``.

The client's ``migrationPayload`` shortcut only returns the legacy copy verbatim when
re-converting it reproduces the document's semantic core byte-for-byte — which the two
encoders do not, because they disagree about ``metadata``. So a document written the
backend's way falls through to the ``node_data``/``node_canvas`` path, finds nothing there,
and loses ``parentId`` and the module labels: children of collapsed modules leak out as
loose top-level cards and every block renders as "module / 0 layers".

Ingestion therefore writes the CLIENT encoding. Verified visually end to end on an imported
Stable Diffusion UNet and a DiT-XL/2.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from models.architecture_models import GraphDocumentV2

# The client only preserves these keys of node.data when it saves
# (graphDocumentAdapter.ts SEMANTIC_DATA_KEYS). Anything else — a sibling `meta` object,
# for instance — is dropped on the first autosave, which is why importer metadata rides as
# `_`-prefixed entries inside data.parameters instead.
CLIENT_DATA_KEYS = (
    "layerType", "parentId", "type", "isGroup", "moduleInstanceId", "moduleTemplateId",
    "moduleEntryNodeId", "moduleExitNodeId", "moduleLayerCount", "moduleMembers",
    "modulePreviewNodes", "modulePreviewEdges", "repeatCount", "repeatBlockLabel",
    "repeatHidden", "moduleCapture",
)


_REPEAT_SUFFIX = re.compile(r"\s*[x×]\s*\d+\s*$")


def _semantic_op(layer_type: Dict[str, Any], raw: Dict[str, Any]) -> str:
    """The op for the SEMANTIC graph, which must stay a bare identifier.

    The canvas title for a repeated block is "BasicBlock x2" — the count is display, and
    it is already carried structurally in ``data.repeatCount`` and ``params._repeat_count``.
    Writing the title here put a SPACE into the semantic op, and codegen derives a Python
    class name from it, so an imported resnet18 generated `class BasicBlock x2` and the
    candidate export check failed with "invalid syntax". Strip the suffix; keep everything
    else byte-identical.
    """
    name = str(
        layer_type.get("name") or layer_type.get("id") or raw.get("type") or "Unknown"
    )
    return _REPEAT_SUFFIX.sub("", name).strip() or "Unknown"


def _kind(node: Dict[str, Any]) -> str:
    data = node.get("data") or {}
    layer_type = data.get("layerType") or {}
    candidate = str(layer_type.get("id") or node.get("type") or "").lower()
    if candidate == "input":
        return "input"
    if candidate == "output":
        return "output"
    if node.get("type") == "moduleNode" or data.get("isGroup"):
        return "module"
    return "layer"


def to_document(name: str, graph: Dict[str, Any], graph_id: str = "pending") -> GraphDocumentV2:
    """Build a validated V2 document from ``{nodes, edges}``."""
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    positions: Dict[str, Dict[str, float]] = {}
    node_canvas: Dict[str, Dict[str, Any]] = {}
    node_data: Dict[str, Dict[str, Any]] = {}
    collapsed: List[str] = []
    inputs: List[Dict[str, Any]] = []
    incoming: Dict[str, List[str]] = {}
    outgoing: Dict[str, List[str]] = {}
    for raw in graph.get("edges") or []:
        for field, table, side, fallback in (("source", outgoing, "sourceHandle", "y"),
                                              ("target", incoming, "targetHandle", "x")):
            ports = table.setdefault(str(raw.get(field)), [])
            port = raw.get(side) or (raw.get("data") or {}).get(side) or fallback
            if port not in ports:
                ports.append(port)

    for raw in graph.get("nodes") or []:
        node_id = str(raw.get("id"))
        data = dict(raw.get("data") or {})
        layer_type = data.get("layerType") or {}
        kind = _kind(raw)

        position = raw.get("position") or {}
        positions[node_id] = {
            "x": float(position.get("x") or 0.0),
            "y": float(position.get("y") or 0.0),
        }
        if kind == "module" and data.get("expanded") is not True:
            collapsed.append(node_id)
        params = dict(data.pop("parameters", {}) or {})
        if kind == "input":
            inputs.append({"name": node_id, "shape": params.get("shape")})
        data.pop("layerType", None)

        node_canvas[node_id] = {"type": raw.get("type")}
        node_data[node_id] = data
        nodes.append(
            {
                "node_id": node_id,
                "kind": kind,
                "op": _semantic_op(layer_type, raw),
                "params": params,
                "in_ports": data.get("inputPorts", [] if kind == "input" else incoming.get(node_id, ["x"])),
                "out_ports": data.get("outputPorts", [] if kind == "output" else outgoing.get(node_id, ["y"])),
                # A static container can remain an opaque custom_code op after
                # workspace resolution. Preserve that identity on canvas reads
                # instead of replacing it with the generic Module palette op.
                "metadata": {
                    **({"parentId": data["parentId"]} if "parentId" in data else {}),
                    **({"layerType": layer_type} if data.get("moduleType") == "container" else {}),
                },
            }
        )

    for raw in graph.get("edges") or []:
        role = raw.get("data") or {}
        edges.append(
            {
                "edge_id": str(raw.get("id")),
                "source_node_id": str(raw.get("source")),
                "source_port": raw.get("sourceHandle") or role.get("sourceHandle") or "y",
                "target_node_id": str(raw.get("target")),
                "target_port": raw.get("targetHandle") or role.get("targetHandle") or "x",
                "metadata": {
                    "role": "residual", "roleEvidence": role["roleEvidence"],
                } if role.get("role") == "residual" and role.get("roleEvidence") in (
                    "identity-bypass", "projection-shortcut", "declared",
                ) else {},
            }
        )

    return GraphDocumentV2.model_validate(
        {
            "schemaVersion": 2,
            "semantic": {
                "graph_id": graph_id,
                "name": name,
                "revision": 0,
                "inputs": inputs,
                "nodes": nodes,
                "edges": edges,
                "modules": [],
                "operations": [],
                "metadata": {"importer": "services.ingest"},
            },
            "view": {
                "graph_id": graph_id,
                "node_positions": positions,
                "collapsed_modules": collapsed,
                "selection": [],
                "layout": {
                    "strategy": "imported",
                    "node_canvas": node_canvas,
                    "node_data": node_data,
                    "edge_canvas": {},
                },
            },
        }
    )
