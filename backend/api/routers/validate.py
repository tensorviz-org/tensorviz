"""Legacy /validate_shapes compatibility router backed by canonical validator.

This endpoint keeps the frontend response contract stable while delegating
shape/graph validation to the same backend engine used by agent/runtime APIs.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.pytorch_requirement import load as load_pytorch_service
from core.op_registry import spec_for
from models.architecture_models import ArchitectureEdge, ArchitectureGraph, ArchitectureNode

router = APIRouter()
#: Built on first use. Canonical validation runs layers, so it loads PyTorch, and
#: importing this router must not: the server starts without it.
_validator = None


def _canonical_validation():
    """The validation module and its shared validator, or a 503 naming PyTorch."""
    global _validator
    validation = load_pytorch_service("services.canonical_validation", "Shape validation")
    if _validator is None:
        _validator = validation.CanonicalValidationService()
    return validation, _validator


class NodeData(BaseModel):
    layerType: Optional[Dict[str, Any]] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    parentId: Optional[str] = None
    isGroup: Optional[bool] = None
    collapsed: Optional[bool] = None


class Node(BaseModel):
    id: str
    type: str
    position: Dict[str, float]
    data: NodeData


class Edge(BaseModel):
    id: str
    source: str
    target: str


class ValidationRequest(BaseModel):
    nodes: List[Node]
    edges: List[Edge]
    input_shape: List[int] = Field(default_factory=lambda: [1, 3, 224, 224])


def _summarize_repair_ops(repairs: List[Dict[str, Any]]) -> str:
    if not repairs:
        return "Check node params and graph connectivity"
    first = repairs[0]
    op = first.get("op")
    if op == "set_param":
        node_id = first.get("node_id")
        param = first.get("param")
        value = first.get("value")
        if node_id and param is not None:
            return f"Set {node_id}.{param} to {value}"
        return "Apply suggested parameter fix"
    if op == "remove_connection":
        return "Remove conflicting connection"
    if op == "connect":
        return "Add missing connection"
    if op == "add_layer":
        return "Insert suggested adapter layer"
    return str(op or "Apply suggested repair")


def _normalize_layer_type(raw: Optional[str]) -> str:
    if not raw:
        return ""
    value = str(raw).strip()
    alias = {
        "conv2d": "nn.Conv2d",
        "linear": "nn.Linear",
        "relu": "nn.ReLU",
        "maxpool2d": "nn.MaxPool2d",
        "avgpool2d": "nn.AvgPool2d",
        "adaptiveavgpool2d": "nn.AdaptiveAvgPool2d",
        "flatten": "nn.Flatten",
        "lstm": "nn.LSTM",
        "gru": "nn.GRU",
        "rnn": "nn.RNN",
        "gelu": "nn.GELU",
        "layernorm": "nn.LayerNorm",
        "batchnorm2d": "nn.BatchNorm2d",
        "dropout": "nn.Dropout",
        "sigmoid": "nn.Sigmoid",
        "tanh": "nn.Tanh",
        "multiheadattention": "nn.MultiheadAttention",
        "input": "Input",
        "output": "Output",
        "module": "Module",
    }
    compact = value.replace(".", "").lower()
    return alias.get(compact, value)


def _coerce_shape(value: Any) -> Optional[List[int]]:
    if isinstance(value, list) and value and all(isinstance(v, int) and v > 0 for v in value):
        return list(value)
    if isinstance(value, str):
        raw = [chunk.strip() for chunk in value.split(",")]
        if not raw:
            return None
        out: List[int] = []
        for chunk in raw:
            if not chunk:
                return None
            if not chunk.lstrip("-").isdigit():
                return None
            dim = int(chunk)
            if dim <= 0:
                return None
            out.append(dim)
        return out or None
    return None


def _extract_layer_type(node: Node) -> str:
    layer_type = node.data.layerType or {}
    if isinstance(layer_type, dict):
        raw = layer_type.get("name") or layer_type.get("id")
        return _normalize_layer_type(raw)
    return ""


def _keep_module_node(
    node: Any,
    edge_sources: set[str],
    edge_targets: set[str],
    frame_ids: set[str],
) -> bool:
    """Should this module node reach the validator, or is it a frame to drop?

    Three conditions, all load-bearing:

    * it is not a FRAME. `_module_roles` decides that by asking whether the module's
      children carry the dataflow — the earlier "does a dotted child exist" string
      test misread every traced `nn.MultiheadAttention`, whose one decoration child
      carries no edges, and cut 84 templates in half.
    * the registry knows its op. A traced BERT expands into twelve
      `moduleNode` children whose op is `transformerencoderlayer` —
      `nn.TransformerEncoderLayer`, a real operation — and THOSE carry the whole
      chain: token_embeddings -> layer_0 -> ... -> layer_11 -> classifier_head.
      Dropping them took every edge with them, which is what actually left `output`
      "unreachable from input" on a model that had just run. Placeholders pass this
      too (`custom_code` resolves), so the opaque case is unaffected. An op the
      registry does not know is still dropped, because keeping it would trade a
      cascade for an "unknown layer type" error.
    * it has a real incident edge. A partially wired boundary still belongs in
      the graph; dropping it also drops source topology and hides real defects.

    A measured `_out_shape` is NOT required, and used to be. Keeping only measured
    blocks still deleted the edges of every unmeasured one, so an UNTRACED modular
    model came back with "Node is unreachable from input" and a global unreachable
    output — false wiring errors about a graph that is wired correctly. The node has
    to stay either way; what differs is what is known about it. With a measurement it
    resolves and the chain holds; without one it is the opaque cause, and the
    unchecked machinery reports that honestly, once, without calling it broken.
    """
    if node.id in frame_ids:
        return False
    op = _extract_layer_type(node)
    # Static ModuleList declarations now have module semantics. An actual wire
    # to one must survive too: the validator reports that boundary as opaque,
    # rather than deleting source topology because no executable op is registered.
    if op.lower() == "modulelist":
        return node.id in edge_sources or node.id in edge_targets
    if spec_for(op or "") is None:
        return False
    return node.id in edge_sources or node.id in edge_targets


def _module_roles(
    request: ValidationRequest,
    edge_sources: set[str],
    edge_targets: set[str],
) -> tuple[Dict[str, tuple[str, str]], set[str]]:
    """Which module nodes are display FRAMES, and which children are decoration.

    The question is whether the CHILDREN carry the dataflow — not, as the first
    version of this asked, whether a dotted child exists at all. That earlier test
    was a string match (`id.rsplit(".", 1)[0]`), and it misread every traced
    `nn.MultiheadAttention`: at the worker's default depth an attention node
    materialises one decoration child, `<attn>.out_proj`, which carries no edges.
    The attention itself sits on the trunk with an edge in and an edge out, and
    dropping it deleted `norm1 -> attn` and `attn -> norm2` — cutting the chain in
    half and answering seven connectivity errors on a ViT block that is wired
    correctly. 84 templates, 497 such errors, none of them true.

    So, per module node:

    * if none of its children touch an edge, the PARENT is the computation and the
      children are decoration — keep the parent, drop the children;
    * otherwise the children carry the chain, the parent is a frame, and its
      (entry, exit) children are returned so the caller can rewire the frame's
      boundary edges onto them rather than deleting them.

    Only a module node may own children, so a user who names a plain layer
    `block.norm` can no longer delete their `block`.

    The canvas already does exactly this on its own side, in
    `shapeInferencePayload.ts::buildInferablePayload` (`moduleBoundaryChildren`);
    this is the same rule for the path the extension uses.
    """
    module_ids = {
        node.id
        for node in request.nodes
        if node.type == "moduleNode" or bool(node.data.isGroup)
    }
    children: Dict[str, list[str]] = {}
    for node in request.nodes:
        if "." not in node.id:
            continue
        parent = node.id.rsplit(".", 1)[0]
        if parent in module_ids:
            children.setdefault(parent, []).append(node.id)

    boundaries: Dict[str, tuple[str, str]] = {}
    decoration: set[str] = set()
    for parent, child_ids in children.items():
        wired = [c for c in child_ids if c in edge_sources or c in edge_targets]
        if not wired:
            # The parent IS the computation; its children are decoration a deeper
            # trace materialised (`<attn>.out_proj`). Validating them adds a "has no
            # input connections" error about a node nobody wired on purpose.
            decoration.update(child_ids)
            continue
        siblings = set(child_ids)
        internal_targets = {
            e.target for e in request.edges if e.source in siblings and e.target in siblings
        }
        internal_sources = {
            e.source for e in request.edges if e.source in siblings and e.target in siblings
        }
        # Candidates restricted to `wired`: an edge-free Dropout inside a real frame
        # must not be picked as its exit.
        entry = next((c for c in wired if c not in internal_targets), None)
        exit_child = next((c for c in wired if c not in internal_sources), None)
        if entry is not None and exit_child is not None:
            boundaries[parent] = (entry, exit_child)
    return boundaries, decoration


def _build_canonical_graph(request: ValidationRequest) -> ArchitectureGraph:
    canonical_nodes: List[ArchitectureNode] = []
    included_ids: set[str] = set()

    fallback_shape = _coerce_shape(request.input_shape) or [1, 3, 224, 224]
    explicit_input_shape: Optional[List[int]] = None

    # Which ids an edge actually touches, so a node is only kept when dropping it
    # would break a wire. Computed once, read by `_measured_block` below.
    edge_sources = {edge.source for edge in request.edges}
    edge_targets = {edge.target for edge in request.edges}
    #: Which modules are display FRAMES, and which children are decoration. A traced
    #: projection sends the block AND its expansion — `encoder_stack` plus twelve
    #: `encoder_stack.encoder_stack__layer_N` children — and there the block is a
    #: frame whose children carry the computation, so validating both would check the
    #: same model twice. `_module_roles` decides that by asking whether the children
    #: carry dataflow; the prefix match it replaces answered "a dotted child exists",
    #: which is a different question and was wrong for 84 templates.
    boundaries, decoration_ids = _module_roles(request, edge_sources, edge_targets)
    frame_ids = set(boundaries)

    for node in request.nodes:
        # Decoration a deeper trace materialised under a module that IS the
        # computation (`<attn>.out_proj`). It carries no edges by construction, so
        # validating it only produces "Node has no input connections" about a node
        # nobody wired.
        if node.id in decoration_ids:
            continue
        # Group/module placeholders are ignored — UNLESS the trace measured one.
        #
        # A module node is a display frame for a user class the parser cannot type,
        # and dropping it was right while it carried nothing. But the trace worker
        # writes the shape it MEASURED onto that same node, and dropping it deleted
        # the one fact that could resolve everything after it. Worse, the edges went
        # with it (see the `included_ids` filter below), so a model that had just
        # executed came back with "Node has no input connections" — carrying a
        # `connect` repair offering to re-make a connection the user already had —
        # plus "unreachable from input" and a global unreachable-output error.
        # Tracing a model made validation WORSE than not tracing it: three errors
        # against the untraced one, on a graph that ran.
        if node.type == "moduleNode" or bool(node.data.isGroup):
            if not _keep_module_node(node, edge_sources, edge_targets, frame_ids):
                continue

        kind = "layer"
        if node.type == "inputNode":
            kind = "input"
        elif node.type == "outputNode":
            kind = "output"
        elif node.type == "moduleNode":
            kind = "module"

        op = _extract_layer_type(node)
        if not op:
            if kind == "input":
                op = "Input"
            elif kind == "output":
                op = "Output"
            else:
                op = "Module"

        params = dict(node.data.parameters or {})
        # Provenance keys are not constructor arguments and are stripped — except the
        # two the validator reads: what the trace measured for this node, and whether
        # the run reached it at all. Stripping those turned a GELU the run never
        # reached into one the static engine confidently sized from out_features.
        params = {
            k: v
            for k, v in params.items()
            if k != "name"
            and (not str(k).startswith("_") or k in ("_shape_provenance", "_out_shape"))
        }

        if kind == "input":
            parsed = _coerce_shape(params.get("shape"))
            if parsed:
                explicit_input_shape = parsed
            params["shape"] = explicit_input_shape or fallback_shape

        canonical_nodes.append(
            ArchitectureNode(
                node_id=node.id,
                kind=kind,  # type: ignore[arg-type]
                op=op,
                params=params,
                in_ports=[] if kind == "input" else ["x"],
                out_ports=[] if kind == "output" else ["y"],
            )
        )
        included_ids.add(node.id)

    edges: List[ArchitectureEdge] = []
    sent_pairs = {(edge.source, edge.target) for edge in request.edges}
    for edge in request.edges:
        # A frame's boundary edges name the FRAME, not the child that actually
        # begins or ends its chain: `input -> features` and `classifier -> output`.
        # Dropping the frame used to delete both, which orphaned `output` and
        # produced "Output node is not connected" on a model that is wired. Rewire
        # onto the entry/exit child instead.
        source, target = edge.source, edge.target
        if source in boundaries:
            source = boundaries[source][1]  # leaving the frame: its exit child
        if target in boundaries:
            target = boundaries[target][0]  # arriving at the frame: its entry child
        # Only a REWIRED edge can be suppressed as a duplicate — a duplicate the
        # caller actually sent is still their graph, and still gets reported.
        if (source, target) != (edge.source, edge.target) and (source, target) in sent_pairs:
            continue
        if source not in included_ids or target not in included_ids:
            continue
        edges.append(
            ArchitectureEdge(
                edge_id=edge.id or f"{source}:y->{target}:x",
                source_node_id=source,
                source_port="y",
                target_node_id=target,
                target_port="x",
            )
        )

    input_shape = explicit_input_shape or fallback_shape
    request_digest = hashlib.sha256(
        json.dumps(request.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ArchitectureGraph(
        graph_id=f"legacy_validate_graph:{request_digest}",
        name="legacy_validate_graph",
        revision=0,
        inputs=[{"name": "x", "shape": input_shape}],
        nodes=canonical_nodes,
        edges=edges,
        modules=[],
        metadata={},
    )


@router.post("/validate_shapes")
def validate_shapes(request: ValidationRequest):
    graph = _build_canonical_graph(request)
    canonical, validator = _canonical_validation()
    validation = validator.validate(graph=graph, strict=False)

    shapes: Dict[str, List[int]] = {}
    errors: List[Dict[str, Any]] = []
    #: cause node -> the nodes it prevented checking (possibly none).
    blocked: Dict[str, List[str]] = {}
    #: every node the validator could not check, victims or not.
    unchecked_causes: List[str] = []

    for node in graph.nodes:
        result = validation.node_results.get(node.node_id)
        if result and result.output_shape:
            shapes[node.node_id] = list(result.output_shape)
        if result and result.blocked_by:
            blocked.setdefault(result.blocked_by, []).append(node.node_id)
        # A block can be unchecked and starve nothing — when the only thing after it
        # is another opaque block, which is its own cause. Keying the report off the
        # victims alone made such a block vanish from the answer entirely, which is
        # the silence this is supposed to end.
        if result and result.message and canonical._UNCHECKED_MARK in result.message and not result.blocked_by:
            unchecked_causes.append(node.node_id)
        if result and result.status == "error":
            suggested_repairs = list(result.suggested_repairs or [])
            errors.append(
                {
                    "nodeId": node.node_id,
                    "message": result.message or "Validation error",
                    "error_type": "validation_error",
                    "suggestion": _summarize_repair_ops(suggested_repairs),
                    "suggested_repairs": suggested_repairs,
                }
            )

    for idx, message in enumerate(validation.global_errors):
        errors.append(
            {
                "nodeId": f"global_{idx}",
                "message": message,
                "error_type": "graph_error",
                "suggestion": "Fix graph structure and connectivity",
            }
        )

    # The third answer. `errors` is named `errors`, and both consumers paint red on
    # membership — ModelBuilder writes `data.error`, and the extension files a
    # Diagnostic — so putting "we could not check this" in there would be a lie told
    # by the structure. It goes in its own array instead, one entry per CAUSE rather
    # than one per starved node, so the collapse from N to 1 is a property of the
    # response and not of how carefully someone worded a message.
    unchecked = [
        {
            "nodeId": cause,
            "message": (
                (validation.node_results.get(cause).message if validation.node_results.get(cause) else None)
                or f"'{cause}' is opaque to static analysis"
            ),
            "blocked": sorted(blocked.get(cause, [])),
        }
        for cause in sorted(set(unchecked_causes) | set(blocked))
    ]

    return {
        # A graph nobody could fully check is not a graph that failed. "valid" still
        # means "no errors"; `unchecked` says what the tick does not cover, which is
        # the distinction the whole of #196 turns on.
        "status": "valid" if not errors else "error",
        "shapes": shapes,
        "errors": errors,
        "unchecked": unchecked,
    }
