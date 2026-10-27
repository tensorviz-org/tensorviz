"""Model verification service.

Runs the architecture through shape inference and an optional dry forward
pass on synthetic input, then reports per-node status so the UI can light
up which layer broke. Used by /api/models/verify.
"""
from __future__ import annotations

import io
import logging
import traceback
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch

from core.layer_builder import build_layer, call_module, topological_sort
from core.op_registry import is_placeholder_op, spec_for
from core.shape_inference import infer_hierarchical_shapes
from models.architecture_models import ArchitectureGraph
from models.graph_models import Graph, HierarchicalGraph
from services.code_generation import (
    _FUNCTIONAL_REQUIRED,
    _FUNCTIONAL_REQUIRED_ANY,
    CodeGenerationService,
)

logger = logging.getLogger(__name__)


#: Ops that PRODUCE a tensor and consume nothing. Imported rather than redeclared: three
#: engines now agree on this set, and a fourth spelling of it is how the first three drifted.
from core.shape_inference import _SOURCE_OPS  # noqa: E402


def _bare_source_op(layer_type) -> str:
    """`torch.nn.Parameter` -> `parameter`, from either a dict or a model object."""
    name = ""
    if isinstance(layer_type, dict):
        name = layer_type.get("name") or layer_type.get("id") or ""
    else:
        name = getattr(layer_type, "name", None) or getattr(layer_type, "id", None) or ""
    return str(name).rsplit(".", 1)[-1].lower()


def _coerce_input_shape(raw: Any, fallback: Optional[List[int]] = None) -> List[int]:
    """Best-effort conversion of a user-supplied input shape into a list of ints.
    Accepts list[int], list[str], comma-separated string, or falls back."""
    if isinstance(raw, list):
        try:
            return [int(x) for x in raw if x not in (None, "")]
        except (TypeError, ValueError):
            pass
    if isinstance(raw, str):
        try:
            return [int(x.strip()) for x in raw.split(",") if x.strip()]
        except (TypeError, ValueError):
            pass
    return list(fallback or [1, 3, 224, 224])


# Lowercase layer-id → canonical torch class name. The frontend palette
# stores ids like "conv2d"/"relu" but layer_builder.build_layer matches
# exact class names like "Conv2d"/"ReLU". This translation table covers
# the layers users can drop on the canvas; anything not in here falls
# back to layerType.name.
_LAYER_ID_TO_CLASS: Dict[str, str] = {
    "conv1d": "Conv1d", "conv2d": "Conv2d", "conv3d": "Conv3d",
    "convtranspose1d": "ConvTranspose1d",
    "convtranspose2d": "ConvTranspose2d",
    "convtranspose3d": "ConvTranspose3d",
    "linear": "Linear",
    "relu": "ReLU", "leakyrelu": "LeakyReLU", "sigmoid": "Sigmoid",
    "tanh": "Tanh", "gelu": "GELU", "softmax": "Softmax", "silu": "SiLU",
    "batchnorm1d": "BatchNorm1d", "batchnorm2d": "BatchNorm2d", "batchnorm3d": "BatchNorm3d",
    "layernorm": "LayerNorm", "groupnorm": "GroupNorm", "instancenorm2d": "InstanceNorm2d",
    # rmsnorm is a REAL torch class (nn.RMSNorm, torch >= 2.4), not a custom op — it was
    # only unbuildable because the templates emit the lowercase id and build_layer matches
    # class names. Without this line it fell through to getattr(nn, "rmsnorm").
    "rmsnorm": "RMSNorm",
    "maxpool1d": "MaxPool1d", "maxpool2d": "MaxPool2d", "maxpool3d": "MaxPool3d",
    "avgpool1d": "AvgPool1d", "avgpool2d": "AvgPool2d", "avgpool3d": "AvgPool3d",
    "adaptiveavgpool1d": "AdaptiveAvgPool1d", "adaptiveavgpool2d": "AdaptiveAvgPool2d",
    "adaptiveavgpool3d": "AdaptiveAvgPool3d",
    "dropout": "Dropout", "dropout2d": "Dropout2d",
    "flatten": "Flatten",
    "embedding": "Embedding",
    "lstm": "LSTM", "gru": "GRU", "rnn": "RNN",
    "multiheadattention": "MultiheadAttention",
    "transformerencoderlayer": "TransformerEncoderLayer",
    "transformerdecoderlayer": "TransformerDecoderLayer",
}


def _resolve_layer_class(layer_type_obj: Any) -> str:
    """Map a LayerType-like object to the torch nn class name."""
    name = getattr(layer_type_obj, "name", None) or (layer_type_obj.get("name") if isinstance(layer_type_obj, dict) else None)
    id_ = getattr(layer_type_obj, "id", None) or (layer_type_obj.get("id") if isinstance(layer_type_obj, dict) else None)
    if id_ and id_.lower() in _LAYER_ID_TO_CLASS:
        return _LAYER_ID_TO_CLASS[id_.lower()]
    if name:
        return name
    if id_:
        return id_
    return "Unknown"


def _op_key(layer_type_obj: Any) -> str:
    """Lowercased op identity (id preferred, else name) with nn./torch. prefixes
    stripped — used to recognize merge/custom-passthrough ops in the forward."""
    id_ = getattr(layer_type_obj, "id", None) or (layer_type_obj.get("id") if isinstance(layer_type_obj, dict) else None)
    name = getattr(layer_type_obj, "name", None) or (layer_type_obj.get("name") if isinstance(layer_type_obj, dict) else None)
    key = (id_ or name or "").strip().lower()
    for prefix in ("torch.nn.", "nn.", "torch."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key

# Merge ops combine multiple inputs into one tensor (no torch.nn module to build).
_MERGE_OP_KEYS = {
    "add", "torch.add", "sum", "torch.sum",
    "mul", "torch.mul", "multiply",
    "cat", "concat", "concatenate", "torch.cat", "torch.concat",
    "stack", "torch.stack",
}

# Shape-preserving custom blocks with no torch.nn equivalent. The dry forward
# can't execute their real math (rotary phase rotation, global response norm,
# adaLN time/text modulation, the sinusoidal timestep basis), so they run as a
# shape-preserving identity on their primary input — same structural-pass
# convention the canonical validator uses (custom_passthrough_ops). Real torch
# ops (Identity/RMSNorm/LayerNorm) are NOT here; they build and run normally.
_CUSTOM_PASSTHROUGH_OPS = {
    "grn", "global_response_norm",
    # gaussian_noise (a diffusion block's forward-process step) is x + sigma*eps:
    # shape-preserving, and the dry forward has no reason to sample the noise.
    # code_generation emits the real addition, so exported and verified shapes agree.
    "gaussian_noise",
    "rotary_embedding", "rope",
    "ada_layer_norm", "ada_layer_norm_zero", "ada_layer_norm_final", "adalayernorm",
    "sinusoidal_pos_embedding", "sinusoidal_position_embedding", "sinus_position_embedding",
}

# nn containers: they HOLD submodules rather than compute, so their constructor
# argument is other nn.Modules and no JSON parameter bag can ever carry one. None of
# them is a registered op (`spec_for` answers None for every one), so they only ever
# reach here from the static tier reading a user's own `__init__` — where the members
# are display data on one card, not a bag to construct from. See the skip in the
# leaf-layer loop below for what that cost the user.
#
# `nn.Transformer` is deliberately NOT here: it takes scalars (d_model, nhead, ...),
# builds from them and computes, so it is a layer like any other.
_CONTAINER_OP_KEYS = {
    "modulelist", "moduledict", "sequential",
    "parameterlist", "parameterdict",
    "transformerencoder", "transformerdecoder",
}


# Ops the runtime executes as a torch.*/F.* CALL rather than as an nn.Module.
# code_generation dispatches on exactly this rule — `layerType.name` starting with
# "torch.", plus the bare-canonical functional names — so verify and export cannot
# disagree about which side of the line an op falls on. There are 44 of them in the
# catalogue, and every one that is not also a merge op used to fail the dry forward
# with "Unknown layer type 'torch.narrow': not found in torch.nn", because this
# stage's only way to run a node was to build an nn module for it.
#
# `interpolate` carries a BARE canonical rather than a prefixed one: the
# _LAYER_OPERATION guard in graph_input_safety allows at most one prefix level, so
# "torch.nn.functional.interpolate" is rejected on purpose and the bare name has to
# be named explicitly here. Same reason code_generation keeps _FUNCTIONAL_NAMESPACE.
_FUNCTIONAL_BARE_OPS = {"interpolate"}


def _is_functional_op_name(op: str) -> bool:
    """True when `op` names a torch function call, not an nn.Module class."""
    name = str(op or "").strip()
    if name.startswith("torch.nn."):
        return False
    return name.startswith("torch.") or name.lower() in _FUNCTIONAL_BARE_OPS


def _is_functional_layer_type(layer_type_obj: Any) -> bool:
    """The object form, reading the same `name` field code_generation reads."""
    name = getattr(layer_type_obj, "name", None) or (
        layer_type_obj.get("name") if isinstance(layer_type_obj, dict) else None
    )
    return _is_functional_op_name(name or "") or _op_key(layer_type_obj) in _FUNCTIONAL_BARE_OPS


def _functional_output_shape(
    layer_type_obj: Any,
    params: Dict[str, Any],
    inputs: List["torch.Tensor"],
) -> Optional[List[int]]:
    """The output shape of a functional op, from the op registry.

    Deliberately not a second implementation of narrow/reshape/permute/... in this
    file. `shape_fn` is the same function shape inference and validation call, and
    tests/validation/test_op_shape_parity.py checks 38 of them against a real torch call — so
    the dry forward cannot report a shape the card disagrees with. Returns None when
    the op declines (an out-of-range slice, a multi-output op like split/chunk),
    which the caller surfaces as an error rather than guessing.
    """
    name = getattr(layer_type_obj, "name", None) or (
        layer_type_obj.get("name") if isinstance(layer_type_obj, dict) else None
    )
    spec = spec_for(str(name or "")) or spec_for(_op_key(layer_type_obj))
    if spec is None or spec.shape_fn is None:
        return None
    shapes = [list(t.shape) for t in inputs]
    try:
        return spec.shape_fn(params or {}, shapes[0] if shapes else [], shapes)
    except Exception:
        return None


def _missing_functional_params(op_key: str, params: Dict[str, Any]) -> List[str]:
    """Which required arguments of a functional op are absent or left blank.

    The same two tables code_generation refuses to emit on, so verify names the
    parameter instead of reporting an unexplained "cannot be applied": a palette
    permute carries `dims` unset, and "needs dims" is actionable where the shape
    engine's decline is not. A node that fails here fails Export for the same
    reason, which is the point of sharing the tables.
    """
    supplied = {
        key for key, value in (params or {}).items()
        if value != "" and value is not None and not key.startswith("_") and key != "name"
    }
    missing = [key for key in _FUNCTIONAL_REQUIRED.get(op_key, ()) if key not in supplied]
    either = _FUNCTIONAL_REQUIRED_ANY.get(op_key)
    if either and not (supplied & set(either)):
        missing.append(" or ".join(either))
    return missing


def _apply_merge(op_key: str, inputs: List["torch.Tensor"], params: Dict[str, Any]) -> "torch.Tensor":
    """Combine multiple input tensors for a merge op. cat concatenates (default
    last dim), stack stacks, mul multiplies, add/sum sum elementwise (all with
    broadcasting). A single input passes through unchanged."""
    from core.scalar_operations import apply_scalar
    scalar_result = apply_scalar(op_key, inputs, params)
    if scalar_result is not None:
        return scalar_result
    if op_key in {"cat", "concat", "concatenate", "torch.cat", "torch.concat"}:
        try:
            dim = int(params.get("dim", -1))
        except (TypeError, ValueError):
            dim = -1
        return torch.cat(inputs, dim=dim)
    if op_key in {"stack", "torch.stack"}:
        try:
            dim = int(params.get("dim", 0))
        except (TypeError, ValueError):
            dim = 0
        return torch.stack(inputs, dim=dim)
    out = inputs[0]
    if op_key in {"mul", "torch.mul", "multiply"}:
        for t in inputs[1:]:
            out = out * t
        return out
    for t in inputs[1:]:  # add / torch.add / sum / torch.sum
        out = out + t
    return out


def _build_synthetic_input(shape: List[int], *, use_long: bool = False, high: int = 100) -> torch.Tensor:
    """Create a small synthetic tensor matching the requested shape.

    Uses float32 by default. Set use_long=True only when the graph input feeds
    directly into an Embedding layer (token indices); `high` then bounds the
    indices to the Embedding's num_embeddings so a small vocab isn't fed
    out-of-range ids.
    """
    safe_shape = [max(1, int(d)) for d in shape] or [1]
    if use_long:
        return torch.randint(low=0, high=max(1, int(high)), size=tuple(safe_shape))
    return torch.randn(*safe_shape)


def _flatten_for_torch(graph_dict: Dict[str, Any]) -> Graph:
    """Flatten a hierarchical graph payload into a flat Graph of leaf nodes, and
    REWIRE module-boundary edges through to the module internals so the forward
    pass stays connected. A root edge `a -> module` becomes `a -> module.entry`
    and `module -> b` becomes `module.exit -> b`; the module wrapper node itself
    is dropped (it has no tensor). Without this the module internals would be
    orphaned and receive the wrong input."""
    root_nodes = list(graph_dict.get("root_nodes") or graph_dict.get("nodes") or [])
    root_edges = list(graph_dict.get("root_edges") or graph_dict.get("edges") or [])
    modules = graph_dict.get("modules") or {}

    def _id(n):
        return n.get("id") if isinstance(n, dict) else getattr(n, "id", None)
    def _get(obj, key):
        return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

    entry_exit: Dict[str, Tuple[Any, Any]] = {}
    leaf_nodes: List[Any] = []
    internal_edges: List[Any] = []
    for mid, mod in modules.items():
        internal = _get(mod, "internal_nodes") or []
        leaf_nodes.extend(internal)
        internal_edges.extend(_get(mod, "internal_edges") or [])
        entry = _get(mod, "input_node_id") or (_id(internal[0]) if internal else None)
        exit_ = _get(mod, "output_node_id") or (_id(internal[-1]) if internal else None)
        node_mid = _get(mod, "module_id") or mid
        entry_exit[node_mid] = (entry, exit_)

    module_ids = set(entry_exit.keys())
    flat_nodes = [n for n in root_nodes if _id(n) not in module_ids] + leaf_nodes

    def remap_source(s):
        return entry_exit[s][1] if s in entry_exit else s
    def remap_target(t):
        return entry_exit[t][0] if t in entry_exit else t

    flat_edges: List[Any] = []
    for e in root_edges:
        src, tgt = _get(e, "source"), _get(e, "target")
        flat_edges.append({"id": _get(e, "id") or f"{src}->{tgt}",
                           "source": remap_source(src), "target": remap_target(tgt)})
    flat_edges.extend(internal_edges)
    return Graph(nodes=flat_nodes, edges=flat_edges)


_REGISTRY = None


def _get_registry():
    """Lazy singleton for the canonical template registry (imported lazily to
    avoid a heavy import when verification runs on plain layer graphs)."""
    global _REGISTRY
    if _REGISTRY is None:
        from services.canonical_template_registry import CanonicalTemplateRegistry
        _REGISTRY = CanonicalTemplateRegistry()
    return _REGISTRY


def _normalize_op_name(op: str) -> str:
    """Registry templates emit ops like 'nn.Linear' / 'torch.add'. Strip the
    module prefix and map to the bare torch class name build_layer expects."""
    s = str(op or "").strip()
    for pref in ("torch.nn.", "nn.", "torch.", "functional.", "F."):
        if s.startswith(pref):
            s = s[len(pref):]
            break
    return _LAYER_ID_TO_CLASS.get(s.lower(), s)


def _arch_node_to_flat(node: Any) -> Dict[str, Any]:
    """Convert a registry ArchitectureNode into the flat canvas node shape the
    verify path consumes (id / type / data.layerType / data.parameters)."""
    op = _normalize_op_name(getattr(node, "op", ""))
    kind = getattr(node, "kind", "layer")
    node_type = "inputNode" if kind == "input" else "outputNode" if kind == "output" else "layerNode"
    return {
        "id": getattr(node, "node_id"),
        "type": node_type,
        "position": {"x": 0.0, "y": 0.0},
        "data": {
            "layerType": {"id": op, "name": op, "parameters": []},
            "parameters": dict(getattr(node, "params", {}) or {}),
        },
    }


def architecture_graph_to_runtime_payload(graph: ArchitectureGraph) -> Dict[str, Any]:
    """Project the canonical semantic graph into the runtime/codegen schema.

    The verification service predates ``ArchitectureGraph`` and consumes the
    canvas-shaped ``Graph`` model. Keeping the conversion here gives validation,
    verification, and export one explicit boundary instead of relying on callers
    to recreate a subtly different payload.
    """

    candidate = graph.model_copy(deep=True)

    # A kind="module" node with NO incident edge is canvas grouping, not computation: the
    # dataflow runs through its children and the parent/child link lives in the VIEW, so a
    # semantic-only consumer sees an orphan. Passed through, codegen turned an imported
    # resnet18's block card into `nn.BasicBlock` and the candidate export check failed with
    # "module 'torch.nn' has no attribute 'BasicBlock'". Its children are already in the
    # payload, so dropping the frame loses nothing executable.
    incident = set()
    for edge in candidate.edges:
        incident.add(edge.source_node_id)
        incident.add(edge.target_node_id)
    executable = [
        node
        for node in candidate.nodes
        if not (node.kind == "module" and node.node_id not in incident)
    ]

    return {
        "nodes": [_arch_node_to_flat(node) for node in executable],
        "edges": [
            {
                "id": edge.edge_id,
                "source": edge.source_node_id,
                "target": edge.target_node_id,
            }
            for edge in candidate.edges
        ],
    }


def architecture_graph_runtime_support(
    graph: ArchitectureGraph,
) -> tuple[bool, List[str]]:
    """Report whether ``verify_model`` can run a faithful PyTorch forward.

    Merge operations are executed faithfully. Custom passthroughs and semantic
    module placeholders are intentionally unsupported: ``verify_model`` treats
    those as identities, which is useful for canvas shape previews but is not
    evidence of the model's real computation.
    """

    reasons: List[str] = []
    if graph.modules:
        reasons.append(
            "ArchitectureGraph.modules must be materialized as explicit nodes and edges"
        )

    for node in graph.nodes:
        if node.kind in {"input", "output"}:
            continue
        if node.kind == "module":
            reasons.append(f"{node.node_id}: semantic module has no materialized runtime")
            continue

        normalized = _normalize_op_name(node.op)
        key = normalized.strip().lower()
        if key in _MERGE_OP_KEYS:
            continue
        if key in _CUSTOM_PASSTHROUGH_OPS:
            reasons.append(
                f"{node.node_id}: '{node.op}' is only a structural runtime passthrough"
            )
            continue

        # A functional op has no nn class by definition, so looking for one rejected
        # every tensor op an AI turn proposed — a Slice, a reshape, a permute — as
        # having "no faithful PyTorch runtime". The registry knowing it IS the
        # runtime: the forward below executes it through the same shape_fn.
        if _is_functional_op_name(node.op) or key in _FUNCTIONAL_BARE_OPS:
            if spec_for(str(node.op or "")) is None and spec_for(key) is None:
                reasons.append(f"{node.node_id}: no faithful PyTorch runtime for '{node.op}'")
            continue

        layer_class = getattr(torch.nn, normalized, None)
        if not (
            isinstance(layer_class, type)
            and issubclass(layer_class, torch.nn.Module)
        ):
            reasons.append(f"{node.node_id}: no faithful PyTorch runtime for '{node.op}'")

    return not reasons, reasons


def _expand_module_templates(graph_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Replace hollow template module nodes (those carrying
    data.moduleTemplateId) with their real leaf layers via the canonical
    template registry, rewiring boundary edges through the template
    entry/exit. Keeps the payload flat so the existing verify path runs on
    real layers instead of treating the module as an identity passthrough.

    No-op for plain graphs, already-hierarchical payloads, or unknown
    templates (those are left untouched so behaviour only ever improves)."""
    if not isinstance(graph_payload, dict):
        return graph_payload
    if (graph_payload.get("modules")) or graph_payload.get("root_nodes"):
        return graph_payload
    nodes = graph_payload.get("nodes")
    if not isinstance(nodes, list):
        return graph_payload

    def _data(n):
        d = n.get("data") if isinstance(n, dict) else None
        return d if isinstance(d, dict) else {}

    if not any(_data(n).get("moduleTemplateId") for n in nodes):
        return graph_payload

    registry = _get_registry()
    new_nodes: List[Any] = []
    new_edges: List[Any] = list(graph_payload.get("edges") or [])
    io_map: Dict[str, Tuple[str, str]] = {}

    for n in nodes:
        tpl = _data(n).get("moduleTemplateId")
        if not tpl:
            new_nodes.append(n)
            continue
        node_id = n.get("id")
        config = {k: v for k, v in (_data(n).get("parameters") or {}).items() if k != "name"}
        try:
            anodes, aedges, entry, exit_, _cfg = registry.instantiate(
                template_id=tpl, instance_id=node_id, config=config,
            )
        except Exception:
            # Unknown / uninstantiable template — leave the module node as-is.
            new_nodes.append(n)
            continue
        for an in anodes:
            new_nodes.append(_arch_node_to_flat(an))
        for ae in aedges:
            new_edges.append({"id": ae.edge_id, "source": ae.source_node_id, "target": ae.target_node_id})
        io_map[node_id] = (entry, exit_)

    if not io_map:
        return graph_payload

    def _remap(edge):
        src, tgt = edge.get("source"), edge.get("target")
        if src in io_map:
            src = io_map[src][1]
        if tgt in io_map:
            tgt = io_map[tgt][0]
        return {"id": edge.get("id") or f"{src}->{tgt}", "source": src, "target": tgt}

    out = dict(graph_payload)
    out["nodes"] = new_nodes
    out["edges"] = [_remap(e) for e in new_edges]
    return out


def _reconstruct_hierarchy(graph_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild a hierarchical graph from a FLAT payload that still carries
    module structure metadata (a `moduleNode` plus children tagged with
    `data.parentId`). The frontend flattens hierarchical templates (e.g. ViT)
    for the canvas; the flat verify path then can't rewire the module
    boundary, so the module's internals get the wrong input and shape/forward
    fails. Reconstructing the `modules` dict routes the graph back through the
    working hierarchical path (infer_hierarchical_shapes + boundary-aware
    flatten).

    No-op for already-hierarchical payloads or flat graphs with no modules."""
    if not isinstance(graph_payload, dict):
        return graph_payload
    if graph_payload.get("modules") or graph_payload.get("root_nodes"):
        return graph_payload
    nodes = graph_payload.get("nodes")
    if not isinstance(nodes, list):
        return graph_payload

    def _data(n):
        d = n.get("data") if isinstance(n, dict) else None
        return d if isinstance(d, dict) else {}

    def _parent(n):
        return _data(n).get("parentId")

    module_nodes = [n for n in nodes if n.get("type") == "moduleNode" or _data(n).get("isGroup")]
    if not module_nodes:
        return graph_payload

    children_by_parent: Dict[str, List[Any]] = {}
    for n in nodes:
        p = _parent(n)
        if p:
            children_by_parent.setdefault(p, []).append(n)

    edges = graph_payload.get("edges") or []
    child_ids = {n.get("id") for n in nodes if _parent(n)}

    # Reconstruction exists for a graph the CLIENT flattened, where edges terminate on
    # the module box and its internals would otherwise be unreachable. An imported graph
    # is the opposite: every edge is child-level real dataflow and the module cards are
    # display-only grouping, carrying no edges at all. Reconstructing that graph replaces
    # exact per-layer dataflow with a coarse box-to-box chain, and the boxes then fail to
    # build because a group's display name ("BasicBlock x2") is not a layer type. So when
    # nothing references a module box, the flat graph is already complete — verify it as
    # it is. Measured: resnet18 depth 2 goes from 11 errors to a full forward pass.
    module_ids = {mn.get("id") for mn in module_nodes}
    if child_ids and not any(
        e.get("source") in module_ids or e.get("target") in module_ids for e in edges
    ):
        return graph_payload

    modules: Dict[str, Any] = {}
    node_hierarchy: Dict[str, str] = {}
    for mn in module_nodes:
        mid = mn.get("id")
        children = children_by_parent.get(mid, [])
        if not children:
            # Hollow module (no inlined children) — leave it for template
            # expansion / identity handling rather than forcing a bad module.
            continue
        cids = {c.get("id") for c in children}
        internal_edges = [e for e in edges if e.get("source") in cids and e.get("target") in cids]
        d = _data(mn)
        entry = d.get("moduleEntryNodeId") or children[0].get("id")
        exit_ = d.get("moduleExitNodeId") or children[-1].get("id")
        modules[mid] = {
            "module_id": mid,
            "parent_id": d.get("parentId"),
            "internal_nodes": children,
            "internal_edges": internal_edges,
            "input_node_id": entry,
            "output_node_id": exit_,
            "parameters": d.get("parameters") or {},
        }
        for c in children:
            node_hierarchy[c.get("id")] = mid

    if not modules:
        return graph_payload

    root_nodes = [n for n in nodes if not _parent(n)]

    # An edge that CROSSES a module boundary must be re-pointed at the module, not
    # dropped. Dropping it left the node after a module with no input at all: an
    # imported resnet18 reported "Node avgpool has no input connections" (11 such
    # errors) because every edge out of a block came from a child — while the same
    # graph verified fine once the grouping was stripped. This mirrors what the canvas
    # does with a collapsed module (see ModelEdgeData.synthetic in types/index.ts).
    def _to_root(node_id: Any) -> Any:
        seen_ids = set()
        current = node_id
        while current in node_hierarchy and current not in seen_ids:
            seen_ids.add(current)
            current = node_hierarchy[current]
        return current

    root_edges: List[Any] = []
    seen_pairs = set()
    for e in edges:
        source, target = _to_root(e.get("source")), _to_root(e.get("target"))
        if source == target:
            # Both endpoints resolved into the same module: an internal edge, already
            # carried by that module's internal_edges.
            continue
        pair = (source, target)
        if pair in seen_pairs:
            # Several children of one module feeding the same neighbour collapse to one
            # boundary edge, exactly as the canvas dedupes them.
            continue
        seen_pairs.add(pair)
        root_edges.append({**e, "source": source, "target": target})
    return {
        "root_nodes": root_nodes,
        "root_edges": root_edges,
        "modules": modules,
        "node_hierarchy": node_hierarchy,
    }


def verify_model(
    graph_payload: Dict[str, Any],
    input_shape: Any = None,
    run_forward: bool = True,
) -> Dict[str, Any]:
    """Run verification on a flat or hierarchical graph payload.

    Returns:
      {
        "ok": bool,
        "stage": "build" | "shape_inference" | "forward" | "complete",
        "input_shape": [...],
        "shapes": { node_id: shape | error },
        "output_shape": [...] | None,
        "summary": { layer_count, param_count, ... } | None,
        "errors": [ { node_id?, message } ],
        "trace": str | None,
      }
    """
    graph_payload = _expand_module_templates(graph_payload)
    graph_payload = _reconstruct_hierarchy(graph_payload)
    is_hier = ("modules" in graph_payload and graph_payload["modules"]) or "root_nodes" in graph_payload
    shape = _coerce_input_shape(input_shape, [1, 3, 224, 224])
    result: Dict[str, Any] = {
        "ok": False,
        "stage": "shape_inference",
        "input_shape": shape,
        "shapes": {},
        "output_shape": None,
        "errors": [],
        "trace": None,
        "summary": None,
    }

    # ── Stage 1: shape inference ─────────────────────────────────────────
    try:
        if is_hier:
            hg = HierarchicalGraph(**graph_payload)
            # infer_hierarchical_shapes returns {"shapes", "errors"} — it has no
            # output shape to give. Unpacking it as a (shapes, output_shape)
            # tuple silently yielded the dict's KEYS, so the response carried the
            # literal strings "shapes"/"errors" instead of shape data. The dry
            # forward below fills output_shape from the real tensor.
            inferred = infer_hierarchical_shapes(hg, shape)
            result["shapes"] = inferred.get("shapes", {})
            hier_errors = inferred.get("errors") or []
            if any(n.data.layerType.name == "Input" for n in hg.root_nodes):
                # A module's internals never contain an Input node — their input
                # arrives across the module boundary — but the flat inference run
                # on those internals reports `missing_input` regardless. Those are
                # structural artefacts once the root graph has its Input node; a
                # genuinely input-less root still reports.
                hier_errors = [e for e in hier_errors if e.get("error_type") != "missing_input"]
            # One root-level problem is reported once per module by the per-module
            # inference run, so an input-less graph produced the same error a dozen
            # times. Deduplicate on (error_type, node_id, message).
            seen: set = set()
            deduped = []
            for err in hier_errors:
                key = (err.get("error_type"), err.get("node_id"), err.get("message"))
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(err)
            result["errors"].extend(deduped)
        else:
            from core.shape_inference import ShapeInferenceService
            flat = _flatten_for_torch(graph_payload)
            inferred = ShapeInferenceService().infer_shapes(flat, shape)
            result["shapes"] = inferred.get("shapes", {})
            result["output_shape"] = inferred.get("output_shape")
    except Exception as exc:
        result["errors"].append({"message": f"Shape inference failed: {exc}"})
        result["trace"] = traceback.format_exc()
        return result

    if not run_forward:
        # ok must not contradict the errors we just surfaced. Before hierarchical
        # inference errors were reported at all this was unconditionally True, which is
        # how a graph with real structural errors could answer ok=True.
        result["ok"] = not result["errors"]
        result["stage"] = "complete"
        return result

    # ── Stage 2: dry torch forward pass on synthetic input ───────────────
    result["stage"] = "build"
    try:
        flat = _flatten_for_torch(graph_payload)
        # build leaf layers
        leaf_layers: Dict[str, torch.nn.Module] = {}
        for n in flat.nodes:
            if n.type != "layerNode":
                continue
            # Merge ops (add/cat/...) and shape-preserving custom blocks (GRN,
            # RoPE, adaLN, ...) have no torch.nn module — they're handled
            # directly in the forward loop, so skip building a layer for them.
            if _op_key(n.data.layerType) in _MERGE_OP_KEYS or _op_key(n.data.layerType) in _CUSTOM_PASSTHROUGH_OPS:
                continue
            # Functional torch.*/F.* ops are called in the forward, not constructed
            # here. Without this, build_layer was handed the op NAME and looked it up
            # in torch.nn: "Unknown layer type 'torch.narrow'".
            if _is_functional_layer_type(n.data.layerType):
                continue
            # `python` / `custom_code` are the user's own source, not an op. There is
            # nothing to construct — the real object lives in their file — so building
            # one would fail with "Unknown layer type 'custom_code'" and take the whole
            # verification down with it. Left out of leaf_layers, the forward below
            # reaches its `_out_shape` branch and uses what the trace measured, which is
            # the only thing anyone knows about this node (task.md M6).
            if is_placeholder_op(_op_key(n.data.layerType)):
                continue
            # A SOURCE is not a layer to build. `nn.Parameter` is a Tensor, so
            # `build_layer` hands back something with no `.eval()` and no `__call__`,
            # and the smoke-test loop below died on
            #     Model build failed: 'Parameter' object has no attribute 'eval'
            # — reported to the user as a broken model, on a graph the extension's own
            # validator calls clean. It is seeded with its own tensor in the forward
            # stage instead, exactly like an input node.
            if _bare_source_op(n.data.layerType) in _SOURCE_OPS:
                continue
            # A CONTAINER holds submodules; it is not computation with a parameter
            # bag. Its constructor argument is other nn.Modules, and no graph payload
            # can carry one — `nn.ModuleList` got a list of `{op, params}` dicts and
            # `nn.TransformerEncoder(layer, 2)` got nothing at all, because `layer` is
            # a local variable in the user's __init__. Both aborted the whole build
            # and told the user their model was broken:
            #     ModuleList — build error: dict is not a Module subclass
            #     TransformerEncoder — build error: missing 1 required positional
            #                          argument: 'encoder_layer'
            # — on models that are correct. A container the static tier could not read
            # is structurally unbuildable, which is a fact about containers, not a
            # fault to report. Left out of leaf_layers it reaches the forward's
            # identity/`_out_shape` branch, the same place `custom_code` lands.
            if _op_key(n.data.layerType) in _CONTAINER_OP_KEYS:
                continue
            try:
                cls_name = _resolve_layer_class(n.data.layerType)
                leaf_layers[n.id] = build_layer(cls_name, n.data.parameters or {})
            except Exception as build_err:
                result["errors"].append({"node_id": n.id, "message": f"build error: {build_err}"})
                result["trace"] = traceback.format_exc()
                return result

        order = topological_sort(flat.nodes, flat.edges)
        node_by_id = {n.id: n for n in flat.nodes}
        # Smoke-test in eval mode: a single-sample dry forward must not trip
        # BatchNorm's train-mode "expected more than 1 value per channel" once a
        # deep conv stack collapses the spatial dims to 1x1.
        for _layer in leaf_layers.values():
            _layer.eval()
    except Exception as exc:
        result["errors"].append({"message": f"Model build failed: {exc}"})
        result["trace"] = traceback.format_exc()
        return result

    result["stage"] = "forward"
    try:
        # Seed each input node with a synthetic tensor of ITS OWN shape/dtype.
        # Multi-input models (e.g. a TTS model taking text ids + timestep + mel)
        # have input nodes of differing shapes, so a single shared tensor won't
        # do. Long is used per-node only when that input feeds an Embedding.
        _embedding_ops = {"nn.Embedding", "Embedding", "embedding"}

        def _embedding_vocab(node_id: str) -> Optional[int]:
            """If this input feeds one or more Embedding layers, return the
            smallest num_embeddings among them (a synthetic id must be < every
            target vocab); else None. Bounding ids by the real vocab stops a
            valid small-vocab embedding graph being false-rejected."""
            vocabs: List[int] = []
            for e in flat.edges:
                if e.source != node_id:
                    continue
                tgt = node_by_id.get(e.target)
                if tgt is not None and getattr(tgt, "type", "") == "layerNode" \
                        and _resolve_layer_class(tgt.data.layerType) in _embedding_ops:
                    try:
                        vocabs.append(int((tgt.data.parameters or {}).get("num_embeddings", 100)))
                    except (TypeError, ValueError):
                        vocabs.append(100)
            return min(vocabs) if vocabs else None

        def _node_shape(n) -> List[int]:
            raw = (n.data.parameters or {}).get("shape") if n.data else None
            if isinstance(raw, str) and raw.strip():
                try:
                    return [int(v) for v in raw.split(",")]
                except ValueError:
                    pass
            if isinstance(raw, (list, tuple)) and raw:
                try:
                    return [int(v) for v in raw]
                except (TypeError, ValueError):
                    pass
            return list(shape)

        def _source_shape(n) -> Optional[List[int]]:
            """A source's own shape: what it contributes, since nothing flows into it."""
            params = (n.data.parameters or {}) if n.data else {}
            for key in ("_out_shape", "shape"):
                raw = params.get(key)
                if isinstance(raw, str) and raw.strip():
                    for sep in ("x", ","):
                        if sep in raw:
                            try:
                                return [int(v) for v in raw.split(sep)]
                            except ValueError:
                                continue
                if isinstance(raw, (list, tuple)) and raw:
                    try:
                        return [int(v) for v in raw]
                    except (TypeError, ValueError):
                        pass
            return None

        tensors: Dict[str, torch.Tensor] = {}
        x = None
        # Sources first: they are roots like the input, and whatever consumes them needs
        # their tensor present before the topological walk reaches it.
        for n in flat.nodes:
            if n.data is None or _bare_source_op(n.data.layerType) not in _SOURCE_OPS:
                continue
            src_shape = _source_shape(n)
            if src_shape:
                tensors[n.id] = torch.zeros(*src_shape)
        for n in flat.nodes:
            if n.type == "inputNode":
                vocab = _embedding_vocab(n.id)
                t = _build_synthetic_input(
                    _node_shape(n),
                    use_long=vocab is not None,
                    high=vocab if vocab is not None else 100,
                )
                tensors[n.id] = t
                if x is None:
                    x = t
        if x is None:  # no explicit input node — fall back to the passed shape
            x = _build_synthetic_input(shape)

        last_tensor = x
        for node_id in order:
            n = node_by_id.get(node_id)
            if n is None:
                continue
            if n.type == "inputNode":
                continue
            if n.data is not None and _bare_source_op(n.data.layerType) in _SOURCE_OPS:
                continue  # seeded above; there is nothing to apply
            # gather input tensors via edges
            inputs = []
            for e in flat.edges:
                if e.target == node_id and e.source in tensors:
                    inputs.append(tensors[e.source])
            if not inputs:
                # fall back to the most recently produced tensor — handles
                # graphs where an embedding/linear starts the chain but the
                # explicit input edge was missing.
                inputs = [last_tensor]

            if n.type == "layerNode":
                op_key = _op_key(n.data.layerType)
                try:
                    if op_key in _MERGE_OP_KEYS:
                        # Multi-input combiner (cat/add/mul/...). Runs even on a
                        # single input (identity), which keeps the graph valid.
                        t_out = _apply_merge(op_key, inputs, n.data.parameters or {})
                    elif _is_functional_layer_type(n.data.layerType):
                        # A torch function call. Ask the registry what it produces and
                        # hand on a tensor of that shape — the same convention the
                        # imported-block branch below uses. Must sit ABOVE that branch:
                        # a functional op has no entry in leaf_layers, so it would
                        # otherwise fall into the identity passthrough and the dry
                        # forward would silently keep the UNSLICED tensor.
                        shape = _functional_output_shape(
                            n.data.layerType, n.data.parameters or {}, inputs
                        )
                        if shape is None:
                            cls_name = _resolve_layer_class(n.data.layerType)
                            missing = _missing_functional_params(op_key, n.data.parameters or {})
                            if missing:
                                message = (
                                    f"{cls_name} needs {' and '.join(missing)} before it can "
                                    f"run — set {'them' if len(missing) > 1 else 'it'} in Properties"
                                )
                            else:
                                dims = "x".join(str(d) for d in inputs[0].shape)
                                message = (
                                    f"shape error: {cls_name} cannot be applied to ({dims}) "
                                    f"with these parameters"
                                )
                            result["errors"].append({"node_id": node_id, "message": message})
                            result["trace"] = f"Functional op declined at node {node_id}"
                            return result
                        t_out = torch.zeros(*shape, dtype=inputs[0].dtype)
                    elif op_key in _CUSTOM_PASSTHROUGH_OPS or node_id not in leaf_layers:
                        # Shape-preserving custom block (GRN/RoPE/adaLN/...): the
                        # real math can't run in this structural forward, so it
                        # passes its primary (first) input through unchanged.
                        #
                        # An IMPORTED block is different: it is not shape-preserving and
                        # ingest recorded what the real model produced (`_out_shape`).
                        # Passing through kept a depth-1 resnet18 at 64 channels into an
                        # fc expecting 512. Trust the measurement.
                        recorded = (n.data.parameters or {}).get("_out_shape")
                        dims = []
                        if isinstance(recorded, str):
                            try:
                                dims = [int(p) for p in recorded.split("x") if p]
                            except ValueError:
                                dims = []
                        t_out = torch.zeros(*dims) if dims else inputs[0]
                    else:
                        # Real torch.nn op: must receive exactly one input.
                        if len(inputs) > 1:
                            cls_name = _resolve_layer_class(n.data.layerType)
                            result["errors"].append({
                                "node_id": node_id,
                                "message": (
                                    f"arity error: {cls_name} received {len(inputs)} inputs "
                                    f"but expects exactly 1 (use a merge/concat op for multi-input)"
                                ),
                            })
                            result["trace"] = f"Arity error at node {node_id}"
                            return result
                        t_out = call_module(leaf_layers[node_id], inputs[0])
                    tensors[node_id] = t_out
                    last_tensor = t_out
                except Exception as fwd_err:
                    result["errors"].append({
                        "node_id": node_id,
                        "message": f"forward error: {fwd_err}",
                    })
                    result["trace"] = traceback.format_exc()
                    # Replace the crude pre-forward analytic estimates with the
                    # ground-truth shapes of the nodes that DID forward, so a
                    # failed run never paints a misleading [C,H,W] estimate onto
                    # the Shape Table for a node that actually produced a real
                    # tensor. Nodes that never ran simply carry no shape.
                    result["shapes"] = {nid: list(t.shape) for nid, t in tensors.items()}
                    return result
            elif n.type == "moduleNode" and any(e.target == node_id for e in flat.edges):
                # A block card that CARRIES dataflow, i.e. a depth-limited import whose
                # children are not in this graph. It cannot be executed, and the loop had
                # no branch for it at all, so the next node fell back to `last_tensor` —
                # the block's own INPUT. A depth-1 resnet18 therefore reached `fc` still
                # at 64 channels: "mat1 and mat2 shapes cannot be multiplied (1x64 and
                # 512x1000)". ingest recorded what the real model produced, so use it.
                # (At depth 2 the card carries no edges and is skipped, leaving the real
                # child-level dataflow untouched.)
                recorded = (n.data.parameters or {}).get("_out_shape")
                dims = []
                if isinstance(recorded, str):
                    try:
                        dims = [int(p) for p in recorded.split("x") if p]
                    except ValueError:
                        dims = []
                t_out = torch.zeros(*dims) if dims else inputs[0]
                tensors[node_id] = t_out
                last_tensor = t_out
            elif n.type == "outputNode":
                t_in = inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=-1)
                tensors[node_id] = t_in
                last_tensor = t_in

        result["output_shape"] = list(last_tensor.shape)
        # Ground-truth per-node shapes from the REAL forward pass override the
        # crude pre-forward analytic estimates (which mishandle the batch dim for
        # conv/pool). This is what the UI Shape Table reads, so it must be the
        # tensors that actually flowed — full [N, C, H, W], not [C, H, W].
        result["shapes"] = {nid: list(t.shape) for nid, t in tensors.items()}
    except Exception as exc:
        result["errors"].append({"message": f"Forward pass failed: {exc}"})
        result["trace"] = traceback.format_exc()
        return result

    # ── Summary ──────────────────────────────────────────────────────────
    total_params = sum(p.numel() for layer in leaf_layers.values() for p in layer.parameters())
    trainable = sum(p.numel() for layer in leaf_layers.values() for p in layer.parameters() if p.requires_grad)
    summary: Dict[str, Any] = {
        "layer_count": len(leaf_layers),
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable),
        "model_size_mb": round(total_params * 4 / (1024 * 1024), 2),
    }

    # Enrich with torchinfo (total MACs / memory) when available.
    try:
        import torchinfo  # type: ignore

        _ti_node_types = {n.id: n.type for n in flat.nodes}
        _ti_edge_pairs = [(e.source, e.target) for e in flat.edges]
        # A merge node must be combined with ITS OWN op, exactly as the dry forward
        # above does via _apply_merge. Concatenating instead turned every residual
        # `add` into a width-doubling cat, so activations grew 2^(residual depth):
        # a ResNet-50 candidate consumed every byte of RAM the process could get
        # (34 GB on CI, where it killed the runner) inside this optional metrics
        # pass — and the `except Exception` below swallowed the MemoryError, so the
        # only visible symptom was a missing total_mult_adds.
        _ti_merge_ops = {
            n.id: (_op_key(n.data.layerType), n.data.parameters or {})
            for n in flat.nodes
            if n.type == "layerNode" and _op_key(n.data.layerType) in _MERGE_OP_KEYS
        }

        class _ForwardModule(torch.nn.Module):
            """Wraps the graph as a proper nn.Module so torchinfo can instrument it."""
            def __init__(self):
                super().__init__()
                for nid, layer in leaf_layers.items():
                    safe = "l_" + "".join(c if c.isalnum() else "_" for c in nid)
                    setattr(self, safe, layer)
                self._fwd_args = (leaf_layers, order, _ti_edge_pairs, _ti_node_types, _ti_merge_ops)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                ll, ord_, ep, nt, mo = self._fwd_args
                tensors: Dict[str, torch.Tensor] = {}
                last = x
                for nid in ord_:
                    node_type = nt.get(nid, "layerNode")
                    if node_type == "inputNode":
                        tensors[nid] = x
                        last = x
                        continue
                    inputs = [tensors[s] for s, t in ep if t == nid and s in tensors]
                    if nid in mo and inputs:
                        op_key, params = mo[nid]
                        t_in = _apply_merge(op_key, inputs, params)
                    else:
                        t_in = inputs[0] if len(inputs) == 1 else (
                            torch.cat(inputs, dim=-1) if inputs else last
                        )
                    if node_type == "layerNode" and nid in ll:
                        t_out = ll[nid](t_in)
                        tensors[nid] = t_out
                        last = t_out
                    else:
                        tensors[nid] = t_in
                        last = t_in
                return last

        model_wrap = _ForwardModule()
        # Use float input for torchinfo (same dtype fix as ONNX export)
        ti_input = torch.randn(*[max(1, int(d)) for d in shape])
        ti = torchinfo.summary(
            model_wrap,
            input_data=ti_input,
            verbose=0,
            col_names=("output_size", "num_params", "mult_adds"),
        )
        summary["total_mult_adds"] = int(ti.total_mult_adds)
        summary["total_memory_mb"] = round(
            ti.to_megabytes(ti.total_output_bytes + ti.total_param_bytes), 2
        )
    except Exception:
        pass  # torchinfo is optional; unsupported model shape — skip silently

    result["summary"] = summary
    result["ok"] = True
    result["stage"] = "complete"
    return result


# ─────────────────────────────────────────────────────────────────────
# Package export — a self-contained zip with model.py, README, etc.
# ─────────────────────────────────────────────────────────────────────

def export_package(
    graph_payload: Dict[str, Any],
    input_shape: Any,
    model_name: str = "GeneratedModel",
) -> Tuple[bytes, str]:
    """Build an in-memory zip containing model.py, __init__.py, README.md,
    requirements.txt and a small sample run script. Returns (zip_bytes,
    filename). The package is independently runnable: `python sample_run.py`
    should print the output tensor shape.
    """
    shape = _coerce_input_shape(input_shape, [1, 3, 224, 224])
    is_hier = ("modules" in graph_payload and graph_payload["modules"]) or "root_nodes" in graph_payload
    codegen = CodeGenerationService()
    # Nodes/edges to derive the runnable sample inputs from (the root graph).
    sample_nodes, sample_edges = [], []
    if is_hier:
        hg = HierarchicalGraph(**graph_payload)
        code = codegen.generate_hierarchical_code(hg, shape, model_name)
        sample_nodes, sample_edges = hg.root_nodes, hg.root_edges
    else:
        # Factor repeated blocks into reusable ×N modules so the exported code
        # is modular (e.g. the flat 267-node F5-TTS -> one DiTBlock instantiated
        # 22× via nn.ModuleList). Falls back to flat codegen when nothing
        # groupable is found, so non-repetitive models export exactly as before.
        hg = None
        try:
            from services.graph_modularizer import modularize_for_codegen
            hg = modularize_for_codegen(graph_payload)
        except Exception:
            hg = None
        if hg is not None:
            code = codegen.generate_hierarchical_code(hg, shape, model_name)
            sample_nodes, sample_edges = hg.root_nodes, hg.root_edges
        else:
            flat = _flatten_for_torch(graph_payload)
            code = codegen.generate_flat_code(flat, shape, model_name)
            sample_nodes, sample_edges = flat.nodes, flat.edges

    # Build the sample run with one tensor per model input (multi-input aware).
    cons, args = codegen.sample_inputs(sample_nodes, sample_edges, shape)
    cons_block = "\n".join(cons)
    sample = (
        f"\"\"\"Sample run for {model_name}. Generated by tensorViz.\"\"\"\n"
        "import torch\n"
        f"from model import {model_name}\n\n"
        f"model = {model_name}()\n"
        "model.eval()  # inference mode: BatchNorm/Dropout use running stats\n"
        f"{cons_block}\n"
        f"y = model({', '.join(args)})\n"
        "print('output shape:', tuple(y.shape) if torch.is_tensor(y) else type(y))\n"
    )

    readme = (
        f"# {model_name}\n\n"
        f"Exported from tensorViz on {datetime.utcnow().isoformat()}Z.\n\n"
        "## Quickstart\n"
        "```bash\n"
        "pip install -r requirements.txt\n"
        "python sample_run.py\n"
        "```\n"
    )

    requirements = "torch>=2.1\n"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{model_name}/model.py", code)
        z.writestr(f"{model_name}/__init__.py", f"from .model import {model_name}\n")
        z.writestr(f"{model_name}/sample_run.py", sample)
        z.writestr(f"{model_name}/README.md", readme)
        z.writestr(f"{model_name}/requirements.txt", requirements)
    return buf.getvalue(), f"{model_name}.zip"
