"""Deterministic adapter-chain builder for rank/channel/feature bridging.

This module is the single source of truth for adapter insertion logic.
Both the template-validation sweep and deterministic operation resolver delegate
here so the adapter math is implemented once.

Public API
----------
plan_adapter_transition(...)
    The unified transition orchestrator: probe source shape → resolve the real
    (passthrough-walked) target → pick required rank → insert_adapter_chain.
    Both run_template_validation_sweep.py and OperationResolver call this, so
    the adapter path is exercised through one implementation.
insert_adapter_chain(...)
    Pure-functional: modifies `nodes` and `edges` lists in-place and
    returns (new_source_id, inserted_nodes, inserted_edges, issue_msg).

required_rank_for_op(op) -> int | None
target_feature_for_op(op, params, fallback) -> int
target_channel_for_rank4(params, fallback) -> int
target_spatial_extent_for_rank4(...) -> int
target_sequence_length_for_conv1d(params, fallback) -> int
rank_from_input_expectation(expectation) -> int | None
append_edge(edges, source, target, ...) -> None
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from models.architecture_models import ArchitectureEdge, ArchitectureNode


# ─── rank/feature helpers ──────────────────────────────────────────────────────

def required_rank_for_op(op: str) -> Optional[int]:
    """Return the required input rank for the given PyTorch op, or None if flexible."""
    if op in {
        "nn.Conv2d", "Conv2d",
        "nn.BatchNorm2d", "BatchNorm2d",
        "nn.MaxPool2d", "MaxPool2d",
        "nn.AvgPool2d", "AvgPool2d",
        "nn.AdaptiveAvgPool2d", "AdaptiveAvgPool2d",
        "nn.GroupNorm", "GroupNorm",
    }:
        return 4
    if op in {
        "nn.Conv1d", "Conv1d",
        "nn.BatchNorm1d", "BatchNorm1d",
    }:
        return 3
    if op in {
        "nn.LSTM", "LSTM",
        "nn.GRU", "GRU",
        "nn.RNN", "RNN",
        "nn.TransformerEncoderLayer", "TransformerEncoderLayer",
        "nn.TransformerDecoderLayer", "TransformerDecoderLayer",
    }:
        return 3
    if op in {
        "nn.Linear", "Linear",
        "nn.MultiheadAttention", "MultiheadAttention",
        "nn.LayerNorm", "LayerNorm",
    }:
        return 2
    return None


# Sequence models accept an *unbatched* rank-2 input in PyTorch (e.g. LSTM's
# (L, H_in)), but the canvas is always batched: a rank-2 [N, F] tensor here means
# [batch, features] and must gain a sequence dim before entering them. So these
# still get the rank-2→rank-3 expansion even though the registry lists rank 2.
_RANK2_IS_UNBATCHED_OPS = {
    "nn.LSTM", "LSTM", "nn.GRU", "GRU", "nn.RNN", "RNN",
    "nn.TransformerEncoderLayer", "TransformerEncoderLayer",
    "nn.TransformerDecoderLayer", "TransformerDecoderLayer",
}

# 1d normalizations accept a rank-3 [N, C, L] input for ANY sequence length L —
# only the channel dim must equal num_features. Unlike Conv1d (which needs L >=
# kernel_size), they must NOT be given the conv1d minimum-length expansion, or an
# already-valid feature map gets a spurious Linear that reshapes L and destroys it.
_RANK3_ANY_LENGTH_OPS = {
    "nn.BatchNorm1d", "BatchNorm1d", "nn.InstanceNorm1d", "InstanceNorm1d",
}


def _accepts_any_rank3_length(op: str) -> bool:
    return op in _RANK3_ANY_LENGTH_OPS


def _op_accepts_batched_rank(op: str, rank: int) -> bool:
    """True if the op natively accepts a *batched* tensor of the given rank.

    ``required_rank_for_op`` collapses range-ops (e.g. BatchNorm1d accepts rank
    2 or 3) to a single required rank, which would otherwise force a spurious
    rank change on an already-valid input. This consults the registry so we skip
    adapting when the current rank is genuinely acceptable — except for sequence
    models, whose rank-2 acceptance is the unbatched form (see above).
    """
    if op in _RANK2_IS_UNBATCHED_OPS and rank == 2:
        return False
    from core.op_registry import spec_for

    spec = spec_for(op)
    ranks = getattr(spec, "input_rank", None) if spec else None
    return bool(ranks) and rank in tuple(ranks)


def target_feature_for_op(op: str, params: Dict[str, object], fallback: int = 256) -> int:
    """Return the expected last-dimension feature size for `op`."""
    for key in (
        "in_features", "embed_dim", "d_model", "input_size",
        "num_features", "in_channels", "normalized_shape",
    ):
        value = params.get(key)
        if key == "normalized_shape":
            if isinstance(value, int) and value > 0:
                return value
            if isinstance(value, list) and value and isinstance(value[-1], int) and value[-1] > 0:
                return value[-1]
        if isinstance(value, int) and value > 0:
            return value
    return fallback


def target_channel_for_rank4(params: Dict[str, object], fallback: int) -> int:
    """Return the expected channel count for a rank-4 op."""
    for key in ("in_channels", "num_features", "num_channels", "channels"):
        value = params.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return fallback


def target_sequence_length_for_conv1d(params: Dict[str, object], fallback: int = 16) -> int:
    """Return a suitable sequence length for Conv1d-family targets."""
    kernel = params.get("kernel_size")
    if isinstance(kernel, int) and kernel > 0:
        return max(fallback, kernel)
    if isinstance(kernel, list) and kernel and isinstance(kernel[0], int) and kernel[0] > 0:
        return max(fallback, int(kernel[0]))
    return fallback


def target_spatial_extent_for_rank4(
    *,
    target_template_id: Optional[str],
    target_op: str,
    target_params: Dict[str, object],
    fallback: int = 16,
) -> int:
    """Return a suitable spatial extent (H=W) for rank-4 adapter tensors."""
    tid = (target_template_id or "").lower()
    if any(token in tid for token in ("aspp", "hrnet", "convnext", "diffusion")):
        return 224
    if any(token in tid for token in ("unet", "vgg", "densenet", "resnet", "fpn", "mask_rcnn")):
        return 32

    output_size = target_params.get("output_size")
    if isinstance(output_size, int) and output_size > 1:
        return int(output_size)
    if isinstance(output_size, list):
        dims = [int(v) for v in output_size if isinstance(v, int) and v > 1]
        if dims:
            return max(dims)

    if target_op in {
        "nn.MaxPool2d", "MaxPool2d",
        "nn.AvgPool2d", "AvgPool2d",
        "nn.AdaptiveAvgPool2d", "AdaptiveAvgPool2d",
    }:
        kernel = target_params.get("kernel_size")
        if isinstance(kernel, int) and kernel > 1:
            return max(fallback, kernel * 2)
    return fallback


def rank_from_input_expectation(expectation: str) -> Optional[int]:
    """Parse a rank integer from a contract input_expectation string."""
    text = (expectation or "").lower()
    marker = "rank-"
    idx = text.find(marker)
    if idx == -1:
        return None
    idx += len(marker)
    digits: List[str] = []
    while idx < len(text) and text[idx].isdigit():
        digits.append(text[idx])
        idx += 1
    return int("".join(digits)) if digits else None


# ─── edge helper ───────────────────────────────────────────────────────────────

def append_edge(
    edges: List[ArchitectureEdge],
    source: str,
    target: str,
    source_port: str = "y",
    target_port: str = "x",
) -> None:
    """Append an ArchitectureEdge to `edges` (no-dedup here; caller manages)."""
    edges.append(
        ArchitectureEdge(
            edge_id=f"{source}:{source_port}->{target}:{target_port}",
            source_node_id=source,
            source_port=source_port,
            target_node_id=target,
            target_port=target_port,
        )
    )


# ─── shape probe ───────────────────────────────────────────────────────────────

def infer_node_shape(
    *,
    validator,  # CanonicalValidationService (no import to avoid circular)
    graph_id: str,
    input_shape: List[int],
    nodes: List[ArchitectureNode],
    edges: List[ArchitectureEdge],
    node_id: str,
) -> Optional[List[int]]:
    """Run validation on a partial graph to determine the output shape at node_id."""
    from models.architecture_models import ArchitectureGraph  # local import

    probe_graph = ArchitectureGraph(
        graph_id=f"{graph_id}__probe__{node_id}__n{len(nodes)}__e{len(edges)}",
        name="shape_probe",
        revision=0,
        inputs=[{"name": "x", "shape": input_shape}],
        nodes=nodes,
        edges=edges,
        metadata={},
    )
    validation = validator.validate(graph=probe_graph, strict=False)
    result = validation.node_results.get(node_id)
    if not result or not result.output_shape:
        return None
    return list(result.output_shape)


# ─── main adapter builder ─────────────────────────────────────────────────────

def insert_adapter_chain(
    *,
    source_shape: List[int],
    nodes: List[ArchitectureNode],
    edges: List[ArchitectureEdge],
    source_node_id: str,
    target_op: str,
    target_params: Dict[str, object],
    adapter_prefix: str,
    target_required_rank: Optional[int] = None,
    target_template_id: Optional[str] = None,
    # Optional: for feature-align probing when shape tracking is imprecise
    validator=None,
    graph_id: str = "g",
    input_shape: Optional[List[int]] = None,
) -> Tuple[str, List[ArchitectureNode], List[ArchitectureEdge], Optional[str]]:
    """Insert bridging adapter nodes/edges between source and target.

    Modifies `nodes` and `edges` in-place.

    Returns
    -------
    (new_source_id, inserted_nodes, inserted_edges, issue_msg)
        new_source_id   — the adapter's exit node id to connect into target
        inserted_nodes  — nodes added by this call (for operation_history)
        inserted_edges  — edges added by this call (for operation_history)
        issue_msg       — None on success, reason string on unsupported transition
    """
    nodes_before = len(nodes)
    edges_before = len(edges)

    required_rank = target_required_rank or required_rank_for_op(target_op)
    if required_rank is None:
        return source_node_id, [], [], None

    current_source = source_node_id
    current_shape = list(source_shape)

    # ── rank-2 target: flatten higher-rank tensors ─────────────────────────────
    if required_rank == 2 and len(current_shape) > 2:
        flat_id = f"{adapter_prefix}_flat2"
        nodes.append(ArchitectureNode(
            node_id=flat_id, kind="layer", op="nn.Flatten",
            params={"start_dim": 1}, in_ports=["x"], out_ports=["y"],
        ))
        append_edge(edges, current_source, flat_id)
        current_source = flat_id
        current_shape = [current_shape[0], int(math.prod(current_shape[1:]))]

    # ── rank-4 target: build image-like tensor when not already rank-4 ─────────
    if required_rank == 4 and len(current_shape) != 4:
        if len(current_shape) > 2:
            flat_id = f"{adapter_prefix}_flat4"
            nodes.append(ArchitectureNode(
                node_id=flat_id, kind="layer", op="nn.Flatten",
                params={"start_dim": 1}, in_ports=["x"], out_ports=["y"],
            ))
            append_edge(edges, current_source, flat_id)
            current_source = flat_id
            current_shape = [current_shape[0], int(math.prod(current_shape[1:]))]

        if len(current_shape) == 2:
            target_channels = int(target_feature_for_op(target_op, target_params, fallback=current_shape[-1]))
            spatial_extent = target_spatial_extent_for_rank4(
                target_template_id=target_template_id,
                target_op=target_op,
                target_params=target_params,
                fallback=16,
            )
            projected_features = int(target_channels * spatial_extent * spatial_extent)
            linear_id = f"{adapter_prefix}_to4_linear"
            nodes.append(ArchitectureNode(
                node_id=linear_id, kind="layer", op="nn.Linear",
                params={"in_features": int(current_shape[-1]), "out_features": projected_features},
                in_ports=["x"], out_ports=["y"],
            ))
            append_edge(edges, current_source, linear_id)
            current_source = linear_id

            unflatten_id = f"{adapter_prefix}_to4_unflatten"
            nodes.append(ArchitectureNode(
                node_id=unflatten_id, kind="layer", op="nn.Unflatten",
                params={"dim": 1, "unflattened_size": [target_channels, spatial_extent, spatial_extent]},
                in_ports=["x"], out_ports=["y"],
            ))
            append_edge(edges, current_source, unflatten_id)
            current_source = unflatten_id
            current_shape = [current_shape[0], target_channels, spatial_extent, spatial_extent]

    # ── rank-4 to rank-4: channel alignment via 1×1 conv ──────────────────────
    if required_rank == 4 and len(current_shape) == 4:
        declared_in_channels = target_channel_for_rank4(target_params, fallback=int(current_shape[1]))
        if declared_in_channels > 0 and current_shape[1] != declared_in_channels:
            conv1x1_id = f"{adapter_prefix}_to4_chan_align"
            nodes.append(ArchitectureNode(
                node_id=conv1x1_id, kind="layer", op="nn.Conv2d",
                params={"in_channels": int(current_shape[1]), "out_channels": int(declared_in_channels), "kernel_size": 1},
                in_ports=["x"], out_ports=["y"],
            ))
            append_edge(edges, current_source, conv1x1_id)
            current_source = conv1x1_id
            current_shape = [current_shape[0], int(declared_in_channels), current_shape[2], current_shape[3]]

    is_conv1d_family = target_op in {"nn.Conv1d", "Conv1d", "nn.BatchNorm1d", "BatchNorm1d"}

    # ── rank-3 target: align when already rank-3 ──────────────────────────────
    if required_rank == 3 and len(current_shape) == 3:
        target_feature = int(target_feature_for_op(target_op, target_params, fallback=current_shape[-1]))
        if is_conv1d_family:
            target_channels = target_feature
            current_channels = int(current_shape[1]) if isinstance(current_shape[1], int) else target_channels
            current_length = int(current_shape[2]) if isinstance(current_shape[2], int) else 16
            if _accepts_any_rank3_length(target_op):
                # BatchNorm1d/InstanceNorm1d accept any L — keep the sequence length
                # so only a genuine channel mismatch triggers an adapter (never a
                # spurious length expansion on an already-valid feature map).
                target_length = current_length
            else:
                target_length = target_sequence_length_for_conv1d(
                    target_params, fallback=max(16, int(current_shape[-1])),
                )
            if current_channels != target_channels or current_length < target_length:
                flat_id = f"{adapter_prefix}_to3_conv1d_flat"
                nodes.append(ArchitectureNode(
                    node_id=flat_id, kind="layer", op="nn.Flatten",
                    params={"start_dim": 1}, in_ports=["x"], out_ports=["y"],
                ))
                append_edge(edges, current_source, flat_id)
                current_source = flat_id

                projected_features = int(target_channels * target_length)
                linear_id = f"{adapter_prefix}_to3_conv1d_linear"
                nodes.append(ArchitectureNode(
                    node_id=linear_id, kind="layer", op="nn.Linear",
                    params={"in_features": int(current_channels * current_length), "out_features": projected_features},
                    in_ports=["x"], out_ports=["y"],
                ))
                append_edge(edges, current_source, linear_id)
                current_source = linear_id

                unflatten_id = f"{adapter_prefix}_to3_conv1d_unflatten"
                nodes.append(ArchitectureNode(
                    node_id=unflatten_id, kind="layer", op="nn.Unflatten",
                    params={"dim": 1, "unflattened_size": [target_channels, target_length]},
                    in_ports=["x"], out_ports=["y"],
                ))
                append_edge(edges, current_source, unflatten_id)
                current_source = unflatten_id
                current_shape = [current_shape[0], target_channels, target_length]
        else:
            current_feature = int(current_shape[-1]) if isinstance(current_shape[-1], int) else target_feature
            if current_feature != target_feature:
                align_id = f"{adapter_prefix}_to3_feature_align"
                nodes.append(ArchitectureNode(
                    node_id=align_id, kind="layer", op="nn.Linear",
                    params={"in_features": current_feature, "out_features": target_feature},
                    in_ports=["x"], out_ports=["y"],
                ))
                append_edge(edges, current_source, align_id)
                current_source = align_id
                current_shape[-1] = target_feature

    # ── rank-3 target: expand from rank-2 ──────────────────────────────────────
    # ...unless the target natively accepts rank 2 (e.g. BatchNorm1d), in which
    # case the rank-2 tensor is already valid and needs no adapter.
    if required_rank == 3 and len(current_shape) == 2 and not _op_accepts_batched_rank(target_op, 2):
        target_feature = int(target_feature_for_op(target_op, target_params, fallback=current_shape[-1]))
        target_length = target_sequence_length_for_conv1d(target_params, fallback=16) if is_conv1d_family else 1
        in_features = int(current_shape[-1])
        projected_features = int(target_feature * target_length) if is_conv1d_family else int(target_feature)
        linear_id = f"{adapter_prefix}_to3_linear"
        nodes.append(ArchitectureNode(
            node_id=linear_id, kind="layer", op="nn.Linear",
            params={"in_features": in_features, "out_features": projected_features},
            in_ports=["x"], out_ports=["y"],
        ))
        append_edge(edges, current_source, linear_id)
        current_source = linear_id

        unflatten_id = f"{adapter_prefix}_to3_unflatten"
        if is_conv1d_family:
            unflatten_size: List[int] = [target_feature, target_length]
            current_shape = [current_shape[0], target_feature, target_length]
        else:
            unflatten_size = [1, target_feature]
            current_shape = [current_shape[0], 1, target_feature]
        nodes.append(ArchitectureNode(
            node_id=unflatten_id, kind="layer", op="nn.Unflatten",
            params={"dim": 1, "unflattened_size": unflatten_size},
            in_ports=["x"], out_ports=["y"],
        ))
        append_edge(edges, current_source, unflatten_id)
        current_source = unflatten_id

    # ── rank-3 target: collapse from rank>3 ────────────────────────────────────
    if required_rank == 3 and len(current_shape) > 3:
        flat_id = f"{adapter_prefix}_flat3"
        nodes.append(ArchitectureNode(
            node_id=flat_id, kind="layer", op="nn.Flatten",
            params={"start_dim": 1}, in_ports=["x"], out_ports=["y"],
        ))
        append_edge(edges, current_source, flat_id)
        current_source = flat_id
        flattened_feature = int(math.prod(current_shape[1:]))

        target_feature = int(target_feature_for_op(target_op, target_params, fallback=flattened_feature))
        target_length = target_sequence_length_for_conv1d(target_params, fallback=16) if is_conv1d_family else 1
        projected_features = int(target_feature * target_length) if is_conv1d_family else int(target_feature)
        linear_id = f"{adapter_prefix}_to3_from4_linear"
        nodes.append(ArchitectureNode(
            node_id=linear_id, kind="layer", op="nn.Linear",
            params={"in_features": flattened_feature, "out_features": projected_features},
            in_ports=["x"], out_ports=["y"],
        ))
        append_edge(edges, current_source, linear_id)
        current_source = linear_id

        unflatten_id = f"{adapter_prefix}_to3_from4_unflatten"
        if is_conv1d_family:
            unflatten_size_4_to_3: List[int] = [target_feature, target_length]
            current_shape = [current_shape[0], target_feature, target_length]
        else:
            unflatten_size_4_to_3 = [1, target_feature]
            current_shape = [current_shape[0], 1, target_feature]
        nodes.append(ArchitectureNode(
            node_id=unflatten_id, kind="layer", op="nn.Unflatten",
            params={"dim": 1, "unflattened_size": unflatten_size_4_to_3},
            in_ports=["x"], out_ports=["y"],
        ))
        append_edge(edges, current_source, unflatten_id)
        current_source = unflatten_id

    # ── Linear / MHA / LayerNorm: align feature dim ───────────────────────────
    if target_op in {
        "nn.Linear", "Linear",
        "nn.MultiheadAttention", "MultiheadAttention",
        "nn.LayerNorm", "LayerNorm",
    }:
        current_feature: Optional[int] = (
            int(current_shape[-1])
            if len(current_shape) >= 2 and isinstance(current_shape[-1], int)
            else None
        )
        if current_feature is None and validator is not None and input_shape is not None:
            probed = infer_node_shape(
                validator=validator,
                graph_id=graph_id,
                input_shape=input_shape,
                nodes=nodes,
                edges=edges,
                node_id=current_source,
            )
            if probed and len(probed) >= 2:
                current_feature = probed[-1]

        if isinstance(current_feature, int) and current_feature > 0:
            tgt_feature = target_feature_for_op(target_op, target_params, fallback=current_feature)
            if current_feature != tgt_feature:
                align_id = f"{adapter_prefix}_align"
                nodes.append(ArchitectureNode(
                    node_id=align_id, kind="layer", op="nn.Linear",
                    params={"in_features": current_feature, "out_features": tgt_feature},
                    in_ports=["x"], out_ports=["y"],
                ))
                append_edge(edges, current_source, align_id)
                current_source = align_id
                if len(current_shape) >= 2:
                    current_shape[-1] = int(tgt_feature)

    inserted_nodes = nodes[nodes_before:]
    inserted_edges = edges[edges_before:]
    return current_source, inserted_nodes, inserted_edges, None


# ─── transition-target resolution (shared: sweep + product via plan_adapter_transition) ─

def resolve_transition_target_node(
    *,
    template_nodes: List[ArchitectureNode],
    template_edges: List[ArchitectureEdge],
    entry_node_id: str,
) -> Optional[ArchitectureNode]:
    """Walk from entry through passthrough ops to the first feature-sensitive node."""
    node_map: Dict[str, ArchitectureNode] = {n.node_id: n for n in template_nodes}
    current = node_map.get(entry_node_id)
    if current is None:
        return None

    passthrough_ops = {
        "Input", "input",
        "nn.Identity", "Identity",
        "nn.Dropout", "Dropout",
        "nn.Dropout2d", "Dropout2d",
        "nn.Dropout3d", "Dropout3d",
    }
    feature_sensitive_ops = {
        "nn.Linear", "Linear",
        "nn.Conv1d", "Conv1d",
        "nn.Conv2d", "Conv2d",
        "nn.Conv3d", "Conv3d",
        "nn.MultiheadAttention", "MultiheadAttention",
        "nn.LayerNorm", "LayerNorm",
        "nn.BatchNorm1d", "BatchNorm1d",
        "nn.BatchNorm2d", "BatchNorm2d",
        "nn.BatchNorm3d", "BatchNorm3d",
        "nn.GroupNorm", "GroupNorm",
    }
    feature_param_keys = ("in_features", "in_channels", "embed_dim", "input_size", "normalized_shape", "num_features")

    def _is_feature_sensitive(node: ArchitectureNode) -> bool:
        if node.op in feature_sensitive_ops:
            return True
        return any(key in node.params for key in feature_param_keys)

    adjacency: Dict[str, List[str]] = {}
    for edge in template_edges:
        adjacency.setdefault(edge.source_node_id, []).append(edge.target_node_id)

    visited: set = set()
    while current.node_id not in visited and (current.kind == "input" or current.op in passthrough_ops):
        visited.add(current.node_id)
        downstream_ids = sorted([
            e.target_node_id
            for e in template_edges
            if e.source_node_id == current.node_id and e.target_node_id in node_map
        ])
        if not downstream_ids:
            break
        next_nodes = [node_map[nid] for nid in downstream_ids if node_map[nid].kind != "output"]
        if len(next_nodes) == 1:
            current = next_nodes[0]
            continue
        prioritized = sorted([n for n in next_nodes if _is_feature_sensitive(n)], key=lambda n: n.node_id)
        if prioritized:
            current = prioritized[0]
            continue
        break

    if _is_feature_sensitive(current):
        return current

    # BFS fallback to nearest downstream feature-sensitive node
    queue = [current.node_id]
    visited_bfs: set = {current.node_id}
    while queue:
        node_id = queue.pop(0)
        for nxt in adjacency.get(node_id, []):
            if nxt in visited_bfs:
                continue
            visited_bfs.add(nxt)
            node = node_map[nxt]
            if node.kind == "output":
                continue
            if _is_feature_sensitive(node):
                return node
            queue.append(nxt)

    return current


# Pure shape-preserving placeholders the PRODUCT may safely walk past: the node
# the user/agent connected to is respected for everything else (Dropout, pooling,
# activations, real layers). Input/Identity change nothing, so routing an adapter
# through them to the layer that actually constrains shape is behaviour-preserving.
_PLACEHOLDER_OPS = frozenset({"Input", "input", "nn.Identity", "Identity"})


def resolve_placeholder_target(
    *,
    nodes: List[ArchitectureNode],
    edges: List[ArchitectureEdge],
    entry_node_id: str,
) -> Optional[ArchitectureNode]:
    """Conservative product-runtime walk: step past PURE placeholders (Input /
    Identity) along an unambiguous single successor, stopping at the first real
    node. Unlike ``resolve_transition_target_node`` (used by the sweep) this does
    NOT walk past Dropout/etc. and has NO BFS-past-everything fallback — it never
    overrides a meaningful node the user chose. Returns the resolved node (or the
    entry node itself when it is already a real node / branches / dead-ends)."""
    node_map: Dict[str, ArchitectureNode] = {n.node_id: n for n in nodes}
    current = node_map.get(entry_node_id)
    if current is None:
        return None
    visited: set = set()
    while current.node_id not in visited and (current.kind == "input" or current.op in _PLACEHOLDER_OPS):
        visited.add(current.node_id)
        downstream = sorted(
            e.target_node_id for e in edges
            if e.source_node_id == current.node_id and e.target_node_id in node_map
        )
        nxt = [node_map[nid] for nid in downstream if node_map[nid].kind != "output"]
        if len(nxt) != 1:  # only follow a single unambiguous successor
            break
        current = nxt[0]
    return current


# ─── unified transition planner (product + sweep share this) ───────────────────

def plan_adapter_transition(
    *,
    validator,  # CanonicalValidationService (no import to avoid circular)
    graph_id: str,
    input_shape: List[int],
    nodes: List[ArchitectureNode],
    edges: List[ArchitectureEdge],
    source_node_id: str,
    target_node: ArchitectureNode,
    adapter_prefix: str,
    contract_rank: Optional[int] = None,
    target_template_id: Optional[str] = None,
    template_nodes: Optional[List[ArchitectureNode]] = None,
    template_edges: Optional[List[ArchitectureEdge]] = None,
    entry_node_id: Optional[str] = None,
    source_shape: Optional[List[int]] = None,
) -> Tuple[str, List[ArchitectureNode], List[ArchitectureEdge], Optional[str]]:
    """Probe the source shape, resolve the real transition target (walking past
    passthrough ops when template internals are supplied), pick the required rank,
    and insert the bridging adapter chain — the single orchestration both the
    validation sweep and the product runtime call, so the adapter path is shared.

    The contract rank (if any) is computed by the caller and passed in as
    ``contract_rank`` to keep this module free of the template-registry dependency.
    ``source_shape`` may be supplied by callers that need a specialised probe
    (e.g. the product runtime probes a module-aware graph); when omitted the flat
    ``infer_node_shape`` probe is used (correct for the sweep's flat graphs).
    Returns ``insert_adapter_chain``'s 4-tuple; on an unprobeable source the
    source node id is returned unchanged with no adapters and no issue.
    """
    if source_shape is None:
        source_shape = infer_node_shape(
            validator=validator,
            graph_id=graph_id,
            input_shape=input_shape,
            nodes=nodes,
            edges=edges,
            node_id=source_node_id,
        )
    if not source_shape:
        return source_node_id, [], [], None

    resolved = target_node
    if template_nodes is not None and template_edges is not None and entry_node_id is not None:
        resolved = resolve_transition_target_node(
            template_nodes=template_nodes,
            template_edges=template_edges,
            entry_node_id=entry_node_id,
        ) or target_node

    required_rank = contract_rank
    if required_rank is None:
        required_rank = required_rank_for_op(resolved.op)

    return insert_adapter_chain(
        source_shape=source_shape,
        nodes=nodes,
        edges=edges,
        source_node_id=source_node_id,
        target_op=resolved.op,
        target_params=resolved.params,
        adapter_prefix=adapter_prefix,
        target_required_rank=required_rank,
        target_template_id=target_template_id,
        validator=validator,
        graph_id=graph_id,
        input_shape=input_shape,
    )
