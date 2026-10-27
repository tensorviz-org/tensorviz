"""Analytic shape + dtype + arity propagation engine.

This module is the single place where topological traversal and per-op
shape math happen.  It consumes ``core/op_registry.py`` for per-op rules
and is imported by both the control plane (canonical_validation) and the
data plane (verification_service) as a cheap pre-check.

Dependency rule: this module only imports from ``core/``.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

from core.op_registry import spec_for

if TYPE_CHECKING:
    from models.architecture_models import ArchitectureEdge, ArchitectureNode


@dataclass
class NodeShapeResult:
    """Shape-inference outcome for a single node."""

    output_shape: list[int] | None = None
    input_shapes: list[list[int]] = field(default_factory=list)
    dtype: Literal["float", "int"] = "float"

    # Structural flags — set at most one per node.
    cycle_member: bool = False           # true → this node is in a cycle
    arity_warning: str | None = None     # set when single-input op received >1 parent
    shape_error: str | None = None       # shape_fn returned None for a concrete reason
    needs_fallback: bool = False         # no spec in registry; caller may try runtime
    used_passthrough: bool = False       # shape forwarded unchanged (placeholder op)

    # Lazy param resolution — non-empty when a defining_param was inferred.
    infilled_params: dict = field(default_factory=dict)   # param_name → inferred value
    set_param_repairs: list = field(default_factory=list) # list of repair dicts

    # Param-schema violations — non-empty when a param fails its constraint.
    schema_violations: list = field(default_factory=list) # list of violation dicts


# ── Lazy param resolution ─────────────────────────────────────────────────────

def _lazy_fill_params(
    canonical_op: str,
    params: dict,
    input_shape: list[int],
    node_id: str,
    defining_params: "tuple[str, ...]",
) -> "tuple[dict, dict, list[dict]]":
    """Infer missing defining_params from input_shape.

    Returns
    -------
    (filled_params, infilled, repairs)
    filled_params : copy of params with missing values filled in
    infilled      : dict of {param_name: inferred_value} (newly filled only)
    repairs       : list of set_param repair dicts
    """
    if not defining_params or not input_shape:
        return params, {}, []

    filled = dict(params)
    infilled: dict = {}
    repairs: list = []

    def _infer(param: str) -> None:
        """Infer a single param from input_shape if currently missing."""
        if filled.get(param) is not None:
            return
        val = None
        if param == "in_features":
            val = input_shape[-1] if input_shape else None
        elif param in {"in_channels"}:
            val = input_shape[1] if len(input_shape) > 1 else None
        elif param == "num_features":
            val = input_shape[1] if len(input_shape) > 1 else None
        elif param == "num_channels":
            val = input_shape[1] if len(input_shape) > 1 else None
        elif param == "normalized_shape":
            val = [input_shape[-1]] if input_shape else None
        elif param == "input_size":
            val = input_shape[-1] if input_shape else None
        elif param == "embed_dim":
            val = input_shape[-1] if input_shape else None
        elif param == "d_model":
            val = input_shape[-1] if input_shape else None
        if val is not None:
            filled[param] = val
            infilled[param] = val
            repairs.append({
                "op": "set_param",
                "node_id": node_id,
                "param": param,
                "value": val,
            })

    for dp in defining_params:
        _infer(dp)

    return filled, infilled, repairs


# ── Param-schema validation ───────────────────────────────────────────────────

def _validate_param_schema(
    canonical_op: str,
    param_schema: "dict[str, Any]",
    params: dict,
    node_id: str,
) -> "list[dict]":
    """Check params against OpSpec.param_schema; return list of violation dicts.

    Schema key formats
    ------------------
    ``"p"``                         → range check; value must be ``(lo, hi)`` tuple.
                                      *lo* / *hi* are floats; ``None`` means unbounded.
    ``"X|divides"``                 → divisibility: params[divisor] must divide params[X].
    ``"X|divides_both"``            → divisibility: divisor divides *both* params in the
                                      tuple value ``(A, B)``.

    Each returned violation dict has at minimum:
    ``{"error_type": str, "param": str, "message": str, "repair": dict | None}``
    """
    violations: list[dict] = []

    for key, constraint in param_schema.items():
        if "|divides" in key:
            # "X|divides_both": ("A", "B") or "X|divides": "Y"
            divisor_name = key.split("|")[0]
            divisor_val = params.get(divisor_name)
            if not isinstance(divisor_val, int) or divisor_val <= 0:
                continue  # can't check without a valid divisor

            targets: tuple
            if "divides_both" in key:
                if not isinstance(constraint, (list, tuple)) or len(constraint) != 2:
                    continue
                targets = tuple(constraint)
            else:
                targets = (constraint,)

            for target_name in targets:
                target_val = params.get(target_name)
                if not isinstance(target_val, int) or target_val <= 0:
                    continue
                if target_val % divisor_val != 0:
                    violations.append({
                        "error_type": "param_divisibility",
                        "param": divisor_name,
                        "message": (
                            f"'{divisor_name}={divisor_val}' must evenly divide "
                            f"'{target_name}={target_val}' for op '{canonical_op}'"
                        ),
                        "repair": {
                            "op": "set_param",
                            "node_id": node_id,
                            "param": divisor_name,
                            "value": 1,  # safest default; UI should offer better choice
                        },
                    })
        else:
            # Plain range constraint: key is param name, value is (lo, hi)
            param_name = key
            param_val = params.get(param_name)
            if param_val is None:
                continue
            if not isinstance(constraint, (list, tuple)) or len(constraint) != 2:
                continue
            lo, hi = constraint
            try:
                v = float(param_val)
            except (TypeError, ValueError):
                continue
            out_of_range = False
            if lo is not None and v < lo:
                out_of_range = True
            if hi is not None and v > hi:
                out_of_range = True
            if out_of_range:
                lo_str = str(lo) if lo is not None else "-∞"
                hi_str = str(hi) if hi is not None else "+∞"
                violations.append({
                    "error_type": "param_range",
                    "param": param_name,
                    "message": (
                        f"'{param_name}={param_val}' is out of valid range "
                        f"[{lo_str}, {hi_str}] for op '{canonical_op}'"
                    ),
                    "repair": {
                        "op": "set_param",
                        "node_id": node_id,
                        "param": param_name,
                        "value": lo if lo is not None else 0,
                    },
                })

    return violations


def _canonical(op: str) -> str:
    """Strip torch.nn./nn./torch. prefix and lowercase for uniform lookup."""
    s = str(op or "").strip()
    for pref in ("torch.nn.", "nn.", "torch."):
        if s.lower().startswith(pref):
            s = s[len(pref):]
            break
    return s


def infer(
    nodes: "list[ArchitectureNode]",
    edges: "list[ArchitectureEdge]",
    initial_shapes: "dict[str, list[int]]",
    passthrough_ops: "frozenset[str] | set[str] | None" = None,
    runtime_fallback_fn: "Callable[[Any, list[list[int]]], list[int] | None] | None" = None,
    fixed_shapes: "dict[str, list[int] | None] | None" = None,
) -> "dict[str, NodeShapeResult]":
    """Propagate shapes analytically through a graph.

    Parameters
    ----------
    nodes:
        All nodes in the graph (any kind).
    edges:
        All directed edges (source → target).
    initial_shapes:
        Pre-resolved output shapes for nodes whose shapes are already known
        (typically ``kind == "input"`` nodes).  ``node_id → shape``.
    passthrough_ops:
        Set of canonicalized (prefix-stripped, lower) op names that should
        forward their primary input shape unchanged when no registry spec is
        found.  If *None* an empty set is used.
    runtime_fallback_fn:
        Optional callable ``(node, parent_shapes) -> output_shape | None``
        invoked for ops not found in the registry (before the passthrough
        check).  When it returns a shape, that shape is propagated to
        downstream nodes exactly like a registry result.  This keeps the
        inline runtime fallback working through the full topological traversal.
    fixed_shapes:
        Authoritative observations for selected nodes, including explicit unknown
        outputs. These nodes never run analytic rules or runtime fallback. Unlike
        initial_shapes, a missing observation cannot be replaced by a derivation.

    Returns
    -------
    dict mapping ``node_id`` to ``NodeShapeResult``.
    Note: input nodes are *not* included in the result dict (their shapes
    come from *initial_shapes*).
    """
    passthrough = frozenset(passthrough_ops or ())

    node_map = {n.node_id: n for n in nodes}
    incoming: dict[str, list[str]] = defaultdict(list)
    adjacency: dict[str, list[str]] = defaultdict(list)
    indegree: dict[str, int] = {n.node_id: 0 for n in nodes}

    for edge in edges:
        incoming[edge.target_node_id].append(edge.source_node_id)
        adjacency[edge.source_node_id].append(edge.target_node_id)
        indegree[edge.target_node_id] = indegree.get(edge.target_node_id, 0) + 1

    # ── Kahn topological sort ─────────────────────────────────────────────────
    queue: deque[str] = deque(nid for nid, deg in indegree.items() if deg == 0)
    topo_order: list[str] = []
    remaining_indegree = dict(indegree)
    while queue:
        nid = queue.popleft()
        topo_order.append(nid)
        for nxt in adjacency.get(nid, []):
            remaining_indegree[nxt] -= 1
            if remaining_indegree[nxt] == 0:
                queue.append(nxt)

    # Nodes still with indegree > 0 are in a cycle.
    cycle_nodes: set[str] = {nid for nid, deg in remaining_indegree.items() if deg > 0}

    # ── Shape map (also tracks dtype) ─────────────────────────────────────────
    shape_map: dict[str, list[int]] = dict(initial_shapes)
    dtype_map: dict[str, Literal["float", "int"]] = {
        nid: "float" for nid in initial_shapes
    }

    results: dict[str, NodeShapeResult] = {}

    # Mark cycle members and stop processing them.
    for nid in cycle_nodes:
        results[nid] = NodeShapeResult(cycle_member=True)

    for node_id in topo_order:
        node = node_map[node_id]

        if fixed_shapes is not None and node_id in fixed_shapes:
            shape = fixed_shapes[node_id]
            shape_map.pop(node_id, None)
            if shape is not None:
                shape_map[node_id] = shape
            results[node_id] = NodeShapeResult(output_shape=shape)
            continue

        # Skip input nodes — their shapes are in initial_shapes.
        if node.kind == "input":
            continue

        # Output nodes just forward their primary parent's shape.
        if node.kind == "output":
            parent_ids = incoming.get(node_id, [])
            if parent_ids:
                primary_shape = shape_map.get(parent_ids[0])
                primary_dtype = dtype_map.get(parent_ids[0], "float")
                if primary_shape is not None:
                    shape_map[node_id] = primary_shape
                    dtype_map[node_id] = primary_dtype
                    results[node_id] = NodeShapeResult(
                        output_shape=primary_shape,
                        input_shapes=[s for pid in parent_ids if (s := shape_map.get(pid)) is not None],
                        dtype=primary_dtype,
                    )
                    continue
            results[node_id] = NodeShapeResult()
            continue

        parent_ids = incoming.get(node_id, [])
        if not parent_ids:
            # No parents — shape engine can't infer; let validation handle.
            results[node_id] = NodeShapeResult()
            continue

        parent_shapes = [shape_map.get(pid) for pid in parent_ids]
        parent_dtypes: list[Literal["float", "int"]] = [
            dtype_map.get(pid, "float") for pid in parent_ids
        ]

        # Skip if any parent shape is unknown (upstream error already recorded).
        if any(s is None for s in parent_shapes):
            results[node_id] = NodeShapeResult()
            continue

        all_shapes: list[list[int]] = parent_shapes  # type: ignore[assignment]
        primary_shape = all_shapes[0]
        primary_dtype = parent_dtypes[0]

        # ── Registry lookup ───────────────────────────────────────────────────
        spec = spec_for(node.op)

        res = NodeShapeResult(input_shapes=all_shapes)

        if spec is not None:
            # Arity check — emit a warning (not error) to avoid breaking
            # existing templates that wire multiple inputs to single-input ops.
            if spec.arity == "one" and len(all_shapes) > 1:
                res.arity_warning = (
                    f"Op '{spec.canonical}' expects a single input "
                    f"but received {len(all_shapes)} connections"
                )

            # dtype propagation — res.dtype is the OUTPUT dtype
            if spec.out_dtype == "int":
                res.dtype = "int"
            elif spec.out_dtype == "float":
                res.dtype = "float"
            else:  # "same" — follow the primary input dtype
                res.dtype = primary_dtype

            # Lazy param resolution: fill any missing defining_params from input_shape.
            filled_params, infilled, repairs = _lazy_fill_params(
                canonical_op=spec.canonical,
                params=node.params or {},
                input_shape=primary_shape,
                node_id=node_id,
                defining_params=spec.defining_params,
            )
            if infilled:
                res.infilled_params = infilled
                res.set_param_repairs = repairs

            # Param-schema validation (using filled params for completeness).
            if spec.param_schema:
                violations = _validate_param_schema(
                    canonical_op=spec.canonical,
                    param_schema=spec.param_schema,
                    params=filled_params,
                    node_id=node_id,
                )
                if violations:
                    res.schema_violations = violations

            # Analytic shape inference (using filled params so lazy resolution works)
            try:
                out = spec.shape_fn(filled_params, primary_shape, all_shapes)
            except Exception:
                out = None

            if out is not None:
                shape_map[node_id] = out
                dtype_map[node_id] = res.dtype
                res.output_shape = out
            else:
                res.shape_error = (
                    f"shape_fn for '{spec.canonical}' returned None "
                    f"(params={node.params}, input={primary_shape})"
                )
                # Still forward dtype so downstream can track.
                dtype_map[node_id] = res.dtype

        else:
            # No registry spec.
            # 1. Try the caller-supplied runtime fallback (e.g. build_layer + torch forward).
            if runtime_fallback_fn is not None:
                try:
                    fallback_out = runtime_fallback_fn(node, all_shapes)
                except Exception:
                    fallback_out = None
                if fallback_out is not None:
                    shape_map[node_id] = fallback_out
                    dtype_map[node_id] = primary_dtype
                    res.output_shape = fallback_out
                    res.dtype = primary_dtype
                    results[node_id] = res
                    continue

            # 2. Check passthrough set (placeholder / custom ops).
            canonical_lower = _canonical(node.op).lower()
            if canonical_lower in passthrough or node.op in passthrough:
                shape_map[node_id] = primary_shape
                dtype_map[node_id] = primary_dtype
                res.output_shape = primary_shape
                res.dtype = primary_dtype
                res.used_passthrough = True
            else:
                res.needs_fallback = True
                res.dtype = primary_dtype
                # Don't update shape_map — caller may still handle this node.

        results[node_id] = res

    return results
