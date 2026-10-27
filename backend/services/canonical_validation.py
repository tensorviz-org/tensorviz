"""Canonical validation pipeline for architecture graphs."""

import hashlib
import json
import math
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set

import torch
from core.layer_builder import build_layer, call_module
from core.op_registry import is_placeholder_op

#: The phrase every unreached result carries, so later passes can recognise one.
_UNREACHED_MARK = "was not reached"
#: Carried by every node downstream of an opaque block nobody has measured.
#:
#: Distinct from `_UNREACHED_MARK`, and deliberately not merged with it: "unreached"
#: means the run STOPPED and there is a real error earlier for the user to fix, so
#: `status="error"` is right there. "Unchecked" means no error exists anywhere — the
#: model may be perfectly correct, and nobody has looked. Same shape, opposite fact.
_UNCHECKED_MARK = "was not shape-checked"
from core.shape_inference import parse_input_shape
from models.architecture_models import (
    ArchitectureGraph,
    ArchitectureNode,
    NodeValidationResult,
    ValidationResult,
)
from pydantic import BaseModel, Field


# Lowercase torch.nn class name -> canonical class name (e.g. "embedding" ->
# "Embedding"). Built once from torch.nn so it covers every layer without a
# hand-maintained table.
_NN_CLASS_BY_LOWER = {
    name.lower(): name
    for name in dir(torch.nn)
    if isinstance(getattr(torch.nn, name, None), type)
    and issubclass(getattr(torch.nn, name), torch.nn.Module)
}


def _canonical_op_name(op: str) -> str:
    """Resolve an op string to its canonical torch.nn class name when possible.

    Frontend ids and the various graph converters emit ops inconsistently —
    'embedding', 'nn.Embedding', 'Embedding'. The shape-inference rules below
    match exact-cased class names, so an unnormalized 'embedding' misses every
    rule and falls through to a silent input-shape passthrough (the bug where
    an Embedding showed output == input). Strip the module prefix and recover
    the proper case from torch.nn; non-nn ops (torch.add, cat) are returned
    prefix-stripped and otherwise untouched, which the rules already accept.
    """
    s = str(op or "").strip()
    for pref in ("torch.nn.", "nn.", "torch."):
        if s.startswith(pref):
            s = s[len(pref):]
            break
    return _NN_CLASS_BY_LOWER.get(s.lower(), s)


class CanonicalValidationStage(BaseModel):
    """Evidence captured at one stable boundary of canonical validation."""

    valid: bool
    errors: list[str] = Field(default_factory=list)
    node_results: dict[str, NodeValidationResult] = Field(default_factory=dict)


class CanonicalValidationReport(BaseModel):
    """Stage-oriented canonical result used by the deterministic AI harness."""

    structural: CanonicalValidationStage
    shape: CanonicalValidationStage
    semantic: CanonicalValidationStage
    result: ValidationResult


def _semantic_content_digest(graph: ArchitectureGraph) -> str:
    """Hash all semantic content so speculative revisions cannot share cache rows."""

    payload = json.dumps(
        graph.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _copy_node_results(
    node_results: dict[str, NodeValidationResult],
) -> dict[str, NodeValidationResult]:
    return {
        node_id: result.model_copy(deep=True)
        for node_id, result in node_results.items()
    }


def _changed_error_results(
    before: dict[str, NodeValidationResult],
    after: dict[str, NodeValidationResult],
) -> dict[str, NodeValidationResult]:
    changed: dict[str, NodeValidationResult] = {}
    for node_id, result in after.items():
        if result.status != "error":
            continue
        previous = before.get(node_id)
        if previous is None or previous.model_dump(mode="json") != result.model_dump(mode="json"):
            changed[node_id] = result.model_copy(deep=True)
    return changed


def _node_error_messages(
    node_results: dict[str, NodeValidationResult],
) -> list[str]:
    return [
        f"{node_id}: {result.message or 'validation error'}"
        for node_id, result in node_results.items()
        if result.status == "error"
    ]


#: Ops that PRODUCE a tensor without consuming one. Asking them for an input
#: connection is asking the wrong question: `nn.Parameter` is a tensor that exists,
#: not a computation over something else, and it can never satisfy the rule.
#:
#: `self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))` used at
#: `self.embed(tokens) + self.pos[:, :n]` was reported as "Node has no input
#: connections" on a model that runs correctly — the one problem a freshly opened
#: demo showed. A parameter having no input is its definition, not its defect.
_SOURCE_OPS = {"parameter", "buffer", "embeddingbag_weight"}


def _bare(op: object) -> str:
    """`torch.nn.Parameter` -> `parameter`, so the set above is written once."""
    return str(op or "").rsplit(".", 1)[-1].lower()


def _container_ids(graph: ArchitectureGraph) -> set[str]:
    """Edge-free module frames declare grouping, not a forward boundary."""
    incident = {endpoint for edge in graph.edges for endpoint in (edge.source_node_id, edge.target_node_id)}
    return {node.node_id for node in graph.nodes if node.kind == "module" and node.node_id not in incident}


def _source_correspondence(node):
    from services.semantic.contracts import SourceCorrespondence
    raw = (node.params or {}).get("_capture_coverage")
    if not isinstance(raw, dict):
        return None
    try:
        fact = SourceCorrespondence.model_validate(raw)
        return fact if fact.nodeId == node.node_id else None
    except ValueError:
        return None


def _metadata_shape(value):
    if isinstance(value, list):
        return list(value) if all(type(dim) is int and dim >= 0 for dim in value) else None
    if not isinstance(value, str):
        return None
    try:
        dims = [int(part) for part in value.split("x") if part.strip()]
    except ValueError:
        return None
    return dims if dims and all(dim >= 0 for dim in dims) else None


class CanonicalValidationService:
    """Validation entrypoint for canonical graph sessions."""

    def __init__(self) -> None:
        self._cache: dict[
            tuple[str, int, bool, str], CanonicalValidationReport
        ] = {}
        self._cache_order: deque[tuple[str, int, bool, str]] = deque()
        self._cache_limit = 256

    def validate(self, graph: ArchitectureGraph, strict: bool = False) -> ValidationResult:
        """Compatibility facade returning the existing aggregate result."""

        return self.validate_stages(graph=graph, strict=strict).result

    def validate_stages(
        self,
        graph: ArchitectureGraph,
        strict: bool = False,
    ) -> CanonicalValidationReport:
        """Validate without mutation and retain truthful per-stage evidence."""

        if "capture" in graph.metadata:
            return self._validate_capture_projection(graph, strict)

        cache_key = (
            graph.graph_id,
            graph.revision,
            strict,
            _semantic_content_digest(graph),
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached.model_copy(deep=True)

        # Normalize op names to canonical torch class names on a private copy so
        # case/prefix variants ('embedding', 'nn.Embedding') match the analytic
        # shape rules instead of silently passing the input shape through. We
        # copy rather than mutate the caller's graph.
        graph = graph.model_copy(deep=True)
        for node in graph.nodes:
            node.op = _canonical_op_name(node.op)

        global_errors: list[str] = []
        node_results: dict[str, NodeValidationResult] = {}

        # Structural validation is split around shape inference because
        # connectivity requires the propagated shape map. Capture both halves
        # while retaining the long-standing aggregate ValidationResult.
        self._validate_graph_structure(graph=graph, global_errors=global_errors, node_results=node_results)
        structural_errors = list(global_errors)

        self._validate_input_shapes(graph=graph, node_results=node_results)
        node_ids = [node.node_id for node in graph.nodes]
        node_id_set = set(node_ids)
        shape_safe = (
            len(node_ids) == len(node_id_set)
            and all(
                edge.source_node_id in node_id_set
                and edge.target_node_id in node_id_set
                for edge in graph.edges
            )
        )
        if shape_safe:
            self._infer_shapes(graph=graph, node_results=node_results)

        after_shape = _copy_node_results(node_results)
        cycle_results = {
            node_id: result.model_copy(deep=True)
            for node_id, result in after_shape.items()
            if result.status == "error"
            and "part of a cycle" in (result.message or "")
        }
        shape_results = {
            node_id: result.model_copy(deep=True)
            for node_id, result in after_shape.items()
            if node_id not in cycle_results
        }
        shape_errors = _node_error_messages(shape_results)
        if not shape_safe:
            shape_errors.append(
                "Shape inference skipped because node references are structurally invalid"
            )

        before_connectivity_errors = len(global_errors)
        before_connectivity_results = _copy_node_results(node_results)
        self._validate_connectivity_and_resolution(graph=graph, node_results=node_results, global_errors=global_errors)

        structural_errors.extend(global_errors[before_connectivity_errors:])
        connectivity_results = _changed_error_results(
            before_connectivity_results,
            node_results,
        )
        structural_node_results = {
            **cycle_results,
            **connectivity_results,
        }
        structural_errors.extend(_node_error_messages(structural_node_results))

        before_semantic_errors = len(global_errors)
        before_semantic_results = _copy_node_results(node_results)
        self._validate_semantic_patterns(graph=graph, node_results=node_results, global_errors=global_errors)
        semantic_errors = list(global_errors[before_semantic_errors:])
        semantic_node_results = _changed_error_results(
            before_semantic_results,
            node_results,
        )

        if strict:
            before_strict_errors = len(global_errors)
            before_strict_results = _copy_node_results(node_results)
            self._apply_strict_mode_checks(graph=graph, node_results=node_results, global_errors=global_errors)
            strict_node_results = _changed_error_results(
                before_strict_results,
                node_results,
            )
            semantic_node_results.update(strict_node_results)
            # ``Strict validation failed`` is an aggregate marker. Attribute it
            # to this stage only when strict/semantic checks introduced an error,
            # rather than when a prior shape/structure error merely persisted.
            strict_global_errors = global_errors[before_strict_errors:]
            if semantic_node_results:
                semantic_errors.extend(strict_global_errors)

        semantic_errors.extend(_node_error_messages(semantic_node_results))
        result = ValidationResult(
            valid=(
                len(global_errors) == 0
                and all(v.status != "error" for v in node_results.values())
                and (not strict or all(v.status == "ok" for v in node_results.values()))
            ),
            graph_id=graph.graph_id,
            revision=graph.revision,
            global_errors=global_errors,
            node_results=node_results,
        )

        report = CanonicalValidationReport(
            structural=CanonicalValidationStage(
                valid=not structural_errors,
                errors=structural_errors,
                node_results=structural_node_results,
            ),
            shape=CanonicalValidationStage(
                valid=not shape_errors,
                errors=shape_errors,
                node_results=shape_results,
            ),
            semantic=CanonicalValidationStage(
                valid=not semantic_errors,
                errors=semantic_errors,
                node_results=semantic_node_results,
            ),
            result=result,
        )
        self._cache[cache_key] = report
        self._cache_order.append(cache_key)
        while len(self._cache_order) > self._cache_limit:
            evicted = self._cache_order.popleft()
            self._cache.pop(evicted, None)
        return report.model_copy(deep=True)

    def _validate_capture_projection(self, graph: ArchitectureGraph, strict: bool) -> CanonicalValidationReport:
        from services.semantic.document import validate_document_projection
        errors = []
        results = {}
        try:
            validate_document_projection(graph)
        except (KeyError, TypeError, ValueError):
            errors.append("Captured document references or argument ports are invalid.")
        if not errors:
            for node in graph.nodes:
                provenance = node.params.get("_shape_provenance")
                status = "error" if provenance == "failed" else "ok" if provenance == "observed" else "warning"
                results[node.node_id] = NodeValidationResult(status=status,
                    message="Recorded call failed for this input." if status == "error" else
                        "Recorded capture evidence; no new execution or shape inference." if status == "ok" else
                        "No completed boundary evidence was captured.",
                    input_shape=node.params.get("_in_shape"), output_shape=node.params.get("_out_shape"))
        measured_errors = _node_error_messages(results)
        result = ValidationResult(graph_id=graph.graph_id, revision=graph.revision, global_errors=errors,
            node_results=results, valid=not errors and not measured_errors and (not strict or all(r.status == "ok" for r in results.values())))
        return CanonicalValidationReport(structural=CanonicalValidationStage(valid=not errors, errors=errors),
            shape=CanonicalValidationStage(valid=not errors and not measured_errors, errors=measured_errors, node_results=results),
            semantic=CanonicalValidationStage(valid=not errors, errors=errors), result=result)

    def _validate_graph_structure(
        self,
        graph: ArchitectureGraph,
        global_errors: list[str],
        node_results: dict[str, NodeValidationResult],
    ) -> None:
        node_ids = [n.node_id for n in graph.nodes]
        duplicates = {node_id for node_id in node_ids if node_ids.count(node_id) > 1}
        if duplicates:
            global_errors.append(f"Duplicate node IDs found: {sorted(duplicates)}")

        edge_ids = [edge.edge_id for edge in graph.edges]
        duplicate_edge_ids = {
            edge_id for edge_id in edge_ids if edge_ids.count(edge_id) > 1
        }
        if duplicate_edge_ids:
            global_errors.append(
                f"Duplicate edge IDs found: {sorted(duplicate_edge_ids)}"
            )

        node_id_set = set(node_ids)
        node_by_id = {node.node_id: node for node in graph.nodes}
        has_input = any(n.kind == "input" for n in graph.nodes)
        has_output = any(n.kind == "output" for n in graph.nodes)
        if not has_input:
            global_errors.append("Graph must contain at least one input node")
        if not has_output:
            global_errors.append("Graph must contain at least one output node")

        for edge in graph.edges:
            if edge.source_node_id not in node_id_set:
                global_errors.append(f"Edge '{edge.edge_id}' source node not found: {edge.source_node_id}")
            elif edge.source_port not in node_by_id[edge.source_node_id].out_ports:
                global_errors.append(
                    f"Edge '{edge.edge_id}' source port not found: "
                    f"{edge.source_node_id}.{edge.source_port}"
                )
            if edge.target_node_id not in node_id_set:
                global_errors.append(f"Edge '{edge.edge_id}' target node not found: {edge.target_node_id}")
            elif edge.target_port not in node_by_id[edge.target_node_id].in_ports:
                global_errors.append(
                    f"Edge '{edge.edge_id}' target port not found: "
                    f"{edge.target_node_id}.{edge.target_port}"
                )

        for node in graph.nodes:
            if node.node_id not in node_results:
                node_results[node.node_id] = NodeValidationResult(status="ok")

    def _validate_input_shapes(
        self,
        graph: ArchitectureGraph,
        node_results: dict[str, NodeValidationResult],
    ) -> None:
        for node in graph.nodes:
            if node.kind != "input":
                continue
            raw = node.params.get("shape")
            # The canvas stores this as the string the user typed ("1,16"); only the
            # code-import path emits a list. Requiring a list here failed the Input
            # node of every real graph, and because the suggested repair below is an
            # image shape, the AI repair loop obediently rewrote MLP inputs to
            # [1, 3, 224, 224] and inserted flatten/align bridges to match.
            shape = parse_input_shape(raw)
            if shape is None:
                node_results[node.node_id] = NodeValidationResult(
                    status="error",
                    message=(
                        "Input node shape must be positive integers, as a list "
                        f"or comma-separated string; got {raw!r}"
                    ),
                    suggested_repairs=[
                        {"op": "set_param", "node_id": node.node_id, "param": "shape", "value": [1, 3, 224, 224]}
                    ],
                )
                continue
            node_results[node.node_id].output_shape = shape

    def _infer_shapes(
        self,
        graph: ArchitectureGraph,
        node_results: dict[str, NodeValidationResult],
    ) -> None:
        from core import shape_engine as _se

        custom_passthrough_ops = {
            "rmsnorm",
            "flash_attention",
            "block_sparse_attention",
            "global_attention",
            "random_attention",
            "sliding_window_attention",
            "scaled_dot_attention",
            "multi_query_attention",
            "speech_transformer_attention",
            "disentangled_attention",
            "spatial_attention",
            "multi_scale_deformable_attention",
            "rotary_embedding",
            "rope",
            "grn",
            "global_response_norm",
            "ada_layer_norm",
            "ada_layer_norm_zero",
            "ada_layer_norm_final",
            "sinusoidal_pos_embedding",
            "sinusoidal_position_embedding",
            "seq_projection",
            "temperature_scale",
            "neighbor_sampler",
            "message_mlp",
            "attention_aggregate",
            "lstm_aggregate",
            "max_aggregate",
            "mean_aggregate",
            "sum_aggregate",
            "epsilon_add",
            "topk_router",
            "mixture_merge",
            "gated_fusion",
            "ghost_module",
            "se_block",
            "masked_conv2d",
            "roi_align",
            "vector_quantizer",
            "gaussian_noise",
            "conformer_conv_module",
            "mamba_depthwise_conv1d",
            "SelectiveSSM",
            "channel_split",
            "channel_split_a",
            "channel_split_b",
            "channel_concat",
            "channel_shuffle",
            "reshape",
            "split_path_a",
            "split_path_b",
            "dot_product_similarity",
            "cosine_similarity",
            "identity",
            "feature_mask",
            "factorization_machine",
            "affine_coupling",
            "weighted_add",
            "glu_activation",
            "torch.sum",
            "torch.mul",
            "sum",
            "mul",
            "patch_reshape",
            "learned_latents",
            "kernel_feature_map",
            "linear_attention",
            "relation_split",
            "relation_conv",
            "relation_sum",
            "window_partition",
            "window_reverse",
            "Reparameterize",
        }

        # Build initial_shapes from already-resolved input nodes, and from every node
        # that already carries a shape something wrote down after running it.
        #
        # The engine recomputes a seeded non-input node anyway, so a measurement never
        # overrides a derivation that succeeds — it is there for the ones that do not.
        # Where the graph is missing a step the model really performs (a reshape inside a
        # nested module, an opaque statement), the derivation fails and every node after
        # it starves: one error on the node itself and "input shape could not be resolved"
        # on each of its successors. A recorded shape ends that chain at the first node
        # that has one, which is the whole point of having measured it.
        initial_shapes: dict[str, list[int]] = {}
        for node in graph.nodes:
            if node.kind == "input":
                r = node_results.get(node.node_id)
                if r and r.output_shape:
                    initial_shapes[node.node_id] = r.output_shape
                continue
            recorded = self._recorded_output_shape(node)
            if recorded is not None:
                initial_shapes[node.node_id] = recorded

        # Runtime fallback wrapper so shape_engine can update shape_map inline.
        def _runtime_fallback(node, parent_shapes):
            shape, _ = self._infer_runtime_fallback(node, parent_shapes)
            return shape

        engine_results = _se.infer(
            nodes=graph.nodes,
            edges=graph.edges,
            initial_shapes=initial_shapes,
            passthrough_ops=frozenset(custom_passthrough_ops),
            runtime_fallback_fn=_runtime_fallback,
            fixed_shapes={node.node_id: self._recorded_output_shape(node)
                          for node in graph.nodes if _source_correspondence(node) is not None},
        )

        node_map = {n.node_id: n for n in graph.nodes}

        # Nodes the run never reached, and everything downstream of them.
        #
        # The trace marks each node after a failing statement `unreached`. The shape
        # engine above knows nothing of that and derives a shape for it anyway — a
        # GELU after a Linear that raised was reported at (2, 20), confidently, from
        # out_features alone — so a card showed a number for a layer whose input
        # never existed while the header said the run had failed. A run that
        # stopped is a fact about the model; a shape derived past the stopping
        # point is a guess in the same font. The mark is propagated forward here
        # so the output node does not report a shape either.
        # A SOURCE carries the mark structurally, not because anything failed: the tracer
        # enters frames, and `nn.Parameter` has no frame to enter, so it is "unreached" on
        # every successful run forever. Reporting that as "the run failed before it" is
        # false on a model that ran fine — it was the single problem a freshly opened
        # demo showed. The mark itself has to stay: `_merge_edges` keys the parser's
        # wiring off it, and stripping it would take the parameter's only edge with it.
        unreached: Set[str] = {
            n.node_id
            for n in graph.nodes
            if isinstance(n.params, dict)
            and n.params.get("_shape_provenance") == "unreached"
            and _bare(n.op) not in _SOURCE_OPS
        }
        downstream: Dict[str, List[str]] = {}
        for edge in graph.edges:
            downstream.setdefault(edge.source_node_id, []).append(edge.target_node_id)

        # A node the trace MEASURED is evidence it ran, and both floods below defer to
        # it: neither "unreached" nor "starved by an opaque block" can be true of a node
        # whose shape was observed.
        measured = {
            n.node_id for n in graph.nodes if self._recorded_output_shape(n) is not None
        }

        if unreached:
            # A node the trace MEASURED is not unreached, whatever sits upstream of it.
            # The mark propagated forward with no exemption at all, so three ghost
            # statements in F5-TTS — loop bodies the projection wrongly marked, which
            # the trace had in fact executed — flooded 143 nodes that each carry an
            # observed `_out_shape`, and every one was reported "was not reached: the
            # run failed before it" on a model whose trace error is None.
            #
            # The `opaque` blame flood twenty lines below has always had this guard
            # (`self._recorded_output_shape(n) is None`); this one never did. A
            # measurement is evidence the node ran, and it outranks an inference about
            # its neighbours — which is the same rule the rest of this file follows.
            frontier = list(unreached)
            while frontier:
                for target in downstream.get(frontier.pop(), []):
                    if target not in unreached and target not in measured:
                        unreached.add(target)
                        frontier.append(target)

        # Everything downstream of an opaque block, and which block is to blame.
        #
        # A placeholder nobody has measured has no output shape — correctly, since an
        # opaque statement is not shape-preserving. But the shape chain then breaks
        # there, and the connectivity pass used to report EVERY node after it as
        # "input shape could not be resolved": one error each, on layers that are
        # perfectly correct and whose only fault is standing downstream of something
        # unknown. On a transformer that is fifty errors about one unknown block.
        #
        # So the blame is collected here and attributed to the block itself. Nodes
        # carrying a measured `_out_shape` are NOT opaque — the trace answered for
        # them, and they resolve normally.
        opaque = {
            n.node_id
            for n in graph.nodes
            if (is_placeholder_op(n.op or "") or n.op == "ModuleList")
            and self._recorded_output_shape(n) is None
        }
        blocked_by: Dict[str, str] = {}
        if opaque:
            frontier = [(cause, cause) for cause in sorted(opaque)]
            while frontier:
                source, cause = frontier.pop()
                for target in downstream.get(source, []):
                    # `unreached` outranks this: a run that stopped is a stronger fact
                    # than one that never ran. An opaque node is its own cause, never
                    # another's victim, and the nearest cause wins.
                    # Same rule as the unreached flood above: a node the trace
                    # MEASURED is checked, so it is not starved by anything upstream
                    # and does not belong in the blocked set. Without this, thirteen
                    # nodes carrying an observed `_out_shape` were still reported as
                    # "not shape-checked" because a ghost statement sat in front of
                    # them.
                    if (
                        target in blocked_by
                        or target in opaque
                        or target in unreached
                        or target in measured
                    ):
                        continue
                    blocked_by[target] = cause
                    frontier.append((target, cause))

        containers = _container_ids(graph)
        for node_id, res in engine_results.items():
            node = node_map[node_id]
            nvr = node_results.setdefault(node_id, NodeValidationResult(status="ok"))
            correspondence = _source_correspondence(node)
            if correspondence is not None:
                output = self._recorded_output_shape(node)
                input_ = _metadata_shape((node.params or {}).get("_in_shape"))
                state = correspondence.status
                messages = {
                    "not_observed": "No invocation of this module was retained for this run.",
                    "capture_limited": "Capture limits prevent establishing whether this source entity ran.",
                    "static_only": "This source declaration has no corresponding captured module instance.",
                    "unresolved": "The source expression has no unambiguous captured correspondence.",
                    "state": "No retained parameter or buffer value establishes a shape; state is not a module call.",
                    "observed": "Captured calls have no single common tensor output shape; inspect their output trees.",
                }
                node_results[node_id] = NodeValidationResult(
                    status="error" if state == "failed" else "ok" if state in ("observed", "state") and output is not None else "warning",
                    message="A captured module call failed for this input." if state == "failed" else (
                        None if state in ("observed", "state") and output is not None else f"{_UNCHECKED_MARK}: {messages[state]}"),
                    input_shape=input_, output_shape=output if state in ("observed", "state") else None,
                )
                continue
            # Workspace resolution can label a frame custom_code. It still has
            # no execution to shape-check, even in strict mode; the actual opaque
            # loop statement and every wired boundary retain their diagnostics.
            if node_id in containers:
                continue

            if node_id in unreached:
                node_results[node_id] = NodeValidationResult(
                    status="error",
                    message=(
                        f"Node '{node_id}' {_UNREACHED_MARK}: the run failed before it, "
                        "so it has no shape. Fix the error the trace reported on the "
                        "earlier line, then save to run again."
                    ),
                    input_shape=None,
                    output_shape=None,
                )
                continue

            if res.cycle_member:
                node_results[node_id] = NodeValidationResult(
                    status="error",
                    message=f"Node '{node_id}' is part of a cycle in the graph",
                )
                continue

            # Downstream of an opaque block. Not an error: nothing about this node is
            # known to be wrong, and nobody has checked it. It must be marked BEFORE
            # the early-out below, because that `continue` is exactly what used to
            # leave a starved node looking untouched for the connectivity pass to
            # then misread as a wiring failure.
            if node_id in blocked_by and res.output_shape is None:
                node_results[node_id] = NodeValidationResult(
                    status="warning",
                    message=(
                        f"Node '{node_id}' {_UNCHECKED_MARK}: '{blocked_by[node_id]}' before it "
                        "is opaque to static analysis, so nothing after it has a known input shape."
                    ),
                    input_shape=res.input_shapes[0] if res.input_shapes else None,
                    output_shape=None,
                    blocked_by=blocked_by[node_id],
                )
                continue

            if node.kind == "output":
                nvr.input_shape = res.input_shapes[0] if res.input_shapes else None
                if res.output_shape is not None:
                    nvr.output_shape = res.output_shape
                else:
                    node_results[node_id] = NodeValidationResult(
                        status="warning",
                        message="Output shape unavailable until upstream shape errors are fixed",
                    )
                continue

            if is_placeholder_op(node.op or "") or node.op == "ModuleList":
                resolved, fallback_error = self._infer_runtime_fallback(node, res.input_shapes)
                # A placeholder someone MEASURED is checked, not unchecked. The trace
                # (or a `# tensorviz: out_shape=` declaration) answered for it, so it
                # resolves like any other node and starves nothing after it. Reporting
                # it as unchecked anyway would tell a user who did trace their model
                # that it still had not been looked at.
                if resolved is not None:
                    nvr.input_shape = res.input_shapes[0] if res.input_shapes else None
                    nvr.output_shape = list(resolved)
                    continue
                blocked = [nid for nid, cause in blocked_by.items() if cause == node_id]
                message = fallback_error or f"No deterministic rule available for operation '{node.op}'"
                # Carries the mark so the connectivity pass spares it. An opaque node
                # downstream of ANOTHER opaque node has no input shape either, and
                # without this it fell through to be reported as a wiring failure —
                # the same false blame this whole change exists to remove, landing on
                # the block instead of on the layer after it.
                message = f"{_UNCHECKED_MARK}: {message}"
                if blocked:
                    # The one statement this whole collapse exists to produce. Saying it
                    # here, once, is the difference between "your model has fifty errors"
                    # and "one block could not be checked, so fifty nodes went unchecked".
                    message = message.rstrip().rstrip(".")
                    message += (
                        f". The {len(blocked)} node(s) after it were not shape-checked."
                    )
                node_results[node_id] = NodeValidationResult(
                    status="warning",
                    message=message,
                    input_shape=res.input_shapes[0] if res.input_shapes else None,
                    output_shape=None,
                )
                continue

            if not res.input_shapes:
                continue

            primary_shape = res.input_shapes[0]
            nvr.input_shape = primary_shape

            # Schema violations take priority over shape errors (they provide repairs).
            if res.schema_violations:
                violation_msgs = "; ".join(v["message"] for v in res.schema_violations)
                repairs = [v["repair"] for v in res.schema_violations if v.get("repair")]
                node_results[node_id] = NodeValidationResult(
                    status="error",
                    message=f"Param constraint violation: {violation_msgs}",
                    input_shape=primary_shape,
                    output_shape=res.output_shape,
                    suggested_repairs=repairs,
                )
                continue

            if res.output_shape is not None:
                nvr.output_shape = res.output_shape
                # Propagate set_param repairs from lazy param resolution.
                if res.set_param_repairs:
                    nvr.suggested_repairs = list(res.set_param_repairs)
                # Arity warnings are informational only in this adapter; strict
                # arity enforcement is deferred to Step 4 (param-range checks)
                # to avoid breaking graphs that legitimately have bridge adapters
                # producing multiple edges into a single-input op.
                continue

            # A placeholder op whose shape_fn declined: it has no recorded shape.
            # Reported as the "no deterministic rule" case rather than as a shape
            # error, because nothing about it is wrong — nobody has run it. Strict
            # mode fails on the warning, which is the point: a graph carrying
            # unexecuted source is not a checked model.
            if res.needs_fallback:
                _, fallback_error = self._infer_runtime_fallback(node, res.input_shapes)
                if node.op in custom_passthrough_ops:
                    nvr.output_shape = primary_shape
                    continue
                node_results[node_id] = NodeValidationResult(
                    status="warning",
                    message=fallback_error or f"No deterministic rule available for operation '{node.op}'",
                    input_shape=primary_shape,
                    output_shape=primary_shape,
                )
                continue

            # A measurement outranks an inference, here as much as in the two floods
            # above. The engine could not derive a shape for this node from the graph it
            # was given — but if something already RAN the node and wrote down what came
            # out, the engine's failure is a statement about the graph, not about the
            # model.
            #
            # It is routinely a statement about the graph. The projection draws a node per
            # `nn.Module` call and per statement it can name, and the wiring between them
            # is reconstructed, not measured: TinyViT's `PatchEmbed` does
            # `self.proj(x)`, then `x.flatten(2).transpose(1, 2)`, then `self.norm(x)`,
            # and the reshape in the middle is inside a nested module, so the merged graph
            # joins a Conv2d straight to a LayerNorm. The engine then reads the LayerNorm's
            # input as [1, 64, 8, 8], compares 8 against normalized_shape=64 and reports
            # "LayerNorm normalized_shape must match trailing input dimensions" — on a
            # layer torch had just run, and whose measured output the same node carries.
            # That was one of five red squiggles on a model that runs (2026-09-12).
            recorded = self._recorded_output_shape(node)
            if recorded is not None:
                nvr.output_shape = recorded
                continue

            node_results[node_id] = self._make_shape_error(
                node, primary_shape, res.input_shapes
            )

    def _make_shape_error(
        self,
        node: ArchitectureNode,
        input_shape: list[int],
        all_shapes: list[list[int]],
    ) -> NodeValidationResult:
        """Return a NodeValidationResult with per-op error message + repairs."""
        op = node.op
        p = node.params

        if op in {"nn.Conv2d", "Conv2d"}:
            # A 4-D input with the wrong channel count has no rank problem: say so,
            # the way Linear does, instead of pointing at [N, C, H, W] and sending
            # the reader after a flatten that is not there.
            expected_channels = p.get("in_channels")
            message = "Conv2d requires [N, C, H, W] input shape"
            if (
                len(input_shape) == 4
                and isinstance(expected_channels, int)
                and expected_channels != input_shape[1]
            ):
                message = (
                    f"Conv2d in_channels mismatch: expected {expected_channels}, "
                    f"got {input_shape[1]}"
                )
            return NodeValidationResult(
                status="error",
                message=message,
                input_shape=input_shape,
                suggested_repairs=[
                    {
                        "op": "set_param",
                        "node_id": node.node_id,
                        "param": "in_channels",
                        "value": input_shape[1] if len(input_shape) > 1 else 1,
                    }
                ],
            )
        if op in {"nn.Conv1d", "Conv1d"}:
            return NodeValidationResult(
                status="error",
                message="Conv1d requires [N, C, L] input shape with matching in_channels",
                input_shape=input_shape,
            )
        if op in {"nn.Conv3d", "Conv3d"}:
            return NodeValidationResult(
                status="error",
                message="Conv3d requires [N, C, D, H, W] input shape with matching in_channels",
                input_shape=input_shape,
            )
        if op in {"nn.ConvTranspose1d", "ConvTranspose1d"}:
            return NodeValidationResult(
                status="error",
                message="ConvTranspose1d requires [N, C, L] input shape with matching in_channels",
                input_shape=input_shape,
            )
        if op in {"nn.ConvTranspose2d", "ConvTranspose2d"}:
            return NodeValidationResult(
                status="error",
                message="ConvTranspose2d requires [N, C, H, W] input shape with matching in_channels",
                input_shape=input_shape,
            )
        if op in {"nn.ConvTranspose3d", "ConvTranspose3d"}:
            return NodeValidationResult(
                status="error",
                message="ConvTranspose3d requires [N, C, D, H, W] input shape with matching in_channels",
                input_shape=input_shape,
            )
        if op in {"nn.MaxPool2d", "MaxPool2d", "nn.AvgPool2d", "AvgPool2d"}:
            return NodeValidationResult(
                status="error",
                message="Pool2d requires [N, C, H, W] input shape",
                input_shape=input_shape,
            )
        if op in {"nn.MaxPool1d", "MaxPool1d", "nn.AvgPool1d", "AvgPool1d"}:
            return NodeValidationResult(
                status="error",
                message="Pool1d requires [N, C, L] input shape",
                input_shape=input_shape,
            )
        if op in {"nn.MaxPool3d", "MaxPool3d", "nn.AvgPool3d", "AvgPool3d"}:
            return NodeValidationResult(
                status="error",
                message="Pool3d requires [N, C, D, H, W] input shape",
                input_shape=input_shape,
            )
        if op in {"nn.AdaptiveAvgPool2d", "AdaptiveAvgPool2d"}:
            return NodeValidationResult(
                status="error",
                message="AdaptiveAvgPool2d requires [N, C, H, W] input shape",
                input_shape=input_shape,
            )
        if op in {
            "nn.AdaptiveAvgPool1d", "AdaptiveAvgPool1d",
            "nn.AdaptiveMaxPool1d", "AdaptiveMaxPool1d",
        }:
            return NodeValidationResult(
                status="error",
                message="AdaptivePool1d requires [N, C, L] input shape",
                input_shape=input_shape,
            )
        if op in {"nn.AdaptiveMaxPool2d", "AdaptiveMaxPool2d"}:
            return NodeValidationResult(
                status="error",
                message="AdaptiveMaxPool2d requires [N, C, H, W] input shape",
                input_shape=input_shape,
            )
        if op in {
            "nn.AdaptiveAvgPool3d", "AdaptiveAvgPool3d",
            "nn.AdaptiveMaxPool3d", "AdaptiveMaxPool3d",
        }:
            return NodeValidationResult(
                status="error",
                message="AdaptivePool3d requires [N, C, D, H, W] input shape",
                input_shape=input_shape,
            )
        if op in {"nn.Flatten", "Flatten"}:
            return NodeValidationResult(
                status="error",
                message="Flatten received invalid input shape",
                input_shape=input_shape,
            )
        if op in {"nn.Unflatten", "Unflatten"}:
            return NodeValidationResult(
                status="error",
                message="Unflatten received incompatible input shape or unflattened_size",
                input_shape=input_shape,
            )
        if op in {"nn.Linear", "Linear"}:
            expected = p.get("in_features")
            actual = input_shape[-1] if input_shape else None
            return NodeValidationResult(
                status="error",
                message=f"Linear in_features mismatch: expected {expected}, got {actual}",
                input_shape=input_shape,
                suggested_repairs=[
                    {"op": "set_param", "node_id": node.node_id, "param": "in_features", "value": actual}
                ],
            )
        if op in {"nn.LSTM", "LSTM", "nn.GRU", "GRU", "nn.RNN", "RNN"}:
            actual_feature = self._infer_rnn_feature_size(
                input_shape=input_shape,
                batch_first=bool(p.get("batch_first", False)),
            )
            return NodeValidationResult(
                status="error",
                message="RNN-family layer expects rank-3 input and matching input_size",
                input_shape=input_shape,
                suggested_repairs=[
                    {
                        "op": "set_param",
                        "node_id": node.node_id,
                        "param": "input_size",
                        "value": actual_feature if isinstance(actual_feature, int) else p.get("input_size"),
                    }
                ],
            )
        if op in {"nn.MultiheadAttention", "MultiheadAttention"}:
            return NodeValidationResult(
                status="error",
                message="MultiheadAttention expects rank-2/3 input and matching embed_dim",
                input_shape=input_shape,
            )
        if op in {
            "nn.TransformerEncoderLayer", "TransformerEncoderLayer",
            "nn.TransformerDecoderLayer", "TransformerDecoderLayer",
        }:
            return NodeValidationResult(
                status="error",
                message="Transformer layer expects rank-3 input and matching d_model",
                input_shape=input_shape,
            )
        if op in {"nn.BatchNorm2d", "BatchNorm2d"}:
            return NodeValidationResult(
                status="error",
                message="BatchNorm2d expects [N, C, H, W] and matching num_features",
                input_shape=input_shape,
            )
        if op in {
            "nn.BatchNorm1d", "BatchNorm1d",
            "nn.BatchNorm3d", "BatchNorm3d",
            "nn.InstanceNorm1d", "InstanceNorm1d",
            "nn.InstanceNorm2d", "InstanceNorm2d",
            "nn.InstanceNorm3d", "InstanceNorm3d",
        }:
            return NodeValidationResult(
                status="error",
                message=f"{op} requires matching num_features and rank-2+ input",
                input_shape=input_shape,
            )
        if op in {"nn.GroupNorm", "GroupNorm"}:
            return NodeValidationResult(
                status="error",
                message="GroupNorm requires positive num_groups dividing num_channels matching input channels",
                input_shape=input_shape,
            )
        if op in {"nn.LayerNorm", "LayerNorm"}:
            return NodeValidationResult(
                status="error",
                message="LayerNorm normalized_shape must match trailing input dimensions",
                input_shape=input_shape,
            )
        if op in {"nn.RMSNorm", "RMSNorm"}:
            return NodeValidationResult(
                status="error",
                message="RMSNorm normalized_shape must match trailing input dimensions",
                input_shape=input_shape,
            )
        if op in {"nn.Embedding", "Embedding"}:
            return NodeValidationResult(
                status="error",
                message="Embedding requires positive num_embeddings and embedding_dim",
                input_shape=input_shape,
            )
        if op in {"torch.add", "add", "Add"}:
            return NodeValidationResult(
                status="error",
                message="Add requires all input shapes to be equal",
                input_shape=input_shape,
            )
        if op in {"torch.cat", "cat", "Concat"}:
            return NodeValidationResult(
                status="error",
                message="Cat requires matching ranks and matching dims except concat dim",
                input_shape=input_shape,
            )
        if op in {"nn.Upsample", "Upsample"}:
            return NodeValidationResult(
                status="error",
                message="Upsample requires rank-3/4/5 input and a size or scale_factor",
                input_shape=input_shape,
            )
        if op in {"nn.PixelShuffle", "PixelShuffle"}:
            return NodeValidationResult(
                status="error",
                message="PixelShuffle requires [N, C, H, W] with C divisible by upscale_factor^2",
                input_shape=input_shape,
            )
        return NodeValidationResult(
            status="error",
            message=f"Shape inference failed for operation '{op}'",
            input_shape=input_shape,
        )


    def _validate_connectivity_and_resolution(
        self,
        graph: ArchitectureGraph,
        node_results: dict[str, NodeValidationResult],
        global_errors: list[str],
    ) -> None:
        if not graph.nodes:
            return

        # Detect self-loops and duplicate edges upfront.
        seen_edges: set[tuple[str, str]] = set()
        for edge in graph.edges:
            src, tgt = edge.source_node_id, edge.target_node_id
            if src == tgt:
                global_errors.append(
                    f"Self-loop detected on node '{src}': an edge cannot connect a node to itself"
                )
            edge_key = (src, tgt)
            if edge_key in seen_edges:
                global_errors.append(
                    f"Duplicate edge from '{src}' to '{tgt}': only one edge between any two nodes is allowed"
                )
            seen_edges.add(edge_key)

        incoming = defaultdict(list)
        outgoing = defaultdict(list)
        for edge in graph.edges:
            incoming[edge.target_node_id].append(edge.source_node_id)
            outgoing[edge.source_node_id].append(edge.target_node_id)

        input_ids = [node.node_id for node in graph.nodes if node.kind == "input"]
        adjacency = outgoing
        reachable = set()
        queue = deque(input_ids)
        while queue:
            node_id = queue.popleft()
            if node_id in reachable:
                continue
            reachable.add(node_id)
            for nxt in adjacency.get(node_id, []):
                if nxt not in reachable:
                    queue.append(nxt)

        has_non_boundary = any(node.kind not in {"input", "output"} for node in graph.nodes)
        if not has_non_boundary and not graph.edges:
            return

        # A kind="module" node with NO incident edge is a canvas group FRAME, not an
        # executable op. An imported hierarchical model runs its dataflow through the
        # block's CHILDREN and gives the wrapper no semantic edges at all (parent/child
        # lives in view.layout.node_data[id].parentId), so treating a frame as a layer made
        # a raw depth-2 resnet18 fail strict validation with exactly 7 errors — one per
        # block. That was not a harmless message: the AI repair loop acted on it, wiring
        # the wrappers into the trunk and then DELETING all 7, so accepting a candidate
        # silently destroyed the model's grouping (71 nodes on the canvas, 65 after).
        container_ids = _container_ids(graph)

        for node in graph.nodes:
            result = node_results.setdefault(node.node_id, NodeValidationResult(status="ok"))
            parents = incoming.get(node.node_id, [])
            # Source cards are joined observations, not editable computational
            # vertices in this legacy drawing. Missing display edges cannot disprove
            # a captured call or justify a source repair. Structural edge validity
            # is still checked above; canonical value wiring replaces this adapter.
            if _source_correspondence(node) is not None:
                continue

            # These suggested_repairs are fed to the AI repair planner as "apply or
            # adapt these", so a placeholder id here is not a harmless hint — it is an
            # instruction to reference a node that does not exist. The literals
            # "input", "last_layer" and "output" used to be emitted verbatim, and
            # planners copied them straight through: gpt-5.5 and gpt-5.6-terra both
            # died on `ValueError: Source node 'input' not found`. Emit a real id or
            # emit nothing; the message alone is already actionable.
            if (
                node.kind in {"layer", "module", "op"}
                and not parents
                and node.node_id not in container_ids
                and _bare(node.op) not in _SOURCE_OPS
            ):
                node_results[node.node_id] = NodeValidationResult(
                    status="error",
                    message="Node has no input connections",
                    suggested_repairs=[
                        {
                            "op": "connect",
                            "source_node_id": input_ids[0],
                            "source_port": "y",
                            "target_node_id": node.node_id,
                            "target_port": "x",
                        }
                    ] if len(input_ids) == 1 else [],
                )
                continue

            if node.kind == "output" and not parents:
                # Only suggest a source when exactly one node has no consumer, which
                # is then unambiguously the dangling tail. Otherwise say nothing
                # rather than nominate an arbitrary layer.
                tails = [
                    candidate.node_id
                    for candidate in graph.nodes
                    if candidate.kind in {"layer", "module", "op"}
                    and not outgoing.get(candidate.node_id)
                    # An edge-free group frame is not a dangling tail. Counting the 7
                    # resnet18 frames inflated `tails` to 8 and suppressed the one real
                    # "connect the tail to output" hint.
                    and candidate.node_id not in container_ids
                ]
                node_results[node.node_id] = NodeValidationResult(
                    status="error",
                    message="Output node is not connected",
                    suggested_repairs=[
                        {
                            "op": "connect",
                            "source_node_id": tails[0],
                            "source_port": "y",
                            "target_node_id": node.node_id,
                            "target_port": "x",
                        }
                    ] if len(tails) == 1 else [],
                )
                continue

            if (
                node.kind != "input"
                and node.node_id not in reachable
                # A frame is never reachable — it has no edges by definition. Without this
                # the exemption above just moves the same 7 errors to this rule.
                and node.node_id not in container_ids
                # And a source is not reachable FROM the input either — it is a second
                # root. `nn.Parameter` is downstream of nothing; it flows INTO the graph.
                # Exempting it from the rule above without exempting it here does exactly
                # what that comment warns about: the same error under a new name.
                and _bare(node.op) not in _SOURCE_OPS
            ):
                node_results[node.node_id] = NodeValidationResult(
                    status="error",
                    message="Node is unreachable from input",
                )
                continue

            if (
                node.kind in {"layer", "module", "op"}
                and parents
                and result.input_shape is None
                # A node the run never reached has no input shape for a reason this
                # pass cannot see, and that reason is already on the result. Replacing
                # it with "could not be resolved" would send the user to check wiring
                # that is fine.
                and not (result.message and _UNREACHED_MARK in result.message)
                # Same exemption, opposite fact: this node has no input shape because
                # an opaque block upstream has no output shape. That is already said
                # once, on the block. Repeating it here as "could not be resolved"
                # sends the user to check wiring that is fine.
                and result.blocked_by is None
                and not (result.message and _UNCHECKED_MARK in result.message)
                # And a node whose OWN output shape was measured is not unresolved,
                # whatever its input looks like from here. The trace ran it and wrote
                # the answer down; "could not be resolved" would send the user to check
                # wiring in front of a node that demonstrably executed.
                and self._recorded_output_shape(node) is None
            ):
                node_results[node.node_id] = NodeValidationResult(
                    status="error",
                    message="Node input shape could not be resolved",
                )

        unresolved_outputs = [node.node_id for node in graph.nodes if node.kind == "output" and node.node_id not in reachable]
        if unresolved_outputs:
            global_errors.append(f"Unreachable output nodes: {sorted(unresolved_outputs)}")

    def _validate_semantic_patterns(
        self,
        graph: ArchitectureGraph,
        node_results: dict[str, NodeValidationResult],
        global_errors: list[str],
    ) -> None:
        node_map = {n.node_id: n for n in graph.nodes}
        incoming = defaultdict(list)
        for edge in graph.edges:
            incoming[edge.target_node_id].append(edge.source_node_id)

        self._validate_attention_semantics(
            graph=graph,
            node_map=node_map,
            incoming=incoming,
            node_results=node_results,
            global_errors=global_errors,
        )
        self._augment_merge_lineage(
            graph=graph,
            node_map=node_map,
            incoming=incoming,
            node_results=node_results,
        )
        self._validate_recurrent_bridges(
            graph=graph,
            node_map=node_map,
            incoming=incoming,
            node_results=node_results,
            global_errors=global_errors,
        )

    def _validate_attention_semantics(
        self,
        graph: ArchitectureGraph,
        node_map: dict[str, ArchitectureNode],
        incoming: dict[str, list[str]],
        node_results: dict[str, NodeValidationResult],
        global_errors: list[str],
    ) -> None:
        attention_ops = {"nn.MultiheadAttention", "MultiheadAttention"}
        for node in graph.nodes:
            if node.op not in attention_ops:
                continue
            parent_ids = incoming.get(node.node_id, [])
            if len(parent_ids) not in {1, 3}:
                current = node_results.get(node.node_id, NodeValidationResult(status="ok"))
                node_results[node.node_id] = NodeValidationResult(
                    status="error",
                    message=(
                        "MultiheadAttention expects either 1 input (self-attention) "
                        "or 3 inputs (query/key/value branches)."
                    ),
                    input_shape=current.input_shape,
                    output_shape=current.output_shape,
                    # Same rule as the structural repairs above: never name a node id
                    # that is not in the graph. With no parent there is nothing
                    # truthful to suggest, so suggest nothing.
                    suggested_repairs=[
                        {
                            "op": "connect",
                            "source_node_id": parent_ids[0],
                            "source_port": "y",
                            "target_node_id": node.node_id,
                            "target_port": "x",
                        }
                    ] if parent_ids else [],
                )
                global_errors.append(f"{node.node_id}: invalid attention branch count {len(parent_ids)}")
                continue

            if len(parent_ids) == 3:
                parent_shapes = []
                for parent_id in parent_ids:
                    parent_result = node_results.get(parent_id)
                    parent_shapes.append(parent_result.output_shape if parent_result else None)
                if any(shape is None for shape in parent_shapes):
                    continue
                if any(len(shape) not in {2, 3} for shape in parent_shapes if shape is not None):
                    node_results[node.node_id] = NodeValidationResult(
                        status="error",
                        message="Q/K/V branches into MultiheadAttention must all be rank-2 or rank-3 tensors.",
                        input_shape=node_results.get(node.node_id).input_shape if node_results.get(node.node_id) else None,
                        output_shape=node_results.get(node.node_id).output_shape if node_results.get(node.node_id) else None,
                        suggested_repairs=[
                            {
                                "op": "set_param",
                                "node_id": node.node_id,
                                "param": "embed_dim",
                                "value": node.params.get("embed_dim", 128),
                            }
                        ],
                    )
                    global_errors.append(f"{node.node_id}: qkv rank mismatch")
                    continue
                feature_sizes = {shape[-1] for shape in parent_shapes if shape is not None}
                if len(feature_sizes) != 1:
                    lineage = self._branch_lineage(parent_ids=parent_ids, node_results=node_results)
                    node_results[node.node_id] = NodeValidationResult(
                        status="error",
                        message=f"Q/K/V feature sizes must match for attention. {lineage}",
                        input_shape=node_results.get(node.node_id).input_shape if node_results.get(node.node_id) else None,
                        output_shape=node_results.get(node.node_id).output_shape if node_results.get(node.node_id) else None,
                        suggested_repairs=[
                            {
                                "op": "set_param",
                                "node_id": parent_ids[-1],
                                "param": "out_features",
                                "value": parent_shapes[0][-1],
                            }
                        ],
                    )
                    global_errors.append(f"{node.node_id}: qkv feature mismatch")

    def _augment_merge_lineage(
        self,
        graph: ArchitectureGraph,
        node_map: dict[str, ArchitectureNode],
        incoming: dict[str, list[str]],
        node_results: dict[str, NodeValidationResult],
    ) -> None:
        del node_map
        merge_ops = {"torch.add", "add", "Add", "torch.cat", "cat", "Concat"}
        for node in graph.nodes:
            if node.op not in merge_ops:
                continue
            result = node_results.get(node.node_id)
            if not result or result.status != "error":
                continue
            parent_ids = incoming.get(node.node_id, [])
            if len(parent_ids) < 2:
                continue
            lineage = self._branch_lineage(parent_ids=parent_ids, node_results=node_results)
            message = result.message or "merge shape mismatch"
            if "Branch lineage:" not in message:
                message = f"{message}. {lineage}"
            suggested = list(result.suggested_repairs)
            if not suggested:
                if node.op in {"torch.add", "add", "Add"}:
                    suggested = [
                        {
                            "op": "remove_connection",
                            "source_node_id": parent_ids[-1],
                            "target_node_id": node.node_id,
                        }
                    ]
                else:
                    suggested = [
                        {
                            "op": "set_param",
                            "node_id": node.node_id,
                            "param": "dim",
                            "value": 1,
                        }
                    ]
            node_results[node.node_id] = NodeValidationResult(
                status=result.status,
                message=message,
                input_shape=result.input_shape,
                output_shape=result.output_shape,
                suggested_repairs=suggested,
            )

    def _validate_recurrent_bridges(
        self,
        graph: ArchitectureGraph,
        node_map: dict[str, ArchitectureNode],
        incoming: dict[str, list[str]],
        node_results: dict[str, NodeValidationResult],
        global_errors: list[str],
    ) -> None:
        rnn_ops = {"nn.LSTM", "LSTM", "nn.GRU", "GRU", "nn.RNN", "RNN"}
        conv_family = {"nn.Conv2d", "Conv2d", "nn.MaxPool2d", "MaxPool2d", "nn.AvgPool2d", "AvgPool2d"}
        for node in graph.nodes:
            if node.op not in rnn_ops:
                continue
            result = node_results.get(node.node_id)
            if not result:
                continue
            ancestors = self._collect_ancestors(node.node_id, incoming)
            has_conv_ancestor = any(node_map.get(ancestor) and node_map[ancestor].op in conv_family for ancestor in ancestors)
            if not has_conv_ancestor:
                continue
            input_shape = result.input_shape
            if input_shape is None:
                continue
            if len(input_shape) == 4:
                node_results[node.node_id] = NodeValidationResult(
                    status="error",
                    message=(
                        "Conv->RNN bridge is invalid: recurrent layers require sequence tensors "
                        "(rank-3) or vector tokens (rank-2)."
                    ),
                    input_shape=result.input_shape,
                    output_shape=result.output_shape,
                    suggested_repairs=[
                        {"op": "add_layer", "node_id": f"{node.node_id}_flatten", "layer_op": "nn.Flatten", "params": {"start_dim": 2}},
                        {"op": "set_param", "node_id": node.node_id, "param": "batch_first", "value": True},
                    ],
                )
                global_errors.append(f"{node.node_id}: conv->rnn rank transition failure")
                continue
            if len(input_shape) == 2:
                node_results[node.node_id] = NodeValidationResult(
                    status="warning",
                    message=(
                        "Conv->RNN bridge uses rank-2 tensor; sequence semantics may be lost. "
                        "Prefer explicit rank-3 [batch, seq, features] adapters."
                    ),
                    input_shape=result.input_shape,
                    output_shape=result.output_shape,
                    suggested_repairs=[
                        {"op": "set_param", "node_id": node.node_id, "param": "batch_first", "value": True}
                    ],
                )

    @staticmethod
    def _collect_ancestors(node_id: str, incoming: dict[str, list[str]]) -> set[str]:
        visited: set[str] = set()
        queue = deque(incoming.get(node_id, []))
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            for parent in incoming.get(current, []):
                if parent not in visited:
                    queue.append(parent)
        return visited

    @staticmethod
    def _branch_lineage(parent_ids: list[str], node_results: dict[str, NodeValidationResult]) -> str:
        fragments = []
        for parent_id in parent_ids:
            parent_result = node_results.get(parent_id)
            shape = parent_result.output_shape if parent_result else None
            fragments.append(f"{parent_id}:{shape}")
        return f"Branch lineage: {', '.join(fragments)}"

    def _apply_strict_mode_checks(
        self,
        graph: ArchitectureGraph,
        node_results: dict[str, NodeValidationResult],
        global_errors: list[str],
    ) -> None:
        # Promote warnings to errors for fail-fast strict mode.
        for node_id, result in list(node_results.items()):
            if result.status == "warning":
                node_results[node_id] = NodeValidationResult(
                    status="error",
                    message=f"Strict mode: {result.message or 'warning promoted to error'}",
                    input_shape=result.input_shape,
                    output_shape=result.output_shape,
                    suggested_repairs=result.suggested_repairs,
                )

        rnn_ops = {"nn.LSTM", "LSTM", "nn.GRU", "GRU", "nn.RNN", "RNN"}
        for node in graph.nodes:
            if node.op not in rnn_ops:
                continue
            result = node_results.get(node.node_id)
            if not result or result.status != "ok":
                continue
            if not result.input_shape or len(result.input_shape) not in {2, 3}:
                node_results[node.node_id] = NodeValidationResult(
                    status="error",
                    message="Strict mode: RNN-family layers require rank-2 or rank-3 input",
                    input_shape=result.input_shape,
                    output_shape=result.output_shape,
                )

        attention_ops = {"nn.MultiheadAttention", "MultiheadAttention", "nn.TransformerEncoderLayer", "TransformerEncoderLayer", "nn.TransformerDecoderLayer", "TransformerDecoderLayer"}
        for node in graph.nodes:
            if node.op not in attention_ops:
                continue
            params = node.params or {}
            nhead = params.get("nhead", params.get("num_heads"))
            d_model = params.get("d_model", params.get("embed_dim"))
            if isinstance(nhead, int) and isinstance(d_model, int) and nhead > 0 and d_model > 0:
                if d_model % nhead != 0:
                    node_results[node.node_id] = NodeValidationResult(
                        status="error",
                        message="Strict mode: d_model/embed_dim must be divisible by nhead/num_heads",
                        input_shape=node_results.get(node.node_id).input_shape if node_results.get(node.node_id) else None,
                        output_shape=node_results.get(node.node_id).output_shape if node_results.get(node.node_id) else None,
                    )

        if any(v.status == "error" for v in node_results.values()):
            global_errors.append("Strict validation failed")


    @staticmethod
    def _infer_rnn_feature_size(input_shape: list[int], batch_first: bool):
        if len(input_shape) == 2:
            return input_shape[1]
        if len(input_shape) == 3:
            return input_shape[2] if batch_first else input_shape[2]
        return None


    @staticmethod
    def _recorded_output_shape(node: ArchitectureNode) -> Optional[list[int]]:
        """The shape a node already carries, parsed. `"2x16"` -> `[2, 16]`.

        Written by the trace worker (measured) or by a `# tensorviz: out_shape=`
        declaration in the class's own source (task.md M6). Never computed here.
        """
        correspondence = _source_correspondence(node)
        if correspondence is not None and correspondence.status not in ("observed", "state"):
            return None
        return _metadata_shape((node.params or {}).get("_out_shape"))

    def _infer_runtime_fallback(
        self,
        node: ArchitectureNode,
        parent_shapes: list[list[int]],
    ):
        if node.op == "ModuleList":
            return None, "ModuleList declares members; this static graph has no executable member for this connection"
        # `python` / `custom_code` name the user's own source. There is no class to
        # build and no forward to run here, so the only honest answer is the one the
        # node was given by something that DID run it. Without one this declines —
        # it must not fall through to the passthrough below, because an opaque
        # statement is not shape-preserving and saying so would put a wrong number
        # in front of every node downstream (task.md M6).
        if is_placeholder_op(node.op or ""):
            recorded = self._recorded_output_shape(node)
            if recorded is not None:
                return recorded, None
            return None, (
                f"'{node.op}' is source code, not an operation: its output shape is "
                "known only from an executed trace or a `# tensorviz: out_shape=` "
                "declaration, and this node carries neither"
            )
        try:
            layer_type = node.op
            if node.op.startswith("nn."):
                layer_type = node.op.replace("nn.", "")
            layer = build_layer(layer_type=layer_type, params=node.params)
            tensors = [torch.randn(*shape) for shape in parent_shapes]
            # A single parent tensor goes through call_module, which supplies the
            # extra positional args (memory / key / value) that transformer and
            # attention modules require. With multiple real parents we pass them
            # through as-is, unwrapping tuple outputs (e.g. MultiheadAttention).
            if len(tensors) == 1:
                output = call_module(layer, tensors[0])
            else:
                output = layer(*tensors)
                if isinstance(output, tuple):
                    output = output[0]
            return list(output.shape), None
        except Exception as exc:
            return None, f"Runtime fallback failed for '{node.op}': {exc}"
