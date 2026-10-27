"""Run canonical validation over all templates and template combinations.

Produces a markdown report with concrete failures so template gaps can be fixed
systematically.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.architecture_models import (  # noqa: E402
    ArchitectureEdge,
    ArchitectureGraph,
    ArchitectureNode,
)
from services.bridge_planner import (  # noqa: E402
    append_edge as _append_edge,
    plan_adapter_transition as _plan_adapter_transition,
    rank_from_input_expectation as _rank_from_input_expectation,
)
from services.canonical_template_registry import CanonicalTemplateRegistry  # noqa: E402
from services.canonical_validation import CanonicalValidationService  # noqa: E402


@dataclass
class SweepIssue:
    level: str  # single | combo
    templates: Tuple[str, ...]
    mode: str  # standard | strict
    issue_type: str  # instantiate_error | invalid
    summary: str
    global_errors: List[str]
    failing_nodes: List[str]


def _sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value.lower())


def _shape_from_entry_node(registry: CanonicalTemplateRegistry, template_id: str) -> List[int] | None:
    try:
        nodes, edges, entry_node_id, _exit_node_id, _cfg = registry.instantiate(
            template_id, f"probe_{template_id}", {}
        )
    except Exception:
        return None

    node_by_id = {node.node_id: node for node in nodes}
    outgoing: Dict[str, List[str]] = {}
    for edge in edges:
        outgoing.setdefault(edge.source_node_id, []).append(edge.target_node_id)

    contract = registry.contract(template_id)
    contract_rank = _rank_from_input_expectation(contract.input_expectation) if contract else None

    # Walk from entry until we hit a node that constrains input rank/features.
    visited: set[str] = set()
    current = entry_node_id
    while current and current not in visited:
        visited.add(current)
        node = node_by_id.get(current)
        if not node:
            break
        op = node.op
        params = node.params or {}

        # Contract-aware fast paths first.
        if contract_rank == 4:
            c = params.get("in_channels") or params.get("num_features") or params.get("num_channels")
            if isinstance(c, int) and c > 0:
                return [1, c, 224, 224]
        if contract_rank == 3:
            c = params.get("in_channels")
            if isinstance(c, int) and c > 0:
                return [1, c, 256]
            f = (
                params.get("input_size")
                or params.get("embed_dim")
                or params.get("in_features")
                or params.get("normalized_shape")
            )
            if isinstance(f, int) and f > 0:
                return [1, 16, f]
            if isinstance(f, list) and f and isinstance(f[0], int) and f[0] > 0:
                return [1, 16, f[0]]
        if contract_rank == 2:
            f = params.get("in_features") or params.get("embed_dim") or params.get("normalized_shape")
            if isinstance(f, int) and f > 0:
                return [1, f]
            if isinstance(f, list) and f and isinstance(f[0], int) and f[0] > 0:
                return [1, f[0]]

        if op in {"nn.Conv2d", "Conv2d"}:
            c = params.get("in_channels")
            if isinstance(c, int) and c > 0:
                return [1, c, 224, 224]
            return [1, 3, 224, 224]
        if op in {"nn.BatchNorm2d", "BatchNorm2d", "nn.GroupNorm", "GroupNorm"}:
            c = params.get("num_features") or params.get("num_channels")
            if isinstance(c, int) and c > 0:
                return [1, c, 224, 224]
        if op in {"nn.Conv1d", "Conv1d"}:
            c = params.get("in_channels")
            if isinstance(c, int) and c > 0:
                return [1, c, 256]
            return [1, 64, 256]
        if op in {"nn.Linear", "Linear"}:
            f = params.get("in_features")
            if isinstance(f, int) and f > 0:
                return [1, f]
            return [1, 256]
        if op in {"nn.Embedding", "Embedding"}:
            return [1, 16]
        if op in {"nn.MultiheadAttention", "MultiheadAttention"}:
            d = params.get("embed_dim")
            if isinstance(d, int) and d > 0:
                return [1, 16, d]
            return [1, 16, 256]
        if op in {"nn.LayerNorm", "LayerNorm"}:
            norm = params.get("normalized_shape")
            if isinstance(norm, int) and norm > 0:
                return [1, 16, norm]
            if isinstance(norm, list) and norm and all(isinstance(v, int) and v > 0 for v in norm):
                if len(norm) == 1:
                    return [1, 16, norm[0]]
                return [1, *norm]
        if op in {"nn.LSTM", "LSTM", "nn.GRU", "GRU", "nn.RNN", "RNN"}:
            f = params.get("input_size")
            if isinstance(f, int) and f > 0:
                return [1, 16, f]
            return [1, 16, 256]

        next_nodes = outgoing.get(current) or []
        current = next_nodes[0] if next_nodes else ""

    if contract_rank == 4:
        return [1, 3, 224, 224]
    if contract_rank == 3:
        return [1, 16, 256]
    if contract_rank == 2:
        return [1, 256]
    return None


def _shape_for_template(registry: CanonicalTemplateRegistry, template_id: str) -> List[int]:
    inferred = _shape_from_entry_node(registry, template_id)
    if inferred:
        return inferred

    tid = template_id.lower()
    if any(token in tid for token in ("conv", "resnet", "vgg", "dense", "unet", "yolo", "fpn", "aspp", "detr")):
        return [1, 3, 224, 224]
    if any(token in tid for token in ("lstm", "gru", "rnn", "seq", "conformer", "wav2vec", "hubert", "whisper", "tcn", "mamba")):
        return [1, 16, 256]
    if any(token in tid for token in ("transformer", "attention", "bert", "gpt", "llama", "qwen", "deberta", "electra", "t5", "bart", "vit", "swin", "patch", "linformer", "performer", "bigbird", "flash")):
        return [1, 128, 768]
    if any(token in tid for token in ("gan", "vae", "flow", "tower", "projection", "fusion", "clap", "qformer", "flamingo", "multimodal")):
        return [1, 512]
    if any(token in tid for token in ("gcn", "gat", "gin", "graph", "rgcn", "message_passing")):
        return [1, 128]
    return [1, 256]


def _build_graph_for_sequence(
    registry: CanonicalTemplateRegistry,
    validator: CanonicalValidationService,
    sequence: Sequence[str],
    graph_index: int,
    enable_adapters: bool,
) -> Tuple[ArchitectureGraph, str | None]:
    input_shape = _shape_for_template(registry, sequence[0])
    graph_id = f"sweep_graph_{graph_index}"
    nodes: List[ArchitectureNode] = [
        ArchitectureNode(node_id="input", kind="input", op="Input", params={"shape": input_shape}, in_ports=[], out_ports=["y"]),
        ArchitectureNode(node_id="output", kind="output", op="Output", params={}, in_ports=["x"], out_ports=[]),
    ]
    edges: List[ArchitectureEdge] = []

    previous_exit = "input"
    unsupported_transition_reason: str | None = None
    for idx, template_id in enumerate(sequence, start=1):
        digest = hashlib.sha1(f"{graph_index}:{idx}:{template_id}".encode("utf-8")).hexdigest()[:8]
        instance_id = f"t{idx}_{_sanitize(template_id)}_{digest}"
        template_nodes, template_edges, entry_node_id, exit_node_id, _ = registry.instantiate(
            template_id=template_id,
            instance_id=instance_id,
            config={},
        )
        incoming_counts: Dict[str, int] = {node.node_id: 0 for node in template_nodes}
        for edge in template_edges:
            if edge.target_node_id in incoming_counts:
                incoming_counts[edge.target_node_id] += 1
        template_roots = [node_id for node_id, count in incoming_counts.items() if count == 0]
        if not template_roots:
            template_roots = [entry_node_id]

        nodes.extend(template_nodes)
        edges.extend(template_edges)
        entry_node = next((n for n in template_nodes if n.node_id == entry_node_id), None)
        if enable_adapters and entry_node is not None and idx > 1:
            adapter_prefix = f"bridge_{idx}_{_sanitize(template_id)}_{digest}"
            # Contract-rank lookup stays in the sweep (registry coupling); the
            # probe/resolve/rank/insert orchestration is the shared product path.
            contract_rank = _rank_from_input_expectation(registry.contract(template_id).input_expectation)
            previous_exit, _ins_nodes, _ins_edges, transition_issue = _plan_adapter_transition(
                validator=validator,
                graph_id=graph_id,
                input_shape=input_shape,
                nodes=nodes,
                edges=edges,
                source_node_id=previous_exit,
                target_node=entry_node,
                adapter_prefix=adapter_prefix,
                contract_rank=contract_rank,
                target_template_id=template_id,
                template_nodes=template_nodes,
                template_edges=template_edges,
                entry_node_id=entry_node_id,
            )
            if transition_issue and unsupported_transition_reason is None:
                unsupported_transition_reason = f"{sequence[idx - 2]} -> {template_id}: {transition_issue}"

        for root_node_id in template_roots:
            _append_edge(
                edges=edges,
                source=previous_exit,
                target=root_node_id,
                source_port="y",
                target_port="x",
            )
        previous_exit = exit_node_id

    _append_edge(edges=edges, source=previous_exit, target="output", source_port="y", target_port="x")

    graph = ArchitectureGraph(
        graph_id=graph_id,
        name=f"sweep_{'_'.join(sequence)}",
        revision=0,
        inputs=[{"name": "x", "shape": input_shape}],
        nodes=nodes,
        edges=edges,
        metadata={"sweep_sequence": list(sequence)},
    )
    return graph, unsupported_transition_reason


def _failing_nodes(validation_result) -> List[str]:
    nodes = []
    for node_id, result in validation_result.node_results.items():
        if result.status == "error":
            msg = result.message or "Unknown node validation error"
            nodes.append(f"{node_id}: {msg}")
    return nodes


def _run_case(
    *,
    registry: CanonicalTemplateRegistry,
    validator: CanonicalValidationService,
    sequence: Sequence[str],
    graph_index: int,
    strict: bool,
    enable_adapters: bool,
) -> Tuple[bool, SweepIssue | None]:
    mode = "strict" if strict else "standard"
    try:
        graph, unsupported_transition_reason = _build_graph_for_sequence(
            registry=registry,
            validator=validator,
            sequence=sequence,
            graph_index=graph_index,
            enable_adapters=enable_adapters,
        )
    except Exception as exc:  # noqa: BLE001
        return False, SweepIssue(
            level="single" if len(sequence) == 1 else "combo",
            templates=tuple(sequence),
            mode=mode,
            issue_type="instantiate_error",
            summary=str(exc),
            global_errors=[],
            failing_nodes=[],
        )

    if enable_adapters and unsupported_transition_reason:
        return False, SweepIssue(
            level="single" if len(sequence) == 1 else "combo",
            templates=tuple(sequence),
            mode=mode,
            issue_type="unsupported_transition",
            summary=unsupported_transition_reason,
            global_errors=[],
            failing_nodes=[],
        )

    validation = validator.validate(graph=graph, strict=strict)
    if validation.valid:
        return True, None

    global_errors = list(validation.global_errors)
    failing_nodes = _failing_nodes(validation)
    summary = global_errors[0] if global_errors else (failing_nodes[0] if failing_nodes else "Unknown validation failure")
    issue = SweepIssue(
        level="single" if len(sequence) == 1 else "combo",
        templates=tuple(sequence),
        mode=mode,
        issue_type="invalid",
        summary=summary,
        global_errors=global_errors,
        failing_nodes=failing_nodes,
    )
    return False, issue


def _iter_combinations(template_ids: Sequence[str], combo_size: int, allow_repeat: bool) -> Iterable[Tuple[str, ...]]:
    if combo_size <= 1:
        for template_id in template_ids:
            yield (template_id,)
        return

    if allow_repeat:
        yield from itertools.product(template_ids, repeat=combo_size)
    else:
        # Order matters for directed graph chaining; skip repeated ids in a sequence.
        for seq in itertools.product(template_ids, repeat=combo_size):
            if len(set(seq)) != len(seq):
                continue
            yield seq


def _render_markdown(
    *,
    template_ids: Sequence[str],
    single_total: int,
    single_passed: int,
    combo_total: int,
    combo_passed: int,
    duration_seconds: float,
    issues: List[SweepIssue],
    top_n_issues: int,
    adapters_enabled: bool,
) -> str:
    def _md_cell(value: str) -> str:
        return value.replace("|", "\\|")

    lines: List[str] = []
    lines.append("# Template Validation Sweep Report")
    lines.append("")
    lines.append(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    lines.append(f"- Total templates discovered: **{len(template_ids)}**")
    lines.append(f"- Adapter mode: **{'enabled' if adapters_enabled else 'disabled'}**")
    lines.append(f"- Single-template checks: **{single_passed}/{single_total} passed**")
    lines.append(f"- Combination checks: **{combo_passed}/{combo_total} passed**")
    lines.append(f"- Total duration: **{duration_seconds:.2f}s**")
    lines.append("")

    issue_counter = Counter(issue.summary for issue in issues)
    lines.append("## Top Issue Buckets")
    lines.append("")
    if issue_counter:
        lines.append("| Count | Summary |")
        lines.append("|---:|---|")
        for summary, count in issue_counter.most_common(top_n_issues):
            lines.append(f"| {count} | {_md_cell(summary)} |")
    else:
        lines.append("_No issues found._")
    lines.append("")

    lines.append("## Single-Template Failures")
    lines.append("")
    single_issues = [issue for issue in issues if issue.level == "single"]
    if single_issues:
        lines.append("| Template | Mode | Type | Summary |")
        lines.append("|---|---|---|---|")
        for issue in single_issues:
            lines.append(
                f"| `{issue.templates[0]}` | {issue.mode} | {issue.issue_type} | {_md_cell(issue.summary)} |"
            )
    else:
        lines.append("_No single-template failures._")
    lines.append("")

    lines.append("## Combination Failures")
    lines.append("")
    combo_issues = [issue for issue in issues if issue.level == "combo"]
    if combo_issues:
        lines.append("| Sequence | Mode | Type | Summary |")
        lines.append("|---|---|---|---|")
        for issue in combo_issues[:1000]:
            seq = " -> ".join(f"`{template}`" for template in issue.templates)
            lines.append(
                f"| {seq} | {issue.mode} | {issue.issue_type} | {_md_cell(issue.summary)} |"
            )
        if len(combo_issues) > 1000:
            lines.append("")
            lines.append(f"_Truncated to first 1000 rows (total combo failures: {len(combo_issues)})._")
    else:
        lines.append("_No combination failures._")
    lines.append("")

    lines.append("## Detailed Failures")
    lines.append("")
    for idx, issue in enumerate(issues[:200], start=1):
        lines.append(f"### {idx}. {' -> '.join(issue.templates)} ({issue.mode}, {issue.issue_type})")
        lines.append("")
        lines.append(f"- Summary: {issue.summary}")
        if issue.global_errors:
            lines.append("- Global errors:")
            for err in issue.global_errors[:8]:
                lines.append(f"  - {err}")
        if issue.failing_nodes:
            lines.append("- Failing nodes:")
            for node in issue.failing_nodes[:10]:
                lines.append(f"  - {node}")
        lines.append("")
    if len(issues) > 200:
        lines.append(f"_Detailed section truncated to first 200 issues (total issues: {len(issues)})._")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run canonical template validation sweep.")
    parser.add_argument("--combo-size", type=int, default=2, help="Template sequence size for combination checks.")
    parser.add_argument("--max-combinations", type=int, default=0, help="Optional cap on combination attempts (0 = all).")
    parser.add_argument("--allow-repeat", action="store_true", help="Allow repeated template IDs in combinations.")
    parser.add_argument("--strict", action="store_true", help="Run strict validation mode in addition to standard mode.")
    parser.add_argument(
        "--enable-adapters",
        action="store_true",
        help="Enable inter-template adapter insertion (bridge mode).",
    )
    parser.add_argument(
        "--disable-adapters",
        action="store_true",
        help="Disable inter-template adapter insertion (override default).",
    )
    parser.add_argument("--top-n-issues", type=int, default=25, help="Top issue buckets in summary.")
    parser.add_argument(
        "--out",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports", "template-validation-report.md")),
        help="Output markdown report path.",
    )
    args = parser.parse_args()
    adapters_enabled = args.enable_adapters or not args.disable_adapters

    registry = CanonicalTemplateRegistry()
    validator = CanonicalValidationService()
    template_ids = [contract.template_id for contract in registry.list_contracts()]

    issues: List[SweepIssue] = []
    single_total = 0
    single_passed = 0
    combo_total = 0
    combo_passed = 0

    graph_index = 0
    started_at = time.perf_counter()
    modes = [False, True] if args.strict else [False]

    # Single-template pass.
    for template_id in template_ids:
        for strict_mode in modes:
            graph_index += 1
            single_total += 1
            passed, issue = _run_case(
                registry=registry,
                validator=validator,
                sequence=(template_id,),
                graph_index=graph_index,
                strict=strict_mode,
                enable_adapters=adapters_enabled,
            )
            if passed:
                single_passed += 1
            elif issue:
                issues.append(issue)

    # Combination pass.
    combinations_iter = _iter_combinations(
        template_ids=template_ids,
        combo_size=max(2, args.combo_size),
        allow_repeat=args.allow_repeat,
    )
    for combo_idx, sequence in enumerate(combinations_iter, start=1):
        if args.max_combinations > 0 and combo_idx > args.max_combinations:
            break
        for strict_mode in modes:
            graph_index += 1
            combo_total += 1
            passed, issue = _run_case(
                registry=registry,
                validator=validator,
                sequence=sequence,
                graph_index=graph_index,
                strict=strict_mode,
                enable_adapters=adapters_enabled,
            )
            if passed:
                combo_passed += 1
            elif issue:
                issues.append(issue)
        if combo_idx % 200 == 0:
            print(f"[sweep] combinations processed: {combo_idx}")

    duration = time.perf_counter() - started_at
    markdown = _render_markdown(
        template_ids=template_ids,
        single_total=single_total,
        single_passed=single_passed,
        combo_total=combo_total,
        combo_passed=combo_passed,
        duration_seconds=duration,
        issues=issues,
        top_n_issues=args.top_n_issues,
        adapters_enabled=adapters_enabled,
    )

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")

    print(f"[sweep] templates={len(template_ids)} single={single_passed}/{single_total} combo={combo_passed}/{combo_total}")
    print(f"[sweep] issues={len(issues)} duration={duration:.2f}s")
    print(f"[sweep] report={out_path}")


if __name__ == "__main__":
    main()
