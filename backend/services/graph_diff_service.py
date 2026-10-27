"""Structural diff between two ArchitectureGraphs.

Given two graphs (e.g. snapshots before/after a refactor), compute which
nodes/edges were added, removed, or modified. Node identity is matched
by ID first; remaining nodes are matched by (op, params-fingerprint)
with topological-position as a tiebreaker, so renamed-but-structurally-
identical nodes are still treated as the same conceptual layer.
"""

import hashlib
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from models.architecture_models import (
    ArchitectureEdge,
    ArchitectureGraph,
    ArchitectureNode,
)


class NodeDiff(BaseModel):
    """One node-level diff entry."""

    status: str  # "added" | "removed" | "modified" | "unchanged"
    node_id_a: Optional[str] = None
    node_id_b: Optional[str] = None
    op: str
    param_changes: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    # param_changes maps param name -> {"before": ..., "after": ...}


class EdgeDiff(BaseModel):
    """One edge-level diff entry."""

    status: str  # "added" | "removed" | "unchanged"
    edge_id_a: Optional[str] = None
    edge_id_b: Optional[str] = None
    source_node_id_a: Optional[str] = None
    target_node_id_a: Optional[str] = None
    source_node_id_b: Optional[str] = None
    target_node_id_b: Optional[str] = None


class GraphDiffResult(BaseModel):
    """Full diff between graph A and graph B."""

    nodes: List[NodeDiff] = Field(default_factory=list)
    edges: List[EdgeDiff] = Field(default_factory=list)
    correspondence: Dict[str, str] = Field(default_factory=dict)
    # correspondence: node_id_a -> node_id_b for matched pairs
    summary: Dict[str, int] = Field(default_factory=dict)
    # summary: counts of {added_nodes, removed_nodes, modified_nodes, added_edges, removed_edges}


def _fingerprint_params(params: Dict[str, Any]) -> str:
    """Stable hash of a node's params dict. Used as a structural key for
    matching nodes across two graphs."""
    canonical = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _topo_index(graph: ArchitectureGraph) -> Dict[str, int]:
    """Best-effort topological position per node. Independent of ID
    naming, so two graphs with the same shape but renamed nodes still
    line up. Falls back to insertion order when there's a cycle."""
    indeg: Dict[str, int] = {n.node_id: 0 for n in graph.nodes}
    succ: Dict[str, List[str]] = defaultdict(list)
    for e in graph.edges:
        if e.source_node_id in indeg and e.target_node_id in indeg:
            indeg[e.target_node_id] += 1
            succ[e.source_node_id].append(e.target_node_id)
    queue = [n.node_id for n in graph.nodes if indeg[n.node_id] == 0]
    order: List[str] = []
    while queue:
        node_id = queue.pop(0)
        order.append(node_id)
        for nxt in succ[node_id]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    if len(order) != len(graph.nodes):
        # Cycle; fall back to insertion order.
        order = [n.node_id for n in graph.nodes]
    return {node_id: i for i, node_id in enumerate(order)}


def _diff_params(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return per-key {before, after} for keys whose values differ."""
    changes: Dict[str, Dict[str, Any]] = {}
    for key in set(a.keys()) | set(b.keys()):
        va, vb = a.get(key), b.get(key)
        if va != vb:
            changes[key] = {"before": va, "after": vb}
    return changes


class GraphDiffService:
    """Compute the structural diff between two ArchitectureGraphs."""

    def diff(
        self,
        graph_a: ArchitectureGraph,
        graph_b: ArchitectureGraph,
        *,
        correspondence: Optional[Dict[str, str]] = None,
    ) -> GraphDiffResult:
        nodes_by_id_a: Dict[str, ArchitectureNode] = {n.node_id: n for n in graph_a.nodes}
        nodes_by_id_b: Dict[str, ArchitectureNode] = {n.node_id: n for n in graph_b.nodes}

        matched_a_to_b: Dict[str, str] = {}
        used_b: set[str] = set()

        # 1. Honor user-provided correspondence first.
        if correspondence:
            for a_id, b_id in correspondence.items():
                if a_id in nodes_by_id_a and b_id in nodes_by_id_b and b_id not in used_b:
                    matched_a_to_b[a_id] = b_id
                    used_b.add(b_id)

        # 2. Exact node_id match for unmatched nodes.
        for a_id in nodes_by_id_a:
            if a_id in matched_a_to_b:
                continue
            if a_id in nodes_by_id_b and a_id not in used_b:
                matched_a_to_b[a_id] = a_id
                used_b.add(a_id)

        # 3. Structural match by (op, params_fingerprint) for remaining,
        #    breaking ties by topological position.
        topo_a = _topo_index(graph_a)
        topo_b = _topo_index(graph_b)
        unmatched_a = [n for n in graph_a.nodes if n.node_id not in matched_a_to_b]
        bucket_b: Dict[Tuple[str, str], List[ArchitectureNode]] = defaultdict(list)
        for n in graph_b.nodes:
            if n.node_id in used_b:
                continue
            key = (n.op, _fingerprint_params(n.params))
            bucket_b[key].append(n)
        for bucket in bucket_b.values():
            bucket.sort(key=lambda n: topo_b.get(n.node_id, 0))

        unmatched_a.sort(key=lambda n: topo_a.get(n.node_id, 0))
        for node_a in unmatched_a:
            key = (node_a.op, _fingerprint_params(node_a.params))
            candidates = bucket_b.get(key)
            if candidates:
                node_b = candidates.pop(0)
                matched_a_to_b[node_a.node_id] = node_b.node_id
                used_b.add(node_b.node_id)

        # 4. Fall back to (op only) match — params changed, structure
        #    preserved. Used for the "modified" bucket.
        unmatched_a = [n for n in graph_a.nodes if n.node_id not in matched_a_to_b]
        bucket_b_op: Dict[str, List[ArchitectureNode]] = defaultdict(list)
        for n in graph_b.nodes:
            if n.node_id in used_b:
                continue
            bucket_b_op[n.op].append(n)
        for bucket in bucket_b_op.values():
            bucket.sort(key=lambda n: topo_b.get(n.node_id, 0))
        unmatched_a.sort(key=lambda n: topo_a.get(n.node_id, 0))
        for node_a in unmatched_a:
            candidates = bucket_b_op.get(node_a.op)
            if candidates:
                node_b = candidates.pop(0)
                matched_a_to_b[node_a.node_id] = node_b.node_id
                used_b.add(node_b.node_id)

        # 5. Build the node diff list.
        node_diffs: List[NodeDiff] = []
        for node_a in graph_a.nodes:
            b_id = matched_a_to_b.get(node_a.node_id)
            if b_id is None:
                node_diffs.append(NodeDiff(
                    status="removed",
                    node_id_a=node_a.node_id,
                    op=node_a.op,
                ))
                continue
            node_b = nodes_by_id_b[b_id]
            changes = _diff_params(node_a.params, node_b.params)
            if changes or node_a.op != node_b.op:
                node_diffs.append(NodeDiff(
                    status="modified",
                    node_id_a=node_a.node_id,
                    node_id_b=b_id,
                    op=node_b.op,
                    param_changes=changes,
                ))
            else:
                node_diffs.append(NodeDiff(
                    status="unchanged",
                    node_id_a=node_a.node_id,
                    node_id_b=b_id,
                    op=node_b.op,
                ))
        for node_b in graph_b.nodes:
            if node_b.node_id not in used_b:
                node_diffs.append(NodeDiff(
                    status="added",
                    node_id_b=node_b.node_id,
                    op=node_b.op,
                ))

        # 6. Edge diff. Translate edges in A to their B-equivalents via
        #    the node correspondence, then compare the resulting edge
        #    sets.
        edges_by_pair_b: Dict[Tuple[str, str, str, str], ArchitectureEdge] = {
            (e.source_node_id, e.source_port, e.target_node_id, e.target_port): e
            for e in graph_b.edges
        }
        seen_b_pairs: set[Tuple[str, str, str, str]] = set()
        edge_diffs: List[EdgeDiff] = []
        for edge_a in graph_a.edges:
            src_b = matched_a_to_b.get(edge_a.source_node_id)
            tgt_b = matched_a_to_b.get(edge_a.target_node_id)
            if src_b and tgt_b:
                key = (src_b, edge_a.source_port, tgt_b, edge_a.target_port)
                if key in edges_by_pair_b:
                    edge_b = edges_by_pair_b[key]
                    seen_b_pairs.add(key)
                    edge_diffs.append(EdgeDiff(
                        status="unchanged",
                        edge_id_a=edge_a.edge_id,
                        edge_id_b=edge_b.edge_id,
                        source_node_id_a=edge_a.source_node_id,
                        target_node_id_a=edge_a.target_node_id,
                        source_node_id_b=edge_b.source_node_id,
                        target_node_id_b=edge_b.target_node_id,
                    ))
                    continue
            edge_diffs.append(EdgeDiff(
                status="removed",
                edge_id_a=edge_a.edge_id,
                source_node_id_a=edge_a.source_node_id,
                target_node_id_a=edge_a.target_node_id,
            ))
        for edge_b in graph_b.edges:
            key = (edge_b.source_node_id, edge_b.source_port, edge_b.target_node_id, edge_b.target_port)
            if key in seen_b_pairs:
                continue
            edge_diffs.append(EdgeDiff(
                status="added",
                edge_id_b=edge_b.edge_id,
                source_node_id_b=edge_b.source_node_id,
                target_node_id_b=edge_b.target_node_id,
            ))

        summary = {
            "added_nodes": sum(1 for d in node_diffs if d.status == "added"),
            "removed_nodes": sum(1 for d in node_diffs if d.status == "removed"),
            "modified_nodes": sum(1 for d in node_diffs if d.status == "modified"),
            "unchanged_nodes": sum(1 for d in node_diffs if d.status == "unchanged"),
            "added_edges": sum(1 for d in edge_diffs if d.status == "added"),
            "removed_edges": sum(1 for d in edge_diffs if d.status == "removed"),
            "unchanged_edges": sum(1 for d in edge_diffs if d.status == "unchanged"),
        }
        return GraphDiffResult(
            nodes=node_diffs,
            edges=edge_diffs,
            correspondence=matched_a_to_b,
            summary=summary,
        )
