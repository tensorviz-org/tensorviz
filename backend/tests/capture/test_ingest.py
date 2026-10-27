"""Tests for model ingestion.

Each test encodes WHY the behaviour matters, because most of these guard against failure
modes that are invisible in JSON and only show up as a wrong picture on the canvas.

Only torch + torchvision are required. transformers/diffusers/timm are optional extras and
the tests that would need them are skipped rather than silently passing.
"""

from __future__ import annotations

import os
import sys
import textwrap

import pytest
import torch
import torch.nn as nn

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import ingest as ingest_router
from services.ingest import ModelRef, available_libraries, catalog, ingest, scan_only
from services.ingest.document import to_document
from services.ingest.repo_scan import StubbedDependencyError
from services.ingest.introspect import (
    apply_call_counts,
    build_tree,
    constructor_kwargs,
    fused_parameter_counts,
    tied_parameter_groups,
)
from services.ingest.materialise import Plan, splittable_leaves
from services.ingest.normalise import layered_positions
from services.ingest.trace import dataflow_edges, resolve_dataflow

CARD_W, CARD_H = 200, 120

_HAS_TORCHVISION = available_libraries()["torchvision"]
requires_torchvision = pytest.mark.skipif(
    not _HAS_TORCHVISION, reason="torchvision not installed"
)


# ───────────────────────────────────────────────── introspection ─────────────
def test_constructor_kwargs_recovers_real_arguments():
    """Without real constructor kwargs the backend layer builder cannot rebuild a node:
    it fails with "GroupNorm.__init__() missing 2 required positional arguments"."""
    kwargs = constructor_kwargs(nn.GroupNorm(num_groups=32, num_channels=320))
    assert kwargs["num_groups"] == 32
    assert kwargs["num_channels"] == 320

    conv = constructor_kwargs(nn.Conv2d(3, 16, kernel_size=3, padding=1))
    assert conv["in_channels"] == 3
    assert conv["out_channels"] == 16
    assert conv["kernel_size"] == [3, 3]


def test_importer_metadata_is_underscore_prefixed_so_it_cannot_break_construction():
    """data.parameters IS the constructor-kwarg namespace — the layer builder splats it
    into nn.<Class>(**params). Bare metadata there produces
    "SiLU.__init__() got an unexpected keyword argument 'module_path'". The `_` prefix is
    what core/layer_builder.py strips."""
    result = ingest(ModelRef(library="torchvision", arch="resnet18", depth=1))
    metadata_keys = set()
    for node in result.graph["nodes"]:
        for key in node["data"]["parameters"]:
            if key in ("name", "shape"):
                continue  # boundary-node conventions predate ingestion
            if key.startswith("_"):
                metadata_keys.add(key)
            else:
                # every non-underscore key must be a plausible constructor argument
                assert key.islower(), f"{key!r} looks like metadata but is not _-prefixed"
    assert "_module_path" in metadata_keys
    assert "_class" in metadata_keys


def test_tied_parameters_are_detected_by_identity():
    """Weight tying is a real architectural fact (an embedding reused as the output head)
    and it is recoverable deterministically from id() aliasing — no heuristics."""

    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(50, 8)
            self.head = nn.Linear(8, 50, bias=False)
            self.head.weight = self.embed.weight

    groups = tied_parameter_groups(Tied())
    assert groups == [["embed.weight", "head.weight"]]


def test_fused_parameter_rule_sees_experts_that_are_not_modules():
    """Modern Mixture-of-Experts layers hold their experts as a leading axis on fused 3-D
    parameters, not as modules. An 8-expert Mixtral block is only ~4 modules, so a
    ModuleList-based collapse reports nothing and an 8-expert sparse layer would render as
    one dense box."""

    class FusedExperts(nn.Linear):
        def __init__(self):
            super().__init__(64, 64)
            self.gate_up_proj = nn.Parameter(torch.zeros(8, 256, 64))

    assert fused_parameter_counts(FusedExperts()) == 8
    assert fused_parameter_counts(nn.Linear(8, 8)) is None


def test_module_list_repeats_collapse_to_one_canonical_block():
    """A 12-block stack must render as one card with x12, not twelve cards."""

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 16)

    class Stack(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Block() for _ in range(12)])

    intro = build_tree(Stack(), max_depth=2)
    repeats = {r.path: (r.count, r.reason) for r in intro.repeats.values()}
    assert repeats.get("blocks.0") == (12, "module-list")
    # the ModuleList itself is a Python container and must not become a card
    assert "blocks" not in intro.nodes


def test_call_count_repeats_apply_to_composites_but_never_to_reused_leaves():
    """A repeated *leaf* is reuse, not depth. ResNet's BasicBlock applies a single
    self.relu twice; stamping "ReLU x2" on it describes one activation as two layers.
    A repeated composite (ALBERT's shared layer group) genuinely is N layers."""

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 8)

    class Looped(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = Inner()
            self.act = nn.ReLU()

    intro = build_tree(Looped(), max_depth=3)
    apply_call_counts(intro, {"block": 6, "block.fc": 6, "act": 2})
    repeats = {r.path: (r.count, r.reason) for r in intro.repeats.values()}
    assert repeats.get("block") == (6, "call-count"), "composite loop must yield xN"
    assert "act" not in repeats, "a reused leaf activation must NOT be reported as xN"


# ───────────────────────────────────────────────────── dataflow ──────────────
def test_a_module_called_repeatedly_does_not_self_edge():
    """Call k's output is call k+1's input on the same node id. Emitting that as an edge
    makes the graph cyclic, and longest-path layout and codegen both walk a DAG."""
    ops = [
        {"op": "aten.linear", "owner": "block", "in_ids": [1], "out_ids": [2]},
        {"op": "aten.linear", "owner": "block", "in_ids": [2], "out_ids": [3]},
    ]
    pairs = dataflow_edges(ops, rendered={"block"}, parent={})
    assert not [p for p in pairs if p[0] == p[1]]


def test_a_merge_op_keeps_every_contributing_producer():
    """A residual add or torch.cat has several producers. Collapsing to one silently drops
    real edges — gpt2 loses `wpe -> drop` entirely."""
    ops = [
        {"op": "aten.embedding", "owner": "wte", "in_ids": [1], "out_ids": [10]},
        {"op": "aten.embedding", "owner": "wpe", "in_ids": [2], "out_ids": [11]},
        # the add lives in the parent frame, which owns rendered children
        {"op": "aten.add", "owner": "root", "in_ids": [10, 11], "out_ids": [12]},
        {"op": "aten.native_dropout", "owner": "drop", "in_ids": [12], "out_ids": [13]},
    ]
    rendered = {"root", "wte", "wpe", "drop"}
    parent = {"wte": "root", "wpe": "root", "drop": "root"}
    pairs = dataflow_edges(ops, rendered, parent)
    assert ("wte", "drop") in pairs
    assert ("wpe", "drop") in pairs, "the second producer of a merge must survive"


# ──────────────────────────────────── materialised merges & call sites ──────
def test_the_fold_is_untouched_when_nothing_is_materialised():
    """`dataflow_edges` is the pinned contract. Passing no plan must reproduce it
    exactly, so the two transparency tests above still guard the live code path."""
    ops = [
        {"op": "aten.embedding", "owner": "wte", "in_ids": [1], "out_ids": [10]},
        {"op": "aten.embedding", "owner": "wpe", "in_ids": [2], "out_ids": [11]},
        {"op": "aten.add", "owner": "root", "in_ids": [10, 11], "out_ids": [12]},
        {"op": "aten.native_dropout", "owner": "drop", "in_ids": [12], "out_ids": [13]},
    ]
    rendered = {"root", "wte", "wpe", "drop"}
    parent = {"wte": "root", "wpe": "root", "drop": "root"}

    assert dataflow_edges(ops, rendered, parent) == set(
        resolve_dataflow(ops, rendered, parent)
    )


def test_a_residual_add_becomes_its_own_node_instead_of_fanning_into_the_consumer():
    """The defect: both producers were unioned into the CONSUMER, so a real nn.Module
    received two inputs and the runtime raised "arity error: ... expects exactly 1".
    Measured before the fix — mobilenet_v2 5 such nodes, densenet121 58."""
    ops = [
        # The identity path's producer must be a real node — in resnet18 it is maxpool.
        {"op": "aten.max_pool2d", "owner": "stem", "fn": "max_pool2d",
         "owner_call": 1, "in_ids": [1], "out_ids": [2]},
        {"op": "aten.convolution", "owner": "conv", "fn": "convolution",
         "owner_call": 1, "in_ids": [2], "out_ids": [10]},
        {"op": "aten.add_.Tensor", "owner": "root", "fn": "add", "owner_call": 1,
         "in_ids": [10, 2], "out_ids": [11], "out_shape": [1, 8]},
        {"op": "aten.relu", "owner": "act", "fn": "relu", "owner_call": 1,
         "in_ids": [11], "out_ids": [12]},
    ]
    rendered = {"root", "stem", "conv", "act"}
    parent = {"conv": "root", "act": "root"}
    plan = Plan(taken=frozenset(rendered))

    pairs = resolve_dataflow(ops, rendered, parent, plan)

    merges = [n for n, s in plan.nodes.items() if s.kind == "merge"]
    assert len(merges) == 1, f"expected one materialised add, got {plan.nodes}"
    add = merges[0]
    assert plan.nodes[add].layer_id == "add", "must key on the id both consumers read"
    # The consumer now takes ONE input, from the merge, instead of two.
    assert [p for p in pairs if p[1] == "act"] == [(add, "act")]
    assert ("conv", add) in pairs
    assert ("stem", add) in pairs, "the identity path must reach the merge"


def test_a_scalar_add_is_not_materialised():
    """`x + 1` has a single producer. Making a node for it would litter the canvas with
    merges that merge nothing."""
    ops = [
        {"op": "aten.convolution", "owner": "conv", "fn": "convolution",
         "owner_call": 1, "in_ids": [1], "out_ids": [10]},
        {"op": "aten.add.Tensor", "owner": "root", "fn": "add", "owner_call": 1,
         "in_ids": [10], "out_ids": [11], "out_shape": [1, 8]},
        {"op": "aten.relu", "owner": "act", "fn": "relu", "owner_call": 1,
         "in_ids": [11], "out_ids": [12]},
    ]
    rendered = {"root", "conv", "act"}
    plan = Plan(taken=frozenset(rendered))

    pairs = resolve_dataflow(ops, rendered, {"conv": "root", "act": "root"}, plan)

    assert not plan.nodes, "a one-producer add must stay transparent"
    assert ("conv", "act") in pairs


def test_a_reused_leaf_gets_one_node_per_call_site():
    """torchvision's BasicBlock calls self.relu twice. One name meant one node, which
    collected edges from both call sites and closed a cycle
    (relu -> conv2 -> bn2 -> relu): the imported graph was not a DAG."""
    ops = [
        {"op": "aten.relu", "owner": "relu", "fn": "relu", "owner_call": 1,
         "in_ids": [1], "out_ids": [10]},
        {"op": "aten.convolution", "owner": "conv2", "fn": "convolution",
         "owner_call": 1, "in_ids": [10], "out_ids": [11]},
        {"op": "aten.relu", "owner": "relu", "fn": "relu", "owner_call": 2,
         "in_ids": [11], "out_ids": [12]},
    ]
    rendered = {"relu", "conv2"}
    plan = Plan(splittable=frozenset({"relu"}), taken=frozenset(rendered))

    pairs = resolve_dataflow(ops, rendered, {}, plan)

    assert ("relu", "conv2") in pairs
    assert ("conv2", "relu") not in pairs, "this pair is what made the graph cyclic"
    assert ("conv2", "relu_call2") in pairs
    assert plan.nodes["relu_call2"].origin == "relu"


def test_a_reused_leaf_is_left_alone_when_it_was_not_declared_splittable():
    ops = [
        {"op": "aten.relu", "owner": "relu", "fn": "relu", "owner_call": 1,
         "in_ids": [1], "out_ids": [10]},
        {"op": "aten.relu", "owner": "relu", "fn": "relu", "owner_call": 2,
         "in_ids": [10], "out_ids": [11]},
    ]
    plan = Plan(taken=frozenset({"relu"}))

    resolve_dataflow(ops, {"relu"}, {}, plan)

    assert not plan.nodes


def test_a_scope_holding_a_parameterised_module_refuses_to_split():
    """`for _ in range(4): x = relu(fc(x))` — splitting only the stateless relu would
    leave fc newly receiving four inbound edges, turning a clean graph into a broken
    one. So the whole scope refuses, and says so."""
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    intro = build_tree(model, 3)
    names = {n for n in intro.nodes if n}
    relu = next(n for n in names if intro.nodes[n].cls == "ReLU")
    linear = next(n for n in names if intro.nodes[n].cls == "Linear")

    ok, refused = splittable_leaves(intro, [frozenset({relu, linear})])

    assert ok == frozenset()
    assert set(refused) == {relu, linear}

    ok_alone, refused_alone = splittable_leaves(intro, [frozenset({relu})])
    assert ok_alone == frozenset({relu})
    assert refused_alone == []


def test_an_unrendered_reuse_member_is_not_reported_as_refused():
    """resnet18's layer1.1 is a collapsed repeat instance: its children are not nodes at
    all, so its reused relu cannot be split and that is not a refusal. Reporting it as
    one sends someone hunting a weight-sharing problem that does not exist."""
    model = nn.Sequential(nn.ReLU())
    intro = build_tree(model, 3)

    ok, refused = splittable_leaves(intro, [frozenset({"layer1.1.relu"})])

    assert ok == frozenset()
    assert refused == []


def test_imported_resnet18_is_acyclic_and_every_node_respects_its_arity():
    """The end-to-end claim. Before this change the imported graph had 7 nodes receiving
    more inputs than their op accepts AND was cyclic, so `verify_model` refused it with
    "Cycle detected or graph is not connected"."""
    result = ingest(ModelRef(library="torchvision", arch="resnet18", depth=2))
    graph = result.graph
    by_id = {n["id"]: n for n in graph["nodes"]}

    def op_of(node):
        return str(((node.get("data") or {}).get("layerType") or {}).get("id") or "").lower()

    inbound = {}
    for edge in graph["edges"]:
        inbound[edge["target"]] = inbound.get(edge["target"], 0) + 1

    merge_ids = {"add", "mul", "cat", "concat", "sum", "stack"}
    over_arity = [
        node_id for node_id, count in inbound.items()
        if count > 1 and node_id in by_id and op_of(by_id[node_id]) not in merge_ids
        and (by_id[node_id].get("data") or {}).get("isGroup") is not True
    ]
    assert over_arity == [], f"nodes receiving more inputs than their op accepts: {over_arity}"

    # Acyclic (Kahn) — the property that made the graph unbuildable.
    indegree = {n["id"]: 0 for n in graph["nodes"]}
    adjacency = {n["id"]: [] for n in graph["nodes"]}
    for edge in graph["edges"]:
        if edge["source"] in adjacency and edge["target"] in indegree:
            adjacency[edge["source"]].append(edge["target"])
            indegree[edge["target"]] += 1
    queue = [k for k, v in indegree.items() if v == 0]
    seen = 0
    while queue:
        current = queue.pop()
        seen += 1
        for nxt in adjacency[current]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    assert seen == len(graph["nodes"]), "graph is cyclic"

    assert result.diagnostics.materialised_ops, "the residual adds must be reported"
    assert result.diagnostics.split_call_sites, "the second relu call site must be reported"


def test_functional_global_pooling_becomes_a_node():
    """MobileNetV2 calls nn.functional.adaptive_avg_pool2d instead of holding a module, so
    no node existed for it and the classifier received the unpooled map flattened to
    1x62720 instead of 1x1280."""
    plan = Plan(flatten=True)
    op = {
        "fn": "mean", "owner": "", "owner_call": 1,
        "in_shape": [1, 1280, 7, 7], "out_shape": [1, 1280, 1, 1],
    }

    assert plan.wants_pool(op) is True
    node_id = plan.pool_node(op, None, [])
    assert plan.nodes[node_id].layer_id == "adaptiveavgpool2d"
    assert plan.nodes[node_id].params == {"output_size": 1}


def test_a_mean_that_is_not_a_global_pool_is_left_alone():
    """`tensor.mean()` also lowers to aten.mean. Materialising every reduction as a pool
    would invent pooling layers a model does not have."""
    plan = Plan(flatten=True)

    # Reduces the channel axis, not the spatial one.
    assert plan.wants_pool({"fn": "mean", "in_shape": [1, 1280, 7, 7], "out_shape": [1, 1, 7, 7]}) is False
    # Already 1x1: nothing was pooled.
    assert plan.wants_pool({"fn": "mean", "in_shape": [1, 8, 1, 1], "out_shape": [1, 8, 1, 1]}) is False
    # Not rank 4.
    assert plan.wants_pool({"fn": "mean", "in_shape": [1, 8], "out_shape": [1, 1]}) is False


def test_a_merge_inside_a_childless_composite_becomes_a_node():
    """At depth 1 a DenseNet _DenseLayer is one card, so its torch.cat was attributed to
    the card itself and it collected up to 17 inbound edges (58 such nodes across
    densenet121). The concat is real structure and gets a node."""
    ops = [
        {"op": "aten.relu", "owner": "a", "fn": "relu", "owner_call": 1,
         "in_ids": [1], "out_ids": [10]},
        {"op": "aten.relu", "owner": "b", "fn": "relu", "owner_call": 1,
         "in_ids": [2], "out_ids": [11]},
        {"op": "aten.cat.default", "owner": "block", "fn": "cat", "owner_call": 1,
         "in_ids": [10, 11], "out_ids": [12], "out_shape": [1, 16]},
    ]
    rendered = {"a", "b", "block"}
    plan = Plan(taken=frozenset(rendered), composites=frozenset({"block"}))

    pairs = resolve_dataflow(ops, rendered, {}, plan)

    cats = [n for n, s in plan.nodes.items() if s.layer_id == "cat"]
    assert len(cats) == 1
    cat = cats[0]
    assert plan.nodes[cat].params.get("dim") == 1, "feature maps concatenate on channels"
    assert ("a", cat) in pairs and ("b", cat) in pairs
    assert (cat, "block") in pairs, "the composite is fed BY the merge"


def test_a_composite_that_is_its_own_producer_keeps_its_merge():
    """A BasicBlock at depth 1 adds its own output to the identity. Materialising that
    would emit layer1.0 -> add -> layer1.0 and make the graph cyclic — measured: it turned
    four clean depth-1 imports cyclic."""
    ops = [
        {"op": "aten.convolution", "owner": "stem", "fn": "convolution", "owner_call": 1,
         "in_ids": [1], "out_ids": [2]},
        # inside the block: its own internals folded to the block itself
        {"op": "aten.convolution", "owner": "block", "fn": "convolution", "owner_call": 1,
         "in_ids": [2], "out_ids": [10]},
        {"op": "aten.add_.Tensor", "owner": "block", "fn": "add", "owner_call": 1,
         "in_ids": [10, 2], "out_ids": [11], "out_shape": [1, 8]},
    ]
    rendered = {"stem", "block"}
    plan = Plan(taken=frozenset(rendered), composites=frozenset({"block"}))

    pairs = resolve_dataflow(ops, rendered, {}, plan)

    assert not [n for n, s in plan.nodes.items() if s.kind == "merge"]
    assert not [p for p in pairs if p[0] == p[1]]


def test_layout_folds_deep_pairs_up_so_the_graph_is_not_one_column():
    """Dataflow pairs mostly connect deep children. Without folding them to their
    top-level ancestor the top-level DAG is edgeless, every node lands at rank 0, and the
    model renders as a single vertical column."""
    tops = ["a", "b"]
    parent = {"a.x": "a", "b.y": "b"}
    positions, rank = layered_positions(tops, {("a.x", "b.y")}, parent)
    assert rank["a"] == 0 and rank["b"] == 1
    assert positions["a"]["x"] < positions["b"]["x"]


# ─────────────────────────────────────────────── end-to-end (Tier A) ─────────
@requires_torchvision
def test_catalog_is_populated_and_paginable():
    entries = catalog("torchvision")
    assert len(entries) > 50
    assert all(e.library == "torchvision" for e in entries)


@requires_torchvision
@pytest.mark.parametrize("arch", ["resnet18", "mobilenet_v3_small"])
def test_tier_a_ingest_produces_observed_shapes_and_no_overlap(arch):
    """The product claim is shape-annotated architecture. Two things must hold: nearly
    every node carries a shape OBSERVED from a real forward, and the emitted coordinates
    must not collide — a graph saved at the origin can never be rescued later, because
    reconcileModuleView only runs dagre when relayout is true, which is false on the
    saved-graph load path."""
    result = ingest(ModelRef(library="torchvision", arch=arch, depth=1))
    assert result.ok
    diagnostics = result.diagnostics
    assert diagnostics.tier == "fake"
    assert diagnostics.aten_ops > 0
    nodes = result.graph["nodes"]
    assert diagnostics.nodes_with_observed_shapes >= len(nodes) - 3

    boxes = [
        (n["position"]["x"], n["position"]["y"])
        for n in nodes
        if not n["data"].get("parentId")
    ]
    assert len(set(boxes)) == len(boxes), "two top-level nodes share coordinates"
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ax, ay = boxes[i]
            bx, by = boxes[j]
            overlap_x = max(0, min(ax + CARD_W, bx + CARD_W) - max(ax, bx))
            overlap_y = max(0, min(ay + CARD_H, by + CARD_H) - max(ay, by))
            assert overlap_x * overlap_y <= 0.25 * CARD_W * CARD_H


@requires_torchvision
def test_module_cards_carry_the_fields_the_canvas_actually_reads():
    """A module card takes its title from data.moduleTemplateId and its count from
    data.moduleLayerCount (canvas/ModuleShapeUtil.tsx), never from layerType.name — which
    the client recomputes from the semantic op. Omit them and every block renders as
    "module / 0 layers"."""
    result = ingest(ModelRef(library="torchvision", arch="resnet18", depth=2))
    modules = [n for n in result.graph["nodes"] if n["type"] == "moduleNode"]
    assert modules, "resnet18 at depth 2 must produce module cards"
    for node in modules:
        assert node["data"]["moduleTemplateId"]
        assert node["data"]["moduleLayerCount"] > 0


@requires_torchvision
def test_module_cards_carry_moduleInstanceId_or_expand_silently_does_nothing():
    """This is what makes a module card INTERACTIVE, and its absence fails silently.

    Both canvas/ModuleShapeUtil.tsx and components/ModuleNode.tsx guard their handlers with
    `const moduleInstanceId = data.moduleInstanceId || ''` followed by
    `if (!moduleInstanceId || !data.onToggleExpand) return`. Neither falls back to the shape
    id, so an imported graph without this field renders a perfectly normal-looking
    "Expand Block" button that does nothing when clicked, and shows the literal word
    "module" as the card subtitle. Measured before the fix: clicking expand left the card
    count at 7; after it, 7 -> 9.

    ModelBuilder's handleFlatToggleExpand matches
    `n.id === key || n.data.moduleInstanceId === key`, so the node's own id is a valid key.
    """
    result = ingest(ModelRef(library="torchvision", arch="resnet18", depth=2))
    modules = [n for n in result.graph["nodes"] if n["type"] == "moduleNode"]
    assert modules, "resnet18 at depth 2 must produce module cards"
    for node in modules:
        instance = node["data"].get("moduleInstanceId")
        assert instance, f"{node['id']} has no moduleInstanceId — expand would be inert"
        assert instance == node["id"], "the id must be a key handleFlatToggleExpand accepts"


@requires_torchvision
def test_children_are_positioned_relative_to_their_parent_not_at_the_origin():
    """A module child's position is stored RELATIVE to its module's origin
    (canvas/useTldrawBridge.ts). Emitting (0,0) stacks every child on the parent's
    top-left corner."""
    result = ingest(ModelRef(library="torchvision", arch="resnet18", depth=2))
    children = [n for n in result.graph["nodes"] if n["data"].get("parentId")]
    assert children
    assert all(n["position"]["y"] > 0 for n in children)
    by_parent: dict = {}
    for node in children:
        by_parent.setdefault(node["data"]["parentId"], []).append(node["position"]["y"])
    for parent, ys in by_parent.items():
        assert len(set(ys)) == len(ys), f"children of {parent} share a y position"


@requires_torchvision
def test_document_encoding_preserves_hierarchy_the_way_the_client_reads_it():
    """The client rebuilds node data from view.layout.node_data / node_canvas. The
    backend's migration encoder instead nests it under semantic.metadata.canvas, which the
    client never reads — so parentId is lost and children of collapsed modules leak out as
    loose top-level cards."""
    result = ingest(ModelRef(library="torchvision", arch="resnet18", depth=2))
    document = to_document(result.name, result.graph)
    layout = document.view.layout
    assert layout["node_canvas"], "node type must ride in view.layout.node_canvas"
    assert layout["node_data"], "per-node canvas data must ride in view.layout.node_data"
    assert document.semantic.nodes, "a flat ModelGraph must yield non-empty semantic nodes"

    parented = [
        nid for nid, data in layout["node_data"].items() if data.get("parentId")
    ]
    assert parented, "parentId must survive into the document"
    assert document.view.collapsed_modules, "modules must start collapsed"


# ──────────────────────────────────────────────── repo scan (Tier B) ─────────
def _write_fixture_repo(root: str) -> None:
    os.makedirs(os.path.join(root, "mymodel"), exist_ok=True)
    with open(os.path.join(root, "mymodel", "__init__.py"), "w") as handle:
        handle.write("")
    with open(os.path.join(root, "mymodel", "net.py"), "w") as handle:
        handle.write(
            textwrap.dedent(
                """
                import torch.nn as nn

                class Block(nn.Module):
                    def __init__(self, dim=8):
                        super().__init__()
                        self.fc = nn.Linear(dim, dim)
                    def forward(self, x):
                        return self.fc(x)

                class TinyNet(nn.Module):
                    def __init__(self, dim=8, depth=3):
                        super().__init__()
                        self.blocks = nn.ModuleList([Block(dim) for _ in range(depth)])
                        self.head = nn.Linear(dim, 2)
                    def forward(self, x):
                        for b in self.blocks:
                            x = b(x)
                        return self.head(x)
                """
            )
        )
    with open(os.path.join(root, "README.md"), "w") as handle:
        handle.write("# Tiny\n\nUsage:\n\n```python\nmodel = TinyNet(dim=8, depth=3)\n```\n")


def test_repo_scan_ranks_the_top_level_model_above_its_building_blocks(tmp_path):
    """Entrypoint ranking is a heuristic, so it is surfaced as a ranked list rather than
    auto-applied — but the top-level model must outrank the block it contains, otherwise
    the default pick is useless."""
    root = str(tmp_path / "repo")
    _write_fixture_repo(root)
    candidates = scan_only(root)
    names = [c.name for c in candidates]
    assert "TinyNet" in names and "Block" in names
    assert names.index("TinyNet") < names.index("Block")
    top = candidates[0]
    assert top.name == "TinyNet"
    assert top.reasons, "a ranking must explain itself"


def test_repo_ingest_end_to_end_without_being_told_the_entrypoint(tmp_path):
    """The product case: point at a repository and get a graph, with no prior knowledge of
    which class is the model, what its constructor needs, or what its forward accepts."""
    root = str(tmp_path / "repo")
    _write_fixture_repo(root)
    result = ingest(ModelRef(repo=root, depth=2))
    assert result.ok, result.error
    assert result.diagnostics.raw_modules > 0
    assert result.diagnostics.nodes_with_observed_shapes > 0
    repeats = {r.path: r.count for r in result.diagnostics.repeat_groups}
    assert repeats.get("blocks.0") == 3, "three identical blocks must collapse to x3"


# ────────────────────────── stubbed dependencies must not be silent ──────────
def _write_stubbed_backbone_repo(root: str, body: str) -> None:
    """A repo importing a package that cannot be installed here, so the stub path fires
    deterministically rather than depending on what this machine happens to have."""
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "stubnet.py"), "w") as handle:
        handle.write(textwrap.dedent(body))


def test_a_model_partly_built_from_an_uninstalled_package_is_refused_not_silently_truncated(
    tmp_path,
):
    """The worst possible ingestion outcome: HTTP 200 on a graph missing a whole backbone.

    ``ghostlib.create_model(...)`` does not raise — the stub returns another stub — and a
    stub is a ModuleType, not an nn.Module, so ``nn.Module.__setattr__`` files it in
    ``__dict__`` and ``named_modules()`` never yields it. This used to report success on a
    graph that was Input -> head -> Output, with nothing anywhere recording that a
    backbone had been dropped. Refusing names the package to install; succeeding teaches
    the user a lie about their own model.
    """
    root = str(tmp_path / "repo")
    _write_stubbed_backbone_repo(
        root,
        """
        import torch.nn as nn
        import ghostlib_solo_absent as ghostlib

        class Classifier(nn.Module):
            def __init__(self, num_classes=10):
                super().__init__()
                self.backbone = ghostlib.create_model('resnet18')
                self.head = nn.Linear(512, num_classes)
            def forward(self, x):
                return self.head(self.backbone(x))
        """,
    )
    with pytest.raises(StubbedDependencyError) as raised:
        ingest(ModelRef(repo=root, depth=2))
    message = str(raised.value)
    assert "pip install ghostlib_solo_absent" in message, (
        "a refusal without the install command is just a different dead end"
    )
    assert "backbone" in message, "name WHICH part of the model does not exist"


def test_a_repeat_ingest_is_refused_just_as_loudly(tmp_path):
    """Enough real layers survive here that the old result looked like a model: five real
    modules, one stubbed backbone, plus three ``input_node ->`` edges that exist in no
    model. Plausible enough that nobody would check.

    Ingested TWICE on purpose, because the server is long-lived. Stubs used to be left in
    ``sys.modules``, so the second import of the same repo succeeded outright and reported
    an EMPTY stubbed list — the refusal must not be built from that list alone, or every
    repo silently regains the old behaviour from its second ingest onwards.
    """
    root = str(tmp_path / "repo")
    _write_stubbed_backbone_repo(
        root,
        """
        import torch.nn as nn
        import ghostlib_repeat_absent as ghostlib

        class MixedNet(nn.Module):
            def __init__(self, num_classes=10):
                super().__init__()
                self.stem = nn.Conv2d(3, 16, kernel_size=3, padding=1)
                self.bn = nn.BatchNorm2d(16)
                self.backbone = ghostlib.create_model('resnet18')
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc1 = nn.Linear(16, 64)
                self.fc2 = nn.Linear(64, num_classes)
            def forward(self, x):
                x = self.pool(self.bn(self.stem(x))).flatten(1)
                return self.fc2(self.fc1(x))
        """,
    )
    for attempt in ("first", "second"):
        with pytest.raises(StubbedDependencyError) as raised:
            ingest(ModelRef(repo=root, entrypoint="MixedNet", depth=2))
        message = str(raised.value)
        assert "pip install ghostlib_repeat_absent" in message, f"{attempt} ingest: {message}"
        assert "backbone" in message, f"{attempt} ingest: {message}"


def test_a_stub_the_model_never_uses_still_ingests(tmp_path):
    """The whole point of stubbing: a training-only import at module scope must not block
    a model definition that never touches it (repo-wide import success on a bare torch
    environment is ~33% without this). This is what stops the refusal above from
    degenerating into "any unresolved import fails"."""
    root = str(tmp_path / "repo")
    _write_stubbed_backbone_repo(
        root,
        """
        import torch.nn as nn
        import ghostlib_unused_absent  # a training dependency, never used below

        class CleanNet(nn.Module):
            def __init__(self, dim=8):
                super().__init__()
                self.fc = nn.Linear(dim, 2)
            def forward(self, x):
                return self.fc(x)
        """,
    )
    result = ingest(ModelRef(repo=root, entrypoint="CleanNet", depth=2))
    assert result.ok, result.error
    assert result.diagnostics.stubbed_dependencies == ["ghostlib_unused_absent"]
    assert "ghostlib_unused_absent" not in sys.modules, (
        "a stub left in sys.modules makes the whole process believe the package is "
        "installed: /api/ingest/libraries reported timm:true after one such import"
    )


@pytest.mark.skipif(
    available_libraries()["timm"], reason="timm is installed here, so nothing is missing"
)
def test_a_missing_optional_extra_answers_422_with_the_install_command():
    """``500 ingestion failed: RuntimeError: timm is not installed (optional extra)``
    blames the server for a request it simply cannot serve here, and leaves the caller
    with nothing to do about it."""
    app = FastAPI()
    app.include_router(ingest_router.router)
    response = TestClient(app).post(
        "/api/ingest/model", json={"library": "timm", "arch": "resnet18", "save": False}
    )
    assert response.status_code == 422, response.text
    assert "pip install timm" in response.json()["detail"]
