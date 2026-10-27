"""Tests for services/bridge_planner.py — rank transition adapter insertion."""

from __future__ import annotations



import pytest

from models.architecture_models import ArchitectureEdge, ArchitectureGraph, ArchitectureNode
from services.bridge_planner import (
    append_edge,
    insert_adapter_chain,
    plan_adapter_transition,
    rank_from_input_expectation,
    required_rank_for_op,
    resolve_placeholder_target,
    resolve_transition_target_node,
    target_feature_for_op,
)
from services.canonical_validation import CanonicalValidationService


# ─── unit helpers ─────────────────────────────────────────────────────────────

class TestRequiredRankForOp:
    def test_conv2d_returns_4(self):
        assert required_rank_for_op("nn.Conv2d") == 4

    def test_conv1d_returns_3(self):
        assert required_rank_for_op("nn.Conv1d") == 3

    def test_linear_returns_2(self):
        assert required_rank_for_op("nn.Linear") == 2

    def test_unknown_returns_none(self):
        assert required_rank_for_op("nn.ReLU") is None

    def test_lstm_returns_3(self):
        assert required_rank_for_op("nn.LSTM") == 3


class TestTargetFeatureForOp:
    def test_linear_in_features(self):
        assert target_feature_for_op("nn.Linear", {"in_features": 512}) == 512

    def test_mha_embed_dim(self):
        assert target_feature_for_op("nn.MultiheadAttention", {"embed_dim": 768}) == 768

    def test_normalized_shape_list(self):
        assert target_feature_for_op("nn.LayerNorm", {"normalized_shape": [256]}) == 256

    def test_fallback(self):
        assert target_feature_for_op("nn.ReLU", {}, fallback=99) == 99


class TestRankFromInputExpectation:
    def test_rank4(self):
        assert rank_from_input_expectation("requires rank-4 NCHW") == 4

    def test_rank2(self):
        assert rank_from_input_expectation("requires rank-2") == 2

    def test_missing(self):
        assert rank_from_input_expectation("no rank info") is None

    def test_empty(self):
        assert rank_from_input_expectation("") is None


class TestAppendEdge:
    def test_creates_correct_edge(self):
        edges: list[ArchitectureEdge] = []
        append_edge(edges, "a", "b")
        assert len(edges) == 1
        e = edges[0]
        assert e.source_node_id == "a"
        assert e.target_node_id == "b"
        assert e.edge_id == "a:y->b:x"


# ─── insert_adapter_chain integration ─────────────────────────────────────────

def _linear_node(node_id: str, in_f: int, out_f: int) -> ArchitectureNode:
    return ArchitectureNode(
        node_id=node_id, kind="layer", op="nn.Linear",
        params={"in_features": in_f, "out_features": out_f},
        in_ports=["x"], out_ports=["y"],
    )


def _make_validator() -> CanonicalValidationService:
    return CanonicalValidationService()


class TestInsertAdapterChain:
    def test_rank2_to_rank4_inserts_linear_unflatten(self):
        """rank-2 source → Conv2d target: inserts linear + unflatten."""
        nodes: list[ArchitectureNode] = [
            ArchitectureNode(node_id="input", kind="input", op="Input",
                             params={"shape": [1, 256]}, in_ports=[], out_ports=["y"]),
            _linear_node("lin", 256, 256),
        ]
        edges: list[ArchitectureEdge] = []
        append_edge(edges, "input", "lin")

        target_node = ArchitectureNode(
            node_id="conv", kind="layer", op="nn.Conv2d",
            params={"in_channels": 64, "out_channels": 128, "kernel_size": 3},
            in_ports=["x"], out_ports=["y"],
        )
        new_source, ins_nodes, ins_edges, issue = insert_adapter_chain(
            source_shape=[1, 256],
            nodes=nodes,
            edges=edges,
            source_node_id="lin",
            target_op=target_node.op,
            target_params=dict(target_node.params),
            adapter_prefix="bridge1",
        )
        assert issue is None
        assert new_source != "lin"
        assert len(ins_nodes) >= 2  # at least linear + unflatten
        ops = [n.op for n in ins_nodes]
        assert "nn.Linear" in ops
        assert "nn.Unflatten" in ops
        # The final adapter node should produce [1, C, H, W] shaped output
        last_node = ins_nodes[-1]
        assert last_node.op == "nn.Unflatten"

    def test_rank4_to_rank2_inserts_flatten(self):
        """rank-4 source → Linear target: inserts Flatten."""
        nodes: list[ArchitectureNode] = [
            ArchitectureNode(node_id="input", kind="input", op="Input",
                             params={"shape": [1, 64, 8, 8]}, in_ports=[], out_ports=["y"]),
        ]
        edges: list[ArchitectureEdge] = []
        new_source, ins_nodes, ins_edges, issue = insert_adapter_chain(
            source_shape=[1, 64, 8, 8],
            nodes=nodes,
            edges=edges,
            source_node_id="input",
            target_op="nn.Linear",
            target_params={"in_features": 4096},
            adapter_prefix="bridge2",
        )
        assert issue is None
        assert new_source != "input"
        ops = [n.op for n in ins_nodes]
        assert "nn.Flatten" in ops

    def test_rank2_to_rank3_inserts_unflatten(self):
        """rank-2 source → LSTM target: inserts linear + unflatten."""
        nodes: list[ArchitectureNode] = [
            ArchitectureNode(node_id="input", kind="input", op="Input",
                             params={"shape": [1, 512]}, in_ports=[], out_ports=["y"]),
        ]
        edges: list[ArchitectureEdge] = []
        new_source, ins_nodes, ins_edges, issue = insert_adapter_chain(
            source_shape=[1, 512],
            nodes=nodes,
            edges=edges,
            source_node_id="input",
            target_op="nn.LSTM",
            target_params={"input_size": 256, "hidden_size": 128},
            adapter_prefix="bridge3",
        )
        assert issue is None
        assert new_source != "input"
        assert len(ins_nodes) >= 2
        ops = [n.op for n in ins_nodes]
        assert "nn.Unflatten" in ops

    def test_rank4_to_rank4_channel_align(self):
        """rank-4 → rank-4 with mismatched channels: inserts 1×1 conv."""
        nodes: list[ArchitectureNode] = [
            ArchitectureNode(node_id="input", kind="input", op="Input",
                             params={"shape": [1, 64, 16, 16]}, in_ports=[], out_ports=["y"]),
        ]
        edges: list[ArchitectureEdge] = []
        new_source, ins_nodes, ins_edges, issue = insert_adapter_chain(
            source_shape=[1, 64, 16, 16],
            nodes=nodes,
            edges=edges,
            source_node_id="input",
            target_op="nn.Conv2d",
            target_params={"in_channels": 128, "out_channels": 256, "kernel_size": 3},
            adapter_prefix="bridge4",
        )
        assert issue is None
        ops = [n.op for n in ins_nodes]
        assert "nn.Conv2d" in ops  # 1×1 channel aligner

    def test_same_rank_same_feature_no_adapters(self):
        """rank-2 source → Linear with matching in_features: no adapters inserted."""
        nodes: list[ArchitectureNode] = [
            ArchitectureNode(node_id="input", kind="input", op="Input",
                             params={"shape": [1, 256]}, in_ports=[], out_ports=["y"]),
        ]
        edges: list[ArchitectureEdge] = []
        new_source, ins_nodes, ins_edges, issue = insert_adapter_chain(
            source_shape=[1, 256],
            nodes=nodes,
            edges=edges,
            source_node_id="input",
            target_op="nn.Linear",
            target_params={"in_features": 256, "out_features": 128},
            adapter_prefix="bridge5",
        )
        assert issue is None
        assert new_source == "input"
        assert ins_nodes == []
        assert ins_edges == []

    def test_unknown_op_no_adapters(self):
        """No required rank for ReLU: nothing inserted."""
        nodes: list[ArchitectureNode] = [
            ArchitectureNode(node_id="input", kind="input", op="Input",
                             params={"shape": [1, 256]}, in_ports=[], out_ports=["y"]),
        ]
        edges: list[ArchitectureEdge] = []
        new_source, ins_nodes, ins_edges, issue = insert_adapter_chain(
            source_shape=[1, 256],
            nodes=nodes,
            edges=edges,
            source_node_id="input",
            target_op="nn.ReLU",
            target_params={},
            adapter_prefix="bridge6",
        )
        assert ins_nodes == []


class TestInsertAdapterChainValidates:
    """Adapter-inserted graphs must pass validation end-to-end."""

    def _build_and_validate(self, source_shape, target_op, target_params, target_id="out"):
        validator = _make_validator()
        nodes = [
            ArchitectureNode(node_id="input", kind="input", op="Input",
                             params={"shape": source_shape}, in_ports=[], out_ports=["y"]),
        ]
        edges: list[ArchitectureEdge] = []

        new_source, ins_nodes, ins_edges, issue = insert_adapter_chain(
            source_shape=source_shape,
            nodes=nodes,
            edges=edges,
            source_node_id="input",
            target_op=target_op,
            target_params=target_params,
            adapter_prefix="br",
        )
        # Add target node
        target_node = ArchitectureNode(
            node_id=target_id, kind="layer", op=target_op,
            params=target_params, in_ports=["x"], out_ports=["y"],
        )
        nodes.append(target_node)
        append_edge(edges, new_source, target_id)
        # Add output
        nodes.append(ArchitectureNode(node_id="output", kind="output", op="Output",
                                      params={}, in_ports=["x"], out_ports=[]))
        append_edge(edges, target_id, "output")

        graph = ArchitectureGraph(
            graph_id="test_graph",
            name="test",
            revision=0,
            inputs=[{"name": "x", "shape": source_shape}],
            nodes=nodes,
            edges=edges,
            metadata={},
        )
        result = validator.validate(graph=graph, strict=False)
        return result

    def test_rank2_to_conv2d_validates(self):
        result = self._build_and_validate(
            source_shape=[1, 256],
            target_op="nn.Conv2d",
            target_params={"in_channels": 64, "out_channels": 128, "kernel_size": 3},
        )
        node_result = result.node_results.get("out")
        assert node_result is not None
        assert node_result.status != "error", f"Expected valid, got: {node_result.message}"

    def test_rank4_to_linear_validates(self):
        result = self._build_and_validate(
            source_shape=[1, 32, 8, 8],
            target_op="nn.Linear",
            target_params={"in_features": 2048, "out_features": 128},
        )
        node_result = result.node_results.get("out")
        assert node_result is not None
        assert node_result.status != "error", f"Expected valid, got: {node_result.message}"

    def test_rank2_to_lstm_validates(self):
        result = self._build_and_validate(
            source_shape=[1, 512],
            target_op="nn.LSTM",
            target_params={"input_size": 256, "hidden_size": 128, "num_layers": 1},
        )
        node_result = result.node_results.get("out")
        assert node_result is not None
        assert node_result.status != "error", f"Expected valid, got: {node_result.message}"


class TestResolveTransitionTargetNode:
    def _make_node(self, node_id: str, op: str, kind: str = "layer", params=None) -> ArchitectureNode:
        return ArchitectureNode(
            node_id=node_id, kind=kind, op=op,
            params=params or {}, in_ports=["x"], out_ports=["y"],
        )

    def _make_edge(self, src: str, tgt: str) -> ArchitectureEdge:
        return ArchitectureEdge(
            edge_id=f"{src}:y->{tgt}:x",
            source_node_id=src, source_port="y",
            target_node_id=tgt, target_port="x",
        )

    def test_skips_dropout_finds_linear(self):
        nodes = [
            self._make_node("input", "Input", kind="input"),
            self._make_node("drop", "nn.Dropout"),
            self._make_node("lin", "nn.Linear", params={"in_features": 256, "out_features": 128}),
        ]
        edges = [self._make_edge("input", "drop"), self._make_edge("drop", "lin")]
        result = resolve_transition_target_node(
            template_nodes=nodes, template_edges=edges, entry_node_id="input"
        )
        assert result is not None
        assert result.node_id == "lin"

    def test_returns_direct_feature_sensitive_node(self):
        nodes = [
            self._make_node("conv", "nn.Conv2d", params={"in_channels": 64, "out_channels": 128, "kernel_size": 3}),
        ]
        edges = []
        result = resolve_transition_target_node(
            template_nodes=nodes, template_edges=edges, entry_node_id="conv"
        )
        assert result is not None
        assert result.node_id == "conv"

    def test_missing_entry_returns_none(self):
        result = resolve_transition_target_node(
            template_nodes=[], template_edges=[], entry_node_id="nonexistent"
        )
        assert result is None


class TestPlanAdapterTransition:
    """plan_adapter_transition = probe source shape → (optionally) walk to the
    real feature-sensitive target → pick rank → insert_adapter_chain. This is the
    unified path both the sweep and the product runtime now call."""

    def test_probes_source_and_inserts_chain_for_direct_target(self):
        # rank-2 source (Linear out [1,256]) → Conv2d target (needs rank-4).
        validator = _make_validator()
        nodes: list[ArchitectureNode] = [
            ArchitectureNode(node_id="input", kind="input", op="Input",
                             params={"shape": [1, 256]}, in_ports=[], out_ports=["y"]),
            _linear_node("lin", 256, 256),
        ]
        edges: list[ArchitectureEdge] = []
        append_edge(edges, "input", "lin")
        target = ArchitectureNode(
            node_id="conv", kind="layer", op="nn.Conv2d",
            params={"in_channels": 64, "out_channels": 128, "kernel_size": 3},
            in_ports=["x"], out_ports=["y"],
        )
        new_source, ins_nodes, ins_edges, issue = plan_adapter_transition(
            validator=validator, graph_id="g", input_shape=[1, 256],
            nodes=nodes, edges=edges, source_node_id="lin",
            target_node=target, adapter_prefix="b",
        )
        # It probed the source itself (no source_shape arg) and bridged rank-2→4.
        assert issue is None
        assert new_source != "lin"
        assert "nn.Unflatten" in [n.op for n in ins_nodes]

    def test_walks_passthrough_to_the_feature_sensitive_target(self):
        # Template entry is a passthrough (Dropout); the real target is the Linear.
        # Passing template internals must resolve to the Linear, so a rank-4 source
        # gets a Flatten toward it rather than being pinned to the passthrough op.
        validator = _make_validator()
        nodes: list[ArchitectureNode] = [
            ArchitectureNode(node_id="input", kind="input", op="Input",
                             params={"shape": [1, 3, 8, 8]}, in_ports=[], out_ports=["y"]),
        ]
        edges: list[ArchitectureEdge] = []
        passthrough_entry = ArchitectureNode(
            node_id="drop", kind="layer", op="nn.Dropout", params={"p": 0.1},
            in_ports=["x"], out_ports=["y"],
        )
        template_nodes = [
            passthrough_entry,
            _linear_node("head", 192, 10),
        ]
        template_edges: list[ArchitectureEdge] = []
        append_edge(template_edges, "drop", "head")

        resolved = resolve_transition_target_node(
            template_nodes=template_nodes, template_edges=template_edges, entry_node_id="drop",
        )
        assert resolved is not None and resolved.node_id == "head"  # walked past Dropout

        nodes.extend(template_nodes)
        edges.extend(template_edges)
        append_edge(edges, "input", "drop")
        new_source, ins_nodes, ins_edges, issue = plan_adapter_transition(
            validator=validator, graph_id="g", input_shape=[1, 3, 8, 8],
            nodes=nodes, edges=edges, source_node_id="input",
            target_node=passthrough_entry, adapter_prefix="b",
            template_nodes=template_nodes, template_edges=template_edges, entry_node_id="drop",
        )
        # rank-4 source → rank-2 Linear target ⇒ a Flatten is inserted.
        assert issue is None
        assert "nn.Flatten" in [n.op for n in ins_nodes]


class TestResolvePlaceholderTarget:
    """Conservative product walk policy: step past PURE placeholders only."""

    def _nodes(self, *specs):
        # specs: (node_id, kind, op)
        out = []
        for node_id, kind, op in specs:
            out.append(ArchitectureNode(node_id=node_id, kind=kind, op=op, params={},
                                        in_ports=["x"], out_ports=["y"]))
        return out

    def test_walks_past_input_and_identity_to_real_layer(self):
        nodes = self._nodes(
            ("inp", "input", "Input"),
            ("id", "layer", "nn.Identity"),
            ("lin", "layer", "nn.Linear"),
        )
        edges: list[ArchitectureEdge] = []
        append_edge(edges, "inp", "id")
        append_edge(edges, "id", "lin")
        result = resolve_placeholder_target(nodes=nodes, edges=edges, entry_node_id="inp")
        assert result is not None and result.node_id == "lin"  # walked Input + Identity

    def test_stops_at_dropout_does_not_walk_past_it(self):
        # The whole point of option 3: Dropout is a real chosen node — respect it.
        nodes = self._nodes(
            ("inp", "input", "Input"),
            ("drop", "layer", "nn.Dropout"),
            ("lin", "layer", "nn.Linear"),
        )
        edges: list[ArchitectureEdge] = []
        append_edge(edges, "inp", "drop")
        append_edge(edges, "drop", "lin")
        result = resolve_placeholder_target(nodes=nodes, edges=edges, entry_node_id="inp")
        assert result is not None and result.node_id == "drop"  # stopped at Dropout, not lin

    def test_real_target_is_returned_unchanged(self):
        nodes = self._nodes(("lin", "layer", "nn.Linear"), ("relu", "layer", "nn.ReLU"))
        edges: list[ArchitectureEdge] = []
        append_edge(edges, "lin", "relu")
        result = resolve_placeholder_target(nodes=nodes, edges=edges, entry_node_id="lin")
        assert result is not None and result.node_id == "lin"  # already real → no walk

    def test_stops_at_a_branch_after_placeholder(self):
        nodes = self._nodes(
            ("inp", "input", "Input"),
            ("a", "layer", "nn.Linear"),
            ("b", "layer", "nn.Linear"),
        )
        edges: list[ArchitectureEdge] = []
        append_edge(edges, "inp", "a")
        append_edge(edges, "inp", "b")  # branch: two successors
        result = resolve_placeholder_target(nodes=nodes, edges=edges, entry_node_id="inp")
        assert result is not None and result.node_id == "inp"  # ambiguous → stop
