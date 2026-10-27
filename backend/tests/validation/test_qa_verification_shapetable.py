"""Regression tests for PRs #56 / #59 — "Verification service & Shape Table truth".

The Shape Table and the per-node badges on the canvas read
``verify_model()['shapes']``. After PR #56/#59, a *successful* dry forward pass
OVERRIDES the crude pre-forward analytic estimates (which mishandle the batch
dim — e.g. ``[C, H, W]``) with the REAL tensor shapes that flowed through torch
(full ``[N, C, H, W]``). These tests pin that "shapes are ground truth"
contract, its boundary (the override is forward-only and produces a summary),
the per-node input seeding that feeds it, and a hand-computed end-to-end case.

Existing coverage lives in ``test_verification_service.py`` and is NOT
duplicated here: that file already covers ``_build_synthetic_input`` dtype,
dtype-error-free MLP/embedding forwards, arity enforcement, and the *single*
``Conv2d`` batch-dim case. This file targets the gaps — multi-layer chains, MLP
per-node shapes, the forward-only boundary, per-node input seeding, a
hand-computed CNN classifier, and the forward-FAILURE fallback bug.
"""

from __future__ import annotations



from services.verification_service import verify_model  # noqa: E402


def _node(node_id, node_type, lt_id, lt_name, params=None):
    """Build a flat canvas node in the shape the verify path consumes."""
    return {
        "id": node_id,
        "type": node_type,
        "position": {"x": 0.0, "y": 0.0},
        "data": {
            "layerType": {"id": lt_id, "name": lt_name, "parameters": []},
            "parameters": params or {},
        },
    }


class TestGroundTruthShapeOverride:
    """A successful forward replaces analytic estimates with the real per-node
    tensor shapes the Shape Table displays (full rank, batch dim kept)."""

    def _conv_pool_relu(self):
        return {
            "nodes": [
                _node("input", "inputNode", "input", "Input", {"shape": [1, 3, 32, 32]}),
                _node("conv", "layerNode", "conv2d", "nn.Conv2d",
                      {"in_channels": 3, "out_channels": 8, "kernel_size": 3, "padding": 1}),
                _node("pool", "layerNode", "maxpool2d", "nn.MaxPool2d", {"kernel_size": 2}),
                _node("relu", "layerNode", "relu", "nn.ReLU", {}),
                _node("output", "outputNode", "output", "Output", {}),
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "conv"},
                {"id": "e2", "source": "conv", "target": "pool"},
                {"id": "e3", "source": "pool", "target": "relu"},
                {"id": "e4", "source": "relu", "target": "output"},
            ],
        }

    def test_multilayer_chain_shapes_are_full_rank_ground_truth(self):
        """Every node's shape is the REAL forward tensor, hand-verified across a
        chain that both preserves (ReLU) and shrinks (MaxPool) spatial dims."""
        result = verify_model(self._conv_pool_relu(), input_shape=[1, 3, 32, 32], run_forward=True)
        assert result["ok"] is True, result["errors"]
        shapes = result["shapes"]
        assert shapes["input"] == [1, 3, 32, 32]
        assert shapes["conv"] == [1, 8, 32, 32]   # Conv2d(3->8, k3, p1) keeps 32x32
        assert shapes["pool"] == [1, 8, 16, 16]   # MaxPool2d(k2) halves H and W
        assert shapes["relu"] == [1, 8, 16, 16]   # activation is shape-preserving
        assert shapes["output"] == [1, 8, 16, 16]
        # output_shape agrees with the tensor sitting at the output node.
        assert result["output_shape"] == shapes["output"]
        # No crude 3-element ([C,H,W]) estimate leaked through the override.
        assert all(len(shapes[n]) == 4 for n in ("conv", "pool", "relu"))


class TestRank2MlpShapeTruth:
    """Rank-2 MLPs: the Shape Table must show [N, features] per node. Existing
    coverage only asserts the forward is dtype-error-free; the *shapes* are the
    gap this pins."""

    def test_mlp_per_node_shapes_keep_batch_dim(self):
        payload = {
            "nodes": [
                _node("input", "inputNode", "input", "Input", {"shape": [4, 256]}),
                _node("lin1", "layerNode", "linear", "nn.Linear",
                      {"in_features": 256, "out_features": 128}),
                _node("lin2", "layerNode", "linear", "nn.Linear",
                      {"in_features": 128, "out_features": 64}),
                _node("output", "outputNode", "output", "Output", {}),
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "lin1"},
                {"id": "e2", "source": "lin1", "target": "lin2"},
                {"id": "e3", "source": "lin2", "target": "output"},
            ],
        }
        result = verify_model(payload, input_shape=[4, 256], run_forward=True)
        assert result["ok"] is True, result["errors"]
        assert result["shapes"]["input"] == [4, 256]
        assert result["shapes"]["lin1"] == [4, 128]
        assert result["shapes"]["lin2"] == [4, 64]
        assert result["output_shape"] == [4, 64]


class TestForwardOnlyOverride:
    """The ground-truth override AND the param summary are produced only when a
    forward actually runs. ``run_forward=False`` stops at shape inference."""

    def _single_conv(self):
        return {
            "nodes": [
                _node("input", "inputNode", "input", "Input", {"shape": [1, 3, 8, 8]}),
                _node("conv", "layerNode", "conv2d", "nn.Conv2d",
                      {"in_channels": 3, "out_channels": 4, "kernel_size": 3, "padding": 1}),
                _node("output", "outputNode", "output", "Output", {}),
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "conv"},
                {"id": "e2", "source": "conv", "target": "output"},
            ],
        }

    def test_forward_produces_ground_truth_and_summary(self):
        result = verify_model(self._single_conv(), input_shape=[1, 3, 8, 8], run_forward=True)
        assert result["ok"] is True, result["errors"]
        assert result["stage"] == "complete"
        assert result["shapes"]["conv"] == [1, 4, 8, 8]   # full-rank ground truth
        # A real forward computes the parameter/MAC summary.
        assert result["summary"] is not None
        assert result["summary"]["layer_count"] == 1

    def test_no_forward_completes_at_shape_inference_without_summary(self):
        """Skipping the forward means no ground-truth override and no param
        summary — the run stops at the shape_inference stage but still reports
        every node so the Shape Table has something to display."""
        result = verify_model(self._single_conv(), input_shape=[1, 3, 8, 8], run_forward=False)
        assert result["ok"] is True
        assert result["stage"] == "complete"
        # No forward ran → the param/MAC summary is never computed.
        assert result["summary"] is None
        assert set(result["shapes"]) >= {"input", "conv", "output"}


class TestInputNodeSeeding:
    """The forward seeds each input node from ITS OWN data.parameters.shape, so
    the Shape Table reflects the dims the user set on the node — while the
    request-level ``input_shape`` field echoes the coerced argument."""

    def test_input_node_seeded_from_its_own_shape_param(self):
        payload = {
            "nodes": [
                _node("input", "inputNode", "input", "Input", {"shape": [2, 3, 8, 8]}),
                _node("conv", "layerNode", "conv2d", "nn.Conv2d",
                      {"in_channels": 3, "out_channels": 4, "kernel_size": 3, "padding": 1}),
                _node("output", "outputNode", "output", "Output", {}),
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "conv"},
                {"id": "e2", "source": "conv", "target": "output"},
            ],
        }
        # Node declares batch=2; the request arg deliberately says something else.
        result = verify_model(payload, input_shape=[1, 3, 224, 224], run_forward=True)
        assert result["ok"] is True, result["errors"]
        # Ground-truth shapes follow the node's own shape (batch=2), not the arg.
        assert result["shapes"]["input"] == [2, 3, 8, 8]
        assert result["shapes"]["conv"] == [2, 4, 8, 8]
        # The request-level input_shape field still echoes the coerced argument.
        assert result["input_shape"] == [1, 3, 224, 224]


class TestHandComputedCnnClassifier:
    """End-to-end hand-computed classifier exercised through the whole verify
    pipeline (build → shape inference → forward → summary)."""

    def test_cnn_classifier_shapes_and_params_hand_computed(self):
        """[1,3,8,8] -> Conv2d(3->4,k3,p1) -> MaxPool2d(2) -> Flatten -> Linear(64->10).
        Every per-node badge and the parameter summary must match by-hand math."""
        payload = {
            "nodes": [
                _node("input", "inputNode", "input", "Input", {"shape": [1, 3, 8, 8]}),
                _node("conv", "layerNode", "conv2d", "nn.Conv2d",
                      {"in_channels": 3, "out_channels": 4, "kernel_size": 3, "padding": 1}),
                _node("pool", "layerNode", "maxpool2d", "nn.MaxPool2d", {"kernel_size": 2}),
                _node("flat", "layerNode", "flatten", "nn.Flatten", {}),
                _node("fc", "layerNode", "linear", "nn.Linear",
                      {"in_features": 64, "out_features": 10}),
                _node("output", "outputNode", "output", "Output", {}),
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "conv"},
                {"id": "e2", "source": "conv", "target": "pool"},
                {"id": "e3", "source": "pool", "target": "flat"},
                {"id": "e4", "source": "flat", "target": "fc"},
                {"id": "e5", "source": "fc", "target": "output"},
            ],
        }
        result = verify_model(payload, input_shape=[1, 3, 8, 8], run_forward=True)
        assert result["ok"] is True, result["errors"]
        shapes = result["shapes"]
        assert shapes["input"] == [1, 3, 8, 8]
        assert shapes["conv"] == [1, 4, 8, 8]    # padding 1 preserves 8x8
        assert shapes["pool"] == [1, 4, 4, 4]    # MaxPool2d(2) -> 4x4
        assert shapes["flat"] == [1, 64]         # 4 * 4 * 4 = 64
        assert shapes["fc"] == [1, 10]
        assert shapes["output"] == [1, 10]
        assert result["output_shape"] == [1, 10]
        # Params: conv = 4*(3*3*3) + 4 = 112 ; linear = 64*10 + 10 = 650 ; total 762.
        assert result["summary"]["total_parameters"] == 762
        # Four built layers (conv, pool, flatten, linear); input/output aren't layers.
        assert result["summary"]["layer_count"] == 4


class TestForwardFailureShapeTruth:
    """A graph that fails mid-forward. The failure must be localized to the
    offending node; the shapes it reports are the subject of a known bug."""

    def _failing_channel_mismatch(self):
        """conv1 forwards fine to [1,8,32,32]; conv2 declares in_channels=3 but is
        actually handed 8 channels, so torch raises at conv2 during the forward."""
        return {
            "nodes": [
                _node("input", "inputNode", "input", "Input", {"shape": [1, 3, 32, 32]}),
                _node("conv1", "layerNode", "conv2d", "nn.Conv2d",
                      {"in_channels": 3, "out_channels": 8, "kernel_size": 3, "padding": 1}),
                _node("conv2", "layerNode", "conv2d", "nn.Conv2d",
                      {"in_channels": 3, "out_channels": 16, "kernel_size": 3, "padding": 1}),
                _node("output", "outputNode", "output", "Output", {}),
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "conv1"},
                {"id": "e2", "source": "conv1", "target": "conv2"},
                {"id": "e3", "source": "conv2", "target": "output"},
            ],
        }

    def test_forward_failure_reports_stage_and_offending_node(self):
        """The failure path localizes the break: ok False, stage 'forward', and an
        error naming conv2 with torch's channel-mismatch message. This is what
        lets the UI light up the layer that broke."""
        result = verify_model(self._failing_channel_mismatch(), input_shape=[1, 3, 32, 32], run_forward=True)
        assert result["ok"] is False
        assert result["stage"] == "forward"
        assert result["errors"], "expected a forward error to be reported"
        err = result["errors"][-1]
        assert err.get("node_id") == "conv2"
        assert "channel" in err.get("message", "").lower()

    def test_failing_forward_keeps_ground_truth_for_succeeded_nodes(self):
        """conv1 forwarded to the real [1,8,32,32] (torch's own error even prints
        'expected input[1, 8, 32, 32]'), so the Shape Table for conv1 should show
        that ground truth — never the crude [8,3,32] estimate the override exists
        to replace."""
        result = verify_model(self._failing_channel_mismatch(), input_shape=[1, 3, 32, 32], run_forward=True)
        assert result["ok"] is False  # sanity: this really is a failing forward
        assert result["shapes"]["conv1"] == [1, 8, 32, 32]
