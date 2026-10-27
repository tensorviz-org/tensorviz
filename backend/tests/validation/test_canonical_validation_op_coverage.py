from __future__ import annotations

"""Regression tests for newly-added PyTorch op shape rules in
CanonicalValidationService. Each test exercises a single-layer graph
(input -> op -> output) and asserts the inferred output shape."""



from models.architecture_models import (  # noqa: E402
    ArchitectureEdge,
    ArchitectureGraph,
    ArchitectureNode,
)
from services.canonical_validation import CanonicalValidationService  # noqa: E402


def _single_layer_graph(op: str, params: dict, input_shape: list[int]) -> ArchitectureGraph:
    return ArchitectureGraph(
        graph_id="g1",
        name="single_layer",
        revision=1,
        inputs=[{"name": "x", "shape": input_shape}],
        nodes=[
            ArchitectureNode(node_id="input", kind="input", op="Input", params={"shape": input_shape}, in_ports=[], out_ports=["y"]),
            ArchitectureNode(node_id="layer", kind="layer", op=op, params=params),
            ArchitectureNode(node_id="output", kind="output", op="Output", params={}, in_ports=["x"], out_ports=[]),
        ],
        edges=[
            ArchitectureEdge(edge_id="e1", source_node_id="input", source_port="y", target_node_id="layer", target_port="x"),
            ArchitectureEdge(edge_id="e2", source_node_id="layer", source_port="y", target_node_id="output", target_port="x"),
        ],
    )


def _infer(op: str, params: dict, input_shape: list[int]) -> list[int] | None:
    result = CanonicalValidationService().validate(_single_layer_graph(op, params, input_shape))
    layer = result.node_results.get("layer")
    return None if layer is None else layer.output_shape


# ── Conv1d / Conv3d ─────────────────────────────────────────────────────────

def test_conv1d_basic():
    assert _infer("nn.Conv1d", {"in_channels": 4, "out_channels": 8, "kernel_size": 3}, [1, 4, 16]) == [1, 8, 14]


def test_conv1d_with_stride_and_padding():
    assert _infer("nn.Conv1d", {"in_channels": 4, "out_channels": 8, "kernel_size": 3, "stride": 2, "padding": 1}, [1, 4, 16]) == [1, 8, 8]


def test_conv3d_basic():
    assert _infer("nn.Conv3d", {"in_channels": 3, "out_channels": 16, "kernel_size": 3}, [1, 3, 8, 16, 16]) == [1, 16, 6, 14, 14]


def test_conv1d_rejects_wrong_in_channels():
    assert _infer("nn.Conv1d", {"in_channels": 8, "out_channels": 16, "kernel_size": 3}, [1, 4, 16]) is None


# ── ConvTranspose 1d/2d/3d ──────────────────────────────────────────────────

def test_conv_transpose2d_doubles_spatial():
    # ConvTranspose2d with stride 2, kernel 4, padding 1 doubles H,W
    assert _infer("nn.ConvTranspose2d", {"in_channels": 8, "out_channels": 4, "kernel_size": 4, "stride": 2, "padding": 1}, [1, 8, 16, 16]) == [1, 4, 32, 32]


def test_conv_transpose1d_basic():
    assert _infer("nn.ConvTranspose1d", {"in_channels": 4, "out_channels": 8, "kernel_size": 3, "stride": 2}, [1, 4, 8]) == [1, 8, 17]


def test_conv_transpose3d_basic():
    assert _infer("nn.ConvTranspose3d", {"in_channels": 4, "out_channels": 4, "kernel_size": 2, "stride": 2}, [1, 4, 4, 8, 8]) == [1, 4, 8, 16, 16]


# ── Pool 1d/3d ──────────────────────────────────────────────────────────────

def test_max_pool1d():
    assert _infer("nn.MaxPool1d", {"kernel_size": 2}, [1, 4, 16]) == [1, 4, 8]


def test_avg_pool3d():
    assert _infer("nn.AvgPool3d", {"kernel_size": 2}, [1, 4, 8, 8, 8]) == [1, 4, 4, 4, 4]


# ── Adaptive pools ──────────────────────────────────────────────────────────

def test_adaptive_avg_pool1d():
    assert _infer("nn.AdaptiveAvgPool1d", {"output_size": 4}, [1, 8, 32]) == [1, 8, 4]


def test_adaptive_max_pool2d():
    assert _infer("nn.AdaptiveMaxPool2d", {"output_size": [4, 4]}, [1, 8, 32, 32]) == [1, 8, 4, 4]


def test_adaptive_avg_pool3d():
    assert _infer("nn.AdaptiveAvgPool3d", {"output_size": [2, 2, 2]}, [1, 4, 8, 8, 8]) == [1, 4, 2, 2, 2]


# ── Normalization layers ────────────────────────────────────────────────────

def test_batchnorm1d():
    assert _infer("nn.BatchNorm1d", {"num_features": 8}, [1, 8, 32]) == [1, 8, 32]


def test_batchnorm3d():
    assert _infer("nn.BatchNorm3d", {"num_features": 4}, [1, 4, 8, 8, 8]) == [1, 4, 8, 8, 8]


def test_instance_norm2d():
    assert _infer("nn.InstanceNorm2d", {"num_features": 16}, [1, 16, 32, 32]) == [1, 16, 32, 32]


def test_group_norm_valid():
    assert _infer("nn.GroupNorm", {"num_groups": 4, "num_channels": 16}, [1, 16, 32, 32]) == [1, 16, 32, 32]


def test_group_norm_rejects_non_dividing_groups():
    # num_channels not divisible by num_groups
    assert _infer("nn.GroupNorm", {"num_groups": 3, "num_channels": 16}, [1, 16, 32, 32]) is None


def test_rms_norm_with_int_normalized_shape():
    assert _infer("nn.RMSNorm", {"normalized_shape": 16}, [1, 32, 16]) == [1, 32, 16]


def test_batchnorm_rejects_channel_mismatch():
    assert _infer("nn.BatchNorm1d", {"num_features": 4}, [1, 8, 16]) is None


# ── Activations (shape-preserving) ──────────────────────────────────────────

def test_silu_preserves_shape():
    assert _infer("nn.SiLU", {}, [1, 16, 32, 32]) == [1, 16, 32, 32]


def test_mish_preserves_shape():
    assert _infer("nn.Mish", {}, [1, 16, 32, 32]) == [1, 16, 32, 32]


def test_elu_preserves_shape():
    assert _infer("nn.ELU", {}, [1, 16, 32, 32]) == [1, 16, 32, 32]


def test_leaky_relu_preserves_shape():
    assert _infer("nn.LeakyReLU", {"negative_slope": 0.01}, [1, 16, 32, 32]) == [1, 16, 32, 32]


def test_softmax_preserves_shape():
    assert _infer("nn.Softmax", {"dim": -1}, [1, 10]) == [1, 10]


def test_log_softmax_preserves_shape():
    assert _infer("nn.LogSoftmax", {"dim": -1}, [1, 10]) == [1, 10]


# ── Upsample / PixelShuffle ─────────────────────────────────────────────────

def test_upsample_with_scale_factor():
    assert _infer("nn.Upsample", {"scale_factor": 2}, [1, 8, 16, 16]) == [1, 8, 32, 32]


def test_upsample_with_explicit_size():
    assert _infer("nn.Upsample", {"size": [64, 64]}, [1, 8, 16, 16]) == [1, 8, 64, 64]


def test_pixel_shuffle_doubles_spatial():
    # PixelShuffle r=2: [N, C*4, H, W] -> [N, C, H*2, W*2]
    assert _infer("nn.PixelShuffle", {"upscale_factor": 2}, [1, 16, 8, 8]) == [1, 4, 16, 16]


def test_pixel_shuffle_rejects_non_divisible_channels():
    # 17 channels is not divisible by 2*2 = 4
    assert _infer("nn.PixelShuffle", {"upscale_factor": 2}, [1, 17, 8, 8]) is None


# ── Embedding: appends embedding_dim as a new trailing dimension ─────────────
# Regression for the bug where an Embedding node showed output == input
# (e.g. embedding_dim=768 but Out=(1,1024)). Root cause: the op name arrived
# as lowercase "embedding", missed the exact-cased rule, and fell through to a
# silent input-shape passthrough. Every op-name variant must now resolve.

def test_embedding_appends_embedding_dim():
    assert _infer("nn.Embedding", {"num_embeddings": 50257, "embedding_dim": 768}, [1, 1024]) == [1, 1024, 768]


def test_embedding_lowercase_op_still_resolves():
    # The exact screenshot failure: lowercase op used to pass (1,1024) through.
    assert _infer("embedding", {"num_embeddings": 50257, "embedding_dim": 768}, [1, 1024]) == [1, 1024, 768]


def test_embedding_nn_lowercase_op_still_resolves():
    assert _infer("nn.embedding", {"num_embeddings": 50257, "embedding_dim": 768}, [1, 1024]) == [1, 1024, 768]


def test_embedding_bare_capitalized_op_resolves():
    assert _infer("Embedding", {"num_embeddings": 30522, "embedding_dim": 128}, [2, 16]) == [2, 16, 128]


def test_lowercase_conv2d_op_resolves():
    # The normalization fix is general, not embedding-specific: any layer whose
    # op arrives lowercase must still match its analytic rule.
    assert _infer("conv2d", {"in_channels": 3, "out_channels": 8, "kernel_size": 3}, [1, 3, 32, 32]) == [1, 8, 30, 30]
