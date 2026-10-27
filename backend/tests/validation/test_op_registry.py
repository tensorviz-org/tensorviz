"""Tests for core/op_registry.py — Step 1 gate.

Gate: registry covers all ops explicitly handled in canonical_validation._infer_shapes,
all ops used in the template sweep, and all common palette aliases.
"""

import pytest
from core.op_registry import spec_for, all_specs, OpSpec


# ── Basic lookup ──────────────────────────────────────────────────────────────

def test_spec_for_returns_opspec():
    spec = spec_for("Conv2d")
    assert isinstance(spec, OpSpec)
    assert spec.canonical == "nn.Conv2d"


def test_spec_for_case_insensitive():
    assert spec_for("conv2d") is spec_for("Conv2d")
    assert spec_for("CONV2D") is spec_for("Conv2d")


def test_spec_for_strips_nn_prefix():
    assert spec_for("nn.Conv2d") is spec_for("Conv2d")
    assert spec_for("torch.nn.Conv2d") is spec_for("Conv2d")


def test_spec_for_strips_torch_prefix():
    assert spec_for("torch.add") is spec_for("add")


def test_spec_for_unknown_returns_none():
    assert spec_for("totally_unknown_op_xyz") is None
    assert spec_for("") is None


def test_all_specs_returns_list():
    specs = all_specs()
    assert len(specs) > 0
    assert all(isinstance(s, OpSpec) for s in specs)


def test_all_specs_no_duplicates():
    specs = all_specs()
    canonicals = [s.canonical for s in specs]
    assert len(canonicals) == len(set(canonicals)), "duplicate canonicals in registry"


# ── All ops currently handled in canonical_validation._infer_shapes ──────────
# These are the ops that had explicit analytic rules before the registry.

@pytest.mark.parametrize("op_str", [
    # Conv family
    "nn.Conv1d", "Conv1d", "conv1d",
    "nn.Conv2d", "Conv2d", "conv2d",
    "nn.Conv3d", "Conv3d", "conv3d",
    "nn.ConvTranspose1d", "ConvTranspose1d",
    "nn.ConvTranspose2d", "ConvTranspose2d",
    "nn.ConvTranspose3d", "ConvTranspose3d",
    # Pool family
    "nn.MaxPool1d", "nn.MaxPool2d", "nn.MaxPool3d",
    "nn.AvgPool1d", "nn.AvgPool2d", "nn.AvgPool3d",
    "nn.AdaptiveAvgPool1d", "nn.AdaptiveAvgPool2d", "nn.AdaptiveAvgPool3d",
    "nn.AdaptiveMaxPool1d", "nn.AdaptiveMaxPool2d", "nn.AdaptiveMaxPool3d",
    "nn.LPPool1d", "nn.LPPool2d",
    # Norm family
    "nn.BatchNorm1d", "nn.BatchNorm2d", "nn.BatchNorm3d",
    "nn.GroupNorm",
    "nn.InstanceNorm1d", "nn.InstanceNorm2d", "nn.InstanceNorm3d",
    "nn.LayerNorm",
    "nn.RMSNorm",
    "nn.LocalResponseNorm",
    # Linear / Embedding
    "nn.Linear",
    "nn.Embedding",
    "nn.EmbeddingBag",
    # Recurrent
    "nn.LSTM", "nn.GRU", "nn.RNN",
    "nn.RNNCell", "nn.GRUCell", "nn.LSTMCell",
    # Attention / Transformer
    "nn.MultiheadAttention",
    "nn.TransformerEncoderLayer",
    "nn.TransformerDecoderLayer",
    "nn.Transformer",
    # Tensor ops
    "nn.Flatten",
    "nn.Unflatten",
    "nn.Upsample",
    "nn.PixelShuffle",
    "nn.PixelUnshuffle",
    # Activations (sample)
    "nn.ReLU", "nn.GELU", "nn.Sigmoid", "nn.Tanh",
    "nn.SiLU", "nn.Mish", "nn.ELU", "nn.SELU", "nn.CELU",
    "nn.LeakyReLU", "nn.PReLU", "nn.ReLU6",
    "nn.Softmax", "nn.LogSoftmax",
    "nn.GLU",
    # Dropout
    "nn.Dropout", "nn.Dropout1d", "nn.Dropout2d", "nn.Dropout3d",
    # torch.* merge ops
    "torch.add", "add", "torch.cat", "cat",
    "torch.mul", "mul", "torch.sub", "sub",
    # Reductions (passthrough placeholder)
    "torch.sum", "sum", "torch.mean", "mean",
])
def test_all_canonical_validation_ops_resolve(op_str):
    spec = spec_for(op_str)
    assert spec is not None, f"spec_for({op_str!r}) returned None — missing from registry"


# ── Template-specific aliases (ops used in canonical_template_registry) ───────

@pytest.mark.parametrize("op_str", [
    "rmsnorm", "RMSNorm",
    "LayerNorm", "layernorm",
    "mul", "sum",  # message_passing and t5_ffn
])
def test_template_ops_resolve(op_str):
    spec = spec_for(op_str)
    assert spec is not None, f"{op_str!r} used in templates must resolve"


# ── Shape function smoke tests ────────────────────────────────────────────────

def test_conv2d_shape_fn():
    spec = spec_for("Conv2d")
    params = {"in_channels": 3, "out_channels": 16, "kernel_size": 3, "padding": 1}
    out = spec.shape_fn(params, [1, 3, 32, 32], [[1, 3, 32, 32]])
    assert out == [1, 16, 32, 32]


def test_linear_shape_fn():
    spec = spec_for("Linear")
    params = {"in_features": 128, "out_features": 64}
    out = spec.shape_fn(params, [2, 128], [[2, 128]])
    assert out == [2, 64]


def test_layernorm_shape_fn():
    spec = spec_for("LayerNorm")
    params = {"normalized_shape": [128]}
    out = spec.shape_fn(params, [2, 10, 128], [[2, 10, 128]])
    assert out == [2, 10, 128]


def test_rmsnorm_shape_fn_same_as_layernorm():
    ln_spec = spec_for("LayerNorm")
    rms_spec = spec_for("RMSNorm")
    params = {"normalized_shape": [64]}
    inp = [4, 64]
    assert ln_spec.shape_fn(params, inp, [inp]) == rms_spec.shape_fn(params, inp, [inp])


def test_flatten_shape_fn():
    spec = spec_for("Flatten")
    out = spec.shape_fn({}, [2, 3, 4, 4], [[2, 3, 4, 4]])
    assert out == [2, 48]


def test_flatten_shape_fn_custom_dims():
    spec = spec_for("Flatten")
    out = spec.shape_fn({"start_dim": 2, "end_dim": -1}, [2, 3, 4, 4], [[2, 3, 4, 4]])
    assert out == [2, 3, 16]


def test_embedding_shape_fn():
    spec = spec_for("Embedding")
    params = {"num_embeddings": 1000, "embedding_dim": 128}
    out = spec.shape_fn(params, [2, 10], [[2, 10]])
    assert out == [2, 10, 128]


def test_add_shape_fn_same_shapes():
    spec = spec_for("add")
    inputs = [[2, 64], [2, 64]]
    out = spec.shape_fn({}, inputs[0], inputs)
    assert out == [2, 64]


def test_add_shape_fn_mismatched_shapes_returns_none():
    spec = spec_for("add")
    inputs = [[2, 64], [2, 32]]
    out = spec.shape_fn({}, inputs[0], inputs)
    assert out is None


def test_cat_shape_fn():
    spec = spec_for("cat")
    inputs = [[2, 32, 8, 8], [2, 64, 8, 8]]
    out = spec.shape_fn({"dim": 1}, inputs[0], inputs)
    assert out == [2, 96, 8, 8]


def test_pixel_shuffle_shape_fn():
    spec = spec_for("PixelShuffle")
    params = {"upscale_factor": 2}
    out = spec.shape_fn(params, [1, 4, 8, 8], [[1, 4, 8, 8]])
    assert out == [1, 1, 16, 16]


def test_glu_shape_fn():
    spec = spec_for("GLU")
    out = spec.shape_fn({}, [2, 128], [[2, 128]])
    assert out == [2, 64]


def test_rnn_shape_fn_batch_first():
    spec = spec_for("LSTM")
    params = {"input_size": 32, "hidden_size": 64, "batch_first": True}
    out = spec.shape_fn(params, [4, 10, 32], [[4, 10, 32]])
    assert out == [4, 10, 64]


def test_upsample_scale_factor():
    spec = spec_for("Upsample")
    params = {"scale_factor": 2}
    out = spec.shape_fn(params, [1, 3, 8, 8], [[1, 3, 8, 8]])
    assert out == [1, 3, 16, 16]


def test_adaptiveavgpool2d_shape_fn():
    spec = spec_for("AdaptiveAvgPool2d")
    params = {"output_size": [1, 1]}
    out = spec.shape_fn(params, [2, 64, 8, 8], [[2, 64, 8, 8]])
    assert out == [2, 64, 1, 1]


# ── Arity classification ─────────────────────────────────────────────────────

def test_single_input_ops_have_arity_one():
    for name in ["Conv2d", "Linear", "LSTM", "Flatten"]:
        spec = spec_for(name)
        assert spec.arity == "one", f"{name} should have arity 'one'"


def test_merge_ops_have_arity_merge():
    for name in ["add", "cat", "mul"]:
        spec = spec_for(name)
        assert spec.arity == "merge", f"{name} should have arity 'merge'"


# ── dtype classification ──────────────────────────────────────────────────────

def test_embedding_has_int_in_dtype():
    spec = spec_for("Embedding")
    assert spec.in_dtype == "int"


def test_conv2d_has_float_in_dtype():
    spec = spec_for("Conv2d")
    assert spec.in_dtype == "float"


# ── defining_params ───────────────────────────────────────────────────────────

def test_defining_params_covered():
    conv = spec_for("Conv2d")
    assert "in_channels" in conv.defining_params
    assert "out_channels" in conv.defining_params

    linear = spec_for("Linear")
    assert "in_features" in linear.defining_params

    layernorm = spec_for("LayerNorm")
    assert "normalized_shape" in layernorm.defining_params
