"""Starter templates: small, foundational neural networks built only from
basic building blocks (Linear, Conv, activations, pooling, normalization,
Embedding, RNNs, Flatten).

These exist to exercise the shape-inference / canonical-validation engine with
the simplest possible architectures. Every one is a single-input chain of
torch.nn modules whose channel/feature dims chain exactly, so each instantiates,
passes the verify_model dry forward, and passes canonical shape validation with
zero errors. Input/Output nodes are synthesized by
``TemplateService._ensure_io_nodes`` from ``minimum_input_shape``.
"""
from typing import Any, Dict, List, Tuple

from models.template_models import ModelTemplate, ModuleTemplate, PopularModel

# A layer spec is (layerType.id, layerType.name, constructor params).
LayerSpec = Tuple[str, str, Dict[str, Any]]

# Grid pitch for the generated chain layout. Cards render 200px wide and up to
# ~220px tall (tldraw shape utils), so both pitches must clear those extents or
# adjacent layers visually overlap on the canvas.
_COL_PITCH = 220
_ROW_PITCH = 240


# ── Leaf-spec helpers (keep the architecture table declarative + readable) ────
def _linear(i: int, o: int) -> LayerSpec:
    return ("linear", "Linear", {"in_features": i, "out_features": o})


def _conv2d(i: int, o: int, k: int = 3, s: int = 1, p: int = 1) -> LayerSpec:
    return ("conv2d", "Conv2d",
            {"in_channels": i, "out_channels": o, "kernel_size": k, "stride": s, "padding": p})


def _convt2d(i: int, o: int, k: int = 4, s: int = 2, p: int = 1) -> LayerSpec:
    return ("convtranspose2d", "ConvTranspose2d",
            {"in_channels": i, "out_channels": o, "kernel_size": k, "stride": s, "padding": p})


def _conv1d(i: int, o: int, k: int = 3, s: int = 1, p: int = 1) -> LayerSpec:
    return ("conv1d", "Conv1d",
            {"in_channels": i, "out_channels": o, "kernel_size": k, "stride": s, "padding": p})


def _embedding(n: int, d: int) -> LayerSpec:
    return ("embedding", "Embedding", {"num_embeddings": n, "embedding_dim": d})


def _lstm(i: int, h: int) -> LayerSpec:
    return ("lstm", "LSTM", {"input_size": i, "hidden_size": h, "batch_first": True})


def _gru(i: int, h: int) -> LayerSpec:
    return ("gru", "GRU", {"input_size": i, "hidden_size": h, "batch_first": True})


def _rnn(i: int, h: int) -> LayerSpec:
    return ("rnn", "RNN", {"input_size": i, "hidden_size": h, "batch_first": True})


_relu: LayerSpec = ("relu", "ReLU", {"inplace": True})
_flatten: LayerSpec = ("flatten", "Flatten", {})
_maxpool2d: LayerSpec = ("maxpool2d", "MaxPool2d", {"kernel_size": 2, "stride": 2})
_avgpool2d: LayerSpec = ("avgpool2d", "AvgPool2d", {"kernel_size": 2, "stride": 2})
_maxpool1d: LayerSpec = ("maxpool1d", "MaxPool1d", {"kernel_size": 2, "stride": 2})
_gap2d: LayerSpec = ("adaptiveavgpool2d", "AdaptiveAvgPool2d", {"output_size": [1, 1]})
_gap1d: LayerSpec = ("adaptiveavgpool1d", "AdaptiveAvgPool1d", {"output_size": 1})
_gmp1d: LayerSpec = ("adaptivemaxpool1d", "AdaptiveMaxPool1d", {"output_size": 1})
_gelu: LayerSpec = ("gelu", "GELU", {})
_silu: LayerSpec = ("silu", "SiLU", {})
_elu: LayerSpec = ("elu", "ELU", {})
_tanh: LayerSpec = ("tanh", "Tanh", {})
_sigmoid: LayerSpec = ("sigmoid", "Sigmoid", {})


def _bn1d(c: int) -> LayerSpec:
    return ("batchnorm1d", "BatchNorm1d", {"num_features": c})


def _bn2d(c: int) -> LayerSpec:
    return ("batchnorm2d", "BatchNorm2d", {"num_features": c})


def _groupnorm(g: int, c: int) -> LayerSpec:
    return ("groupnorm", "GroupNorm", {"num_groups": g, "num_channels": c})


def _layernorm(shape: List[int]) -> LayerSpec:
    return ("layernorm", "LayerNorm", {"normalized_shape": shape})


def _dropout(p: float = 0.5) -> LayerSpec:
    return ("dropout", "Dropout", {"p": p})


def _leakyrelu(slope: float = 0.1) -> LayerSpec:
    return ("leakyrelu", "LeakyReLU", {"negative_slope": slope})


def _softmax() -> LayerSpec:
    return ("softmax", "Softmax", {"dim": -1})


def _logsoftmax() -> LayerSpec:
    return ("logsoftmax", "LogSoftmax", {"dim": -1})


def _starter(
    template_id: str,
    name: str,
    description: str,
    sub_category: str,
    input_shape: List[int],
    layers: List[LayerSpec],
    tags: List[str],
) -> ModelTemplate:
    """Build a flat single-input-chain ModelTemplate from a layer-spec list."""
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    for i, (layer_id, layer_name, params) in enumerate(layers):
        node_id = f"{template_id}__n{i}"
        row, col = divmod(i, 8)
        nodes.append({
            "id": node_id,
            "type": "layerNode",
            "position": {"x": 120 + col * _COL_PITCH, "y": row * _ROW_PITCH},
            "data": {
                "layerType": {"id": layer_id, "name": layer_name, "parameters": []},
                "parameters": dict(params),
            },
        })
        if i > 0:
            edges.append({
                "id": f"{template_id}__e{i}",
                "source": f"{template_id}__n{i - 1}",
                "target": node_id,
            })
    return ModelTemplate(
        id=template_id,
        name=name,
        description=description,
        category="starters",
        sub_category=sub_category,
        difficulty="beginner",
        tags=tags,
        base_config={},
        customizable_params=[],
        template_structure={"nodes": nodes, "edges": edges},
        minimum_input_shape=list(input_shape),
        recommended_input_shape=list(input_shape),
    )


def _build_all() -> List[ModelTemplate]:
    """The starter architecture table. Each entry is a small, valid network.

    Shapes are traced so every op's input rank / channels / features chain
    exactly (Conv in_channels == incoming, Linear in_features == incoming,
    pooling keeps the spatial math positive). Comments note the running shape.
    """
    return [
        # ── MLP family (rank-2 inputs) ───────────────────────────────────────
        _starter(
            "mlp_tiny", "Tiny MLP",
            "Two-layer perceptron: the smallest fully-connected network.",
            "mlp", [1, 16],
            [_linear(16, 32), _relu, _linear(32, 4)],
            ["mlp", "linear", "relu"],
        ),
        _starter(
            "mlp_relu_stack", "MLP ReLU Stack",
            "Three Linear layers with ReLU activations between them.",
            "mlp", [1, 32],
            [_linear(32, 64), _relu, _linear(64, 64), _relu, _linear(64, 10)],
            ["mlp", "linear", "relu"],
        ),
        _starter(
            "mlp_softmax_classifier", "MLP Softmax Classifier",
            "Feed-forward classifier ending in a Softmax over 5 classes.",
            "mlp", [1, 20],
            [_linear(20, 40), _relu, _linear(40, 5), _softmax()],
            ["mlp", "softmax", "classifier"],
        ),
        _starter(
            "mlp_dropout", "MLP with Dropout",
            "Fully-connected network with Dropout regularization.",
            "mlp", [1, 64],
            [_linear(64, 128), _relu, _dropout(0.5), _linear(128, 10)],
            ["mlp", "dropout", "regularization"],
        ),
        _starter(
            "mlp_batchnorm1d", "MLP with BatchNorm1d",
            "Linear -> BatchNorm1d -> ReLU block followed by a classifier head.",
            "mlp", [1, 32],
            [_linear(32, 64), _bn1d(64), _relu, _linear(64, 10)],
            ["mlp", "batchnorm", "normalization"],
        ),
        _starter(
            "mlp_layernorm", "MLP with LayerNorm",
            "Linear projection normalized with LayerNorm before the head.",
            "mlp", [1, 48],
            [_linear(48, 64), _layernorm([64]), _relu, _linear(64, 8)],
            ["mlp", "layernorm", "normalization"],
        ),
        _starter(
            "mlp_gelu", "MLP with GELU",
            "Three Linear layers using GELU activations.",
            "mlp", [1, 16],
            [_linear(16, 32), _gelu, _linear(32, 16), _gelu, _linear(16, 2)],
            ["mlp", "gelu", "activation"],
        ),
        _starter(
            "mlp_sigmoid_binary", "MLP Binary Classifier",
            "Feed-forward binary classifier with a single Sigmoid output.",
            "mlp", [1, 30],
            [_linear(30, 16), _relu, _linear(16, 1), _sigmoid],
            ["mlp", "sigmoid", "binary"],
        ),
        _starter(
            "mlp_tanh", "MLP with Tanh",
            "Small fully-connected network using Tanh activation.",
            "mlp", [1, 12],
            [_linear(12, 24), _tanh, _linear(24, 3)],
            ["mlp", "tanh", "activation"],
        ),
        _starter(
            "mlp_deep", "Deep MLP",
            "Five-layer fully-connected network with ReLU activations.",
            "mlp", [1, 64],
            [_linear(64, 64), _relu, _linear(64, 64), _relu, _linear(64, 32),
             _relu, _linear(32, 16), _relu, _linear(16, 10)],
            ["mlp", "deep", "linear"],
        ),
        _starter(
            "mlp_leakyrelu", "MLP with LeakyReLU",
            "Two-layer perceptron using a LeakyReLU activation.",
            "mlp", [1, 16],
            [_linear(16, 32), _leakyrelu(0.1), _linear(32, 8)],
            ["mlp", "leakyrelu", "activation"],
        ),
        _starter(
            "mlp_elu", "MLP with ELU",
            "Fully-connected network using the ELU activation.",
            "mlp", [1, 16],
            [_linear(16, 32), _elu, _linear(32, 4)],
            ["mlp", "elu", "activation"],
        ),
        _starter(
            "mlp_silu", "MLP with SiLU",
            "Fully-connected network using the SiLU (Swish) activation.",
            "mlp", [1, 16],
            [_linear(16, 32), _silu, _linear(32, 4)],
            ["mlp", "silu", "swish", "activation"],
        ),
        _starter(
            "mlp_logsoftmax", "MLP LogSoftmax Classifier",
            "Classifier head producing log-probabilities via LogSoftmax.",
            "mlp", [1, 20],
            [_linear(20, 32), _relu, _linear(32, 10), _logsoftmax()],
            ["mlp", "logsoftmax", "classifier"],
        ),
        _starter(
            "autoencoder_mlp", "MLP Autoencoder",
            "Encoder-bottleneck-decoder fully-connected autoencoder.",
            "mlp", [1, 64],
            [_linear(64, 16), _relu, _linear(16, 64), _sigmoid],
            ["mlp", "autoencoder", "bottleneck"],
        ),

        # ── CNN family (rank-4 inputs) ───────────────────────────────────────
        _starter(
            "cnn_tiny", "Tiny CNN",
            "Single conv layer feeding a flattened linear classifier.",
            "cnn", [1, 3, 32, 32],
            [_conv2d(3, 8), _relu, _flatten, _linear(8 * 32 * 32, 10)],
            ["cnn", "conv2d", "classifier"],
        ),
        _starter(
            "cnn_maxpool", "CNN with MaxPool",
            "Conv -> ReLU -> MaxPool downsampling before the classifier.",
            "cnn", [1, 3, 32, 32],
            [_conv2d(3, 16), _relu, _maxpool2d, _flatten, _linear(16 * 16 * 16, 10)],
            ["cnn", "conv2d", "maxpool"],
        ),
        _starter(
            "lenet5", "LeNet-5",
            "Classic LeNet-5: two conv+pool stages and three linear layers.",
            "cnn", [1, 1, 28, 28],
            [_conv2d(1, 6, k=5, p=0), _relu, _maxpool2d,
             _conv2d(6, 16, k=5, p=0), _relu, _maxpool2d,
             _flatten, _linear(16 * 4 * 4, 120), _relu,
             _linear(120, 84), _relu, _linear(84, 10)],
            ["cnn", "lenet", "classic"],
        ),
        _starter(
            "cnn_batchnorm", "CNN with BatchNorm2d",
            "Two Conv -> BatchNorm2d -> ReLU blocks with a global-pool head.",
            "cnn", [1, 3, 32, 32],
            [_conv2d(3, 16), _bn2d(16), _relu,
             _conv2d(16, 32), _bn2d(32), _relu,
             _gap2d, _flatten, _linear(32, 10)],
            ["cnn", "batchnorm", "normalization"],
        ),
        _starter(
            "cnn_avgpool", "CNN with AvgPool",
            "Two conv stages with average pooling and a linear classifier.",
            "cnn", [1, 3, 32, 32],
            [_conv2d(3, 8), _relu, _avgpool2d,
             _conv2d(8, 16), _relu, _avgpool2d,
             _flatten, _linear(16 * 8 * 8, 10)],
            ["cnn", "avgpool", "conv2d"],
        ),
        _starter(
            "cnn_3conv", "Three-Layer CNN",
            "Three stacked conv layers with a global-average-pool head.",
            "cnn", [1, 3, 32, 32],
            [_conv2d(3, 16), _relu, _conv2d(16, 32), _relu, _conv2d(32, 64), _relu,
             _gap2d, _flatten, _linear(64, 10)],
            ["cnn", "deep", "conv2d"],
        ),
        _starter(
            "cnn_groupnorm", "CNN with GroupNorm",
            "Conv layer normalized with GroupNorm and a global-pool head.",
            "cnn", [1, 3, 32, 32],
            [_conv2d(3, 16), _groupnorm(4, 16), _relu, _gap2d, _flatten, _linear(16, 10)],
            ["cnn", "groupnorm", "normalization"],
        ),
        _starter(
            "cnn_softmax_head", "CNN Softmax Classifier",
            "Conv + pool feature extractor with a Softmax classifier head.",
            "cnn", [1, 3, 32, 32],
            [_conv2d(3, 8), _relu, _maxpool2d, _flatten,
             _linear(8 * 16 * 16, 10), _softmax()],
            ["cnn", "softmax", "classifier"],
        ),
        _starter(
            "mnist_cnn", "MNIST CNN",
            "Two conv+pool stages sized for 28x28 grayscale digit images.",
            "cnn", [1, 1, 28, 28],
            [_conv2d(1, 32), _relu, _maxpool2d,
             _conv2d(32, 64), _relu, _maxpool2d,
             _flatten, _linear(64 * 7 * 7, 128), _relu, _linear(128, 10)],
            ["cnn", "mnist", "classifier"],
        ),
        _starter(
            "conv_autoencoder", "Convolutional Autoencoder",
            "Strided-conv encoder and transposed-conv decoder reconstructing the input.",
            "cnn", [1, 3, 32, 32],
            [_conv2d(3, 16, k=3, s=2, p=1), _relu,
             _conv2d(16, 8, k=3, s=2, p=1), _relu,
             _convt2d(8, 16), _relu, _convt2d(16, 3), _sigmoid],
            ["cnn", "autoencoder", "convtranspose"],
        ),
        _starter(
            "conv_global_pool", "Conv + Global Pool",
            "Single conv layer with global average pooling and Softmax head.",
            "cnn", [1, 3, 16, 16],
            [_conv2d(3, 32), _relu, _gap2d, _flatten, _linear(32, 10), _softmax()],
            ["cnn", "global-pool", "softmax"],
        ),

        # ── Conv1d family (rank-3 inputs) ────────────────────────────────────
        _starter(
            "conv1d_net", "1D CNN",
            "Conv1d feature extractor with global pooling for sequence inputs.",
            "conv1d", [1, 4, 32],
            [_conv1d(4, 8), _relu, _gap1d, _flatten, _linear(8, 4)],
            ["conv1d", "sequence", "pooling"],
        ),
        _starter(
            "conv1d_maxpool", "1D CNN with Pooling",
            "Two Conv1d stages with max pooling for 1D signal classification.",
            "conv1d", [1, 8, 64],
            [_conv1d(8, 16), _relu, _maxpool1d, _conv1d(16, 32), _relu,
             _gmp1d, _flatten, _linear(32, 5)],
            ["conv1d", "maxpool", "signal"],
        ),

        # ── Embedding / NLP family (rank-2 integer inputs) ───────────────────
        _starter(
            "embedding_mlp", "Embedding + MLP",
            "Token embedding flattened into a feed-forward classifier.",
            "embedding", [1, 16],
            [_embedding(1000, 32), _flatten, _linear(16 * 32, 64), _relu, _linear(64, 10)],
            ["embedding", "nlp", "classifier"],
        ),
        _starter(
            "text_classifier", "Text Classifier",
            "Embedding bag-of-tokens classifier with Dropout and Softmax.",
            "embedding", [1, 20],
            [_embedding(5000, 64), _flatten, _linear(20 * 64, 128), _relu,
             _dropout(0.5), _linear(128, 4), _softmax()],
            ["embedding", "nlp", "text", "softmax"],
        ),
        _starter(
            "embedding_lstm", "Embedding + LSTM",
            "Token embedding fed through an LSTM and a linear classifier.",
            "embedding", [1, 16],
            [_embedding(2000, 32), _lstm(32, 64), _flatten, _linear(16 * 64, 10)],
            ["embedding", "lstm", "nlp"],
        ),
        _starter(
            "embedding_gru", "Embedding + GRU",
            "Token embedding fed through a GRU and a linear classifier.",
            "embedding", [1, 12],
            [_embedding(1000, 16), _gru(16, 32), _flatten, _linear(12 * 32, 5)],
            ["embedding", "gru", "nlp"],
        ),

        # ── RNN family (rank-3 float inputs) ─────────────────────────────────
        _starter(
            "rnn_net", "Vanilla RNN",
            "Elman RNN over a float sequence with a linear classifier.",
            "recurrent", [1, 10, 8],
            [_rnn(8, 16), _flatten, _linear(10 * 16, 4)],
            ["rnn", "recurrent", "sequence"],
        ),
        _starter(
            "lstm_net", "LSTM Sequence Model",
            "LSTM over a float sequence with a linear classifier.",
            "recurrent", [1, 10, 8],
            [_lstm(8, 16), _flatten, _linear(10 * 16, 4)],
            ["lstm", "recurrent", "sequence"],
        ),
        _starter(
            "gru_net", "GRU Sequence Model",
            "GRU over a float sequence with a linear classifier.",
            "recurrent", [1, 10, 8],
            [_gru(8, 16), _flatten, _linear(10 * 16, 4)],
            ["gru", "recurrent", "sequence"],
        ),

        # ── Tensor-op showcase ───────────────────────────────────────────────
        _starter(
            "flatten_linear", "Flatten + Linear",
            "Flattens a small image tensor straight into an MLP classifier.",
            "mlp", [1, 3, 8, 8],
            [_flatten, _linear(3 * 8 * 8, 32), _relu, _linear(32, 10)],
            ["flatten", "mlp", "linear"],
        ),
    ]


class StarterTemplates:
    """Foundational starter architectures for the 'Starters' category."""

    def get_all_templates(self) -> Dict[str, ModelTemplate]:
        return {t.id: t for t in _build_all()}

    def get_module_templates(self) -> Dict[str, ModuleTemplate]:
        return {}

    def get_popular_models(self) -> Dict[str, PopularModel]:
        return {}
