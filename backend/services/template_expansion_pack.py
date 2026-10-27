"""Built-in expansion pack for canonical templates (non-audio families).

These are registered as dynamic templates at startup (persist=False) so they
behave like first-class templates without requiring 30+ bespoke builder methods.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _chain_nodes(nodes: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    edges: List[Dict[str, str]] = []
    for idx in range(len(nodes) - 1):
        edges.append(
            {
                "source_suffix": str(nodes[idx]["suffix"]),
                "target_suffix": str(nodes[idx + 1]["suffix"]),
            }
        )
    return edges


def _template(
    *,
    template_id: str,
    name: str,
    family: str,
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, str]] | None = None,
    training_minimums: List[str] | None = None,
    input_expectation: str,
    allowed_residual_patterns: List[str] | None = None,
    aliases: List[str] | None = None,
) -> Dict[str, Any]:
    final_edges = edges or _chain_nodes(nodes)
    return {
        "template_id": template_id,
        "name": name,
        "family": family,
        "nodes": nodes,
        "edges": final_edges,
        "required_nodes": [str(node["layer_op"]) for node in nodes],
        "required_edges": [
            f"{edge['source_suffix']} -> {edge['target_suffix']}" for edge in final_edges
        ],
        "allowed_residual_patterns": allowed_residual_patterns or [],
        "training_minimums": training_minimums or [],
        "input_expectation": input_expectation,
        "aliases": aliases or [],
        "metadata": {"source": "expansion_pack"},
    }


EXPANSION_TEMPLATES: List[Dict[str, Any]] = [
    # Transformer / LLM variants
    _template(
        template_id="transformer_encoder_prenorm_block",
        name="Transformer Encoder PreNorm",
        family="transformer",
        nodes=[
            {"suffix": "norm1", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "attn", "layer_op": "nn.MultiheadAttention", "default_params": {"batch_first": True}},
            {"suffix": "add1", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "norm2", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "ff1", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "gelu", "layer_op": "nn.GELU", "default_params": {}},
            {"suffix": "ff2", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "add2", "layer_op": "torch.add", "default_params": {}},
        ],
        edges=[
            {"source_suffix": "norm1", "target_suffix": "attn"},
            {"source_suffix": "attn", "target_suffix": "add1"},
            {"source_suffix": "add1", "target_suffix": "norm2"},
            {"source_suffix": "norm2", "target_suffix": "ff1"},
            {"source_suffix": "ff1", "target_suffix": "gelu"},
            {"source_suffix": "gelu", "target_suffix": "ff2"},
            {"source_suffix": "ff2", "target_suffix": "add2"},
        ],
        training_minimums=["d_model % num_heads == 0", "ffn_dim >= d_model"],
        input_expectation="rank-3 tensor [batch, tokens, d_model]",
        allowed_residual_patterns=["post-attn residual", "post-mlp residual"],
        aliases=["transformer_encoder", "encoder_prenorm"],
    ),
    _template(
        template_id="transformer_decoder_block",
        name="Transformer Decoder Block",
        family="transformer",
        nodes=[
            {"suffix": "norm1", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "self_attn", "layer_op": "nn.MultiheadAttention", "default_params": {"batch_first": True}},
            {"suffix": "add1", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "norm2", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "cross_attn", "layer_op": "nn.MultiheadAttention", "default_params": {"batch_first": True}},
            {"suffix": "add2", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "norm3", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "ff1", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "gelu", "layer_op": "nn.GELU", "default_params": {}},
            {"suffix": "ff2", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "add3", "layer_op": "torch.add", "default_params": {}},
        ],
        training_minimums=["d_model % num_heads == 0", "ffn_dim >= d_model"],
        input_expectation="rank-3 tensor [batch, tokens, d_model]",
        allowed_residual_patterns=["self-attn residual", "cross-attn residual", "ff residual"],
        aliases=["transformer_decoder", "decoder_cross_attention"],
    ),
    _template(
        template_id="transformer_cross_attention_block",
        name="Transformer Cross-Attention Block",
        family="transformer",
        nodes=[
            {"suffix": "q_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "kv_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "attn", "layer_op": "nn.MultiheadAttention", "default_params": {"batch_first": True}},
            {"suffix": "out_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "add", "layer_op": "torch.add", "default_params": {}},
        ],
        training_minimums=["embed_dim % num_heads == 0"],
        input_expectation="rank-3 tensor [batch, tokens, embed_dim]",
        aliases=["cross_attention", "encoder_decoder_attention"],
    ),
    _template(
        template_id="transformer_moe_block",
        name="Transformer MoE Feedforward Block",
        family="transformer",
        nodes=[
            {"suffix": "gate", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "topk", "layer_op": "topk_router", "default_params": {"k": 2}},
            {"suffix": "expert1", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "expert2", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "merge", "layer_op": "mixture_merge", "default_params": {}},
            {"suffix": "out", "layer_op": "nn.Linear", "default_params": {}},
        ],
        training_minimums=["num_experts >= 2", "top_k >= 1"],
        input_expectation="rank-3 tensor [batch, tokens, d_model]",
        aliases=["moe_block", "mixture_of_experts"],
    ),
    _template(
        template_id="transformer_rmsnorm_block",
        name="Transformer RMSNorm Block",
        family="transformer",
        nodes=[
            {"suffix": "rms1", "layer_op": "rmsnorm", "default_params": {}},
            {"suffix": "attn", "layer_op": "nn.MultiheadAttention", "default_params": {"batch_first": True}},
            {"suffix": "add1", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "rms2", "layer_op": "rmsnorm", "default_params": {}},
            {"suffix": "ff", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "add2", "layer_op": "torch.add", "default_params": {}},
        ],
        input_expectation="rank-3 tensor [batch, tokens, d_model]",
        aliases=["llama_block", "rmsnorm_transformer"],
    ),
    _template(
        template_id="transformer_glu_mlp_block",
        name="Transformer GLU MLP Block",
        family="transformer",
        nodes=[
            {"suffix": "in_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "gate_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "glu", "layer_op": "glu_activation", "default_params": {}},
            {"suffix": "out_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "add", "layer_op": "torch.add", "default_params": {}},
        ],
        input_expectation="rank-3 tensor [batch, tokens, d_model]",
        aliases=["swiglu_block", "gated_mlp"],
    ),
    _template(
        template_id="long_context_attention_block",
        name="Long Context Attention Block",
        family="transformer",
        nodes=[
            {"suffix": "qkv", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "sliding_attn", "layer_op": "sliding_window_attention", "default_params": {"window": 512}},
            {"suffix": "global_attn", "layer_op": "global_attention", "default_params": {}},
            {"suffix": "merge", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "out", "layer_op": "nn.Linear", "default_params": {}},
        ],
        input_expectation="rank-3 tensor [batch, tokens, d_model]",
        aliases=["longformer_block", "sparse_attention_block"],
    ),
    _template(
        template_id="rotary_attention_block",
        name="Rotary Position Attention Block",
        family="transformer",
        nodes=[
            {"suffix": "qkv", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "rope", "layer_op": "rotary_embedding", "default_params": {}},
            {"suffix": "attn", "layer_op": "nn.MultiheadAttention", "default_params": {"batch_first": True}},
            {"suffix": "out", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "add", "layer_op": "torch.add", "default_params": {}},
        ],
        input_expectation="rank-3 tensor [batch, tokens, d_model]",
        aliases=["rope_attention", "gpt_block"],
    ),
    _template(
        template_id="t5_ffn_block",
        name="T5 Feedforward Block",
        family="transformer",
        nodes=[
            {"suffix": "norm", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "wi0", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "wi1", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "relu", "layer_op": "nn.ReLU", "default_params": {}},
            {"suffix": "mul", "layer_op": "torch.mul", "default_params": {}},
            {"suffix": "wo", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "add", "layer_op": "torch.add", "default_params": {}},
        ],
        input_expectation="rank-3 tensor [batch, tokens, d_model]",
        aliases=["t5_block", "t5_ffn"],
    ),
    # Vision/CNN variants
    _template(
        template_id="convnext_block",
        name="ConvNeXt Block",
        family="cnn",
        nodes=[
            {"suffix": "dwconv", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 7, "padding": 3, "groups": 1}},
            {"suffix": "norm", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "pw1", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
            {"suffix": "gelu", "layer_op": "nn.GELU", "default_params": {}},
            {"suffix": "pw2", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
            {"suffix": "add", "layer_op": "torch.add", "default_params": {}},
        ],
        input_expectation="rank-4 tensor [batch, channels, h, w]",
        aliases=["convnext", "convnext_stage"],
    ),
    _template(
        template_id="efficientnet_mbconv_block",
        name="EfficientNet MBConv Block",
        family="cnn",
        nodes=[
            {"suffix": "expand", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
            {"suffix": "bn1", "layer_op": "nn.BatchNorm2d", "default_params": {}},
            {"suffix": "swish", "layer_op": "nn.SiLU", "default_params": {}},
            {"suffix": "depthwise", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            {"suffix": "bn2", "layer_op": "nn.BatchNorm2d", "default_params": {}},
            {"suffix": "se", "layer_op": "se_block", "default_params": {}},
            {"suffix": "project", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
            {"suffix": "add", "layer_op": "torch.add", "default_params": {}},
        ],
        input_expectation="rank-4 tensor [batch, channels, h, w]",
        aliases=["efficientnet", "mbconv"],
    ),
    _template(
        template_id="swin_window_block",
        name="Swin Window Attention Block",
        family="vision_transformer",
        nodes=[
            {"suffix": "norm1", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "window_partition", "layer_op": "window_partition", "default_params": {"window_size": 7}},
            {"suffix": "attn", "layer_op": "nn.MultiheadAttention", "default_params": {"batch_first": True}},
            {"suffix": "window_reverse", "layer_op": "window_reverse", "default_params": {"window_size": 7}},
            {"suffix": "add1", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "norm2", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "mlp", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "add2", "layer_op": "torch.add", "default_params": {}},
        ],
        input_expectation="rank-3 tensor [batch, tokens, embed_dim]",
        aliases=["swin", "window_attention"],
    ),
    _template(
        template_id="inception_mixed_block",
        name="Inception Mixed Branch Block",
        family="cnn",
        nodes=[
            {"suffix": "branch1x1", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
            {"suffix": "branch3x3_reduce", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
            {"suffix": "branch3x3", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            {"suffix": "branch5x5_reduce", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
            {"suffix": "branch5x5", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 5, "padding": 2}},
            {"suffix": "pool_proj", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
            {"suffix": "concat", "layer_op": "torch.cat", "default_params": {"dim": 1}},
        ],
        edges=[
            {"source_suffix": "branch3x3_reduce", "target_suffix": "branch3x3"},
            {"source_suffix": "branch5x5_reduce", "target_suffix": "branch5x5"},
            {"source_suffix": "branch1x1", "target_suffix": "concat"},
            {"source_suffix": "branch3x3", "target_suffix": "concat"},
            {"source_suffix": "branch5x5", "target_suffix": "concat"},
            {"source_suffix": "pool_proj", "target_suffix": "concat"},
        ],
        input_expectation="rank-4 tensor [batch, channels, h, w]",
        aliases=["inception_block"],
    ),
    _template(
        template_id="csp_bottleneck_block",
        name="CSP Bottleneck Block",
        family="cnn",
        nodes=[
            {"suffix": "split1", "layer_op": "split_path_a", "default_params": {}},
            {"suffix": "split2", "layer_op": "split_path_b", "default_params": {}},
            {"suffix": "conv1", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
            {"suffix": "conv2", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            {"suffix": "concat", "layer_op": "torch.cat", "default_params": {"dim": 1}},
            {"suffix": "fuse", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
        ],
        edges=[
            {"source_suffix": "split1", "target_suffix": "conv1"},
            {"source_suffix": "conv1", "target_suffix": "conv2"},
            {"source_suffix": "conv2", "target_suffix": "concat"},
            {"source_suffix": "split2", "target_suffix": "concat"},
            {"source_suffix": "concat", "target_suffix": "fuse"},
        ],
        input_expectation="rank-4 tensor [batch, channels, h, w]",
        aliases=["csp_block", "yolo_csp"],
    ),
    _template(
        template_id="se_residual_block",
        name="Squeeze-Excitation Residual Block",
        family="cnn",
        nodes=[
            {"suffix": "conv1", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            {"suffix": "bn1", "layer_op": "nn.BatchNorm2d", "default_params": {}},
            {"suffix": "relu", "layer_op": "nn.ReLU", "default_params": {}},
            {"suffix": "conv2", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            {"suffix": "se", "layer_op": "se_block", "default_params": {}},
            {"suffix": "add", "layer_op": "torch.add", "default_params": {}},
        ],
        input_expectation="rank-4 tensor [batch, channels, h, w]",
        aliases=["senet_block", "se_block"],
    ),
    _template(
        template_id="ghost_bottleneck_block",
        name="GhostNet Bottleneck Block",
        family="cnn",
        nodes=[
            {"suffix": "ghost1", "layer_op": "ghost_module", "default_params": {}},
            {"suffix": "dwconv", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            {"suffix": "se", "layer_op": "se_block", "default_params": {}},
            {"suffix": "ghost2", "layer_op": "ghost_module", "default_params": {}},
            {"suffix": "add", "layer_op": "torch.add", "default_params": {}},
        ],
        input_expectation="rank-4 tensor [batch, channels, h, w]",
        aliases=["ghostnet_block", "ghost_bottleneck"],
    ),
    _template(
        template_id="fpn_fusion_block",
        name="Feature Pyramid Fusion Block",
        family="vision",
        nodes=[
            {"suffix": "lateral", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
            {"suffix": "upsample", "layer_op": "nn.Upsample", "default_params": {"scale_factor": 2}},
            {"suffix": "add", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "smooth", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
        ],
        input_expectation="rank-4 tensor [batch, channels, h, w]",
        aliases=["fpn_block", "feature_pyramid_block"],
    ),
    _template(
        template_id="aspp_block",
        name="ASPP Block",
        family="segmentation",
        nodes=[
            {"suffix": "conv1x1", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
            {"suffix": "dil6", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 6, "dilation": 6}},
            {"suffix": "dil12", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 12, "dilation": 12}},
            {"suffix": "dil18", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 18, "dilation": 18}},
            {"suffix": "image_pool", "layer_op": "nn.AdaptiveAvgPool2d", "default_params": {"output_size": [1, 1]}},
            {"suffix": "image_up", "layer_op": "nn.Upsample", "default_params": {"size": [224, 224], "mode": "bilinear"}},
            {"suffix": "concat", "layer_op": "torch.cat", "default_params": {"dim": 1}},
            {"suffix": "project", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
        ],
        edges=[
            {"source_suffix": "conv1x1", "target_suffix": "concat"},
            {"source_suffix": "dil6", "target_suffix": "concat"},
            {"source_suffix": "dil12", "target_suffix": "concat"},
            {"source_suffix": "dil18", "target_suffix": "concat"},
            {"source_suffix": "image_pool", "target_suffix": "image_up"},
            {"source_suffix": "image_up", "target_suffix": "concat"},
            {"source_suffix": "concat", "target_suffix": "project"},
        ],
        input_expectation="rank-4 tensor [batch, channels, h, w]",
        aliases=["deeplab_aspp", "aspp"],
    ),
    _template(
        template_id="unetpp_dense_skip_block",
        name="UNet++ Dense Skip Block",
        family="segmentation",
        nodes=[
            {"suffix": "enc", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            {"suffix": "skip1", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            {"suffix": "skip2", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            {"suffix": "concat", "layer_op": "torch.cat", "default_params": {"dim": 1}},
            {"suffix": "fuse", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
        ],
        edges=[
            {"source_suffix": "enc", "target_suffix": "skip1"},
            {"source_suffix": "skip1", "target_suffix": "skip2"},
            {"source_suffix": "enc", "target_suffix": "concat"},
            {"source_suffix": "skip1", "target_suffix": "concat"},
            {"source_suffix": "skip2", "target_suffix": "concat"},
            {"source_suffix": "concat", "target_suffix": "fuse"},
        ],
        input_expectation="rank-4 tensor [batch, channels, h, w]",
        aliases=["unetpp_block", "nested_unet_block"],
    ),
    # GNN variants
    _template(
        template_id="graphsage_block",
        name="GraphSAGE Block",
        family="gnn",
        nodes=[
            {"suffix": "sample", "layer_op": "neighbor_sampler", "default_params": {}},
            {"suffix": "aggregate", "layer_op": "mean_aggregate", "default_params": {}},
            {"suffix": "concat", "layer_op": "torch.cat", "default_params": {"dim": -1}},
            {"suffix": "linear", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "relu", "layer_op": "nn.ReLU", "default_params": {}},
        ],
        input_expectation="rank-2 node features [num_nodes, features]",
        aliases=["graphsage", "sage_layer"],
    ),
    _template(
        template_id="gin_block",
        name="GIN Block",
        family="gnn",
        nodes=[
            {"suffix": "aggregate", "layer_op": "sum_aggregate", "default_params": {}},
            {"suffix": "eps_add", "layer_op": "epsilon_add", "default_params": {}},
            {"suffix": "mlp1", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "relu", "layer_op": "nn.ReLU", "default_params": {}},
            {"suffix": "mlp2", "layer_op": "nn.Linear", "default_params": {}},
        ],
        input_expectation="rank-2 node features [num_nodes, features]",
        aliases=["gin_layer", "graph_isomorphism_network"],
    ),
    _template(
        template_id="rgcn_block",
        name="Relational GCN Block",
        family="gnn",
        nodes=[
            {"suffix": "relation_split", "layer_op": "relation_split", "default_params": {}},
            {"suffix": "relation_conv", "layer_op": "relation_conv", "default_params": {}},
            {"suffix": "sum", "layer_op": "relation_sum", "default_params": {}},
            {"suffix": "linear", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "relu", "layer_op": "nn.ReLU", "default_params": {}},
        ],
        input_expectation="rank-2 node features [num_nodes, features]",
        aliases=["rgcn_layer", "relational_gcn"],
    ),
    # Sequence / recurrent variants
    _template(
        template_id="gru_block",
        name="GRU Block",
        family="sequence",
        nodes=[
            {"suffix": "gru", "layer_op": "nn.GRU", "default_params": {"batch_first": True}},
            {"suffix": "dropout", "layer_op": "nn.Dropout", "default_params": {"p": 0.1}},
            {"suffix": "proj", "layer_op": "nn.Linear", "default_params": {}},
        ],
        input_expectation="rank-3 tensor [batch, sequence, features]",
        aliases=["gru", "recurrent_gru"],
    ),
    _template(
        template_id="bilstm_block",
        name="BiLSTM Block",
        family="sequence",
        nodes=[
            {"suffix": "bilstm", "layer_op": "nn.LSTM", "default_params": {"batch_first": True, "bidirectional": True}},
            {"suffix": "proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "norm", "layer_op": "nn.LayerNorm", "default_params": {}},
        ],
        input_expectation="rank-3 tensor [batch, sequence, features]",
        aliases=["bilstm", "bidirectional_lstm"],
    ),
    _template(
        template_id="temporal_attention_block",
        name="Temporal Attention Block",
        family="sequence",
        nodes=[
            {"suffix": "q_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "k_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "v_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "attn", "layer_op": "scaled_dot_attention", "default_params": {}},
            {"suffix": "out", "layer_op": "nn.Linear", "default_params": {}},
        ],
        input_expectation="rank-3 tensor [batch, sequence, features]",
        aliases=["temporal_self_attention", "time_attention"],
    ),
    _template(
        template_id="conformer_block",
        name="Conformer Block",
        family="sequence",
        nodes=[
            {"suffix": "ffn1", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "mhsa", "layer_op": "nn.MultiheadAttention", "default_params": {"batch_first": True}},
            {"suffix": "conv", "layer_op": "conformer_conv_module", "default_params": {"kernel_size": 15, "padding": 7}},
            {"suffix": "ffn2", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "norm", "layer_op": "nn.LayerNorm", "default_params": {}},
        ],
        input_expectation="rank-3 tensor [batch, sequence, features]",
        aliases=["conformer", "conformer_encoder"],
    ),
    # Generative / diffusion variants
    _template(
        template_id="diffusion_residual_block",
        name="Diffusion Residual Block",
        family="generative",
        nodes=[
            {"suffix": "norm1", "layer_op": "nn.GroupNorm", "default_params": {}},
            {"suffix": "silu1", "layer_op": "nn.SiLU", "default_params": {}},
            {"suffix": "conv1", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            # No time projection. A DDPM residual block conditions on the timestep by
            # projecting the time EMBEDDING and adding it to the convolution's output — a
            # side branch from a second input. These templates are a linear chain with one
            # input, so the projection was sitting in series between conv1 and norm2,
            # feeding a Linear the 4-D output of a Conv2d: "Incompatible connection
            # conv1 -> time_proj: 2D tensor cannot be directly connected", and the block
            # could be drawn but never turned into code. What a single-input chain can
            # honestly say is the residual block without its conditioning, which is what
            # this now is.
            {"suffix": "norm2", "layer_op": "nn.GroupNorm", "default_params": {}},
            {"suffix": "silu2", "layer_op": "nn.SiLU", "default_params": {}},
            {"suffix": "conv2", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            {"suffix": "add", "layer_op": "torch.add", "default_params": {}},
        ],
        input_expectation="rank-4 tensor [batch, channels, h, w]",
        aliases=["ddpm_resblock", "unet_resblock"],
    ),
    _template(
        template_id="diffusion_attention_block",
        name="Diffusion Attention Block",
        family="generative",
        nodes=[
            {"suffix": "norm", "layer_op": "nn.GroupNorm", "default_params": {}},
            {"suffix": "qkv", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
            {"suffix": "attn", "layer_op": "spatial_attention", "default_params": {}},
            {"suffix": "proj", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
            {"suffix": "add", "layer_op": "torch.add", "default_params": {}},
        ],
        input_expectation="rank-4 tensor [batch, channels, h, w]",
        aliases=["ddpm_attention", "spatial_attention_block"],
    ),
    _template(
        template_id="latent_diffusion_block",
        name="Latent Diffusion Block",
        family="generative",
        nodes=[
            {"suffix": "vae_encode", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            {"suffix": "noise", "layer_op": "gaussian_noise", "default_params": {}},
            {"suffix": "denoise", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            {"suffix": "vae_decode", "layer_op": "nn.ConvTranspose2d", "default_params": {"kernel_size": 3, "padding": 1}},
        ],
        input_expectation="rank-4 tensor [batch, channels, h, w]",
        aliases=["latent_diffusion", "stable_diffusion_block"],
    ),
    _template(
        template_id="normalizing_flow_block",
        name="Normalizing Flow Coupling Block",
        family="generative",
        nodes=[
            {"suffix": "split", "layer_op": "channel_split", "default_params": {}},
            {"suffix": "scale_net", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "shift_net", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "affine", "layer_op": "affine_coupling", "default_params": {}},
            {"suffix": "concat", "layer_op": "channel_concat", "default_params": {}},
        ],
        input_expectation="rank-2 tensor [batch, features]",
        aliases=["flow_block", "realnvp_block"],
    ),
    _template(
        template_id="autoregressive_decoder_block",
        name="Autoregressive Decoder Block",
        family="generative",
        nodes=[
            {"suffix": "masked_conv", "layer_op": "masked_conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            {"suffix": "relu", "layer_op": "nn.ReLU", "default_params": {}},
            {"suffix": "residual", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "head", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
        ],
        input_expectation="rank-4 tensor [batch, channels, h, w]",
        aliases=["pixelcnn_block", "autoregressive_block"],
    ),
    # Multimodal variants
    _template(
        template_id="cross_modal_attention_block",
        name="Cross-Modal Attention Block",
        family="multimodal",
        nodes=[
            {"suffix": "vision_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "text_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "cross_attn", "layer_op": "nn.MultiheadAttention", "default_params": {"batch_first": True}},
            {"suffix": "fusion", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "add", "layer_op": "torch.add", "default_params": {}},
        ],
        input_expectation="rank-3 tensor [batch, tokens, embed_dim]",
        aliases=["multimodal_cross_attention", "flamingo_block"],
    ),
    _template(
        template_id="dual_tower_projection_block",
        name="Dual Tower Projection Block",
        family="multimodal",
        nodes=[
            {"suffix": "img_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "txt_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "img_norm", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "txt_norm", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "sim", "layer_op": "cosine_similarity", "default_params": {}},
        ],
        input_expectation="rank-2 tensor [batch, features]",
        aliases=["clip_projection", "contrastive_projection_block"],
    ),
    # Detection / pyramid
    _template(
        template_id="retinanet_head_block",
        name="RetinaNet Head Block",
        family="vision",
        nodes=[
            {"suffix": "conv1", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            {"suffix": "relu1", "layer_op": "nn.ReLU", "default_params": {}},
            {"suffix": "conv2", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            {"suffix": "relu2", "layer_op": "nn.ReLU", "default_params": {}},
            {"suffix": "cls_head", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            {"suffix": "box_head", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
        ],
        edges=[
            {"source_suffix": "conv1", "target_suffix": "relu1"},
            {"source_suffix": "relu1", "target_suffix": "conv2"},
            {"source_suffix": "conv2", "target_suffix": "relu2"},
            {"source_suffix": "relu2", "target_suffix": "cls_head"},
            {"source_suffix": "relu2", "target_suffix": "box_head"},
        ],
        input_expectation="rank-4 tensor [batch, channels, h, w]",
        aliases=["retinanet_head", "detection_head"],
    ),
    _template(
        template_id="yolo_head_block",
        name="YOLO Head Block",
        family="vision",
        nodes=[
            {"suffix": "conv_reduce", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
            {"suffix": "conv_refine", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            {"suffix": "pred", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
        ],
        input_expectation="rank-4 tensor [batch, channels, h, w]",
        aliases=["yolo_head", "detection_yolo_head"],
    ),
    _template(
        template_id="patch_merge_block",
        name="Patch Merge Block",
        family="vision_transformer",
        nodes=[
            {"suffix": "reshape", "layer_op": "patch_reshape", "default_params": {}},
            {"suffix": "concat", "layer_op": "torch.cat", "default_params": {"dim": -1}},
            {"suffix": "norm", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "linear", "layer_op": "nn.Linear", "default_params": {}},
        ],
        input_expectation="rank-3 tensor [batch, tokens, embed_dim]",
        aliases=["swin_patch_merge", "patch_merging"],
    ),
    _template(
        template_id="regnet_block",
        name="RegNet Residual Stage Block",
        family="cnn",
        nodes=[
            {"suffix": "conv1", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
            {"suffix": "bn1", "layer_op": "nn.BatchNorm2d", "default_params": {}},
            {"suffix": "relu1", "layer_op": "nn.ReLU", "default_params": {}},
            {"suffix": "conv2", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            {"suffix": "bn2", "layer_op": "nn.BatchNorm2d", "default_params": {}},
            {"suffix": "relu2", "layer_op": "nn.ReLU", "default_params": {}},
            {"suffix": "proj", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
            {"suffix": "add", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "relu_out", "layer_op": "nn.ReLU", "default_params": {}},
        ],
        edges=[
            {"source_suffix": "conv1", "target_suffix": "bn1"},
            {"source_suffix": "bn1", "target_suffix": "relu1"},
            {"source_suffix": "relu1", "target_suffix": "conv2"},
            {"source_suffix": "conv2", "target_suffix": "bn2"},
            {"source_suffix": "bn2", "target_suffix": "relu2"},
            {"source_suffix": "relu2", "target_suffix": "add"},
            {"source_suffix": "proj", "target_suffix": "add"},
            {"source_suffix": "add", "target_suffix": "relu_out"},
        ],
        input_expectation="rank-4 tensor [batch, channels, h, w]",
        aliases=["regnet_stage", "regnet"],
    ),
    _template(
        template_id="darts_cell",
        name="DARTS Normal Cell",
        family="nas",
        nodes=[
            {"suffix": "preproc0", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
            {"suffix": "preproc1", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
            {"suffix": "sepconv3x3", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            {"suffix": "dilconv3x3", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 2, "dilation": 2}},
            {"suffix": "skip", "layer_op": "identity", "default_params": {}},
            {"suffix": "maxpool", "layer_op": "nn.MaxPool2d", "default_params": {"kernel_size": 3, "stride": 1, "padding": 1}},
            {"suffix": "add1", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "add2", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "concat", "layer_op": "torch.cat", "default_params": {"dim": 1}},
        ],
        edges=[
            {"source_suffix": "preproc0", "target_suffix": "sepconv3x3"},
            {"source_suffix": "preproc1", "target_suffix": "dilconv3x3"},
            {"source_suffix": "sepconv3x3", "target_suffix": "add1"},
            {"source_suffix": "dilconv3x3", "target_suffix": "add1"},
            {"source_suffix": "preproc0", "target_suffix": "skip"},
            {"source_suffix": "preproc1", "target_suffix": "maxpool"},
            {"source_suffix": "skip", "target_suffix": "add2"},
            {"source_suffix": "maxpool", "target_suffix": "add2"},
            {"source_suffix": "add1", "target_suffix": "concat"},
            {"source_suffix": "add2", "target_suffix": "concat"},
        ],
        input_expectation="rank-4 tensor [batch, channels, h, w]",
        aliases=["darts_normal_cell", "nas_cell_darts"],
    ),
    _template(
        template_id="resnext_bottleneck_block",
        name="ResNeXt Bottleneck Block",
        family="cnn",
        nodes=[
            {"suffix": "conv1", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
            {"suffix": "bn1", "layer_op": "nn.BatchNorm2d", "default_params": {}},
            {"suffix": "relu1", "layer_op": "nn.ReLU", "default_params": {}},
            {"suffix": "group_conv", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1, "groups": 32}},
            {"suffix": "bn2", "layer_op": "nn.BatchNorm2d", "default_params": {}},
            {"suffix": "relu2", "layer_op": "nn.ReLU", "default_params": {}},
            {"suffix": "conv3", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
            {"suffix": "bn3", "layer_op": "nn.BatchNorm2d", "default_params": {}},
            {"suffix": "proj", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
            {"suffix": "add", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "relu_out", "layer_op": "nn.ReLU", "default_params": {}},
        ],
        edges=[
            {"source_suffix": "conv1", "target_suffix": "bn1"},
            {"source_suffix": "bn1", "target_suffix": "relu1"},
            {"source_suffix": "relu1", "target_suffix": "group_conv"},
            {"source_suffix": "group_conv", "target_suffix": "bn2"},
            {"source_suffix": "bn2", "target_suffix": "relu2"},
            {"source_suffix": "relu2", "target_suffix": "conv3"},
            {"source_suffix": "conv3", "target_suffix": "bn3"},
            {"source_suffix": "bn3", "target_suffix": "add"},
            {"source_suffix": "proj", "target_suffix": "add"},
            {"source_suffix": "add", "target_suffix": "relu_out"},
        ],
        input_expectation="rank-4 tensor [batch, channels, h, w]",
        aliases=["resnext_block", "resnext_bottleneck"],
    ),
    _template(
        template_id="wide_resnet_block",
        name="Wide ResNet Block",
        family="cnn",
        nodes=[
            {"suffix": "bn1", "layer_op": "nn.BatchNorm2d", "default_params": {}},
            {"suffix": "relu1", "layer_op": "nn.ReLU", "default_params": {}},
            {"suffix": "conv1", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            {"suffix": "dropout", "layer_op": "nn.Dropout", "default_params": {"p": 0.3}},
            {"suffix": "bn2", "layer_op": "nn.BatchNorm2d", "default_params": {}},
            {"suffix": "relu2", "layer_op": "nn.ReLU", "default_params": {}},
            {"suffix": "conv2", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            {"suffix": "proj", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
            {"suffix": "add", "layer_op": "torch.add", "default_params": {}},
        ],
        edges=[
            {"source_suffix": "bn1", "target_suffix": "relu1"},
            {"source_suffix": "relu1", "target_suffix": "conv1"},
            {"source_suffix": "conv1", "target_suffix": "dropout"},
            {"source_suffix": "dropout", "target_suffix": "bn2"},
            {"source_suffix": "bn2", "target_suffix": "relu2"},
            {"source_suffix": "relu2", "target_suffix": "conv2"},
            {"source_suffix": "conv2", "target_suffix": "add"},
            {"source_suffix": "proj", "target_suffix": "add"},
        ],
        input_expectation="rank-4 tensor [batch, channels, h, w]",
        aliases=["wideresnet_block", "wrn_block"],
    ),
    _template(
        template_id="xception_block",
        name="Xception Separable Conv Block",
        family="cnn",
        nodes=[
            {"suffix": "sepconv1_dw", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1, "groups": 1}},
            {"suffix": "sepconv1_pw", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
            {"suffix": "bn1", "layer_op": "nn.BatchNorm2d", "default_params": {}},
            {"suffix": "relu1", "layer_op": "nn.ReLU", "default_params": {}},
            {"suffix": "sepconv2_dw", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1, "groups": 1}},
            {"suffix": "sepconv2_pw", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
            {"suffix": "bn2", "layer_op": "nn.BatchNorm2d", "default_params": {}},
            {"suffix": "relu2", "layer_op": "nn.ReLU", "default_params": {}},
            {"suffix": "proj", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
            {"suffix": "add", "layer_op": "torch.add", "default_params": {}},
        ],
        edges=[
            {"source_suffix": "sepconv1_dw", "target_suffix": "sepconv1_pw"},
            {"source_suffix": "sepconv1_pw", "target_suffix": "bn1"},
            {"source_suffix": "bn1", "target_suffix": "relu1"},
            {"source_suffix": "relu1", "target_suffix": "sepconv2_dw"},
            {"source_suffix": "sepconv2_dw", "target_suffix": "sepconv2_pw"},
            {"source_suffix": "sepconv2_pw", "target_suffix": "bn2"},
            {"source_suffix": "bn2", "target_suffix": "relu2"},
            {"source_suffix": "relu2", "target_suffix": "add"},
            {"source_suffix": "proj", "target_suffix": "add"},
        ],
        input_expectation="rank-4 tensor [batch, channels, h, w]",
        aliases=["xception_separable_block", "xception"],
    ),
    _template(
        template_id="shufflenet_v2_unit",
        name="ShuffleNetV2 Unit",
        family="cnn",
        nodes=[
            {"suffix": "split_a", "layer_op": "nn.Identity", "default_params": {}},
            {"suffix": "split_b", "layer_op": "nn.Identity", "default_params": {}},
            {"suffix": "pw1", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
            {"suffix": "dw", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1, "groups": 1}},
            {"suffix": "pw2", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
            {"suffix": "concat", "layer_op": "torch.cat", "default_params": {"dim": 1}},
            {"suffix": "shuffle", "layer_op": "channel_shuffle", "default_params": {}},
        ],
        edges=[
            {"source_suffix": "split_b", "target_suffix": "pw1"},
            {"source_suffix": "pw1", "target_suffix": "dw"},
            {"source_suffix": "dw", "target_suffix": "pw2"},
            {"source_suffix": "split_a", "target_suffix": "concat"},
            {"source_suffix": "pw2", "target_suffix": "concat"},
            {"source_suffix": "concat", "target_suffix": "shuffle"},
        ],
        input_expectation="rank-4 tensor [batch, channels, h, w]",
        aliases=["shufflenetv2_block", "shufflenet_v2"],
    ),
    _template(
        template_id="hrnet_fusion_block",
        name="HRNet Multi-Resolution Fusion Block",
        family="cnn",
        nodes=[
            {"suffix": "high_branch", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            {"suffix": "mid_branch", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            {"suffix": "low_branch", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            {"suffix": "up_low", "layer_op": "nn.Upsample", "default_params": {"size": [224, 224], "mode": "bilinear"}},
            {"suffix": "up_mid", "layer_op": "nn.Upsample", "default_params": {"size": [224, 224], "mode": "bilinear"}},
            {"suffix": "fuse_high", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "fuse_mid", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "fuse_low", "layer_op": "torch.add", "default_params": {}},
        ],
        edges=[
            {"source_suffix": "low_branch", "target_suffix": "up_low"},
            {"source_suffix": "mid_branch", "target_suffix": "up_mid"},
            {"source_suffix": "high_branch", "target_suffix": "fuse_high"},
            {"source_suffix": "up_mid", "target_suffix": "fuse_high"},
            {"source_suffix": "mid_branch", "target_suffix": "fuse_mid"},
            {"source_suffix": "up_low", "target_suffix": "fuse_mid"},
            {"source_suffix": "low_branch", "target_suffix": "fuse_low"},
            {"source_suffix": "mid_branch", "target_suffix": "fuse_low"},
        ],
        input_expectation="rank-4 tensor [batch, channels, h, w]",
        aliases=["hrnet_block", "high_resolution_fusion"],
    ),
    _template(
        template_id="repvgg_block",
        name="RepVGG Re-Parameterizable Block",
        family="cnn",
        nodes=[
            {"suffix": "conv3x3", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            {"suffix": "conv1x1", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
            {"suffix": "identity", "layer_op": "identity", "default_params": {}},
            {"suffix": "add1", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "add2", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "relu", "layer_op": "nn.ReLU", "default_params": {}},
        ],
        edges=[
            {"source_suffix": "conv3x3", "target_suffix": "add1"},
            {"source_suffix": "conv1x1", "target_suffix": "add1"},
            {"source_suffix": "add1", "target_suffix": "add2"},
            {"source_suffix": "identity", "target_suffix": "add2"},
            {"source_suffix": "add2", "target_suffix": "relu"},
        ],
        input_expectation="rank-4 tensor [batch, channels, h, w]",
        aliases=["repvgg", "reparam_conv_block"],
    ),
    _template(
        template_id="faster_rcnn_head_block",
        name="Faster R-CNN Head Block",
        family="detection",
        nodes=[
            {"suffix": "roi_align", "layer_op": "roi_align", "default_params": {}},
            {"suffix": "fc1", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "relu1", "layer_op": "nn.ReLU", "default_params": {}},
            {"suffix": "fc2", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "relu2", "layer_op": "nn.ReLU", "default_params": {}},
            {"suffix": "cls_head", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "box_head", "layer_op": "nn.Linear", "default_params": {}},
        ],
        edges=[
            {"source_suffix": "roi_align", "target_suffix": "fc1"},
            {"source_suffix": "fc1", "target_suffix": "relu1"},
            {"source_suffix": "relu1", "target_suffix": "fc2"},
            {"source_suffix": "fc2", "target_suffix": "relu2"},
            {"source_suffix": "relu2", "target_suffix": "cls_head"},
            {"source_suffix": "relu2", "target_suffix": "box_head"},
        ],
        input_expectation="rank-4 region features [batch, channels, h, w]",
        aliases=["faster_rcnn_head", "frcnn_head"],
    ),
    _template(
        template_id="mask_rcnn_head_block",
        name="Mask R-CNN Mask Head Block",
        family="detection",
        nodes=[
            {"suffix": "roi_align", "layer_op": "roi_align", "default_params": {}},
            {"suffix": "conv1", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            {"suffix": "relu1", "layer_op": "nn.ReLU", "default_params": {}},
            {"suffix": "conv2", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            {"suffix": "relu2", "layer_op": "nn.ReLU", "default_params": {}},
            {"suffix": "deconv", "layer_op": "nn.ConvTranspose2d", "default_params": {"kernel_size": 2, "stride": 2}},
            {"suffix": "mask_logits", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
        ],
        input_expectation="rank-4 roi features [batch, channels, h, w]",
        aliases=["mask_rcnn_head", "mask_head"],
    ),
    _template(
        template_id="ssd_head_block",
        name="SSD Detection Head Block",
        family="detection",
        nodes=[
            {"suffix": "cls_conv", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            {"suffix": "box_conv", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            # Flatten, not reshape. These meant "flatten the spatial axes so the
            # two heads can be concatenated on dim 1", but a param-less reshape
            # cannot express that: it has no target shape, so its output shape was
            # unknowable AND codegen emitted `torch.reshape(x)`, which raises
            # TypeError for a missing `shape`. nn.Flatten(start_dim=2) is the same
            # operation, derivable and runnable: (N,C,H,W) -> (N,C,H*W).
            {"suffix": "reshape_cls", "layer_op": "nn.Flatten", "default_params": {"start_dim": 2}},
            {"suffix": "reshape_box", "layer_op": "nn.Flatten", "default_params": {"start_dim": 2}},
            {"suffix": "concat", "layer_op": "torch.cat", "default_params": {"dim": 1}},
        ],
        edges=[
            {"source_suffix": "cls_conv", "target_suffix": "reshape_cls"},
            {"source_suffix": "box_conv", "target_suffix": "reshape_box"},
            {"source_suffix": "reshape_cls", "target_suffix": "concat"},
            {"source_suffix": "reshape_box", "target_suffix": "concat"},
        ],
        input_expectation="rank-4 feature maps [batch, channels, h, w]",
        aliases=["ssd_head", "multibox_head"],
    ),
    _template(
        template_id="yolov5_c3_block",
        name="YOLOv5 C3 Block",
        family="detection",
        nodes=[
            {"suffix": "cv1", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
            {"suffix": "cv2", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
            {"suffix": "bottleneck1", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            {"suffix": "bottleneck2", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            {"suffix": "concat", "layer_op": "torch.cat", "default_params": {"dim": 1}},
            {"suffix": "cv3", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
        ],
        edges=[
            {"source_suffix": "cv1", "target_suffix": "bottleneck1"},
            {"source_suffix": "bottleneck1", "target_suffix": "bottleneck2"},
            {"source_suffix": "bottleneck2", "target_suffix": "concat"},
            {"source_suffix": "cv2", "target_suffix": "concat"},
            {"source_suffix": "concat", "target_suffix": "cv3"},
        ],
        input_expectation="rank-4 tensor [batch, channels, h, w]",
        aliases=["yolov5_c3", "csp_c3_block"],
    ),
    _template(
        template_id="yolov8_c2f_block",
        name="YOLOv8 C2f Block",
        family="detection",
        nodes=[
            {"suffix": "cv1", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
            {"suffix": "split", "layer_op": "channel_split", "default_params": {}},
            {"suffix": "b1", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            {"suffix": "b2", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}},
            {"suffix": "concat", "layer_op": "torch.cat", "default_params": {"dim": 1}},
            {"suffix": "cv2", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
        ],
        edges=[
            {"source_suffix": "cv1", "target_suffix": "split"},
            {"source_suffix": "split", "target_suffix": "b1"},
            {"source_suffix": "b1", "target_suffix": "b2"},
            {"source_suffix": "split", "target_suffix": "concat"},
            {"source_suffix": "b1", "target_suffix": "concat"},
            {"source_suffix": "b2", "target_suffix": "concat"},
            {"source_suffix": "concat", "target_suffix": "cv2"},
        ],
        input_expectation="rank-4 tensor [batch, channels, h, w]",
        aliases=["yolov8_c2f", "c2f_block"],
    ),
    _template(
        template_id="deformable_detr_block",
        name="Deformable DETR Decoder Block",
        family="detection",
        nodes=[
            {"suffix": "query_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "ms_deform_attn", "layer_op": "multi_scale_deformable_attention", "default_params": {}},
            {"suffix": "add1", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "ffn1", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "relu", "layer_op": "nn.ReLU", "default_params": {}},
            {"suffix": "ffn2", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "add2", "layer_op": "torch.add", "default_params": {}},
        ],
        input_expectation="rank-3 tensor [batch, queries, d_model]",
        aliases=["deformable_detr", "ms_deformable_detr_block"],
    ),
    _template(
        template_id="bifpn_block",
        name="BiFPN Weighted Fusion Block",
        family="detection",
        nodes=[
            {"suffix": "p3", "layer_op": "identity", "default_params": {}},
            {"suffix": "p4", "layer_op": "identity", "default_params": {}},
            {"suffix": "p5", "layer_op": "identity", "default_params": {}},
            {"suffix": "up_p5", "layer_op": "nn.Upsample", "default_params": {"scale_factor": 2}},
            {"suffix": "w_add_p4", "layer_op": "weighted_add", "default_params": {}},
            {"suffix": "up_p4", "layer_op": "nn.Upsample", "default_params": {"scale_factor": 2}},
            {"suffix": "w_add_p3", "layer_op": "weighted_add", "default_params": {}},
            {"suffix": "down_p3", "layer_op": "nn.MaxPool2d", "default_params": {"kernel_size": 2, "stride": 2}},
            {"suffix": "w_add_td_p4", "layer_op": "weighted_add", "default_params": {}},
            {"suffix": "down_p4", "layer_op": "nn.MaxPool2d", "default_params": {"kernel_size": 2, "stride": 2}},
            {"suffix": "w_add_td_p5", "layer_op": "weighted_add", "default_params": {}},
        ],
        edges=[
            {"source_suffix": "p5", "target_suffix": "up_p5"},
            {"source_suffix": "up_p5", "target_suffix": "w_add_p4"},
            {"source_suffix": "p4", "target_suffix": "w_add_p4"},
            {"source_suffix": "w_add_p4", "target_suffix": "up_p4"},
            {"source_suffix": "up_p4", "target_suffix": "w_add_p3"},
            {"source_suffix": "p3", "target_suffix": "w_add_p3"},
            {"source_suffix": "w_add_p3", "target_suffix": "down_p3"},
            {"source_suffix": "down_p3", "target_suffix": "w_add_td_p4"},
            {"source_suffix": "w_add_p4", "target_suffix": "w_add_td_p4"},
            {"source_suffix": "w_add_td_p4", "target_suffix": "down_p4"},
            {"source_suffix": "down_p4", "target_suffix": "w_add_td_p5"},
            {"source_suffix": "p5", "target_suffix": "w_add_td_p5"},
        ],
        input_expectation="multi-scale rank-4 pyramid features",
        aliases=["bifpn_fusion_block", "efficientdet_bifpn"],
    ),
    _template(
        template_id="gpt2_decoder_block",
        name="GPT-2 Decoder Block",
        family="transformer",
        nodes=[
            {"suffix": "ln1", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "self_attn", "layer_op": "nn.MultiheadAttention", "default_params": {"batch_first": True}},
            {"suffix": "add1", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "ln2", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "mlp_fc", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "gelu", "layer_op": "nn.GELU", "default_params": {}},
            {"suffix": "mlp_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "add2", "layer_op": "torch.add", "default_params": {}},
        ],
        input_expectation="rank-3 tensor [batch, tokens, d_model]",
        aliases=["gpt2_block", "gpt2_decoder"],
    ),
    _template(
        template_id="gpt_neox_block",
        name="GPT-NeoX Decoder Block",
        family="transformer",
        nodes=[
            {"suffix": "ln1", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "attn", "layer_op": "nn.MultiheadAttention", "default_params": {"batch_first": True}},
            {"suffix": "add1", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "ln2", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "fc1", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "gelu", "layer_op": "nn.GELU", "default_params": {}},
            {"suffix": "fc2", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "add2", "layer_op": "torch.add", "default_params": {}},
        ],
        input_expectation="rank-3 tensor [batch, tokens, d_model]",
        aliases=["gptneox_block", "neox_decoder_block"],
    ),
    _template(
        template_id="llama2_block",
        name="LLaMA-2 Decoder Block",
        family="transformer",
        nodes=[
            {"suffix": "rms1", "layer_op": "rmsnorm", "default_params": {}},
            {"suffix": "qkv_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "rope", "layer_op": "rotary_embedding", "default_params": {}},
            {"suffix": "attn", "layer_op": "nn.MultiheadAttention", "default_params": {"batch_first": True}},
            {"suffix": "add1", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "rms2", "layer_op": "rmsnorm", "default_params": {}},
            {"suffix": "gate_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "up_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "swiglu", "layer_op": "glu_activation", "default_params": {}},
            {"suffix": "down_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "add2", "layer_op": "torch.add", "default_params": {}},
        ],
        input_expectation="rank-3 tensor [batch, tokens, d_model]",
        aliases=["llama_block", "llama2_decoder"],
    ),
    _template(
        template_id="mistral_block",
        name="Mistral Decoder Block",
        family="transformer",
        nodes=[
            {"suffix": "rms1", "layer_op": "rmsnorm", "default_params": {}},
            {"suffix": "qkv", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "sw_attn", "layer_op": "sliding_window_attention", "default_params": {"window": 4096}},
            {"suffix": "add1", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "rms2", "layer_op": "rmsnorm", "default_params": {}},
            {"suffix": "ffn1", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "silu", "layer_op": "nn.SiLU", "default_params": {}},
            {"suffix": "ffn2", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "add2", "layer_op": "torch.add", "default_params": {}},
        ],
        input_expectation="rank-3 tensor [batch, tokens, d_model]",
        aliases=["mistral_decoder", "mistral_sliding_window_block"],
    ),
    _template(
        template_id="mixtral_moe_block",
        name="Mixtral MoE Decoder Block",
        family="transformer",
        nodes=[
            {"suffix": "rms1", "layer_op": "rmsnorm", "default_params": {}},
            {"suffix": "attn", "layer_op": "nn.MultiheadAttention", "default_params": {"batch_first": True}},
            {"suffix": "add1", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "rms2", "layer_op": "rmsnorm", "default_params": {}},
            {"suffix": "router", "layer_op": "topk_router", "default_params": {"k": 2}},
            {"suffix": "expert1", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "expert2", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "merge", "layer_op": "mixture_merge", "default_params": {}},
            {"suffix": "add2", "layer_op": "torch.add", "default_params": {}},
        ],
        input_expectation="rank-3 tensor [batch, tokens, d_model]",
        aliases=["mixtral_block", "mixtral_decoder_block"],
    ),
    _template(
        template_id="qwen_block",
        name="Qwen Decoder Block",
        family="transformer",
        nodes=[
            {"suffix": "rms1", "layer_op": "rmsnorm", "default_params": {}},
            {"suffix": "attn_qkv", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "attn", "layer_op": "nn.MultiheadAttention", "default_params": {"batch_first": True}},
            {"suffix": "add1", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "rms2", "layer_op": "rmsnorm", "default_params": {}},
            {"suffix": "mlp1", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "silu", "layer_op": "nn.SiLU", "default_params": {}},
            {"suffix": "mlp2", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "add2", "layer_op": "torch.add", "default_params": {}},
        ],
        input_expectation="rank-3 tensor [batch, tokens, d_model]",
        aliases=["qwen_decoder", "qwen_transformer_block"],
    ),
    _template(
        template_id="deberta_block",
        name="DeBERTa Encoder Block",
        family="transformer",
        nodes=[
            {"suffix": "ln1", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "dis_attn", "layer_op": "disentangled_attention", "default_params": {}},
            {"suffix": "add1", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "ln2", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "ffn1", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "gelu", "layer_op": "nn.GELU", "default_params": {}},
            {"suffix": "ffn2", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "add2", "layer_op": "torch.add", "default_params": {}},
        ],
        input_expectation="rank-3 tensor [batch, tokens, d_model]",
        aliases=["deberta_encoder_block", "deberta_transformer_block"],
    ),
    _template(
        template_id="electra_block",
        name="ELECTRA Encoder Block",
        family="transformer",
        nodes=[
            {"suffix": "ln1", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "attn", "layer_op": "nn.MultiheadAttention", "default_params": {"batch_first": True}},
            {"suffix": "add1", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "ln2", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "ffn1", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "gelu", "layer_op": "nn.GELU", "default_params": {}},
            {"suffix": "ffn2", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "add2", "layer_op": "torch.add", "default_params": {}},
        ],
        input_expectation="rank-3 tensor [batch, tokens, d_model]",
        aliases=["electra_encoder_block", "electra_transformer_block"],
    ),
    _template(
        template_id="t5_decoder_block",
        name="T5 Decoder Block",
        family="transformer",
        nodes=[
            {"suffix": "ln1", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "self_attn", "layer_op": "nn.MultiheadAttention", "default_params": {"batch_first": True}},
            {"suffix": "add1", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "ln2", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "cross_attn", "layer_op": "nn.MultiheadAttention", "default_params": {"batch_first": True}},
            {"suffix": "add2", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "ln3", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "ffn1", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "relu", "layer_op": "nn.ReLU", "default_params": {}},
            {"suffix": "ffn2", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "add3", "layer_op": "torch.add", "default_params": {}},
        ],
        input_expectation="rank-3 tensor [batch, tokens, d_model]",
        aliases=["t5_decoder", "t5_cross_attention_block"],
    ),
    _template(
        template_id="bart_block",
        name="BART Encoder-Decoder Block",
        family="transformer",
        nodes=[
            {"suffix": "enc_ln", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "enc_attn", "layer_op": "nn.MultiheadAttention", "default_params": {"batch_first": True}},
            {"suffix": "enc_add", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "dec_ln1", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "dec_self_attn", "layer_op": "nn.MultiheadAttention", "default_params": {"batch_first": True}},
            {"suffix": "dec_add1", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "dec_ln2", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "dec_cross_attn", "layer_op": "nn.MultiheadAttention", "default_params": {"batch_first": True}},
            {"suffix": "dec_add2", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "ffn", "layer_op": "nn.Linear", "default_params": {}},
        ],
        input_expectation="rank-3 tensor [batch, tokens, d_model]",
        aliases=["bart_encoder_decoder_block", "bart_transformer_block"],
    ),
    _template(
        template_id="linformer_block",
        name="Linformer Attention Block",
        family="transformer",
        nodes=[
            {"suffix": "ln1", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "q_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "k_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "v_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "k_reduce", "layer_op": "seq_projection", "default_params": {}},
            {"suffix": "v_reduce", "layer_op": "seq_projection", "default_params": {}},
            {"suffix": "attn", "layer_op": "scaled_dot_attention", "default_params": {}},
            {"suffix": "out_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "add1", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "ln2", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "ffn", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "add2", "layer_op": "torch.add", "default_params": {}},
        ],
        input_expectation="rank-3 tensor [batch, tokens, d_model]",
        aliases=["linformer_attention_block", "linformer"],
    ),
    _template(
        template_id="performer_block",
        name="Performer Attention Block",
        family="transformer",
        nodes=[
            {"suffix": "ln1", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "qkv_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "feature_map", "layer_op": "kernel_feature_map", "default_params": {}},
            {"suffix": "linear_attn", "layer_op": "linear_attention", "default_params": {}},
            {"suffix": "out_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "add1", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "ln2", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "ffn1", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "gelu", "layer_op": "nn.GELU", "default_params": {}},
            {"suffix": "ffn2", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "add2", "layer_op": "torch.add", "default_params": {}},
        ],
        input_expectation="rank-3 tensor [batch, tokens, d_model]",
        aliases=["performer_attention_block", "performer"],
    ),
    _template(
        template_id="bigbird_block",
        name="BigBird Sparse Attention Block",
        family="transformer",
        nodes=[
            {"suffix": "ln1", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "qkv_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "block_sparse_attn", "layer_op": "block_sparse_attention", "default_params": {}},
            {"suffix": "global_attn", "layer_op": "global_attention", "default_params": {}},
            {"suffix": "random_attn", "layer_op": "random_attention", "default_params": {}},
            {"suffix": "merge", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "out_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "add1", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "ln2", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "ffn", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "add2", "layer_op": "torch.add", "default_params": {}},
        ],
        input_expectation="rank-3 tensor [batch, tokens, d_model]",
        aliases=["bigbird_sparse_block", "bigbird_attention"],
    ),
    _template(
        template_id="flash_attention_block",
        name="FlashAttention Block",
        family="transformer",
        nodes=[
            {"suffix": "ln1", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "qkv_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "flash_attn", "layer_op": "flash_attention", "default_params": {}},
            {"suffix": "out_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "add1", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "ln2", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "mlp1", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "silu", "layer_op": "nn.SiLU", "default_params": {}},
            {"suffix": "mlp2", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "add2", "layer_op": "torch.add", "default_params": {}},
        ],
        input_expectation="rank-3 tensor [batch, tokens, d_model]",
        aliases=["flashattn_block", "flash_attention"],
    ),
    _template(
        template_id="multi_query_attention_block",
        name="Multi-Query Attention Block",
        family="transformer",
        nodes=[
            {"suffix": "ln1", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "q_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "k_proj_shared", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "v_proj_shared", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "mqa_attn", "layer_op": "multi_query_attention", "default_params": {}},
            {"suffix": "out_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "add1", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "ln2", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "ffn", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "add2", "layer_op": "torch.add", "default_params": {}},
        ],
        input_expectation="rank-3 tensor [batch, tokens, d_model]",
        aliases=["mqa_block", "multi_query_attention"],
    ),
    _template(
        template_id="wav2vec2_encoder_block",
        name="Wav2Vec2 Encoder Block",
        family="speech",
        nodes=[
            {"suffix": "feature_extractor", "layer_op": "nn.Conv1d", "default_params": {"kernel_size": 10, "stride": 5}},
            {"suffix": "layer_norm", "layer_op": "nn.BatchNorm1d", "default_params": {}},
            {"suffix": "pos_conv_embed", "layer_op": "nn.Conv1d", "default_params": {"kernel_size": 5, "padding": 2, "groups": 16}},
            {"suffix": "transformer_enc", "layer_op": "speech_transformer_attention", "default_params": {"batch_first": True}},
            {"suffix": "ffn", "layer_op": "nn.Conv1d", "default_params": {"kernel_size": 1}},
            {"suffix": "quantizer", "layer_op": "vector_quantizer", "default_params": {}},
            {"suffix": "proj", "layer_op": "nn.Conv1d", "default_params": {"kernel_size": 1}},
        ],
        input_expectation="rank-3 waveform tensor [batch, channels, time]",
        training_minimums=["feature_dim > 0", "num_heads >= 1", "hidden_dim % num_heads == 0"],
        aliases=["wav2vec2_block", "wav2vec_2_0_block"],
    ),
    _template(
        template_id="hubert_encoder_block",
        name="HuBERT Encoder Block",
        family="speech",
        nodes=[
            {"suffix": "feature_extractor", "layer_op": "nn.Conv1d", "default_params": {"kernel_size": 10, "stride": 5}},
            {"suffix": "layer_norm", "layer_op": "nn.BatchNorm1d", "default_params": {}},
            {"suffix": "mask_embed", "layer_op": "nn.Identity", "default_params": {}},
            {"suffix": "transformer_enc", "layer_op": "speech_transformer_attention", "default_params": {"batch_first": True}},
            {"suffix": "proj", "layer_op": "nn.Conv1d", "default_params": {"kernel_size": 1}},
            {"suffix": "pred_head", "layer_op": "nn.Conv1d", "default_params": {"kernel_size": 1}},
        ],
        input_expectation="rank-3 waveform tensor [batch, channels, time]",
        training_minimums=["hidden_dim > 0", "codebook_size > 0"],
        aliases=["hubert_block", "hidden_unit_bert_block"],
    ),
    _template(
        template_id="whisper_encoder_decoder_block",
        name="Whisper Encoder-Decoder Block",
        family="speech",
        nodes=[
            {"suffix": "logmel_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "enc_attn", "layer_op": "nn.MultiheadAttention", "default_params": {"batch_first": True}},
            {"suffix": "enc_ffn", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "dec_self_attn", "layer_op": "nn.MultiheadAttention", "default_params": {"batch_first": True}},
            {"suffix": "dec_cross_attn", "layer_op": "nn.MultiheadAttention", "default_params": {"batch_first": True}},
            {"suffix": "dec_ffn", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "lm_head", "layer_op": "nn.Linear", "default_params": {}},
        ],
        input_expectation="rank-3 log-mel tensor [batch, frames, mel_bins]",
        training_minimums=["d_model > 0", "vocab_size > 0", "num_heads >= 1"],
        aliases=["whisper_block", "asr_encoder_decoder_block"],
    ),
    _template(
        template_id="clap_dual_encoder_block",
        name="CLAP Dual Encoder Block",
        family="multimodal",
        nodes=[
            {"suffix": "audio_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "audio_norm", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "text_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "text_norm", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "similarity", "layer_op": "dot_product_similarity", "default_params": {}},
            {"suffix": "temp_scale", "layer_op": "temperature_scale", "default_params": {}},
        ],
        input_expectation="paired audio/text embeddings",
        training_minimums=["audio_dim == text_dim", "temperature > 0"],
        aliases=["clap_block", "audio_text_contrastive_block"],
    ),
    _template(
        template_id="perceiver_resampler_block",
        name="Perceiver Resampler Block",
        family="multimodal",
        nodes=[
            {"suffix": "input_proj", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "latent_array", "layer_op": "learned_latents", "default_params": {}},
            {"suffix": "cross_attn", "layer_op": "nn.MultiheadAttention", "default_params": {"batch_first": True}},
            {"suffix": "latent_ffn", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "add", "layer_op": "torch.add", "default_params": {}},
        ],
        input_expectation="rank-3 tensor [batch, tokens, dim]",
        training_minimums=["num_latents > 0", "latent_dim > 0", "num_heads >= 1"],
        aliases=["perceiver_block", "perceiver_io_block", "resampler_block"],
    ),
    _template(
        template_id="qformer_fusion_block",
        name="Q-Former Fusion Block",
        family="multimodal",
        nodes=[
            {"suffix": "query_tokens", "layer_op": "nn.Embedding", "default_params": {}},
            {"suffix": "self_attn", "layer_op": "nn.MultiheadAttention", "default_params": {"batch_first": True}},
            {"suffix": "cross_attn", "layer_op": "nn.MultiheadAttention", "default_params": {"batch_first": True}},
            {"suffix": "mlp", "layer_op": "nn.Linear", "default_params": {}},
            {"suffix": "out_proj", "layer_op": "nn.Linear", "default_params": {}},
        ],
        input_expectation="query tokens + encoder tokens",
        training_minimums=["num_query_tokens > 0", "query_dim > 0", "encoder_dim > 0"],
        aliases=["qformer_block", "blip2_qformer_block"],
    ),
    _template(
        template_id="flamingo_gated_cross_attn_block",
        name="Flamingo Gated Cross-Attention Block",
        family="multimodal",
        nodes=[
            {"suffix": "lang_norm", "layer_op": "nn.LayerNorm", "default_params": {}},
            {"suffix": "cross_attn", "layer_op": "nn.MultiheadAttention", "default_params": {"batch_first": True}},
            {"suffix": "gate", "layer_op": "nn.Sigmoid", "default_params": {}},
            {"suffix": "gated_add", "layer_op": "torch.add", "default_params": {}},
            {"suffix": "ffn", "layer_op": "nn.Linear", "default_params": {}},
        ],
        input_expectation="language tokens with visual context tokens",
        training_minimums=["d_model > 0", "vision_dim > 0", "num_heads >= 1"],
        aliases=["flamingo_block", "gated_cross_attention_block"],
    ),
]


def _build_programmatic_variants() -> List[Dict[str, Any]]:
    variants: List[Dict[str, Any]] = []

    for norm in ["layernorm", "rmsnorm"]:
        for act in ["relu", "gelu", "silu", "tanh"]:
            for heads in [4, 8, 12, 16]:
                norm_layer = "nn.LayerNorm" if norm == "layernorm" else "rmsnorm"
                if act == "relu":
                    act_layer = "nn.ReLU"
                elif act == "gelu":
                    act_layer = "nn.GELU"
                elif act == "silu":
                    act_layer = "nn.SiLU"
                else:
                    act_layer = "nn.Tanh"
                template_id = f"tfm_enc_{norm}_{act}_h{heads}_block"
                variants.append(
                    _template(
                        template_id=template_id,
                        name=f"Transformer Encoder {norm}/{act}/h{heads}",
                        family="transformer",
                        nodes=[
                            {"suffix": "norm1", "layer_op": norm_layer, "default_params": {}},
                            {
                                "suffix": "attn",
                                "layer_op": "nn.MultiheadAttention",
                                "default_params": {"batch_first": True, "num_heads": heads},
                            },
                            {"suffix": "add1", "layer_op": "torch.add", "default_params": {}},
                            {"suffix": "norm2", "layer_op": norm_layer, "default_params": {}},
                            {"suffix": "ff1", "layer_op": "nn.Linear", "default_params": {}},
                            {"suffix": "act", "layer_op": act_layer, "default_params": {}},
                            {"suffix": "ff2", "layer_op": "nn.Linear", "default_params": {}},
                            {"suffix": "add2", "layer_op": "torch.add", "default_params": {}},
                        ],
                        input_expectation="rank-3 tensor [batch, tokens, d_model]",
                        training_minimums=["d_model % num_heads == 0", "ffn_dim >= d_model"],
                        allowed_residual_patterns=["post-attn residual", "post-mlp residual"],
                        aliases=[template_id.replace("_block", ""), template_id.replace("_", "-")],
                    )
                )

    for depth in [2, 3, 4, 5, 6]:
        for kernel in [3, 5, 7]:
            for residual in [False, True]:
                template_id = f"cnn_conv{depth}_k{kernel}_{'res' if residual else 'plain'}_block"
                nodes: List[Dict[str, Any]] = []
                for idx in range(depth):
                    nodes.extend(
                        [
                            {
                                "suffix": f"conv{idx+1}",
                                "layer_op": "nn.Conv2d",
                                "default_params": {"kernel_size": kernel, "padding": kernel // 2},
                            },
                            {"suffix": f"bn{idx+1}", "layer_op": "nn.BatchNorm2d", "default_params": {}},
                            {"suffix": f"relu{idx+1}", "layer_op": "nn.ReLU", "default_params": {}},
                        ]
                    )
                if residual:
                    nodes.append({"suffix": "add", "layer_op": "torch.add", "default_params": {}})
                variants.append(
                    _template(
                        template_id=template_id,
                        name=f"CNN Conv{depth} k{kernel} {'Residual' if residual else 'Plain'}",
                        family="cnn",
                        nodes=nodes,
                        input_expectation="rank-4 tensor [batch, channels, h, w]",
                        training_minimums=["in_channels > 0", "out_channels > 0"],
                        allowed_residual_patterns=["identity shortcut"] if residual else [],
                        aliases=[template_id.replace("_block", "")],
                    )
                )

    for patch in [8, 14, 16, 32]:
        for heads in [8, 12, 16]:
            template_id = f"vit_patch{patch}_h{heads}_block"
            variants.append(
                _template(
                    template_id=template_id,
                    name=f"ViT Patch{patch} Heads{heads}",
                    family="vision_transformer",
                    nodes=[
                        {"suffix": "patch_embed", "layer_op": "nn.Linear", "default_params": {}},
                        {"suffix": "norm1", "layer_op": "nn.LayerNorm", "default_params": {}},
                        {
                            "suffix": "attn",
                            "layer_op": "nn.MultiheadAttention",
                            "default_params": {"batch_first": True, "num_heads": heads},
                        },
                        {"suffix": "add1", "layer_op": "torch.add", "default_params": {}},
                        {"suffix": "norm2", "layer_op": "nn.LayerNorm", "default_params": {}},
                        {"suffix": "mlp1", "layer_op": "nn.Linear", "default_params": {}},
                        {"suffix": "gelu", "layer_op": "nn.GELU", "default_params": {}},
                        {"suffix": "mlp2", "layer_op": "nn.Linear", "default_params": {}},
                        {"suffix": "add2", "layer_op": "torch.add", "default_params": {}},
                    ],
                    input_expectation="rank-3 tensor [batch, tokens, embed_dim]",
                    training_minimums=[f"patch_size == {patch}", "embed_dim % num_heads == 0"],
                    allowed_residual_patterns=["attn residual", "mlp residual"],
                    aliases=[template_id.replace("_block", ""), f"vit-{patch}"],
                )
            )

    for aggr in ["mean", "sum", "max", "attention", "lstm"]:
        for hops in [1, 2, 3, 4, 5]:
            template_id = f"gnn_{aggr}_{hops}hop_block"
            if aggr == "attention":
                aggr_layer = "attention_aggregate"
            elif aggr == "lstm":
                aggr_layer = "lstm_aggregate"
            else:
                aggr_layer = f"{aggr}_aggregate"
            variants.append(
                _template(
                    template_id=template_id,
                    name=f"GNN {aggr.title()} {hops}-Hop Block",
                    family="gnn",
                    nodes=[
                        {"suffix": "sample", "layer_op": "neighbor_sampler", "default_params": {"hops": hops}},
                        {"suffix": "message", "layer_op": "message_mlp", "default_params": {}},
                        {"suffix": "aggregate", "layer_op": aggr_layer, "default_params": {}},
                        {"suffix": "update", "layer_op": "nn.Linear", "default_params": {}},
                        {"suffix": "relu", "layer_op": "nn.ReLU", "default_params": {}},
                    ],
                    input_expectation="rank-2 node features [num_nodes, features]",
                    aliases=[template_id.replace("_block", "")],
                )
            )

    for cell in ["lstm", "gru", "rnn"]:
        for layers in [2, 4, 6, 8]:
            for bidirectional in [False, True]:
                if cell == "lstm":
                    layer_op = "nn.LSTM"
                elif cell == "gru":
                    layer_op = "nn.GRU"
                else:
                    layer_op = "nn.RNN"
                template_id = f"{cell}_{layers}layer_{'bi' if bidirectional else 'uni'}_block"
                variants.append(
                    _template(
                        template_id=template_id,
                        name=f"{cell.upper()} {layers}-Layer {'Bi' if bidirectional else 'Uni'}",
                        family="sequence",
                        nodes=[
                            {
                                "suffix": "rnn",
                                "layer_op": layer_op,
                                "default_params": {
                                    "num_layers": layers,
                                    "batch_first": True,
                                    "bidirectional": bidirectional,
                                },
                            },
                            {"suffix": "dropout", "layer_op": "nn.Dropout", "default_params": {"p": 0.1}},
                            {"suffix": "proj", "layer_op": "nn.Linear", "default_params": {}},
                        ],
                        input_expectation="rank-3 tensor [batch, sequence, features]",
                        training_minimums=["input_size > 0", "hidden_size > 0", "num_layers >= 1"],
                        aliases=[template_id.replace("_block", "")],
                    )
                )

    for width in [32, 64, 128, 256]:
        for depth in [2, 3, 4]:
            template_id = f"unet_w{width}_d{depth}_block"
            nodes: List[Dict[str, Any]] = []
            for idx in range(depth):
                nodes.append(
                    {
                        "suffix": f"enc{idx+1}",
                        "layer_op": "nn.Conv2d",
                        "default_params": {"kernel_size": 3, "padding": 1, "out_channels": width * (2**idx)},
                    }
                )
                nodes.append(
                    {"suffix": f"pool{idx+1}", "layer_op": "nn.MaxPool2d", "default_params": {"kernel_size": 2}}
                )
            nodes.append(
                {"suffix": "bottleneck", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}}
            )
            for idx in range(depth, 0, -1):
                nodes.append(
                    {
                        "suffix": f"up{idx}",
                        "layer_op": "nn.ConvTranspose2d",
                        "default_params": {"kernel_size": 2, "stride": 2},
                    }
                )
                nodes.append({"suffix": f"cat{idx}", "layer_op": "torch.cat", "default_params": {"dim": 1}})
                nodes.append(
                    {
                        "suffix": f"dec{idx}",
                        "layer_op": "nn.Conv2d",
                        "default_params": {"kernel_size": 3, "padding": 1},
                    }
                )
            variants.append(
                _template(
                    template_id=template_id,
                    name=f"UNet Width{width} Depth{depth}",
                    family="segmentation",
                    nodes=nodes,
                    input_expectation="rank-4 tensor [batch, channels, h, w]",
                    aliases=[template_id.replace("_block", ""), f"unet-{width}-{depth}"],
                )
            )

    for channels in [64, 128, 256, 512]:
        for blocks in [2, 4, 6]:
            template_id = f"diffusion_unet_c{channels}_b{blocks}_block"
            nodes: List[Dict[str, Any]] = []
            for idx in range(blocks):
                nodes.extend(
                    [
                        {
                            "suffix": f"res{idx+1}",
                            "layer_op": "nn.Conv2d",
                            "default_params": {
                                "kernel_size": 3,
                                "padding": 1,
                                "out_channels": channels,
                            },
                        },
                        {"suffix": f"norm{idx+1}", "layer_op": "nn.GroupNorm", "default_params": {}},
                        {"suffix": f"act{idx+1}", "layer_op": "nn.SiLU", "default_params": {}},
                    ]
                )
            nodes.append(
                {"suffix": "out", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 3, "padding": 1}}
            )
            variants.append(
                _template(
                    template_id=template_id,
                    name=f"Diffusion UNet C{channels} B{blocks}",
                    family="generative",
                    nodes=nodes,
                    input_expectation="rank-4 tensor [batch, channels, h, w]",
                    aliases=[template_id.replace("_block", "")],
                )
            )

    for fusion in ["concat", "add", "cross_attn", "gated"]:
        for dims in [512, 768, 1024]:
            template_id = f"multimodal_{fusion}_{dims}_block"
            if fusion == "cross_attn":
                fusion_layer = "nn.MultiheadAttention"
                fusion_params: Dict[str, Any] = {"batch_first": True}
            elif fusion == "gated":
                fusion_layer = "gated_fusion"
                fusion_params = {}
            elif fusion == "concat":
                fusion_layer = "torch.cat"
                fusion_params = {"dim": -1}
            else:
                fusion_layer = "torch.add"
                fusion_params = {}
            variants.append(
                _template(
                    template_id=template_id,
                    name=f"Multimodal {fusion} D{dims}",
                    family="multimodal",
                    nodes=[
                        {
                            "suffix": "vision_proj",
                            "layer_op": "nn.Linear",
                            "default_params": {"out_features": dims},
                        },
                        {
                            "suffix": "text_proj",
                            "layer_op": "nn.Linear",
                            "default_params": {"out_features": dims},
                        },
                        {"suffix": "fuse", "layer_op": fusion_layer, "default_params": fusion_params},
                        {"suffix": "head", "layer_op": "nn.Linear", "default_params": {"out_features": dims}},
                    ],
                    input_expectation="rank-2 or rank-3 tensor depending on fusion mode",
                    aliases=[template_id.replace("_block", "")],
                )
            )

    for detector in ["yolo", "retina", "fcos", "detr"]:
        for levels in [3, 4, 5, 6]:
            template_id = f"{detector}_fpn_l{levels}_head_block"
            nodes = [
                {
                    "suffix": f"p{idx}",
                    "layer_op": "nn.Conv2d",
                    "default_params": {"kernel_size": 3, "padding": 1},
                }
                for idx in range(3, levels + 3)
            ]
            nodes.extend(
                [
                    # The pyramid levels have to be MERGED before the heads. A Conv2d
                    # takes exactly one tensor, so wiring all L levels straight into
                    # cls_head/box_head described a graph torch cannot run: "Conv2d
                    # received 4 inputs but expects exactly 1". Concatenating on the
                    # channel axis is the fan-in the canvas expresses, the same way
                    # aspp_block's branches reach its concat.
                    {"suffix": "fuse", "layer_op": "torch.cat", "default_params": {"dim": 1}},
                    {"suffix": "cls_head", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
                    {"suffix": "box_head", "layer_op": "nn.Conv2d", "default_params": {"kernel_size": 1}},
                ]
            )
            edges = [{"source_suffix": f"p{idx}", "target_suffix": "fuse"} for idx in range(3, levels + 3)]
            edges.extend(
                [
                    {"source_suffix": "fuse", "target_suffix": "cls_head"},
                    {"source_suffix": "fuse", "target_suffix": "box_head"},
                ]
            )
            variants.append(
                _template(
                    template_id=template_id,
                    name=f"{detector.upper()} FPN L{levels} Head",
                    family="vision",
                    nodes=nodes,
                    edges=list(edges),
                    input_expectation="rank-4 tensor pyramid features",
                    aliases=[template_id.replace("_block", "")],
                )
            )

    for backbone in ["mlp", "resnet", "transformer"]:
        for fusion in ["concat", "dot", "cross"]:
            for width in [128, 256, 512]:
                template_id = f"two_tower_{backbone}_{fusion}_{width}_block"
                fusion_layer = (
                    "torch.cat" if fusion == "concat" else ("dot_product_similarity" if fusion == "dot" else "nn.MultiheadAttention")
                )
                fusion_params: Dict[str, Any] = {"dim": -1} if fusion == "concat" else ({"batch_first": True} if fusion == "cross" else {})
                variants.append(
                    _template(
                        template_id=template_id,
                        name=f"Two-Tower {backbone} {fusion} W{width}",
                        family="recsys",
                        nodes=[
                            {"suffix": "tower_a", "layer_op": "nn.Linear", "default_params": {"out_features": width}},
                            {"suffix": "tower_b", "layer_op": "nn.Linear", "default_params": {"out_features": width}},
                            {"suffix": "fusion", "layer_op": fusion_layer, "default_params": fusion_params},
                            {"suffix": "head", "layer_op": "nn.Linear", "default_params": {"out_features": width}},
                        ],
                        input_expectation="rank-2 feature tensors",
                        aliases=[template_id.replace("_block", "")],
                    )
                )

    for model in ["tabnet", "tabtransformer", "deepfm", "wide_deep"]:
        for width in [128, 256, 512]:
            template_id = f"{model}_{width}_block"
            if model == "tabnet":
                nodes = [
                    {"suffix": "feat_transform", "layer_op": "nn.Linear", "default_params": {"out_features": width}},
                    {"suffix": "attn_mask", "layer_op": "feature_mask", "default_params": {}},
                    {"suffix": "decision", "layer_op": "nn.Linear", "default_params": {"out_features": width}},
                ]
            elif model == "tabtransformer":
                nodes = [
                    {"suffix": "embed", "layer_op": "nn.Embedding", "default_params": {}},
                    {"suffix": "attn", "layer_op": "nn.MultiheadAttention", "default_params": {"batch_first": True}},
                    {"suffix": "mlp", "layer_op": "nn.Linear", "default_params": {"out_features": width}},
                ]
            elif model == "deepfm":
                nodes = [
                    {"suffix": "emb", "layer_op": "nn.Embedding", "default_params": {}},
                    {"suffix": "fm", "layer_op": "factorization_machine", "default_params": {}},
                    {"suffix": "deep", "layer_op": "nn.Linear", "default_params": {"out_features": width}},
                    {"suffix": "merge", "layer_op": "torch.add", "default_params": {}},
                ]
            else:
                nodes = [
                    {"suffix": "wide", "layer_op": "nn.Linear", "default_params": {"out_features": width}},
                    {"suffix": "deep1", "layer_op": "nn.Linear", "default_params": {"out_features": width}},
                    {"suffix": "deep2", "layer_op": "nn.ReLU", "default_params": {}},
                    {"suffix": "merge", "layer_op": "torch.add", "default_params": {}},
                ]
            variants.append(
                _template(
                    template_id=template_id,
                    name=f"{model} W{width}",
                    family="tabular",
                    nodes=nodes,
                    input_expectation="rank-2 tabular features [batch, features]",
                    aliases=[template_id.replace("_block", ""), model],
                )
            )

    unique: Dict[str, Dict[str, Any]] = {}
    for entry in variants:
        unique[str(entry["template_id"])] = entry
    return list(unique.values())


EXPANSION_TEMPLATES.extend(_build_programmatic_variants())


def register_expansion_templates(registry: Any) -> int:
    """Register expansion templates into a CanonicalTemplateRegistry instance."""
    registered = 0
    for entry in EXPANSION_TEMPLATES:
        template_id = str(entry["template_id"])
        if template_id in registry._contracts:  # noqa: SLF001
            continue
        registry.register_dynamic_template(
            template_id=template_id,
            name=str(entry["name"]),
            family=str(entry["family"]),
            nodes=list(entry["nodes"]),
            edges=list(entry["edges"]),
            required_nodes=list(entry["required_nodes"]),
            required_edges=list(entry["required_edges"]),
            allowed_residual_patterns=list(entry["allowed_residual_patterns"]),
            training_minimums=list(entry["training_minimums"]),
            input_expectation=str(entry["input_expectation"]),
            metadata=dict(entry.get("metadata") or {}),
            persist=False,
        )
        registered += 1
    return registered
