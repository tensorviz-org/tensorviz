"""Recommended complex-model roadmap for broad practical coverage."""

from typing import Any, Dict, List


def get_complex_model_roadmap() -> Dict[str, Any]:
    packs: List[Dict[str, Any]] = [
        {
            "id": "transformer_core",
            "name": "Transformer Core Pack",
            "priority": 1,
            "implemented_templates": [
                "full_transformer_block",
                "vit_block",
                "seq2seq_attention_block",
            ],
            "models": [
                "TransformerEncoderLayer (true MHA block)",
                "TransformerDecoderLayer",
                "Encoder-Decoder Transformer",
                "BERT-style encoder stack",
                "GPT-style decoder stack",
                "T5-style seq2seq stack",
            ],
            "validations": [
                "d_model divisible by nhead",
                "attention embed_dim consistency",
                "residual branch shape equality",
            ],
        },
        {
            "id": "vision_modern",
            "name": "Modern Vision Pack",
            "priority": 2,
            "implemented_templates": [
                "unet_block",
            ],
            "models": [
                "ResNet/ResNeXt family",
                "ConvNeXt",
                "Vision Transformer (ViT)",
                "Swin Transformer blocks",
                "UNet segmentation backbone",
            ],
            "validations": [
                "conv channel continuity",
                "downsample/upsample spatial consistency",
                "classification-head feature compatibility",
            ],
        },
        {
            "id": "sequence_time_series",
            "name": "Sequence & Time-Series Pack",
            "priority": 3,
            "models": [
                "Stacked LSTM/GRU",
                "Temporal Convolutional Network (TCN)",
                "Transformer time-series encoder",
                "Seq2Seq with attention",
            ],
            "validations": [
                "rank-2/rank-3 recurrent input support",
                "sequence feature-size matching",
                "decoder/encoder hidden-state compatibility",
            ],
        },
        {
            "id": "generative_core",
            "name": "Generative Core Pack",
            "priority": 4,
            "models": [
                "Variational Autoencoder (VAE)",
                "GAN (generator/discriminator pair)",
                "Diffusion UNet blocks",
                "Autoregressive token generator",
            ],
            "validations": [
                "latent bottleneck shape constraints",
                "generator/discriminator interface checks",
                "skip-connection shape checks",
            ],
        },
        {
            "id": "multimodal_retrieval",
            "name": "Multimodal & Retrieval Pack",
            "priority": 5,
            "implemented_templates": [
                "dual_encoder_fusion_block",
            ],
            "models": [
                "Dual encoder (image/text)",
                "Cross-attention fusion block",
                "Siamese embedding tower",
                "RAG embedding projector",
            ],
            "validations": [
                "cross-branch feature-dim alignment",
                "contrastive-head projection compatibility",
                "fusion-attention shape checks",
            ],
        },
        {
            "id": "graph_models",
            "name": "Graph Neural Network Pack",
            "priority": 6,
            "models": [
                "GCN",
                "GraphSAGE",
                "GAT",
                "GIN",
            ],
            "validations": [
                "node/edge feature compatibility",
                "message passing layer consistency",
            ],
        },
    ]

    return {
        "version": "2026-05-11",
        "goal": "Cover the most practically important model families used in modern production and research workflows.",
        "packs": packs,
    }
