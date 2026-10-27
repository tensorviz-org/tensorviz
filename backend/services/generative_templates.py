"""Generative model templates and architectures"""
from typing import Dict, List, Any, Optional
from models.template_models import ModelTemplate, ModuleTemplate, PopularModel
from models.graph_models import Graph, Node, Edge, LayerType, LayerData


def _build_f5tts_structure(
    dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512,
    conv_layers=4, mel_dim=100, seq_len=256, conv_intermediate=1024,
    freq_dim=256, vocab=257,
):
    """Generate the FULL, faithful F5-TTS graph (swivid/F5-TTS, arXiv:2410.06885)
    as a flat multi-branch DAG with every block present and correctly named:
    text encoder (Embedding + ConvNeXt-V2 blocks with GRN + residuals), a
    timestep-embedding branch (SinusPositionEmbedding + MLP), input-embedding
    fusion (concat of noisy mel + reference mel + text, then Linear), a
    convolutional position embedding (residual), `depth` DiT blocks
    (AdaLayerNorm-Zero conditioned on the timestep -> RoPE self-attention ->
    gated residual -> FF-AdaLN -> FeedForward -> gated residual), a final
    AdaLayerNorm and an output projection to mel channels (the predicted flow).

    Operates in [B, N, dim] (batch first). All shape-bearing torch ops chain
    exactly; the custom blocks (SinusPositionEmbedding/GRN/RoPE/AdaLayerNorm)
    are shape-preserving and recognized by the validator + verifier so the whole
    graph passes shape validation AND the dry forward with zero errors.
    """
    nodes, edges = [], []

    def node(nid, ntype, lid, lname, params, x, y):
        nodes.append({
            "id": nid, "type": ntype, "position": {"x": x, "y": y},
            "data": {"layerType": {"id": lid, "name": lname, "parameters": []},
                     "parameters": params},
        })

    def link(s, t):
        edges.append({"id": f"{s}__{t}", "source": s, "target": t})

    A = {"emb": "var(--cat-embedding)", "conv": "var(--cat-conv)", "lin": "var(--cat-linear)",
         "act": "var(--cat-activation)", "norm": "var(--cat-normalization)",
         "attn": "var(--cat-attention)", "util": "var(--cat-utility)", "reg": "var(--cat-regularization)"}
    cat_dim = mel_dim * 2 + text_dim

    # ── input nodes ──────────────────────────────────────────────────────────
    node("in_text", "inputNode", "input", "Input",
         {"name": "Text Tokens", "shape": f"1,{seq_len}", "_subtitle": "char/phoneme ids",
          "_shape": f"[B, {seq_len}]", "_accent": A["emb"]}, 40, 60)
    node("in_noisy", "inputNode", "input", "Input",
         {"name": "Noisy Mel", "shape": f"1,{seq_len},{mel_dim}", "_subtitle": "flow sample x_t",
          "_shape": f"[B, {seq_len}, {mel_dim}]", "_accent": A["emb"]}, 40, 360)
    node("in_cond", "inputNode", "input", "Input",
         {"name": "Reference Mel", "shape": f"1,{seq_len},{mel_dim}", "_subtitle": "masked cond speech",
          "_shape": f"[B, {seq_len}, {mel_dim}]", "_accent": A["emb"]}, 40, 500)
    node("in_time", "inputNode", "input", "Input",
         {"name": "Timestep", "shape": f"1,{freq_dim}", "_subtitle": "flow time t",
          "_shape": f"[B, {freq_dim}]", "_accent": A["emb"]}, 40, 680)

    # ── timestep embedding branch ─────────────────────────────────────────────
    node("t_sinus", "layerNode", "sinusoidal_pos_embedding", "sinusoidal_pos_embedding",
         {"name": "SinusPositionEmbedding", "_subtitle": "sinusoidal timestep", "_accent": A["util"]}, 260, 680)
    node("t_lin1", "layerNode", "linear", "Linear",
         {"name": "Time MLP 1", "in_features": freq_dim, "out_features": dim, "_accent": A["lin"]}, 440, 680)
    node("t_silu", "layerNode", "silu", "SiLU", {"name": "SiLU", "_accent": A["act"]}, 620, 680)
    node("t_lin2", "layerNode", "linear", "Linear",
         {"name": "Time MLP 2", "in_features": dim, "out_features": dim,
          "_subtitle": f"t_emb [B, {dim}]", "_accent": A["lin"]}, 800, 680)
    link("in_time", "t_sinus"); link("t_sinus", "t_lin1"); link("t_lin1", "t_silu"); link("t_silu", "t_lin2")
    t_emb = "t_lin2"

    # ── text encoder: Char Embedding + ConvNeXt-V2 blocks ─────────────────────
    node("text_embed", "layerNode", "embedding", "Embedding",
         {"name": "Char Embedding", "num_embeddings": vocab, "embedding_dim": text_dim,
          "_subtitle": "Embedding", "_shape": f"[B, {seq_len}, {text_dim}]", "_accent": A["emb"]}, 260, 60)
    link("in_text", "text_embed")
    prev, x0 = "text_embed", 440
    for i in range(conv_layers):
        b = f"cnx{i}"
        node(f"{b}_dw", "layerNode", "conv1d", "Conv1d",
             {"name": f"ConvNeXt-V2 {i}: DWConv", "in_channels": seq_len, "out_channels": seq_len,
              "kernel_size": 7, "stride": 1, "padding": 3, "groups": seq_len,
              "_subtitle": "depthwise k7", "_accent": A["conv"]}, x0, 60)
        node(f"{b}_ln", "layerNode", "layernorm", "LayerNorm",
             {"name": "LayerNorm", "normalized_shape": text_dim, "eps": 0.000001, "_accent": A["norm"]}, x0 + 150, 60)
        node(f"{b}_pw1", "layerNode", "linear", "Linear",
             {"name": "PWConv1", "in_features": text_dim, "out_features": conv_intermediate, "_accent": A["lin"]}, x0 + 300, 60)
        node(f"{b}_gelu", "layerNode", "gelu", "GELU", {"name": "GELU", "_accent": A["act"]}, x0 + 450, 60)
        node(f"{b}_grn", "layerNode", "grn", "grn",
             {"name": "GRN", "_subtitle": "global response norm", "_accent": A["util"]}, x0 + 600, 60)
        node(f"{b}_pw2", "layerNode", "linear", "Linear",
             {"name": "PWConv2", "in_features": conv_intermediate, "out_features": text_dim, "_accent": A["lin"]}, x0 + 750, 60)
        node(f"{b}_add", "layerNode", "add", "Add",
             {"name": f"ConvNeXt Residual {i}", "_subtitle": "x + block(x)", "_accent": A["util"]}, x0 + 900, 60)
        link(prev, f"{b}_dw"); link(f"{b}_dw", f"{b}_ln"); link(f"{b}_ln", f"{b}_pw1")
        link(f"{b}_pw1", f"{b}_gelu"); link(f"{b}_gelu", f"{b}_grn"); link(f"{b}_grn", f"{b}_pw2")
        link(prev, f"{b}_add"); link(f"{b}_pw2", f"{b}_add")   # residual: prev FIRST (primary)
        prev, x0 = f"{b}_add", x0 + 1050
    text_feat = prev

    # ── input embedding: concat(noisy, cond, text) -> Linear ──────────────────
    node("in_concat", "layerNode", "cat", "Concat",
         {"name": "InputEmbedding Concat", "dim": -1, "_subtitle": "[noisy ‖ cond ‖ text]",
          "_shape": f"[B, {seq_len}, {cat_dim}]", "_accent": A["util"]}, x0, 360)
    link("in_noisy", "in_concat"); link("in_cond", "in_concat"); link(text_feat, "in_concat")
    node("in_proj", "layerNode", "linear", "Linear",
         {"name": "Input Projection", "in_features": cat_dim, "out_features": dim,
          "_subtitle": f"Linear -> {dim}", "_shape": f"[B, {seq_len}, {dim}]", "_accent": A["lin"]}, x0 + 170, 360)
    link("in_concat", "in_proj")

    # ── convolutional position embedding (residual) ───────────────────────────
    node("cpe_c1", "layerNode", "conv1d", "Conv1d",
         {"name": "ConvPosEmbed C1", "in_channels": seq_len, "out_channels": seq_len, "kernel_size": 31,
          "stride": 1, "padding": 15, "groups": seq_len, "_subtitle": "Conv1d k31", "_accent": A["conv"]}, x0 + 340, 360)
    node("cpe_m1", "layerNode", "mish", "Mish", {"name": "Mish", "_accent": A["act"]}, x0 + 490, 360)
    node("cpe_c2", "layerNode", "conv1d", "Conv1d",
         {"name": "ConvPosEmbed C2", "in_channels": seq_len, "out_channels": seq_len, "kernel_size": 31,
          "stride": 1, "padding": 15, "groups": seq_len, "_subtitle": "Conv1d k31", "_accent": A["conv"]}, x0 + 640, 360)
    node("cpe_m2", "layerNode", "mish", "Mish", {"name": "Mish", "_accent": A["act"]}, x0 + 790, 360)
    node("cpe_add", "layerNode", "add", "Add",
         {"name": "ConvPos Residual", "_subtitle": "x + conv(x)", "_accent": A["util"]}, x0 + 940, 360)
    link("in_proj", "cpe_c1"); link("cpe_c1", "cpe_m1"); link("cpe_m1", "cpe_c2"); link("cpe_c2", "cpe_m2")
    link("in_proj", "cpe_add"); link("cpe_m2", "cpe_add")   # residual: in_proj FIRST
    x, bx = "cpe_add", x0 + 1110

    # ── DiT backbone: `depth` blocks ──────────────────────────────────────────
    for i in range(depth):
        b = f"dit{i}"
        # AdaLayerNorm-Zero modulation projection: SiLU(t_emb) -> Linear(dim, 6*dim)
        # producing shift/scale/gate for both the attention and FF paths. This is
        # the bulk of a DiT block's parameters (~6.3M each at dim=1024).
        node(f"{b}_mod", "layerNode", "linear", "Linear",
             {"name": f"DiT {i}: AdaLN-Zero Proj", "in_features": dim, "out_features": dim * 6,
              "_subtitle": "SiLU+Linear -> 6x(shift/scale/gate)", "_accent": A["lin"]}, bx, 200)
        node(f"{b}_adaln1", "layerNode", "ada_layer_norm", "ada_layer_norm",
             {"name": f"DiT {i}: AdaLayerNorm-Zero", "_subtitle": "modulate (shift/scale)", "_accent": A["norm"]}, bx, 360)
        node(f"{b}_rope", "layerNode", "rotary_embedding", "rotary_embedding",
             {"name": "RoPE", "_subtitle": "rotary pos emb (q,k)", "_accent": A["util"]}, bx + 150, 360)
        node(f"{b}_attn", "layerNode", "multiheadattention", "MultiheadAttention",
             {"name": "Self-Attention", "embed_dim": dim, "num_heads": heads, "dropout": 0.1,
              "_subtitle": f"{heads} heads", "_accent": A["attn"]}, bx + 300, 360)
        node(f"{b}_add1", "layerNode", "add", "Add",
             {"name": "Gated Residual (attn)", "_subtitle": "x + gate * attn", "_accent": A["util"]}, bx + 450, 360)
        node(f"{b}_adaln2", "layerNode", "ada_layer_norm", "ada_layer_norm",
             {"name": "FF AdaLN", "_subtitle": "scale/shift", "_accent": A["norm"]}, bx + 600, 360)
        node(f"{b}_ff1", "layerNode", "linear", "Linear",
             {"name": "FeedForward In", "in_features": dim, "out_features": dim * ff_mult, "_accent": A["lin"]}, bx + 750, 360)
        node(f"{b}_ffg", "layerNode", "gelu", "GELU", {"name": "GELU", "approximate": "tanh", "_accent": A["act"]}, bx + 900, 360)
        node(f"{b}_ffd", "layerNode", "dropout", "Dropout", {"name": "Dropout", "p": 0.1, "_accent": A["reg"]}, bx + 1050, 360)
        node(f"{b}_ff2", "layerNode", "linear", "Linear",
             {"name": "FeedForward Out", "in_features": dim * ff_mult, "out_features": dim, "_accent": A["lin"]}, bx + 1200, 360)
        node(f"{b}_add2", "layerNode", "add", "Add",
             {"name": "Gated Residual (ff)", "_subtitle": "x + gate * ff", "_accent": A["util"]}, bx + 1350, 360)
        link(t_emb, f"{b}_mod")                                    # timestep -> modulation proj
        link(x, f"{b}_adaln1"); link(f"{b}_mod", f"{b}_adaln1")    # sequence FIRST, then modulation
        link(f"{b}_adaln1", f"{b}_rope"); link(f"{b}_rope", f"{b}_attn")
        link(x, f"{b}_add1"); link(f"{b}_attn", f"{b}_add1")       # residual: sequence FIRST
        link(f"{b}_add1", f"{b}_adaln2"); link(f"{b}_mod", f"{b}_adaln2")  # FF path reuses the 6x modulation
        link(f"{b}_adaln2", f"{b}_ff1"); link(f"{b}_ff1", f"{b}_ffg")
        link(f"{b}_ffg", f"{b}_ffd"); link(f"{b}_ffd", f"{b}_ff2")
        link(f"{b}_add1", f"{b}_add2"); link(f"{b}_ff2", f"{b}_add2")
        x, bx = f"{b}_add2", bx + 1520

    # ── final AdaLayerNorm + output projection ────────────────────────────────
    node("final_mod", "layerNode", "linear", "Linear",
         {"name": "AdaLN-Final Proj", "in_features": dim, "out_features": dim * 2,
          "_subtitle": "SiLU+Linear -> 2x(shift/scale)", "_accent": A["lin"]}, bx, 200)
    node("final_norm", "layerNode", "ada_layer_norm_final", "ada_layer_norm_final",
         {"name": "AdaLayerNorm-Final", "_subtitle": "modulate (shift/scale)", "_accent": A["norm"]}, bx, 360)
    node("out_proj", "layerNode", "linear", "Linear",
         {"name": "Output Projection", "in_features": dim, "out_features": mel_dim,
          "_subtitle": f"Linear -> {mel_dim} mel", "_shape": f"[B, {seq_len}, {mel_dim}]", "_accent": A["lin"]}, bx + 170, 360)
    node("out_node", "outputNode", "output", "Output",
         {"name": "Output", "_subtitle": "predicted flow (mel)", "_shape": f"[B, {seq_len}, {mel_dim}]", "_accent": A["emb"]}, bx + 340, 360)
    link(t_emb, "final_mod")
    link(x, "final_norm"); link("final_mod", "final_norm")
    link("final_norm", "out_proj"); link("out_proj", "out_node")

    return {"root_nodes": nodes, "root_edges": edges, "modules": {}}


class GenerativeTemplates:
    """Service for Generative model architectures"""
    
    def get_all_templates(self) -> Dict[str, ModelTemplate]:
        """Get all generative templates"""
        return {
            "gan": self._create_gan_template(),
            "vae": self._create_vae_template(),
            "stylegan": self._create_stylegan_template(),
            "dcgan": self._create_dcgan_template(),
            "cyclegan": self._create_cyclegan_template(),
            "diffusion_transformer": self._create_diffusion_transformer(),
            "f5_tts": self._create_f5tts_template(),
        }
    
    def get_module_templates(self) -> Dict[str, ModuleTemplate]:
        """Get generative module templates"""
        return {
            "generator_block": self._create_generator_block(),
            "discriminator_block": self._create_discriminator_block(),
            "style_block": self._create_style_block(),
            "attention_gate": self._create_attention_gate(),
        }
    
    def get_popular_models(self) -> Dict[str, PopularModel]:
        """Get popular generative models"""
        return {
            "stylegan2": self._create_stylegan2(),
            "biggan": self._create_biggan(),
            "progressive_gan": self._create_progressive_gan(),
            "nanodit": self._create_nanodit(),
            "f5tts": self._create_f5tts(),
        }
    
    def _create_gan_template(self) -> ModelTemplate:
        """Create basic GAN template"""
        return ModelTemplate(
            id="gan",
            name="Generative Adversarial Network",
            description="Basic GAN with generator and discriminator for image generation",
            category="generative",
            sub_category="gan",
            difficulty="intermediate",
            tags=["gan", "generator", "discriminator", "adversarial"],
            
            base_config={
                "latent_dim": 100,
                "image_channels": 3,
                "image_size": 64,
                "generator_features": [512, 256, 128, 64],
                "discriminator_features": [64, 128, 256, 512],
                "use_batch_norm": True,
                "use_spectral_norm": False,
                "activation": "relu"
            },
            
            customizable_params=[
                {
                    "name": "latent_dim",
                    "type": "number",
                    "min": 32,
                    "max": 512,
                    "description": "Latent vector dimension"
                },
                {
                    "name": "image_size",
                    "type": "select",
                    "options": [32, 64, 128, 256, 512],
                    "description": "Output image size"
                },
                {
                    "name": "use_spectral_norm",
                    "type": "boolean",
                    "description": "Use spectral normalization"
                }
            ],
            
            template_structure={
                "nodes": [
                    {
                        "id": "generator",
                        "type": "moduleNode",
                        "position": {"x": 200, "y": 100},
                        "data": {
                            "layerType": {"id": "generator", "name": "Generator", "parameters": []},
                            "parameters": {
                                "latent_dim": "{latent_dim}",
                                "output_channels": "{image_channels}",
                                "output_size": "{image_size}",
                                "features": "{generator_features}",
                                "use_batch_norm": "{use_batch_norm}",
                                "activation": "{activation}"
                            }
                        }
                    },
                    {
                        "id": "discriminator",
                        "type": "moduleNode",
                        "position": {"x": 200, "y": 400},
                        "data": {
                            "layerType": {"id": "discriminator", "name": "Discriminator", "parameters": []},
                            "parameters": {
                                "input_channels": "{image_channels}",
                                "input_size": "{image_size}",
                                "features": "{discriminator_features}",
                                "use_spectral_norm": "{use_spectral_norm}",
                                "activation": "leaky_relu"
                            }
                        }
                    }
                ],
                "edges": [
                    # Chain G -> D so the single-input validator flows the latent
                    # through the generator to an image, then into the
                    # discriminator (latent -> G -> image -> D -> score).
                    {"id": "e1", "source": "generator", "target": "discriminator"}
                ]
            },

            variants=[
                {
                    "id": "dcgan",
                    "name": "DCGAN",
                    "description": "Deep Convolutional GAN",
                    "config_overrides": {
                        "use_batch_norm": True,
                        "activation": "relu"
                    }
                },
                {
                    "id": "wgan",
                    "name": "Wasserstein GAN",
                    "description": "GAN with Wasserstein loss",
                    "config_overrides": {
                        "use_spectral_norm": True
                    }
                }
            ],
            
            typical_use_cases=[
                "Image Generation",
                "Data Augmentation",
                "Style Transfer",
                "Super Resolution"
            ],
            
            paper_reference="Generative Adversarial Nets (Goodfellow et al., 2014)",
            minimum_input_shape=[1, 100],
            recommended_input_shape=[64, 100]
        )
    
    def _create_vae_template(self) -> ModelTemplate:
        """Create Variational Autoencoder template"""
        return ModelTemplate(
            id="vae",
            name="Variational Autoencoder",
            description="VAE for learning latent representations and generating new samples",
            category="generative",
            sub_category="autoencoder",
            difficulty="intermediate",
            tags=["vae", "encoder", "decoder", "latent", "reconstruction"],
            
            base_config={
                "input_channels": 3,
                "input_size": 64,
                "latent_dim": 128,
                "encoder_features": [32, 64, 128, 256],
                "decoder_features": [256, 128, 64, 32],
                "beta": 1.0,  # β-VAE parameter
                "use_batch_norm": True
            },
            
            customizable_params=[
                {
                    "name": "latent_dim",
                    "type": "number",
                    "min": 32,
                    "max": 512,
                    "description": "Latent space dimension"
                },
                {
                    "name": "beta",
                    "type": "number",
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "description": "β parameter for KL divergence"
                }
            ],
            
            template_structure={
                "nodes": [
                    {
                        "id": "encoder",
                        "type": "moduleNode",
                        "position": {"x": 100, "y": 200},
                        "data": {
                            "layerType": {"id": "vae_encoder", "name": "VAEEncoder", "parameters": []},
                            "parameters": {
                                "input_channels": "{input_channels}",
                                "input_size": "{input_size}",
                                "latent_dim": "{latent_dim}",
                                "features": "{encoder_features}",
                                "use_batch_norm": "{use_batch_norm}"
                            }
                        }
                    },
                    {
                        "id": "reparameterize",
                        "type": "moduleNode",
                        "position": {"x": 300, "y": 200},
                        "data": {
                            "layerType": {"id": "reparameterize", "name": "ReparameterizationTrick", "parameters": []},
                            "parameters": {
                                "latent_dim": "{latent_dim}"
                            }
                        }
                    },
                    {
                        "id": "decoder",
                        "type": "moduleNode",
                        "position": {"x": 500, "y": 200},
                        "data": {
                            "layerType": {"id": "vae_decoder", "name": "VAEDecoder", "parameters": []},
                            "parameters": {
                                "latent_dim": "{latent_dim}",
                                "output_channels": "{input_channels}",
                                "output_size": "{input_size}",
                                "features": "{decoder_features}",
                                "use_batch_norm": "{use_batch_norm}"
                            }
                        }
                    }
                ],
                "edges": [
                    {"id": "e1", "source": "encoder", "target": "reparameterize"},
                    {"id": "e2", "source": "reparameterize", "target": "decoder"}
                ]
            },
            
            variants=[
                {
                    "id": "beta_vae",
                    "name": "β-VAE",
                    "description": "VAE with adjustable β for disentangled representations",
                    "config_overrides": {"beta": 4.0}
                },
                {
                    "id": "conditional_vae",
                    "name": "Conditional VAE",
                    "description": "VAE conditioned on class labels",
                    "structural_changes": [
                        {"type": "add_module", "module_id": "label_embedding"}
                    ]
                }
            ],
            
            typical_use_cases=[
                "Image Generation",
                "Representation Learning",
                "Anomaly Detection",
                "Data Compression"
            ],
            
            paper_reference="Auto-Encoding Variational Bayes (Kingma & Welling, 2013)",
            minimum_input_shape=[1, 3, 28, 28],
            recommended_input_shape=[32, 3, 64, 64]
        )
    
    def _create_stylegan_template(self) -> ModelTemplate:
        """Create StyleGAN template"""
        return ModelTemplate(
            id="stylegan",
            name="StyleGAN",
            description="Style-based generator for high-quality image synthesis",
            category="generative",
            sub_category="stylegan",
            difficulty="advanced",
            tags=["stylegan", "style", "mapping", "synthesis", "progressive"],
            
            base_config={
                "latent_dim": 512,
                "style_dim": 512,
                "num_layers": 8,
                "image_size": 1024,
                "image_channels": 3,
                "use_noise": True,
                "style_mixing": True,
                "truncation_psi": 0.7
            },
            
            customizable_params=[
                {
                    "name": "image_size",
                    "type": "select",
                    "options": [64, 128, 256, 512, 1024],
                    "description": "Output image resolution"
                },
                {
                    "name": "truncation_psi",
                    "type": "number",
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "description": "Truncation trick parameter"
                }
            ],
            
            template_structure={
                "nodes": [
                    {
                        "id": "mapping_network",
                        "type": "moduleNode",
                        "position": {"x": 100, "y": 100},
                        "data": {
                            "layerType": {"id": "mapping_network", "name": "MappingNetwork", "parameters": []},
                            "parameters": {
                                "latent_dim": "{latent_dim}",
                                "style_dim": "{style_dim}",
                                "num_layers": "{num_layers}"
                            }
                        }
                    },
                    {
                        "id": "synthesis_network",
                        "type": "moduleNode",
                        "position": {"x": 400, "y": 200},
                        "data": {
                            "layerType": {"id": "synthesis_network", "name": "SynthesisNetwork", "parameters": []},
                            "parameters": {
                                "style_dim": "{style_dim}",
                                "image_size": "{image_size}",
                                "image_channels": "{image_channels}",
                                "use_noise": "{use_noise}"
                            }
                        }
                    },
                    {
                        "id": "style_discriminator",
                        "type": "moduleNode",
                        "position": {"x": 400, "y": 500},
                        "data": {
                            "layerType": {"id": "style_discriminator", "name": "StyleDiscriminator", "parameters": []},
                            "parameters": {
                                "image_channels": "{image_channels}",
                                "image_size": "{image_size}"
                            }
                        }
                    }
                ],
                "edges": [
                    {"id": "e1", "source": "mapping_network", "target": "synthesis_network"},
                    # Chain synthesis -> discriminator so the single-input
                    # validator flows the latent through mapping + synthesis to a
                    # generated image, then into the discriminator.
                    {"id": "e2", "source": "synthesis_network", "target": "style_discriminator"}
                ]
            },

            variants=[
                {
                    "id": "stylegan2",
                    "name": "StyleGAN2",
                    "description": "Improved StyleGAN with better image quality",
                    "config_overrides": {
                        "use_noise": True,
                        "style_mixing": True
                    }
                }
            ],
            
            typical_use_cases=[
                "High-Quality Image Generation",
                "Face Generation",
                "Style Transfer",
                "Image Editing"
            ],
            
            paper_reference="Analyzing and Improving the Image Quality of StyleGAN (Karras et al., 2019)",
            minimum_input_shape=[1, 512],
            recommended_input_shape=[16, 512]
        )
    
    def _create_dcgan_template(self) -> ModelTemplate:
        """Create DCGAN template"""
        return ModelTemplate(
            id="dcgan",
            name="DCGAN",
            description="Deep Convolutional GAN with architectural guidelines",
            category="generative",
            sub_category="gan",
            difficulty="beginner",
            tags=["dcgan", "convolutional", "guidelines"],
            
            base_config={
                "latent_dim": 100,
                "image_channels": 3,
                "image_size": 64,
                "feature_maps_g": 64,
                "feature_maps_d": 64,
                "use_batch_norm": True,
                "leaky_relu_slope": 0.2
            },
            
            customizable_params=[
                {
                    "name": "feature_maps_g",
                    "type": "number",
                    "min": 32,
                    "max": 512,
                    "description": "Generator feature map size"
                },
                {
                    "name": "feature_maps_d", 
                    "type": "number",
                    "min": 32,
                    "max": 512,
                    "description": "Discriminator feature map size"
                }
            ],
            
            template_structure={
                "nodes": [
                    {
                        "id": "dcgan_generator",
                        "type": "moduleNode",
                        "position": {"x": 200, "y": 100},
                        "data": {
                            "layerType": {"id": "dcgan_generator", "name": "DCGANGenerator", "parameters": []},
                            "parameters": {
                                "latent_dim": "{latent_dim}",
                                "feature_maps": "{feature_maps_g}",
                                "output_channels": "{image_channels}",
                                "output_size": "{image_size}"
                            }
                        }
                    },
                    {
                        "id": "dcgan_discriminator",
                        "type": "moduleNode",
                        "position": {"x": 200, "y": 400},
                        "data": {
                            "layerType": {"id": "dcgan_discriminator", "name": "DCGANDiscriminator", "parameters": []},
                            "parameters": {
                                "input_channels": "{image_channels}",
                                "feature_maps": "{feature_maps_d}",
                                "leaky_relu_slope": "{leaky_relu_slope}"
                            }
                        }
                    }
                ],
                "edges": []
            },
            
            typical_use_cases=[
                "Image Generation",
                "Educational GAN Implementation",
                "Benchmark Comparisons"
            ],
            
            paper_reference="Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (Radford et al., 2015)",
            minimum_input_shape=[1, 100],
            recommended_input_shape=[64, 100]
        )
    
    def _create_cyclegan_template(self) -> ModelTemplate:
        """Create CycleGAN template"""
        return ModelTemplate(
            id="cyclegan",
            name="CycleGAN",
            description="Cycle-consistent adversarial networks for unpaired image-to-image translation",
            category="generative",
            sub_category="translation",
            difficulty="advanced",
            tags=["cyclegan", "translation", "cycle-consistent", "unpaired"],
            
            base_config={
                "input_channels": 3,
                "output_channels": 3,
                "image_size": 256,
                "generator_filters": 64,
                "discriminator_filters": 64,
                "num_residual_blocks": 9,
                "lambda_cycle": 10.0,
                "lambda_identity": 0.5
            },
            
            customizable_params=[
                {
                    "name": "num_residual_blocks",
                    "type": "number",
                    "min": 6,
                    "max": 12,
                    "description": "Number of residual blocks in generator"
                },
                {
                    "name": "lambda_cycle",
                    "type": "number",
                    "min": 1.0,
                    "max": 20.0,
                    "description": "Cycle consistency loss weight"
                }
            ],
            
            template_structure={
                "nodes": [
                    {
                        "id": "generator_a_to_b",
                        "type": "moduleNode",
                        "position": {"x": 100, "y": 100},
                        "data": {
                            "layerType": {"id": "resnet_generator", "name": "ResNetGenerator", "parameters": []},
                            "parameters": {
                                "input_channels": "{input_channels}",
                                "output_channels": "{output_channels}",
                                "filters": "{generator_filters}",
                                "num_residual_blocks": "{num_residual_blocks}"
                            }
                        }
                    },
                    {
                        "id": "generator_b_to_a",
                        "type": "moduleNode",
                        "position": {"x": 500, "y": 100},
                        "data": {
                            "layerType": {"id": "resnet_generator", "name": "ResNetGenerator", "parameters": []},
                            "parameters": {
                                "input_channels": "{output_channels}",
                                "output_channels": "{input_channels}",
                                "filters": "{generator_filters}",
                                "num_residual_blocks": "{num_residual_blocks}"
                            }
                        }
                    },
                    {
                        "id": "discriminator_a",
                        "type": "moduleNode",
                        "position": {"x": 100, "y": 400},
                        "data": {
                            "layerType": {"id": "patch_discriminator", "name": "PatchDiscriminator", "parameters": []},
                            "parameters": {
                                "input_channels": "{input_channels}",
                                "filters": "{discriminator_filters}"
                            }
                        }
                    },
                    {
                        "id": "discriminator_b",
                        "type": "moduleNode",
                        "position": {"x": 500, "y": 400},
                        "data": {
                            "layerType": {"id": "patch_discriminator", "name": "PatchDiscriminator", "parameters": []},
                            "parameters": {
                                "input_channels": "{output_channels}",
                                "filters": "{discriminator_filters}"
                            }
                        }
                    }
                ],
                "edges": [
                    {"id": "e1", "source": "generator_a_to_b", "target": "generator_b_to_a"}
                ]
            },
            
            typical_use_cases=[
                "Style Transfer",
                "Domain Adaptation",
                "Image-to-Image Translation",
                "Data Augmentation"
            ],
            
            paper_reference="Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (Zhu et al., 2017)",
            minimum_input_shape=[1, 3, 128, 128],
            recommended_input_shape=[16, 3, 256, 256]
        )
    
    # Module template implementations (simplified)
    def _create_generator_block(self) -> ModuleTemplate:
        pass
    
    def _create_discriminator_block(self) -> ModuleTemplate:
        pass
    
    def _create_style_block(self) -> ModuleTemplate:
        pass
    
    def _create_attention_gate(self) -> ModuleTemplate:
        pass
    
    # Popular model implementations
    def _create_stylegan2(self) -> PopularModel:
        """Create StyleGAN2 model configuration"""
        return PopularModel(
            id="stylegan2",
            name="StyleGAN2",
            full_name="Analyzing and Improving the Image Quality of StyleGAN",
            description="Improved StyleGAN with better image quality and reduced artifacts",
            category="generative",
            architecture_type="stylegan",
            parameter_count="30M",
            training_dataset="FFHQ, LSUN, etc.",
            template_id="stylegan",
            config={
                "latent_dim": 512,
                "style_dim": 512,
                "image_size": 1024,
                "use_noise": True,
                "style_mixing": True
            },
            typical_applications=[
                "High-Quality Face Generation",
                "Style Transfer",
                "Image Editing",
                "Data Augmentation"
            ]
        )
    
    def _create_biggan(self) -> PopularModel:
        """Create BigGAN model configuration"""
        return PopularModel(
            id="biggan",
            name="BigGAN",
            full_name="Large Scale GAN Training for High Fidelity Natural Image Synthesis",
            description="Large-scale GAN for high-fidelity natural image synthesis",
            category="generative",
            architecture_type="gan",
            parameter_count="158M",
            training_dataset="ImageNet",
            template_id="gan",
            config={
                "latent_dim": 128,
                "image_size": 512,
                "use_spectral_norm": True,
                "use_self_attention": True
            },
            typical_applications=[
                "High-Resolution Image Generation",
                "Class-Conditional Generation",
                "Large-Scale Synthesis",
                "Research Benchmarks"
            ]
        )
    
    def _create_progressive_gan(self) -> PopularModel:
        """Create Progressive GAN model configuration"""
        return PopularModel(
            id="progressive_gan",
            name="Progressive GAN",
            full_name="Progressive Growing of GANs for Improved Quality, Stability, and Variation",
            description="GAN that progressively grows from low to high resolution",
            category="generative",
            architecture_type="progressive_gan",
            parameter_count="46M",
            training_dataset="CelebA-HQ",
            template_id="gan",
            config={
                "latent_dim": 512,
                "image_size": 1024,
                "progressive_growing": True,
                "use_pixel_norm": True
            },
            typical_applications=[
                "High-Resolution Image Generation",
                "Stable Training",
                "Progressive Image Synthesis",
                "Face Generation"
            ]
        )

    def _create_diffusion_transformer(self) -> ModelTemplate:
        """Create the nanoDiT (Diffusion Transformer) template.

        Ported from https://github.com/sayakpaul/nanoDiT (DiT, Peebles & Xie
        2022). Renders left->right: Input (noised latent/image) -> Patch
        Embedding (Conv2d) -> Tokenize (Flatten) -> DiT Block (moduleNode,
        expanded, x depth) -> Final Norm (LayerNorm) -> Detokenize (Unflatten)
        -> Unpatchify (ConvTranspose2d) -> Output. Unlike a classifier, a DiT
        predicts a tensor with the SAME shape as its input (the denoised /
        noise prediction), so the chain patchifies, runs the transformer
        backbone over the patch tokens, then unpatchifies back to image space.

        Runnable structural template: every step is a single-input torch op so
        the dry forward pass + code export run end-to-end. The DiT block is one
        pre-norm nn.TransformerEncoderLayer (LN->attn->residual->LN->MLP->
        residual) repeated x depth. The adaLN timestep + class-label
        conditioning of the real DiT is auxiliary and can't run in the single-
        input executor, so it is omitted here exactly as the ViT template omits
        its CLS token / positional embedding.
        """
        return ModelTemplate(
            id="diffusion_transformer",
            name="Diffusion Transformer (nanoDiT)",
            description="Transformer-backbone diffusion model that patchifies a noised image, applies a stack of DiT blocks, then unpatchifies back to image space (Peebles & Xie, 2022; nanoDiT by sayakpaul)",
            category="generative",
            sub_category="diffusion",
            difficulty="advanced",
            tags=["diffusion", "transformer", "dit", "generative", "patches"],

            base_config={
                "input_size": 32,
                "patch_size": 2,
                "in_channels": 3,
                "hidden_size": 384,
                "num_layers": 12,
                "num_heads": 6,
                "num_classes": 10,
                "seq_shape": "[B, 384, 256]",
            },

            customizable_params=[
                {"name": "hidden_size", "type": "number", "min": 64, "max": 1536, "step": 64,
                 "description": "Transformer hidden / patch-embed channel dim", "category": "architecture"},
                {"name": "num_layers", "type": "number", "min": 1, "max": 32,
                 "description": "Number of DiT blocks (depth)", "category": "architecture"},
                {"name": "num_heads", "type": "number", "min": 1, "max": 16,
                 "description": "Number of attention heads", "category": "attention"},
                {"name": "num_classes", "type": "number", "min": 1, "max": 1000,
                 "description": "Number of conditioning classes", "category": "conditioning"},
            ],

            paper_reference="Scalable Diffusion Models with Transformers (Peebles & Xie, 2022)",
            typical_use_cases=["Class-conditional image generation", "Latent diffusion", "Research baselines"],
            minimum_input_shape=[1, 3, 32, 32],
            recommended_input_shape=[1, 3, 32, 32],

            template_structure={
                "root_nodes": [
                    {
                        "id": "input_node",
                        "type": "inputNode",
                        "position": {"x": 80, "y": 200},
                        "data": {
                            "layerType": {"id": "input", "name": "Input", "parameters": []},
                            "parameters": {
                                "name": "Input",
                                "shape": "1,{in_channels},{input_size},{input_size}",
                                "_subtitle": "Noised latent",
                                "_shape": "[B, 3, 32, 32]",
                                "_accent": "var(--cat-embedding)",
                            },
                        },
                    },
                    {
                        "id": "patch_embedding",
                        "type": "layerNode",
                        "position": {"x": 300, "y": 200},
                        "data": {
                            "layerType": {"id": "conv2d", "name": "Conv2d", "parameters": []},
                            "parameters": {
                                "name": "Patch Embedding",
                                "in_channels": "{in_channels}",
                                "out_channels": "{hidden_size}",
                                "kernel_size": "{patch_size}",
                                "stride": "{patch_size}",
                                "padding": 0,
                                "_subtitle": "Conv2d 2×2",
                                "_shape": "[B, 384, 16, 16]",
                                "_accent": "var(--cat-conv)",
                            },
                        },
                    },
                    {
                        "id": "tokenize",
                        "type": "layerNode",
                        "position": {"x": 520, "y": 200},
                        "data": {
                            "layerType": {"id": "flatten", "name": "Flatten", "parameters": []},
                            "parameters": {
                                "name": "Tokenize",
                                "start_dim": 2,
                                "_subtitle": "flatten patches",
                                "_shape": "[B, 384, 256]",
                                "_accent": "var(--cat-utility)",
                            },
                        },
                    },
                    {
                        "id": "dit_blocks",
                        "type": "moduleNode",
                        "position": {"x": 740, "y": 200},
                        "data": {
                            "layerType": {"id": "module", "name": "Module", "parameters": []},
                            "parameters": {
                                "name": "DiT Blocks",
                                "module_type": "dit_block_stack",
                                "num_layers": "{num_layers}",
                            },
                            "moduleType": "custom",
                            "expanded": True,
                        },
                    },
                    {
                        "id": "final_norm",
                        "type": "layerNode",
                        "position": {"x": 960, "y": 200},
                        "data": {
                            "layerType": {"id": "layernorm", "name": "LayerNorm", "parameters": []},
                            "parameters": {
                                "name": "Final Norm",
                                "normalized_shape": 256,
                                "eps": 0.000001,
                                "_subtitle": "LayerNorm",
                                "_shape": "[B, 384, 256]",
                                "_accent": "var(--cat-normalization)",
                            },
                        },
                    },
                    {
                        "id": "detokenize",
                        "type": "layerNode",
                        "position": {"x": 1180, "y": 200},
                        "data": {
                            "layerType": {"id": "unflatten", "name": "Unflatten", "parameters": []},
                            "parameters": {
                                "name": "Detokenize",
                                "dim": 2,
                                "unflattened_size": [16, 16],
                                "_subtitle": "tokens → grid",
                                "_shape": "[B, 384, 16, 16]",
                                "_accent": "var(--cat-utility)",
                            },
                        },
                    },
                    {
                        "id": "unpatchify",
                        "type": "layerNode",
                        "position": {"x": 1400, "y": 200},
                        "data": {
                            "layerType": {"id": "convtranspose2d", "name": "ConvTranspose2d", "parameters": []},
                            "parameters": {
                                "name": "Unpatchify",
                                "in_channels": "{hidden_size}",
                                "out_channels": "{in_channels}",
                                "kernel_size": "{patch_size}",
                                "stride": "{patch_size}",
                                "padding": 0,
                                "_subtitle": "ConvTranspose2d 2×2",
                                "_shape": "[B, 3, 32, 32]",
                                "_accent": "var(--cat-conv)",
                            },
                        },
                    },
                    {
                        "id": "output_node",
                        "type": "outputNode",
                        "position": {"x": 1620, "y": 200},
                        "data": {
                            "layerType": {"id": "output", "name": "Output", "parameters": []},
                            "parameters": {
                                "name": "Output",
                                "_subtitle": "Noise prediction",
                                "_shape": "[B, 3, 32, 32]",
                                "_accent": "var(--cat-embedding)",
                            },
                        },
                    },
                ],
                "root_edges": [
                    {"id": "e_in_patch", "source": "input_node", "target": "patch_embedding"},
                    {"id": "e_patch_tok", "source": "patch_embedding", "target": "tokenize"},
                    {"id": "e_tok_blocks", "source": "tokenize", "target": "dit_blocks"},
                    {"id": "e_blocks_norm", "source": "dit_blocks", "target": "final_norm"},
                    {"id": "e_norm_detok", "source": "final_norm", "target": "detokenize"},
                    {"id": "e_detok_unpatch", "source": "detokenize", "target": "unpatchify"},
                    {"id": "e_unpatch_out", "source": "unpatchify", "target": "output_node"},
                ],
                "modules": {},
            },
        )

    def _create_nanodit(self) -> PopularModel:
        """nanoDiT — minimal Diffusion Transformer configuration."""
        return PopularModel(
            id="nanodit",
            name="nanoDiT",
            full_name="nanoDiT — Minimal Diffusion Transformer (DiT-S/2)",
            description="Compact Diffusion Transformer (DiT) backbone: patchify → 12 DiT blocks → unpatchify, predicting a same-shape noise tensor (sayakpaul/nanoDiT)",
            category="generative",
            architecture_type="diffusion_transformer",
            parameter_count="~33M",
            training_dataset="CIFAR-10 / class-conditional latents",
            training_details={
                "input_size": 32,
                "patch_size": 2,
                "depth": 12,
                "hidden_size": 384,
                "num_heads": 6,
                "num_classes": 10,
            },
            template_id="diffusion_transformer",
            config={
                "input_size": 32,
                "patch_size": 2,
                "in_channels": 3,
                "hidden_size": 384,
                "num_layers": 12,
                "num_heads": 6,
                "num_classes": 10,
                "seq_shape": "[B, 384, 256]",
            },
            typical_applications=["Class-Conditional Image Generation", "Latent Diffusion", "Research Baselines"],
        )

    def _create_f5tts_template(self) -> ModelTemplate:
        """F5-TTS — flow-matching text-to-speech with a DiT backbone.

        Faithful to the swivid/F5-TTS architecture (Chen et al., 2024,
        arXiv:2410.06885): a character/text encoder (embedding + ConvNeXt-V2
        blocks), an input-embedding fusion, a convolutional position embedding,
        a deep stack of DiT blocks (AdaLayerNorm-Zero → RoPE attention →
        FeedForward), a final adaptive LayerNorm and an output projection to mel
        channels (the predicted flow / vector field).

        The full, faithful multi-branch graph is generated by
        ``_build_f5tts_structure``: a text branch (Char Embedding + ConvNeXt-V2
        blocks with GRN + residuals), a timestep-embedding branch
        (SinusPositionEmbedding + MLP), the input-embedding fusion (concat of
        noisy mel + reference mel + text → Linear), a convolutional position
        embedding (residual), `depth` DiT blocks each with AdaLayerNorm-Zero
        conditioned on the timestep → RoPE self-attention → gated residual →
        FF-AdaLN → FeedForward → gated residual, a final AdaLayerNorm and the mel
        output projection. Every block is a real, correctly-named node and the
        conditioning + residual wiring is explicit. It operates in [B, N, dim]
        (batch first); shape-bearing torch ops chain exactly while the custom
        blocks (SinusPositionEmbedding / GRN / RoPE / AdaLayerNorm) are
        shape-preserving and recognized by both the shape validator and the dry
        forward, so the whole graph validates AND verifies with zero errors."""
        return ModelTemplate(
            id="f5_tts",
            name="F5-TTS",
            description="Flow-matching TTS with a DiT backbone: text encoder (ConvNeXt-V2) + input/conv-position embedding + DiT block stack + output mel projection (Chen et al., 2024)",
            category="generative",
            sub_category="tts",
            difficulty="advanced",
            tags=["tts", "speech", "flow-matching", "dit", "transformer", "convnext", "generative"],

            base_config={
                "seq_len": 256,
                "mel_dim": 100,
                "text_dim": 512,
                "dim": 1024,
                "depth": 22,
                "heads": 16,
                "ff_mult": 2,
                "conv_layers": 4,
                "conv_intermediate": 1024,
                "text_num_embeds": 256,
            },

            customizable_params=[
                {"name": "dim", "type": "number", "min": 64, "max": 2048, "step": 64,
                 "description": "Transformer model dim", "category": "architecture"},
                {"name": "depth", "type": "number", "min": 1, "max": 32,
                 "description": "Number of DiT blocks (depth)", "category": "architecture"},
                {"name": "heads", "type": "number", "min": 1, "max": 32,
                 "description": "Number of attention heads", "category": "attention"},
                {"name": "ff_mult", "type": "number", "min": 1, "max": 8,
                 "description": "FeedForward expansion multiplier", "category": "architecture"},
                {"name": "text_dim", "type": "number", "min": 64, "max": 1024, "step": 64,
                 "description": "Text/character embedding dim", "category": "text"},
                {"name": "conv_layers", "type": "number", "min": 0, "max": 12,
                 "description": "ConvNeXt-V2 blocks in the text encoder", "category": "text"},
                {"name": "mel_dim", "type": "number", "min": 1, "max": 256,
                 "description": "Mel channels (output flow dim)", "category": "audio"},
            ],

            paper_reference="F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching (Chen et al., 2024, arXiv:2410.06885)",
            typical_use_cases=["Zero-shot text-to-speech", "Voice cloning", "Speech synthesis research"],
            minimum_input_shape=[1, 256],
            recommended_input_shape=[1, 256],

            template_structure=_build_f5tts_structure(
                dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512,
                conv_layers=4, mel_dim=100, seq_len=256, conv_intermediate=1024,
            ),
        )

    def _create_f5tts(self) -> PopularModel:
        """F5-TTS Base — the released 335M flow-matching TTS model."""
        return PopularModel(
            id="f5tts",
            name="F5-TTS",
            full_name="F5-TTS Base — Fairytaler Fakes Fluent and Faithful Speech with Flow Matching",
            description="Non-autoregressive flow-matching TTS: ConvNeXt-V2 text encoder + 22-layer DiT backbone with RoPE attention, predicting a mel-spectrogram vector field (swivid/F5-TTS)",
            category="generative",
            architecture_type="flow_matching_dit",
            parameter_count="335.8M",
            training_dataset="Emilia (95k h multilingual speech)",
            training_details={
                "dim": 1024,
                "depth": 22,
                "heads": 16,
                "ff_mult": 2,
                "text_dim": 512,
                "conv_layers": 4,
                "mel_dim": 100,
            },
            template_id="f5_tts",
            config={
                "seq_len": 256,
                "mel_dim": 100,
                "text_dim": 512,
                "dim": 1024,
                "depth": 22,
                "heads": 16,
                "ff_mult": 2,
                "conv_layers": 4,
                "conv_intermediate": 1024,
                "text_num_embeds": 256,
            },
            typical_applications=["Zero-Shot TTS", "Voice Cloning", "Speech Synthesis"],
        )
