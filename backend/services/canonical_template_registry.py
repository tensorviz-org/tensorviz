"""Canonical operation templates for complex model blocks."""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from models.architecture_models import ArchitectureEdge, ArchitectureNode


@dataclass(frozen=True)
class TemplateContract:
    template_id: str
    name: str
    required_nodes: List[str]
    required_edges: List[str]
    allowed_residual_patterns: List[str]
    training_minimums: List[str]
    input_expectation: str


#: Ops that pass a tensor through unchanged, so they say nothing about its shape.
#: A block whose first node is one of these — the GNN layers open on `nn.Identity`,
#: DenseNet on a BatchNorm-free Identity — left the seed with no information, and the
#: first real layer then refused the default: `nn.Linear(in_features=64)` handed a
#: [1, 256] tensor is "mat1 and mat2 shapes cannot be multiplied".
_SHAPE_NEUTRAL_OPS = frozenset({
    "identity", "relu", "leakyrelu", "gelu", "silu", "elu", "tanh", "sigmoid",
    "hardswish", "hardsigmoid", "mish", "softplus", "dropout", "dropout1d",
    "dropout2d", "dropout3d", "alphadropout",
})


def _bare_op(op: Optional[str]) -> str:
    """`nn.Identity` -> `identity`; the registry stores ops both ways."""
    return str(op or "").rsplit(".", 1)[-1].strip().lower()


class CanonicalTemplateRegistry:
    """Registry for canonical complex templates and planner-facing constraints."""

    def __init__(self) -> None:
        self._dynamic_templates_path = os.getenv(
            "PAPER_DYNAMIC_TEMPLATES_REGISTRY_PATH",
            os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", ".runtime", "dynamic-templates.json")
            ),
        )
        self._dynamic_templates: Dict[str, Dict[str, Any]] = {}
        #: Lazily computed set of ids the code generator can actually build (list_contracts).
        self._buildable: Optional[Set[str]] = None
        self._contracts: Dict[str, TemplateContract] = {
            "full_transformer_block": TemplateContract(
                template_id="full_transformer_block",
                name="Full Transformer Block",
                required_nodes=[
                    "LayerNorm (x2)",
                    "MultiheadAttention",
                    "Linear (x2)",
                    "GELU",
                    "Add residual (x2)",
                ],
                required_edges=[
                    "input -> norm1 -> attn -> add_attn",
                    "input -> add_attn (residual)",
                    "add_attn -> norm2 -> mlp1 -> gelu -> mlp2 -> add_mlp",
                    "add_attn -> add_mlp (residual)",
                ],
                allowed_residual_patterns=["post-attention add", "post-mlp add"],
                training_minimums=[
                    "d_model % num_heads == 0",
                    "ffn_dim >= d_model",
                    "dropout in [0.0, 0.5]",
                ],
                input_expectation="rank-3 tensor [batch, tokens, d_model]",
            ),
            "vit_block": TemplateContract(
                template_id="vit_block",
                name="ViT Token Block",
                required_nodes=[
                    "Linear patch embedding",
                    "LayerNorm",
                    "MultiheadAttention",
                    "MLP (Linear -> GELU -> Linear)",
                    "Add residual (x2)",
                ],
                required_edges=[
                    "input -> patch_embed -> norm1 -> attn -> add_attn",
                    "patch_embed -> add_attn (residual)",
                    "add_attn -> norm2 -> mlp1 -> gelu -> mlp2 -> add_mlp",
                    "add_attn -> add_mlp (residual)",
                ],
                allowed_residual_patterns=["token residual after attn", "token residual after mlp"],
                training_minimums=[
                    "d_model % num_heads == 0",
                    "mlp_dim >= d_model",
                    "patch_dim > 0",
                ],
                input_expectation="rank-3 tensor [batch, tokens, patch_dim]",
            ),
            "unet_block": TemplateContract(
                template_id="unet_block",
                name="UNet Skip Block",
                required_nodes=[
                    "Encoder conv stack",
                    "Bottleneck conv",
                    "Decoder projection conv",
                    "Concat skip merge",
                    "Fusion conv",
                ],
                required_edges=[
                    "input -> enc_conv1 -> enc_relu1 -> enc_conv2",
                    "enc_conv2 -> bottleneck -> dec_proj",
                    "enc_conv2 + dec_proj -> concat_skip",
                    "concat_skip -> fuse_conv",
                ],
                allowed_residual_patterns=["skip concat between encoder and decoder branch"],
                training_minimums=[
                    "encoder_channels > 0",
                    "decoder_channels == encoder_channels",
                    "fusion_in_channels == encoder_channels * 2",
                ],
                input_expectation="rank-4 tensor [batch, channels, height, width]",
            ),
            "seq2seq_attention_block": TemplateContract(
                template_id="seq2seq_attention_block",
                name="Seq2Seq Attention Block",
                required_nodes=[
                    "Encoder LSTM",
                    "Decoder LSTM",
                    "Cross attention",
                    "Context residual add",
                    "Projection head",
                ],
                required_edges=[
                    "input -> encoder -> decoder",
                    "decoder -> attn -> add_ctx",
                    "decoder -> add_ctx (residual)",
                    "add_ctx -> projection",
                ],
                allowed_residual_patterns=["decoder-context add"],
                training_minimums=[
                    "encoder_hidden == decoder_hidden",
                    "hidden_dim % num_heads == 0",
                    "num_layers >= 1",
                ],
                input_expectation="rank-3 tensor [batch, sequence, features]",
            ),
            "dual_encoder_fusion_block": TemplateContract(
                template_id="dual_encoder_fusion_block",
                name="Dual Encoder Fusion Block",
                required_nodes=[
                    "Image encoder projection",
                    "Text encoder projection",
                    "Concat fusion",
                    "Fusion projection",
                    "Residual add",
                ],
                required_edges=[
                    "input -> image_proj",
                    "input -> text_proj",
                    "image_proj + text_proj -> fuse_cat",
                    "fuse_cat -> fuse_proj -> fuse_add",
                    "image_proj -> fuse_add (residual)",
                ],
                allowed_residual_patterns=["fused projection added to image branch projection"],
                training_minimums=[
                    "image_dim == text_dim",
                    "fusion_dim > 0",
                    "fuse_proj.in_features == fusion_dim * 2",
                ],
                input_expectation="rank-2 tensor [batch, features]",
            ),
            "resnet_stem": TemplateContract(
                template_id="resnet_stem",
                name="ResNet Stem",
                required_nodes=[
                    "Conv2d 7x7 stride2",
                    "BatchNorm2d",
                    "ReLU",
                    "MaxPool2d stride2",
                ],
                required_edges=[
                    "conv1 -> bn1 -> relu -> maxpool",
                ],
                allowed_residual_patterns=[],
                training_minimums=[
                    "in_channels > 0",
                    "stem_channels > 0",
                ],
                input_expectation="rank-4 tensor [batch, channels, height, width]",
            ),
            "resnet_bottleneck_block": TemplateContract(
                template_id="resnet_bottleneck_block",
                name="ResNet Bottleneck Block",
                required_nodes=[
                    "Conv1x1 reduce",
                    "Conv3x3",
                    "Conv1x1 expand",
                    "BatchNorm2d",
                    "ReLU",
                    "Residual add",
                ],
                required_edges=[
                    "conv1 -> bn1 -> relu1 -> conv2 -> bn2 -> relu2 -> conv3 -> bn3 -> add -> relu3",
                ],
                allowed_residual_patterns=["identity shortcut", "projection shortcut"],
                training_minimums=[
                    "out_channels % 4 == 0",
                    "bottleneck_channels > 0",
                    "stride in [1, 2]",
                ],
                input_expectation="rank-4 tensor [batch, channels, height, width]",
            ),
            "vgg_block": TemplateContract(
                template_id="vgg_block",
                name="VGG Conv Block",
                required_nodes=[
                    "Conv2d (x2-4)",
                    "ReLU (x2-4)",
                    "MaxPool2d",
                ],
                required_edges=[
                    "conv1 -> relu1 -> conv2 -> relu2 -> ... -> maxpool",
                ],
                allowed_residual_patterns=[],
                training_minimums=[
                    "in_channels > 0",
                    "out_channels > 0",
                    "num_convs >= 2",
                ],
                input_expectation="rank-4 tensor [batch, channels, height, width]",
            ),
            "densenet_dense_block": TemplateContract(
                template_id="densenet_dense_block",
                name="DenseNet Dense Block",
                required_nodes=[
                    "BatchNorm2d",
                    "ReLU",
                    "Conv1x1 bottleneck",
                    "Conv3x3",
                    "Concat",
                ],
                required_edges=[
                    "bn1 -> relu1 -> conv1x1 -> bn2 -> relu2 -> conv3x3 -> concat",
                    "input -> concat (dense skip)",
                ],
                allowed_residual_patterns=["dense concat from input"],
                training_minimums=[
                    "in_channels > 0",
                    "growth_rate > 0",
                    "bn_size >= 1",
                ],
                input_expectation="rank-4 tensor [batch, channels, height, width]",
            ),
            "densenet_transition": TemplateContract(
                template_id="densenet_transition",
                name="DenseNet Transition Layer",
                required_nodes=[
                    "BatchNorm2d",
                    "ReLU",
                    "Conv1x1",
                    "AvgPool2d",
                ],
                required_edges=[
                    "bn -> relu -> conv1x1 -> avgpool",
                ],
                allowed_residual_patterns=[],
                training_minimums=[
                    "in_channels > 0",
                    "compression > 0.0",
                    "compression <= 1.0",
                ],
                input_expectation="rank-4 tensor [batch, channels, height, width]",
            ),
            "mobilenet_depthwise_block": TemplateContract(
                template_id="mobilenet_depthwise_block",
                name="MobileNet Depthwise Separable Block",
                required_nodes=[
                    "Conv2d depthwise",
                    "BatchNorm2d (x2)",
                    "ReLU6 (x2)",
                    "Conv2d pointwise 1x1",
                ],
                required_edges=[
                    "dw_conv -> bn1 -> relu1 -> pw_conv -> bn2 -> relu2",
                ],
                allowed_residual_patterns=["identity shortcut when stride=1 and in==out"],
                training_minimums=[
                    "in_channels > 0",
                    "out_channels > 0",
                    "stride in [1, 2]",
                ],
                input_expectation="rank-4 tensor [batch, channels, height, width]",
            ),
            "resnet_basic_block": TemplateContract(
                template_id="resnet_basic_block",
                name="ResNet Basic Block",
                required_nodes=[
                    "Conv3x3 (x2)",
                    "BatchNorm2d (x2)",
                    "ReLU (x2)",
                    "Residual add",
                ],
                required_edges=[
                    "conv1 -> bn1 -> relu1 -> conv2 -> bn2 -> add -> relu2",
                ],
                allowed_residual_patterns=["identity shortcut", "projection shortcut"],
                training_minimums=[
                    "in_channels > 0",
                    "out_channels > 0",
                    "stride in [1, 2]",
                ],
                input_expectation="rank-4 tensor [batch, channels, height, width]",
            ),
            "gcn_layer": TemplateContract(
                template_id="gcn_layer",
                name="Graph Convolutional Layer",
                required_nodes=["Linear", "Activation"],
                required_edges=["linear -> activation"],
                allowed_residual_patterns=["identity shortcut"],
                training_minimums=["in_features > 0", "out_features > 0"],
                input_expectation="rank-2 node features [num_nodes, in_features]",
            ),
            "gat_layer": TemplateContract(
                template_id="gat_layer",
                name="Graph Attention Layer",
                required_nodes=["Linear", "LeakyReLU", "Attention softmax"],
                required_edges=["linear -> attn_score -> softmax -> aggregate"],
                allowed_residual_patterns=["identity shortcut"],
                training_minimums=["in_features > 0", "out_features > 0", "num_heads >= 1"],
                input_expectation="rank-2 node features [num_nodes, in_features]",
            ),
            "message_passing_block": TemplateContract(
                template_id="message_passing_block",
                name="Message Passing Neural Network Block",
                required_nodes=["Message fn", "Aggregate fn", "Update fn"],
                required_edges=["message -> aggregate -> update"],
                allowed_residual_patterns=["identity shortcut"],
                training_minimums=["in_features > 0", "out_features > 0"],
                input_expectation="rank-2 node features [num_nodes, in_features]",
            ),
            "mamba_block": TemplateContract(
                template_id="mamba_block",
                name="Mamba Selective State Space Block",
                required_nodes=["Linear proj_in", "Conv1d", "SSM", "Linear proj_out", "LayerNorm"],
                required_edges=["proj_in -> conv1d -> ssm -> proj_out"],
                allowed_residual_patterns=["identity shortcut"],
                training_minimums=["d_model > 0", "d_state > 0", "expand_factor >= 1"],
                input_expectation="rank-3 tensor [batch, seq_len, d_model]",
            ),
            "tcn_block": TemplateContract(
                template_id="tcn_block",
                name="Temporal Convolutional Network Block",
                required_nodes=["Conv1d dilated (x2)", "ReLU (x2)", "Dropout (x2)"],
                required_edges=["conv1 -> relu1 -> dropout1 -> conv2 -> relu2 -> dropout2 -> add"],
                allowed_residual_patterns=["identity shortcut", "projection shortcut"],
                training_minimums=["in_channels > 0", "out_channels > 0", "kernel_size >= 2"],
                input_expectation="rank-3 tensor [batch, channels, seq_len]",
            ),
            "vae_encoder_decoder": TemplateContract(
                template_id="vae_encoder_decoder",
                name="Variational Autoencoder Encoder-Decoder",
                required_nodes=["Encoder Linear (x2+)", "mu head", "logvar head", "Decoder Linear (x2+)"],
                required_edges=["encoder -> mu + logvar -> reparameterize -> decoder"],
                allowed_residual_patterns=[],
                training_minimums=["input_dim > 0", "latent_dim > 0"],
                input_expectation="rank-2 tensor [batch, input_dim]",
            ),
            "gan_pair": TemplateContract(
                template_id="gan_pair",
                name="GAN Generator-Discriminator Pair",
                required_nodes=["Generator Linear (x3+)", "Discriminator Linear (x3+)"],
                required_edges=["gen: noise -> linear -> ... -> output", "disc: input -> linear -> ... -> sigmoid"],
                allowed_residual_patterns=[],
                training_minimums=["latent_dim > 0", "output_dim > 0"],
                input_expectation="rank-2 noise vector [batch, latent_dim]",
            ),
        }
        self._load_dynamic_templates()
        self._load_builtin_expansion_templates()

    def contract(self, template_id: str) -> TemplateContract:
        if template_id not in self._contracts:
            raise ValueError(f"Unknown canonical template '{template_id}'")
        return self._contracts[template_id]

    def list_contracts(self) -> List[TemplateContract]:
        """Every block the product can actually build.

        A block whose operations the code generator has no entry for is not a template,
        it is a picture: it instantiates onto the canvas and then cannot be turned into a
        `.py`, exported, or run. 85 of these were being offered — the graph-neural-network
        family and friends, whose `neighbor_sampler` needs an edge index that a generated
        `forward(x)` has no parameter for, and whose `message_mlp` carries no width to
        build a Linear from. Offering them costs the user a choice that dead-ends.

        The rule, rather than a list of 85: a block is offered when every operation it
        instantiates to is one the generator knows. It is self-maintaining — teach the
        generator `dot_product_similarity` and the two-tower blocks reappear on their own,
        with nothing here to update.
        """
        return [
            self._contracts[key]
            for key in sorted(self._contracts.keys())
            if key in self.buildable_ids()
        ]

    def all_contracts(self) -> List[TemplateContract]:
        """Every contract, buildable or not. For diagnostics that need the whole set."""
        return [self._contracts[key] for key in sorted(self._contracts.keys())]

    def buildable_ids(self) -> Set[str]:
        """Ids whose every instantiated operation the code generator can emit.

        Computed once. Instantiating a block is dict construction — no torch, no forward —
        so the whole registry costs milliseconds. The oracle is the GENERATOR's own
        acceptance rule, not the op registry: the generator also ships shape-preserving
        implementations for custom ops (grn, sinusoidal_pos_embedding, rotary_embedding,
        adaLN), and asking `spec_for` alone wrongly drops every template built on them.
        """
        if self._buildable is not None:
            return self._buildable

        from services.code_generation import generator_supports_op  # local: avoids an import cycle

        exempt = {"input", "output", "module"}
        buildable: Set[str] = set()
        for template_id in self._contracts:
            try:
                nodes, _edges, _entry, _exit, _meta = self.instantiate(template_id, template_id, {})
            except Exception:
                continue
            ok = True
            for node in nodes:
                op = getattr(node, "op", None) or ""
                bare = str(op).split(".")[-1].lower()
                if bare in exempt:
                    continue
                if not generator_supports_op(str(op)):
                    ok = False
                    break
            if ok:
                buildable.add(template_id)
        self._buildable = buildable
        return buildable

    def instantiate(
        self,
        template_id: str,
        instance_id: str,
        config: Dict[str, Any],
    ) -> Tuple[List[ArchitectureNode], List[ArchitectureEdge], str, str, Dict[str, Any]]:
        if template_id in self._dynamic_templates:
            return self._build_dynamic_template(
                template_id=template_id,
                instance_id=instance_id,
                config=config,
            )
        if template_id == "full_transformer_block":
            return self._build_full_transformer_block(instance_id=instance_id, config=config)
        if template_id == "vit_block":
            return self._build_vit_block(instance_id=instance_id, config=config)
        if template_id == "unet_block":
            return self._build_unet_block(instance_id=instance_id, config=config)
        if template_id == "seq2seq_attention_block":
            return self._build_seq2seq_attention_block(instance_id=instance_id, config=config)
        if template_id == "dual_encoder_fusion_block":
            return self._build_dual_encoder_fusion_block(instance_id=instance_id, config=config)
        if template_id == "resnet_stem":
            return self._build_resnet_stem(instance_id=instance_id, config=config)
        if template_id == "resnet_bottleneck_block":
            return self._build_resnet_bottleneck_block(instance_id=instance_id, config=config)
        if template_id == "vgg_block":
            return self._build_vgg_block(instance_id=instance_id, config=config)
        if template_id == "densenet_dense_block":
            return self._build_densenet_dense_block(instance_id=instance_id, config=config)
        if template_id == "densenet_transition":
            return self._build_densenet_transition(instance_id=instance_id, config=config)
        if template_id == "mobilenet_depthwise_block":
            return self._build_mobilenet_depthwise_block(instance_id=instance_id, config=config)
        if template_id == "resnet_basic_block":
            return self._build_resnet_basic_block(instance_id=instance_id, config=config)
        if template_id == "gcn_layer":
            return self._build_gcn_layer(instance_id=instance_id, config=config)
        if template_id == "gat_layer":
            return self._build_gat_layer(instance_id=instance_id, config=config)
        if template_id == "message_passing_block":
            return self._build_message_passing_block(instance_id=instance_id, config=config)
        if template_id == "mamba_block":
            return self._build_mamba_block(instance_id=instance_id, config=config)
        if template_id == "tcn_block":
            return self._build_tcn_block(instance_id=instance_id, config=config)
        if template_id == "vae_encoder_decoder":
            return self._build_vae_encoder_decoder(instance_id=instance_id, config=config)
        if template_id == "gan_pair":
            return self._build_gan_pair(instance_id=instance_id, config=config)
        raise ValueError(f"Unknown canonical template '{template_id}'")

    def seed_input_shape(
        self,
        *,
        template_id: str,
        nodes: List[ArchitectureNode],
        entry_node_id: str,
        edges: Optional[List[ArchitectureEdge]] = None,
    ) -> List[int]:
        """The tensor this block's entry layer expects.

        A canonical block is a FRAGMENT: it has an entry and an exit and no Input
        node at all, so `CodeGenerationService.sample_inputs` has nothing to size
        the example tensor from and falls back to `/api/models/generate-code`'s
        ImageNet default. A token-sequence block was therefore declared as
        `torch.randn([1, 3, 224, 224])` — and since #202 the trace worker OBEYS the
        declaration rather than searching, so `nn.LayerNorm(normalized_shape=256)`,
        the first layer of these blocks, raised before anything else ran. Every node
        after it was then marked unreached, which is the "was not reached" wall the
        QA sweep saw on block after block.

        The answer was already being computed and thrown away: `_build_dynamic_template`
        sizes every layer in the block from this same seed and then drops it, because
        `instantiate` returns five values and none of them is the shape.

        Derived from the INSTANTIATED entry node rather than the template's declared
        config, so it reflects a caller's overrides: instantiating with
        `{"ln1": {"normalized_shape": 128}}` answers [1, 16, 128] where the declared
        config would still answer [1, 16, 256].
        """
        contract = self._contracts.get(template_id)
        expectation = contract.input_expectation if contract else ""
        default = self._default_shape_from_expectation(expectation)

        # Walk the chain from the entry until a node actually says something about the
        # shape. A block whose first node is `nn.Identity` — the GNN layers, the
        # DenseNet blocks — told us nothing, so the seed fell back to the expectation's
        # default and the FIRST real layer then refused it: `nn.Linear(in_features=64)`
        # handed a [1, 256] tensor is "mat1 and mat2 shapes cannot be multiplied".
        # Identity, activations and dropout do not change a shape, so reading through
        # them to the layer that does is the same question asked of a node that can
        # answer it.
        successors: Dict[str, List[str]] = {}
        for edge in edges or []:
            successors.setdefault(edge.source_node_id, []).append(edge.target_node_id)
        by_id = {n.node_id: n for n in nodes}

        seen: Set[str] = set()
        cursor = entry_node_id if entry_node_id in by_id else (nodes[0].node_id if nodes else None)
        while cursor and cursor not in seen:
            seen.add(cursor)
            node = by_id.get(cursor)
            if node is None:
                break
            # Walk past a node only when it is SHAPE-NEUTRAL and so cannot answer.
            # Not "when its answer equals the default": a Conv2d whose in_channels is
            # genuinely 3 answers [1, 3, 224, 224], which IS the default, and treating
            # that as silence walked on to a later Conv2d and seeded 61 blocks with the
            # wrong channel count. Measured: 198 clean -> 137. The question is whether
            # the node has a rule, not whether its answer is surprising.
            if _bare_op(node.op) not in _SHAPE_NEUTRAL_OPS:
                return self._seed_dynamic_shape(
                    input_expectation=expectation,
                    nodes_cfg=[{"layer_op": node.op, "default_params": node.params}],
                )
            nxt = successors.get(cursor) or []
            cursor = nxt[0] if nxt else None
        return default

    def register_dynamic_template(
        self,
        *,
        template_id: str,
        name: str,
        family: str,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        required_nodes: List[str],
        required_edges: List[str],
        allowed_residual_patterns: List[str],
        training_minimums: List[str],
        input_expectation: str,
        metadata: Optional[Dict[str, Any]] = None,
        persist: bool = True,
    ) -> None:
        if not template_id.strip():
            raise ValueError("template_id cannot be empty")
        if not nodes:
            raise ValueError("dynamic template requires at least one node")

        normalized_nodes: List[Dict[str, Any]] = []
        for node in nodes:
            suffix = str(node.get("suffix", "")).strip()
            layer_op = str(node.get("layer_op", "")).strip()
            if not suffix or not layer_op:
                raise ValueError("each dynamic node must include non-empty suffix and layer_op")
            default_params = node.get("default_params") or {}
            if not isinstance(default_params, dict):
                default_params = {}
            normalized_nodes.append(
                {
                    "suffix": suffix,
                    "layer_op": layer_op,
                    "default_params": default_params,
                }
            )

        normalized_edges: List[Dict[str, str]] = []
        for edge in edges:
            source_suffix = str(edge.get("source_suffix", "")).strip()
            target_suffix = str(edge.get("target_suffix", "")).strip()
            if not source_suffix or not target_suffix:
                continue
            normalized_edges.append(
                {
                    "source_suffix": source_suffix,
                    "target_suffix": target_suffix,
                }
            )

        if not normalized_edges:
            # Safe default chain when generator emits nodes but no edges.
            for idx in range(len(normalized_nodes) - 1):
                normalized_edges.append(
                    {
                        "source_suffix": normalized_nodes[idx]["suffix"],
                        "target_suffix": normalized_nodes[idx + 1]["suffix"],
                    }
                )

        payload = {
            "template_id": template_id,
            "name": name or template_id,
            "family": family,
            "nodes": normalized_nodes,
            "edges": normalized_edges,
            "required_nodes": required_nodes or [node["layer_op"] for node in normalized_nodes],
            "required_edges": required_edges or [
                f"{edge['source_suffix']} -> {edge['target_suffix']}" for edge in normalized_edges
            ],
            "allowed_residual_patterns": allowed_residual_patterns or [],
            "training_minimums": training_minimums or [],
            "input_expectation": input_expectation or "rank-2 tensor [batch, features]",
            "metadata": metadata or {},
        }
        self._dynamic_templates[template_id] = payload
        self._contracts[template_id] = TemplateContract(
            template_id=template_id,
            name=payload["name"],
            required_nodes=payload["required_nodes"],
            required_edges=payload["required_edges"],
            allowed_residual_patterns=payload["allowed_residual_patterns"],
            training_minimums=payload["training_minimums"],
            input_expectation=payload["input_expectation"],
        )
        if persist:
            self._persist_dynamic_templates()

    def list_dynamic_template_ids(self) -> List[str]:
        return sorted(self._dynamic_templates.keys())

    def _load_dynamic_templates(self) -> None:
        path = self._dynamic_templates_path
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return
        raw_templates = payload.get("templates", []) if isinstance(payload, dict) else payload
        if not isinstance(raw_templates, list):
            return

        for raw in raw_templates:
            if not isinstance(raw, dict):
                continue
            try:
                self.register_dynamic_template(
                    template_id=str(raw.get("template_id", "")).strip(),
                    name=str(raw.get("name", "")).strip() or str(raw.get("template_id", "")).strip(),
                    family=str(raw.get("family", "")).strip(),
                    nodes=list(raw.get("nodes") or []),
                    edges=list(raw.get("edges") or []),
                    required_nodes=list(raw.get("required_nodes") or []),
                    required_edges=list(raw.get("required_edges") or []),
                    allowed_residual_patterns=list(raw.get("allowed_residual_patterns") or []),
                    training_minimums=list(raw.get("training_minimums") or []),
                    input_expectation=str(raw.get("input_expectation", "")).strip(),
                    metadata=dict(raw.get("metadata") or {}),
                    persist=False,
                )
            except Exception:
                continue

    def _persist_dynamic_templates(self) -> None:
        path = self._dynamic_templates_path
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "templates": [
                self._dynamic_templates[template_id]
                for template_id in sorted(self._dynamic_templates.keys())
            ]
        }
        temp_path = f"{path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(temp_path, path)

    def _load_builtin_expansion_templates(self) -> None:
        try:
            from services.template_expansion_pack import register_expansion_templates
        except Exception:
            return
        try:
            register_expansion_templates(self)
        except Exception:
            # Expansion templates are additive. If loading fails, keep core registry working.
            return

    def _build_dynamic_template(
        self,
        *,
        template_id: str,
        instance_id: str,
        config: Dict[str, Any],
    ) -> Tuple[List[ArchitectureNode], List[ArchitectureEdge], str, str, Dict[str, Any]]:
        template = self._dynamic_templates.get(template_id)
        if not template:
            raise ValueError(f"Unknown canonical template '{template_id}'")

        nodes_cfg = template.get("nodes") or []
        edges_cfg = template.get("edges") or []
        if not nodes_cfg:
            raise ValueError(f"Dynamic template '{template_id}' is invalid: no nodes configured")

        override_by_suffix: Dict[str, Dict[str, Any]] = {}
        for key, value in (config or {}).items():
            if isinstance(value, dict):
                override_by_suffix[str(key)] = dict(value)
                continue
            if isinstance(key, str) and "." in key:
                suffix, param = key.split(".", 1)
                override_by_suffix.setdefault(suffix, {})[param] = value

        incoming_by_suffix: Dict[str, int] = {}
        for edge_cfg in edges_cfg:
            target_suffix = str(edge_cfg.get("target_suffix", "")).strip()
            if target_suffix:
                incoming_by_suffix[target_suffix] = incoming_by_suffix.get(target_suffix, 0) + 1

        contract = self._contracts.get(template_id)
        input_expectation = contract.input_expectation if contract else ""
        current_shape = self._seed_dynamic_shape(
            input_expectation=input_expectation,
            nodes_cfg=nodes_cfg,
        )

        nodes: List[ArchitectureNode] = []
        node_ids_in_order: List[str] = []
        for node_cfg in nodes_cfg:
            suffix = str(node_cfg.get("suffix", "")).strip()
            layer_op = str(node_cfg.get("layer_op", "")).strip()
            default_params = dict(node_cfg.get("default_params") or {})
            default_params.update(override_by_suffix.get(suffix, {}))
            current_shape = self._align_dynamic_input_shape(
                layer_op=layer_op,
                params=default_params,
                current_shape=current_shape,
            )
            default_params = self._normalize_dynamic_node_params(
                layer_op=layer_op,
                params=default_params,
                current_shape=current_shape,
            )
            node = self._node(instance_id, suffix, layer_op, default_params)
            nodes.append(node)
            node_ids_in_order.append(node.node_id)
            current_shape = self._estimate_dynamic_output_shape(
                layer_op=layer_op,
                params=default_params,
                current_shape=current_shape,
                merge_inputs=incoming_by_suffix.get(suffix, 1),
            )

        edges: List[ArchitectureEdge] = []
        incoming: Dict[str, int] = {node_id: 0 for node_id in node_ids_in_order}
        outgoing: Dict[str, int] = {node_id: 0 for node_id in node_ids_in_order}
        for edge_cfg in edges_cfg:
            source_suffix = str(edge_cfg.get("source_suffix", "")).strip()
            target_suffix = str(edge_cfg.get("target_suffix", "")).strip()
            if not source_suffix or not target_suffix:
                continue
            source_node_id = self._id(instance_id, source_suffix)
            target_node_id = self._id(instance_id, target_suffix)
            if source_node_id not in incoming or target_node_id not in incoming:
                continue
            edges.append(
                ArchitectureEdge(
                    edge_id=f"{source_node_id}:y->{target_node_id}:x",
                    source_node_id=source_node_id,
                    source_port="y",
                    target_node_id=target_node_id,
                    target_port="x",
                )
            )
            outgoing[source_node_id] += 1
            incoming[target_node_id] += 1

        # Auto-heal disconnected nodes for imperfect dynamic template definitions.
        # If a non-first node has no incoming edge, attach it to its immediate
        # predecessor in declaration order so validation can proceed.
        existing_edges = {edge.edge_id for edge in edges}
        for idx, node_id in enumerate(node_ids_in_order):
            if idx == 0 or incoming.get(node_id, 0) > 0:
                continue
            prev_node_id = node_ids_in_order[idx - 1]
            edge_id = f"{prev_node_id}:y->{node_id}:x"
            if edge_id in existing_edges:
                continue
            edges.append(
                ArchitectureEdge(
                    edge_id=edge_id,
                    source_node_id=prev_node_id,
                    source_port="y",
                    target_node_id=node_id,
                    target_port="x",
                )
            )
            existing_edges.add(edge_id)
            outgoing[prev_node_id] += 1
            incoming[node_id] += 1

        entry_node_id = next((node_id for node_id in node_ids_in_order if incoming[node_id] == 0), node_ids_in_order[0])
        exit_node_id = next((node_id for node_id in reversed(node_ids_in_order) if outgoing[node_id] == 0), node_ids_in_order[-1])
        return nodes, edges, entry_node_id, exit_node_id, dict(config or {})

    @staticmethod
    def _default_shape_from_expectation(expectation: str) -> List[int]:
        text = (expectation or "").lower()
        if "rank-4" in text:
            return [1, 3, 224, 224]
        if "rank-3" in text:
            if "tokens" in text or "sequence" in text:
                return [1, 16, 256]
            return [1, 16, 256]
        if "rank-2" in text:
            return [1, 256]
        return [1, 256]

    def _seed_dynamic_shape(self, *, input_expectation: str, nodes_cfg: List[Dict[str, Any]]) -> List[int]:
        current_shape = self._default_shape_from_expectation(input_expectation)
        if not nodes_cfg:
            return current_shape

        first = dict(nodes_cfg[0] or {})
        op = str(first.get("layer_op", "")).strip()
        params = dict(first.get("default_params") or {})
        text = (input_expectation or "").lower()

        in_channels = params.get("in_channels")
        if op in {"nn.Conv2d", "Conv2d"} and isinstance(in_channels, int) and in_channels > 0:
            return [1, in_channels, 224, 224]

        if op in {"nn.Conv1d", "Conv1d"} and isinstance(in_channels, int) and in_channels > 0:
            return [1, in_channels, 256]

        # Normalisation layers state their own channel count, and several blocks open
        # on one — DenseNet's transition begins `nn.BatchNorm2d(num_features=128)`.
        # Without a rule here the seed fell back to the expectation's ImageNet default
        # and the layer refused it: "running_mean should contain 3 elements not 128".
        channels = params.get("num_features")
        if channels is None:
            channels = params.get("num_channels")
        if (
            op in {
                "nn.BatchNorm2d", "BatchNorm2d", "nn.BatchNorm3d", "BatchNorm3d",
                "nn.GroupNorm", "GroupNorm", "nn.InstanceNorm2d", "InstanceNorm2d",
            }
            and isinstance(channels, int)
            and channels > 0
        ):
            return [1, channels, 224, 224]
        if (
            op in {"nn.BatchNorm1d", "BatchNorm1d", "nn.InstanceNorm1d", "InstanceNorm1d"}
            and isinstance(channels, int)
            and channels > 0
        ):
            # BatchNorm1d takes (N, C) or (N, C, L); a rank-2 block wants the former.
            return [1, channels] if "rank-2" in text else [1, channels, 256]

        in_features = params.get("in_features")
        if op in {"nn.Linear", "Linear"} and isinstance(in_features, int) and in_features > 0:
            if "rank-3" in text or "tokens" in text or "sequence" in text:
                return [1, 16, in_features]
            return [1, in_features]

        normalized_shape = params.get("normalized_shape")
        if op in {"nn.LayerNorm", "LayerNorm"}:
            if isinstance(normalized_shape, int) and normalized_shape > 0:
                if "rank-3" in text or "tokens" in text or "sequence" in text:
                    return [1, 16, normalized_shape]
                return [1, normalized_shape]
            if isinstance(normalized_shape, (list, tuple)) and normalized_shape:
                dims = [int(v) for v in normalized_shape if isinstance(v, int) and v > 0]
                if dims:
                    if len(dims) == 1:
                        if "rank-3" in text or "tokens" in text or "sequence" in text:
                            return [1, 16, dims[0]]
                        return [1, dims[0]]
                    return [1] + dims

        embed_dim = params.get("embed_dim")
        if op in {"nn.MultiheadAttention", "MultiheadAttention"} and isinstance(embed_dim, int) and embed_dim > 0:
            return [1, 16, embed_dim]

        input_size = params.get("input_size")
        if op in {"nn.LSTM", "LSTM", "nn.GRU", "GRU", "nn.RNN", "RNN"} and isinstance(input_size, int) and input_size > 0:
            return [1, 16, input_size]

        # An Embedding is the one layer whose input is not shaped like its output: it
        # TAKES a tensor of indices, [batch, tokens], and produces [batch, tokens, dim].
        # Seeding with the output shape handed a rank-3 float tensor to a lookup that
        # wants rank-2 integers, which torch reports as "Expected tensor for argument
        # #1 'indices' to have one of the following scalar types: Long, Int".
        if op in {"nn.Embedding", "Embedding"}:
            return [1, 16]

        return current_shape

    @staticmethod
    def _align_dynamic_input_shape(
        *,
        layer_op: str,
        params: Dict[str, Any],
        current_shape: List[int],
    ) -> List[int]:
        shape = list(current_shape or [1, 256])
        op = layer_op.replace("torch.", "nn.") if layer_op.startswith("torch.") else layer_op

        if op in {"nn.Linear", "Linear"}:
            expected = params.get("in_features")
            if isinstance(expected, int) and expected > 0:
                if len(shape) >= 1:
                    shape[-1] = expected
                else:
                    shape = [1, expected]
            return shape

        if op in {"nn.LayerNorm", "LayerNorm"}:
            normalized = params.get("normalized_shape")
            if isinstance(normalized, int) and normalized > 0:
                if len(shape) >= 1:
                    shape[-1] = normalized
                else:
                    shape = [1, normalized]
            elif isinstance(normalized, (list, tuple)) and normalized:
                dims = [int(v) for v in normalized if isinstance(v, int) and v > 0]
                if dims:
                    if len(dims) == 1:
                        if len(shape) >= 1:
                            shape[-1] = dims[0]
                        else:
                            shape = [1, dims[0]]
                    else:
                        shape = [shape[0] if shape else 1] + dims
            return shape

        if op in {"nn.MultiheadAttention", "MultiheadAttention"}:
            embed_dim = params.get("embed_dim")
            if isinstance(embed_dim, int) and embed_dim > 0:
                if len(shape) == 3:
                    return [shape[0], shape[1], embed_dim]
                return [shape[0] if shape else 1, 16, embed_dim]
            return shape

        if op in {"nn.Conv2d", "Conv2d", "nn.BatchNorm2d", "BatchNorm2d", "nn.GroupNorm", "GroupNorm"}:
            channels = params.get("in_channels")
            if op in {"nn.BatchNorm2d", "BatchNorm2d", "nn.GroupNorm", "GroupNorm"}:
                channels = params.get("num_features") or params.get("num_channels") or channels
            if isinstance(channels, int) and channels > 0:
                if len(shape) == 4:
                    return [shape[0], channels, shape[2], shape[3]]
                return [shape[0] if shape else 1, channels, 224, 224]
            return shape

        if op in {"nn.Conv1d", "Conv1d"}:
            channels = params.get("in_channels")
            if (not isinstance(channels, int) or channels <= 0) and len(shape) == 3:
                # For rank-3 sequence tensors [B, T, F], treat features as channels.
                channels = shape[-1]
            if isinstance(channels, int) and channels > 0:
                if len(shape) == 3:
                    return [shape[0], shape[1], channels]
                return [shape[0] if shape else 1, channels, 256]
            return shape

        if op in {"nn.ConvTranspose2d", "ConvTranspose2d"}:
            channels = params.get("in_channels")
            if isinstance(channels, int) and channels > 0:
                if len(shape) == 4:
                    return [shape[0], channels, shape[2], shape[3]]
                return [shape[0] if shape else 1, channels, 56, 56]
            if len(shape) == 4:
                return shape
            return [shape[0] if shape else 1, 32, 56, 56]

        if op in {"nn.LSTM", "LSTM", "nn.GRU", "GRU", "nn.RNN", "RNN"}:
            input_size = params.get("input_size")
            if isinstance(input_size, int) and input_size > 0:
                if len(shape) == 3:
                    return [shape[0], shape[1], input_size]
                return [shape[0] if shape else 1, 16, input_size]
            return shape

        if op in {"nn.Embedding", "Embedding"}:
            embedding_dim = params.get("embedding_dim")
            if isinstance(embedding_dim, int) and embedding_dim > 0:
                if len(shape) == 3:
                    return [shape[0], shape[1], embedding_dim]
                return [shape[0] if shape else 1, 16, embedding_dim]
            return shape

        return shape

    @staticmethod
    def _best_num_heads(embed_dim: int) -> int:
        if embed_dim <= 0:
            return 1
        for candidate in (16, 12, 8, 6, 4, 3, 2):
            if embed_dim % candidate == 0:
                return candidate
        return 1

    @staticmethod
    def _best_group_divisor(num_channels: int, cap: int = 32) -> int:
        if num_channels <= 0:
            return 1
        upper = min(cap, num_channels)
        for groups in range(upper, 0, -1):
            if num_channels % groups == 0:
                return groups
        return 1

    def _normalize_dynamic_node_params(
        self,
        *,
        layer_op: str,
        params: Dict[str, Any],
        current_shape: List[int],
    ) -> Dict[str, Any]:
        normalized = dict(params or {})
        op = layer_op.replace("torch.", "nn.") if layer_op.startswith("torch.") else layer_op
        if op == "rmsnorm":
            op = "nn.RMSNorm"
        feature_dim = current_shape[-1] if current_shape else 256
        channel_dim = current_shape[1] if len(current_shape) > 1 else feature_dim

        if op in {"nn.Linear", "Linear"}:
            if not isinstance(normalized.get("in_features"), int) or normalized["in_features"] <= 0:
                normalized["in_features"] = int(feature_dim)
            if not isinstance(normalized.get("out_features"), int) or normalized["out_features"] <= 0:
                normalized["out_features"] = int(normalized["in_features"])

        # RMSNorm shares LayerNorm's normalized_shape semantics; without this
        # branch covering it, rmsnorm templates leave normalized_shape unset and
        # fail canonical validation.
        if op in {"nn.LayerNorm", "LayerNorm", "nn.RMSNorm", "RMSNorm"}:
            shape = normalized.get("normalized_shape")
            if isinstance(shape, int) and shape > 0:
                pass
            elif isinstance(shape, (list, tuple)) and len(shape) > 0 and all(isinstance(v, int) and v > 0 for v in shape):
                pass
            else:
                normalized["normalized_shape"] = [int(feature_dim)]

        if op in {"nn.Embedding", "Embedding"}:
            if not isinstance(normalized.get("num_embeddings"), int) or normalized["num_embeddings"] <= 0:
                normalized["num_embeddings"] = 32000
            if not isinstance(normalized.get("embedding_dim"), int) or normalized["embedding_dim"] <= 0:
                normalized["embedding_dim"] = int(feature_dim if feature_dim > 0 else 256)

        if op in {"nn.GroupNorm", "GroupNorm"}:
            if not isinstance(normalized.get("num_channels"), int) or normalized["num_channels"] <= 0:
                normalized["num_channels"] = int(channel_dim if channel_dim > 0 else 32)
            channels = int(normalized["num_channels"])
            groups = normalized.get("num_groups")
            if not isinstance(groups, int) or groups <= 0 or channels % groups != 0:
                normalized["num_groups"] = self._best_group_divisor(channels)

        if op in {"nn.Conv2d", "Conv2d"}:
            if len(current_shape) == 4:
                if not isinstance(normalized.get("in_channels"), int) or normalized["in_channels"] <= 0:
                    normalized["in_channels"] = int(current_shape[1])
            if not isinstance(normalized.get("out_channels"), int) or normalized["out_channels"] <= 0:
                normalized["out_channels"] = int(normalized.get("in_channels", 32))
            if "kernel_size" not in normalized:
                normalized["kernel_size"] = 3
            # If groups is set, ensure it divides both in/out channels by snapping it down.
            groups = normalized.get("groups")
            if isinstance(groups, int) and groups > 1:
                in_ch = normalized.get("in_channels")
                out_ch = normalized.get("out_channels")
                effective_groups = groups
                if isinstance(in_ch, int) and in_ch % effective_groups != 0:
                    effective_groups = self._best_group_divisor(in_ch, effective_groups)
                if isinstance(out_ch, int) and out_ch % effective_groups != 0:
                    effective_groups = self._best_group_divisor(out_ch, effective_groups)
                if effective_groups != groups:
                    normalized["groups"] = effective_groups

        if op in {"nn.Conv1d", "Conv1d"}:
            if len(current_shape) == 3:
                if not isinstance(normalized.get("in_channels"), int) or normalized["in_channels"] <= 0:
                    normalized["in_channels"] = int(current_shape[-1])
            if not isinstance(normalized.get("out_channels"), int) or normalized["out_channels"] <= 0:
                normalized["out_channels"] = int(normalized.get("in_channels", 64))
            if "kernel_size" not in normalized:
                normalized["kernel_size"] = 3
            normalized.setdefault("padding", 1)

        if op in {"nn.ConvTranspose2d", "ConvTranspose2d"}:
            if len(current_shape) == 4:
                if not isinstance(normalized.get("in_channels"), int) or normalized["in_channels"] <= 0:
                    normalized["in_channels"] = int(current_shape[1])
            if not isinstance(normalized.get("out_channels"), int) or normalized["out_channels"] <= 0:
                normalized["out_channels"] = int(normalized.get("in_channels", 32))
            if "kernel_size" not in normalized:
                normalized["kernel_size"] = 3

        if op in {"nn.BatchNorm2d", "BatchNorm2d"}:
            if len(current_shape) == 4:
                if not isinstance(normalized.get("num_features"), int) or normalized["num_features"] <= 0:
                    normalized["num_features"] = int(current_shape[1])

        if op in {"nn.BatchNorm1d", "BatchNorm1d"}:
            if len(current_shape) == 3:
                if not isinstance(normalized.get("num_features"), int) or normalized["num_features"] <= 0:
                    normalized["num_features"] = int(current_shape[-1])

        if op in {"nn.LSTM", "LSTM", "nn.GRU", "GRU", "nn.RNN", "RNN"}:
            if not isinstance(normalized.get("input_size"), int) or normalized["input_size"] <= 0:
                normalized["input_size"] = int(feature_dim if feature_dim > 0 else 128)
            if not isinstance(normalized.get("hidden_size"), int) or normalized["hidden_size"] <= 0:
                normalized["hidden_size"] = int(normalized["input_size"])
            if "batch_first" not in normalized:
                normalized["batch_first"] = True

        if op in {"nn.MultiheadAttention", "MultiheadAttention"}:
            if not isinstance(normalized.get("embed_dim"), int) or normalized["embed_dim"] <= 0:
                normalized["embed_dim"] = int(feature_dim if feature_dim > 0 else 256)
            embed_dim = int(normalized["embed_dim"])
            heads = normalized.get("num_heads")
            if not isinstance(heads, int) or heads <= 0 or embed_dim % heads != 0:
                normalized["num_heads"] = self._best_num_heads(embed_dim)

        return normalized

    @staticmethod
    def _estimate_dynamic_output_shape(
        *,
        layer_op: str,
        params: Dict[str, Any],
        current_shape: List[int],
        merge_inputs: int = 1,
    ) -> List[int]:
        op = layer_op.replace("torch.", "nn.") if layer_op.startswith("torch.") else layer_op
        if op in {"nn.Linear", "Linear"} and current_shape:
            out_features = params.get("out_features")
            if isinstance(out_features, int) and out_features > 0:
                return current_shape[:-1] + [out_features]
        if op in {"nn.LayerNorm", "LayerNorm", "nn.GroupNorm", "GroupNorm", "nn.BatchNorm2d", "BatchNorm2d"}:
            return current_shape
        if op in {"nn.Embedding", "Embedding"} and current_shape:
            embedding_dim = params.get("embedding_dim")
            if isinstance(embedding_dim, int) and embedding_dim > 0:
                return current_shape + [embedding_dim]
        if op in {"nn.LSTM", "LSTM", "nn.GRU", "GRU", "nn.RNN", "RNN"} and current_shape:
            hidden_size = params.get("hidden_size")
            if not isinstance(hidden_size, int) or hidden_size <= 0:
                return current_shape
            # A bidirectional RNN concatenates the two directions, so its output width
            # is TWICE hidden_size. Without this every `*_bi_block` sized the projection
            # that follows it at hidden_size and the generated file died on the first
            # matmul: "mat1 and mat2 shapes cannot be multiplied (16x512 and 256x256)".
            if params.get("bidirectional"):
                hidden_size *= 2
            if len(current_shape) == 3:
                return [current_shape[0], current_shape[1], hidden_size]
            if len(current_shape) == 2:
                return [current_shape[0], 1, hidden_size]
        if op in {"nn.Conv2d", "Conv2d"} and len(current_shape) == 4:
            out_channels = params.get("out_channels")
            if isinstance(out_channels, int) and out_channels > 0:
                return [current_shape[0], out_channels, current_shape[2], current_shape[3]]
        if op in {"nn.Conv1d", "Conv1d"} and len(current_shape) == 3:
            out_channels = params.get("out_channels")
            if isinstance(out_channels, int) and out_channels > 0:
                return [current_shape[0], current_shape[1], out_channels]
        if op in {"nn.ConvTranspose2d", "ConvTranspose2d"} and len(current_shape) == 4:
            out_channels = params.get("out_channels")
            if isinstance(out_channels, int) and out_channels > 0:
                return [current_shape[0], out_channels, current_shape[2], current_shape[3]]
        if op in {"nn.MultiheadAttention", "MultiheadAttention"}:
            return current_shape
        if layer_op == "torch.cat" or op == "nn.cat":
            dim = params.get("dim", 1)
            if not isinstance(dim, int):
                return current_shape
            if not current_shape:
                return current_shape
            out = list(current_shape)
            rank = len(out)
            resolved_dim = dim if dim >= 0 else rank + dim
            if resolved_dim < 0 or resolved_dim >= rank:
                return current_shape
            if merge_inputs > 1:
                out[resolved_dim] = max(1, int(out[resolved_dim])) * int(merge_inputs)
            return out
        return current_shape

    def _build_full_transformer_block(
        self, instance_id: str, config: Dict[str, Any]
    ) -> Tuple[List[ArchitectureNode], List[ArchitectureEdge], str, str, Dict[str, Any]]:
        d_model = int(config.get("d_model", 128))
        num_heads = int(config.get("num_heads", 8))
        ffn_dim = int(config.get("ffn_dim", 256))
        dropout = float(config.get("dropout", 0.1))
        self._enforce(
            "full_transformer_block",
            [
                (d_model > 0, "d_model must be > 0"),
                (num_heads > 0, "num_heads must be > 0"),
                (d_model % num_heads == 0, "d_model must be divisible by num_heads"),
                (ffn_dim >= d_model, "ffn_dim must be >= d_model"),
                (0.0 <= dropout <= 0.5, "dropout must be between 0.0 and 0.5"),
            ],
        )
        cfg = {"d_model": d_model, "num_heads": num_heads, "ffn_dim": ffn_dim, "dropout": dropout}

        nodes = [
            self._node(instance_id, "norm1", "nn.LayerNorm", {"normalized_shape": d_model}),
            self._node(instance_id, "attn", "nn.MultiheadAttention", {"embed_dim": d_model, "num_heads": num_heads}),
            self._node(instance_id, "add_attn", "torch.add", {}),
            self._node(instance_id, "norm2", "nn.LayerNorm", {"normalized_shape": d_model}),
            self._node(instance_id, "mlp1", "nn.Linear", {"in_features": d_model, "out_features": ffn_dim}),
            self._node(instance_id, "gelu", "nn.GELU", {}),
            self._node(instance_id, "mlp2", "nn.Linear", {"in_features": ffn_dim, "out_features": d_model}),
            self._node(instance_id, "add_mlp", "torch.add", {}),
        ]
        edges = self._edges(
            instance_id,
            [
                ("norm1", "attn"),
                ("attn", "add_attn"),
                ("norm2", "mlp1"),
                ("mlp1", "gelu"),
                ("gelu", "mlp2"),
                ("mlp2", "add_mlp"),
                ("add_attn", "norm2"),
                ("add_attn", "add_mlp"),
            ],
        )
        return nodes, edges, self._id(instance_id, "norm1"), self._id(instance_id, "add_mlp"), cfg

    def _build_vit_block(
        self, instance_id: str, config: Dict[str, Any]
    ) -> Tuple[List[ArchitectureNode], List[ArchitectureEdge], str, str, Dict[str, Any]]:
        patch_dim = int(config.get("patch_dim", 64))
        d_model = int(config.get("d_model", 128))
        num_heads = int(config.get("num_heads", 8))
        mlp_dim = int(config.get("mlp_dim", 256))
        self._enforce(
            "vit_block",
            [
                (patch_dim > 0, "patch_dim must be > 0"),
                (d_model > 0, "d_model must be > 0"),
                (num_heads > 0, "num_heads must be > 0"),
                (d_model % num_heads == 0, "d_model must be divisible by num_heads"),
                (mlp_dim >= d_model, "mlp_dim must be >= d_model"),
            ],
        )
        cfg = {"patch_dim": patch_dim, "d_model": d_model, "num_heads": num_heads, "mlp_dim": mlp_dim}

        nodes = [
            self._node(instance_id, "patch_embed", "nn.Linear", {"in_features": patch_dim, "out_features": d_model}),
            self._node(instance_id, "norm1", "nn.LayerNorm", {"normalized_shape": d_model}),
            self._node(instance_id, "attn", "nn.MultiheadAttention", {"embed_dim": d_model, "num_heads": num_heads}),
            self._node(instance_id, "add_attn", "torch.add", {}),
            self._node(instance_id, "norm2", "nn.LayerNorm", {"normalized_shape": d_model}),
            self._node(instance_id, "mlp1", "nn.Linear", {"in_features": d_model, "out_features": mlp_dim}),
            self._node(instance_id, "gelu", "nn.GELU", {}),
            self._node(instance_id, "mlp2", "nn.Linear", {"in_features": mlp_dim, "out_features": d_model}),
            self._node(instance_id, "add_mlp", "torch.add", {}),
        ]
        edges = self._edges(
            instance_id,
            [
                ("patch_embed", "norm1"),
                ("norm1", "attn"),
                ("attn", "add_attn"),
                ("patch_embed", "add_attn"),
                ("add_attn", "norm2"),
                ("norm2", "mlp1"),
                ("mlp1", "gelu"),
                ("gelu", "mlp2"),
                ("mlp2", "add_mlp"),
                ("add_attn", "add_mlp"),
            ],
        )
        return nodes, edges, self._id(instance_id, "patch_embed"), self._id(instance_id, "add_mlp"), cfg

    def _build_unet_block(
        self, instance_id: str, config: Dict[str, Any]
    ) -> Tuple[List[ArchitectureNode], List[ArchitectureEdge], str, str, Dict[str, Any]]:
        in_channels = int(config.get("in_channels", 3))
        encoder_channels = int(config.get("encoder_channels", 32))
        bottleneck_channels = int(config.get("bottleneck_channels", 64))
        decoder_channels = int(config.get("decoder_channels", 32))
        fusion_out_channels = int(config.get("fusion_out_channels", 32))
        self._enforce(
            "unet_block",
            [
                (in_channels > 0, "in_channels must be > 0"),
                (encoder_channels > 0, "encoder_channels must be > 0"),
                (bottleneck_channels > 0, "bottleneck_channels must be > 0"),
                (decoder_channels == encoder_channels, "decoder_channels must equal encoder_channels"),
                (fusion_out_channels > 0, "fusion_out_channels must be > 0"),
            ],
        )
        cfg = {
            "in_channels": in_channels,
            "encoder_channels": encoder_channels,
            "bottleneck_channels": bottleneck_channels,
            "decoder_channels": decoder_channels,
            "fusion_out_channels": fusion_out_channels,
        }

        nodes = [
            self._node(instance_id, "enc_conv1", "nn.Conv2d", {"in_channels": in_channels, "out_channels": encoder_channels, "kernel_size": 3, "padding": 1}),
            self._node(instance_id, "enc_relu1", "nn.ReLU", {}),
            self._node(instance_id, "enc_conv2", "nn.Conv2d", {"in_channels": encoder_channels, "out_channels": encoder_channels, "kernel_size": 3, "padding": 1}),
            self._node(instance_id, "bottleneck", "nn.Conv2d", {"in_channels": encoder_channels, "out_channels": bottleneck_channels, "kernel_size": 3, "padding": 1}),
            self._node(instance_id, "dec_proj", "nn.Conv2d", {"in_channels": bottleneck_channels, "out_channels": decoder_channels, "kernel_size": 1}),
            self._node(instance_id, "concat_skip", "torch.cat", {"dim": 1}),
            self._node(instance_id, "fuse_conv", "nn.Conv2d", {"in_channels": encoder_channels + decoder_channels, "out_channels": fusion_out_channels, "kernel_size": 3, "padding": 1}),
        ]
        edges = self._edges(
            instance_id,
            [
                ("enc_conv1", "enc_relu1"),
                ("enc_relu1", "enc_conv2"),
                ("enc_conv2", "bottleneck"),
                ("bottleneck", "dec_proj"),
                ("enc_conv2", "concat_skip"),
                ("dec_proj", "concat_skip"),
                ("concat_skip", "fuse_conv"),
            ],
        )
        return nodes, edges, self._id(instance_id, "enc_conv1"), self._id(instance_id, "fuse_conv"), cfg

    def _build_seq2seq_attention_block(
        self, instance_id: str, config: Dict[str, Any]
    ) -> Tuple[List[ArchitectureNode], List[ArchitectureEdge], str, str, Dict[str, Any]]:
        input_size = int(config.get("input_size", 128))
        encoder_hidden = int(config.get("encoder_hidden", 128))
        decoder_hidden = int(config.get("decoder_hidden", 128))
        num_layers = int(config.get("num_layers", 2))
        num_heads = int(config.get("num_heads", 8))
        self._enforce(
            "seq2seq_attention_block",
            [
                (input_size > 0, "input_size must be > 0"),
                (encoder_hidden > 0, "encoder_hidden must be > 0"),
                (decoder_hidden > 0, "decoder_hidden must be > 0"),
                (encoder_hidden == decoder_hidden, "encoder_hidden must equal decoder_hidden"),
                (num_layers >= 1, "num_layers must be >= 1"),
                (num_heads > 0, "num_heads must be > 0"),
                (decoder_hidden % num_heads == 0, "decoder_hidden must be divisible by num_heads"),
            ],
        )
        cfg = {
            "input_size": input_size,
            "encoder_hidden": encoder_hidden,
            "decoder_hidden": decoder_hidden,
            "num_layers": num_layers,
            "num_heads": num_heads,
        }

        nodes = [
            self._node(instance_id, "encoder", "nn.LSTM", {"input_size": input_size, "hidden_size": encoder_hidden, "num_layers": num_layers, "batch_first": True}),
            self._node(instance_id, "decoder", "nn.LSTM", {"input_size": encoder_hidden, "hidden_size": decoder_hidden, "num_layers": num_layers, "batch_first": True}),
            self._node(instance_id, "attn", "nn.MultiheadAttention", {"embed_dim": decoder_hidden, "num_heads": num_heads}),
            self._node(instance_id, "add_ctx", "torch.add", {}),
            self._node(instance_id, "projection", "nn.Linear", {"in_features": decoder_hidden, "out_features": decoder_hidden}),
        ]
        edges = self._edges(
            instance_id,
            [
                ("encoder", "decoder"),
                ("decoder", "attn"),
                ("attn", "add_ctx"),
                ("decoder", "add_ctx"),
                ("add_ctx", "projection"),
            ],
        )
        return nodes, edges, self._id(instance_id, "encoder"), self._id(instance_id, "projection"), cfg

    def _build_dual_encoder_fusion_block(
        self, instance_id: str, config: Dict[str, Any]
    ) -> Tuple[List[ArchitectureNode], List[ArchitectureEdge], str, str, Dict[str, Any]]:
        image_dim = int(config.get("image_dim", 256))
        text_dim = int(config.get("text_dim", 256))
        fusion_dim = int(config.get("fusion_dim", 256))
        self._enforce(
            "dual_encoder_fusion_block",
            [
                (image_dim > 0, "image_dim must be > 0"),
                (text_dim > 0, "text_dim must be > 0"),
                (image_dim == text_dim, "image_dim must equal text_dim"),
                (fusion_dim > 0, "fusion_dim must be > 0"),
            ],
        )
        cfg = {"image_dim": image_dim, "text_dim": text_dim, "fusion_dim": fusion_dim}

        nodes = [
            self._node(instance_id, "image_proj", "nn.Linear", {"in_features": image_dim, "out_features": fusion_dim}),
            self._node(instance_id, "text_proj", "nn.Linear", {"in_features": text_dim, "out_features": fusion_dim}),
            self._node(instance_id, "fuse_cat", "torch.cat", {"dim": -1}),
            self._node(instance_id, "fuse_proj", "nn.Linear", {"in_features": fusion_dim * 2, "out_features": fusion_dim}),
            self._node(instance_id, "fuse_add", "torch.add", {}),
        ]
        edges = self._edges(
            instance_id,
            [
                ("image_proj", "fuse_cat"),
                ("text_proj", "fuse_cat"),
                ("fuse_cat", "fuse_proj"),
                ("fuse_proj", "fuse_add"),
                ("image_proj", "fuse_add"),
            ],
        )
        return nodes, edges, self._id(instance_id, "image_proj"), self._id(instance_id, "fuse_add"), cfg

    def _build_resnet_stem(
        self, instance_id: str, config: Dict[str, Any]
    ) -> Tuple[List[ArchitectureNode], List[ArchitectureEdge], str, str, Dict[str, Any]]:
        in_channels = int(config.get("in_channels", 3))
        stem_channels = int(config.get("stem_channels", 64))
        self._enforce(
            "resnet_stem",
            [
                (in_channels > 0, "in_channels must be > 0"),
                (stem_channels > 0, "stem_channels must be > 0"),
            ],
        )
        cfg = {"in_channels": in_channels, "stem_channels": stem_channels}
        nodes = [
            self._node(instance_id, "conv1", "nn.Conv2d", {"in_channels": in_channels, "out_channels": stem_channels, "kernel_size": 7, "stride": 2, "padding": 3, "bias": False}),
            self._node(instance_id, "bn1", "nn.BatchNorm2d", {"num_features": stem_channels}),
            self._node(instance_id, "relu", "nn.ReLU", {"inplace": True}),
            self._node(instance_id, "maxpool", "nn.MaxPool2d", {"kernel_size": 3, "stride": 2, "padding": 1}),
        ]
        edges = self._edges(
            instance_id,
            [
                ("conv1", "bn1"),
                ("bn1", "relu"),
                ("relu", "maxpool"),
            ],
        )
        return nodes, edges, self._id(instance_id, "conv1"), self._id(instance_id, "maxpool"), cfg

    def _build_resnet_bottleneck_block(
        self, instance_id: str, config: Dict[str, Any]
    ) -> Tuple[List[ArchitectureNode], List[ArchitectureEdge], str, str, Dict[str, Any]]:
        in_channels = int(config.get("in_channels", 256))
        bottleneck_channels = int(config.get("bottleneck_channels", 64))
        out_channels = int(config.get("out_channels", 256))
        stride = int(config.get("stride", 1))
        projection = bool(config.get("projection", in_channels != out_channels or stride != 1))
        self._enforce(
            "resnet_bottleneck_block",
            [
                (in_channels > 0, "in_channels must be > 0"),
                (bottleneck_channels > 0, "bottleneck_channels must be > 0"),
                (out_channels > 0, "out_channels must be > 0"),
                (out_channels % 4 == 0, "out_channels must be divisible by 4"),
                (stride in {1, 2}, "stride must be 1 or 2"),
            ],
        )
        cfg = {
            "in_channels": in_channels,
            "bottleneck_channels": bottleneck_channels,
            "out_channels": out_channels,
            "stride": stride,
            "projection": projection,
        }

        nodes = [
            self._node(instance_id, "identity", "nn.Identity", {}),
            self._node(instance_id, "conv1", "nn.Conv2d", {"in_channels": in_channels, "out_channels": bottleneck_channels, "kernel_size": 1, "stride": 1, "bias": False}),
            self._node(instance_id, "bn1", "nn.BatchNorm2d", {"num_features": bottleneck_channels}),
            self._node(instance_id, "relu1", "nn.ReLU", {"inplace": True}),
            self._node(instance_id, "conv2", "nn.Conv2d", {"in_channels": bottleneck_channels, "out_channels": bottleneck_channels, "kernel_size": 3, "stride": stride, "padding": 1, "bias": False}),
            self._node(instance_id, "bn2", "nn.BatchNorm2d", {"num_features": bottleneck_channels}),
            self._node(instance_id, "relu2", "nn.ReLU", {"inplace": True}),
            self._node(instance_id, "conv3", "nn.Conv2d", {"in_channels": bottleneck_channels, "out_channels": out_channels, "kernel_size": 1, "stride": 1, "bias": False}),
            self._node(instance_id, "bn3", "nn.BatchNorm2d", {"num_features": out_channels}),
            self._node(instance_id, "add", "torch.add", {}),
            self._node(instance_id, "relu3", "nn.ReLU", {"inplace": True}),
        ]

        if projection:
            nodes.extend(
                [
                    self._node(instance_id, "proj_conv", "nn.Conv2d", {"in_channels": in_channels, "out_channels": out_channels, "kernel_size": 1, "stride": stride, "bias": False}),
                    self._node(instance_id, "proj_bn", "nn.BatchNorm2d", {"num_features": out_channels}),
                ]
            )

        edges = self._edges(
            instance_id,
            [
                ("identity", "conv1"),
                ("conv1", "bn1"),
                ("bn1", "relu1"),
                ("relu1", "conv2"),
                ("conv2", "bn2"),
                ("bn2", "relu2"),
                ("relu2", "conv3"),
                ("conv3", "bn3"),
                ("bn3", "add"),
                ("add", "relu3"),
            ],
        )
        if projection:
            edges.extend(
                self._edges(
                    instance_id,
                    [
                        ("identity", "proj_conv"),
                        ("proj_conv", "proj_bn"),
                        ("proj_bn", "add"),
                    ],
                )
            )
        else:
            edges.extend(self._edges(instance_id, [("identity", "add")]))

        return nodes, edges, self._id(instance_id, "identity"), self._id(instance_id, "relu3"), cfg

    def _build_vgg_block(
        self, instance_id: str, config: Dict[str, Any]
    ) -> Tuple[List[ArchitectureNode], List[ArchitectureEdge], str, str, Dict[str, Any]]:
        in_channels = int(config.get("in_channels", 3))
        out_channels = int(config.get("out_channels", 64))
        num_convs = int(config.get("num_convs", 2))
        self._enforce(
            "vgg_block",
            [
                (in_channels > 0, "in_channels must be > 0"),
                (out_channels > 0, "out_channels must be > 0"),
                (num_convs >= 2, "num_convs must be >= 2"),
            ],
        )
        cfg = {"in_channels": in_channels, "out_channels": out_channels, "num_convs": num_convs}

        nodes: List[ArchitectureNode] = []
        link_pairs: List[Tuple[str, str]] = []
        prev_channels = in_channels
        for i in range(1, num_convs + 1):
            conv_id = f"conv{i}"
            relu_id = f"relu{i}"
            nodes.append(self._node(instance_id, conv_id, "nn.Conv2d", {"in_channels": prev_channels, "out_channels": out_channels, "kernel_size": 3, "padding": 1}))
            nodes.append(self._node(instance_id, relu_id, "nn.ReLU", {"inplace": True}))
            link_pairs.append((conv_id, relu_id))
            if i > 1:
                link_pairs.append((f"relu{i - 1}", conv_id))
            prev_channels = out_channels
        nodes.append(self._node(instance_id, "maxpool", "nn.MaxPool2d", {"kernel_size": 2, "stride": 2}))
        link_pairs.append((f"relu{num_convs}", "maxpool"))

        edges = self._edges(instance_id, link_pairs)
        return nodes, edges, self._id(instance_id, "conv1"), self._id(instance_id, "maxpool"), cfg

    def _build_densenet_dense_block(
        self, instance_id: str, config: Dict[str, Any]
    ) -> Tuple[List[ArchitectureNode], List[ArchitectureEdge], str, str, Dict[str, Any]]:
        in_channels = int(config.get("in_channels", 64))
        growth_rate = int(config.get("growth_rate", 32))
        bn_size = int(config.get("bn_size", 4))
        self._enforce(
            "densenet_dense_block",
            [
                (in_channels > 0, "in_channels must be > 0"),
                (growth_rate > 0, "growth_rate must be > 0"),
                (bn_size >= 1, "bn_size must be >= 1"),
            ],
        )
        cfg = {"in_channels": in_channels, "growth_rate": growth_rate, "bn_size": bn_size}
        inter_channels = bn_size * growth_rate

        nodes = [
            self._node(instance_id, "identity", "nn.Identity", {}),
            self._node(instance_id, "bn1", "nn.BatchNorm2d", {"num_features": in_channels}),
            self._node(instance_id, "relu1", "nn.ReLU", {"inplace": True}),
            self._node(instance_id, "conv1x1", "nn.Conv2d", {"in_channels": in_channels, "out_channels": inter_channels, "kernel_size": 1, "bias": False}),
            self._node(instance_id, "bn2", "nn.BatchNorm2d", {"num_features": inter_channels}),
            self._node(instance_id, "relu2", "nn.ReLU", {"inplace": True}),
            self._node(instance_id, "conv3x3", "nn.Conv2d", {"in_channels": inter_channels, "out_channels": growth_rate, "kernel_size": 3, "padding": 1, "bias": False}),
            self._node(instance_id, "concat", "torch.cat", {"dim": 1}),
        ]
        edges = self._edges(
            instance_id,
            [
                ("identity", "bn1"),
                ("bn1", "relu1"),
                ("relu1", "conv1x1"),
                ("conv1x1", "bn2"),
                ("bn2", "relu2"),
                ("relu2", "conv3x3"),
                ("conv3x3", "concat"),
                ("identity", "concat"),
            ],
        )
        return nodes, edges, self._id(instance_id, "identity"), self._id(instance_id, "concat"), cfg

    def _build_densenet_transition(
        self, instance_id: str, config: Dict[str, Any]
    ) -> Tuple[List[ArchitectureNode], List[ArchitectureEdge], str, str, Dict[str, Any]]:
        in_channels = int(config.get("in_channels", 128))
        compression = float(config.get("compression", 0.5))
        self._enforce(
            "densenet_transition",
            [
                (in_channels > 0, "in_channels must be > 0"),
                (0.0 < compression <= 1.0, "compression must be in (0.0, 1.0]"),
            ],
        )
        out_channels = max(1, int(in_channels * compression))
        cfg = {"in_channels": in_channels, "compression": compression, "out_channels": out_channels}

        nodes = [
            self._node(instance_id, "bn", "nn.BatchNorm2d", {"num_features": in_channels}),
            self._node(instance_id, "relu", "nn.ReLU", {"inplace": True}),
            self._node(instance_id, "conv1x1", "nn.Conv2d", {"in_channels": in_channels, "out_channels": out_channels, "kernel_size": 1, "bias": False}),
            self._node(instance_id, "avgpool", "nn.AvgPool2d", {"kernel_size": 2, "stride": 2}),
        ]
        edges = self._edges(
            instance_id,
            [
                ("bn", "relu"),
                ("relu", "conv1x1"),
                ("conv1x1", "avgpool"),
            ],
        )
        return nodes, edges, self._id(instance_id, "bn"), self._id(instance_id, "avgpool"), cfg

    def _build_mobilenet_depthwise_block(
        self, instance_id: str, config: Dict[str, Any]
    ) -> Tuple[List[ArchitectureNode], List[ArchitectureEdge], str, str, Dict[str, Any]]:
        in_channels = int(config.get("in_channels", 32))
        out_channels = int(config.get("out_channels", 64))
        stride = int(config.get("stride", 1))
        use_residual = bool(config.get("use_residual", stride == 1 and in_channels == out_channels))
        self._enforce(
            "mobilenet_depthwise_block",
            [
                (in_channels > 0, "in_channels must be > 0"),
                (out_channels > 0, "out_channels must be > 0"),
                (stride in {1, 2}, "stride must be 1 or 2"),
            ],
        )
        cfg = {"in_channels": in_channels, "out_channels": out_channels, "stride": stride, "use_residual": use_residual}

        nodes = [
            self._node(instance_id, "identity", "nn.Identity", {}),
            self._node(instance_id, "dw_conv", "nn.Conv2d", {"in_channels": in_channels, "out_channels": in_channels, "kernel_size": 3, "stride": stride, "padding": 1, "groups": in_channels, "bias": False}),
            self._node(instance_id, "bn1", "nn.BatchNorm2d", {"num_features": in_channels}),
            self._node(instance_id, "relu1", "nn.ReLU6", {"inplace": True}),
            self._node(instance_id, "pw_conv", "nn.Conv2d", {"in_channels": in_channels, "out_channels": out_channels, "kernel_size": 1, "bias": False}),
            self._node(instance_id, "bn2", "nn.BatchNorm2d", {"num_features": out_channels}),
            self._node(instance_id, "relu2", "nn.ReLU6", {"inplace": True}),
        ]
        link_pairs = [
            ("identity", "dw_conv"),
            ("dw_conv", "bn1"),
            ("bn1", "relu1"),
            ("relu1", "pw_conv"),
            ("pw_conv", "bn2"),
            ("bn2", "relu2"),
        ]

        if use_residual:
            nodes.append(self._node(instance_id, "add", "torch.add", {}))
            link_pairs.append(("relu2", "add"))
            link_pairs.append(("identity", "add"))
            edges = self._edges(instance_id, link_pairs)
            return nodes, edges, self._id(instance_id, "identity"), self._id(instance_id, "add"), cfg

        edges = self._edges(instance_id, link_pairs)
        return nodes, edges, self._id(instance_id, "identity"), self._id(instance_id, "relu2"), cfg

    def _build_resnet_basic_block(
        self, instance_id: str, config: Dict[str, Any]
    ) -> Tuple[List[ArchitectureNode], List[ArchitectureEdge], str, str, Dict[str, Any]]:
        in_channels = int(config.get("in_channels", 64))
        out_channels = int(config.get("out_channels", 64))
        stride = int(config.get("stride", 1))
        projection = bool(config.get("projection", in_channels != out_channels or stride != 1))
        self._enforce(
            "resnet_basic_block",
            [
                (in_channels > 0, "in_channels must be > 0"),
                (out_channels > 0, "out_channels must be > 0"),
                (stride in {1, 2}, "stride must be 1 or 2"),
            ],
        )
        cfg = {"in_channels": in_channels, "out_channels": out_channels, "stride": stride, "projection": projection}

        nodes = [
            self._node(instance_id, "identity", "nn.Identity", {}),
            self._node(instance_id, "conv1", "nn.Conv2d", {"in_channels": in_channels, "out_channels": out_channels, "kernel_size": 3, "stride": stride, "padding": 1, "bias": False}),
            self._node(instance_id, "bn1", "nn.BatchNorm2d", {"num_features": out_channels}),
            self._node(instance_id, "relu1", "nn.ReLU", {"inplace": True}),
            self._node(instance_id, "conv2", "nn.Conv2d", {"in_channels": out_channels, "out_channels": out_channels, "kernel_size": 3, "stride": 1, "padding": 1, "bias": False}),
            self._node(instance_id, "bn2", "nn.BatchNorm2d", {"num_features": out_channels}),
            self._node(instance_id, "add", "torch.add", {}),
            self._node(instance_id, "relu2", "nn.ReLU", {"inplace": True}),
        ]
        if projection:
            nodes.extend([
                self._node(instance_id, "proj_conv", "nn.Conv2d", {"in_channels": in_channels, "out_channels": out_channels, "kernel_size": 1, "stride": stride, "bias": False}),
                self._node(instance_id, "proj_bn", "nn.BatchNorm2d", {"num_features": out_channels}),
            ])

        link_pairs = [
            ("identity", "conv1"),
            ("conv1", "bn1"),
            ("bn1", "relu1"),
            ("relu1", "conv2"),
            ("conv2", "bn2"),
            ("bn2", "add"),
            ("add", "relu2"),
        ]
        edges = self._edges(instance_id, link_pairs)
        if projection:
            edges.extend(self._edges(instance_id, [("identity", "proj_conv"), ("proj_conv", "proj_bn"), ("proj_bn", "add")]))
        else:
            edges.extend(self._edges(instance_id, [("identity", "add")]))

        return nodes, edges, self._id(instance_id, "identity"), self._id(instance_id, "relu2"), cfg

    def _build_gcn_layer(
        self, instance_id: str, config: Dict[str, Any]
    ) -> Tuple[List[ArchitectureNode], List[ArchitectureEdge], str, str, Dict[str, Any]]:
        in_features = int(config.get("in_features", 64))
        out_features = int(config.get("out_features", 64))
        use_residual = bool(config.get("use_residual", in_features == out_features))
        self._enforce("gcn_layer", [
            (in_features > 0, "in_features must be > 0"),
            (out_features > 0, "out_features must be > 0"),
        ])
        cfg = {"in_features": in_features, "out_features": out_features, "use_residual": use_residual}

        nodes = [
            self._node(instance_id, "identity", "nn.Identity", {}),
            self._node(instance_id, "linear", "nn.Linear", {"in_features": in_features, "out_features": out_features, "bias": False}),
            self._node(instance_id, "relu", "nn.ReLU", {"inplace": True}),
        ]
        link_pairs = [("identity", "linear"), ("linear", "relu")]

        if use_residual:
            nodes.append(self._node(instance_id, "add", "torch.add", {}))
            link_pairs.extend([("relu", "add"), ("identity", "add")])
            edges = self._edges(instance_id, link_pairs)
            return nodes, edges, self._id(instance_id, "identity"), self._id(instance_id, "add"), cfg

        edges = self._edges(instance_id, link_pairs)
        return nodes, edges, self._id(instance_id, "identity"), self._id(instance_id, "relu"), cfg

    def _build_gat_layer(
        self, instance_id: str, config: Dict[str, Any]
    ) -> Tuple[List[ArchitectureNode], List[ArchitectureEdge], str, str, Dict[str, Any]]:
        in_features = int(config.get("in_features", 64))
        out_features = int(config.get("out_features", 64))
        num_heads = int(config.get("num_heads", 4))
        self._enforce("gat_layer", [
            (in_features > 0, "in_features must be > 0"),
            (out_features > 0, "out_features must be > 0"),
            (num_heads >= 1, "num_heads must be >= 1"),
        ])
        head_dim = max(1, out_features // num_heads)
        cfg = {"in_features": in_features, "out_features": out_features, "num_heads": num_heads, "head_dim": head_dim}

        nodes = [
            self._node(instance_id, "identity", "nn.Identity", {}),
            self._node(instance_id, "linear", "nn.Linear", {"in_features": in_features, "out_features": num_heads * head_dim, "bias": False}),
            self._node(instance_id, "leaky_relu", "nn.LeakyReLU", {"negative_slope": 0.2}),
            self._node(instance_id, "attn_softmax", "nn.Softmax", {"dim": -1}),
            self._node(instance_id, "dropout", "nn.Dropout", {"p": 0.6}),
            self._node(instance_id, "elu", "nn.ELU", {"alpha": 1.0}),
        ]
        edges = self._edges(instance_id, [
            ("identity", "linear"), ("linear", "leaky_relu"),
            ("leaky_relu", "attn_softmax"), ("attn_softmax", "dropout"),
            ("dropout", "elu"),
        ])
        return nodes, edges, self._id(instance_id, "identity"), self._id(instance_id, "elu"), cfg

    def _build_message_passing_block(
        self, instance_id: str, config: Dict[str, Any]
    ) -> Tuple[List[ArchitectureNode], List[ArchitectureEdge], str, str, Dict[str, Any]]:
        in_features = int(config.get("in_features", 64))
        out_features = int(config.get("out_features", 64))
        self._enforce("message_passing_block", [
            (in_features > 0, "in_features must be > 0"),
            (out_features > 0, "out_features must be > 0"),
        ])
        cfg = {"in_features": in_features, "out_features": out_features}

        nodes = [
            self._node(instance_id, "identity", "nn.Identity", {}),
            self._node(instance_id, "message_fn", "nn.Linear", {"in_features": in_features, "out_features": out_features}),
            # Aggregate over NEIGHBOURS (dim 0), not over features: the declared input is
            # rank-2 [num_nodes, in_features], so dim 1 is the feature axis and summing it
            # collapses every node to a scalar. keepdim holds the rank-2 contract.
            self._node(instance_id, "aggregate", "torch.sum", {"dim": 0, "keepdim": True}),
            self._node(instance_id, "update_fn", "nn.Linear", {"in_features": out_features, "out_features": out_features}),
            self._node(instance_id, "relu", "nn.ReLU", {"inplace": True}),
        ]
        edges = self._edges(instance_id, [
            ("identity", "message_fn"), ("message_fn", "aggregate"),
            ("aggregate", "update_fn"), ("update_fn", "relu"),
        ])
        return nodes, edges, self._id(instance_id, "identity"), self._id(instance_id, "relu"), cfg

    def _build_mamba_block(
        self, instance_id: str, config: Dict[str, Any]
    ) -> Tuple[List[ArchitectureNode], List[ArchitectureEdge], str, str, Dict[str, Any]]:
        d_model = int(config.get("d_model", 128))
        d_state = int(config.get("d_state", 16))
        expand_factor = int(config.get("expand_factor", 2))
        dt_rank = int(config.get("dt_rank", max(1, d_model // 16)))
        self._enforce("mamba_block", [
            (d_model > 0, "d_model must be > 0"),
            (d_state > 0, "d_state must be > 0"),
            (expand_factor >= 1, "expand_factor must be >= 1"),
        ])
        d_inner = d_model * expand_factor
        cfg = {"d_model": d_model, "d_state": d_state, "expand_factor": expand_factor, "d_inner": d_inner, "dt_rank": dt_rank}

        nodes = [
            self._node(instance_id, "identity", "nn.Identity", {}),
            self._node(instance_id, "norm", "nn.LayerNorm", {"normalized_shape": d_model}),
            self._node(instance_id, "proj_in", "nn.Linear", {"in_features": d_model, "out_features": d_inner, "bias": False}),
            self._node(instance_id, "conv1d", "mamba_depthwise_conv1d", {"in_channels": d_inner, "out_channels": d_inner, "kernel_size": 4, "padding": 3, "groups": d_inner}),
            self._node(instance_id, "silu", "nn.SiLU", {}),
            self._node(instance_id, "ssm", "SelectiveSSM", {"d_inner": d_inner, "d_state": d_state, "dt_rank": dt_rank}),
            self._node(instance_id, "proj_out", "nn.Linear", {"in_features": d_inner, "out_features": d_model, "bias": False}),
            self._node(instance_id, "add", "torch.add", {}),
        ]
        edges = self._edges(instance_id, [
            ("identity", "norm"), ("norm", "proj_in"), ("proj_in", "conv1d"),
            ("conv1d", "silu"), ("silu", "ssm"), ("ssm", "proj_out"),
            ("proj_out", "add"), ("identity", "add"),
        ])
        return nodes, edges, self._id(instance_id, "identity"), self._id(instance_id, "add"), cfg

    def _build_tcn_block(
        self, instance_id: str, config: Dict[str, Any]
    ) -> Tuple[List[ArchitectureNode], List[ArchitectureEdge], str, str, Dict[str, Any]]:
        in_channels = int(config.get("in_channels", 64))
        out_channels = int(config.get("out_channels", 64))
        kernel_size = int(config.get("kernel_size", 3))
        dilation = int(config.get("dilation", 1))
        dropout = float(config.get("dropout", 0.2))
        projection = bool(config.get("projection", in_channels != out_channels))
        self._enforce("tcn_block", [
            (in_channels > 0, "in_channels must be > 0"),
            (out_channels > 0, "out_channels must be > 0"),
            (kernel_size >= 2, "kernel_size must be >= 2"),
        ])
        # Use "same-ish" padding so residual add can preserve temporal length.
        padding = ((kernel_size - 1) * dilation) // 2
        cfg = {"in_channels": in_channels, "out_channels": out_channels, "kernel_size": kernel_size, "dilation": dilation, "dropout": dropout, "projection": projection}

        nodes = [
            self._node(instance_id, "identity", "nn.Identity", {}),
            self._node(instance_id, "conv1", "nn.Conv1d", {"in_channels": in_channels, "out_channels": out_channels, "kernel_size": kernel_size, "dilation": dilation, "padding": padding}),
            self._node(instance_id, "relu1", "nn.ReLU", {"inplace": True}),
            self._node(instance_id, "dropout1", "nn.Dropout", {"p": dropout}),
            self._node(instance_id, "conv2", "nn.Conv1d", {"in_channels": out_channels, "out_channels": out_channels, "kernel_size": kernel_size, "dilation": dilation, "padding": padding}),
            self._node(instance_id, "relu2", "nn.ReLU", {"inplace": True}),
            self._node(instance_id, "dropout2", "nn.Dropout", {"p": dropout}),
            self._node(instance_id, "add", "torch.add", {}),
            self._node(instance_id, "relu_out", "nn.ReLU", {"inplace": True}),
        ]
        link_pairs = [
            ("identity", "conv1"), ("conv1", "relu1"), ("relu1", "dropout1"),
            ("dropout1", "conv2"), ("conv2", "relu2"), ("relu2", "dropout2"),
            ("dropout2", "add"), ("add", "relu_out"),
        ]
        if projection:
            nodes.append(self._node(instance_id, "proj", "nn.Conv1d", {"in_channels": in_channels, "out_channels": out_channels, "kernel_size": 1}))
            link_pairs.append(("identity", "proj"))
            link_pairs.append(("proj", "add"))
        else:
            link_pairs.append(("identity", "add"))

        edges = self._edges(instance_id, link_pairs)
        return nodes, edges, self._id(instance_id, "identity"), self._id(instance_id, "relu_out"), cfg

    def _build_vae_encoder_decoder(
        self, instance_id: str, config: Dict[str, Any]
    ) -> Tuple[List[ArchitectureNode], List[ArchitectureEdge], str, str, Dict[str, Any]]:
        input_dim = int(config.get("input_dim", 784))
        hidden_dim = int(config.get("hidden_dim", 400))
        latent_dim = int(config.get("latent_dim", 20))
        self._enforce("vae_encoder_decoder", [
            (input_dim > 0, "input_dim must be > 0"),
            (latent_dim > 0, "latent_dim must be > 0"),
        ])
        cfg = {"input_dim": input_dim, "hidden_dim": hidden_dim, "latent_dim": latent_dim}

        nodes = [
            # Encoder
            self._node(instance_id, "enc_linear1", "nn.Linear", {"in_features": input_dim, "out_features": hidden_dim}),
            self._node(instance_id, "enc_relu", "nn.ReLU", {"inplace": True}),
            self._node(instance_id, "mu_head", "nn.Linear", {"in_features": hidden_dim, "out_features": latent_dim}),
            self._node(instance_id, "logvar_head", "nn.Linear", {"in_features": hidden_dim, "out_features": latent_dim}),
            # Reparameterize (represented as op node)
            self._node(instance_id, "reparameterize", "Reparameterize", {"latent_dim": latent_dim}),
            # Decoder
            self._node(instance_id, "dec_linear1", "nn.Linear", {"in_features": latent_dim, "out_features": hidden_dim}),
            self._node(instance_id, "dec_relu", "nn.ReLU", {"inplace": True}),
            self._node(instance_id, "dec_linear2", "nn.Linear", {"in_features": hidden_dim, "out_features": input_dim}),
            self._node(instance_id, "sigmoid", "nn.Sigmoid", {}),
        ]
        edges = self._edges(instance_id, [
            ("enc_linear1", "enc_relu"),
            ("enc_relu", "mu_head"),
            ("enc_relu", "logvar_head"),
            ("mu_head", "reparameterize"),
            ("logvar_head", "reparameterize"),
            ("reparameterize", "dec_linear1"),
            ("dec_linear1", "dec_relu"),
            ("dec_relu", "dec_linear2"),
            ("dec_linear2", "sigmoid"),
        ])
        return nodes, edges, self._id(instance_id, "enc_linear1"), self._id(instance_id, "sigmoid"), cfg

    def _build_gan_pair(
        self, instance_id: str, config: Dict[str, Any]
    ) -> Tuple[List[ArchitectureNode], List[ArchitectureEdge], str, str, Dict[str, Any]]:
        latent_dim = int(config.get("latent_dim", 100))
        hidden_dim = int(config.get("hidden_dim", 256))
        output_dim = int(config.get("output_dim", 784))
        self._enforce("gan_pair", [
            (latent_dim > 0, "latent_dim must be > 0"),
            (output_dim > 0, "output_dim must be > 0"),
        ])
        cfg = {"latent_dim": latent_dim, "hidden_dim": hidden_dim, "output_dim": output_dim}

        nodes = [
            # Generator
            self._node(instance_id, "gen_linear1", "nn.Linear", {"in_features": latent_dim, "out_features": hidden_dim}),
            self._node(instance_id, "gen_relu1", "nn.LeakyReLU", {"negative_slope": 0.2}),
            self._node(instance_id, "gen_linear2", "nn.Linear", {"in_features": hidden_dim, "out_features": hidden_dim * 2}),
            self._node(instance_id, "gen_relu2", "nn.LeakyReLU", {"negative_slope": 0.2}),
            self._node(instance_id, "gen_linear3", "nn.Linear", {"in_features": hidden_dim * 2, "out_features": output_dim}),
            self._node(instance_id, "gen_tanh", "nn.Tanh", {}),
            # Discriminator
            self._node(instance_id, "disc_linear1", "nn.Linear", {"in_features": output_dim, "out_features": hidden_dim * 2}),
            self._node(instance_id, "disc_relu1", "nn.LeakyReLU", {"negative_slope": 0.2}),
            self._node(instance_id, "disc_linear2", "nn.Linear", {"in_features": hidden_dim * 2, "out_features": hidden_dim}),
            self._node(instance_id, "disc_relu2", "nn.LeakyReLU", {"negative_slope": 0.2}),
            self._node(instance_id, "disc_linear3", "nn.Linear", {"in_features": hidden_dim, "out_features": 1}),
            self._node(instance_id, "disc_sigmoid", "nn.Sigmoid", {}),
        ]
        edges = self._edges(instance_id, [
            # Generator chain
            ("gen_linear1", "gen_relu1"), ("gen_relu1", "gen_linear2"),
            ("gen_linear2", "gen_relu2"), ("gen_relu2", "gen_linear3"),
            ("gen_linear3", "gen_tanh"),
            # Generator output feeds discriminator
            ("gen_tanh", "disc_linear1"),
            # Discriminator chain
            ("disc_linear1", "disc_relu1"), ("disc_relu1", "disc_linear2"),
            ("disc_linear2", "disc_relu2"), ("disc_relu2", "disc_linear3"),
            ("disc_linear3", "disc_sigmoid"),
        ])
        return nodes, edges, self._id(instance_id, "gen_linear1"), self._id(instance_id, "disc_sigmoid"), cfg

    @staticmethod
    def _enforce(template_id: str, checks: List[Tuple[bool, str]]) -> None:
        failures = [message for ok, message in checks if not ok]
        if failures:
            joined = "; ".join(failures)
            raise ValueError(f"Template '{template_id}' failed post-checks: {joined}")

    @staticmethod
    def _node(instance_id: str, local_id: str, op: str, params: Dict[str, Any]) -> ArchitectureNode:
        return ArchitectureNode(
            node_id=f"{instance_id}__{local_id}",
            kind="layer" if not op.startswith("torch.") else "op",
            op=op,
            params=params,
            in_ports=["x"],
            out_ports=["y"],
        )

    @staticmethod
    def _id(instance_id: str, local_id: str) -> str:
        return f"{instance_id}__{local_id}"

    @staticmethod
    def _edges(instance_id: str, links: List[Tuple[str, str]]) -> List[ArchitectureEdge]:
        edges: List[ArchitectureEdge] = []
        for source_local, target_local in links:
            source = f"{instance_id}__{source_local}"
            target = f"{instance_id}__{target_local}"
            edges.append(
                ArchitectureEdge(
                    edge_id=f"{source}:y->{target}:x",
                    source_node_id=source,
                    source_port="y",
                    target_node_id=target,
                    target_port="x",
                )
            )
        return edges
