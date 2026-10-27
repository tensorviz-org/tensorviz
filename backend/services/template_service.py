"""
Central template service for managing all model templates, popular models, and customization.
Supports hierarchical template instantiation and user customization workflow.
"""
from typing import Dict, List, Any, Optional, Union
from models.template_models import ModelTemplate, ModuleTemplate, PopularModel, TemplateVariant
from models.graph_models import Graph, HierarchicalGraph, Node, Edge, ModuleDefinition
from .transformer_templates import TransformerTemplates
from .computer_vision_templates import ComputerVisionTemplates
from .nlp_templates import NLPTemplates
from .generative_templates import GenerativeTemplates
from .starter_templates import StarterTemplates

# Horizontal step between adjacent cards. The tldraw shape utils render layer /
# input / output cards 200px wide, so anything below 200 overlaps.
_CARD_PITCH_X = 220


class TemplateService:
    """Central service for managing all templates and enabling user customization"""
    
    def __init__(self):
        self.transformer_templates = TransformerTemplates()
        self.cv_templates = ComputerVisionTemplates()
        self.nlp_templates = NLPTemplates()
        self.generative_templates = GenerativeTemplates()
        self.starter_templates = StarterTemplates()
    
    # Template Discovery Methods
    def get_all_categories(self) -> List[Dict[str, Any]]:
        """Get all available template categories"""
        return [
            {
                "id": "starters",
                "name": "Starters",
                "description": "Small foundational networks built from basic blocks",
                "icon": "zap",
                "count": len(self.starter_templates.get_all_templates())
            },
            {
                "id": "transformer",
                "name": "Transformer Models",
                "description": "Attention-based models for NLP and beyond",
                "icon": "brain",
                "count": len(self.transformer_templates.get_all_templates())
            },
            {
                "id": "computer_vision",
                "name": "Computer Vision",
                "description": "CNN and vision transformer architectures",
                "icon": "eye",
                "count": len(self.cv_templates.get_all_templates())
            },
            {
                "id": "nlp",
                "name": "NLP Specialized",
                "description": "Specialized NLP architectures beyond transformers",
                "icon": "message-square",
                "count": len(self.nlp_templates.get_all_templates())
            },
            {
                "id": "generative",
                "name": "Generative Models",
                "description": "GANs, VAEs, and other generative architectures",
                "icon": "sparkles",
                "count": len(self.generative_templates.get_all_templates())
            }
        ]
    
    def get_popular_models_by_category(self, category: str) -> Dict[str, PopularModel]:
        """Get popular models for a specific category"""
        if category == "starters":
            return self.starter_templates.get_popular_models()
        if category == "transformer":
            return self.transformer_templates.get_popular_models()
        elif category == "computer_vision":
            return self.cv_templates.get_popular_models()
        elif category == "nlp":
            return self.nlp_templates.get_popular_models()
        elif category == "generative":
            return self.generative_templates.get_popular_models()
        else:
            return {}
    
    def get_all_popular_models(self) -> Dict[str, PopularModel]:
        """Get all popular models across categories"""
        models = {}
        models.update(self.starter_templates.get_popular_models())
        models.update(self.transformer_templates.get_popular_models())
        models.update(self.cv_templates.get_popular_models())
        models.update(self.nlp_templates.get_popular_models())
        models.update(self.generative_templates.get_popular_models())
        return models
    
    def get_templates_by_category(self, category: str) -> Dict[str, ModelTemplate]:
        """Get templates for a specific category"""
        if category == "starters":
            return self.starter_templates.get_all_templates()
        if category == "transformer":
            return self.transformer_templates.get_all_templates()
        elif category == "computer_vision":
            return self.cv_templates.get_all_templates()
        elif category == "nlp":
            return self.nlp_templates.get_all_templates()
        elif category == "generative":
            return self.generative_templates.get_all_templates()
        else:
            return {}
    
    def get_module_templates_by_category(self, category: str) -> Dict[str, ModuleTemplate]:
        """Get module templates for a specific category"""
        if category == "starters":
            return self.starter_templates.get_module_templates()
        if category == "transformer":
            return self.transformer_templates.get_module_templates()
        elif category == "computer_vision":
            return self.cv_templates.get_module_templates()
        elif category == "nlp":
            return self.nlp_templates.get_module_templates()
        elif category == "generative":
            return self.generative_templates.get_module_templates()
        else:
            return {}
    
    # Template Instantiation Methods
    def instantiate_popular_model(self, model_id: str, customizations: Optional[Dict[str, Any]] = None) -> HierarchicalGraph:
        """Instantiate a popular model with optional customizations"""
        # Get the popular model configuration
        popular_model = self.get_popular_model_by_id(model_id)
        if not popular_model:
            raise ValueError(f"Popular model '{model_id}' not found")
        
        # Get the base template
        template = self.get_template_by_id(popular_model.template_id)
        if not template:
            raise ValueError(f"Template '{popular_model.template_id}' not found")
        
        # Merge configurations: base config + popular model config + user customizations
        final_config = template.base_config.copy()
        final_config.update(popular_model.config)
        if customizations:
            final_config.update(customizations)
        
        # Instantiate the template with the final configuration
        return self.instantiate_template(popular_model.template_id, final_config)
    
    def instantiate_template(self, template_id: str, config: Dict[str, Any]) -> HierarchicalGraph:
        """Instantiate a template with specific configuration"""
        template = self.get_template_by_id(template_id)
        if not template:
            raise ValueError(f"Template '{template_id}' not found")

        # Merge the template's declared defaults under the caller's config so a
        # bare instantiation (e.g. the gallery "Templates" tab, which posts an
        # empty config) still substitutes every `{placeholder}` — otherwise the
        # structure ships literal "{seq_len}"/"{dim}" strings and shape
        # validation breaks. Caller-supplied keys win over the defaults.
        config = {**(template.base_config or {}), **(config or {})}

        # Process template structure and substitute configuration values
        processed_structure = self._process_template_structure(template.template_structure, config)

        # Convert to hierarchical graph format
        graph = self._convert_to_hierarchical_graph(processed_structure, template_id, config)

        # Templates declare placeholder moduleNodes (encoder_stack, decoder_stack,
        # etc.) without an entry in `modules`. Expand them here so the
        # instantiated graph actually carries the per-layer internals.
        self._expand_module_placeholders(graph, config)
        # Many CV/NLP/generative templates omit explicit Input/Output nodes,
        # which makes shape validation report "missing input node" /
        # "unreachable" for the whole graph. Synthesize them from the template's
        # declared input shape so every instantiated model is anchored end-to-end.
        self._ensure_io_nodes(graph, template, config)
        return graph
    
    def customize_model(self, base_graph: HierarchicalGraph, customizations: List[Dict[str, Any]]) -> HierarchicalGraph:
        """Apply customizations to an existing model graph"""
        modified_graph = base_graph.copy()
        
        for customization in customizations:
            action = customization.get("action")
            
            if action == "remove_module":
                modified_graph = self._remove_module(modified_graph, customization)
            elif action == "add_module":
                modified_graph = self._add_module(modified_graph, customization)
            elif action == "modify_module":
                modified_graph = self._modify_module(modified_graph, customization)
            elif action == "change_parameter":
                modified_graph = self._change_parameter(modified_graph, customization)
            elif action == "replace_module":
                modified_graph = self._replace_module(modified_graph, customization)
        
        return modified_graph
    
    # User Journey Support Methods
    def get_customization_options(self, template_id: str) -> Dict[str, Any]:
        """Get available customization options for a template"""
        template = self.get_template_by_id(template_id)
        if not template:
            return {}
        
        return {
            "parameters": template.customizable_params,
            "removable_modules": self._get_removable_modules(template),
            "addable_modules": self._get_addable_modules(template),
            "replaceable_modules": self._get_replaceable_modules(template),
            "variants": template.variants
        }
    
    def validate_customization(self, template_id: str, customizations: Dict[str, Any]) -> List[str]:
        """Validate user customizations and return any errors"""
        template = self.get_template_by_id(template_id)
        if not template:
            return ["Template not found"]
        
        errors = []
        
        # Validate parameter ranges
        for param_name, param_value in customizations.items():
            param_def = next((p for p in template.customizable_params if p["name"] == param_name), None)
            if param_def:
                if param_def["type"] == "number":
                    if "min" in param_def and param_value < param_def["min"]:
                        errors.append(f"{param_name} must be >= {param_def['min']}")
                    if "max" in param_def and param_value > param_def["max"]:
                        errors.append(f"{param_name} must be <= {param_def['max']}")
        
        # Validate architectural constraints
        if "d_model" in customizations and "num_heads" in customizations:
            if customizations["d_model"] % customizations["num_heads"] != 0:
                errors.append("d_model must be divisible by num_heads")
        
        return errors
    
    # Helper Methods
    def get_template_by_id(self, template_id: str) -> Optional[ModelTemplate]:
        """Get a template by its ID across all categories"""
        all_templates = {}
        all_templates.update(self.starter_templates.get_all_templates())
        all_templates.update(self.transformer_templates.get_all_templates())
        all_templates.update(self.cv_templates.get_all_templates())
        all_templates.update(self.nlp_templates.get_all_templates())
        all_templates.update(self.generative_templates.get_all_templates())
        return all_templates.get(template_id)
    
    def get_popular_model_by_id(self, model_id: str) -> Optional[PopularModel]:
        """Get a popular model by its ID across all categories"""
        all_models = self.get_all_popular_models()
        return all_models.get(model_id)
    
    def get_module_template_by_id(self, module_id: str) -> Optional[ModuleTemplate]:
        """Get a module template by its ID across all categories"""
        all_modules = {}
        all_modules.update(self.starter_templates.get_module_templates())
        all_modules.update(self.transformer_templates.get_module_templates())
        all_modules.update(self.cv_templates.get_module_templates())
        all_modules.update(self.nlp_templates.get_module_templates()) 
        all_modules.update(self.generative_templates.get_module_templates())
        return all_modules.get(module_id)
    
    def _process_template_structure(self, template_structure: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Process template structure by substituting configuration values"""
        import json
        import re
        
        # Convert to string for easier substitution
        structure_str = json.dumps(template_structure)
        
        # Substitute configuration values (format: {config_key})
        for key, value in config.items():
            pattern = f"{{{key}}}"
            structure_str = structure_str.replace(pattern, str(value))
        
        # Convert back to dictionary
        return json.loads(structure_str)
    
    def _convert_to_hierarchical_graph(self, processed_structure: Dict[str, Any], template_id: str, config: Dict[str, Any]) -> HierarchicalGraph:
        """Convert processed template structure to HierarchicalGraph"""
        root_nodes = []
        root_edges = []
        modules = {}
        
        # Process root nodes - handle both 'nodes' and 'root_nodes' formats
        nodes_data = processed_structure.get("root_nodes", processed_structure.get("nodes", []))
        for node_data in nodes_data:
            node = self._create_node_from_data(node_data)
            root_nodes.append(node)
        
        # Process root edges - handle both 'edges' and 'root_edges' formats
        edges_data = processed_structure.get("root_edges", processed_structure.get("edges", []))
        for edge_data in edges_data:
            edge = Edge(
                id=edge_data["id"],
                source=edge_data["source"],
                target=edge_data["target"]
            )
            root_edges.append(edge)
        
        # Process modules
        for module_id, module_data in processed_structure.get("modules", {}).items():
            module_def = self._create_module_definition_from_data(module_data)
            modules[module_id] = module_def
        
        return HierarchicalGraph(
            root_nodes=root_nodes,
            root_edges=root_edges,
            modules=modules
        )
    
    @staticmethod
    def _coerce_numeric_params(parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Config substitution (`_process_template_structure`) injects values as
        strings inside the JSON template (e.g. out_channels "{d_model}" -> "768"),
        which breaks torch layers that need ints (Conv2d/LayerNorm/Linear). Coerce
        numeric-looking strings back to int/float, and boolean-looking strings
        ("True"/"False", from `str(bool)`) back to bool (else nn.LSTM/Conv get a
        str where they expect a bool, e.g. `bidirectional="True"`). Display-only
        keys (prefixed with `_`) and other strings ("1,3,224,224") are untouched."""
        def coerce(value: Any) -> Any:
            if not isinstance(value, str):
                return value
            s = value.strip()
            if s in ("True", "true"):
                return True
            if s in ("False", "false"):
                return False
            try:
                return int(s)
            except ValueError:
                pass
            try:
                return float(s)
            except ValueError:
                return value
        return {k: (v if k.startswith("_") else coerce(v)) for k, v in parameters.items()}

    def _create_node_from_data(self, node_data: Dict[str, Any]) -> Node:
        """Create a Node object from template data"""
        from models.graph_models import LayerType, LayerData

        layer_type_data = node_data["data"]["layerType"]
        layer_type = LayerType(
            id=layer_type_data["id"],
            name=layer_type_data["name"],
            parameters=layer_type_data["parameters"]
        )

        layer_data = LayerData(
            layerType=layer_type,
            parameters=self._coerce_numeric_params(node_data["data"]["parameters"]),
            moduleType=node_data["data"].get("moduleType"),
            expanded=node_data["data"].get("expanded")
        )
        
        return Node(
            id=node_data["id"],
            type=node_data["type"],
            position=node_data["position"],
            data=layer_data
        )
    
    # ── Placeholder expansion ────────────────────────────────────────────
    def _ensure_io_nodes(self, graph: HierarchicalGraph, template: ModelTemplate, config: Optional[Dict[str, Any]] = None) -> None:
        """Guarantee the graph has an Input node feeding its source layers and an
        Output node draining its sink layers. Templates that already declare
        them (transformers, ViT, nanoDiT) are left untouched. The input shape
        comes from the template's `minimum_input_shape` (batch forced to 1) so
        downstream shape inference has a concrete anchor to propagate from."""
        from models.graph_models import LayerType, LayerData

        roots = graph.root_nodes
        has_input = any(n.type == "inputNode" for n in roots)
        has_output = any(n.type == "outputNode" for n in roots)
        if has_input and has_output:
            return

        shape = list(template.minimum_input_shape or [1, 3, 224, 224])
        if shape:
            shape[0] = 1  # batch of 1 for inference
        # For latent-vector inputs ([N, D], e.g. GAN/StyleGAN generators) the
        # channel dim must track the (possibly popular-model-overridden)
        # latent_dim so the synthesized generator's reshape/conv channel math
        # matches the actual input width.
        if config and len(shape) == 2 and "latent_dim" in config:
            shape[1] = self._to_int(config.get("latent_dim"), shape[1])
        shape_str = ",".join(str(int(d)) for d in shape)

        original_ids = [n.id for n in roots]
        # Anchor the synthesized boundary nodes on the chain they attach to:
        # Input one card-pitch LEFT of the leftmost source layer, Output one pitch
        # RIGHT of the rightmost sink. Cards render 200px wide, so the 220px pitch
        # matches the hand-authored templates (transformers, nanoDiT) and keeps the
        # boundary nodes from landing on top of a layer — or on each other.
        positions = {n.id: n.position for n in roots}

        def _anchor(node_ids: List[str], side: int) -> Dict[str, float]:
            known = [positions[nid] for nid in node_ids if nid in positions]
            if not known:
                return {"x": 0.0, "y": 0.0}
            edge_pos = (min if side < 0 else max)(known, key=lambda p: p.get("x", 0.0))
            return {"x": edge_pos.get("x", 0.0) + side * _CARD_PITCH_X,
                    "y": edge_pos.get("y", 0.0)}

        if not has_input:
            targets = {e.target for e in graph.root_edges}
            sources = [nid for nid in original_ids if nid not in targets]
            graph.root_nodes.insert(0, Node(
                id="input_node", type="inputNode", position=_anchor(sources, -1),
                data=LayerData(
                    layerType=LayerType(id="input", name="Input", parameters=[]),
                    parameters={"name": "Input", "shape": shape_str,
                                "_subtitle": "Input", "_shape": f"[{', '.join(str(d) for d in shape)}]",
                                "_accent": "var(--cat-embedding)"},
                ),
            ))
            for sid in sources:
                graph.root_edges.append(Edge(id=f"input_node->{sid}", source="input_node", target=sid))
        if not has_output:
            outgoing = {e.source for e in graph.root_edges}
            sinks = [nid for nid in original_ids if nid not in outgoing]
            graph.root_nodes.append(Node(
                id="output_node", type="outputNode", position=_anchor(sinks, 1),
                data=LayerData(
                    layerType=LayerType(id="output", name="Output", parameters=[]),
                    parameters={"name": "Output", "_subtitle": "Output", "_accent": "var(--cat-embedding)"},
                ),
            ))
            for sid in sinks:
                graph.root_edges.append(Edge(id=f"{sid}->output_node", source=sid, target="output_node"))

    def _expand_module_placeholders(self, graph: HierarchicalGraph, config: Dict[str, Any]) -> None:
        """For any moduleNode at the root level that lacks an entry in
        `modules`, synthesize one based on its declared `module_type`.

        Today we expand transformer encoder/decoder stacks into N standard
        TransformerEncoderLayer / TransformerDecoderLayer blocks. Each block
        is itself a leaf `layerNode` using the canonical torch op, so the
        generated model exports to runnable PyTorch.
        """
        # layerType.id of each dangling moduleNode -> position among same-id
        # siblings (needed for ResNet layer1..layer4 -> layers[0..3]).
        sibling_index: Dict[str, int] = {}
        for node in list(graph.root_nodes):
            if node.type != "moduleNode":
                continue
            module_id = node.id
            if module_id in graph.modules:
                continue
            params = node.data.parameters if node.data else {}
            layer_id = (
                node.data.layerType.id
                if node.data and node.data.layerType
                else ""
            ).lower()
            module_type = str(params.get("module_type") or "").lower()

            stage_index = sibling_index.get(layer_id, 0)
            sibling_index[layer_id] = stage_index + 1

            module_def: Optional[ModuleDefinition] = None
            if module_type == "vit_encoder_stack":
                module_def = self._build_vit_block(
                    module_id=module_id,
                    config=config,
                    params=params,
                )
            elif module_type == "dit_block_stack":
                module_def = self._build_dit_block(
                    module_id=module_id,
                    config=config,
                    params=params,
                )
            elif module_type in {"encoder_stack", "decoder_stack"}:
                module_def = self._build_transformer_stack(
                    module_id=module_id,
                    module_type=module_type,
                    config=config,
                    params=params,
                )
            else:
                module_def = self._build_named_module(
                    module_id=module_id,
                    layer_id=layer_id,
                    config=config,
                    params=params,
                    stage_index=stage_index,
                )

            if module_def is None:
                continue
            graph.modules[module_id] = module_def
            # Mirror repeat metadata onto the placeholder root node so the
            # frontend can read it from data.parameters directly.
            for key in ("repeat_count", "repeat_block_label"):
                if key in module_def.parameters:
                    node.data.parameters[key] = module_def.parameters[key]

    def _build_transformer_stack(
        self,
        module_id: str,
        module_type: str,
        config: Dict[str, Any],
        params: Dict[str, Any],
    ) -> ModuleDefinition:
        """Build a stack of N TransformerEncoder/DecoderLayer leaves."""
        from models.graph_models import LayerType, LayerData

        # num_layers may come through as the templated literal "{num_layers}"
        # when substitution missed; fall back to config or a safe default.
        raw_n = params.get("num_layers") or config.get("num_layers") or 6
        try:
            num_layers = int(raw_n)
        except (TypeError, ValueError):
            num_layers = int(config.get("num_layers", 6))
        num_layers = max(1, min(num_layers, 96))  # sanity clamp

        d_model = int(config.get("d_model", 768))
        num_heads = int(config.get("num_heads", 12))
        d_ff = int(config.get("d_ff", d_model * 4))

        layer_op = "TransformerEncoderLayer" if module_type == "encoder_stack" else "TransformerDecoderLayer"
        leaf_layer_type = LayerType(
            id=layer_op.lower(),
            name=layer_op,
            parameters=[],
        )

        internal_nodes = []
        internal_edges = []
        for i in range(num_layers):
            node_id = f"{module_id}__layer_{i}"
            layer_params = {
                "d_model": d_model,
                "nhead": num_heads,
                "dim_feedforward": d_ff,
                "dropout": float(config.get("dropout", 0.1)),
                "batch_first": True,
            }
            internal_nodes.append(
                Node(
                    id=node_id,
                    type="layerNode",
                    position={"x": 120 + i * 160, "y": 0},
                    data=LayerData(
                        layerType=leaf_layer_type,
                        parameters=layer_params,
                        moduleType=None,
                    ),
                )
            )
            if i > 0:
                prev_id = f"{module_id}__layer_{i - 1}"
                internal_edges.append(
                    Edge(id=f"{module_id}__e_{i}", source=prev_id, target=node_id)
                )

        return ModuleDefinition(
            module_id=module_id,
            parent_id=None,
            internal_nodes=internal_nodes,
            internal_edges=internal_edges,
            parameters={
                **params,
                "name": params.get("name", module_id),
                "num_layers": num_layers,
                "d_model": d_model,
                "num_heads": num_heads,
                "d_ff": d_ff,
                "repeat_count": num_layers,
                "repeat_block_label": layer_op,
            },
        )

    def _build_vit_block(
        self,
        module_id: str,
        config: Dict[str, Any],
        params: Dict[str, Any],
    ) -> ModuleDefinition:
        """Build the internals of the ViT encoder block as ONE runnable
        nn.TransformerEncoderLayer (the canonical LN+MHA+MLP+residual block),
        repeated ×N. Using a single self-contained, single-input layer keeps the
        dry forward pass + code export runnable (a bare nn.MultiheadAttention
        node can't run in the sequential executor — it needs q,k,v).

        Patches arrive flattened as [B, C, P] (C=embed channels as the sequence,
        P=#patches as the feature dim), so the encoder's d_model is P = (image /
        patch)^2 and nhead must divide it.
        """
        from models.graph_models import LayerType, LayerData

        image_size = int(config.get("image_size", 224))
        patch_size = int(config.get("patch_size", 16))
        num_patches = (image_size // patch_size) ** 2          # 196 for 224/16
        num_layers = int(config.get("num_layers", 12))

        # Pick an attention-head count that divides the model dim.
        d_model = num_patches
        nhead = next((h for h in (8, 7, 4, 2, 1) if d_model % h == 0), 1)
        dim_feedforward = max(64, d_model * 2)
        seq_shape = f"[B, {int(config.get('d_model', 768))}, {num_patches}]"

        node_id = f"{module_id}__encoder_layer"
        encoder = Node(
            id=node_id,
            type="layerNode",
            position={"x": 60, "y": 0},
            data=LayerData(
                layerType=LayerType(id="transformerencoderlayer", name="TransformerEncoderLayer", parameters=[]),
                parameters={
                    "d_model": d_model,
                    "nhead": nhead,
                    "dim_feedforward": dim_feedforward,
                    "dropout": float(config.get("dropout", 0.0)),
                    "batch_first": True,
                    "_title": "Encoder Layer",
                    "_subtitle": f"d={d_model} · heads={nhead}",
                    "_shape": seq_shape,
                    "_accent": "var(--cat-attention)",
                },
                moduleType=None,
            ),
        )

        return ModuleDefinition(
            module_id=module_id,
            parent_id=None,
            internal_nodes=[encoder],
            internal_edges=[],
            input_node_id=node_id,
            output_node_id=node_id,
            parameters={
                **params,
                "name": params.get("name", module_id),
                "d_model": d_model,
                "num_heads": nhead,
                "d_ff": dim_feedforward,
                "repeat_count": num_layers,
                "repeat_block_label": "EncoderLayer",
            },
        )

    def _build_dit_block(
        self,
        module_id: str,
        config: Dict[str, Any],
        params: Dict[str, Any],
    ) -> ModuleDefinition:
        """Build the internals of the DiT (Diffusion Transformer) block stack as
        ONE runnable nn.TransformerEncoderLayer in pre-norm mode (norm_first),
        repeated ×depth. A DiTBlock is LN→attention→residual→LN→MLP→residual;
        the pre-norm encoder layer is the canonical runnable stand-in (the adaLN
        timestep/class conditioning is auxiliary and can't run in the single-
        input dry forward, exactly as ViT omits its CLS/pos-embed).

        Patches arrive flattened as [B, C, P] (C=embed channels as the sequence,
        P=#patches as the feature dim), so the encoder's d_model is P =
        (input_size / patch_size)^2 and nhead must divide it.
        """
        from models.graph_models import LayerType, LayerData

        input_size = int(config.get("input_size", 32))
        patch_size = int(config.get("patch_size", 2))
        num_patches = (input_size // patch_size) ** 2          # 256 for 32/2
        num_layers = int(config.get("num_layers", 12))

        # Pick an attention-head count that divides the model dim.
        d_model = num_patches
        nhead = next((h for h in (8, 6, 4, 2, 1) if d_model % h == 0), 1)
        dim_feedforward = max(64, d_model * 4)                 # DiT mlp_ratio=4
        seq_shape = f"[B, {int(config.get('hidden_size', 384))}, {num_patches}]"

        node_id = f"{module_id}__block"
        block = Node(
            id=node_id,
            type="layerNode",
            position={"x": 60, "y": 0},
            data=LayerData(
                layerType=LayerType(id="transformerencoderlayer", name="TransformerEncoderLayer", parameters=[]),
                parameters={
                    "d_model": d_model,
                    "nhead": nhead,
                    "dim_feedforward": dim_feedforward,
                    "dropout": float(config.get("dropout", 0.0)),
                    "batch_first": True,
                    "norm_first": True,
                    "_title": "DiT Block",
                    "_subtitle": f"d={d_model} · heads={nhead}",
                    "_shape": seq_shape,
                    "_accent": "var(--cat-attention)",
                },
                moduleType=None,
            ),
        )

        return ModuleDefinition(
            module_id=module_id,
            parent_id=None,
            internal_nodes=[block],
            internal_edges=[],
            input_node_id=node_id,
            output_node_id=node_id,
            parameters={
                **params,
                "name": params.get("name", module_id),
                "d_model": d_model,
                "num_heads": nhead,
                "d_ff": dim_feedforward,
                "repeat_count": num_layers,
                "repeat_block_label": "DiTBlock",
            },
        )

    # ── Generic leaf/chain builders ──────────────────────────────────────
    def _chain(self, module_id: str, specs: List[tuple]) -> tuple:
        """Build a sequential chain of leaf layerNodes from specs.

        Each spec is (layer_id, layer_name, params). Returns
        (internal_nodes, internal_edges, input_node_id, output_node_id).
        Layers are laid out left-to-right; every 8 layers wraps to a new row.
        """
        from models.graph_models import LayerType, LayerData

        internal_nodes: List[Node] = []
        internal_edges: List[Edge] = []
        for i, (layer_id, layer_name, layer_params) in enumerate(specs):
            node_id = f"{module_id}__n_{i}"
            row, col = divmod(i, 8)
            internal_nodes.append(
                Node(
                    id=node_id,
                    type="layerNode",
                    position={"x": 120 + col * 160, "y": row * 120},
                    data=LayerData(
                        layerType=LayerType(id=layer_id, name=layer_name, parameters=[]),
                        parameters=dict(layer_params),
                        moduleType=None,
                    ),
                )
            )
            if i > 0:
                internal_edges.append(
                    Edge(
                        id=f"{module_id}__e_{i}",
                        source=f"{module_id}__n_{i - 1}",
                        target=node_id,
                    )
                )
        input_id = f"{module_id}__n_0" if internal_nodes else None
        output_id = f"{module_id}__n_{len(internal_nodes) - 1}" if internal_nodes else None
        return internal_nodes, internal_edges, input_id, output_id

    @staticmethod
    def _to_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _build_named_module(
        self,
        module_id: str,
        layer_id: str,
        config: Dict[str, Any],
        params: Dict[str, Any],
        stage_index: int,
    ) -> Optional[ModuleDefinition]:
        """Synthesize internals for a non-transformer placeholder moduleNode.

        Dispatch is keyed off the node's ``layerType.id``. Each branch emits a
        well-formed, renderable conv/linear chain using canonical torch.nn
        layer names. Channel math is approximate by design; the goal is
        non-empty, exportable internals (see task spec).
        """
        conv = lambda i, o, k, s, p: ("conv2d", "Conv2d", {
            "in_channels": i, "out_channels": o, "kernel_size": k,
            "stride": s, "padding": p,
        })
        bn = lambda c: ("batchnorm2d", "BatchNorm2d", {"num_features": c})
        relu = ("relu", "ReLU", {"inplace": True})
        maxpool = ("maxpool2d", "MaxPool2d", {"kernel_size": 2, "stride": 2})
        linear = lambda i, o: ("linear", "Linear", {"in_features": i, "out_features": o})
        flatten = ("flatten", "Flatten", {})
        dropout = lambda p: ("dropout", "Dropout", {"p": p})

        repeat_count: Optional[int] = None
        repeat_label: Optional[str] = None
        specs: List[tuple] = []

        if layer_id == "resnet_layer":
            # One ResNet stage = `blocks` bottleneck blocks. `blocks` comes
            # through as the un-substituted literal "{layers[N]}"; resolve it
            # from config['layers'][stage_index].
            layers_cfg = config.get("layers", [3, 4, 6, 3])
            if stage_index < len(layers_cfg):
                blocks = self._to_int(layers_cfg[stage_index], 2)
            else:
                blocks = self._to_int(params.get("blocks"), 2)
            blocks = max(1, min(blocks, 64))
            planes = self._to_int(params.get("planes"), 64)
            stride = self._to_int(params.get("stride"), 1)
            width = planes * 4  # bottleneck expansion
            # Channels entering this stage = expansion of the PREVIOUS stage's
            # planes (layer1 receives the 64-ch stem output). Each bottleneck
            # outputs `width`; chain `in_c` so every conv's in_channels equals
            # its incoming channel count across blocks and across stages.
            in_c = self._to_int(params.get("in_channels"),
                                64 if planes == 64 else (planes // 2) * 4)
            for b in range(blocks):
                s = stride if b == 0 else 1
                specs += [
                    conv(in_c, planes, 1, 1, 0), bn(planes), relu,
                    conv(planes, planes, 3, s, 1), bn(planes), relu,
                    conv(planes, width, 1, 1, 0), bn(width), relu,
                ]
                in_c = width
            repeat_count, repeat_label = blocks, "Bottleneck"

        elif layer_id == "vgg_features":
            cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
                   512, 512, 512, "M", 512, 512, 512, "M"]
            in_c = 3
            for v in cfg:
                if v == "M":
                    specs.append(maxpool)
                else:
                    specs += [conv(in_c, v, 3, 1, 1), relu]
                    in_c = v
            repeat_count, repeat_label = 13, "ConvBlock"

        elif layer_id == "vgg_classifier":
            num_classes = self._to_int(params.get("num_classes"), 1000)
            p = float(params.get("dropout", 0.5))
            specs = [
                flatten,
                linear(512 * 7 * 7, 4096), relu, dropout(p),
                linear(4096, 4096), relu, dropout(p),
                linear(4096, num_classes),
            ]

        elif layer_id == "inception_stack":
            # Approximate Inception-v3 mixed blocks as conv stages. The root
            # `conv2d_1a` stem feeds 32 channels into this module, so the first
            # conv must accept 32 (not the textbook 192) to chain cleanly.
            in_c = 32
            stages = [(256, 3), (288, 3), (768, 4), (2048, 1)]
            for out_c, _ in stages:
                specs += [conv(in_c, out_c, 3, 1, 1), bn(out_c), relu, maxpool]
                in_c = out_c
            repeat_count, repeat_label = len(stages), "InceptionBlock"

        elif layer_id == "mobilenet_features":
            # The root `conv1` stem already produces 32 channels (3->32), so
            # this module receives 32 channels — do NOT replay the 3->32 stem
            # here or the first conv would expect 3 and reject the 32-ch input.
            in_c = 32
            channels = [16, 24, 32, 64, 96, 160, 320, 1280]
            strides = [1, 2, 2, 2, 1, 2, 1, 1]
            for out_c, s in zip(channels, strides):
                specs += [conv(in_c, out_c, 3, s, 1), bn(out_c), relu]
                in_c = out_c
            repeat_count, repeat_label = len(channels), "InvertedResidual"

        elif layer_id == "efficientnet_stem":
            out_c = self._to_int(params.get("out_channels"), 32)
            specs = [conv(3, out_c, 3, 2, 1), bn(out_c), relu]

        elif layer_id == "efficientnet_blocks":
            in_c = 32
            channels = [16, 24, 40, 80, 112, 192, 320]
            strides = [1, 2, 2, 2, 1, 2, 1]
            for out_c, s in zip(channels, strides):
                specs += [conv(in_c, out_c, 3, s, 1), bn(out_c), relu]
                in_c = out_c
            repeat_count, repeat_label = len(channels), "MBConvBlock"

        elif layer_id == "efficientnet_head":
            num_classes = self._to_int(params.get("num_classes"), 1000)
            p = float(params.get("dropout_rate", 0.2))
            specs = [
                conv(320, 1280, 1, 1, 0), bn(1280), relu,
                ("adaptiveavgpool2d", "AdaptiveAvgPool2d", {"output_size": [1, 1]}),
                flatten, dropout(p), linear(1280, num_classes),
            ]

        elif layer_id == "mapping_network":
            n = self._to_int(params.get("num_layers"), 8)
            n = max(1, min(n, 32))
            latent = self._to_int(params.get("latent_dim"), 512)
            style = self._to_int(params.get("style_dim"), 512)
            dims = [latent] + [style] * n
            for i in range(n):
                specs += [linear(dims[i], dims[i + 1]),
                          ("leakyrelu", "LeakyReLU", {"negative_slope": 0.2})]
            repeat_count, repeat_label = n, "MappingLayer"

        elif layer_id in ("synthesis_network", "generator"):
            latent = self._to_int(params.get("style_dim") or params.get("latent_dim"), 512)
            # The latent / style vector arrives as a rank-2 [N, latent] tensor,
            # but the ConvTranspose2d stack needs a rank-4 [N, C, H, W] map.
            # Unflatten the channel dim into [latent, 1, 1] to seed a 1x1 grid,
            # exactly the [N, latent, 1, 1] start of a DCGAN/StyleGAN generator.
            specs += [("unflatten", "Unflatten", {
                "dim": 1, "unflattened_size": [latent, 1, 1]})]
            feats = [512, 256, 128, 64, 32]
            in_c = latent
            for f in feats:
                specs += [
                    ("convtranspose2d", "ConvTranspose2d", {
                        "in_channels": in_c, "out_channels": f,
                        "kernel_size": 4, "stride": 2, "padding": 1}),
                    bn(f), relu,
                ]
                in_c = f
            out_ch = self._to_int(params.get("image_channels") or params.get("output_channels"), 3)
            specs += [conv(in_c, out_ch, 3, 1, 1), ("tanh", "Tanh", {})]
            repeat_count, repeat_label = len(feats), "UpsampleBlock"

        elif layer_id in ("style_discriminator", "discriminator"):
            in_ch = self._to_int(params.get("image_channels") or params.get("input_channels"), 3)
            feats = [64, 128, 256, 512]
            in_c = in_ch
            for f in feats:
                specs += [conv(in_c, f, 4, 2, 1), bn(f),
                          ("leakyrelu", "LeakyReLU", {"negative_slope": 0.2})]
                in_c = f
            # Collapse the remaining spatial grid to 1x1 so the flattened width
            # is exactly in_c, independent of the (variable) input image size,
            # which keeps the final Linear's in_features correct.
            specs += [("adaptiveavgpool2d", "AdaptiveAvgPool2d", {"output_size": [1, 1]}),
                      flatten, linear(in_c, 1)]
            repeat_count, repeat_label = len(feats), "DownsampleBlock"

        elif layer_id == "sequence_pooling":
            # The LSTM/GRU emits [N, seq, hidden] (batch_first). To pool over the
            # sequence and hand the classifier a fixed [N, hidden] vector we run a
            # 1x1 Conv1d that reads the sequence positions as channels (in=seq_len)
            # and projects them to `hidden` filters, then collapse the remaining
            # axis with AdaptiveMaxPool1d. (A direct transpose op isn't available
            # in the symbolic validator, so this is the shape-consistent stand-in
            # for max/mean-over-time pooling — analogous to ViT's class-token pool.)
            hidden = self._to_int(params.get("hidden_size"), 512)
            seq_len = self._to_int(params.get("seq_len"), 10)
            specs = [
                ("conv1d", "Conv1d", {
                    "in_channels": seq_len, "out_channels": hidden,
                    "kernel_size": 1, "stride": 1, "padding": 0}),
                ("adaptivemaxpool1d", "AdaptiveMaxPool1d", {"output_size": 1}),
                flatten, linear(hidden, hidden), relu,
            ]

        elif layer_id == "attention":
            # Seq2seq cross-attention context layer. The encoder hands a
            # [N, L, hidden] sequence; attention produces a same-shaped context
            # sequence. Represent it as a shape-preserving Linear(hidden->hidden)
            # projection (the canonical validator can't run a bare q/k/v attention
            # op on a single input, so this is the runnable, shape-consistent
            # stand-in — analogous to how ViT's block stands in for full MHA).
            hidden = self._to_int(params.get("hidden_size"), 512)
            specs = [linear(hidden, hidden), ("tanh", "Tanh", {})]

        elif layer_id == "text_cnn":
            # Embedding emits [N, L, embed_dim]; Conv1d wants [N, C, L]. Without a
            # transpose op in the symbolic validator we read the sequence
            # positions as the channel axis (first conv in_channels = seq_len) and
            # let the filters slide over the embedding axis — a shape-consistent
            # stand-in for the n-gram Conv1d bank. Filters are stacked
            # sequentially, then AdaptiveMaxPool1d collapses to a fixed
            # [N, num_filters] vector for the classifier.
            num_filters = self._to_int(params.get("num_filters"), 100)
            seq_len = self._to_int(params.get("seq_len"), 10)
            try:
                import ast
                filter_sizes = ast.literal_eval(str(params.get("filter_sizes", "[3, 4, 5]")))
            except (ValueError, SyntaxError):
                filter_sizes = [3, 4, 5]
            in_c = seq_len
            for k in filter_sizes:
                specs.append((
                    "conv1d", "Conv1d",
                    {"in_channels": in_c, "out_channels": num_filters,
                     "kernel_size": int(k), "stride": 1, "padding": int(k) // 2},
                ))
                specs.append(relu)
                in_c = num_filters
            specs += [
                ("adaptivemaxpool1d", "AdaptiveMaxPool1d", {"output_size": 1}),
                flatten, linear(num_filters, num_filters),
            ]
            repeat_count, repeat_label = len(filter_sizes), "TextConvBlock"

        else:
            return None

        internal_nodes, internal_edges, input_id, output_id = self._chain(module_id, specs)
        module_params: Dict[str, Any] = {
            **params,
            "name": params.get("name", module_id),
        }
        if repeat_count is not None:
            module_params["repeat_count"] = repeat_count
            module_params["repeat_block_label"] = repeat_label

        return ModuleDefinition(
            module_id=module_id,
            parent_id=None,
            internal_nodes=internal_nodes,
            internal_edges=internal_edges,
            input_node_id=input_id,
            output_node_id=output_id,
            parameters=module_params,
        )

    def _create_module_definition_from_data(self, module_data: Dict[str, Any]) -> ModuleDefinition:
        """Create a ModuleDefinition from template data"""
        internal_nodes = []
        internal_edges = []
        
        # Process internal nodes
        for node_data in module_data.get("internal_nodes", []):
            node = self._create_node_from_data(node_data)
            internal_nodes.append(node)
        
        # Process internal edges
        for edge_data in module_data.get("internal_edges", []):
            edge = Edge(
                id=edge_data["id"],
                source=edge_data["source"],
                target=edge_data["target"]
            )
            internal_edges.append(edge)
        
        return ModuleDefinition(
            module_id=module_data["module_id"],
            parent_id=module_data.get("parent_id"),
            internal_nodes=internal_nodes,
            internal_edges=internal_edges,
            parameters=module_data.get("parameters", {})
        )
    
    # Customization Helper Methods
    def _remove_module(self, graph: HierarchicalGraph, customization: Dict[str, Any]) -> HierarchicalGraph:
        """Remove a module from the graph"""
        module_id = customization.get("module_id")
        
        # Remove from modules dict
        if module_id in graph.modules:
            del graph.modules[module_id]
        
        # Remove nodes referencing this module
        graph.root_nodes = [n for n in graph.root_nodes if n.id != module_id]
        
        # Remove edges connected to this module
        graph.root_edges = [e for e in graph.root_edges if e.source != module_id and e.target != module_id]
        
        return graph
    
    def _add_module(self, graph: HierarchicalGraph, customization: Dict[str, Any]) -> HierarchicalGraph:
        """Add a module to the graph"""
        # Implementation would add new modules based on customization
        return graph
    
    def _modify_module(self, graph: HierarchicalGraph, customization: Dict[str, Any]) -> HierarchicalGraph:
        """Modify an existing module"""
        # Implementation would modify module parameters or structure
        return graph
    
    def _change_parameter(self, graph: HierarchicalGraph, customization: Dict[str, Any]) -> HierarchicalGraph:
        """Change a parameter value"""
        # Implementation would update parameter values in nodes
        return graph
    
    def _replace_module(self, graph: HierarchicalGraph, customization: Dict[str, Any]) -> HierarchicalGraph:
        """Replace one module with another"""
        # Implementation would replace modules
        return graph
    
    def _get_removable_modules(self, template: ModelTemplate) -> List[str]:
        """Get list of modules that can be removed from this template"""
        # Return modules that are optional for the architecture
        removable = []
        
        if template.id == "transformer_encoder_decoder":
            removable.extend(["encoder_module", "decoder_module"])
        elif template.id == "transformer_encoder_only":
            removable.extend(["encoder_stack"])
        
        return removable
    
    def _get_addable_modules(self, template: ModelTemplate) -> List[str]:
        """Get list of modules that can be added to this template"""
        # Return compatible modules for this template type
        addable = []
        
        if "transformer" in template.id:
            addable.extend(["dropout_layer", "layer_norm", "residual_connection"])
        
        return addable
    
    def _get_replaceable_modules(self, template: ModelTemplate) -> List[Dict[str, Any]]:
        """Get list of modules that can be replaced and their alternatives"""
        replaceable = []
        
        if "transformer" in template.id:
            replaceable.append({
                "module": "multi_head_attention",
                "alternatives": ["local_attention", "sparse_attention", "linear_attention"]
            })
        
        return replaceable
