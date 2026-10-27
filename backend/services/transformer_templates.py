"""Transformer-based model templates and architectures - Comprehensive Implementation"""
from typing import Dict, List, Any, Optional
from models.template_models import ModelTemplate, ModuleTemplate, PopularModel, TemplateVariant
from models.graph_models import Graph, Node, Edge, LayerType, LayerData


class TransformerTemplates:
    """Service for Transformer-based architectures with comprehensive customization"""
    
    def get_all_templates(self) -> Dict[str, ModelTemplate]:
        """Get all transformer templates"""
        return {
            "transformer_encoder_decoder": self._create_transformer_encoder_decoder(),
            "transformer_encoder_only": self._create_transformer_encoder_only(),
            "transformer_decoder_only": self._create_transformer_decoder_only(),
            "vision_transformer": self._create_vision_transformer(),
        }
    
    def get_module_templates(self) -> Dict[str, ModuleTemplate]:
        """Get transformer module templates"""
        return {
            "multi_head_attention": self._create_multi_head_attention(),
            "transformer_encoder_block": self._create_transformer_encoder_block(),
            "feed_forward_network": self._create_feed_forward_network(),
        }
    
    def get_popular_models(self) -> Dict[str, PopularModel]:
        """Get popular transformer-based models"""
        return {
            "bert_base": self._create_bert_base(),
            "gpt2_small": self._create_gpt2_small(),
            "t5_base": self._create_t5_base(),
            "vit_base": self._create_vit_base(),
        }
    
    def get_template_variants(self) -> Dict[str, TemplateVariant]:
        """Get template variants for common use cases"""
        return {
            "bert_classifier": self._create_bert_classifier_variant(),
            "gpt_generator": self._create_gpt_generator_variant(),
        }
    
    def _create_transformer_encoder_only(self) -> ModelTemplate:
        """Create encoder-only transformer template (BERT-style)"""
        return ModelTemplate(
            id="transformer_encoder_only",
            name="Transformer Encoder-Only",
            description="Encoder-only transformer for bidirectional tasks like classification (BERT-style)",
            category="nlp",
            sub_category="transformer",
            difficulty="intermediate",
            tags=["transformer", "encoder", "bert", "classification"],
            
            base_config={
                "d_model": 768,
                "num_heads": 12,
                "num_layers": 12,
                "d_ff": 3072,
                "dropout": 0.1,
                "vocab_size": 30522,
                "max_length": 512,
                "num_classes": 2
            },
            
            customizable_params=[
                {
                    "name": "d_model",
                    "type": "number",
                    "min": 128,
                    "max": 2048,
                    "step": 64,
                    "description": "Model dimension (hidden size)",
                    "category": "architecture"
                },
                {
                    "name": "num_heads", 
                    "type": "number",
                    "min": 2,
                    "max": 32,
                    "description": "Number of attention heads",
                    "category": "attention"
                },
                {
                    "name": "num_layers",
                    "type": "number", 
                    "min": 1,
                    "max": 24,
                    "description": "Number of encoder layers",
                    "category": "architecture"
                },
                {
                    "name": "num_classes",
                    "type": "number",
                    "min": 2,
                    "max": 1000,
                    "description": "Number of output classes",
                    "category": "output"
                }
            ],
            
            template_structure={
                "root_nodes": [
                    {
                        "id": "input_node",
                        "type": "inputNode", 
                        "position": {"x": 100, "y": 200},
                        "data": {
                            "layerType": {"id": "input", "name": "Input", "parameters": []},
                            "parameters": {"shape": "1,512", "name": "input_tokens"}
                        }
                    },
                    {
                        "id": "token_embeddings",
                        "type": "layerNode",
                        "position": {"x": 300, "y": 200},
                        "data": {
                            "layerType": {"id": "embedding", "name": "Embedding", "parameters": []},
                            "parameters": {"num_embeddings": "{vocab_size}", "embedding_dim": "{d_model}"}
                        }
                    },
                    {
                        "id": "encoder_stack",
                        "type": "moduleNode",
                        "position": {"x": 500, "y": 200}, 
                        "data": {
                            "layerType": {"id": "module", "name": "Module", "parameters": []},
                            "parameters": {"module_type": "encoder_stack", "num_layers": "{num_layers}"},
                            "moduleType": "custom"
                        }
                    },
                    {
                        "id": "classifier_head",
                        "type": "layerNode",
                        "position": {"x": 700, "y": 200},
                        "data": {
                            "layerType": {"id": "linear", "name": "Linear", "parameters": []},
                            "parameters": {"in_features": "{d_model}", "out_features": "{num_classes}"}
                        }
                    },
                    {
                        "id": "output_node",
                        "type": "outputNode",
                        "position": {"x": 900, "y": 200},
                        "data": {
                            "layerType": {"id": "output", "name": "Output", "parameters": []},
                            "parameters": {"name": "logits"}
                        }
                    }
                ],
                "root_edges": [
                    {"id": "e1", "source": "input_node", "target": "token_embeddings"},
                    {"id": "e2", "source": "token_embeddings", "target": "encoder_stack"},
                    {"id": "e3", "source": "encoder_stack", "target": "classifier_head"},
                    {"id": "e4", "source": "classifier_head", "target": "output_node"}
                ]
            },
            
            variants=[
                {
                    "id": "bert_base",
                    "name": "BERT Base",
                    "description": "Standard BERT Base configuration",
                    "config_overrides": {
                        "d_model": 768,
                        "num_heads": 12,
                        "num_layers": 12,
                        "d_ff": 3072
                    }
                }
            ],
            
            typical_use_cases=[
                "Text Classification",
                "Named Entity Recognition", 
                "Question Answering",
                "Sentiment Analysis"
            ],
            
            paper_reference="BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)",
            
            minimum_input_shape=[1, 1],
            recommended_input_shape=[1, 512]
        )
    
    def _create_transformer_decoder_only(self) -> ModelTemplate:
        """Create decoder-only transformer template (GPT-style)"""
        return ModelTemplate(
            id="transformer_decoder_only",
            name="Transformer Decoder-Only",
            description="Decoder-only transformer for autoregressive generation (GPT-style)",
            category="nlp",
            sub_category="transformer",
            difficulty="intermediate", 
            tags=["transformer", "decoder", "gpt", "generation"],
            
            base_config={
                "d_model": 768,
                "num_heads": 12,
                "num_layers": 12,
                "d_ff": 3072,
                "dropout": 0.1,
                "vocab_size": 50257,
                "max_length": 1024
            },
            
            customizable_params=[
                {
                    "name": "d_model",
                    "type": "number",
                    "min": 128,
                    "max": 4096,
                    "step": 64,
                    "description": "Model dimension",
                    "category": "architecture"
                },
                {
                    "name": "num_heads",
                    "type": "number",
                    "min": 2,
                    "max": 64,
                    "description": "Number of attention heads",
                    "category": "attention"
                },
                {
                    "name": "num_layers",
                    "type": "number",
                    "min": 1,
                    "max": 48,
                    "description": "Number of decoder layers",
                    "category": "architecture"
                }
            ],
            
            template_structure={
                "root_nodes": [
                    {
                        "id": "input_node",
                        "type": "inputNode",
                        "position": {"x": 100, "y": 200},
                        "data": {
                            "layerType": {"id": "input", "name": "Input", "parameters": []},
                            "parameters": {"shape": "1,1024", "name": "input_tokens"}
                        }
                    },
                    {
                        "id": "token_embeddings",
                        "type": "layerNode",
                        "position": {"x": 300, "y": 200},
                        "data": {
                            "layerType": {"id": "embedding", "name": "Embedding", "parameters": []},
                            "parameters": {"num_embeddings": "{vocab_size}", "embedding_dim": "{d_model}"}
                        }
                    },
                    {
                        "id": "decoder_stack",
                        "type": "moduleNode",
                        "position": {"x": 500, "y": 200},
                        "data": {
                            "layerType": {"id": "module", "name": "Module", "parameters": []},
                            "parameters": {"module_type": "decoder_stack", "num_layers": "{num_layers}"},
                            "moduleType": "custom"
                        }
                    },
                    {
                        "id": "lm_head",
                        "type": "layerNode",
                        "position": {"x": 700, "y": 200},
                        "data": {
                            "layerType": {"id": "linear", "name": "Linear", "parameters": []},
                            "parameters": {"in_features": "{d_model}", "out_features": "{vocab_size}"}
                        }
                    },
                    {
                        "id": "output_node",
                        "type": "outputNode",
                        "position": {"x": 900, "y": 200},
                        "data": {
                            "layerType": {"id": "output", "name": "Output", "parameters": []},
                            "parameters": {"name": "logits"}
                        }
                    }
                ],
                "root_edges": [
                    {"id": "e1", "source": "input_node", "target": "token_embeddings"},
                    {"id": "e2", "source": "token_embeddings", "target": "decoder_stack"},
                    {"id": "e3", "source": "decoder_stack", "target": "lm_head"},
                    {"id": "e4", "source": "lm_head", "target": "output_node"}
                ]
            },
            
            variants=[
                {
                    "id": "gpt_small",
                    "name": "GPT Small",
                    "description": "Small GPT configuration",
                    "config_overrides": {
                        "d_model": 768,
                        "num_heads": 12,
                        "num_layers": 12,
                        "d_ff": 3072
                    }
                }
            ],
            
            typical_use_cases=[
                "Text Generation",
                "Code Generation", 
                "Creative Writing",
                "Chatbots"
            ],
            
            paper_reference="Language Models are Unsupervised Multitask Learners (Radford et al., 2019)",
            
            minimum_input_shape=[1, 1],
            recommended_input_shape=[1, 1024]
        )
    
    def _create_transformer_encoder_decoder(self) -> ModelTemplate:
        """Create full Transformer (encoder-decoder) template"""
        return ModelTemplate(
            id="transformer_encoder_decoder",
            name="Transformer (Encoder-Decoder)",
            description="Full transformer with encoder and decoder for seq2seq tasks",
            category="nlp",
            sub_category="transformer",
            difficulty="advanced",
            tags=["transformer", "encoder-decoder", "seq2seq"],
            
            base_config={
                "d_model": 512,
                "num_heads": 8,
                "num_encoder_layers": 6,
                "num_decoder_layers": 6,
                "d_ff": 2048,
                "dropout": 0.1,
                "vocab_size": 30000
            },
            
            customizable_params=[
                {
                    "name": "d_model",
                    "type": "number",
                    "min": 128,
                    "max": 2048,
                    "description": "Model dimension"
                },
                {
                    "name": "num_encoder_layers",
                    "type": "number",
                    "min": 1,
                    "max": 24,
                    "description": "Number of encoder layers"
                },
                {
                    "name": "num_decoder_layers",
                    "type": "number",
                    "min": 1,
                    "max": 24,
                    "description": "Number of decoder layers"
                }
            ],
            
            template_structure={
                "root_nodes": [
                    {
                        "id": "input_node",
                        "type": "inputNode",
                        "position": {"x": 100, "y": 200},
                        "data": {
                            "layerType": {"id": "input", "name": "Input", "parameters": []},
                            "parameters": {"shape": "1,512", "name": "input_tokens"}
                        }
                    },
                    {
                        "id": "token_embeddings",
                        "type": "layerNode",
                        "position": {"x": 280, "y": 200},
                        "data": {
                            "layerType": {"id": "embedding", "name": "Embedding", "parameters": []},
                            "parameters": {"num_embeddings": "{vocab_size}", "embedding_dim": "{d_model}"}
                        }
                    },
                    {
                        "id": "encoder_stack",
                        "type": "moduleNode",
                        "position": {"x": 460, "y": 200},
                        "data": {
                            "layerType": {"id": "module", "name": "Module", "parameters": []},
                            "parameters": {"module_type": "encoder_stack", "num_layers": "{num_encoder_layers}"},
                            "moduleType": "custom"
                        }
                    },
                    {
                        "id": "decoder_stack",
                        "type": "moduleNode",
                        "position": {"x": 640, "y": 200},
                        "data": {
                            "layerType": {"id": "module", "name": "Module", "parameters": []},
                            "parameters": {"module_type": "decoder_stack", "num_layers": "{num_decoder_layers}"},
                            "moduleType": "custom"
                        }
                    },
                    {
                        "id": "output_proj",
                        "type": "layerNode",
                        "position": {"x": 820, "y": 200},
                        "data": {
                            "layerType": {"id": "linear", "name": "Linear", "parameters": []},
                            "parameters": {"in_features": "{d_model}", "out_features": "{vocab_size}"}
                        }
                    },
                    {
                        "id": "output_node",
                        "type": "outputNode",
                        "position": {"x": 1000, "y": 200},
                        "data": {
                            "layerType": {"id": "output", "name": "Output", "parameters": []},
                            "parameters": {"name": "logits"}
                        }
                    }
                ],
                "root_edges": [
                    {"id": "e1", "source": "input_node", "target": "token_embeddings"},
                    {"id": "e2", "source": "token_embeddings", "target": "encoder_stack"},
                    {"id": "e3", "source": "encoder_stack", "target": "decoder_stack"},
                    {"id": "e4", "source": "decoder_stack", "target": "output_proj"},
                    {"id": "e5", "source": "output_proj", "target": "output_node"}
                ]
            },

            variants=[],
            typical_use_cases=["Translation", "Summarization"],
            paper_reference="Attention Is All You Need (Vaswani et al., 2017)",
            minimum_input_shape=[1, 1],
            recommended_input_shape=[1, 512]
        )
    
    def _create_vision_transformer(self) -> ModelTemplate:
        """Create Vision Transformer (ViT) template.

        Renders left->right: Input -> Patch Embedding (Conv2d) ->
        Transformer Block (moduleNode, expanded, x12) -> Layer Norm ->
        Class Token (CLS pooling) -> Linear. Each root node carries the
        frontend display keys (_subtitle/_shape/_accent) inside data.parameters.
        """
        return ModelTemplate(
            id="vision_transformer",
            name="Vision Transformer (ViT)",
            description="Image classification transformer that splits the image into patches and applies a transformer encoder (Dosovitskiy et al., 2020)",
            category="computer_vision",
            sub_category="transformer",
            difficulty="advanced",
            tags=["transformer", "vision", "vit", "classification", "patches"],

            base_config={
                "d_model": 768,
                "num_heads": 12,
                "num_layers": 12,
                "d_ff": 3072,
                "dropout": 0.0,
                "patch_size": 16,
                "image_size": 224,
                "num_classes": 1000,
                "seq_shape": "[B, 196, 768]",
            },

            customizable_params=[
                {"name": "d_model", "type": "number", "min": 128, "max": 2048, "step": 64,
                 "description": "Embedding / hidden dimension", "category": "architecture"},
                {"name": "num_heads", "type": "number", "min": 2, "max": 32,
                 "description": "Number of attention heads", "category": "attention"},
                {"name": "num_layers", "type": "number", "min": 1, "max": 32,
                 "description": "Number of transformer encoder layers", "category": "architecture"},
                {"name": "num_classes", "type": "number", "min": 2, "max": 21843,
                 "description": "Number of output classes", "category": "output"},
            ],

            template_structure={
                # Runnable ViT: every step is a single-input torch op so the dry
                # forward pass + code export both run end-to-end. The encoder
                # block is one nn.TransformerEncoderLayer (the canonical
                # LN+MHA+MLP+residual block) repeated ×N. Patches are flattened
                # to a token sequence after the conv stem and mean-pooled before
                # the classifier.
                "root_nodes": [
                    {
                        "id": "input_node",
                        "type": "inputNode",
                        "position": {"x": 80, "y": 200},
                        "data": {
                            "layerType": {"id": "input", "name": "Input", "parameters": []},
                            "parameters": {
                                "name": "Input",
                                "shape": "1,3,224,224",
                                "_subtitle": "Image",
                                "_shape": "[B, 3, 224, 224]",
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
                                "in_channels": 3,
                                "out_channels": "{d_model}",
                                "kernel_size": "{patch_size}",
                                "stride": "{patch_size}",
                                "padding": 0,
                                "_subtitle": "Conv2d 16×16",
                                "_shape": "[B, 768, 14, 14]",
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
                                "_shape": "[B, 768, 196]",
                                "_accent": "var(--cat-utility)",
                            },
                        },
                    },
                    {
                        "id": "transformer_block",
                        "type": "moduleNode",
                        "position": {"x": 740, "y": 200},
                        "data": {
                            "layerType": {"id": "module", "name": "Module", "parameters": []},
                            "parameters": {
                                "name": "Transformer Block",
                                "module_type": "vit_encoder_stack",
                                "num_layers": "{num_layers}",
                            },
                            "moduleType": "custom",
                            "expanded": True,
                        },
                    },
                    {
                        "id": "pool",
                        "type": "layerNode",
                        "position": {"x": 960, "y": 200},
                        "data": {
                            "layerType": {"id": "adaptiveavgpool1d", "name": "AdaptiveAvgPool1d", "parameters": []},
                            "parameters": {
                                "name": "Class Token",
                                "output_size": 1,
                                "_subtitle": "pool",
                                "_shape": "[B, 768, 1]",
                                "_accent": "var(--cat-activation)",
                            },
                        },
                    },
                    {
                        "id": "flatten_out",
                        "type": "layerNode",
                        "position": {"x": 1140, "y": 200},
                        "data": {
                            "layerType": {"id": "flatten", "name": "Flatten", "parameters": []},
                            "parameters": {
                                "name": "Flatten",
                                "start_dim": 1,
                                "_subtitle": "vector",
                                "_shape": "[B, 768]",
                                "_accent": "var(--cat-utility)",
                            },
                        },
                    },
                    {
                        "id": "classifier_head",
                        "type": "layerNode",
                        "position": {"x": 1340, "y": 200},
                        "data": {
                            "layerType": {"id": "linear", "name": "Linear", "parameters": []},
                            "parameters": {
                                "name": "Linear",
                                "in_features": "{d_model}",
                                "out_features": "{num_classes}",
                                "_subtitle": "1000",
                                "_shape": "[B, 1000]",
                                "_accent": "var(--error)",
                            },
                        },
                    },
                    {
                        "id": "output_node",
                        "type": "outputNode",
                        "position": {"x": 1540, "y": 200},
                        "data": {
                            "layerType": {"id": "output", "name": "Output", "parameters": []},
                            "parameters": {
                                "name": "Output",
                                "_subtitle": "Logits",
                                "_shape": "[B, 1000]",
                                "_accent": "var(--error)",
                            },
                        },
                    },
                ],
                "root_edges": [
                    {"id": "e1", "source": "input_node", "target": "patch_embedding"},
                    {"id": "e2", "source": "patch_embedding", "target": "tokenize"},
                    {"id": "e3", "source": "tokenize", "target": "transformer_block"},
                    {"id": "e4", "source": "transformer_block", "target": "pool"},
                    {"id": "e5", "source": "pool", "target": "flatten_out"},
                    {"id": "e6", "source": "flatten_out", "target": "classifier_head"},
                    {"id": "e7", "source": "classifier_head", "target": "output_node"},
                ],
            },

            variants=[],
            typical_use_cases=["Image Classification", "Feature Extraction", "Transfer Learning"],
            paper_reference="An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Dosovitskiy et al., 2020)",
            minimum_input_shape=[1, 3, 224, 224],
            recommended_input_shape=[1, 3, 224, 224],
        )

    # Module Templates
    def _create_multi_head_attention(self) -> ModuleTemplate:
        """Multi-head attention module template"""
        return ModuleTemplate(
            id="multi_head_attention",
            name="Multi-Head Attention",
            description="Multi-head self-attention mechanism",
            category="attention",
            internal_graph=Graph(nodes=[], edges=[]),
            configurable_params={
                "d_model": "Model dimension",
                "num_heads": "Number of attention heads"
            },
            default_params={
                "d_model": 512,
                "num_heads": 8
            },
            input_specs={"shape": "[batch_size, seq_len, d_model]"},
            output_specs={"shape": "[batch_size, seq_len, d_model]"},
            pytorch_class_template="MultiHeadAttention"
        )
    
    def _create_transformer_encoder_block(self) -> ModuleTemplate:
        """Transformer encoder block template"""
        return ModuleTemplate(
            id="transformer_encoder_block",
            name="Transformer Encoder Block",
            description="Complete transformer encoder block",
            category="transformer",
            internal_graph=Graph(nodes=[], edges=[]),
            configurable_params={
                "d_model": "Model dimension",
                "num_heads": "Number of attention heads",
                "d_ff": "Feed-forward dimension"
            },
            default_params={
                "d_model": 512,
                "num_heads": 8,
                "d_ff": 2048
            },
            input_specs={"shape": "[batch_size, seq_len, d_model]"},
            output_specs={"shape": "[batch_size, seq_len, d_model]"},
            pytorch_class_template="TransformerEncoderBlock"
        )
    
    def _create_feed_forward_network(self) -> ModuleTemplate:
        """Feed-forward network module template"""
        return ModuleTemplate(
            id="feed_forward_network",
            name="Feed Forward Network",
            description="Position-wise feed-forward network",
            category="feedforward",
            internal_graph=Graph(nodes=[], edges=[]),
            configurable_params={
                "d_model": "Input/output dimension",
                "d_ff": "Hidden dimension"
            },
            default_params={
                "d_model": 512,
                "d_ff": 2048
            },
            input_specs={"shape": "[batch_size, seq_len, d_model]"},
            output_specs={"shape": "[batch_size, seq_len, d_model]"},
            pytorch_class_template="FeedForwardNetwork"
        )
    
    # Popular Models
    def _create_bert_base(self) -> PopularModel:
        """BERT Base model configuration"""
        return PopularModel(
            id="bert_base",
            name="BERT Base",
            full_name="Bidirectional Encoder Representations from Transformers - Base",
            description="Base BERT model with 110M parameters",
            category="nlp",
            architecture_type="transformer_encoder",
            parameter_count="110M",
            training_dataset="BookCorpus + English Wikipedia",
            training_details={
                "optimizer": "AdamW",
                "learning_rate": 1e-4,
                "batch_size": 256,
                "sequence_length": 512,
                "training_steps": "1M",
                "warmup_steps": 10000
            },
            template_id="transformer_encoder_only",
            config={
                "d_model": 768,
                "num_heads": 12,
                "num_layers": 12,
                "d_ff": 3072,
                "vocab_size": 30522,
                "num_classes": 2
            },
            typical_applications=["Text Classification", "NER", "QA"],
            performance_metrics={
                "glue_score": 79.6,
                "squad_f1": 88.5,
                "squad_em": 80.8
            }
        )
    
    def _create_gpt2_small(self) -> PopularModel:
        """GPT-2 Small model configuration"""
        return PopularModel(
            id="gpt2_small",
            name="GPT-2 Small",
            full_name="Generative Pre-trained Transformer 2 - Small",
            description="Small GPT-2 model with 117M parameters",
            category="nlp",
            architecture_type="transformer_decoder",
            parameter_count="117M",
            training_dataset="WebText",
            training_details={
                "optimizer": "AdamW",
                "learning_rate": 2.5e-4,
                "batch_size": 512,
                "sequence_length": 1024,
                "training_tokens": "40GB",
                "context_length": 1024
            },
            template_id="transformer_decoder_only",
            config={
                "d_model": 768,
                "num_heads": 12,
                "num_layers": 12,
                "d_ff": 3072,
                "vocab_size": 50257
            },
            typical_applications=["Text Generation", "Completion", "Creative Writing"],
            performance_metrics={
                "perplexity": 35.76,
                "bleu_score": 0.24,
                "rouge_l": 0.19
            }
        )
    
    def _create_t5_base(self) -> PopularModel:
        """T5 Base model configuration"""
        return PopularModel(
            id="t5_base",
            name="T5 Base",
            full_name="Text-to-Text Transfer Transformer - Base",
            description="Base T5 model with 220M parameters",
            category="nlp",
            architecture_type="transformer_encoder_decoder",
            parameter_count="220M",
            training_dataset="C4",
            training_details={
                "optimizer": "Adafactor",
                "learning_rate": 1e-3,
                "batch_size": 128,
                "sequence_length": 512,
                "training_steps": "1M",
                "inverse_sqrt_schedule": True
            },
            template_id="transformer_encoder_decoder",
            config={
                "d_model": 768,
                "num_heads": 12,
                "num_encoder_layers": 12,
                "num_decoder_layers": 12,
                "vocab_size": 32128
            },
            typical_applications=["Summarization", "Translation"],
            performance_metrics={
                "cnndm_rouge1": 42.5,
                "cnndm_rouge2": 20.5,
                "cnndm_rougeL": 39.8,
                "wmt_en_de_bleu": 27.5
            }
        )
    
    def _create_vit_base(self) -> PopularModel:
        """Vision Transformer ViT-B/16 model configuration"""
        return PopularModel(
            id="vit_base",
            name="Vision Transformer (ViT-B/16)",
            full_name="Vision Transformer Base, 16x16 patches",
            description="Base Vision Transformer with 86M parameters, 16x16 patches on 224x224 images",
            category="computer_vision",
            architecture_type="transformer",
            parameter_count="86M",
            training_dataset="ImageNet-21k / ImageNet-1k",
            training_details={
                "optimizer": "AdamW",
                "learning_rate": 1e-3,
                "batch_size": 4096,
                "image_size": 224,
                "patch_size": 16,
                "weight_decay": 0.3,
            },
            template_id="vision_transformer",
            config={
                "d_model": 768,
                "num_heads": 12,
                "num_layers": 12,
                "d_ff": 3072,
                "patch_size": 16,
                "image_size": 224,
                "num_classes": 1000,
                "seq_shape": "[B, 196, 768]",
            },
            typical_applications=["Image Classification", "Transfer Learning"],
            performance_metrics={
                "imagenet_top1": 84.0,
                "imagenet_top5": 97.0,
            },
        )

    # Template Variants
    def _create_bert_classifier_variant(self) -> TemplateVariant:
        """BERT variant optimized for classification"""
        return TemplateVariant(
            id="bert_classifier",
            name="BERT Classifier",
            description="BERT optimized for classification tasks",
            parent_template_id="transformer_encoder_only",
            config_overrides={
                "num_classes": 5,
                "pooling_strategy": "cls_token"
            },
            structural_changes=[
                {"action": "remove", "target": "decoder_stack"},
                {"action": "add", "component": "dropout_layer", "position": "before_classifier"}
            ],
            purpose="classification",
            modifications=["Removed decoder", "Added dropout", "Classifier head"]
        )
    
    def _create_gpt_generator_variant(self) -> TemplateVariant:
        """GPT variant optimized for generation"""
        return TemplateVariant(
            id="gpt_generator",
            name="GPT Generator",
            description="GPT optimized for text generation",
            parent_template_id="transformer_decoder_only",
            config_overrides={
                "max_length": 2048,
                "temperature": 0.8
            },
            structural_changes=[
                {"action": "modify", "target": "attention", "change": "causal_mask"}
            ],
            purpose="generation",
            modifications=["Causal attention", "Extended context"]
        )
