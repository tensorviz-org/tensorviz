"""Natural Language Processing model templates beyond transformers"""
from typing import Dict, List, Any, Optional
from models.template_models import ModelTemplate, ModuleTemplate, PopularModel
from models.graph_models import Graph, Node, Edge, LayerType, LayerData


class NLPTemplates:
    """Service for Non-transformer NLP architectures"""
    
    def get_all_templates(self) -> Dict[str, ModelTemplate]:
        """Get all NLP templates"""
        return {
            "lstm_classifier": self._create_lstm_classifier(),
            "gru_seq2seq": self._create_gru_seq2seq(),
            "cnn_text_classifier": self._create_cnn_text_classifier(),
            "rnn_language_model": self._create_rnn_language_model(),
        }
    
    def get_module_templates(self) -> Dict[str, ModuleTemplate]:
        """Get NLP module templates"""
        return {
            "lstm_cell": self._create_lstm_cell(),
            "gru_cell": self._create_gru_cell(),
            "attention_mechanism": self._create_attention_mechanism(),
            "text_cnn_block": self._create_text_cnn_block(),
        }
    
    def get_popular_models(self) -> Dict[str, PopularModel]:
        """Get popular non-transformer NLP models"""
        return {
            "elmo": self._create_elmo(),
            "word2vec": self._create_word2vec(),
            "fasttext": self._create_fasttext(),
        }
    
    def _create_lstm_classifier(self) -> ModelTemplate:
        """Create LSTM-based text classifier"""
        return ModelTemplate(
            id="lstm_classifier",
            name="LSTM Text Classifier",
            description="LSTM-based neural network for text classification tasks",
            category="nlp",
            sub_category="rnn",
            difficulty="intermediate",
            tags=["lstm", "classification", "rnn", "text"],
            
            base_config={
                "vocab_size": 10000,
                "embedding_dim": 300,
                "hidden_size": 256,
                "num_layers": 2,
                "num_classes": 2,
                "dropout": 0.3,
                "bidirectional": True,
                "pooling": "max"  # max, mean, last
            },
            
            customizable_params=[
                {
                    "name": "hidden_size",
                    "type": "number",
                    "min": 64,
                    "max": 1024,
                    "description": "LSTM hidden dimension"
                },
                {
                    "name": "num_layers",
                    "type": "number",
                    "min": 1,
                    "max": 8,
                    "description": "Number of LSTM layers"
                },
                {
                    "name": "bidirectional",
                    "type": "boolean",
                    "description": "Use bidirectional LSTM"
                },
                {
                    "name": "pooling",
                    "type": "select",
                    "options": ["max", "mean", "last", "attention"],
                    "description": "Sequence pooling method"
                }
            ],
            
            template_structure={
                "nodes": [
                    {
                        "id": "embedding",
                        "type": "layerNode",
                        "position": {"x": 100, "y": 100},
                        "data": {
                            "layerType": {"id": "embedding", "name": "Embedding", "parameters": []},
                            "parameters": {
                                "num_embeddings": "{vocab_size}",
                                "embedding_dim": "{embedding_dim}",
                                "padding_idx": 0
                            }
                        }
                    },
                    {
                        "id": "dropout_emb",
                        "type": "layerNode",
                        "position": {"x": 300, "y": 100},
                        "data": {
                            "layerType": {"id": "dropout", "name": "Dropout", "parameters": []},
                            "parameters": {"p": "{dropout}"}
                        }
                    },
                    {
                        "id": "lstm",
                        "type": "layerNode",
                        "position": {"x": 500, "y": 100},
                        "data": {
                            "layerType": {"id": "lstm", "name": "LSTM", "parameters": []},
                            "parameters": {
                                "input_size": "{embedding_dim}",
                                "hidden_size": "{hidden_size}",
                                "num_layers": "{num_layers}",
                                "dropout": "{dropout}",
                                "bidirectional": "{bidirectional}",
                                "batch_first": True
                            }
                        }
                    },
                    {
                        "id": "pooling",
                        "type": "moduleNode",
                        "position": {"x": 700, "y": 100},
                        "data": {
                            "layerType": {"id": "sequence_pooling", "name": "SequencePooling", "parameters": []},
                            "parameters": {
                                "pooling_type": "{pooling}",
                                "hidden_size": "{hidden_size}",
                                "bidirectional": "{bidirectional}",
                                "seq_len": 10
                            }
                        }
                    },
                    {
                        "id": "dropout_final",
                        "type": "layerNode",
                        "position": {"x": 900, "y": 100},
                        "data": {
                            "layerType": {"id": "dropout", "name": "Dropout", "parameters": []},
                            "parameters": {"p": "{dropout}"}
                        }
                    },
                    {
                        "id": "classifier",
                        "type": "layerNode",
                        "position": {"x": 1100, "y": 100},
                        "data": {
                            "layerType": {"id": "linear", "name": "Linear", "parameters": []},
                            "parameters": {
                                "in_features": "{hidden_size}",  # Will be adjusted for bidirectional
                                "out_features": "{num_classes}"
                            }
                        }
                    }
                ],
                "edges": [
                    {"id": "e1", "source": "embedding", "target": "dropout_emb"},
                    {"id": "e2", "source": "dropout_emb", "target": "lstm"},
                    {"id": "e3", "source": "lstm", "target": "pooling"},
                    {"id": "e4", "source": "pooling", "target": "dropout_final"},
                    {"id": "e5", "source": "dropout_final", "target": "classifier"}
                ]
            },
            
            variants=[
                {
                    "id": "simple_lstm",
                    "name": "Simple LSTM",
                    "description": "Single layer unidirectional LSTM",
                    "config_overrides": {
                        "num_layers": 1,
                        "bidirectional": False,
                        "pooling": "last"
                    }
                },
                {
                    "id": "deep_bilstm",
                    "name": "Deep BiLSTM",
                    "description": "Deep bidirectional LSTM with attention pooling",
                    "config_overrides": {
                        "num_layers": 4,
                        "bidirectional": True,
                        "pooling": "attention"
                    }
                }
            ],
            
            typical_use_cases=[
                "Sentiment Analysis",
                "Text Classification",
                "Intent Recognition",
                "Spam Detection"
            ],
            
            minimum_input_shape=[1, 10],
            recommended_input_shape=[32, 128]
        )
    
    def _create_gru_seq2seq(self) -> ModelTemplate:
        """Create GRU-based sequence-to-sequence model"""
        return ModelTemplate(
            id="gru_seq2seq",
            name="GRU Sequence-to-Sequence",
            description="GRU-based encoder-decoder for sequence-to-sequence tasks",
            category="nlp",
            sub_category="seq2seq",
            difficulty="advanced",
            tags=["gru", "seq2seq", "encoder-decoder", "attention"],
            
            base_config={
                "src_vocab_size": 10000,
                "tgt_vocab_size": 10000,
                "embedding_dim": 256,
                "hidden_size": 512,
                "num_layers": 2,
                "dropout": 0.3,
                "attention": True,
                "attention_type": "dot"  # dot, general, concat
            },
            
            customizable_params=[
                {
                    "name": "attention",
                    "type": "boolean",
                    "description": "Use attention mechanism"
                },
                {
                    "name": "attention_type",
                    "type": "select",
                    "options": ["dot", "general", "concat"],
                    "description": "Type of attention mechanism"
                }
            ],
            
            template_structure={
                "nodes": [
                    {
                        "id": "encoder_embedding",
                        "type": "layerNode",
                        "position": {"x": 100, "y": 100},
                        "data": {
                            "layerType": {"id": "embedding", "name": "Embedding", "parameters": []},
                            "parameters": {
                                "num_embeddings": "{src_vocab_size}",
                                "embedding_dim": "{embedding_dim}"
                            }
                        }
                    },
                    {
                        "id": "encoder_gru",
                        "type": "layerNode",
                        "position": {"x": 300, "y": 100},
                        "data": {
                            "layerType": {"id": "gru", "name": "GRU", "parameters": []},
                            "parameters": {
                                "input_size": "{embedding_dim}",
                                "hidden_size": "{hidden_size}",
                                "num_layers": "{num_layers}",
                                "dropout": "{dropout}",
                                # Unidirectional so the runtime hidden width stays
                                # hidden_size (a bidirectional GRU would emit 2*hidden
                                # at runtime, mismatching the attention/decoder dims).
                                "bidirectional": False,
                                "batch_first": True
                            }
                        }
                    },
                    {
                        "id": "attention_layer",
                        "type": "moduleNode",
                        "position": {"x": 500, "y": 100},
                        "data": {
                            "layerType": {"id": "attention", "name": "Attention", "parameters": []},
                            "parameters": {
                                "hidden_size": "{hidden_size}",
                                "attention_type": "{attention_type}",
                                "enabled": "{attention}"
                            }
                        }
                    },
                    {
                        "id": "decoder_gru",
                        "type": "layerNode",
                        "position": {"x": 700, "y": 100},
                        "data": {
                            "layerType": {"id": "gru", "name": "GRU", "parameters": []},
                            "parameters": {
                                # Receives the encoder/attention context [N, L, hidden],
                                # so its input_size is hidden_size (not embedding_dim).
                                "input_size": "{hidden_size}",
                                "hidden_size": "{hidden_size}",
                                "num_layers": "{num_layers}",
                                "dropout": "{dropout}",
                                "batch_first": True
                            }
                        }
                    },
                    {
                        "id": "output_projection",
                        "type": "layerNode",
                        "position": {"x": 900, "y": 100},
                        "data": {
                            "layerType": {"id": "linear", "name": "Linear", "parameters": []},
                            "parameters": {
                                "in_features": "{hidden_size}",
                                "out_features": "{tgt_vocab_size}"
                            }
                        }
                    }
                ],
                "edges": [
                    {"id": "e1", "source": "encoder_embedding", "target": "encoder_gru"},
                    {"id": "e3", "source": "encoder_gru", "target": "attention_layer"},
                    {"id": "e4", "source": "attention_layer", "target": "decoder_gru"},
                    {"id": "e5", "source": "decoder_gru", "target": "output_projection"}
                ]
            },
            
            typical_use_cases=[
                "Machine Translation",
                "Text Summarization",
                "Dialogue Systems",
                "Code Generation"
            ],
            
            minimum_input_shape=[1, 10],
            recommended_input_shape=[32, 64]
        )
    
    def _create_cnn_text_classifier(self) -> ModelTemplate:
        """Create CNN-based text classifier"""
        return ModelTemplate(
            id="cnn_text_classifier",
            name="CNN Text Classifier",
            description="Convolutional neural network for text classification",
            category="nlp",
            sub_category="cnn",
            difficulty="beginner",
            tags=["cnn", "text", "classification", "n-gram"],
            
            base_config={
                "vocab_size": 10000,
                "embedding_dim": 300,
                "num_filters": 100,
                "filter_sizes": [3, 4, 5],
                "num_classes": 2,
                "dropout": 0.5,
                "max_length": 256
            },
            
            customizable_params=[
                {
                    "name": "filter_sizes",
                    "type": "array",
                    "description": "Convolution filter sizes (n-gram sizes)"
                },
                {
                    "name": "num_filters",
                    "type": "number",
                    "min": 50,
                    "max": 500,
                    "description": "Number of filters per size"
                }
            ],
            
            template_structure={
                "nodes": [
                    {
                        "id": "embedding",
                        "type": "layerNode",
                        "position": {"x": 100, "y": 100},
                        "data": {
                            "layerType": {"id": "embedding", "name": "Embedding", "parameters": []},
                            "parameters": {
                                "num_embeddings": "{vocab_size}",
                                "embedding_dim": "{embedding_dim}"
                            }
                        }
                    },
                    {
                        "id": "text_cnn",
                        "type": "moduleNode",
                        "position": {"x": 400, "y": 200},
                        "data": {
                            "layerType": {"id": "text_cnn", "name": "TextCNN", "parameters": []},
                            "parameters": {
                                "embedding_dim": "{embedding_dim}",
                                "num_filters": "{num_filters}",
                                "filter_sizes": "{filter_sizes}",
                                "seq_len": 10
                            }
                        }
                    },
                    {
                        "id": "dropout",
                        "type": "layerNode",
                        "position": {"x": 600, "y": 200},
                        "data": {
                            "layerType": {"id": "dropout", "name": "Dropout", "parameters": []},
                            "parameters": {"p": "{dropout}"}
                        }
                    },
                    {
                        "id": "classifier",
                        "type": "layerNode",
                        "position": {"x": 800, "y": 200},
                        "data": {
                            "layerType": {"id": "linear", "name": "Linear", "parameters": []},
                            "parameters": {
                                "in_features": "{num_filters}",  # TextCNN module pools to num_filters
                                "out_features": "{num_classes}"
                            }
                        }
                    }
                ],
                "edges": [
                    {"id": "e1", "source": "embedding", "target": "text_cnn"},
                    {"id": "e2", "source": "text_cnn", "target": "dropout"},
                    {"id": "e3", "source": "dropout", "target": "classifier"}
                ]
            },
            
            typical_use_cases=[
                "Sentiment Analysis",
                "Topic Classification",
                "Spam Detection",
                "Short Text Classification"
            ],
            
            minimum_input_shape=[1, 10],
            recommended_input_shape=[32, 256]
        )
    
    def _create_rnn_language_model(self) -> ModelTemplate:
        """Create RNN-based language model"""
        return ModelTemplate(
            id="rnn_language_model",
            name="RNN Language Model",
            description="Recurrent neural network for language modeling",
            category="nlp",
            sub_category="language_model",
            difficulty="intermediate",
            tags=["rnn", "language-model", "generation"],
            
            base_config={
                "vocab_size": 10000,
                "embedding_dim": 200,
                "hidden_size": 200,
                "num_layers": 2,
                "dropout": 0.2,
                "tie_weights": True,
                "rnn_type": "LSTM"  # LSTM, GRU, RNN
            },
            
            customizable_params=[
                {
                    "name": "rnn_type",
                    "type": "select",
                    "options": ["LSTM", "GRU", "RNN"],
                    "description": "Type of RNN cell"
                },
                {
                    "name": "tie_weights",
                    "type": "boolean",
                    "description": "Tie input and output embeddings"
                }
            ],
            
            template_structure={
                "nodes": [
                    {
                        "id": "embedding",
                        "type": "layerNode",
                        "position": {"x": 100, "y": 100},
                        "data": {
                            "layerType": {"id": "embedding", "name": "Embedding", "parameters": []},
                            "parameters": {
                                "num_embeddings": "{vocab_size}",
                                "embedding_dim": "{embedding_dim}"
                            }
                        }
                    },
                    {
                        "id": "dropout_input",
                        "type": "layerNode",
                        "position": {"x": 300, "y": 100},
                        "data": {
                            "layerType": {"id": "dropout", "name": "Dropout", "parameters": []},
                            "parameters": {"p": "{dropout}"}
                        }
                    },
                    {
                        "id": "rnn",
                        "type": "layerNode",
                        "position": {"x": 500, "y": 100},
                        "data": {
                            "layerType": {"id": "lstm", "name": "{rnn_type}", "parameters": []},
                            "parameters": {
                                "input_size": "{embedding_dim}",
                                "hidden_size": "{hidden_size}",
                                "num_layers": "{num_layers}",
                                "dropout": "{dropout}",
                                "batch_first": True
                            }
                        }
                    },
                    {
                        "id": "dropout_output",
                        "type": "layerNode",
                        "position": {"x": 700, "y": 100},
                        "data": {
                            "layerType": {"id": "dropout", "name": "Dropout", "parameters": []},
                            "parameters": {"p": "{dropout}"}
                        }
                    },
                    {
                        "id": "decoder",
                        "type": "layerNode",
                        "position": {"x": 900, "y": 100},
                        "data": {
                            "layerType": {"id": "linear", "name": "Linear", "parameters": []},
                            "parameters": {
                                "in_features": "{hidden_size}",
                                "out_features": "{vocab_size}"
                            }
                        }
                    }
                ],
                "edges": [
                    {"id": "e1", "source": "embedding", "target": "dropout_input"},
                    {"id": "e2", "source": "dropout_input", "target": "rnn"},
                    {"id": "e3", "source": "rnn", "target": "dropout_output"},
                    {"id": "e4", "source": "dropout_output", "target": "decoder"}
                ]
            },
            
            typical_use_cases=[
                "Text Generation",
                "Language Modeling",
                "Autocomplete",
                "Poetry Generation"
            ],
            
            minimum_input_shape=[1, 10],
            recommended_input_shape=[32, 35]
        )
    
    # Module template implementations (simplified)
    def _create_lstm_cell(self) -> ModuleTemplate:
        pass
    
    def _create_gru_cell(self) -> ModuleTemplate:
        pass
    
    def _create_attention_mechanism(self) -> ModuleTemplate:
        pass
    
    def _create_text_cnn_block(self) -> ModuleTemplate:
        pass
    
    # Popular model implementations
    def _create_elmo(self) -> PopularModel:
        """Create ELMo model configuration"""
        return PopularModel(
            id="elmo",
            name="ELMo",
            full_name="Embeddings from Language Models",
            description="Deep contextualized word representations using bidirectional LSTM",
            category="nlp",
            architecture_type="bilstm",
            parameter_count="94M",
            training_dataset="1 Billion Word Benchmark",
            template_id="lstm_classifier",
            config={
                "hidden_size": 512,
                "num_layers": 2,
                "bidirectional": True,
                "embedding_dim": 512
            },
            typical_applications=[
                "Contextual Word Embeddings",
                "Transfer Learning",
                "NLP Feature Extraction",
                "Text Understanding"
            ]
        )
    
    def _create_word2vec(self) -> PopularModel:
        """Create Word2Vec model configuration"""
        return PopularModel(
            id="word2vec",
            name="Word2Vec",
            full_name="Word to Vector Embeddings",
            description="Neural word embeddings using skip-gram or CBOW",
            category="nlp",
            architecture_type="embedding",
            parameter_count="Variable",
            training_dataset="Google News",
            template_id="rnn_language_model",  # Simplified mapping
            config={
                "embedding_dim": 300,
                "vocab_size": 3000000
            },
            typical_applications=[
                "Word Embeddings",
                "Semantic Similarity",
                "Word Analogies",
                "Feature Extraction"
            ]
        )
    
    def _create_fasttext(self) -> PopularModel:
        """Create FastText model configuration"""
        return PopularModel(
            id="fasttext",
            name="FastText",
            full_name="FastText Word Embeddings",
            description="Word embeddings with subword information",
            category="nlp",
            architecture_type="embedding",
            parameter_count="Variable",
            training_dataset="Common Crawl",
            template_id="cnn_text_classifier",  # Simplified mapping
            config={
                "embedding_dim": 300,
                "vocab_size": 2000000
            },
            typical_applications=[
                "Word Embeddings",
                "Text Classification",
                "Language Identification",
                "Out-of-vocabulary Words"
            ]
        )
