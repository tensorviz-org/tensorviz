"""Data models for model templates and popular architectures"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from .graph_models import Graph, Node, Edge


class ModelTemplate(BaseModel):
    """Template for popular neural network architectures"""
    id: str
    name: str
    description: str
    category: str  # 'computer_vision', 'nlp', 'generative', 'custom'
    sub_category: str  # 'cnn', 'transformer', 'gan', etc.
    difficulty: str  # 'beginner', 'intermediate', 'advanced'
    tags: List[str]
    
    # Architecture configuration
    base_config: Dict[str, Any]  # Default parameters
    customizable_params: List[Dict[str, Any]]  # Parameters user can modify
    
    # Template structure
    template_structure: Dict[str, Any]  # Full hierarchical template
    
    # Variants - different configurations of the same model
    variants: List[Dict[str, Any]] = []
    
    # Usage information
    typical_use_cases: List[str] = []
    paper_reference: Optional[str] = None
    
    # Requirements
    minimum_input_shape: List[int]
    recommended_input_shape: List[int]


class ModuleTemplate(BaseModel):
    """Template for individual modules/components"""
    id: str
    name: str
    description: str
    category: str
    
    # Internal structure
    internal_graph: Graph
    
    # Configuration
    configurable_params: Dict[str, Any]
    default_params: Dict[str, Any]
    
    # Connections
    input_specs: Dict[str, Any]  # Expected input format
    output_specs: Dict[str, Any]  # Output format
    
    # Code generation
    pytorch_class_template: str


class TemplateVariant(BaseModel):
    """Specific variant of a model template"""
    id: str
    name: str
    description: str
    parent_template_id: str
    
    # Configuration overrides
    config_overrides: Dict[str, Any]
    structural_changes: List[Dict[str, Any]]  # Add/remove/modify modules
    
    # Metadata
    purpose: str  # 'classification', 'generation', 'feature_extraction', etc.
    modifications: List[str]  # Human-readable list of changes


class PopularModel(BaseModel):
    """Popular pre-trained model configuration"""
    id: str
    name: str
    full_name: str
    description: str
    category: str
    
    # Model details
    architecture_type: str  # 'transformer', 'cnn', 'rnn', etc.
    parameter_count: Optional[str] = None  # '110M', '1.5B', etc.
    
    # Training details
    training_dataset: Optional[str] = None
    training_details: Optional[Dict[str, Any]] = None
    
    # Template reference
    template_id: str  # References ModelTemplate
    config: Dict[str, Any]  # Specific configuration for this model
    
    # Usage
    typical_applications: List[str]
    performance_metrics: Optional[Dict[str, Any]] = None
