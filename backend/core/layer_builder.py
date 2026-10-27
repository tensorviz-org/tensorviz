"""
Core layer building utilities and PyTorch layer management.
Refactored from server.py for modularity.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from core.op_registry import normalized_parameter
from models.graph_models import Node, Edge


def call_module(module: nn.Module, primary: torch.Tensor) -> torch.Tensor:
    """Run module.forward() with the right number of positional inputs.

    Most layers take a single tensor, but a few transformer/attention modules
    need more positional arguments than the single input the graph hands us:

      - TransformerDecoderLayer / TransformerDecoder: forward(tgt, memory)
      - Transformer:                                  forward(src, tgt)
      - MultiheadAttention:                           forward(query, key, value)

    For those we reuse `primary` for the missing arguments (self-attention /
    decoder-reading-its-own-stack semantics) so a valid graph verifies instead
    of raising "missing required positional argument". Always returns a single
    tensor — tuple outputs like MultiheadAttention's (output, weights) are
    unwrapped to the first element.
    """
    if isinstance(module, (nn.TransformerDecoderLayer, nn.TransformerDecoder)):
        out = module(primary, primary)            # (tgt, memory)
    elif isinstance(module, nn.Transformer):
        out = module(primary, primary)            # (src, tgt)
    elif isinstance(module, nn.MultiheadAttention):
        out = module(primary, primary, primary)   # (query, key, value)
    else:
        out = module(primary)
    if isinstance(out, tuple):
        out = out[0]
    return out


def build_layer(layer_type: str, params: Dict[str, Any]) -> nn.Module:
    """Build a PyTorch layer with specific error handling"""
    try:
        # Strip display-only / non-constructor keys before they reach a torch
        # constructor: `name` (semantic label) and `_`-prefixed UI hints
        # (_title/_subtitle/_shape/_accent) are not valid layer kwargs.
        # Read the typed form of a value at the same time: a kernel the user typed
        # is stored as the string "3" or "3,3", and handing that to nn.Conv2d
        # raised `type must be tuple of ints, but got str`. That was the build
        # error behind export refusing a graph validation had already blessed.
        params = {k: normalized_parameter(v) for k, v in params.items()
                  if k != 'name' and not k.startswith('_')}

        # Handle special layer types that need custom handling
        if layer_type == 'ReLU':
            return nn.ReLU()
        elif layer_type == 'Sigmoid':
            return nn.Sigmoid()
        elif layer_type == 'Tanh':
            return nn.Tanh()
        elif layer_type == 'LeakyReLU':
            negative_slope = params.get('negative_slope', 0.01)
            return nn.LeakyReLU(negative_slope=negative_slope)
        elif layer_type == 'Dropout':
            p = params.get('p', 0.5)
            return nn.Dropout(p=p)
        elif layer_type == 'BatchNorm1d':
            num_features = params.get('num_features')
            if num_features is None:
                raise ValueError("BatchNorm1d requires 'num_features' parameter")
            return nn.BatchNorm1d(num_features=num_features)
        elif layer_type == 'BatchNorm2d':
            num_features = params.get('num_features')
            if num_features is None:
                raise ValueError("BatchNorm2d requires 'num_features' parameter")
            return nn.BatchNorm2d(num_features=num_features)
        elif layer_type == 'Flatten':
            start_dim = params.get('start_dim', 1)
            end_dim = params.get('end_dim', -1)
            return nn.Flatten(start_dim=start_dim, end_dim=end_dim)
        elif layer_type == 'AdaptiveAvgPool2d':
            output_size = params.get('output_size', (1, 1))
            return nn.AdaptiveAvgPool2d(output_size=output_size)
        elif layer_type == 'MaxPool2d':
            kernel_size = params.get('kernel_size', 2)
            stride = params.get('stride', None)
            return nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        elif layer_type == 'AvgPool2d':
            kernel_size = params.get('kernel_size', 2)
            stride = params.get('stride', None)
            return nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
        elif layer_type == 'LayerNorm':
            normalized_shape = params.get('normalized_shape')
            if normalized_shape is None:
                raise ValueError("LayerNorm requires 'normalized_shape' parameter")
            eps = params.get('eps', 1e-5)
            return nn.LayerNorm(normalized_shape=normalized_shape, eps=eps)
        elif layer_type == 'MultiheadAttention':
            embed_dim = params.get('embed_dim')
            num_heads = params.get('num_heads')
            if embed_dim is None or num_heads is None:
                raise ValueError("MultiheadAttention requires 'embed_dim' and 'num_heads' parameters")
            dropout = params.get('dropout', 0.0)
            return nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        elif layer_type == 'TransformerEncoderLayer':
            d_model = params.get('d_model')
            nhead = params.get('nhead')
            if d_model is None or nhead is None:
                raise ValueError("TransformerEncoderLayer requires 'd_model' and 'nhead' parameters")
            dim_feedforward = params.get('dim_feedforward', 2048)
            dropout = params.get('dropout', 0.1)
            return nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout
            )
        elif layer_type == 'TransformerDecoderLayer':
            d_model = params.get('d_model')
            nhead = params.get('nhead')
            if d_model is None or nhead is None:
                raise ValueError("TransformerDecoderLayer requires 'd_model' and 'nhead' parameters")
            dim_feedforward = params.get('dim_feedforward', 2048)
            dropout = params.get('dropout', 0.1)
            return nn.TransformerDecoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout
            )
        elif layer_type == 'Embedding':
            num_embeddings = params.get('num_embeddings')
            embedding_dim = params.get('embedding_dim')
            if num_embeddings is None or embedding_dim is None:
                raise ValueError("Embedding requires 'num_embeddings' and 'embedding_dim' parameters")
            return nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        else:
            # Try to get the layer class from torch.nn
            if not hasattr(nn, layer_type):
                raise AttributeError(f"Layer type '{layer_type}' not found in torch.nn")
            
            layer_cls = getattr(nn, layer_type)
            return layer_cls(**params)
            
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid parameters for {layer_type}: {e}")
    except AttributeError as e:
        raise ValueError(f"Unknown layer type '{layer_type}': {e}")
    except Exception as e:
        raise ValueError(f"Error initializing layer {layer_type}: {e}")


def topological_sort(nodes: List[Node], edges: List[Edge]) -> List[str]:
    """Topologically sort nodes based on edge connections"""
    from collections import defaultdict, deque
    
    graph = defaultdict(list)
    indegree = defaultdict(int)
    
    # Initialize all nodes with 0 indegree
    for node in nodes:
        indegree[node.id] = 0
    
    # Build graph and calculate indegrees
    for edge in edges:
        graph[edge.source].append(edge.target)
        indegree[edge.target] += 1
    
    # Find nodes with no incoming edges
    queue = deque([node.id for node in nodes if indegree[node.id] == 0])
    ordered = []
    
    while queue:
        node = queue.popleft()
        ordered.append(node)
        
        # Reduce indegree for all neighbors
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
    
    if len(ordered) != len(nodes):
        raise ValueError("Cycle detected or graph is not connected")
    
    return ordered


def validate_layer_parameters(layer_type: str, params: Dict[str, Any]) -> List[str]:
    """Validate layer parameters and return list of validation errors"""
    errors = []
    
    try:
        # Try building the layer to catch parameter errors
        build_layer(layer_type, params)
    except ValueError as e:
        errors.append(str(e))
    except Exception as e:
        errors.append(f"Unexpected error validating {layer_type}: {e}")
    
    return errors


def get_layer_output_shape(layer_type: str, params: Dict[str, Any], input_shape: List[int]) -> List[int]:
    """Get expected output shape for a layer given input shape"""
    try:
        # Create dummy input tensor
        input_tensor = torch.randn(*input_shape)
        
        # Build and execute layer
        layer = build_layer(layer_type, params)
        with torch.no_grad():
            output_tensor = call_module(layer, input_tensor)
        
        return list(output_tensor.shape)
    except Exception:
        # If we can't determine shape, return input shape as fallback
        return input_shape
