"""Core utilities for shape inference and model building"""
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
from models.graph_models import Graph, Node, Edge, HierarchicalGraph, ModuleDefinition
from core.layer_builder import call_module
from core.op_registry import normalized_parameter
from core.source_ops import SOURCE_OPS as _SOURCE_OPS


#: Ops that PRODUCE a tensor without consuming one — a second root, not a step in the
#: chain. They have no input to check, no path from the graph's input, and no frame for
#: the tracer to enter, so the shape engine skips them rather than reporting all three.


def _bare_op(name: object) -> str:
    """`torch.nn.Parameter` -> `parameter`."""
    return str(name or "").rsplit(".", 1)[-1].lower()


def parse_input_shape(raw: Any) -> Optional[List[int]]:
    """Parse an Input node's shape from either representation the canvas produces.

    The canvas stores this param as the **string** the user typed (``"1,16"``), and
    that is what templates write; the code-import path emits a ``list[int]``
    instead. Validators that accepted only the list form silently failed on every
    real canvas graph — see ``services/canonical_validation._validate_input_shapes``
    and ``ai/domain/candidate_validation._primary_input_shape``, both of which then
    fell back to (or suggested) the ``[1, 3, 224, 224]`` image default.

    Returns None for anything unparseable, so callers keep their own fallback
    rather than being handed a plausible-looking wrong shape.
    """
    if isinstance(raw, (list, tuple)):
        dims = list(raw)
    elif isinstance(raw, str):
        text = raw.strip().strip("[]()")
        if not text:
            return None
        try:
            dims = [int(part.strip()) for part in text.split(",") if part.strip()]
        except ValueError:
            return None
    else:
        return None
    if dims and all(isinstance(dim, int) and not isinstance(dim, bool) and dim > 0 for dim in dims):
        return dims
    return None


class TorchFunctionalLayer(nn.Module):
    """Wrapper wrapper for torch functional operations (like torch.add, torch.cat)"""
    def __init__(self, op_name: str, params: Dict[str, Any]):
        super().__init__()
        self.op_name = op_name
        self.params = params
        
    def forward(self, *inputs):
        if not inputs:
            raise ValueError(f"{self.op_name} requires at least one input tensor")
            
        op = self.op_name.replace("torch.", "")
        
        # Parse params
        kwargs = {}
        if "dim" in self.params and self.params["dim"] != "":
            try:
                dim_str = str(self.params["dim"]).strip()
                if "," in dim_str:
                    kwargs["dim"] = tuple(map(int, dim_str.split(",")))
                else:
                    kwargs["dim"] = int(dim_str)
            except ValueError:
                pass
                
        if op in ["cat", "stack"]:
            func = getattr(torch, op)
            return func(inputs, **kwargs)
        elif op in ["add", "sub", "mul", "div", "matmul", "bmm"]:
            func = getattr(torch, op)
            res = inputs[0]
            for t in inputs[1:]:
                res = func(res, t)
            return res
        else:
            func = getattr(torch, op)
            # most other ops only take 1 input but we pass kwargs just in case
            if "dim" in kwargs and op not in ["squeeze", "unsqueeze", "sum", "mean", "max", "min", "argmax", "argmin", "chunk", "split", "transpose", "flatten"]:
                # some ops don't officially take dim keyword, just skip it if it's there
                pass 
            return func(inputs[0], **kwargs)

# Combiners that take several inputs and have no torch.nn class. Kept in step with
# services/verification_service._MERGE_OP_KEYS, which is what the dry forward uses —
# a node accepted there and rejected here reports a phantom analytic error.
_ANALYTIC_MERGE_OPS = {
    "add", "torch.add", "sum", "torch.sum",
    "mul", "torch.mul", "multiply",
    "cat", "concat", "concatenate", "torch.cat", "torch.concat",
    "stack", "torch.stack",
}


def _apply_analytic_merge(
    op_key: str, inputs: List[Any], params: Dict[str, Any]
) -> Any:
    """Combine several input tensors the same way the dry forward does."""
    if not inputs:
        raise ValueError("merge op received no inputs")
    from core.scalar_operations import apply_scalar
    scalar_result = apply_scalar(op_key, inputs, params)
    if scalar_result is not None:
        return scalar_result
    if op_key in {"cat", "concat", "concatenate", "torch.cat", "torch.concat"}:
        try:
            dim = int(params.get("dim", 1))
        except (TypeError, ValueError):
            dim = 1
        return torch.cat(inputs, dim=dim) if len(inputs) > 1 else inputs[0]
    if op_key in {"stack", "torch.stack"}:
        try:
            dim = int(params.get("dim", 0))
        except (TypeError, ValueError):
            dim = 0
        return torch.stack(inputs, dim=dim) if len(inputs) > 1 else inputs[0]
    if op_key in {"sum", "torch.sum"} and len(inputs) == 1:
        # A one-input `sum` is a REDUCTION, not a merge. It is in this set because
        # a residual `sum` of two or more tensors has no torch.nn class and has to
        # be applied directly — but that route returned inputs[0] unchanged, which
        # is exactly the passthrough that used to be reported as fixed. It was fixed
        # in op_registry's _shape_reduce, and this plane shadows the registry: the
        # /api/models/infer-shapes endpoint reaches the merge branch first, so it
        # still answered [1,3,224,224] for sum(dim=1) of a [1,3,224,224] input
        # where torch answers [1,224,224]. Reduce the same way torch does, and
        # leave the multi-input residual path below untouched.
        dim = params.get("dim", None)
        keepdim = bool(params.get("keepdim", False))
        if dim is None:
            return torch.sum(inputs[0])
        try:
            dim = int(dim)
        except (TypeError, ValueError):
            return inputs[0]
        return torch.sum(inputs[0], dim=dim, keepdim=keepdim)

    out = inputs[0]
    if op_key in {"mul", "torch.mul", "multiply"}:
        for tensor in inputs[1:]:
            out = out * tensor
        return out
    for tensor in inputs[1:]:
        out = out + tensor
    return out


def build_layer(layer_type: str, params: Dict[str, Any]) -> nn.Module:
    """Build a PyTorch layer with specific error handling"""
    try:
        # Strip display-only / non-constructor keys (name + _-prefixed UI hints)
        # so they never reach a torch constructor.
        params = {k: v for k, v in params.items() if k != 'name' and not k.startswith('_')}

        if layer_type.startswith('torch.'):
            # Stored form, untouched: TorchFunctionalLayer reads a comma-separated
            # `dim` itself, and it reads it off the string.
            return TorchFunctionalLayer(layer_type, params)

        # Below here a value reaches a real nn constructor, so read a typed one the
        # same way core.layer_builder does — a kernel of "3,3" is a tuple, not a
        # str. This plane has its own builder, so leaving it out meant
        # /api/models/infer-shapes still reported "type must be tuple of ints, but
        # got str" for a graph that by then validated, exported and ran.
        params = {k: normalized_parameter(v) for k, v in params.items()}

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
    """Perform topological sort on the graph nodes"""
    graph = defaultdict(list)
    indegree = defaultdict(int)
    
    # Initialize indegree for all nodes
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
        
        # Reduce indegree for neighbors
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
    
    if len(ordered) != len(nodes):
        raise ValueError("Cycle detected or graph is not connected")
    
    return ordered


def calculate_model_summary(graph: Dict[str, Any], input_shape: List[int]) -> Dict[str, Any]:
    """Calculate model summary statistics"""
    nodes = graph.get("nodes", [])
    
    # Count parameters (simplified)
    total_params = 0
    trainable_params = 0
    
    for node in nodes:
        layer_type = node.get("data", {}).get("layerType", {}).get("name", "")
        params = node.get("data", {}).get("parameters", {})
        
        # Rough parameter estimation
        if layer_type == "Linear":
            in_features = params.get("in_features", 0)
            out_features = params.get("out_features", 0)
            has_bias = params.get("bias", True)
            layer_params = in_features * out_features + (out_features if has_bias else 0)
            total_params += layer_params
            trainable_params += layer_params
        elif layer_type == "Conv2d":
            in_channels = params.get("in_channels", 0)
            out_channels = params.get("out_channels", 0)
            kernel_size = params.get("kernel_size", 3)
            if isinstance(kernel_size, int):
                kernel_size = [kernel_size, kernel_size]
            layer_params = in_channels * out_channels * kernel_size[0] * kernel_size[1]
            total_params += layer_params
            trainable_params += layer_params
    
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": round(model_size_mb, 2)
    }


def validate_graph(graph: Dict[str, Any], input_shape: List[int]) -> List[Dict[str, str]]:
    """Validate graph for common errors"""
    errors = []
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    
    # Check if there's at least one input node
    input_nodes = [n for n in nodes if n.get("data", {}).get("layerType", {}).get("name") == "Input"]
    if not input_nodes:
        errors.append({
            "nodeId": "general",
            "message": "Model should have at least one Input node"
        })
    
    # Check for disconnected nodes
    connected_nodes = set()
    for edge in edges:
        connected_nodes.add(edge.get("source"))
        connected_nodes.add(edge.get("target"))
    
    for node in nodes:
        node_id = node.get("id")
        if node_id not in connected_nodes and len(nodes) > 1:
            layer_name = node.get("data", {}).get("layerType", {}).get("displayName", "Unknown")
            errors.append({
                "nodeId": node_id,
                "message": f"{layer_name} node is not connected to the graph"
            })
    
    return errors


class HierarchicalShapeInferenceContext:
    """Context for hierarchical shape inference"""
    
    def __init__(self, hierarchical_graph: HierarchicalGraph):
        self.graph = hierarchical_graph
        self.tensor_cache: Dict[str, torch.Tensor] = {}
        self.shape_cache: Dict[str, List[int]] = {}
        self.module_cache: Dict[str, nn.Module] = {}
        self.errors: List[Dict[str, Any]] = []
        
    def build_module_recursively(self, module_def: ModuleDefinition) -> nn.Module:
        """Build PyTorch module with internal structure"""
        
        class DynamicModule(nn.Module):
            def __init__(self, definition: ModuleDefinition, context: 'HierarchicalShapeInferenceContext'):
                super().__init__()
                self.definition = definition
                self.context = context
                self.internal_layers = nn.ModuleDict()
                
                # Build internal layers
                for node in definition.internal_nodes:
                    layer_type = node.data.layerType.name
                    layer_params = node.data.parameters
                    
                    if layer_type not in ['Input', 'Output', 'Module']:
                        try:
                            layer = build_layer(layer_type, layer_params)
                            self.internal_layers[node.id] = layer
                        except Exception as e:
                            context.errors.append({
                                "nodeId": f"{definition.module_id}.{node.id}",
                                "message": f"Error building layer {layer_type}: {e}",
                                "error_type": "layer_build_error"
                            })
            
            def forward(self, x):
                # Execute internal graph
                tensor_map = {self.definition.input_node_id or "input": x}
                
                try:
                    execution_order = topological_sort(
                        self.definition.internal_nodes, 
                        self.definition.internal_edges
                    )
                    
                    for node_id in execution_order:
                        node = next(n for n in self.definition.internal_nodes if n.id == node_id)
                        layer_type = node.data.layerType.name
                        
                        if layer_type == 'Input':
                            continue
                        elif layer_type == 'Output':
                            # Get input from previous layer
                            input_edges = [e for e in self.definition.internal_edges if e.target == node_id]
                            if input_edges and input_edges[0].source in tensor_map:
                                tensor_map[node_id] = tensor_map[input_edges[0].source]
                            continue
                        elif layer_type == 'Module':
                            # Handle nested modules (simplified for now)
                            input_edges = [e for e in self.definition.internal_edges if e.target == node_id]
                            if input_edges and input_edges[0].source in tensor_map:
                                tensor_map[node_id] = tensor_map[input_edges[0].source]
                            continue
                        
                        # Get input for this layer
                        input_edges = [e for e in self.definition.internal_edges if e.target == node_id]
                        if not input_edges:
                            continue
                            
                        source_id = input_edges[0].source
                        if source_id not in tensor_map:
                            continue
                            
                        # Apply layer
                        if node_id in self.internal_layers:
                            layer_input = tensor_map[source_id]
                            layer_output = call_module(self.internal_layers[node_id], layer_input)
                            tensor_map[node_id] = layer_output
                
                except Exception as e:
                    self.context.errors.append({
                        "nodeId": self.definition.module_id,
                        "message": f"Error executing module: {e}",
                        "error_type": "module_execution_error"
                    })
                    return x  # Fallback: pass through
                
                # Return output from output node or last node
                output_nodes = [n for n in self.definition.internal_nodes 
                              if n.data.layerType.name == 'Output']
                if output_nodes and output_nodes[0].id in tensor_map:
                    return tensor_map[output_nodes[0].id]
                elif tensor_map:
                    # Return last computed tensor
                    return list(tensor_map.values())[-1]
                else:
                    return x  # Fallback
        
        return DynamicModule(module_def, self)


def infer_hierarchical_shapes(hierarchical_graph: HierarchicalGraph, input_shape: List[int]):
    """Infer shapes for hierarchical graph with module support"""
    context = HierarchicalShapeInferenceContext(hierarchical_graph)
    shape_map = {}
    
    # Process root level first
    root_shape_result = infer_node_shapes_with_context(
        Graph(nodes=hierarchical_graph.root_nodes, edges=hierarchical_graph.root_edges),
        input_shape,
        context
    )
    
    shape_map.update(root_shape_result.get("shapes", {}))
    context.errors.extend(root_shape_result.get("errors", []))
    
    # Process each module's internal structure
    for module_id, module_def in hierarchical_graph.modules.items():
        if module_def.internal_nodes:
            # Use shape from module's input
            module_input_shape = input_shape  # Simplified - should track actual input shape
            
            module_shape_result = infer_node_shapes_with_context(
                Graph(nodes=module_def.internal_nodes, edges=module_def.internal_edges),
                module_input_shape,
                context
            )
            
            # Prefix module shapes with module ID to avoid conflicts
            for node_id, shape in module_shape_result.get("shapes", {}).items():
                shape_map[f"{module_id}.{node_id}"] = shape
                
            context.errors.extend(module_shape_result.get("errors", []))
    
    return {"shapes": shape_map, "errors": context.errors}


def infer_node_shapes_with_context(graph: Graph, input_shape: List[int], context=None):
    """Enhanced shape inference with module context support"""
    if context is None:
        # Create basic context for backward compatibility
        class SimpleContext:
            def __init__(self):
                self.errors = []
                self.module_cache = {}
        context = SimpleContext()
    
    node_map = {node.id: node for node in graph.nodes}
    shape_map = {}
    tensor_cache = {}
    # Nodes the run never reached, so their successors inherit the same reason.
    unreached_ids: set = set()
    errors = context.errors if hasattr(context, 'errors') else []

    # Strictly require Input node - no fallback to first node
    input_nodes = [n for n in graph.nodes if n.data.layerType.name == 'Input']
    if not input_nodes:
        errors.append({
            "nodeId": "general",
            "message": "No Input node found in the graph. Please add an Input node to define the input shape.",
            "error_type": "missing_input",
            "suggestion": "Add an Input node from the palette and connect it to your model."
        })
        return {"shapes": shape_map, "errors": errors}
    
    # Initialize input tensor from Input node only
    input_node = input_nodes[0]
    try:
        shape_str = input_node.data.parameters.get('shape', '1,3,224,224')
        if not shape_str or not isinstance(shape_str, str):
            raise ValueError("Shape parameter must be a non-empty string")
            
        parsed_input_shape = [int(s.strip()) for s in shape_str.split(',')]
        if len(parsed_input_shape) == 0:
            raise ValueError("Empty shape specification")
        if any(dim <= 0 for dim in parsed_input_shape):
            raise ValueError("All dimensions must be positive")
            
        initial_tensor = torch.randn(*parsed_input_shape)
        tensor_cache[input_node.id] = initial_tensor
        shape_map[input_node.id] = list(initial_tensor.shape)
        
    except (ValueError, TypeError) as e:
        errors.append({
            "nodeId": input_node.id,
            "message": f"Invalid input shape specification: {e}",
            "error_type": "invalid_input_shape",
            "suggestion": "Use comma-separated positive integers like '1,3,224,224' for batch_size,channels,height,width"
        })
        return {"shapes": shape_map, "errors": errors}

    # Get execution order via topological sort
    try:
        execution_order = topological_sort(graph.nodes, graph.edges)
    except ValueError as e:
        errors.append({
            "nodeId": "general",
            "message": f"Graph structure error: {e}",
            "error_type": "invalid_graph",
            "suggestion": "Ensure graph is connected and has no cycles"
        })
        return {"shapes": shape_map, "errors": errors}

    # Process nodes in topological order
    for node_id in execution_order:
        current_node = node_map[node_id]
        layer_type = current_node.data.layerType.name
        layer_params = current_node.data.parameters

        # A SOURCE produces a tensor and consumes nothing, so every question this loop
        # asks below is the wrong one to ask of it.
        #
        # `nn.Parameter` is the case: it has no `forward`, so the tracer can never visit
        # it and the projection marks it `unreached` — structurally, forever, not because
        # a run failed. That one marker then fired three different rules in turn, each
        # reporting a model that runs correctly as broken: "Node has no input
        # connections", then "Node is unreachable from input", then "was not reached: the
        # run failed before it". Exempting them one at a time just moved the error along;
        # the engine has to stop treating a parameter as a computation over an input.
        if _bare_op(layer_type) in _SOURCE_OPS:
            continue

        # Handle special node types
        if layer_type == 'Input':
            continue  # Already processed
            
        elif layer_type == 'Output':
            # Output nodes just pass through their input
            input_edges = [e for e in graph.edges if e.target == node_id]
            if input_edges:
                source_id = input_edges[0].source
                if source_id in tensor_cache:
                    shape_map[node_id] = list(tensor_cache[source_id].shape)
            continue
            
        elif layer_type == 'Module' or current_node.type == 'moduleNode':
            # A declared ModuleList has no forward operation or captured members.
            # Changing its semantic kind must not turn a real connection into an
            # identity-shaped result in this retained flat inference endpoint.
            if _bare_op(layer_type) == 'modulelist':
                if any(e.source == node_id or e.target == node_id for e in graph.edges):
                    errors.append({
                        "nodeId": node_id,
                        "message": "ModuleList declares members; their executable wiring is unavailable in this static graph",
                        "error_type": "unknown_shape",
                    })
                continue
            # A GROUP node is a container whatever its display name. Keying only on the
            # literal name 'Module' meant an imported block card named "BasicBlock x2" or
            # "InvertedResidual" fell through to the layer builder and died with
            # "Unknown layer type 'BasicBlock x2'".
            # A container whose children are NOT in this graph (a depth-limited import)
            # still knows the shape the real model produced. Passing its input through
            # kept a depth-1 resnet18 at 64 channels all the way to `fc`, which expects
            # 512 — "mat1 and mat2 shapes cannot be multiplied (1x64 and 512x1000)".
            recorded = layer_params.get("_out_shape") if isinstance(layer_params, dict) else None
            has_children_here = any(
                (n.data.parentId == node_id) for n in graph.nodes
            )
            if isinstance(recorded, str) and not has_children_here:
                try:
                    dims = [int(part) for part in recorded.split("x") if part]
                except ValueError:
                    dims = []
                if dims:
                    tensor_cache[node_id] = torch.zeros(*dims)
                    shape_map[node_id] = list(dims)
                    continue

            input_edges = [e for e in graph.edges if e.target == node_id]
            if input_edges:
                source_id = input_edges[0].source
                if source_id in tensor_cache:
                    # Check if module has internal structure
                    if hasattr(current_node.data, 'internalGraph') and current_node.data.internalGraph:
                        # Module with internal structure - build and execute
                        try:
                            # This would need proper implementation
                            tensor_cache[node_id] = tensor_cache[source_id]
                            shape_map[node_id] = list(tensor_cache[source_id].shape)
                        except Exception as e:
                            errors.append({
                                "nodeId": node_id,
                                "message": f"Error processing module: {e}",
                                "error_type": "module_error"
                            })
                    else:
                        # Simple module container - pass through
                        tensor_cache[node_id] = tensor_cache[source_id]
                        shape_map[node_id] = list(tensor_cache[source_id].shape)
            continue

        # A node the run never reached carries no shape, and must not be given one.
        #
        # The trace marks every node after a failing statement `unreached`, and this
        # engine used to compute a shape for it anyway — GELU after a Linear that
        # raised showed (2, 20), confidently, from out_features alone. That put a
        # number on a card for a layer whose input never existed, while the header
        # said the run had failed. A run that stopped is a fact about the model; a
        # shape derived past the stopping point is a guess wearing the same font.
        if isinstance(layer_params, dict) and layer_params.get("_shape_provenance") == "unreached":
            unreached_ids.add(node_id)
            errors.append({
                "nodeId": node_id,
                "message": f"Node {node_id} was not reached: the run failed before it",
                "error_type": "unreached",
                "layer_type": layer_type,
                "suggestion": "Fix the error the trace reported on the earlier line, then save to run again",
            })
            continue

        # Get input tensor for computational layers
        input_edges = [e for e in graph.edges if e.target == node_id]
        if not input_edges:
            errors.append({
                "nodeId": node_id,
                "message": f"Node {node_id} has no input connections",
                "error_type": "disconnected_node",
                "layer_type": layer_type,
                "suggestion": "Connect this node to a preceding layer or input node"
            })
            continue

        missing_inputs = []
        layer_input_tensors = []
        # Support operations that take multiple inputs (like torch.add, torch.cat)
        # We need to preserve order, ideally edges come in some topological connection order,
        # but in React Flow edge order might vary. For now we assume input_edges order is fine.
        for edge in input_edges:
            source_id = edge.source
            if source_id not in tensor_cache:
                missing_inputs.append(source_id)
            else:
                layer_input_tensors.append(tensor_cache[source_id])
                
        if missing_inputs:
            # Downstream of a node the run never reached, the honest reason is the
            # same one, not "check connectivity": the wiring is fine, the run stopped.
            if any(source in unreached_ids for source in missing_inputs):
                unreached_ids.add(node_id)
                errors.append({
                    "nodeId": node_id,
                    "message": f"Node {node_id} was not reached: the run failed before it",
                    "error_type": "unreached",
                    "layer_type": layer_type,
                    "suggestion": "Fix the error the trace reported on the earlier line, then save to run again",
                })
                continue
            errors.append({
                "nodeId": node_id,
                "message": f"Input nodes {missing_inputs} have no computed tensor",
                "error_type": "missing_input_tensor",
                "layer_type": layer_type,
                "suggestion": "Check that all preceding layers are properly connected and valid"
            })
            continue

        # For error display compatibility
        layer_input_tensor = layer_input_tensors[0] if len(layer_input_tensors) == 1 else layer_input_tensors

        # An IMPORTED opaque composite carries the shape the real model actually
        # produced (`_out_shape`, written by services/ingest/normalise.py from the
        # trace). There is no analytic rule for a library-private block, and passing
        # its input through unchanged is wrong: a depth-1 resnet18 kept 64 channels all
        # the way to `fc`, which expects 512, and failed with
        # "mat1 and mat2 shapes cannot be multiplied (1x64 and 512x1000)". Trust the
        # measurement over a guess.
        observed_out = layer_params.get("_out_shape") if isinstance(layer_params, dict) else None
        if layer_params.get("_opaque") and isinstance(observed_out, str):
            try:
                observed_dims = [int(part) for part in observed_out.split("x") if part]
            except ValueError:
                observed_dims = []
            if observed_dims:
                tensor_cache[node_id] = torch.zeros(*observed_dims)
                shape_map[node_id] = list(observed_dims)
                continue

        # Multi-input combiners have no torch.nn class, so build_layer cannot make one:
        # the analytic path died with "Unknown layer type 'Add'" on any graph containing a
        # materialised residual, while the dry forward handled the same node fine through
        # its own merge route. Apply the op directly instead.
        merge_key = (current_node.data.layerType.id or "").strip().lower()
        if merge_key in _ANALYTIC_MERGE_OPS:
            try:
                merged = _apply_analytic_merge(merge_key, layer_input_tensors, layer_params)
                tensor_cache[node_id] = merged
                shape_map[node_id] = list(merged.shape)
                continue
            except (RuntimeError, ValueError, TypeError) as merge_err:
                errors.append({
                    "nodeId": node_id,
                    "message": f"Error in {layer_type}: {merge_err}",
                    "error_type": "layer_error",
                    "layer_type": layer_type,
                    "suggestion": "Check that the merged inputs have compatible shapes",
                })
                continue

        # Build and execute layer
        try:
            layer_instance = build_layer(layer_type, layer_params)

            # Pass input tensor(s) to the layer. Single-input modules go
            # through call_module so recurrent/attention/transformer layers
            # (LSTM/GRU/RNN/MultiheadAttention/Transformer*) that return tuples
            # are unwrapped to their primary output tensor — otherwise a
            # top-level RNN node crashes with "'tuple' object has no attribute
            # 'shape'". Multi-input merge ops (cat/add) keep the *-spread.
            if len(layer_input_tensors) == 1:
                layer_output_tensor = call_module(layer_instance, layer_input_tensors[0])
            else:
                layer_output_tensor = layer_instance(*layer_input_tensors)

            # Cache results
            tensor_cache[node_id] = layer_output_tensor
            shape_map[node_id] = list(layer_output_tensor.shape)
            
        except (RuntimeError, ValueError, TypeError) as e:
            error_msg = str(e)
            
            # Generate specific error information
            # For a multi-input node `layer_input_tensor` is a LIST of tensors (see where
            # it is assigned), so reading .shape crashed the error REPORTER itself and the
            # real failure was replaced by "'list' object has no attribute 'shape'".
            if isinstance(layer_input_tensor, (list, tuple)):
                input_shape_info = [
                    list(t.shape) for t in layer_input_tensor if hasattr(t, "shape")
                ]
            elif layer_input_tensor is not None and hasattr(layer_input_tensor, "shape"):
                input_shape_info = list(layer_input_tensor.shape)
            else:
                input_shape_info = "Unknown"
            
            # Provide specific guidance for common errors
            if layer_type == "Linear" and "mat1 and mat2 shapes cannot be multiplied" in error_msg:
                expected_features = layer_params.get('in_features', 'Unknown')
                actual_features = layer_input_tensor.shape[-1] if layer_input_tensor is not None and len(layer_input_tensor.shape) > 1 else "Unknown"
                
                error_detail = {
                    "nodeId": node_id,
                    "message": f"Shape mismatch in Linear layer: expected {expected_features} input features, got {actual_features}",
                    "error_type": "shape_mismatch",
                    "input_shape": input_shape_info,
                    "expected_shape": f"[*, {expected_features}]",
                    "layer_type": layer_type,
                    "suggestion": f"Change in_features to {actual_features} or add a layer to reshape input to [..., {expected_features}]"
                }
            elif layer_type == "Conv2d" and ("Expected" in error_msg and "input" in error_msg):
                expected_channels = layer_params.get('in_channels', 'Unknown')
                actual_channels = layer_input_tensor.shape[1] if layer_input_tensor is not None and len(layer_input_tensor.shape) > 1 else "Unknown"
                
                error_detail = {
                    "nodeId": node_id,
                    "message": f"Shape mismatch in Conv2d layer: expected {expected_channels} input channels, got {actual_channels}",
                    "error_type": "shape_mismatch",
                    "input_shape": input_shape_info,
                    "expected_shape": f"[N, {expected_channels}, H, W]",
                    "layer_type": layer_type,
                    "suggestion": f"Change in_channels to {actual_channels} or add a layer to adjust channel dimensions"
                }
            else:
                error_detail = {
                    "nodeId": node_id,
                    "message": f"Error in {layer_type}: {error_msg}",
                    "error_type": "layer_error",
                    "input_shape": input_shape_info,
                    "layer_type": layer_type,
                    "suggestion": "Check layer parameters and input compatibility"
                }
            
            errors.append(error_detail)
            continue

    return {"shapes": shape_map, "errors": errors}


def infer_node_shapes(graph: Graph, input_shape: List[int]):
    """
    Infer output shapes for all nodes in the graph.
    Requires an explicit Input node to avoid silent fallback errors.
    Thin compatibility wrapper that delegates to infer_node_shapes_with_context().
    """
    return infer_node_shapes_with_context(graph, input_shape)


class ShapeInferenceService:
    """Service for shape inference and model summary calculations"""
    
    def __init__(self):
        self.cache = {}
    
    def infer_shapes(self, graph: Graph, input_shape: List[int]) -> Dict[str, Any]:
        """Infer output shapes for all nodes in a graph"""
        try:
            # Start with input shape
            shapes = {}
            
            # Find input node and set its shape
            input_nodes = [node for node in graph.nodes if node.data.layerType.id == 'input' or 'input' in node.data.layerType.id.lower()]
            if input_nodes:
                shapes[input_nodes[0].id] = input_shape
            
            # Build execution order
            execution_order = topological_sort(graph.nodes, graph.edges)
            
            # Process each node in topological order to infer shapes
            current_shape = input_shape
            
            for node_id in execution_order:
                node = next((n for n in graph.nodes if n.id == node_id), None)
                if not node:
                    continue
                
                layer_type = node.data.layerType.id.lower()
                params = node.data.parameters
                
                # Calculate output shape based on layer type and input shape
                try:
                    if layer_type == 'input':
                        output_shape = input_shape
                    elif layer_type == 'linear':
                        in_features = params.get('in_features', current_shape[-1] if current_shape else 0)
                        out_features = params.get('out_features', in_features)
                        output_shape = current_shape[:-1] + [out_features] if current_shape else [out_features]
                    elif layer_type in ['conv2d', 'conv']:
                        if len(current_shape) >= 3:  # [C, H, W]
                            out_channels = params.get('out_channels', current_shape[0])
                            kernel_size = params.get('kernel_size', 3)
                            stride = params.get('stride', 1)
                            padding = params.get('padding', 0)
                            
                            # Calculate output spatial dimensions
                            h_out = (current_shape[1] + 2 * padding - kernel_size) // stride + 1
                            w_out = (current_shape[2] + 2 * padding - kernel_size) // stride + 1
                            output_shape = [out_channels, h_out, w_out]
                        else:
                            output_shape = current_shape
                    elif layer_type in ['relu', 'sigmoid', 'tanh', 'leakyrelu']:
                        # Activation functions don't change shape
                        output_shape = current_shape
                    elif layer_type == 'flatten':
                        if len(current_shape) > 1:
                            output_shape = [current_shape[0] * current_shape[1] * current_shape[2]] if len(current_shape) >= 3 else [sum(current_shape)]
                        else:
                            output_shape = current_shape
                    elif layer_type in ['maxpool2d', 'avgpool2d']:
                        if len(current_shape) >= 3:
                            kernel_size = params.get('kernel_size', 2)
                            stride = params.get('stride', kernel_size)
                            
                            h_out = current_shape[1] // stride
                            w_out = current_shape[2] // stride
                            output_shape = [current_shape[0], h_out, w_out]
                        else:
                            output_shape = current_shape
                    elif layer_type in ['batchnorm1d', 'batchnorm2d', 'dropout']:
                        # These layers don't change shape
                        output_shape = current_shape
                    elif layer_type == 'embedding':
                        # nn.Embedding appends embedding_dim as a new trailing
                        # dimension: input (N, *) of token indices -> (N, *, dim).
                        embedding_dim = params.get('embedding_dim', 128)
                        output_shape = [*current_shape, embedding_dim] if current_shape else [embedding_dim]
                    elif layer_type == 'lstm':
                        seq_len = current_shape[0] if len(current_shape) >= 2 else 1
                        hidden_size = params.get('hidden_size', 128)
                        bidirectional = params.get('bidirectional', False)
                        output_size = hidden_size * (2 if bidirectional else 1)
                        output_shape = [seq_len, output_size]
                    elif layer_type == 'output':
                        output_shape = current_shape
                    else:
                        # Unknown layer type, keep current shape
                        output_shape = current_shape
                    
                    # Store the shape for this node
                    shapes[node.id] = output_shape
                    
                    # Update current shape for next layer
                    # Find edges that have this node as source
                    outgoing_edges = [e for e in graph.edges if e.source == node.id]
                    if outgoing_edges:
                        current_shape = output_shape
                        
                except Exception as layer_error:
                    print(f"Error processing layer {node.id}: {layer_error}")
                    # On error, keep the current shape
                    shapes[node.id] = current_shape
            
            return {
                "success": True,
                "shapes": shapes,
                "output_shape": current_shape
            }
            
        except Exception as e:
            print(f"Shape inference error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "shapes": {},
                "output_shape": []
            }
    
    def calculate_summary(self, graph: Graph, input_shape: List[int]) -> Dict[str, Any]:
        """Calculate model summary with parameter counts and memory usage"""
        return calculate_model_summary(graph.dict(), input_shape)
    
    def validate_graph(self, graph: Graph, input_shape: List[int]) -> List[Dict[str, str]]:
        """Validate graph for shape compatibility"""
        return validate_graph(graph.dict(), input_shape)
    
    def build_model_from_graph(self, graph: Graph) -> nn.Module:
        """Build a PyTorch model from graph representation"""
        class GraphModel(nn.Module):
            def __init__(self, graph: Graph):
                super().__init__()
                self.graph = graph
                
                # Build layers
                for node in graph.nodes:
                    if node.type == "layerNode":
                        layer_type = node.data.layerType.id
                        params = node.data.parameters
                        layer = build_layer(layer_type, params)
                        setattr(self, node.id, layer)
                
                # Store execution order
                self.execution_order = topological_sort(graph.nodes, graph.edges)
            
            def forward(self, x):
                tensor_map = {"input": x}
                
                for node_id in self.execution_order:
                    node = next((n for n in self.graph.nodes if n.id == node_id), None)
                    if not node or node.type not in ["layerNode", "outputNode"]:
                        continue
                    
                    if node.type == "layerNode":
                        # Get input tensor(s)
                        inputs = []
                        for edge in self.graph.edges:
                            if edge.target == node_id:
                                inputs.append(tensor_map[edge.source])
                        
                        if inputs:
                            input_tensor = inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=1)
                            layer = getattr(self, node_id)
                            tensor_map[node_id] = call_module(layer, input_tensor)
                    
                    elif node.type == "outputNode":
                        # Get final output
                        for edge in self.graph.edges:
                            if edge.target == node_id:
                                return tensor_map[edge.source]
                
                # If no explicit output node, return last computed tensor
                return list(tensor_map.values())[-1]
        
        return GraphModel(graph)


# Keep existing standalone functions for backward compatibility
