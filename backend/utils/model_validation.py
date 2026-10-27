"""Model validation utilities for PyTorch models"""
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
from core.op_registry import is_placeholder_op, spec_for
from services.code_generation import generator_supports_op
from models.graph_models import Graph, Node, Edge, LayerData, HierarchicalGraph

# The three graph node kinds that are not torch operations. They exist nowhere in
# op_registry on purpose — Input and Output carry a shape, Module carries a
# sub-graph — so an "is this a real layer?" check has to exempt them or it
# rejects every graph the product itself builds. codegen already keeps the same
# list (services/code_generation.py, keyed on layerType.name); this is that list
# keyed on layerType.id, which is what reaches this service.
_NON_OPERATION_LAYER_IDS = frozenset({"input", "output", "module"})
_NON_OPERATION_LAYER_NAMES = frozenset({"Input", "Output", "Module"})


class ModelValidationService:
    """Service for validating PyTorch models and their configurations"""
    
    def __init__(self):
        self.validation_cache = {}
    
    def validate_model_graph(self, graph: Graph) -> Dict[str, Any]:
        """Validate a model graph for consistency and correctness"""
        errors = []
        warnings = []
        
        # Check for basic graph structure issues
        structure_issues = self._validate_graph_structure(graph)
        errors.extend(structure_issues.get("errors", []))
        warnings.extend(structure_issues.get("warnings", []))
        
        # Check for layer compatibility
        compatibility_issues = self._validate_layer_compatibility(graph)
        errors.extend(compatibility_issues.get("errors", []))
        warnings.extend(compatibility_issues.get("warnings", []))
        
        # Check for parameter validity
        parameter_issues = self._validate_parameters(graph)
        errors.extend(parameter_issues.get("errors", []))
        warnings.extend(parameter_issues.get("warnings", []))
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "total_issues": len(errors) + len(warnings)
        }
    
    def validate_flat_graph(
        self, graph: Graph, input_shape: List[int]
    ) -> List[Dict[str, Any]]:
        """Validate a flat graph and return the ``[{nodeId, message}]`` list that
        ``/api/models/validate`` reports to the frontend.

        ``server_v2.py`` has always called this on the non-hierarchical branch but it was
        never defined, so any flat graph POSTed to ``/api/models/validate`` answered
        HTTP 500 with ``'ModelValidationService' object has no attribute
        'validate_flat_graph'``. Sibling of :meth:`validate_hierarchical_graph`, and like
        it, ``input_shape`` is accepted for call-site parity: shape checking belongs to the
        verification service, not structural validation.
        """
        return [
            {"nodeId": "root", "message": message}
            for message in self.validate_model_graph(graph).get("errors", [])
        ]

    def validate_hierarchical_graph(
        self, hierarchical_graph: HierarchicalGraph, input_shape: List[int]
    ) -> List[Dict[str, Any]]:
        """Validate a hierarchical graph — the root graph plus every module's
        internal graph — and return the flat ``[{nodeId, message}]`` error list
        that ``/api/models/validate`` reports to the frontend.

        ``input_shape`` is accepted for parity with the flat path's call site;
        structural validation does not need it (shape checking is the
        verification service's job, not this one's).
        """
        errors: List[Dict[str, Any]] = []

        def collect(node_id: str, graph: Graph) -> None:
            # Only hard errors — warnings (e.g. "No input node found", which is
            # normal for a module's internals) are not part of this contract.
            for message in self.validate_model_graph(graph).get("errors", []):
                errors.append({"nodeId": node_id, "message": message})

        collect("root", Graph(nodes=hierarchical_graph.root_nodes, edges=hierarchical_graph.root_edges))

        for module_id, module in hierarchical_graph.modules.items():
            collect(module_id, Graph(nodes=module.internal_nodes, edges=module.internal_edges))

            internal_ids = {node.id for node in module.internal_nodes}
            for boundary in ("input_node_id", "output_node_id"):
                boundary_id = getattr(module, boundary)
                if boundary_id and boundary_id not in internal_ids:
                    errors.append({
                        "nodeId": module_id,
                        "message": (
                            f"Module {module_id}: {boundary} '{boundary_id}' is not one of its "
                            "internal nodes"
                        ),
                    })

        for node_id, module_id in hierarchical_graph.node_hierarchy.items():
            if module_id not in hierarchical_graph.modules:
                errors.append({
                    "nodeId": node_id,
                    "message": f"Node {node_id} is assigned to unknown module '{module_id}'",
                })

        return errors

    def warnings_for_graph(self, graph: Graph, node_id: str = "root") -> List[Dict[str, Any]]:
        """Structural warnings for one graph, shaped like the error list.

        ``/api/models/validate`` used to answer ``"warnings": []`` unconditionally, so
        every warning this service produced — including "no Input node, assuming
        [1, 3, 224, 224]" — was computed and thrown away. That is what let a graph
        whose input shape nobody chose come back as an unqualified green tick.

        Callers ask for the root graph only. A module's internal graph warns about
        its missing Input and Output nodes by design, and repeating that per module
        would bury the warnings that mean something.
        """
        return [
            {"nodeId": node_id, "message": message}
            for message in self.validate_model_graph(graph).get("warnings", [])
        ]

    def _validate_graph_structure(self, graph: Graph) -> Dict[str, List[str]]:
        """Validate basic graph structure"""
        errors = []
        warnings = []
        
        # Check for disconnected nodes
        connected_nodes = set()
        for edge in graph.edges:
            connected_nodes.add(edge.source)
            connected_nodes.add(edge.target)
        
        all_nodes = {node.id for node in graph.nodes}
        disconnected = all_nodes - connected_nodes
        
        if len(disconnected) > 1:  # Allow one input node to be disconnected
            warnings.append(f"Found {len(disconnected)} disconnected nodes: {list(disconnected)}")
        
        # Check for cycles (should not exist in feedforward networks)
        if self._has_cycles(graph):
            errors.append("Graph contains cycles - this may cause infinite loops")
        
        # Check for multiple input/output nodes
        input_nodes = [node for node in graph.nodes if node.type == "inputNode"]
        output_nodes = [node for node in graph.nodes if node.type == "outputNode"]
        
        if len(input_nodes) == 0:
            # Loud, not silent. Everything downstream — /api/models/validate, the
            # verify forward pass, codegen — falls back to [1, 3, 224, 224] when no
            # Input node says otherwise, so a graph can be pronounced valid against
            # a shape the user never chose (the GO-19 case). Saying which shape was
            # assumed is the difference between a green tick that means something
            # and one that does not. It stays a warning rather than an error because
            # a module's internal graph legitimately has no Input node of its own.
            warnings.append(
                "No Input node found — validation assumed the default input shape "
                "[1, 3, 224, 224]. Add an Input node to check the shape you actually "
                "intend to feed this model."
            )
        elif len(input_nodes) > 1:
            warnings.append(f"Multiple input nodes found: {[n.id for n in input_nodes]}")
        
        if len(output_nodes) == 0:
            warnings.append("No output node found")
        elif len(output_nodes) > 1:
            warnings.append(f"Multiple output nodes found: {[n.id for n in output_nodes]}")
        
        return {"errors": errors, "warnings": warnings}
    
    def _validate_layer_compatibility(self, graph: Graph) -> Dict[str, List[str]]:
        """Validate layer input/output compatibility"""
        errors = []
        warnings = []
        
        # Build adjacency map
        adjacency = {}
        for edge in graph.edges:
            if edge.source not in adjacency:
                adjacency[edge.source] = []
            adjacency[edge.source].append(edge.target)
        
        # Check each connection
        for source_id, targets in adjacency.items():
            source_node = next((n for n in graph.nodes if n.id == source_id), None)
            if not source_node:
                continue
                
            for target_id in targets:
                target_node = next((n for n in graph.nodes if n.id == target_id), None)
                if not target_node:
                    continue
                
                # Check tensor shape compatibility
                compatibility_issue = self._check_tensor_compatibility(source_node, target_node)
                if compatibility_issue:
                    errors.append(f"Incompatible connection {source_id} -> {target_id}: {compatibility_issue}")
        
        return {"errors": errors, "warnings": warnings}
    
    def _validate_parameters(self, graph: Graph) -> Dict[str, List[str]]:
        """Validate layer parameters"""
        errors = []
        warnings = []
        
        for node in graph.nodes:
            # Every node that carries layer parameters gets checked, not just the
            # ones whose `type` happens to read "layerNode". The old allow-list
            # silently exempted moduleNode — which does carry real parameters — so
            # a payload only slightly off the canvas's usual spelling bypassed
            # every parameter check and came back valid:true with no warning.
            # inputNode/outputNode are excluded because their "parameters" are a
            # shape, not layer hyper-parameters.
            if node.type not in ("inputNode", "outputNode"):
                errors.extend(self._validate_layer_parameters(node))
        
        return {"errors": errors, "warnings": warnings}
    
    def _has_cycles(self, graph: Graph) -> bool:
        """Check if graph has cycles using DFS"""
        adjacency = {}
        for edge in graph.edges:
            if edge.source not in adjacency:
                adjacency[edge.source] = []
            adjacency[edge.source].append(edge.target)
        
        visited = set()
        rec_stack = set()
        
        def dfs(node):
            if node in rec_stack:
                return True  # Back edge found - cycle detected
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in adjacency.get(node, []):
                if dfs(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in graph.nodes:
            if node.id not in visited:
                if dfs(node.id):
                    return True
        
        return False
    
    def _check_tensor_compatibility(self, source_node: Node, target_node: Node) -> Optional[str]:
        """Check if two layers are tensor-compatible"""
        # This is a simplified check - in practice, you'd need more sophisticated shape inference
        if source_node.type == "inputNode" or target_node.type == "outputNode":
            return None
        
        # Basic compatibility checks based on layer types
        if hasattr(source_node.data, 'layerType') and hasattr(target_node.data, 'layerType'):
            source_type = source_node.data.layerType.id
            target_type = target_node.data.layerType.id
            
            # Check some basic incompatibilities
            if source_type in ["conv2d", "maxpool2d", "avgpool2d"] and target_type == "linear":
                if not self._has_flatten_between(source_node.id, target_node.id):
                    return "2D tensor cannot be directly connected to Linear layer without flattening"
        
        return None
    
    def _has_flatten_between(self, source_id: str, target_id: str) -> bool:
        """Check if there's a flatten operation between two nodes"""
        # This would need access to the full graph structure
        # For now, return False as a conservative approach
        return False
    
    @staticmethod
    def _numbers_in(value: Any) -> List[float]:
        """Every number a stored parameter readably contains.

        An empty list means "nothing numeric to read here", which is not the
        same as zero: it is the caller's cue to hold no opinion rather than to
        report an error.
        """
        if isinstance(value, bool):
            return []
        if isinstance(value, (int, float)):
            return [float(value)]
        if isinstance(value, (list, tuple)):
            return [number for item in value for number in ModelValidationService._numbers_in(item)]
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",") if part.strip()]
            numbers = []
            for part in parts:
                try:
                    numbers.append(float(part))
                except ValueError:
                    return []
            return numbers
        return []

    @classmethod
    def _not_positive(cls, value: Any) -> bool:
        """True only when the value is readably a number that is not positive.

        Parameters reach this service as whatever the canvas stored, and the
        editor stores several shapes for the same field: an asymmetric kernel is
        [3, 3], a typed one arrives as "3" or "3,3". Comparing the raw value
        against 0 raised TypeError for every non-numeric form, and the
        endpoint's bare `except` turned that into a 500 — so a graph with an
        asymmetric kernel could not be validated at all, let alone found
        invalid. Anything unreadable is left alone here: the shape engine and
        codegen both interpret these values, and inventing an error for a form
        this function simply cannot parse would block a model that builds.
        """
        numbers = cls._numbers_in(value)
        return any(number <= 0 for number in numbers)

    @staticmethod
    def _names_an_operation(node: Node) -> bool:
        """Does this node name something that can actually be built?

        True for the non-operation kinds (Input/Output/Module) as well, because
        "unknown" here means "names no operation and is not one of the three kinds
        that legitimately name none".

        The oracle is `generator_supports_op`, not `spec_for`, because "can be
        built" is a question about the GENERATOR, and the generator accepts more
        than the op registry holds: it ships shape-preserving nn.Module sources
        for grn, sinusoidal_pos_embedding, rotary_embedding and the adaLN family.
        Asking `spec_for` alone called F5-TTS unbuildable — the one model those
        implementations were written for — and the catalogue filter that trusts
        this answer then hid it. Both canonical_validation (custom_passthrough_ops)
        and verification_service (_CUSTOM_PASSTHROUGH_OPS) already treat these ops
        as real; this validator was the last one that did not.
        """
        if node.type == "moduleNode":
            return True
        layer_type = node.data.layerType
        if (layer_type.id or "").lower() in _NON_OPERATION_LAYER_IDS:
            return True
        if (getattr(layer_type, "name", "") or "") in _NON_OPERATION_LAYER_NAMES:
            return True
        return generator_supports_op(layer_type.id or "")

    @staticmethod
    def _recorded_shape(parameters: Dict[str, Any]) -> Optional[str]:
        """The shape a placeholder node already carries, if it carries one.

        `_out_shape` is written by exactly two things: the trace worker, which
        measured it, and a `# tensorviz: out_shape=` declaration in the class's
        own source (task.md M6). Both are evidence from outside this validator;
        neither is a guess it made. A node without one is a node nobody has
        checked, which is what the refusal below is for.
        """
        if not isinstance(parameters, dict):
            return None
        recorded = parameters.get("_out_shape")
        return recorded if isinstance(recorded, str) and recorded.strip() else None

    @staticmethod
    def _source_location(parameters: Dict[str, Any]) -> str:
        """`file:line` for a node carrying parser provenance, so a refusal points at
        the statement rather than at a node id the user never chose."""
        source_file = parameters.get("_source_file") if isinstance(parameters, dict) else None
        lineno = parameters.get("_lineno") if isinstance(parameters, dict) else None
        if isinstance(source_file, str) and isinstance(lineno, int):
            return f"{source_file}:{lineno}"
        if isinstance(lineno, int):
            return f"line {lineno}"
        return "no source location recorded"

    def _validate_layer_parameters(self, node: Node) -> List[str]:
        """Validate parameters for a specific layer"""
        errors = []

        if not hasattr(node.data, 'layerType') or not hasattr(node.data, 'parameters'):
            return errors

        layer_type = node.data.layerType.id
        parameters = node.data.parameters

        # A layer id that names no operation cannot be built, shape-inferred or
        # exported, so calling such a graph valid is the worst kind of wrong: the
        # green tick is the only signal the user has, and the failure surfaces
        # later as a codegen 500. op_registry.spec_for is alias-aware and strips
        # the torch.nn./nn. prefixes, so every id the palette can produce resolves
        # (verified: 150 of the palette's 153 ids, the other three being the
        # non-operation kinds exempted here).
        if not self._names_an_operation(node):
            errors.append(
                f"Node {node.id}: unknown layer type '{layer_type}' — it matches no "
                "supported operation"
            )
            return errors

        # `python` / `custom_code` DO resolve through spec_for — that is what lets the
        # canvas render them — but they name source this validator cannot model: an
        # opaque statement, or an instance of the user's own nn.Module. Letting them
        # fall through would answer `valid: true` for a graph nobody has checked,
        # which is the same false green the comment above is about, just wearing a
        # registered id. Their shape is knowable only by running the model.
        if is_placeholder_op(layer_type or ""):
            # ...unless the node carries the answer. An executed trace (or the
            # class's own `# tensorviz: out_shape=` declaration) puts the output
            # shape on the node, and that is precisely what "only known once the
            # model is executed" was waiting for. Nothing else about the code is
            # checked here — there is nothing else this validator could check —
            # so the graph is blessed for exactly what it can show evidence of.
            if self._recorded_shape(parameters) is None:
                errors.append(
                    f"Node {node.id}: '{layer_type}' is source code, not an operation this "
                    f"validator can check ({self._source_location(parameters)}). Its shape "
                    "is only known once the model is executed."
                )
            return errors

        spec = spec_for(layer_type or "")

        # Validate based on layer type. A missing key still defaults to 0, so an
        # unset required parameter is reported as before.
        #
        # Dispatched on the REGISTRY's canonical name, because this chain is where
        # the audit's "or the layer id is capitalized" half of the silent skip
        # lived. Resolving unknown ids through spec_for (above) is alias-aware and
        # case-insensitive, so `Linear` is not "unknown" — but this chain compared
        # against the bare lowercase literal, so `layerType.id: "Linear"` with
        # in_features:-5 fell off the end and came back valid:true with no error.
        #
        # Deliberately NOT a second normalisation of our own. The first attempt at
        # this fix re-implemented spec_for's prefix stripping as a local regex and
        # the two disagreed: spec_for strips exactly one prefix, and the registry
        # keys ARE the prefixed canonical names ('nn.linear'), so `nn.nn.Linear`
        # resolves to a real spec while a one-shot regex leaves 'nn.linear' and
        # matches no branch below — a negative out_features sailed through under
        # that spelling. One canonicalisation, owned by the registry, is the only
        # way these two can't drift apart again.
        canonical = spec.canonical.lower() if spec is not None else (layer_type or "").strip().lower()
        if canonical in ("nn.conv2d", "conv2d"):
            if self._not_positive(parameters.get("kernel_size", 0)):
                errors.append(f"Node {node.id}: Conv2d kernel_size must be positive")
            if self._not_positive(parameters.get("in_channels", 0)):
                errors.append(f"Node {node.id}: Conv2d in_channels must be positive")
            if self._not_positive(parameters.get("out_channels", 0)):
                errors.append(f"Node {node.id}: Conv2d out_channels must be positive")

        elif canonical in ("nn.linear", "linear"):
            if self._not_positive(parameters.get("in_features", 0)):
                errors.append(f"Node {node.id}: Linear in_features must be positive")
            if self._not_positive(parameters.get("out_features", 0)):
                errors.append(f"Node {node.id}: Linear out_features must be positive")

        elif canonical in ("nn.batchnorm2d", "batchnorm2d"):
            if self._not_positive(parameters.get("num_features", 0)):
                errors.append(f"Node {node.id}: BatchNorm2d num_features must be positive")

        return errors
    
    def validate_pytorch_model(self, model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Validate an actual PyTorch model"""
        try:
            # Create dummy input
            dummy_input = torch.randn(1, *input_shape)
            
            # Try forward pass
            model.eval()
            with torch.no_grad():
                output = model(dummy_input)
            
            return {
                "valid": True,
                "output_shape": tuple(output.shape[1:]),  # Remove batch dimension
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "errors": [],
                "warnings": []
            }
        
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Model validation failed: {str(e)}"],
                "warnings": [],
                "output_shape": None,
                "total_parameters": 0,
                "trainable_parameters": 0
            }
    
    def suggest_fixes(self, validation_result: Dict[str, Any]) -> List[str]:
        """Suggest fixes for validation errors"""
        suggestions = []
        
        for error in validation_result.get("errors", []):
            if "disconnected nodes" in error.lower():
                suggestions.append("Connect all nodes in a valid computational graph")
            elif "cycles" in error.lower():
                suggestions.append("Remove cyclic connections to create a feedforward network")
            elif "incompatible connection" in error.lower():
                suggestions.append("Add appropriate reshaping layers (e.g., Flatten) between incompatible layers")
            elif "must be positive" in error.lower():
                suggestions.append("Ensure all layer parameters have positive values")
        
        return suggestions
