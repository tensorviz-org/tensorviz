"""
Code generation service for converting visual graphs to PyTorch code.
Supports hierarchical, modular code generation.
"""
import ast
import keyword
import re
from typing import Dict, List, Any, Optional, Tuple

from core.graph_input_safety import (
    UnsafeGraphInput,
    require_layer_operation_syntax,
    require_python_identifier,
    require_registered_layer_operation,
    require_resource_identifier,
    validate_layer_parameters,
)
from core.op_registry import is_placeholder_op, normalized_parameter
from models.graph_models import Graph, HierarchicalGraph, Node, Edge, ModuleDefinition


# Merge / elementwise ops emitted FUNCTIONALLY in the forward pass (no __init__
# layer) — keyed by op id, so F5-TTS's "Add"/"Concat" become torch.add/torch.cat
# instead of bogus nn.Add()/nn.Concat().
_MERGE_OPS = {"add", "sub", "mul", "div", "matmul", "bmm", "cat", "concat", "stack"}

# Functional torch.* ops whose call is meaningless without these parameters.
# Without this, a node left at its palette defaults emitted e.g. `torch.reshape(x)`
# — syntactically fine, TypeError the moment it runs. core/layer_builder.py has
# guarded exactly this for `nn.` CONSTRUCTORS for a while; the functional path had
# no equivalent, so "never export code that cannot run" only held on one side.
_FUNCTIONAL_REQUIRED = {
    "reshape": ("shape",),
    "narrow": ("dim", "start", "length"),
    "split": ("split_size_or_sections",),
    "chunk": ("chunks",),
    "transpose": ("dims",),
    "permute": ("dims",),
}

# ANY one of these, not all: F.interpolate takes size XOR scale_factor and errors
# when given neither.
_FUNCTIONAL_REQUIRED_ANY = {
    "interpolate": ("size", "scale_factor"),
}

# Ops emitted from torch.nn.functional rather than the torch namespace. There is
# no torch.interpolate at all, so the generic `torch.{op}(...)` tail would emit a
# call to something that does not exist. The generated preamble already imports
# torch.nn.functional as F.
_FUNCTIONAL_NAMESPACE = {"interpolate"}

# Ops that aren't in torch.nn. We emit faithful, shape-preserving nn.Module
# implementations into the generated file so the exported model actually imports
# and runs (the F5-TTS text/DiT blocks rely on these). op id -> class name.
_CUSTOM_OP_CLASS = {
    "grn": "GRN", "global_response_norm": "GRN",
    "rotary_embedding": "RotaryEmbedding", "rope": "RotaryEmbedding",
    "ada_layer_norm": "AdaLayerNorm", "adalayernorm": "AdaLayerNorm",
    "ada_layer_norm_zero": "AdaLayerNorm", "ada_layer_norm_final": "AdaLayerNorm",
    "sinusoidal_pos_embedding": "SinusoidalPosEmbedding",
    "sinusoidal_position_embedding": "SinusoidalPosEmbedding",
    "gaussian_noise": "GaussianNoise",
}

# Reference implementations (class name -> source). Shape-preserving so they
# slot into the structural graph and run; faithful enough to be a sane starting
# point a user can refine.
_CUSTOM_OP_SOURCE: Dict[str, str] = {
    "GRN": '''class GRN(nn.Module):
    """Global Response Normalization (ConvNeXt-V2), parameter-free, shape-preserving."""
    def forward(self, x):
        gx = torch.norm(x, p=2, dim=-2, keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return x + x * nx''',
    "RotaryEmbedding": '''class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (rotate-half), shape-preserving."""
    def forward(self, x):
        D = x.shape[-1]
        N = x.shape[-2] if x.dim() >= 2 else 1
        half = D // 2
        if half == 0:
            return x
        inv = 1.0 / (10000.0 ** (torch.arange(half, device=x.device).float() / half))
        ang = torch.outer(torch.arange(N, device=x.device).float(), inv)
        cos = ang.cos().repeat(1, 2)
        sin = ang.sin().repeat(1, 2)
        if cos.shape[-1] < D:
            cos = F.pad(cos, (0, D - cos.shape[-1]), value=1.0)
            sin = F.pad(sin, (0, D - sin.shape[-1]), value=0.0)
        x1, x2 = x[..., :half], x[..., half:2 * half]
        rot = torch.cat([-x2, x1], dim=-1)
        if rot.shape[-1] < D:
            rot = F.pad(rot, (0, D - rot.shape[-1]))
        for _ in range(x.dim() - 2):
            cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
        return x * cos + rot * sin''',
    "AdaLayerNorm": '''class AdaLayerNorm(nn.Module):
    """Adaptive LayerNorm: normalize x, then modulate by a conditioning signal
    (typically an AdaLayerNorm-Zero projection of the timestep embedding —
    shift/scale/gate chunks). Shape-preserving."""
    def forward(self, x, cond=None):
        x = F.layer_norm(x, (x.shape[-1],))
        if cond is None:
            return x
        if cond.dim() == x.dim() - 1:
            cond = cond.unsqueeze(-2)
        d = x.shape[-1]
        if cond.shape[-1] >= 2 * d and cond.shape[-1] % d == 0:
            shift, scale = cond[..., :d], cond[..., d:2 * d]
            x = x * (1.0 + scale) + shift
        elif cond.shape[-1] == d:
            x = x * (1.0 + torch.tanh(cond)) + cond
        return x''',
    "SinusoidalPosEmbedding": '''class SinusoidalPosEmbedding(nn.Module):
    """Sinusoidal embedding of the input (shape-preserving)."""
    def forward(self, x):
        D = x.shape[-1]
        half = D // 2
        if half == 0:
            return torch.sin(x)
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=x.device).float() / half)
        a = x[..., :half] * freqs
        emb = torch.cat([torch.sin(a), torch.cos(a)], dim=-1)
        if emb.shape[-1] < D:
            emb = F.pad(emb, (0, D - emb.shape[-1]))
        return emb[..., :D]''',
    "GaussianNoise": '''class GaussianNoise(nn.Module):
    """Additive isotropic Gaussian noise: one step of a diffusion forward process,
    x + sigma * eps with eps ~ N(0, I). Parameter-free and shape-preserving —
    replace sigma with a real beta schedule keyed on the timestep to train with it."""
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = float(sigma)

    def forward(self, x):
        return x + self.sigma * torch.randn_like(x)''',
}


#: Marks an ``_emit_op`` result that is a STATEMENT, not an expression to bind.
#: A line the generator cannot write must refuse when it runs rather than assign
#: something plausible — the same choice ``_hollow_module_classes`` makes for an
#: imported module whose body was not exported.
_REFUSAL_PREFIX = "raise "


#: `torch.randn([1, 16, 256])` / `torch.randn(1, 16, 256)` — an input a plain
#: comma-separated declaration can express, and therefore one BOTH tiers can read.
_PLAIN_RANDN = re.compile(r"torch\.randn\(\s*[\[\(]?\s*(?P<dims>\d+(?:\s*,\s*\d+)*)\s*[\]\)]?\s*\)")


def _op_id(node: Node) -> str:
    return (node.data.layerType.id or "").lower()


def generator_supports_op(op: str) -> bool:
    """True when the generator can emit a runnable line for `op`.

    The generator accepts an op from EITHER of two sources: the op registry, or
    `_CUSTOM_OP_CLASS` above, for which it emits a shape-preserving nn.Module
    definition into the exported file. Asking the op registry alone understates
    what will build — that is what dropped F5-TTS, whose DiT blocks are the very
    reason `_CUSTOM_OP_CLASS` exists.
    """
    if not op:
        return False
    if str(op).lower() in _CUSTOM_OP_CLASS:
        return True
    from core.op_registry import spec_for  # local: op_registry imports nothing here

    return spec_for(str(op)) is not None


class CodeGenerationService:
    """Service for generating modular PyTorch code from visual graphs"""
    
    def __init__(self):
        self.indent = "    "  # 4 spaces

    @staticmethod
    def _safe_comment(value: Any) -> str:
        """Keep dynamic diagnostics on one inert comment line."""

        return " ".join(str(value).replace("\r", " ").splitlines())

    @classmethod
    def _validate_nodes(cls, nodes: List[Node], *, label: str) -> None:
        generated_names: Dict[str, str] = {}
        for node in nodes:
            require_resource_identifier(node.id, label=f"{label} node id")
            generated = cls._node_id_to_layer_name(node.id)
            collision = generated_names.get(generated)
            if collision is not None and collision != node.id:
                raise UnsafeGraphInput(
                    f"node ids {collision!r} and {node.id!r} generate the same "
                    "Python identifier"
                )
            generated_names[generated] = node.id

            layer_type = node.data.layerType.name
            op_id = _op_id(node)
            parameters = node.data.parameters or {}
            for name in parameters:
                require_python_identifier(name, label=f"{label} parameter key")

            # `python` / `custom_code` name source the registry does not model:
            # an opaque statement, or an instance of the user's own nn.Module.
            # Neither can be invented from the registry — an `nn.python()` is
            # exactly the "exported code that cannot run" this validation exists
            # to stop. What CAN be written is the node's own record of itself:
            # a `custom_code` node that knows its class and the module it is
            # imported from, or a `python` node carrying the statement's source
            # text (task.md M6). Without those there is still nothing to write,
            # so the refusal stands — and it says WHERE the code lives, because
            # editing it in place is the only way out.
            if is_placeholder_op(op_id) and not cls._writable_placeholder(node):
                raise UnsafeGraphInput(
                    f"{label} node {node.id!r} is source code this generator "
                    f"cannot write ({cls._source_location(parameters)}). Edit it "
                    "in the file it came from."
                )

            if layer_type in {"Input", "Output", "Module"}:
                continue
            if node.type == "moduleNode" or op_id in _CUSTOM_OP_CLASS:
                require_layer_operation_syntax(
                    layer_type, label=f"{label} layer operation"
                )
                continue
            validate_layer_parameters(
                layer_type,
                parameters,
                label=f"{label} {node.id!r} parameters",
                allow_display_metadata=True,
            )

    @staticmethod
    def _custom_import(node: Node) -> Optional[Tuple[str, str]]:
        """``(module, Class)`` for a ``custom_code`` node, when it knows both.

        The class is the user's, so the only faithful way to write it into a new
        file is to import the real one. A node that cannot say where its class
        lives cannot be written at all — reimplementing it from a name would be
        a different model wearing the same word.
        """
        if _op_id(node) != "custom_code":
            return None
        params = getattr(node.data, "parameters", None) or {}
        module = params.get("_import_path")
        class_name = params.get("_class")
        if not isinstance(module, str) or not isinstance(class_name, str):
            return None
        if not class_name.isidentifier() or keyword.iskeyword(class_name):
            return None
        parts = module.split(".")
        if not parts or not all(p.isidentifier() and not keyword.iskeyword(p) for p in parts):
            return None
        return module, class_name

    @classmethod
    def _reconstructible_arguments(cls, node: Node) -> bool:
        """Whether a ``custom_code`` node knows how it was constructed.

        Either it carries the declaration it was read from (``_raw_call``, which
        the AST tier supplies and the projection merge keeps on the traced node)
        or it carries real constructor parameters, which is what a node authored
        on the canvas has. A traced node with neither knows the class and not the
        call — ``MyBlock(16)`` and ``MyBlock()`` are different models.
        """
        params = getattr(node.data, "parameters", None) or {}
        if isinstance(params.get("_raw_call"), str):
            return True
        return any(
            key != "name" and not str(key).startswith("_") for key in params
        )

    @staticmethod
    def _raw_call(node: Node) -> Optional[str]:
        """The source text a ``python`` node was read from, if it has one."""
        if _op_id(node) != "python":
            return None
        raw = (getattr(node.data, "parameters", None) or {}).get("_raw_call")
        return raw if isinstance(raw, str) and raw.strip() else None

    @classmethod
    def _writable_placeholder(cls, node: Node) -> bool:
        """Whether this `python`/`custom_code` node carries enough to be written."""
        return cls._custom_import(node) is not None or cls._raw_call(node) is not None

    #: Names a recorded statement may read that are not its input tensor.
    _STATEMENT_NAMESPACES = frozenset({"self", "torch", "nn", "F", "math", "np", "numpy"})

    @classmethod
    def _rebound(cls, raw: str, primary: str) -> str:
        """``raw`` with its input variable renamed to the one the graph feeds it.

        A ``python`` node carries the statement its author wrote, and that
        statement names THEIR variable — ``x.transpose(1, 2)``. The generated
        forward binds the same value under a different name (``out_fc``), so
        emitting the text unchanged would transpose whatever ``x`` happens to be
        at that point: a different model that still runs, which is the worst
        outcome this generator can produce. Renaming the one free name it reads is
        what keeps the line meaning what it meant.

        Only the unambiguous case is rewritten — exactly one free name that is not
        a module alias. A statement reading two tensors has no single input to
        rename, so it is emitted as written and stays the author's problem.
        """
        try:
            tree = ast.parse(raw, mode="eval")
        except (SyntaxError, ValueError):
            return raw
        free = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id not in cls._STATEMENT_NAMESPACES
        }
        if len(free) != 1:
            return raw
        name = free.pop()
        if name == primary:
            return raw
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == name:
                node.id = primary
        return ast.unparse(tree).strip()

    @staticmethod
    def _parses(expression: str) -> bool:
        """Whether a recorded statement is still Python. The text rides on the
        node through saves, AI turns and hand edits, so it is parsed before being
        interpolated: a file that does not compile is worse than a missing one."""
        try:
            ast.parse(expression, mode="eval")
        except (SyntaxError, ValueError):
            return False
        return True

    @staticmethod
    def _source_location(parameters: Dict[str, Any]) -> str:
        """`file:line` for a node that carries parser provenance, so a refusal
        points at the statement instead of at a node id nobody chose."""
        source_file = parameters.get("_source_file")
        lineno = parameters.get("_lineno")
        if isinstance(source_file, str) and isinstance(lineno, int):
            return f"{CodeGenerationService._safe_comment(source_file)}:{lineno}"
        if isinstance(lineno, int):
            return f"line {lineno}"
        raw_call = parameters.get("_raw_call")
        if isinstance(raw_call, str):
            return CodeGenerationService._safe_comment(raw_call)
        return "no source location recorded"

    @classmethod
    def _validate_flat_graph(cls, graph: Graph, *, class_name: str) -> None:
        require_python_identifier(class_name, label="model class name")
        cls._validate_nodes(graph.nodes, label="graph")
        node_ids = {node.id for node in graph.nodes}
        for edge in graph.edges:
            require_resource_identifier(edge.source, label="edge source")
            require_resource_identifier(edge.target, label="edge target")
            if edge.source not in node_ids or edge.target not in node_ids:
                raise UnsafeGraphInput("edge references an unknown node")

    @classmethod
    def _validate_hierarchical_graph(
        cls,
        graph: HierarchicalGraph,
        *,
        class_name: str,
    ) -> None:
        require_python_identifier(class_name, label="model class name")
        cls._validate_nodes(graph.root_nodes, label="root graph")
        root_ids = {node.id for node in graph.root_nodes}
        for edge in graph.root_edges:
            require_resource_identifier(edge.source, label="root edge source")
            require_resource_identifier(edge.target, label="root edge target")
            if edge.source not in root_ids or edge.target not in root_ids:
                raise UnsafeGraphInput("root edge references an unknown node")

        generated_classes: Dict[str, str] = {}
        for module_id, module in graph.modules.items():
            require_resource_identifier(module_id, label="module id")
            require_resource_identifier(module.module_id, label="module definition id")
            class_token = cls._module_id_to_class_name(module_id)
            collision = generated_classes.get(class_token)
            if collision is not None and collision != module_id:
                raise UnsafeGraphInput(
                    f"module ids {collision!r} and {module_id!r} generate the same "
                    "Python class name"
                )
            generated_classes[class_token] = module_id
            for name in module.parameters:
                require_python_identifier(name, label="module parameter key")
            for name in module.input_arg_names:
                require_python_identifier(name, label="module input argument")
            cls._validate_nodes(
                module.internal_nodes,
                label=f"module {module_id!r}",
            )
    
    def generate_hierarchical_code(
        self, 
        hierarchical_graph: HierarchicalGraph, 
        input_shape: List[int], 
        class_name: str = "GeneratedModel"
    ) -> str:
        """Generate modular PyTorch code from hierarchical graph"""

        self._validate_hierarchical_graph(
            hierarchical_graph,
            class_name=class_name,
        )
        
        # Extract modules and main graph
        modules = hierarchical_graph.modules
        root_nodes = hierarchical_graph.root_nodes
        root_edges = hierarchical_graph.root_edges
        
        # Generate code parts
        all_nodes = list(root_nodes)
        for m in modules.values():
            all_nodes.extend(m.internal_nodes)
        imports = self._generate_imports(all_nodes)
        custom_defs = self._custom_op_defs(all_nodes)
        imported_ids = {n.id for n in root_nodes if self._is_imported(n)}
        module_classes = self._generate_module_classes(modules, imported_ids)
        hollow_classes = self._hollow_module_classes(root_nodes, modules)
        main_class = self._generate_main_class(root_nodes, root_edges, class_name, modules)
        usage_example = self._generate_usage_example(class_name, input_shape, root_nodes, root_edges)

        # Combine all parts. The banner goes FIRST so it survives the file being
        # downloaded, pasted or diffed away from the app that produced it.
        unreproducible = self.unreproducible_modules(root_nodes, modules)
        code_parts = []
        if unreproducible:
            code_parts.append(self._unreproducible_header(unreproducible))
        code_parts.append(imports)
        if custom_defs:
            code_parts.append(custom_defs)
        if module_classes:
            code_parts.extend(module_classes)
        if hollow_classes:
            code_parts.extend(hollow_classes)
        # What to call forward with, said once where the tracer can read it. It sits
        # directly above the model: graph setup gives one outside every class to no
        # class when the helper modules above are models too.
        declaration = self.input_declaration(root_nodes, root_edges, input_shape)
        code_parts.append(declaration + "\n" + main_class if declaration else main_class)
        code_parts.append(usage_example)

        return "\n\n".join(code_parts)

    def _custom_op_defs(self, nodes: List[Node]) -> str:
        """Source for the shape-preserving nn.Module implementations of the
        custom ops actually used in `nodes` (deduped)."""
        seen, defs = set(), []
        for n in nodes:
            cls = _CUSTOM_OP_CLASS.get(_op_id(n))
            if cls and cls not in seen:
                seen.add(cls)
                defs.append(_CUSTOM_OP_SOURCE[cls])
        return "\n\n".join(defs)
    
    def generate_flat_code(
        self, 
        graph: Graph, 
        input_shape: List[int], 
        class_name: str = "GeneratedModel"
    ) -> str:
        """Generate simple PyTorch code from flat graph"""

        self._validate_flat_graph(graph, class_name=class_name)
        
        # Filter out input/output nodes
        layer_nodes = [n for n in graph.nodes 
                      if n.data.layerType.name not in ["Input", "Output"]]
        
        imports = self._generate_imports(graph.nodes)
        custom_defs = self._custom_op_defs(graph.nodes)
        main_class = self._generate_simple_class(graph, class_name)
        usage_example = self._generate_usage_example(class_name, input_shape, graph.nodes, graph.edges)

        parts = [imports]
        if custom_defs:
            parts.append(custom_defs)
        # Directly above the model, as in generate_hierarchical_code.
        declaration = self.input_declaration(graph.nodes, graph.edges, input_shape)
        parts.extend([declaration + "\n" + main_class if declaration else main_class, usage_example])
        return "\n\n".join(parts)
    

    def _emit_op(self, node: Node, in_vars: List[str]) -> str:
        """Return the forward-pass RHS expression for a single node given its
        input variable names. Shared by the main/flat forward and module-class
        forward so the op-emission logic lives in one place."""
        layer_type = node.data.layerType.name
        op_id = _op_id(node)
        primary = in_vars[0] if in_vars else "x"
        if node.data.parameters.get('_source_arguments'):
            raise UnsafeGraphInput(
                f"{node.id} has symbolic source arguments; run the original source or supply concrete arguments before export")

        # A `python` node is a statement the parser could not type, carried as its
        # own source text. It is regenerated VERBATIM: it is the user's line, and
        # rewriting it would be this generator inventing semantics for the one kind
        # of code it explicitly does not model (task.md M6). The text is parsed
        # first — it rides on the node through saves, AI turns and hand edits, and a
        # file that does not compile is worse than an incomplete one — so a
        # statement that is no longer Python refuses at run time instead, and
        # `unreproducible_modules` reports it so `complete` stays false.
        raw = self._raw_call(node)
        if raw is not None:
            if not self._parses(raw):
                return _REFUSAL_PREFIX + "NotImplementedError({!r})".format(
                    f"the statement recorded for {node.id} is not valid Python, so it "
                    f"could not be regenerated "
                    f"({self._source_location(node.data.parameters or {})})"
                )
            return self._rebound(raw, primary)

        # Merge / elementwise ops -> functional. Handles both the legacy
        # "torch.add" (by name) and F5-TTS's "Add"/"Concat" (by id).
        # _FUNCTIONAL_NAMESPACE ops carry a BARE canonical ("interpolate"): the
        # _LAYER_OPERATION guard in graph_input_safety allows at most ONE prefix
        # level, so "torch.nn.functional.interpolate" and "F.interpolate" are both
        # rejected on purpose — an op name must not be able to escape into an
        # arbitrary attribute path. Bare names therefore need naming explicitly
        # here instead of matching on the "torch." prefix.
        if op_id in _MERGE_OPS or op_id in _FUNCTIONAL_NAMESPACE or layer_type.startswith("torch."):
            spec = require_registered_layer_operation(layer_type)
            op = op_id if op_id in _MERGE_OPS else spec.canonical.removeprefix("torch.")
            if op in ("cat", "concat", "stack"):
                fn = "stack" if op == "stack" else "cat"
                dim = node.data.parameters.get("dim", -1)
                return (
                    f"torch.{fn}([{', '.join(in_vars) or primary}], "
                    f"dim={self._fmt_value(dim)})"
                )
            if op in ("add", "sub", "mul", "div", "matmul", "bmm"):
                params = node.data.parameters
                if params.get("_scalar_side") in ("left", "right"):
                    if len(in_vars) != 1 or type(params.get("other")) not in (int, float):
                        raise UnsafeGraphInput("Scalar arithmetic requires one tensor and a numeric literal operand")
                    scalar = self._fmt_value(params["other"])
                    left, right = (scalar, primary) if params["_scalar_side"] == "left" else (primary, scalar)
                    return f"torch.{op}({left}, {right})"
                func_call = primary
                for inv in in_vars[1:]:
                    func_call = f"torch.{op}({func_call}, {inv})"
                return func_call
            # other torch.* ops (flatten, reshape, narrow, …): pass real kwargs
            # only. Note the `v == ""` skip below — a param left at a palette
            # default of '' vanishes from the call, which is how the missing
            # positional args this guard now catches used to get through.
            params = dict(node.data.parameters)

            supplied = {k for k, v in params.items()
                        if v != "" and v is not None and not k.startswith("_") and k != "name"}
            missing = [k for k in _FUNCTIONAL_REQUIRED.get(op, ()) if k not in supplied]
            if missing:
                raise UnsafeGraphInput(
                    f"{layer_type} needs {' and '.join(missing)} before it can be "
                    f"generated — set {'them' if len(missing) > 1 else 'it'} in Properties"
                )
            either = _FUNCTIONAL_REQUIRED_ANY.get(op)
            if either and not (supplied & set(either)):
                raise UnsafeGraphInput(
                    f"{layer_type} needs either {' or '.join(either)} — set one in Properties"
                )

            if op in ("transpose", "permute"):
                # Neither op has a `dim` keyword at all, and graphs saved while the
                # palette exposed one still carry it beside the `dims` the user has
                # since filled in — emitting both is a TypeError.
                params.pop("dim", None)
                # The Properties field is free text, so the axes arrive as the comma
                # string the Input node's shape uses; torch wants real ints.
                dims = self._parse_shape(params.get("dims"))
                if dims is not None:
                    params["dims"] = dims
                # transpose is stored as a 2-list under `dims`, but torch's signature
                # is transpose(input, dim0, dim1) with no `dims` keyword — emitting it
                # verbatim produced `torch.transpose(x, dims=[1, 3])`, a TypeError.
                if op == "transpose" and dims is not None and len(dims) == 2:
                    params.pop("dims")
                    params["dim0"], params["dim1"] = dims[0], dims[1]

            kwargs_strs = []
            for k, v in params.items():
                if k == "name" or k.startswith("_") or v == "":
                    continue
                require_python_identifier(k, label="functional parameter key")
                kwargs_strs.append(f"{k}={self._fmt_parameter_value(k, v)}")
            kwarg_str = (", " + ", ".join(kwargs_strs)) if kwargs_strs else ""
            ns = "F" if op in _FUNCTIONAL_NAMESPACE else "torch"
            return f"{ns}.{op}({primary}{kwarg_str})"

        # Custom ops with emitted nn.Module implementations: pass every input
        # (e.g. AdaLayerNorm takes the sequence + conditioning).
        if op_id in _CUSTOM_OP_CLASS:
            layer_name = self._node_id_to_layer_name(node.id)
            return f"self.{layer_name}({', '.join(in_vars) or primary})"

        layer_name = self._node_id_to_layer_name(node.id)
        # A few transformer/attention modules need more positional inputs than
        # the single tensor the graph hands them; reuse `primary` for the missing
        # args (self-attention / decoder-reading-its-own-stack semantics).
        #
        # Matched on the op ID as well as the display name. The IDE places a canonical
        # block through `catalog.ts placeBlock`, which sets `layerType.name` to the raw
        # op string — "nn.MultiheadAttention", not "MultiheadAttention" — so none of
        # these rules fired and the generic call below emitted `self.attn(x)`, a file
        # that raises "MultiheadAttention.forward() missing 2 required positional
        # arguments". The web canvas was immune because it resolves ids through
        # `getLayerByType`, which supplies the bare class name.
        #
        # Both spellings rather than only the id: `layerType.id` is normalised by every
        # producer in the tree today, but the model allows an empty one, and a
        # hand-written or legacy graph carrying {"id": "", "name": "LSTM"} would then
        # lose the tuple unwrap silently — the exact failure this comment exists about.
        if layer_type in ("TransformerDecoderLayer", "TransformerDecoder", "Transformer") or op_id in (
            "transformerdecoderlayer",
            "transformerdecoder",
            "transformer",
        ):
            return f"self.{layer_name}({primary}, {primary})"
        if layer_type == "MultiheadAttention" or op_id == "multiheadattention":
            return f"self.{layer_name}({primary}, {primary}, {primary})[0]"
        # Recurrent layers return (output, h_n[/c_n]); take the output sequence so
        # the next layer receives a tensor, not a tuple.
        if layer_type in ("LSTM", "GRU", "RNN") or op_id in ("lstm", "gru", "rnn"):
            return f"self.{layer_name}({primary})[0]"
        if len(in_vars) <= 1:
            return f"self.{layer_name}({primary})"
        return f"self.{layer_name}({', '.join(in_vars)})"

    def _generate_forward_pass(
        self,
        nodes: List[Node],
        edges: List[Edge],
        root: bool = False,
        modules: Optional[Dict[str, ModuleDefinition]] = None,
    ) -> List[str]:
        """Generate branched topological forward pass."""
        # One forward arg per Input node. A single-input model keeps `x` (so
        # existing exports are byte-identical); a multi-input model (e.g. F5-TTS:
        # text / noisy mel / reference mel / timestep) gets a named arg each.
        input_node_objs = [n for n in nodes if n.data.layerType.name == "Input"]
        if len(input_node_objs) > 1:
            arg_names = [self._node_id_to_layer_name(n.id) for n in input_node_objs]
        else:
            arg_names = ["x"]
        primary_arg = arg_names[0]
        forward_lines = [
            "",
            f"{self.indent}def forward(self, {', '.join(arg_names)}):",
        ]

        computational_nodes = [n for n in nodes if n.data.layerType.name not in ["Input", "Output", "Module"] or n.data.layerType.name == "Module"]

        if not computational_nodes:
            forward_lines.append(f"{self.indent}{self.indent}return {primary_arg}")
            return forward_lines

        try:
            from core.layer_builder import topological_sort
            execution_order = topological_sort(nodes, edges)

            var_map = {}
            if len(input_node_objs) > 1:
                for n, a in zip(input_node_objs, arg_names):
                    var_map[n.id] = a
            else:
                for n in input_node_objs:
                    var_map[n.id] = "x"

            # Keep track of generated layer vars
            for node_id in execution_order:
                node = next((n for n in nodes if n.id == node_id), None)
                if not node:
                    continue

                layer_type = node.data.layerType.name
                if layer_type in ["Input", "Output"]:
                    continue

                # Find inputs
                incoming_edges = [e for e in edges if e.target == node_id]
                in_vars = []
                for e in incoming_edges:
                    if e.source in var_map:
                        in_vars.append(var_map[e.source])
                    else:
                        in_vars.append(primary_arg)  # Fallback

                if not in_vars:
                    in_vars = [primary_arg]

                out_var = f"out_{self._node_id_to_layer_name(node_id)}"
                var_map[node_id] = out_var

                # A modular repeated block: instantiate once as an nn.ModuleList
                # and run it in a loop, threading the primary (sequence) input and
                # holding any conditioning inputs (e.g. F5-TTS t_emb) constant.
                mdef = modules.get(node_id) if (modules and layer_type == "Module") else None
                if mdef is not None and getattr(mdef, "is_modular_block", False) and int(mdef.parameters.get("repeat_count", 1)) > 1:
                    layer_name = self._node_id_to_layer_name(node_id)
                    cond_set = set(mdef.conditioning_source_ids or [])
                    primary_vars = [var_map[e.source] for e in incoming_edges
                                    if e.source not in cond_set and e.source in var_map]
                    primary = primary_vars[0] if primary_vars else (in_vars[0] if in_vars else "x")
                    cond_vars = []
                    for cs in (mdef.conditioning_source_ids or []):
                        for e in incoming_edges:
                            if e.source == cs and e.source in var_map:
                                cond_vars.append(var_map[e.source])
                                break
                    extra = (", " + ", ".join(cond_vars)) if cond_vars else ""
                    forward_lines.append(f"{self.indent}{self.indent}{out_var} = {primary}")
                    forward_lines.append(f"{self.indent}{self.indent}for _blk in self.{layer_name}:")
                    forward_lines.append(f"{self.indent}{self.indent}{self.indent}{out_var} = _blk({out_var}{extra})")
                    continue

                emitted = self._emit_op(node, in_vars)
                if emitted.startswith(_REFUSAL_PREFIX):
                    forward_lines.append(f"{self.indent}{self.indent}{emitted}")
                    continue
                forward_lines.append(f"{self.indent}{self.indent}{out_var} = {emitted}")

            # Determine return variable. Follow Output nodes or last computed
            output_nodes = [n.id for n in nodes if n.data.layerType.name == "Output"]
            if output_nodes:
                out_incoming = [e.source for e in edges if e.target in output_nodes]
                returns = [var_map[s] for s in out_incoming if s in var_map]
                if returns:
                    if len(returns) == 1:
                        forward_lines.append(f"{self.indent}{self.indent}return {returns[0]}")
                    else:
                        forward_lines.append(f"{self.indent}{self.indent}return {', '.join(returns)}")
                else:
                    forward_lines.append(f"{self.indent}{self.indent}return {primary_arg}")
            else:
                last_var = var_map.get(execution_order[-1], primary_arg)
                forward_lines.append(f"{self.indent}{self.indent}return {last_var}")

        except UnsafeGraphInput:
            raise
        except Exception as e:
            forward_lines.append(
                f"{self.indent}{self.indent}# Error generating topology: "
                f"{self._safe_comment(e)}"
            )
            forward_lines.append(f"{self.indent}{self.indent}return {primary_arg}")
            
        return forward_lines

    def _generate_imports(self, nodes: Optional[List[Node]] = None) -> str:
        """Generate import statements, including one per user class on the canvas.

        A `custom_code` node names a class that lives in the user's own package
        (task.md M6). A generated file that instantiates it without importing it
        is a NameError printed underneath a correct-looking architecture, so the
        import is collected from the nodes rather than left to the reader.
        """
        custom = sorted(
            {pair for pair in (self._custom_import(n) for n in (nodes or [])) if pair}
        )
        extra = "".join(f"\nfrom {module} import {name}" for module, name in custom)
        return '''import torch
import torch.nn as nn
import torch.nn.functional as F
import math''' + extra
    
    def _generate_module_classes(self, modules: Dict[str, ModuleDefinition],
                                imported_ids: Optional[set] = None) -> List[str]:
        """Generate module class definitions"""
        module_classes = []
        
        for module_id, module_def in modules.items():
            class_name = self._module_id_to_class_name(module_id)
            module_class = self._generate_single_module_class(
                class_name, module_def, imported=module_id in (imported_ids or set())
            )
            module_classes.append(module_class)

        return module_classes

    def _hollow_module_classes(self, root_nodes: List[Node],
                               modules: Dict[str, ModuleDefinition]) -> List[str]:
        """Emit a passthrough nn.Module for each root module node that has NO
        ModuleDefinition and no exported internals (e.g. VAE's VAEEncoder /
        VAEDecoder / ReparameterizationTrick). Without this the main class would
        reference an undefined class name. A shape-preserving passthrough mirrors
        verify_model (which treats a module node with no internals as identity)
        and is clearly flagged so a user replaces it — far better than emitting a
        nonexistent nn.<Name> that AttributeErrors on instantiation.

        An IMPORTED module is different: the user did not author it, its real body lives in a
        library, and a passthrough would make the exported model silently compute something
        else entirely while still importing and running. Those stubs raise instead, so a wrong
        answer is impossible and the failure names what to supply.
        """
        defs, seen = [], set()
        for n in root_nodes:
            if getattr(n, "type", None) != "moduleNode" or n.id in (modules or {}):
                continue
            # A user class is imported, not stubbed: a stub of the same name would
            # shadow the real implementation the import just brought in (task.md M6).
            if self._custom_import(n) is not None:
                continue
            cls = self._module_id_to_class_name(n.id)
            if cls in seen:
                continue
            seen.add(cls)
            name = n.data.layerType.name
            params = getattr(n.data, "parameters", None) or {}
            if self._is_imported(n):
                origin = params.get("_class") or name
                path = params.get("_module_path") or n.id
                error_message = (
                    f"{cls} stands in for {origin!s} ({path!s}), which was not "
                    "exported. Supply the real implementation before running "
                    "this model."
                )
                defs.append(
                    f"class {cls}(nn.Module):\n"
                    f"{self.indent}\"\"\"Unreproduced imported module.\n"
                    f"\n"
                    f"{self.indent}Its implementation is not part of the graph and could not be\n"
                    f"{self.indent}generated. Import the real class or\n"
                    f"{self.indent}reimplement it, then delete this stub.\n"
                    f"\n"
                    f"{self.indent}It raises deliberately: a passthrough would let this model run and\n"
                    f"{self.indent}produce numbers from the wrong architecture.\"\"\"\n"
                    f"{self.indent}def __init__(self, **kwargs):\n"
                    f"{self.indent}{self.indent}super().__init__()\n"
                    f"\n"
                    f"{self.indent}def forward(self, *args, **kwargs):\n"
                    f"{self.indent}{self.indent}raise NotImplementedError("
                    f"{error_message!r})"
                )
                continue
            defs.append(
                f"class {cls}(nn.Module):\n"
                f"{self.indent}\"\"\"Placeholder module: no exported internals were\n"
                f"{self.indent}available, so this is a shape-preserving passthrough. Replace with\n"
                f"{self.indent}the real implementation.\"\"\"\n"
                f"{self.indent}def __init__(self, **kwargs):\n"
                f"{self.indent}{self.indent}super().__init__()\n"
                f"\n"
                f"{self.indent}def forward(self, x):\n"
                f"{self.indent}{self.indent}return x"
            )
        return defs
    
    def _generate_single_module_class(self, class_name: str, module_def: ModuleDefinition,
                                      imported: bool = False) -> str:
        """Generate a single module class"""
        
        # Filter computational layers
        layer_nodes = [n for n in module_def.internal_nodes 
                      if n.data.layerType.name not in ["Input", "Output", "Module"]]
        
        # Generate __init__ method
        init_lines = [
            f"class {class_name}(nn.Module):",
            f"{self.indent}def __init__(self, **kwargs):",
            f"{self.indent}{self.indent}super().__init__()"
        ]
        
        # Add configurable parameters from module definition (skip reserved /
        # display-only keys so a modular block's __init__ stays clean).
        _reserved = {"name", "repeat_count", "repeat_block_label"}
        config_params = {
            k: v for k, v in (module_def.parameters or {}).items()
            if k not in _reserved and not k.startswith("_")
        }
        if config_params:
            init_lines.append(f"{self.indent}{self.indent}# Module parameters")
            for param_name, default_value in config_params.items():
                require_python_identifier(param_name, label="module parameter key")
                init_lines.append(
                    f"{self.indent}{self.indent}self.{param_name} = "
                    f"kwargs.get({param_name!r}, {default_value!r})"
                )
            init_lines.append("")

        # Add layer definitions
        if layer_nodes:
            init_lines.append(f"{self.indent}{self.indent}# Layers")
            for node in layer_nodes:
                layer_line = self._generate_layer_definition(node)
                if layer_line:
                    init_lines.append(f"{self.indent}{self.indent}{layer_line}")

        # Generate forward method. An imported module with no body is handled BEFORE the
        # dispatch below, because both branches would otherwise emit a silent passthrough:
        # _generate_module_forward returns the first arg unchanged, and
        # _generate_forward_pass emits `return x`. For a module whose real implementation
        # lives in a library, that makes the exported model run and produce numbers from
        # the wrong architecture with nothing to signal it.
        if imported and not self._has_body(module_def):
            forward_lines = self._raising_forward()
        elif getattr(module_def, "is_modular_block", False):
            forward_lines = self._generate_module_forward(module_def, imported=imported)
        else:
            forward_lines = self._generate_forward_pass(module_def.internal_nodes, module_def.internal_edges)
        return "\n".join(init_lines + forward_lines)

    def _raising_forward(self) -> List[str]:
        """A forward that refuses rather than silently returning its input."""
        I = self.indent
        message = (
            "This module was imported and its implementation was not exported. "
            "Supply the real one before running this model."
        )
        return [
            "",
            f"{I}def forward(self, *args, **kwargs):",
            f"{I}{I}raise NotImplementedError(",
            f'{I}{I}{I}"{message}"',
            f"{I}{I})",
        ]

    def _generate_module_forward(self, module_def: ModuleDefinition,
                                 imported: bool = False) -> List[str]:
        """Forward pass for a modular block: signature = its input arg names
        (primary first, then conditioning), var map seeded from the external
        sources its boundary edges reference, returns the canonical exit node."""
        I = self.indent
        args = list(module_def.input_arg_names) or ["x"]
        lines = ["", f"{I}def forward(self, {', '.join(args)}):"]
        internal = module_def.internal_nodes
        comp = [n for n in internal if n.data.layerType.name not in ("Input", "Output")]
        if not comp:
            # Nothing to emit. For an authored graph a passthrough is a placeholder the user
            # fills in. For an IMPORTED module it would be a lie: the real body lives in a
            # library, and returning the input unchanged makes the exported model run and
            # produce numbers from the wrong architecture with nothing to signal it.
            if imported:
                message = (
                    "This module was imported and its implementation was not exported. "
                    "Supply the real one before running this model."
                )
                lines.append(f"{I}{I}raise NotImplementedError(")
                lines.append(f'{I}{I}{I}"{message}"')
                lines.append(f"{I}{I})")
                return lines
            lines.append(f"{I}{I}return {args[0]}")
            return lines
        try:
            from core.layer_builder import topological_sort
            order = topological_sort(internal, module_def.internal_edges)
            var_map = {}
            for src, arg in zip(module_def.input_node_ids, args):
                var_map[src] = arg
            all_edges = list(module_def.internal_edges) + list(module_def.boundary_edges)
            primary_src = module_def.input_node_ids[0] if module_def.input_node_ids else None
            for node_id in order:
                node = next((n for n in internal if n.id == node_id), None)
                if not node or node.data.layerType.name in ("Input", "Output"):
                    continue
                incoming = [e for e in all_edges if e.target == node_id]
                # The primary (sequence) input must be arg 0 — a boundary edge
                # from the primary source can otherwise land after internal edges
                # (e.g. AdaLayerNorm's modulation), flipping which tensor is normed.
                incoming.sort(key=lambda e: 0 if e.source == primary_src else 1)
                in_vars = [var_map[e.source] for e in incoming if e.source in var_map]
                if not in_vars:
                    in_vars = [args[0]]
                out_var = f"out_{self._node_id_to_layer_name(node_id)}"
                var_map[node_id] = out_var
                emitted = self._emit_op(node, in_vars)
                if emitted.startswith(_REFUSAL_PREFIX):
                    lines.append(f"{I}{I}{emitted}")
                    continue
                lines.append(f"{I}{I}{out_var} = {emitted}")
            out_id = module_def.output_node_id or (order[-1] if order else None)
            lines.append(f"{I}{I}return {var_map.get(out_id, args[0])}")
        except UnsafeGraphInput:
            raise
        except Exception as e:
            lines.append(
                f"{I}{I}# Error generating module topology: {self._safe_comment(e)}"
            )
            lines.append(f"{I}{I}return {args[0]}")
        return lines
    
    def _generate_main_class(
        self, 
        root_nodes: List[Node], 
        root_edges: List[Edge], 
        class_name: str,
        modules: Dict[str, ModuleDefinition]
    ) -> str:
        """Generate main model class"""
        
        # Filter computational nodes
        layer_nodes = [n for n in root_nodes 
                      if n.data.layerType.name not in ["Input", "Output"]]
        
        # Generate __init__ method
        init_lines = [
            f"class {class_name}(nn.Module):",
            f"{self.indent}def __init__(self):",
            f"{self.indent}{self.indent}super().__init__()"
        ]
        
        if layer_nodes:
            init_lines.append(f"{self.indent}{self.indent}# Model layers")
            for node in layer_nodes:
                layer_line = self._generate_layer_definition(node, modules)
                if layer_line:
                    init_lines.append(f"{self.indent}{self.indent}{layer_line}")

        # Generate forward method
        forward_lines = self._generate_forward_pass(root_nodes, root_edges, root=True, modules=modules)
        return "\n".join(init_lines + forward_lines)
    
    def _generate_simple_class(self, graph: Graph, class_name: str) -> str:
        """Generate simple sequential model class"""
        
        
        layer_nodes = [n for n in graph.nodes if n.data.layerType.name not in ["Input", "Output"]]
        
        init_lines = [
            f"class {class_name}(nn.Module):",
            f"{self.indent}def __init__(self):",
            f"{self.indent}{self.indent}super().__init__()"
        ]
        
        if layer_nodes:
            init_lines.append(f"{self.indent}{self.indent}# Model layers")
            for node in layer_nodes:
                if not node.data.layerType.name.startswith('torch.'):
                    layer_line = self._generate_layer_definition(node)
                    if layer_line:
                        init_lines.append(f"{self.indent}{self.indent}{layer_line}")

        forward_lines = self._generate_forward_pass(graph.nodes, graph.edges)
        return "\n".join(init_lines + forward_lines)
    
    def _generate_layer_definition(self, node: Node, modules: Optional[Dict[str, ModuleDefinition]] = None) -> Optional[str]:
        """Generate the __init__ layer line, or None for ops that are emitted
        functionally in the forward pass (merge/elementwise + torch.* ops)."""
        layer_type = node.data.layerType.name
        params = node.data.parameters
        layer_name = self._node_id_to_layer_name(node.id)
        op_id = _op_id(node)

        # Functional ops have no constructor — they're emitted in forward().
        # _FUNCTIONAL_NAMESPACE ops carry a BARE canonical ("interpolate"): the
        # _LAYER_OPERATION guard in graph_input_safety allows at most ONE prefix
        # level, so "torch.nn.functional.interpolate" and "F.interpolate" are both
        # rejected on purpose — an op name must not be able to escape into an
        # arbitrary attribute path. Bare names therefore need naming explicitly
        # here instead of matching on the "torch." prefix.
        if op_id in _MERGE_OPS or op_id in _FUNCTIONAL_NAMESPACE or layer_type.startswith("torch."):
            return None

        # The user's own nn.Module (task.md M6). The real class is IMPORTED — see
        # _generate_imports — never redefined from its name, which would be a
        # different model wearing the same word. This must stand before the module
        # branch below: a custom class whose children are also on the canvas is
        # still that class, and emitting a generated copy of it beside the import
        # would give the file two classes claiming to be the same thing.
        custom = self._custom_import(node)
        if custom is not None:
            # The declaration the user actually wrote, when the node carries it.
            # `MyBlock(16)` and `MyBlock()` are different models, and the live
            # object cannot be asked which arguments built it — a constructor
            # argument that is not also stored as an attribute is gone by the time
            # the tracer sees the module. So the source text wins whenever it is
            # there, and `**params` is the fallback for a node authored on canvas.
            declared = (params or {}).get("_raw_call")
            if (
                isinstance(declared, str)
                and declared.startswith(custom[1] + "(")
                and self._parses(declared)
            ):
                return f"self.{layer_name} = {declared}"
            argument_strings = []
            for key, value in (params or {}).items():
                if key == "name" or key.startswith("_"):
                    continue
                require_python_identifier(key, label="custom layer parameter key")
                argument_strings.append(f"{key}={self._fmt_value(value)}")
            return f"self.{layer_name} = {custom[1]}({', '.join(argument_strings)})"

        # A `python` node is a statement, not a layer: it has no constructor and is
        # emitted in the forward pass by `_emit_op`.
        if op_id == "python":
            return None

        # Custom ops we ship a shape-preserving implementation for.
        if op_id in _CUSTOM_OP_CLASS:
            return f"self.{layer_name} = {_CUSTOM_OP_CLASS[op_id]}()"

        # Module / container nodes. Detect by node.type=='moduleNode' — an
        # instantiated module carries a SEMANTIC layerType.name (Generator,
        # VAEEncoder, ResNetLayer, InceptionStack, …), not "Module", so keying
        # only on the name would fall through and emit a nonexistent nn.<Name>.
        # These must reference a LOCALLY-defined class, never nn.<name>.
        if node.type == "moduleNode" or layer_type == "Module":
            module_class_name = self._module_id_to_class_name(node.id)
            mdef = (modules or {}).get(node.id)
            if mdef is not None:
                if getattr(mdef, "is_modular_block", False):
                    # A repeated block: instantiate ×N as an nn.ModuleList so the
                    # forward loop can reuse one class instead of inlining N copies.
                    n_rep = int(mdef.parameters.get("repeat_count", 1))
                    if n_rep > 1:
                        return f"self.{layer_name} = nn.ModuleList([{module_class_name}() for _ in range({n_rep})])"
                    return f"self.{layer_name} = {module_class_name}()"
                # Pass module parameters to the generated nested class.
                module_params = mdef.parameters
                if module_params:
                    param_parts = []
                    for key, value in module_params.items():
                        require_python_identifier(key, label="module parameter key")
                        param_parts.append(f"{key}={value!r}")
                    param_str = ", ".join(param_parts)
                    return f"self.{layer_name} = {module_class_name}({param_str})"
                return f"self.{layer_name} = {module_class_name}()"
            # No ModuleDefinition: reference the passthrough wrapper class emitted
            # by _hollow_module_classes (mirrors verify_model, which treats a
            # module node with no internals as an identity). Emitting nn.<name>
            # here would AttributeError at instantiation.
            return f"self.{layer_name} = {module_class_name}()"

        # Regular PyTorch layer. Skip display-only / non-constructor keys
        # (name + `_`-prefixed UI hints) so the generated layer is valid.
        param_strs = []
        spec = require_registered_layer_operation(layer_type)
        validate_layer_parameters(
            layer_type,
            params,
            label=f"layer {node.id!r} parameters",
            allow_display_metadata=True,
        )
        for key, value in params.items():
            if key == "name" or key.startswith("_"):
                continue
            param_strs.append(f"{key}={self._fmt_value(value)}")

        param_str = ", ".join(param_strs)
        class_name = spec.canonical.removeprefix("nn.")
        return f"self.{layer_name} = nn.{class_name}({param_str})"
    
    @staticmethod
    def _parse_shape(raw: Any) -> Optional[List[int]]:
        # Accept both a list/tuple (the code parser emits shapes this way) and a
        # comma-separated string (templates emit strings) so the generated usage
        # example is sized from the input node's real shape in both paths.
        if isinstance(raw, (list, tuple)):
            try:
                dims = [int(x) for x in raw]
            except (ValueError, TypeError):
                return None
            return dims or None
        if not isinstance(raw, str) or not raw.strip():
            return None
        try:
            dims = [int(float(p)) for p in raw.split(",") if p.strip()]
        except ValueError:
            return None
        return dims or None

    @staticmethod
    def _fmt_value(value: Any) -> str:
        """Format a constructor arg as valid Python. Numeric-looking strings
        pass through as bare numbers (so e.g. Linear(in_features=512), not '512');
        everything else is repr()'d so strings/lists/dicts/bools/None are always
        syntactically valid (fixes str values whose quotes weren't escaped)."""
        # A typed tuple field is stored as the string the user typed, so a kernel
        # of "3,3" was emitted verbatim as kernel_size='3,3' — a file torch cannot
        # build. Read it with the same reading core.layer_builder uses to build the
        # layer, so the generated file rebuilds the model that was verified.
        normalized = normalized_parameter(value)
        if isinstance(normalized, tuple):
            return repr(normalized)
        if isinstance(value, str) and re.fullmatch(
            r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",
            value,
        ):
            return value
        return repr(value)

    @classmethod
    def _fmt_parameter_value(cls, name: str, value: Any) -> str:
        """Format legacy comma-separated ``dim`` values without raw source."""

        if name == "dim" and isinstance(value, str) and "," in value:
            parts = [part.strip() for part in value.split(",")]
            if parts and all(re.fullmatch(r"[+-]?\d+", part) for part in parts):
                return repr(tuple(int(part) for part in parts))
        return cls._fmt_value(value)

    def sample_inputs(self, nodes: List[Node], edges: List[Edge], default_shape: List[int]) -> Tuple[List[str], List[str]]:
        """Build runnable example inputs — one tensor per Input node, sized from
        the node's declared shape, integer (randint) when it feeds an Embedding.
        Returns (construction_lines, call_arg_names). Single-input → ['x']."""
        inputs = [n for n in nodes if n.data.layerType.name == "Input"]
        by_id = {n.id: n for n in nodes}
        input_ids = {n.id for n in inputs}
        emb_vocab: Dict[str, int] = {}
        for e in edges:
            if e.source in input_ids:
                t = by_id.get(e.target)
                if t is not None and (_op_id(t) == "embedding" or t.data.layerType.name == "Embedding"):
                    emb_vocab[e.source] = int(t.data.parameters.get("num_embeddings", 1000) or 1000)
        # Single-input model: keep the arg named `x`, but size it from the input
        # node's real declared shape and use integer indices (randint) when the
        # first layer is an Embedding (torch.randn floats crash nn.Embedding).
        if len(inputs) <= 1:
            if not inputs:
                # A canonical block has no Input node, so the entry layer decides what
                # the example tensor must be. An Embedding wants integer indices, and
                # `torch.randn` floats crash it with an error about dtype rather than
                # about the input — the same trap the Input-node branch below already
                # avoids, which a fragment never reached.
                targets = {e.target for e in edges}
                entry = next((n for n in nodes if n.id not in targets), None)
                if entry is not None and (
                    _op_id(entry) == "embedding" or entry.data.layerType.name == "Embedding"
                ):
                    vocab = int(entry.data.parameters.get("num_embeddings", 1000) or 1000)
                    return [f"x = torch.randint(0, {vocab}, {tuple(default_shape)})"], ["x"]
                return [f"x = torch.randn({list(default_shape)})"], ["x"]
            n = inputs[0]
            shape = self._parse_shape(n.data.parameters.get("shape")) or list(default_shape)
            if n.id in emb_vocab:
                return [f"x = torch.randint(0, {emb_vocab[n.id]}, {tuple(shape)})"], ["x"]
            return [f"x = torch.randn({list(shape)})"], ["x"]
        lines, args = [], []
        for n in inputs:
            arg = self._node_id_to_layer_name(n.id)
            args.append(arg)
            shape = self._parse_shape(n.data.parameters.get("shape")) or list(default_shape)
            if n.id in emb_vocab:
                lines.append(f"{arg} = torch.randint(0, {emb_vocab[n.id]}, {tuple(shape)})")
            else:
                lines.append(f"{arg} = torch.randn({list(shape)})")
        return lines, args

    def input_declaration(self, nodes: List[Node], edges: List[Edge],
                          default_shape: List[int]) -> str:
        """`# tensorviz: input=(...)` — what to call this model's forward with.

        The generator already works this out for the usage example: one tensor per
        Input node, sized from its declared shape, `randint` when it feeds an
        Embedding. It just never said so anywhere the TRACER could read, so the
        worker fell back to searching a candidate pool — and that pool caps at three
        arguments, so F5-TTS (four inputs: text, noisy, cond, timestep) died with
        "forward() missing 1 required positional argument: 'in_time'" and every one
        of its 290 nodes was then reported "was not reached".

        The web app never hit this because it never RUNS the model; it infers shapes
        analytically, where a four-input graph is unremarkable. Only the IDE traces.

        Emitting the declaration removes the guess entirely: `parse_input_spec`
        already accepts a Python expression precisely so `torch.randint(...)` can be
        said, which is the one thing a shape alone cannot express and the exact thing
        an Embedding needs.
        """
        lines, _args = self.sample_inputs(nodes, edges, default_shape)
        exprs = [line.split("=", 1)[1].strip() for line in lines if "=" in line]
        if not exprs:
            return ""

        # Prefer the PLAIN comma form, because two tiers read this comment and only
        # one of them can evaluate Python. The trace worker imports torch and accepts
        # either; the parser deliberately does not (`code_parser.declared_input_shape`
        # reads the plain form only, "which is exactly what this tier promises not to
        # import"). Emitting the expression form unconditionally therefore made every
        # generated file's declaration invisible to the parser, which then fell back to
        # its [1, 3, 224, 224] default — and the editor reported "input shape is
        # assumed, not stated — not checked", so the model was never validated at all.
        # A green tick covering nothing is worse than a red one.
        plain = [_PLAIN_RANDN.fullmatch(expr) for expr in exprs]
        if all(plain):
            shapes = [
                ",".join(part.strip() for part in match.group("dims").split(","))
                for match in plain
            ]
            return f"# tensorviz: input={';'.join(shapes)}"
        # Something here needs an expression — `torch.randint` for an Embedding, which
        # a shape alone cannot say. The worker still gets it exactly right; the parser
        # falls back to its own inference, which is the honest trade.
        return f"# tensorviz: input=({', '.join(exprs)},)"

    def _generate_usage_example(self, class_name: str, input_shape: List[int],
                                nodes: Optional[List[Node]] = None, edges: Optional[List[Edge]] = None) -> str:
        """Generate a runnable usage example (multi-input aware)."""
        cons, args = self.sample_inputs(nodes or [], edges or [], input_shape)
        cons_block = ("\n" + self.indent).join(cons)
        call = ", ".join(args)
        return f'''# Example usage:
if __name__ == "__main__":
    model = {class_name}()
    model.eval()  # inference mode: BatchNorm/Dropout use running stats
    {cons_block}
    output = model({call})
    out_shape = tuple(output.shape) if torch.is_tensor(output) else type(output)
    print("Output shape:", out_shape)

    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {{total_params:,}}")
    print(f"Trainable parameters: {{trainable_params:,}}")'''
    
    @staticmethod
    def _node_id_to_layer_name(node_id: str) -> str:
        """Convert node ID to valid Python variable name"""
        require_resource_identifier(node_id, label="node id")
        safe_name = re.sub(r"[^0-9A-Za-z_]", "_", node_id)
        # Ensure it doesn't start with a number
        if safe_name and (safe_name[0].isdigit() or keyword.iskeyword(safe_name)):
            safe_name = f"layer_{safe_name}"
        return safe_name or "layer"
    
    @staticmethod
    def _module_id_to_class_name(module_id: str) -> str:
        """Convert a module id to a valid, COLLISION-FREE Python class name.

        The previous version split only on ``-``, ``_`` and space and then dropped any part
        that failed ``str.isalnum()``. Imported module ids are dotted paths, so
        ``down_blocks.0`` split to ``['down', 'blocks.0']``, ``'blocks.0'`` was discarded for
        containing a dot, and the name collapsed to ``Down`` — as did ``down_blocks.1`` and
        every other sibling. An imported UNet emitted several different modules all called
        ``Down``, so the generated file defined the same class repeatedly and the later
        definition silently won.

        Splitting on every non-alphanumeric run keeps the disambiguating segments
        (``DownBlocks0``, ``DownBlocks1``) and preserves internal capitalisation, so a class
        named ``DiTBlock`` stays ``DiTBlock`` instead of becoming ``Ditblock``.
        """
        require_resource_identifier(module_id, label="module id")
        parts = [p for p in re.split(r"[^0-9A-Za-z]+", module_id) if p]
        class_name = "".join(p[:1].upper() + p[1:] for p in parts)

        # Ensure it's a valid class name
        if not class_name or class_name[0].isdigit() or keyword.iskeyword(class_name):
            class_name = f"Module{class_name}"

        return class_name or "CustomModule"

    # ── Honesty about what cannot be reproduced ──────────────────────────────
    @staticmethod
    def _has_body(module_def: ModuleDefinition) -> bool:
        """Whether a module definition has anything to emit a forward pass from."""
        return any(
            n.data.layerType.name not in ("Input", "Output")
            for n in (module_def.internal_nodes or [])
        )

    @staticmethod
    def _is_imported(node: Node) -> bool:
        """An ingested node carries importer provenance under `_`-prefixed params."""
        params = getattr(node.data, "parameters", None) or {}
        return "_module_path" in params or "_class" in params

    def unreproducible_modules(
        self, root_nodes: List[Node], modules: Optional[Dict[str, ModuleDefinition]] = None
    ) -> List[Dict[str, Any]]:
        """Module nodes the generated code CANNOT faithfully reproduce.

        A module node with no exported internals has no body to emit. For a hand-authored
        graph that is a placeholder the user is expected to fill in. For an *imported* graph
        it is a library-private composite (``ResnetBlock2D``, ``Timesteps``,
        ``LlamaRotaryEmbedding``) whose real implementation lives in a package — the user did
        not write it and has no way to notice it is missing from the export.

        A ``python`` node whose recorded statement is no longer valid Python is the
        same failure in a different shape (task.md M6): the one thing that could
        have regenerated that line does not parse, so the file is missing it. It is
        reported here rather than refused so the caller still gets the rest of the
        model with ``complete: false`` beside it.
        """
        out: List[Dict[str, Any]] = []
        for node in root_nodes:
            raw = self._raw_call(node)
            if raw is not None and not self._parses(raw):
                params = getattr(node.data, "parameters", None) or {}
                out.append({
                    "node_id": node.id,
                    "class_name": self._node_id_to_layer_name(node.id),
                    "op": node.data.layerType.name,
                    "source_class": None,
                    "module_path": params.get("_source_file"),
                    "imported": False,
                    "reason": (
                        "the statement recorded for this node is not valid Python, so it "
                        "could not be regenerated"
                    ),
                })
                continue
            if getattr(node, "type", None) != "moduleNode":
                continue
            # The user's own class: its body is not in the graph and does not need
            # to be, because the generated file imports the real one — PROVIDED the
            # arguments it was built with are recoverable. A node that came from a
            # trace and carries neither the declaration it was read from nor any
            # constructor parameter would regenerate as `MyBlock()`, which is a
            # different model that imports and runs. Report it instead.
            if self._custom_import(node) is not None:
                params = getattr(node.data, "parameters", None) or {}
                if self._reconstructible_arguments(node):
                    continue
                out.append({
                    "node_id": node.id,
                    "class_name": self._module_id_to_class_name(node.id),
                    "op": node.data.layerType.name,
                    "source_class": params.get("_class"),
                    "module_path": params.get("_import_path"),
                    "imported": True,
                    "reason": (
                        "the arguments this class was constructed with are not on the "
                        "node, so it would be regenerated with its defaults"
                    ),
                })
                continue
            # A ModuleDefinition can exist and still be EMPTY. `_generate_module_forward`
            # emits `return <first arg>` when a definition has no computational internal
            # nodes, so "has a definition" is not the same as "can be reproduced" — that
            # gap is what let an imported UNet report a complete export while emitting 38
            # silent passthroughs.
            definition = (modules or {}).get(node.id)
            if definition is not None and self._has_body(definition):
                continue
            params = getattr(node.data, "parameters", None) or {}
            out.append({
                "node_id": node.id,
                "class_name": self._module_id_to_class_name(node.id),
                "op": node.data.layerType.name,
                "source_class": params.get("_class"),
                "module_path": params.get("_module_path"),
                "imported": self._is_imported(node),
                "reason": "no exported internals: this module's body is not part of the graph",
            })
        return out

    def _unreproducible_header(self, entries: List[Dict[str, Any]]) -> str:
        """A banner in the generated file itself.

        The file is the artifact that leaves the product — it gets downloaded, pasted and
        run somewhere this warning cannot follow it, so the warning has to be IN it.
        """
        lines = [
            "# " + "=" * 74,
            "# INCOMPLETE EXPORT — this file does not reproduce the original model.",
            "# " + "=" * 74,
            "# The modules below had no exported internals, so no body could be generated:",
        ]
        for e in entries:
            origin = self._safe_comment(e.get("source_class") or e["op"])
            where = (
                f" ({self._safe_comment(e['module_path'])})"
                if e.get("module_path")
                else ""
            )
            lines.append(f"#   - {e['class_name']}: {origin}{where}")
        lines.append("#")
        if any(e["imported"] for e in entries):
            lines.append(
                "# These came from an imported model, so their real implementations live in the"
            )
            lines.append(
                "# source library. Import them from there, or reimplement them, before using this"
            )
            lines.append(
                "# file. The stubs below RAISE rather than return their input, so that running"
            )
            lines.append(
                "# this model cannot silently produce numbers from the wrong architecture."
            )
        else:
            lines.append(
                "# They are emitted as shape-preserving passthroughs for you to fill in."
            )
        lines.append("# " + "=" * 74)
        return "\n".join(lines)
