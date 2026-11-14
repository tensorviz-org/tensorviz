"""Code parser service for converting PyTorch code to graph representation"""
import ast
import copy
import re
from typing import Dict, List, Any, NamedTuple, Optional, Tuple, Set
from models.graph_models import Graph, Node, Edge, LayerData, LayerType, CodeParseRequest
from core.op_registry import spec_for
from utils.residual_roles import annotate_residual_edges
from services.ingest.input_declarations import applicable
from services.torch_imports import canonical_name, torch_imports
from services.validation.signatures import forward_declaration
from utils.source_flow import Flow, UNKNOWN, bind_helper, local_methods
from utils.source_tensor_layout import TORCH_LAYOUT_TARGETS, tensor_layout_call


# Sentinel marking an AST argument whose literal value could not be extracted.
_UNSET = object()


# torch.nn.functional activation name (trailing '_' of in-place variants stripped)
# -> the equivalent torch.nn module class. Only pure tensor-in/tensor-out
# activations are listed so the synthesized node re-exports as a valid, runnable
# nn.<Class>() (e.g. F.relu -> nn.ReLU). Softmax/LogSoftmax additionally carry a
# literal `dim`. Ops needing an external parameter (F.prelu's weight) are omitted.
_F_ACTIVATIONS: Dict[str, str] = {
    "relu": "ReLU", "relu6": "ReLU6", "leaky_relu": "LeakyReLU", "elu": "ELU",
    "selu": "SELU", "celu": "CELU", "gelu": "GELU", "silu": "SiLU", "mish": "Mish",
    "sigmoid": "Sigmoid", "tanh": "Tanh", "softplus": "Softplus", "softsign": "Softsign",
    "hardswish": "Hardswish", "hardsigmoid": "Hardsigmoid", "hardtanh": "Hardtanh",
    "tanhshrink": "Tanhshrink", "softmax": "Softmax", "log_softmax": "LogSoftmax",
}

# RNN-family classes take (*args, **kwargs) in __init__, so inspect.signature
# yields no positional-parameter names — positional constructor args (e.g.
# nn.LSTM(10, 20)) would otherwise be captured as arg_0/arg_1 and break re-export
# + shape inference. Supply the real constructor order for these.
_ARGS_KWARGS_POSITIONAL: Dict[str, List[str]] = {
    "LSTM": ["input_size", "hidden_size", "num_layers", "bias", "batch_first", "dropout", "bidirectional", "proj_size"],
    "GRU": ["input_size", "hidden_size", "num_layers", "bias", "batch_first", "dropout", "bidirectional"],
    "RNN": ["input_size", "hidden_size", "num_layers", "nonlinearity", "bias", "batch_first", "dropout", "bidirectional"],
}


# Supported binary arithmetic, including a tensor with a known literal operand.
# Other operators remain outside this source projection's arithmetic coverage.
_BINOP_MERGE: Dict[type, str] = {
    ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div", ast.MatMult: "matmul",
}

# torch.<fn> calls that merge N tensors into one node. cat/stack additionally
# carry the axis they join on, which the canvas and the code generator both need.
_MERGE_CALLS = ("cat", "stack", "add", "mul", "matmul", "bmm")


#: The graph's two synthetic terminals, as they are named while forward() is being
#: read. NOT "input"/"output": that namespace belongs to the model's own attribute
#: names, and `self.output = nn.Sigmoid()` is ordinary PyTorch. Sharing one spelling
#: made the user's layer and the terminal the same node, so the layer was dropped
#: from the drawing while the toolbar went on counting it. `<` and `>` cannot occur
#: in a Python identifier, so these cannot collide with any attribute.
ENTRY_MARK = "<input>"
EXIT_MARK = "<output>"


def _free_id(preferred: str, taken: Set[str]) -> str:
    """`preferred` if nothing owns it, else the nearest free dunder spelling.

    Terminals yield rather than the user's layer: a layer id is the attribute that
    declared it, and renaming that would break every consumer that maps a node back
    to the source line it came from."""
    if preferred not in taken:
        return preferred
    candidate = f"__{preferred}__"
    suffix = 2
    while candidate in taken:
        candidate = f"__{preferred}_{suffix}__"
        suffix += 1
    return candidate


def _dedupe(items: List[str]) -> List[str]:
    """Order-preserving de-duplication (dict keeps insertion order)."""
    return list(dict.fromkeys(items))


def _dedupe_pairs(pairs: List[Tuple[str, str]], value_targets: Optional[Set[str]] = None) -> List[Tuple[str, str]]:
    """Collapse repeated declaration calls, but retain an operation's value uses."""
    seen = set()
    result = []
    for pair in pairs:
        if pair not in seen or pair[1] in (value_targets or ()):
            result.append(pair)
        seen.add(pair)
    return result


def _is_self_super_call(node: ast.AST, current_class: Optional[str] = None) -> bool:
    """Whether ``node`` is the ``super()`` call of a method-level super expression.

    Both supported spellings name the next class in the MRO from within an
    instance method: the zero-argument ``super()``, and the explicit
    ``super(CurrentClass, self)`` whose second argument is ``self``. The
    When ``current_class`` is supplied, the explicit form must name that exact
    class. Accepting ``super(OtherClass, self)`` as this class's next MRO entry
    would silently compose or bridge the wrong implementation.
    """
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "super"
        and not node.keywords
    ):
        return False
    if not node.args:
        return True
    return (
        len(node.args) == 2
        and isinstance(node.args[0], ast.Name)
        and isinstance(node.args[1], ast.Name)
        and node.args[1].id == "self"
        and (current_class is None or node.args[0].id == current_class)
    )



def declared_input_shape(
    source: str, cls: Optional[str] = None, *, tree: Optional[ast.AST] = None, proven=None
) -> Optional[List[int]]:
    """The shape the ``# tensorviz: input=`` comment for class ``cls`` declares, or ``None``.

    ``# tensorviz: input=2,8`` (or ``2,8;2,8`` for two arguments) is the escape hatch
    for a forward whose input cannot be guessed, and the only one that lives in the
    user's own source. Which class it belongs to is decided in
    ``services.ingest.input_declarations``, the one rule setup, the capture worker and
    the project graph also follow: in a file with several model classes, only a
    declaration directly above or inside ``cls`` is read. ``tree`` and ``proven`` are
    passed through to it.

    Only the plain comma-separated form is read, and only its first argument. This
    tier draws a single Input node, and the expression form the worker also accepts
    (``torch.randint(0, 50257, (1, 16))``) needs torch to evaluate, which is exactly
    what this tier promises not to import. A declaration it cannot read is left to
    the inference below rather than guessed at.
    """
    own, _stray = applicable(source or "", cls, tree=tree, proven=proven)
    if not own:
        return None
    first = own[0][0].split(";")[0].strip()
    if not re.fullmatch(r"\d+(\s*,\s*\d+)*", first):
        return None
    dims = [int(part) for part in first.split(",")]
    return dims or None


#: ``# tensorviz: init=Net(hidden=8, depth=2)`` on or above the class line. The
#: per-class escape hatch for a constructor that cannot be called with no arguments,
#: and the companion to ``input=``: that one says what the model is fed, this one says
#: how the model is built.
#:
#: Duplicated in ``services/ingest/service.py`` because the worker cannot import this
#: module without the user's own ``models`` package shadowing the backend's.
#: ``test_ide_init_comment`` asserts the two cannot drift.
INIT_COMMENT = re.compile(r"#\s*tensorviz:\s*init\s*=\s*(?P<value>[^\n]+)")


class ConfigCall(NamedTuple):
    """``ModelConfig()`` written as the VALUE of an init declaration argument.

    The commonest real ``__init__`` takes a config OBJECT, and no literal describes
    one. This tier cannot build it — it promises not to import the user's module, and
    the class is defined there — so the call is carried as a value rather than
    evaluated, and the ingest tier (which is about to import that module anyway)
    builds it. What matters here is that the declaration is not REFUSED: a user whose
    correct line is squiggled as an error has no way to discover it works.

    Structurally equal to ``services.ingest.service.ConfigCall``, which is a separate
    class for the same reason ``INIT_COMMENT`` is a separate pattern. Tuples compare
    by value, so ``test_ide_init_comment``'s drift guard still holds the two to one
    answer without either importing the other.
    """

    name: str
    kwargs: Dict[str, Any]


class InitDeclaration(NamedTuple):
    """One ``# tensorviz: init=`` line, read.

    ``error`` is what separates "the user said nothing" (no declaration at all, so
    ``declared_init`` returns ``None``) from "the user said something we will not
    act on" — which has to reach the editor as a diagnostic rather than be dropped,
    or the user edits a line that is silently ignored.
    """

    cls: str
    overrides: Dict[str, Any]
    error: Optional[str]


def _read_config_call(node: ast.Call, argument: str) -> Tuple[Optional[ConfigCall], Optional[str]]:
    """``ModelConfig(width=8)`` -> a ``ConfigCall``; see the ingest tier's twin.

    One call, on a bare name, with literal keyword arguments. Not a general
    expression evaluator: a dotted name, a positional argument or a call nested
    inside this one is refused with the reason, because each of those is a step
    towards evaluating arbitrary source to decide how to run the user's source.
    """
    if not isinstance(node.func, ast.Name):
        return None, (
            f"`{argument}` must name a class to build, e.g. `{argument}=ModelConfig()`. "
            "Import the name into this file; tensorViz will not resolve a dotted path."
        )
    if node.args:
        return None, (
            f"Name the arguments: write `{argument}={node.func.id}(width=8)`, "
            "not positional values."
        )
    fields: Dict[str, Any] = {}
    for keyword in node.keywords:
        if keyword.arg is None:
            return None, "Name the arguments: `**kwargs` cannot be read from a comment."
        try:
            fields[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, SyntaxError):
            return None, (
                f"`{keyword.arg}` must be a literal value (a number, string, bool, list, "
                f"tuple or dict). tensorViz builds one {node.func.id}(...), not an expression."
            )
    return ConfigCall(node.func.id, fields), None


def _read_init_call(value: str) -> Tuple[str, Dict[str, Any], Optional[str]]:
    """``"Net(hidden=8)"`` -> ``("Net", {"hidden": 8}, None)``.

    Keyword arguments whose values are literals, or one constructor call to build the
    config object a real ``__init__`` takes. Positional arguments are refused because
    the graph and the worker would have to agree on a parameter order neither of them
    can see from here, and anything richer than a single call is refused because
    evaluating it means running the user's code to decide how to run the user's code.
    """
    try:
        expr = ast.parse(value.strip(), mode="eval").body
    except SyntaxError:
        return "", {}, "tensorViz could not read this init declaration. Write it as a call, e.g. `# tensorviz: init=Net(hidden=8)`."
    if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Name):
        return "", {}, "An init declaration must be a call on the class name, e.g. `# tensorviz: init=Net(hidden=8)`."
    cls = expr.func.id
    if expr.args:
        return cls, {}, f"Name the arguments: write `# tensorviz: init={cls}(hidden=8)`, not positional values."
    overrides: Dict[str, Any] = {}
    for keyword in expr.keywords:
        if keyword.arg is None:
            return cls, {}, "Name the arguments: `**kwargs` cannot be read from a comment."
        if isinstance(keyword.value, ast.Call):
            call, error = _read_config_call(keyword.value, keyword.arg)
            if error is not None:
                return cls, {}, error
            overrides[keyword.arg] = call
            continue
        try:
            overrides[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, SyntaxError):
            return cls, {}, (
                f"`{keyword.arg}` must be a literal value (a number, string, bool, list, "
                "tuple or dict) or a call that builds one, e.g. "
                f"`{keyword.arg}=ModelConfig()`. tensorViz will not evaluate an arbitrary "
                "expression to build your model."
            )
    return cls, overrides, None


def declared_init(source: str, cls: Optional[str] = None) -> Optional[InitDeclaration]:
    """The constructor arguments a ``# tensorviz: init=`` comment declares.

    ``None`` when the file declares nothing for ``cls``. A declaration that names a
    different class is not this class's declaration: a file holding three models
    carries three lines, and reading the wrong one would build the wrong model.
    """
    for match in INIT_COMMENT.finditer(source or ""):
        name, overrides, error = _read_init_call(match.group("value"))
        # A line too malformed to name a class cannot be attributed to one, so the
        # class filter must not swallow it: the user wrote a `# tensorviz: init=`
        # that does nothing, and that is precisely what they need to be told.
        if cls is not None and name and name != cls:
            continue
        return InitDeclaration(cls=name, overrides=overrides, error=error)
    return None


class CodeParserService:
    """Service for parsing PyTorch code and converting it to graph representation"""

    def __init__(self):
        self.layer_mapping = self._create_layer_mapping()
        self.import_mapping = self._create_import_mapping()
        # Memoised ordered constructor param names, keyed by class name.
        self._sig_cache: Dict[str, List[str]] = {}
    
    def parse_pytorch_code(
        self,
        request: CodeParseRequest,
        extra_module_bases: Optional[Set[str]] = None,
        super_forward: Optional[Tuple[str, str]] = None,
        extra_layer_names: Optional[Set[str]] = None,
        constructor_kinds: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        """Parse PyTorch code and convert to graph representation.

        ``extra_module_bases`` is the multi-file projection's way in: base
        spellings (``Block``, ``blocks.Block``) the IDE module index has already
        resolved to known ``nn.Module`` subclasses in *other* supplied files.
        Syntactically a cross-file subclass like ``class Mixer(Block)`` does not
        read as ``nn.Module`` from this one buffer, so without the set it would
        be refused as "no PyTorch model classes". ``None`` keeps the historic
        single-file behaviour bit-for-bit.

        ``super_forward`` is the multi-file projection's second way in: the
        (entry, exit) node ids of the NEXT forward in the resolved inheritance
        chain, so a ``super().forward(x)`` call wires its argument's producers
        into the base flow and reads the base flow's exit as its own value
        instead of silently passing its argument through. ``None`` keeps the
        historic single-file behaviour bit-for-bit. It applies only to classes
        converted in this call; ``_static_subgraph`` recursion never inherits
        it (a same-file child's own ``super()`` target is unknown here).

        ``extra_layer_names`` is the multi-file projection's third way in: the
        ``self.`` attribute names a composed base's ``__init__`` provably
        assigned, so a subclass forward call like ``self.fc(x)`` binds to the
        inherited layer card instead of passing its argument through
        disconnected. It names layers only — no node is created for them here;
        the cross-file composition supplies the card itself.

        ``constructor_kinds`` carries workspace-proven dotted call identities:
        True for module classes (including classes with uncertain ancestry),
        False for functions/plain classes without module ancestry.
        Unknown dotted class-shaped calls stay opaque pending resolution; ordinary
        helper calls are not layers. It is request-local, never parser state.
        """
        try:
            # Parse the code using AST
            tree = ast.parse(request.code)
            # Keep original source positions and call spellings. Request-local
            # AST annotations let constructor resolution use proven import aliases.
            imported_targets = {f'{statement.module}.{alias.name}' for statement in tree.body
                                if isinstance(statement, ast.ImportFrom) and not statement.level
                                and statement.module in {'torch', 'torch.nn.functional'}
                                for alias in statement.names if alias.name != '*'}
            imports = torch_imports(tree, {'torch.nn', 'torch.nn.functional', *TORCH_LAYOUT_TARGETS, *imported_targets})
            bases = set(extra_module_bases or ())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    spelling = ast.unparse(node.func)
                    target = canonical_name(spelling, imports)
                    if spelling.split('.')[0] in imports:
                        node._torch_target = target
                    if target.startswith("torch.nn.") and target.count(".") == 2:
                        node._torch_layer_class = target.rsplit(".", 1)[-1]
                elif isinstance(node, ast.ClassDef):
                    bases.update(ast.unparse(base) for base in node.bases
                                 if canonical_name(ast.unparse(base), imports) == "torch.nn.Module")
            
            # Extract model classes
            model_classes = self._extract_model_classes(tree, bases)
            
            if not model_classes:
                return {
                    "success": False,
                    "error": "No PyTorch model classes found in the code",
                    "graphs": []
                }
            
            # Honour, in order: what the caller asked for, what the CLASS declares,
            # what a previous trace MEASURED, then what can be discovered from the
            # module, then the default.
            #
            # The declaration goes above discovery on purpose. It is the one place
            # the user states what their model is fed, and the trace worker already
            # obeys it — so without this line the drawn graph and the traced graph
            # disagreed about the input, and the drawn one filled the canvas with
            # shape errors that the first save silently cleared.
            #
            # The measurement goes BELOW the declaration and ABOVE discovery, and
            # both halves matter. Below, because a remembered measurement that
            # outranked `# tensorviz: input=` would make a cache beat the line the
            # user just typed. Above, because a shape a forward pass really ran on
            # is a better answer than one read off a literal — and it is the only
            # thing that makes the drawn Input node agree with the measured shapes
            # on the cards beneath it for a model nothing in the source describes.
            # It stands in only for a class without a declaration of its own: one
            # whose declaration this tier cannot read (the expression form) is drawn
            # from its source, since the saved shape can be older than that line.
            #
            # The declaration is read per class, by the rule setup and the worker
            # follow: in a file with several models one class's declaration is not
            # another's input, and one that belongs to no class is drawn for none.
            # Counting the models needs a module index, built at most once per parse
            # and only for a file with a declaration and several classes. Classes
            # whose base the caller resolved in another file count as proven too.
            proof: List[Set[Tuple[int, str]]] = []

            def proven() -> Set[Tuple[int, str]]:
                if not proof:
                    from services.ingest.module_discovery import module_declarations
                    proof.append({(line, name) for _, line, name in module_declarations({"buffer.py": request.code})}
                                 | {(node.lineno, node.name) for node in tree.body if isinstance(node, ast.ClassDef)
                                    and any(ast.unparse(base) in (extra_module_bases or ()) for base in node.bases)})
                return proof[0]

            def measured_input_shape(class_name: str) -> Optional[List[int]]:
                if request.measured_input_shape and not applicable(request.code, class_name, tree=tree, proven=proven)[0]:
                    return request.measured_input_shape
                return None

            inferred_input_shape = self._discover_input_shape(tree)

            # Convert each model class to a graph. A class that instantiates
            # another user-defined nn.Module (self.block = MyBlock()) gets a
            # `custom_code` node carrying that class's static subgraph, so the
            # link between the two is in the graph rather than lost (C5).
            # Which classes are used as submodules by another class in the file.
            # Whichever class nobody instantiates is the model the user meant to
            # import; picking the FIRST class instead made the result depend on
            # source order, silently dropping the real model.
            submodules = self._submodule_class_names(model_classes)

            graphs = []
            for class_name, class_node in model_classes.items():
                discovered_input_shape = (
                    request.input_shape
                    or declared_input_shape(request.code, class_name, tree=tree, proven=proven)
                    or measured_input_shape(class_name)
                    or inferred_input_shape
                )
                try:
                    graph = self._convert_class_to_graph(
                        class_name,
                        class_node,
                        request.model_name or class_name,
                        discovered_input_shape,
                        source_file=request.file,
                        model_classes=model_classes,
                        super_forward=super_forward,
                        extra_layer_names=extra_layer_names,
                        constructor_kinds=constructor_kinds,
                    )
                    graphs.append({
                        "name": class_name,
                        "graph": graph,
                        "hierarchy_level": self._detect_hierarchy_level(class_node),
                        "is_top_level": class_name not in submodules,
                    })
                except Exception as e:
                    graphs.append({
                        "name": class_name,
                        "error": f"Failed to parse class {class_name}: {str(e)}",
                        "graph": None,
                        "is_top_level": class_name not in submodules,
                    })
            
            return {
                "success": True,
                "graphs": graphs,
                "imports": self._extract_imports(tree),
                "detected_framework": "pytorch"
            }
            
        except SyntaxError:
            # Malformed code is a client error: let the endpoint surface it as
            # HTTP 400 rather than a 200 success:false (which read as "parsed").
            raise
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to parse code: {str(e)}",
                "graphs": []
            }
    
    def _discover_input_shape(self, tree: ast.AST) -> Optional[List[int]]:
        """Best-effort: scan module-level code for torch.randn/zeros/ones/
        empty(...) calls with literal int args and treat those dims as
        the intended input shape. Returns None if no such call found."""
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            attr_name = None
            module_name = None
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                module_name = func.value.id
                attr_name = func.attr
            if module_name != "torch" or attr_name not in {"randn", "zeros", "ones", "empty", "rand"}:
                continue
            dims: List[int] = []
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                    dims.append(arg.value)
                else:
                    dims = []
                    break
            if len(dims) >= 2:
                return dims
        return None

    def _submodule_class_names(self, model_classes: Dict[str, ast.ClassDef]) -> set:
        """Names of model classes instantiated inside another model class.

        `self.block = Block()` makes Block a submodule of the enclosing class, so
        Block is not the model the file is about. Only direct calls by name count;
        a class referenced solely by string or factory is treated as top-level,
        which errs toward showing the user more rather than less.
        """
        used = set()
        for owner, class_node in model_classes.items():
            for node in ast.walk(class_node):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.id if isinstance(func, ast.Name) else None
                if name and name != owner and name in model_classes:
                    used.add(name)
        return used

    def _infer_input_shape_from_class(self, class_node: ast.ClassDef) -> Optional[List[int]]:
        """Best-effort input shape from the first shape-bearing layer.

        The field offering "best-effort detection" only ever scanned module-level
        torch.randn calls, so a plain MLP with no example tensor fell back to the
        image default (1, 3, 224, 224) and failed validation with a mat1/mat2
        mismatch. A Linear's in_features or a Conv's in_channels states the same
        fact the example tensor would have.

        The width is read exactly the way the layer's own parameters are read, so a
        name __init__ gives a literal default counts as that default. Only a literal
        counted before, so `nn.Linear(hidden, hidden)` with `hidden=16` fell back to
        the image default and drew a 224x224 tensor feeding a Linear the same parse
        had already resolved to 16 features — a model the canvas showed but nothing
        could run.
        """
        scope = next(
            (
                self._init_defaults(stmt)
                for stmt in class_node.body
                if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__"
            ),
            {},
        )
        for node in ast.walk(class_node):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            attr = func.attr if isinstance(func, ast.Attribute) else (
                func.id if isinstance(func, ast.Name) else None
            )
            if not attr or not node.args:
                continue
            size = self._extract_arg_value(node.args[0], scope)
            if not isinstance(size, int) or isinstance(size, bool):
                continue
            if attr == "Linear":
                return [1, size]
            if attr in {"Conv2d", "BatchNorm2d"}:
                return [1, size, 224, 224]
            if attr in {"Conv1d", "BatchNorm1d"}:
                return [1, size, 128]
            if attr == "Conv3d":
                return [1, size, 16, 112, 112]
        return None

    def _extract_model_classes(
        self, tree: ast.AST, extra_module_bases: Optional[Set[str]] = None
    ) -> Dict[str, ast.ClassDef]:
        """Extract PyTorch model classes from AST"""
        model_classes = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if class inherits from nn.Module
                if self._inherits_from_nn_module(node, extra_module_bases):
                    model_classes[node.name] = node
        
        return model_classes
    
    def _inherits_from_nn_module(
        self, class_node: ast.ClassDef, extra_module_bases: Optional[Set[str]] = None
    ) -> bool:
        """Check if a class inherits from nn.Module"""
        # Keep the direct spellings aligned with static workspace indexing and
        # cross-file resolution. Do not admit arbitrary *.Module attributes.
        known_bases = {"Module", "nn.Module", "torch.nn.Module"}
        known_bases.update(extra_module_bases or ())
        for base in class_node.bases:
            try:
                if ast.unparse(base) in known_bases:
                    return True
            except (ValueError, TypeError, RecursionError):
                continue
        return False
    
    def _convert_class_to_graph(
        self,
        class_name: str,
        class_node: ast.ClassDef,
        model_name: str,
        discovered_input_shape: Optional[List[int]] = None,
        source_file: Optional[str] = None,
        model_classes: Optional[Dict[str, ast.ClassDef]] = None,
        _expanding: Optional[Set[str]] = None,
        super_forward: Optional[Tuple[str, str]] = None,
        extra_layer_names: Optional[Set[str]] = None,
        constructor_kinds: Optional[Dict[str, bool]] = None,
    ) -> Graph:
        """Convert a model class to a graph representation"""
        nodes = []
        edges = []
        layer_counter = 0
        
        # Extract __init__ method to find layer definitions. `alias` maps each
        # source-level attribute name (e.g. "net") to the (entry, exit) node ids
        # it resolves to — identity for a plain layer, or the first/last child
        # for an expanded nn.Sequential. `series_edges` wires the in-series
        # children of Sequential blocks.
        init_method = self._find_init_method(class_node)
        forward_method = self._find_forward_method(class_node)
        series_edges: List[Edge] = []
        alias: Dict[str, Tuple[str, str]] = {}
        has_dotted_layers = False
        if init_method:
            # Calling the assigned attribute in forward is also evidence of a
            # module boundary when an unresolved constructor uses a lowercase
            # alias. A workspace-proven helper (False) still wins over this hint.
            call_kinds = dict(constructor_kinds or {})
            called = {
                n.func.attr for n in ast.walk(forward_method) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "self"
            } if forward_method is not None else set()
            for statement in ast.walk(init_method):
                targets = statement.targets if isinstance(statement, ast.Assign) else (
                    [statement.target] if isinstance(statement, ast.AnnAssign) else [])
                if (len(targets) == 1 and isinstance(targets[0], ast.Attribute)
                        and isinstance(targets[0].value, ast.Name) and targets[0].value.id == "self"
                        and targets[0].attr in called and isinstance(statement.value, ast.Call)
                        and isinstance(statement.value.func, ast.Attribute)):
                    callee = ast.unparse(statement.value.func)
                    if all(part.isidentifier() for part in callee.split(".")) and not callee.startswith(("nn.", "self.")):
                        call_kinds.setdefault(callee, True)
            layers, series_edges, alias = self._extract_layers_from_init(
                init_method, model_classes, call_kinds
            )
            has_dotted_layers = any("." in info["layer_type"] for info in layers.values())

            # Create nodes for each layer
            for i, (layer_name, layer_info) in enumerate(layers.items()):
                layer_counter += 1
                node = self._create_layer_node(
                    layer_name,
                    layer_info,
                    i,
                    source_file=source_file,
                    model_classes=model_classes,
                    _expanding=(_expanding or set()) | {class_name},
                    discovered_input_shape=discovered_input_shape,
                )
                nodes.append(node)

        # Extract forward method to find connections. The pairs stay in
        # attribute/sentinel space until the terminal ids below are settled —
        # only then is it known what ENTRY_MARK and EXIT_MARK resolve to.
        edges = list(series_edges)
        translated: List[Tuple[str, str]] = []
        flow_scopes = []
        if forward_method:
            layer_guards: Dict[str, List[List[str]]] = {}
            connections, synth_infos = self._extract_connections_from_forward(
                forward_method, alias, super_forward, extra_layer_names, class_name,
                self._literal_attributes(class_node, init_method),
                layer_guards,
                class_node,
                flow_scopes,
            )
            for name, guards in layer_guards.items():
                for node in nodes:
                    if node.id in alias.get(name, (name, name)):
                        node.data.parameters["_review_guards"] = guards

            # Synthesize nodes for everything forward() creates that __init__ did
            # not declare: recognized functional / method ops (x.flatten(1),
            # F.relu(x)), merge nodes (residual adds, torch.cat) and opaque
            # `python` nodes for statements the parser cannot type. Each is
            # inserted into `nodes` right after its dataflow predecessor so the
            # init-order-based terminal wiring below still finds the real tail
            # (e.g. a Flatten between pool and fc keeps fc as nodes[-2]).
            for sid, info in synth_infos.items():
                helpers = self._review_helpers(info.get("call"), class_node) if info['emit'] == 'opaque' else None
                if helpers:
                    info.setdefault("parameters", {})["_review_helpers"] = helpers
                layer_counter += 1
                synth_node = self._create_synth_node(sid, info, layer_counter, source_file)
                alias[sid] = (sid, sid)
                # After the LAST predecessor already placed: a merge node has
                # several, and sitting after the earliest of them would read as
                # if it ran before one of its own inputs.
                positions = [
                    i for a, b in connections if b == sid
                    for i, n in enumerate(nodes)
                    if n.id == alias.get(a, (a, a))[1]
                ]
                if positions:
                    nodes.insert(max(positions) + 1, synth_node)
                else:
                    # Every predecessor is the input terminal (or there is none):
                    # this op is the FIRST thing forward() does, e.g.
                    # `x = x.flatten(1)` before any self.X call. It goes at the
                    # front of the chain, right after the input node prepended
                    # below — appending it would scramble the terminal wiring.
                    nodes.insert(0, synth_node)

            # Translate attribute-name connections to node ids: connect the exit
            # of the source block to the entry of the target block. Synthesized
            # ids alias to themselves, so functional ops thread into the chain.
            translated = _dedupe_pairs([
                (alias.get(a, (a, a))[1], alias.get(b, (b, b))[0])
                for a, b in connections
            ], set(synth_infos))
        elif alias and not has_dotted_layers:
            # No forward method: nothing wired the declared layers to each other, so
            # every layer after the first was left dangling and the input tensor was
            # fed straight into the last one (a 784-wide input hitting a 128-wide
            # Linear). Chain them in __init__ order, which is what the canvas shows
            # and the only reading a bare __init__ supports. Sequential children are
            # already chained by series_edges; alias maps each attribute to its
            # (entry, exit), so exit -> entry threads those blocks in correctly.
            names = list(alias.keys())
            translated = [
                (alias[a][1], alias[b][0]) for a, b in zip(names, names[1:])
            ]

        # Unknown dimensions stay empty; image dimensions are not a fallback
        # for arbitrary forwards such as feature-vector losses.
        input_shape = (
            discovered_input_shape
            or self._infer_input_shape_from_class(class_node)
        )
        # The canvas — and core/shape_inference.py, which refuses anything else —
        # carries a shape as the comma-separated string the Input node's palette
        # parameter is typed as. Emitting the raw list here meant every imported
        # model failed inference with "Shape parameter must be a non-empty string"
        # unless the frontend happened to convert it first.
        # A node id is the attribute that declared the layer, and `self.output =
        # nn.Sigmoid()` is ordinary PyTorch — so the terminal's preferred id can
        # already be taken. Two nodes with one id is silent data loss: the canvas
        # keys shapes by id, so the user's layer was dropped from the drawing
        # while the toolbar went on counting it. Terminals yield the name.
        taken = {node.id for node in nodes}
        input_id = _free_id("input", taken)
        taken.add(input_id)
        output_id = _free_id("output", taken)

        # The terminals are known, so forward()'s sentinels can become node ids.
        terminals = {ENTRY_MARK: input_id, EXIT_MARK: output_id}
        edges.extend(self._create_edges_from_connections(
            [(terminals.get(a, a), terminals.get(b, b)) for a, b in translated]
        ))

        input_node = Node(
            id=input_id,
            type="inputNode",
            position={"x": 50, "y": 200},
            data=LayerData(
                layerType=LayerType(id="input", name="Input", parameters=[]),
                parameters={"shape": ",".join(str(dim) for dim in input_shape) if input_shape else "",
                            '_review_scopes': [{'file': source_file, 'scope': scope} for scope in flow_scopes]}
            )
        )
        nodes.insert(0, input_node)
        
        output_node = Node(
            id=output_id,
            type="outputNode",
            position={"x": (layer_counter + 1) * 200 + 50, "y": 200},
            data=LayerData(
                layerType=LayerType(id="output", name="Output", parameters=[]),
                parameters={}
            )
        )
        nodes.append(output_node)
        
        # Always wire the terminals: connect input -> first real layer and
        # last real layer -> output whenever those boundary edges are absent.
        # (Real forwards produce inter-layer edges but never touch the synthetic
        # input/output nodes, so previously both terminals stayed orphaned and
        # re-export collapsed to a no-op `return x`.)
        # A retained dotted constructor is not evidence that forward invokes it.
        # Keep real forward/container edges; never invent its boundary flow from
        # declaration order, including when forward is absent or has no return.
        if len(nodes) > 2 and not has_dotted_layers:
            if not any(e.source == input_id for e in edges):
                edges.insert(0, Edge(id="edge_input_first", source=input_id, target=nodes[1].id))
            if not any(e.target == output_id for e in edges):
                edges.append(Edge(id="edge_last_output", source=nodes[-2].id, target=output_id))
        
        annotate_residual_edges(nodes, edges, forward_method, alias,
                                lambda call: self._functional_op_info(call) or self._merge_call_info(call))
        return Graph(
            id=f"parsed_{model_name.lower()}",
            name=f"Parsed {model_name}",
            description=f"Parsed from PyTorch code: {class_name}",
            nodes=nodes,
            edges=edges,
            metadata={
                "source": "code_parser",
                "original_class": class_name,
                "parsing_method": "ast"
            }
        )
    
    def _find_init_method(self, class_node: ast.ClassDef) -> Optional[ast.FunctionDef]:
        """Find the __init__ method in a class"""
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                return node
        return None
    
    def _find_forward_method(self, class_node: ast.ClassDef) -> Optional[ast.FunctionDef]:
        """Find the forward method in a class"""
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == "forward":
                return node
        return None
    
    def _extract_layers_from_init(
        self,
        init_method: ast.FunctionDef,
        model_classes: Optional[Dict[str, ast.ClassDef]] = None,
        constructor_kinds: Optional[Dict[str, bool]] = None,
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Edge], Dict[str, Tuple[str, str]]]:
        """Extract layer definitions from __init__ method.

        Returns (layers, series_edges, alias):
          - layers:       ordered {node_id: layer_info}
          - series_edges: edges wiring nn.Sequential children in series
          - alias:        {source attribute name: (entry_id, exit_id)} so the
                          forward pass can connect blocks by their boundaries.
        """
        layers: Dict[str, Dict[str, Any]] = {}
        series_edges: List[Edge] = []
        alias: Dict[str, Tuple[str, str]] = {}
        # nn.Linear(784, hidden_dim) is only unreadable if nothing says what
        # hidden_dim is. When __init__ gives it a literal default, that default is
        # the model the author shipped, so use it rather than dropping the argument
        # and importing a Linear with no out_features.
        scope = self._init_defaults(init_method)

        for node in ast.walk(init_method):
            # `self.l1 = nn.Linear(...)` and `self.l1: Any = nn.Linear(...)` declare
            # the same layer. Only the first used to be read, so an annotated
            # declaration produced no node, no alias and no span — the layer simply
            # was not on the canvas, and "Reveal in editor" for anything derived from
            # it pointed into torch's own source instead of the user's line. Annotated
            # attributes are ordinary in typed code and are what this repository's own
            # fixtures are written in, so the omission showed up as "the graph is
            # missing my layer" rather than as anything about annotations.
            targets = None
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                # `self.l1: Any` with no value declares a type, not a layer.
                targets = [node.target]
            if targets is not None:
                # Look for self.layer_name = nn.SomeLayer(...)
                if (len(targets) == 1 and
                    isinstance(targets[0], ast.Attribute) and
                    isinstance(targets[0].value, ast.Name) and
                    targets[0].value.id == "self"):

                    layer_name = targets[0].attr
                    value = node.value

                    # nn.Sequential(...): expand its positional children into
                    # individual nodes wired in series instead of dropping them.
                    if (isinstance(value, ast.Call) and
                            self._get_layer_type_from_call(value, constructor_kinds) == "Sequential"):
                        child_ids = self._flatten_sequential(layer_name, value, layers, scope, constructor_kinds)
                        if child_ids:
                            for a, b in zip(child_ids, child_ids[1:]):
                                series_edges.append(Edge(id=f"edge_{a}_{b}", source=a, target=b))
                            alias[layer_name] = (child_ids[0], child_ids[-1])
                            continue
                        # Empty / unrecognized Sequential: fall through so it is
                        # still emitted as a single node rather than lost.

                    layer_info = self._parse_layer_assignment(value, scope, constructor_kinds)
                    if layer_info:
                        # The declaring statement is what a source edit has to
                        # address, so the span is the Assign, not the call.
                        layer_info["span"] = node
                        if (model_classes or {}).get(layer_info["layer_type"]) is not None:
                            layer_info["custom_class"] = layer_info["layer_type"]
                        layers[layer_name] = layer_info
                        alias[layer_name] = (layer_name, layer_name)

        return layers, series_edges, alias

    def _flatten_sequential(
        self,
        prefix: str,
        call_node: ast.Call,
        layers: Dict[str, Dict[str, Any]],
        scope: Optional[Dict[str, Any]] = None,
        constructor_kinds: Optional[Dict[str, bool]] = None,
    ) -> List[str]:
        """Recurse into an nn.Sequential's positional args, emitting one node
        per child layer (flattening nested Sequentials). Returns the ordered
        child node ids and registers each in `layers`."""
        child_ids: List[str] = []
        for i, arg in enumerate(call_node.args):
            if not isinstance(arg, ast.Call):
                continue
            if self._get_layer_type_from_call(arg, constructor_kinds) == "Sequential":
                child_ids.extend(self._flatten_sequential(f"{prefix}_{i}", arg, layers, scope, constructor_kinds))
                continue
            layer_info = self._parse_layer_assignment(arg, scope, constructor_kinds)
            if layer_info:
                # A Sequential child has no statement of its own; the call IS the
                # declaration, so it is also the span.
                layer_info["span"] = arg
                child_id = f"{prefix}_{i}"
                layers[child_id] = layer_info
                child_ids.append(child_id)
        return child_ids
    
    def _init_defaults(self, init_method: ast.FunctionDef) -> Dict[str, Any]:
        """Literal defaults declared by __init__'s own signature, by parameter name."""
        args = init_method.args
        named = list(args.args) + list(args.kwonlyargs)
        defaults: Dict[str, Any] = {}
        positional = args.args[len(args.args) - len(args.defaults):] if args.defaults else []
        for arg, default in zip(positional, args.defaults):
            value = self._extract_arg_value(default)
            if value is not _UNSET:
                defaults[arg.arg] = value
        for arg, default in zip(args.kwonlyargs, args.kw_defaults):
            if default is None:
                continue
            value = self._extract_arg_value(default)
            if value is not _UNSET:
                defaults[arg.arg] = value
        del named
        return defaults

    def _parse_layer_assignment(
        self, value_node: ast.AST, scope: Optional[Dict[str, Any]] = None,
        constructor_kinds: Optional[Dict[str, bool]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Parse a layer assignment to extract layer type and parameters"""
        if isinstance(value_node, ast.Call):
            # Get the layer type
            layer_type = self._get_layer_type_from_call(value_node, constructor_kinds)
            if not layer_type:
                return None
            
            # Extract parameters (positional args mapped via the layer's
            # constructor signature).
            parameters = self._extract_parameters_from_call(
                value_node, None if "." in layer_type else layer_type, scope
            )
            parameters["_review_call"] = self._constructor_semantics(value_node, layer_type, scope)

            # An nn.ModuleList holds layers, and a list of layers is not a list of
            # literals: the generic argument reader drops every element it cannot
            # read as a constant, so `nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])`
            # arrived carrying `modules: []` — the container saying it is empty while
            # the user is looking at two layers they wrote. The members are the one
            # thing the card has to be able to say, so they are read here instead.
            # They are recorded under `_modules`, not `modules`. `modules` is the real
            # name of nn.ModuleList's first constructor argument, and this dict is the
            # bag every torch-facing consumer passes to the constructor: `build_layer`
            # drops `name` and every `_`-prefixed key and forwards the rest, so
            # `modules` was the only key that survived and torch was handed
            # `nn.ModuleList(modules=[{'op': 'Block', ...}, ...])` — reported on the
            # canvas as `ModuleList — build error: dict is not a Module subclass`, on a
            # model that was correct. A member list is display data; the leading
            # underscore is this codebase's word for that, and it is what `cardParams.ts`
            # already reads to draw "6 × Block".
            if layer_type == "ModuleList":
                members = self._modulelist_members(value_node, scope)
                # Whatever the generic reader left behind under the constructor's own
                # name goes, in both branches. Nothing may reach a torch constructor.
                parameters.pop("modules", None)
                if members is not None:
                    parameters["_modules"] = members
                # Otherwise nothing in the file says what is in there (a name, or a
                # comprehension whose count is an argument with no default). Saying
                # nothing is honest; saying "empty" was not.

            return {
                "layer_type": layer_type,
                "parameters": parameters,
                "raw_call": ast.unparse(value_node) if hasattr(ast, 'unparse') else str(value_node)
            }
        
        return None
    
    def _modulelist_members(
        self, call_node: ast.Call, scope: Optional[Dict[str, Any]] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """The layers an ``nn.ModuleList(...)`` declares, in order, or None when
        this tier cannot read them off the source.

        Two forms are readable without running anything: a list written out
        (``[nn.Linear(8, 8), nn.ReLU()]``) and the repeat idiom
        (``[nn.Linear(8, 8) for _ in range(3)]``), whose count is a literal or an
        ``__init__`` default. Anything else — a name, a count nothing in the file
        gives a value — is None rather than a shorter list, because a member list
        that quietly drops what it could not parse is the same lie as the empty
        one it replaces.
        """
        argument = None
        if call_node.args:
            argument = call_node.args[0]
        else:
            for keyword in call_node.keywords:
                if keyword.arg == "modules":
                    argument = keyword.value
                    break
        if argument is None:
            return []

        if isinstance(argument, (ast.List, ast.Tuple)):
            elements = list(argument.elts)
        elif isinstance(argument, ast.ListComp):
            count = self._comprehension_count(argument, scope)
            if count is None:
                return None
            elements = [argument.elt] * count
        else:
            return None

        members: List[Dict[str, Any]] = []
        for element in elements:
            member = self._parse_layer_assignment(element, scope)
            if member is None:
                return None
            members.append({"op": member["layer_type"], "params": member["parameters"]})
        return members

    def _comprehension_count(
        self, comp: ast.ListComp, scope: Optional[Dict[str, Any]] = None
    ) -> Optional[int]:
        """How many times ``[X for _ in range(N)]`` repeats, or None.

        Only ``range`` over readable literals, because that is the idiom the
        repeat block is written in and the only iterable whose length is knowable
        without evaluating the file.
        """
        if len(comp.generators) != 1:
            return None
        generator = comp.generators[0]
        if generator.ifs or generator.is_async:
            return None
        iterator = generator.iter
        if (
            not isinstance(iterator, ast.Call)
            or not isinstance(iterator.func, ast.Name)
            or iterator.func.id != "range"
            or iterator.keywords
            or not 1 <= len(iterator.args) <= 3
        ):
            return None
        bounds = [self._extract_arg_value(arg, scope) for arg in iterator.args]
        if any(bound is _UNSET or not isinstance(bound, int) or isinstance(bound, bool)
               for bound in bounds):
            return None
        try:
            return len(range(*bounds))
        except ValueError:
            # range(step=0). The user's own file would raise; do not guess for it.
            return None

    def _get_layer_type_from_call(
        self, call_node: ast.Call, constructor_kinds: Optional[Dict[str, bool]] = None,
    ) -> Optional[str]:
        """Extract the layer class name from a function call, preserving its
        original casing (e.g. 'ReLU', 'Conv2d') so it can be resolved to the
        canonical class name later."""
        proven_torch = getattr(call_node, "_torch_layer_class", None)
        if proven_torch:
            return proven_torch
        if isinstance(call_node.func, ast.Attribute):
            callee = ast.unparse(call_node.func)
            proven = (constructor_kinds or {}).get(callee)
            if proven is not None:
                return callee if proven else None
            if (isinstance(call_node.func.value, ast.Name) and
                call_node.func.value.id == "nn"):
                return call_node.func.attr
            # Keep the complete spelling so blocks.Linear cannot borrow torch's
            # signature or registry identity. This is only a candidate, never
            # evidence of a resolved module. Lowercase helpers remain ordinary
            # Python unless the workspace index proves they alias a module class.
            if (call_node.func.attr[:1].isupper()
                    and all(part.isidentifier() for part in callee.split("."))
                    and callee.split(".")[0] != "self"):
                return callee
        elif isinstance(call_node.func, ast.Name):
            # Direct function call
            return call_node.func.id

        return None

    def _constructor_semantics(self, call_node, layer_type, scope):
        """Normalize argument spelling without losing unsupported expressions.

        Unlike display parameters, this retains every argument, including **kwargs
        and nonliteral expressions. Reordering keywords or writing p=0.1 instead
        of positional 0.1 is not a computation change for a known signature.
        """
        class CanonicalCalls(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                target = getattr(node, "_torch_layer_class", None)
                if target:
                    node.func = ast.parse("torch.nn." + target, mode="eval").body
                return node

        def value(node):
            if isinstance(node, ast.Name) and scope and node.id in scope:
                return scope[node.id]
            try:
                return ast.literal_eval(node)
            except (ValueError, TypeError):
                return {"expression": ast.dump(CanonicalCalls().visit(copy.deepcopy(node)), include_attributes=False)}

        names = self._positional_param_names(layer_type) if "." not in layer_type else []
        args = {names[i] if i < len(names) else f"arg_{i}": value(arg) for i, arg in enumerate(call_node.args)}
        for i, keyword in enumerate(call_node.keywords):
            args[keyword.arg if keyword.arg is not None else f"**{i}"] = value(keyword.value)
        # Only trusted torch signatures establish omitted defaults. An unpacked
        # argument can supply any keyword, so no defaults are inferred for it.
        if getattr(call_node, "_torch_layer_class", None) == layer_type and not any(isinstance(a, ast.Starred) for a in call_node.args) and all(k.arg is not None for k in call_node.keywords):
            try:
                import inspect
                import torch.nn as nn
                constructor = nn.RNNBase if layer_type in _ARGS_KWARGS_POSITIONAL else getattr(nn, layer_type)
                for parameter in inspect.signature(constructor).parameters.values():
                    if parameter.name == "proj_size" and layer_type in {"GRU", "RNN"}:
                        continue
                    default = parameter.default
                    if default is None or isinstance(default, (bool, int, float, str)):
                        args.setdefault(parameter.name, default)
                if layer_type == "RNN":
                    args.setdefault("nonlinearity", "tanh")
            except (ImportError, OSError, AttributeError, TypeError, ValueError):
                # The static tool service deliberately has no model framework.
                # Keep explicit arguments; absent signatures cannot prove defaults.
                pass
        return {"target": layer_type, "arguments": args}

    @staticmethod
    def _review_helpers(expression, class_node):
        """Retain reachable source bodies on an opaque helper call, without running it."""
        methods = {n.name: n for n in class_node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        found = {}

        def visit(tree):
            if tree is None:
                return
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"
                        and node.func.attr in methods and node.func.attr not in found):
                    method = methods[node.func.attr]
                    found[node.func.attr] = ast.dump(method, include_attributes=False)
                    visit(method)
        visit(expression)
        return found

    def _extract_parameters_from_call(
        self,
        call_node: ast.Call,
        layer_type: Optional[str] = None,
        scope: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract parameters from a function call.

        Positional args are mapped to their real constructor parameter names
        (from the layer's signature) so e.g. nn.Linear(8, 16) yields
        in_features=8, out_features=16 rather than arg_0/arg_1.
        """
        parameters = {}

        # Positional arguments -> real parameter names when the signature is known.
        positional_names = self._positional_param_names(layer_type) if layer_type else []
        for i, arg in enumerate(call_node.args):
            value = self._extract_arg_value(arg, scope)
            if value is _UNSET:
                continue
            key = positional_names[i] if i < len(positional_names) else f"arg_{i}"
            parameters[key] = value

        # Keyword arguments
        for keyword in call_node.keywords:
            if keyword.arg is None:
                continue
            value = self._extract_arg_value(keyword.value, scope)
            if value is not _UNSET:
                parameters[keyword.arg] = value

        return parameters

    def _extract_arg_value(self, node: ast.AST, scope: Optional[Dict[str, Any]] = None) -> Any:
        """Extract a literal value from an AST argument node, or _UNSET when it
        is not a supported literal (e.g. a name/expression).

        `scope` carries __init__'s own literal defaults, so a layer written in terms
        of a constructor argument still imports with the value the author defaulted
        it to. A name with no default stays _UNSET: nothing in the file says what it
        is, and guessing would be worse than leaving the field empty.
        """
        if isinstance(node, ast.Name) and scope and node.id in scope:
            return scope[node.id]
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Num):  # Python < 3.8 compatibility
            return node.n
        if isinstance(node, ast.Str):  # Python < 3.8 compatibility
            return node.s
        if isinstance(node, (ast.List, ast.Tuple)):
            return self._extract_list_value(node)
        return _UNSET

    def _positional_param_names(self, layer_type: str) -> List[str]:
        """Ordered positional constructor parameter names for a torch.nn class,
        resolved via the op registry's canonical name. Returns [] when the class
        or its signature can't be resolved (positional args then stay arg_i)."""
        if layer_type in self._sig_cache:
            return self._sig_cache[layer_type]

        spec = spec_for(layer_type)
        class_name = spec.canonical.split(".")[-1] if spec else layer_type
        names: List[str] = []
        try:
            import inspect
            import torch.nn as nn
            cls = getattr(nn, class_name)
            sig = inspect.signature(cls)
            names = [
                p.name for p in sig.parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
        except Exception:
            names = []
        # RNN-family (*args, **kwargs) signatures expose no positional names;
        # fall back to the known constructor order so positional args map to
        # input_size/hidden_size instead of arg_0/arg_1.
        if not names:
            names = _ARGS_KWARGS_POSITIONAL.get(class_name, [])

        self._sig_cache[layer_type] = names
        return names
    
    def _extract_list_value(self, list_node: ast.List) -> List[Any]:
        """Extract values from an AST List node"""
        values = []
        for element in list_node.elts:
            if isinstance(element, ast.Constant):
                values.append(element.value)
            elif isinstance(element, ast.Num):
                values.append(element.n)
        return values
    
    def _literal_attributes(self, class_node, init_method):
        """Only unconditional literal constructor attributes written once in source."""
        if init_method is None:
            return {}
        stores = {}
        for node in ast.walk(class_node):
            if (isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del))
                    and isinstance(node.value, ast.Name) and node.value.id == "self"):
                stores[node.attr] = stores.get(node.attr, 0) + 1
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in {"setattr", "delattr"}):
                return {}
        values = {}
        scope = self._init_defaults(init_method)
        for stmt in init_method.body:
            targets = stmt.targets if isinstance(stmt, ast.Assign) else ([stmt.target] if isinstance(stmt, ast.AnnAssign) else [])
            for target in targets:
                if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                        and target.value.id == "self" and stores.get(target.attr) == 1):
                    try:
                        value = scope[stmt.value.id] if isinstance(stmt.value, ast.Name) else ast.literal_eval(stmt.value)
                        if type(value) in (int, float):
                            values[target.attr] = value
                    except (ValueError, TypeError, KeyError):
                        pass
        return values

    def _extract_connections_from_forward(
        self,
        forward_method: ast.FunctionDef,
        alias: Dict[str, Tuple[str, str]],
        super_forward: Optional[Tuple[str, str]] = None,
        extra_layer_names: Optional[Set[str]] = None,
        current_class: Optional[str] = None,
        literal_attributes: Optional[Dict[str, Any]] = None,
        layer_guards: Optional[Dict[str, List[List[str]]]] = None,
        class_node: Optional[ast.ClassDef] = None,
        flow_scopes: Optional[List[List[str]]] = None,
    ) -> Tuple[List[Tuple[str, str]], Dict[str, Dict[str, Any]]]:
        """Read forward() as dataflow (SSA), not as one linear chain.

        ``super_forward`` is the (entry, exit) node pair of the NEXT forward in
        the resolved inheritance chain, supplied by the cross-file projection.
        A ``super().forward(...)`` call then wires the argument's producers into
        the base flow's entry and reads the base flow's exit as the call's own
        value. Without it the historic behaviour stands: the call passes its
        argument's producers through transparently.

        ``extra_layer_names`` are ``self.`` attributes a composed base owns even
        though this class's own ``__init__`` never declares them (the base's
        super call runs first): ``self.fc(x)`` then binds as that layer's call —
        no node is synthesized for it here — instead of leaking the argument's
        producers past the inherited member it flows through.

        Every local name is bound to the node(s) that PRODUCE its tensor; the
        forward arguments are bound to the input terminal and whatever `return`
        yields is wired to the output terminal. An expression therefore resolves
        to its real producers, which is what makes residuals, branches and merges
        expressible: `x + self.l2(h)` becomes an `add` node with two in-edges
        instead of two names concatenated into a fake chain.

        Statements the parser cannot type do not vanish either. An unrecognised
        call that stands as a whole statement becomes an opaque `python` node
        carrying its source text, so the graph says "there is code here I did not
        model" rather than quietly dropping a transpose. Nested unrecognised calls
        (`x.size(0)` inside a view) are traversed for their producers only — they
        are operands, not statements.

        Compound statements are walked instead of skipped; nodes created inside
        one are marked `control_flow` so consumers can show that the op runs
        conditionally or in a loop.

        Returns (connections, ops):
          - connections: (source, target) pairs in ATTRIBUTE/synthetic-id space,
            with the graph's terminals as ENTRY_MARK/EXIT_MARK; the caller maps
            them to node ids through `alias` and the terminal ids it picked.
          - ops: ordered {synthetic id: build info} for the nodes forward()
            itself creates (functional ops, merges, opaque statements).
        """
        layer_names = set(alias) | set(extra_layer_names or ())
        ops: Dict[str, Dict[str, Any]] = {}
        connections: List[Tuple[str, str]] = []
        env: Dict[str, List[str]] = {}
        returns: List[str] = []
        counter = [0]
        guards: List[str] = []
        methods = local_methods(class_node)
        active = [forward_method.name]
        scope = [current_class or '', forward_method.name]
        method = forward_method
        helper_budget = [4096]
        statement_methods = {}
        declared_forward = (forward_declaration(class_node) if class_node is not None else None) or forward_method
        if flow_scopes is not None:
            flow_scopes.append(list(scope))

        # Every parameter of forward() is an input tensor of this graph.
        for arg in (declared_forward.args.posonlyargs + declared_forward.args.args)[1:] + declared_forward.args.kwonlyargs:
            env[arg.arg] = Flow([ENTRY_MARK])

        def connect(source: str, target: str) -> None:
            if source and target and source != target:
                connections.append((source, target))

        def add_op(op_id: str, info: Dict[str, Any], stmt: ast.AST,
                   call: ast.AST, control_flow: bool, sources: List[str]) -> str:
            sid = f"{op_id}_{counter[0]}"
            counter[0] += 1
            built = dict(info)
            built["span"] = stmt
            built["raw_call"] = ast.unparse(call)
            built["call"] = call
            call._tensorviz_emitted = True
            built["control_flow"] = control_flow
            built['parameters'] = {**built.get('parameters', {}), '_review_scope': list(scope),
                                   '_review_order': counter[0], '_review_statement': ast.dump(stmt, include_attributes=False)}
            statement_methods[stmt] = method
            if guards:
                built["parameters"] = {**built.get("parameters", {}), "_review_guards": list(guards)}
            ops[sid] = built
            for source in sources:
                connect(source, sid)
            return sid

        def call_sources(expr: ast.Call, stmt: ast.AST, control_flow: bool, preserve_uses: bool = False) -> List[str]:
            """Producers feeding a call: the callee's base (`x` in `x.flatten(1)`)
            first, then every argument.

            `self.embed` is the exception: it is the thing being CALLED, not a tensor
            flowing in. Resolving it would connect the layer to itself now that a bare
            `self.<attr>` read resolves to its node (see `resolve`)."""
            if (
                isinstance(expr.func, ast.Attribute)
                and isinstance(expr.func.value, ast.Name)
                and expr.func.value.id == "self"
                and expr.func.attr in layer_names
            ):
                out: List[str] = []
            else:
                out = list(resolve(expr.func, stmt, control_flow))
            for arg in expr.args:
                out.extend(resolve(arg, stmt, control_flow))
            for keyword in expr.keywords:
                out.extend(resolve(keyword.value, stmt, control_flow))
            return out if preserve_uses else _dedupe(out)

        def visit_call(expr: ast.Call, stmt: ast.AST, control_flow: bool, root: bool) -> List[str]:
            nonlocal env, returns, method, scope
            func = expr.func

            # super().forward(...): when the cross-file projection resolved the
            # next forward in the MRO, route through it instead of passing the
            # argument through untouched. `super()` zero-arg, or the explicit
            # two-arg spelling `super(ThisClass, self)`, both name the next
            # forward — the planner verified the pair, so either reads the same.
            if (
                super_forward is not None
                and isinstance(func, ast.Attribute)
                and func.attr == "forward"
                and isinstance(func.value, ast.Call)
                and isinstance(func.value.func, ast.Name)
                and func.value.func.id == "super"
                and _is_self_super_call(func.value, current_class)
            ):
                sources = call_sources(expr, stmt, control_flow)
                for source in sources:
                    connect(source, super_forward[0])
                return [super_forward[1]]

            # self.<attr>(...) for an attribute __init__ declared: a real layer.
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "self"
                and func.attr in layer_names
            ):
                if guards and layer_guards is not None:
                    layer_guards.setdefault(func.attr, []).append(list(guards))
                for source in call_sources(expr, stmt, control_flow):
                    connect(source, func.attr)
                return [func.attr]

            # A direct local helper owns a separate local binding/return scope.
            if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                    and func.value.id == 'self' and func.attr in methods):
                helper = methods[func.attr]
                call = expr
                if (method is forward_method and declared_forward is not forward_method
                        and len(expr.args) == 1 and isinstance(expr.args[0], ast.Starred)
                        and isinstance(expr.args[0].value, ast.Name)
                        and expr.args[0].value.id == forward_method.args.vararg.arg):
                    # Reuse the signature preflight's proven transparent *args/**kwargs delegate.
                    call = copy.copy(expr)
                    call.args = [ast.Name(id=arg.arg, ctx=ast.Load())
                                 for arg in (helper.args.posonlyargs + helper.args.args)[1:]]
                    call.keywords = [ast.keyword(arg=arg.arg, value=ast.Name(id=arg.arg, ctx=ast.Load()))
                                     for arg in helper.args.kwonlyargs]
                cost = sum(1 for _ in ast.walk(helper))
                if helper.name not in active and len(active) < 16 and cost <= helper_budget[0]:
                    checkpoint = len(connections), counter[0], helper_budget[0], len(flow_scopes or [])
                    bindings = bind_helper(helper, call, lambda value: resolve(value, stmt, control_flow))
                    if bindings is not None:
                        helper_budget[0] -= cost
                        saved = env, returns, method, scope
                        env, returns, method = bindings, [], helper
                        scope = scope + [ast.unparse(expr)]
                        if flow_scopes is not None:
                            flow_scopes.append(list(scope))
                        active.append(helper.name)
                        try:
                            walk(helper.body, control_flow)
                            return Flow.merge(returns)
                        finally:
                            active.pop()
                            env, returns, method, scope = saved
                    # A refused binding must not resolve argument expressions twice.
                    del connections[checkpoint[0]:]
                    for sid in list(ops)[checkpoint[1]:]:
                        del ops[sid]
                    counter[0], helper_budget[0] = checkpoint[1:3]
                    if flow_scopes is not None:
                        del flow_scopes[checkpoint[3]:]
                sources = call_sources(expr, stmt, control_flow)
                return [add_op('python', {'emit': 'opaque', 'parameters': {}}, stmt, expr, control_flow, sources)]

            # A module constructed at the call site is still an operation, not a registered member.
            if isinstance(func, ast.Call) and getattr(func, '_torch_layer_class', None):
                layer_type = func._torch_layer_class
                if spec_for('nn.' + layer_type) is not None:
                    literals = {name: value.literal for name, value in env.items()
                                if isinstance(value, Flow) and value.literal is not UNKNOWN}
                    parameters = self._extract_parameters_from_call(func, layer_type, literals)
                    parameters['_review_call'] = self._constructor_semantics(func, layer_type, literals)
                    sources = [node for argument in expr.args for node in resolve(argument, stmt, control_flow)]
                    sources += [node for keyword in expr.keywords for node in resolve(keyword.value, stmt, control_flow)]
                    return [add_op(layer_type.lower(), {'emit': 'module', 'type': layer_type, 'parameters': parameters},
                                   stmt, expr, control_flow, sources)]

            merge = self._merge_call_info(expr)
            if merge is not None:
                sources = call_sources(expr, stmt, control_flow, preserve_uses=True)
                if not sources:
                    return []
                return [add_op(merge["op"],
                               {"emit": "functional", "type": f"torch.{merge['op']}",
                                "parameters": merge["parameters"]},
                               stmt, expr, control_flow, sources)]

            layout = tensor_layout_call(expr)
            if layout is not None:
                info, tensor = layout
                sources = resolve(tensor, stmt, control_flow)
                if sources:
                    return [add_op(info['type'].split('.')[-1], info, stmt, expr, control_flow, sources)]

            info = self._functional_op_info(expr)
            if info is not None:
                sources = call_sources(expr, stmt, control_flow)
                op_id = info["type"].split(".")[-1].lower()
                return [add_op(op_id, info, stmt, expr, control_flow, sources)]

            if (isinstance(func, ast.Attribute) and func.attr in ('size', 'dim', 'numel')
                    and not (isinstance(func.value, ast.Name) and func.value.id == 'self')):
                # Tensor metadata feeds scalar options, not tensor connections.
                if resolve(func.value, stmt, control_flow):
                    return Flow()

            sources = call_sources(expr, stmt, control_flow)
            if root and sources:
                # A whole statement we cannot type. Keep it, with its source text,
                # rather than dropping it and lying about the topology.
                return [add_op("python", {"emit": "opaque", "parameters": {}},
                               stmt, expr, control_flow, sources)]
            return sources

        def visit_binop(expr: ast.BinOp, stmt: ast.AST, control_flow: bool) -> List[str]:
            left = resolve(expr.left, stmt, control_flow)
            right = resolve(expr.right, stmt, control_flow)
            sources = left + right
            op_id = _BINOP_MERGE.get(type(expr.op))
            params = {}
            if op_id in {"add", "sub", "mul", "div"} and sources and bool(left) != bool(right):
                scalar = expr.right if left else expr.left
                try:
                    value = ((literal_attributes or {})[scalar.attr]
                             if isinstance(scalar, ast.Attribute) and isinstance(scalar.value, ast.Name) and scalar.value.id == "self"
                             else ast.literal_eval(scalar))
                    if type(value) in (int, float):
                        params = {"other": value, "_scalar_side": "right" if left else "left"}
                except (ValueError, TypeError, KeyError):
                    pass
            if op_id is None or len(sources) < 2:
                if not params:
                    return sources
            return [add_op(op_id,
                           {"emit": "functional", "type": f"torch.{op_id}", "parameters": params},
                           stmt, expr, control_flow, sources)]

        def resolve(expr: Optional[ast.AST], stmt: ast.AST,
                    control_flow: bool, root: bool = False) -> List[str]:
            """Producers of an expression's value."""
            if expr is None:
                return []
            if isinstance(expr, ast.Name):
                return env.get(expr.id, Flow())
            if isinstance(expr, ast.Constant):
                return Flow(literal=expr.value)
            if isinstance(expr, ast.Dict):
                if any(not isinstance(key, ast.Constant) or not isinstance(key.value, (str, int)) for key in expr.keys):
                    return Flow(node for value in expr.values for node in resolve(value, stmt, control_flow))
                return Flow.container({key.value: resolve(value, stmt, control_flow, root=True)
                                       for key, value in zip(expr.keys, expr.values)})
            if isinstance(expr, ast.Subscript):
                value = resolve(expr.value, stmt, control_flow)
                if isinstance(value, Flow) and value.fields is not None and isinstance(expr.slice, ast.Constant):
                    return value.fields.get(expr.slice.value, value)
                return value
            if (
                isinstance(expr, ast.Attribute)
                and isinstance(expr.value, ast.Name)
                and expr.value.id == "self"
                and expr.attr in layer_names
            ):
                # A declared attribute READ, not called: `self.pos` in
                # `self.embed(tokens) + self.pos[:, :n]`. `visit_call` already maps the
                # CALLED form, so only this one was missing — and it fell through to
                # `resolve(self)`, which is nothing.
                #
                # Three symptoms, one cause. The parameter never became a source, so it
                # had no outgoing edge and hung off the canvas; `visit_binop` saw one
                # source instead of two, took its "`x * 2` scales one tensor" branch and
                # emitted no `add`, so the line joining them drew nothing; and the node,
                # a layer with no parents, was reported as "Node has no input
                # connections" — on a parameter, which by definition has no input.
                #
                # Must precede the generic Attribute branch below, or it never fires.
                return [expr.attr]
            if isinstance(expr, (ast.Attribute, ast.Subscript, ast.Starred)):
                if isinstance(expr, ast.Attribute) and expr.attr in ('shape', 'dtype', 'device', 'ndim'):
                    resolve(expr.value, stmt, control_flow)
                    return Flow()
                return resolve(expr.value, stmt, control_flow)
            if isinstance(expr, ast.UnaryOp):
                try:
                    return Flow(literal=ast.literal_eval(expr))
                except (ValueError, TypeError):
                    return Flow(resolve(expr.operand, stmt, control_flow))
            if isinstance(expr, (ast.Tuple, ast.List)):
                return Flow.container({i: resolve(element, stmt, control_flow) for i, element in enumerate(expr.elts)})
            if isinstance(expr, ast.Set):
                return Flow(node for element in expr.elts for node in resolve(element, stmt, control_flow))
            if isinstance(expr, ast.BoolOp):
                out = []
                for value in expr.values:
                    out.extend(resolve(value, stmt, control_flow))
                return _dedupe(out)
            if isinstance(expr, ast.Compare):
                out = list(resolve(expr.left, stmt, control_flow))
                for comparator in expr.comparators:
                    out.extend(resolve(comparator, stmt, control_flow))
                return _dedupe(out)
            if isinstance(expr, ast.IfExp):
                return _dedupe(
                    resolve(expr.body, stmt, control_flow)
                    + resolve(expr.orelse, stmt, control_flow)
                )
            if isinstance(expr, ast.BinOp):
                return visit_binop(expr, stmt, control_flow)
            if isinstance(expr, ast.Call):
                return visit_call(expr, stmt, control_flow, root)
            return []

        def bind(target: ast.AST, producers: List[str]) -> None:
            if isinstance(target, ast.Name):
                env[target.id] = producers
            elif isinstance(target, (ast.Tuple, ast.List)):
                # `x, _ = self.lstm(x)`: every bound name carries the same producer.
                for i, element in enumerate(target.elts):
                    bind(element, producers.fields.get(i, producers) if isinstance(producers, Flow)
                         and producers.fields is not None else producers)
            elif isinstance(target, ast.Starred):
                bind(target.value, producers)
            # `self.cache = x` / `buf[i] = x` are state, not dataflow: not tracked.

        def walk(body: List[ast.stmt], control_flow: bool) -> None:
            for stmt in body:
                statement(stmt, control_flow)
                if isinstance(stmt, (ast.Return, ast.Raise)):
                    break

        def statement(stmt: ast.stmt, control_flow: bool) -> None:
            if isinstance(stmt, ast.Assign):
                producers = resolve(stmt.value, stmt, control_flow, root=True)
                for target in stmt.targets:
                    bind(target, producers)
            elif isinstance(stmt, ast.AnnAssign):
                if stmt.value is not None:
                    bind(stmt.target, resolve(stmt.value, stmt, control_flow, root=True))
            elif isinstance(stmt, ast.AugAssign):
                # `x += self.f(x)` joins the old value with the new one.
                before = resolve(stmt.target, stmt, control_flow)
                producers = _dedupe(
                    before + resolve(stmt.value, stmt, control_flow, root=True)
                )
                op_id = _BINOP_MERGE.get(type(stmt.op))
                if op_id is not None and len(producers) >= 2:
                    producers = [add_op(op_id,
                                        {"emit": "functional", "type": f"torch.{op_id}",
                                         "parameters": {}},
                                        stmt, stmt.value, control_flow, producers)]
                bind(stmt.target, producers)
            elif isinstance(stmt, ast.Expr):
                resolve(stmt.value, stmt, control_flow, root=True)
            elif isinstance(stmt, ast.Return):
                returns.append(resolve(stmt.value, stmt, control_flow, root=True))
            elif isinstance(stmt, ast.If):
                # The test is read for its producers only: `if x.sum() > 0` asks a
                # question about a tensor, it does not transform one.
                resolve(stmt.test, stmt, control_flow)
                guards.append("if " + ast.unparse(stmt.test))
                walk(stmt.body, True)
                guards[-1] = "else " + ast.unparse(stmt.test)
                walk(stmt.orelse, True)
                guards.pop()
            elif isinstance(stmt, ast.While):
                resolve(stmt.test, stmt, control_flow)
                guards.append("while " + ast.unparse(stmt.test))
                walk(stmt.body, True)
                walk(stmt.orelse, True)
                guards.pop()
            elif isinstance(stmt, ast.For):
                resolve(stmt.iter, stmt, control_flow)
                guards.append("for " + ast.unparse(stmt.target) + " in " + ast.unparse(stmt.iter))
                walk(stmt.body, True)
                walk(stmt.orelse, True)
                guards.pop()
            elif isinstance(stmt, ast.With):
                for item in stmt.items:
                    resolve(item.context_expr, stmt, control_flow)
                walk(stmt.body, True)
            elif isinstance(stmt, ast.Try):
                walk(stmt.body, True)
                for handler in stmt.handlers:
                    walk(handler.body, True)
                walk(stmt.orelse, True)
                walk(stmt.finalbody, True)

        walk(forward_method.body, False)
        # An edited expression can retain its unique source binding. Reassignments
        # and statements projecting to multiple operations provide no such identity.
        statement_ops: Dict[ast.AST, List[Dict[str, Any]]] = {}
        for info in ops.values():
            statement_ops.setdefault(info["span"], []).append(info)
        for stmt, entries in statement_ops.items():
            owner = statement_methods[stmt]
            target = (stmt.targets[0] if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                      else stmt.target if isinstance(stmt, ast.AnnAssign) else None)
            stores = sum(isinstance(item, ast.Name) and isinstance(item.ctx, (ast.Store, ast.Del))
                         and item.id == getattr(target, 'id', None) for item in ast.walk(owner))
            if len(entries) == 1 and isinstance(target, ast.Name) and stores == 1:
                info = entries[0]
                info["parameters"] = {**info.get("parameters", {}), "_review_binding":
                    [*info['parameters']['_review_scope'], target.id]}
        for producer in _dedupe(Flow.merge(returns)):
            connect(producer, EXIT_MARK)

        return _dedupe_pairs(connections, set(ops)), ops

    def _merge_call_info(self, call_node: ast.Call) -> Optional[Dict[str, Any]]:
        """torch.cat / torch.stack / torch.add / torch.mul written in forward:
        a merge node with one in-edge per tensor argument. cat and stack also
        carry the axis they join on when it is a literal. Returns None for
        anything else."""
        func = call_node.func
        if not isinstance(func, ast.Attribute):
            return None
        if not (isinstance(func.value, ast.Name) and func.value.id == "torch"):
            return None
        if func.attr not in _MERGE_CALLS:
            return None
        parameters: Dict[str, Any] = {}
        if func.attr in ("cat", "stack"):
            dim = self._literal_int_arg(call_node, 1, "dim")
            if dim is not None:
                parameters["dim"] = dim
        return {"op": func.attr, "parameters": parameters}

    def _functional_op_info(self, call_node: ast.Call) -> Optional[Dict[str, Any]]:
        """Recognize a small set of functional / method forward ops and return
        the build info for a synthesized graph node, or None when the call is not
        one we can type. Info schema:
          {"emit": "module",     "type": <nn class>,   "parameters": {...}}
          {"emit": "functional", "type": <torch.xxx>,  "parameters": {...}}
        "module" ops re-export as `nn.<Class>(...)`; "functional" ops as
        `torch.<op>(x, ...)` in the forward pass (matching CodeGenerationService).
        None no longer means "dropped": a call we cannot type that stands as a
        whole statement becomes an opaque `python` node instead
        (`_extract_connections_from_forward`).
        """
        func = call_node.func
        if not isinstance(func, ast.Attribute):
            return None
        attr = func.attr
        base = func.value
        base_name = base.id if isinstance(base, ast.Name) else None
        is_method = base_name not in ("F", "torch", "nn")  # a tensor method vs a namespaced fn

        # flatten -> nn.Flatten. torch.flatten(x, k): start_dim is arg 1 (arg 0 is
        # the tensor). x.flatten(k): start_dim is arg 0. A no-arg flatten emits a
        # bare nn.Flatten() (start_dim defaults to 1 — the flatten-before-Linear
        # intent), rather than the tensor method's own start_dim=0 which would
        # collapse the batch and never round-trip to a runnable classifier.
        if attr == "flatten" and base_name == "torch":
            start_dim = self._literal_int_arg(call_node, 1, "start_dim")
            return {"emit": "module", "type": "Flatten",
                    "parameters": {} if start_dim is None else {"start_dim": start_dim}}
        if attr == "flatten" and is_method:
            start_dim = self._literal_int_arg(call_node, 0, "start_dim")
            return {"emit": "module", "type": "Flatten",
                    "parameters": {} if start_dim is None else {"start_dim": start_dim}}

        # x.view(b, -1) / x.reshape(b, -1): the canonical "flatten all but dim 0"
        # idiom -> nn.Flatten(start_dim=1). Only this exact 2-arg, trailing -1 form
        # is matched; any other view/reshape needs shape args we can't recover, so
        # it stays untyped (and becomes an opaque node) rather than guessed.
        if attr in ("view", "reshape") and is_method:
            args = call_node.args
            if len(args) == 2 and self._const_int(args[1]) == -1:
                return {"emit": "module", "type": "Flatten", "parameters": {"start_dim": 1}}
            return None

        # x.permute(dims) -> torch.permute(x, dims=[...]), emitted functionally.
        # Only literal-int dims are captured; anything else stays untyped.
        if attr == "permute" and is_method:
            dims = self._literal_dims(call_node.args)
            if dims is None:
                return None
            return {"emit": "functional", "type": "torch.permute", "parameters": {"dims": dims}}

        # F.<act> / nn.functional.<act> / torch.nn.functional.<act> -> nn.<Act>.
        if self._is_functional_namespace(base):
            key = attr.rstrip("_")  # drop in-place suffix (relu_ -> relu)
            nn_cls = _F_ACTIVATIONS.get(key)
            if nn_cls is not None:
                params: Dict[str, Any] = {}
                if nn_cls in ("Softmax", "LogSoftmax"):
                    dim = self._literal_int_arg(call_node, 1, "dim")
                    if dim is not None:
                        params["dim"] = dim
                return {"emit": "module", "type": nn_cls, "parameters": params}

        return None

    @staticmethod
    def _is_functional_namespace(base: ast.AST) -> bool:
        """True when `base` is the torch.nn.functional namespace: the `F` alias,
        or any attribute chain ending in `.functional`."""
        if isinstance(base, ast.Name):
            return base.id == "F"
        if isinstance(base, ast.Attribute):
            return base.attr == "functional"
        return False

    @staticmethod
    def _const_int(node: ast.AST) -> Optional[int]:
        """Int from an int-literal node, including negative literals which the AST
        encodes as UnaryOp(USub, Constant(n)) — so `-1` / `dim=-1` resolve. bool is
        excluded (it's an int subclass)."""
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = node.operand
            if isinstance(inner, ast.Constant) and isinstance(inner.value, int) and not isinstance(inner.value, bool):
                return -inner.value
        return None

    def _literal_int_arg(self, call_node: ast.Call, pos: int, kw: str) -> Optional[int]:
        """Literal int from keyword `kw`, else positional index `pos`; None if
        absent or not an int literal (negatives supported)."""
        for keyword in call_node.keywords:
            if keyword.arg == kw:
                return self._const_int(keyword.value)
        if pos < len(call_node.args):
            return self._const_int(call_node.args[pos])
        return None

    def _literal_dims(self, args: List[ast.AST]) -> Optional[List[int]]:
        """Permutation dims from either a single list/tuple arg (permute((0,2,1)))
        or several int args (permute(0,2,1)); None if any element isn't an int."""
        elts = args[0].elts if len(args) == 1 and isinstance(args[0], (ast.List, ast.Tuple)) else args
        vals: List[int] = []
        for node in elts:
            value = self._const_int(node)
            if value is None:
                return None
            vals.append(value)
        return vals or None

    @staticmethod
    def _provenance(info: Dict[str, Any], source_file: Optional[str]) -> Dict[str, Any]:
        """The `_`-prefixed provenance keys for one node (task.md §6.1).

        They go into `parameters` because that is the ONLY per-node channel that
        survives every plane: `_`-prefixed keys are stripped before torch
        constructors and code generation (core/layer_builder.py,
        services/code_generation.py, services/graph_modularizer.py) and
        `parameters` is kept whole by the frontend's save allowlist. A node that
        cannot say which line it came from cannot be edited in place, which is
        the whole point of the code<->graph pivot.
        """
        out: Dict[str, Any] = {}
        if source_file:
            out["_source_file"] = source_file
        span = info.get("span")
        if span is not None and getattr(span, "lineno", None) is not None:
            out["_lineno"] = span.lineno
            out["_end_lineno"] = getattr(span, "end_lineno", None) or span.lineno
            out["_col_offset"] = span.col_offset
            end_col = getattr(span, "end_col_offset", None)
            out["_end_col_offset"] = span.col_offset if end_col is None else end_col
        if info.get("raw_call"):
            out["_raw_call"] = info["raw_call"]
        call = info.get("call")
        if isinstance(call, ast.Call) and isinstance(call.func, (ast.Name, ast.Attribute)):
            # The result-producing call is distinct from calls preparing its
            # arguments. A conditional argument does not change that outer call.
            result_call = {"name": call.func.id if isinstance(call.func, ast.Name) else call.func.attr}
            if getattr(call, '_torch_target', None):
                result_call['target'] = call._torch_target
            out['_source_result_call'] = result_call
        if isinstance(call, ast.AST) and not any(isinstance(item, (
            ast.IfExp, ast.BoolOp, ast.Compare, ast.Lambda, ast.Dict,
            ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
        )) for item in ast.walk(call)):
            # Function and argument expressions evaluate before the outer call.
            # This describes source call order, never observed tensor execution.
            calls: List[str] = []

            def visit(item: ast.AST) -> None:
                if item is not call and getattr(item, '_tensorviz_emitted', False):
                    return  # This child already owns a separate card.
                for child in ast.iter_child_nodes(item):
                    visit(child)
                if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute):
                    calls.append(item.func.attr)

            visit(call)
            out["_source_calls"] = calls
        if info.get("control_flow"):
            out["_control_flow"] = True
        if info.get("custom_class"):
            out["_class"] = info["custom_class"]
        return out

    def _create_synth_node(
        self, node_id: str, info: Dict[str, Any], index: int,
        source_file: Optional[str] = None,
    ) -> Node:
        """Build a graph node for something forward() creates. "functional" ops
        (and merges) carry their full torch canonical name (e.g. 'torch.permute',
        'torch.add') so the code generator emits them in the forward pass;
        "module" ops reuse the normal layer-node path so they re-export as
        `nn.<Class>(...)`; "opaque" ops are statements we did not model at all."""
        params = dict(info.get("parameters") or {})
        params.update(self._provenance(info, source_file))
        if info["emit"] == "opaque":
            return Node(
                id=node_id,
                type="layerNode",
                position={"x": (index + 1) * 200 + 50, "y": 200},
                data=LayerData(
                    layerType=LayerType(id="python", name="Python", parameters=[]),
                    parameters=params,
                )
            )
        if info["emit"] == "functional":
            name = info["type"]
            op_id = name.split(".")[-1].lower()
            return Node(
                id=node_id,
                type="layerNode",
                position={"x": (index + 1) * 200 + 50, "y": 200},
                data=LayerData(
                    layerType=LayerType(id=op_id, name=name, parameters=[]),
                    parameters=params,
                )
            )
        return self._create_layer_node(
            node_id,
            {"layer_type": info["type"], "parameters": params},
            index,
        )
    
    def _create_layer_node(
        self,
        layer_name: str,
        layer_info: Dict[str, Any],
        index: int,
        source_file: Optional[str] = None,
        model_classes: Optional[Dict[str, ast.ClassDef]] = None,
        _expanding: Optional[Set[str]] = None,
        discovered_input_shape: Optional[List[int]] = None,
    ) -> Node:
        """Create a node from layer information"""
        raw_type = layer_info["layer_type"]
        parameters = dict(layer_info.get("parameters") or {})
        parameters.update(self._provenance(layer_info, source_file))

        if "." in raw_type:
            parameters.update({"_class": raw_type.split(".")[-1], "_opaque": True})
            return Node(
                id=layer_name, type="layerNode",
                position={"x": (index + 1) * 200 + 50, "y": 200},
                data=LayerData(
                    layerType=LayerType(id="custom_code", name="custom_code", parameters=[]),
                    parameters=parameters,
                ),
            )

        # `self.blk = MyBlock(...)` where MyBlock is an nn.Module defined in the
        # same file. There is no OpSpec for a user's class and there must not be
        # one (task.md D5), so it becomes the `custom_code` placeholder carrying
        # the class name and that class's own static subgraph — the link between
        # the two classes, which used to be dropped (C5).
        custom_class = layer_info.get("custom_class")
        if custom_class:
            internal = self._static_subgraph(
                custom_class, model_classes, source_file, _expanding, discovered_input_shape
            )
            return Node(
                id=layer_name,
                type="layerNode",
                position={"x": (index + 1) * 200 + 50, "y": 200},
                data=LayerData(
                    layerType=LayerType(id="custom_code", name="custom_code", parameters=[]),
                    parameters=parameters,
                    internalGraph=internal,
                )
            )

        # Resolve the canonical class name (e.g. 'relu' -> id 'relu', name
        # 'ReLU') so re-export emits a valid `nn.<RealName>` instead of the
        # corrupted 'Relu'/'Batchnorm2d' produced by naive case transforms.
        layer_id, display_name = self._canonical_names(raw_type)

        # Map to our layer type system
        mapped_type = self.layer_mapping.get(layer_id, layer_id)

        return Node(
            id=layer_name,
            # ModuleList declares membership, not a forward operation. Keep the
            # member descriptions in parameters; they are not captured children.
            type="moduleNode" if raw_type == "ModuleList" else "layerNode",
            position={"x": (index + 1) * 200 + 50, "y": 200},
            data=LayerData(
                layerType=LayerType(
                    id=mapped_type,
                    name=display_name,
                    parameters=[]
                ),
                parameters=parameters,
                moduleType="container" if raw_type == "ModuleList" else None,
            )
        )

    def _static_subgraph(
        self,
        class_name: str,
        model_classes: Optional[Dict[str, ast.ClassDef]],
        source_file: Optional[str],
        expanding: Optional[Set[str]],
        discovered_input_shape: Optional[List[int]],
    ) -> Optional[Dict[str, Any]]:
        """The referenced class's own static subgraph, or None when it cannot be
        produced. `expanding` breaks mutual references (A holds a B, B holds an A)
        which would otherwise recurse forever."""
        class_node = (model_classes or {}).get(class_name)
        if class_node is None or class_name in (expanding or set()):
            return None
        try:
            child = self._convert_class_to_graph(
                class_name, class_node, class_name, discovered_input_shape,
                source_file=source_file, model_classes=model_classes,
                _expanding=expanding,
            )
        except Exception:
            # A child that fails to parse must not take the parent's graph
            # with it: the parent is still a usable graph, minus the expansion.
            return None
        return {
            "nodes": [node.model_dump() for node in child.nodes],
            "edges": [edge.model_dump() for edge in child.edges],
        }

    def _canonical_names(self, raw_type: str) -> Tuple[str, str]:
        """Return (layer_id, display_name) for a raw layer class name, using the
        op registry's canonical name when available. `display_name` is the exact
        PyTorch class name (e.g. 'BatchNorm2d', 'LSTM'); `layer_id` is its
        lowercase form used by the layer-type id / op-id conventions."""
        spec = spec_for(raw_type)
        if spec:
            display_name = spec.canonical.split(".")[-1]
        else:
            display_name = raw_type
        return display_name.lower(), display_name
    
    def _create_edges_from_connections(self, connections: List[Tuple[str, str]]) -> List[Edge]:
        """Create edges from connection tuples"""
        edges = []
        for i, (source, target) in enumerate(connections):
            edge_id = f"edge_{source}_{target}_{i}"
            edges.append(Edge(id=edge_id, source=source, target=target))
        return edges
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements from AST"""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)
        
        return imports
    
    def _detect_hierarchy_level(self, class_node: ast.ClassDef) -> str:
        """Detect the hierarchy level of a model class"""
        # Simple heuristic based on class complexity
        num_methods = len([n for n in class_node.body if isinstance(n, ast.FunctionDef)])
        num_assignments = len([n for n in ast.walk(class_node) if isinstance(n, ast.Assign)])
        
        if num_methods <= 2 and num_assignments <= 5:
            return "module"
        elif num_methods <= 4 and num_assignments <= 15:
            return "block"
        else:
            return "model"
    
    def _create_layer_mapping(self) -> Dict[str, str]:
        """Create mapping from PyTorch layer names to our layer IDs"""
        return {
            "conv2d": "conv2d",
            "linear": "linear",
            "relu": "relu",
            "batchnorm2d": "batchnorm2d",
            "maxpool2d": "maxpool2d",
            "avgpool2d": "avgpool2d",
            "dropout": "dropout",
            "flatten": "flatten",
            "sigmoid": "sigmoid",
            "tanh": "tanh",
            "softmax": "softmax",
            "lstm": "lstm",
            "gru": "gru",
            "embedding": "embedding",
            "layernorm": "layernorm",
            "groupnorm": "groupnorm",
            "instancenorm2d": "instancenorm2d",
            "leakyrelu": "leakyrelu",
            "elu": "elu",
            "selu": "selu",
            "gelu": "gelu",
            "swish": "swish"
        }
    
    def _create_import_mapping(self) -> Dict[str, str]:
        """Create mapping from import statements to framework detection"""
        return {
            "torch": "pytorch",
            "torch.nn": "pytorch",
            "torch.nn.functional": "pytorch",
            "torchvision": "pytorch",
            "tensorflow": "tensorflow",
            "keras": "keras",
            "jax": "jax",
            "flax": "flax"
        }
