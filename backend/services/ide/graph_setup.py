"""Read model setup from supplied source. Never import or evaluate project code."""
from __future__ import annotations

import ast
import json
from typing import Any

from services.validation.signatures import forward_declaration

from .module_index import ModuleIndex, module_of_file, package_of_file
from .multifile import MultifileProjectionRequest, validate_root, validate_sources
from .source_decorators import imported_symbol, known_decorators

_MISSING = object()
_MAX_CHOICES = 64


class UnsupportedModelClass(ValueError):
    """A source-proven unsupported class, safe to explain at the API boundary."""


def argument(value: Any) -> Any:
    """Literal Python values expressed in the existing profile binding grammar."""
    if value is None:
        return {"kind": "none"}
    if type(value) is float:
        return {"kind": "float", "value": value}
    if type(value) in (str, bool, int):
        return value
    if isinstance(value, (list, tuple)):
        return {"kind": "tuple" if isinstance(value, tuple) else "list",
                "items": [argument(item) for item in value]}
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {"kind": "dict", "items": {key: argument(item) for key, item in value.items()}}
    raise ValueError("Use a factory for this Python value.")


def _literal(node):
    try:
        value = ast.literal_eval(node)
        # Bound defaults/examples independently of the supplied source budget.
        if len(json.dumps(value, allow_nan=False)) <= 4096:
            argument(value)
            return value
    except (ValueError, TypeError, SyntaxError, RecursionError, OverflowError):
        pass
    return _MISSING


def _parameters(function, *, method=True, decorators_known=False, file=None):
    if not isinstance(function, ast.FunctionDef) or (function.decorator_list and not decorators_known):
        return None
    args = function.args
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    entries = [(param, default, "positional" if i < len(args.posonlyargs) else "keyword")
               for i, (param, default) in enumerate(zip(positional, defaults))]
    if method:
        if not entries:
            return None
        entries = entries[1:]
    entries += [(param, default, "keyword_only") for param, default in zip(args.kwonlyargs, args.kw_defaults)]
    result = []
    for param, default, kind in entries:
        field = {"name": param.arg, "kind": kind, "required": default is None}
        if file and getattr(param, "lineno", None):
            field["source"] = dict(file=file, line=param.lineno)
        if param.annotation is not None:
            field["annotation"] = ast.unparse(param.annotation)[:160]
        if default is not None:
            value = _literal(default)
            field["defaultText"] = ast.unparse(default)[:160]
            if value is not _MISSING:
                field["defaultValue"] = value
        result.append(field)
    for param, kind in [(args.vararg, "varargs"), (args.kwarg, "kwargs")]:
        if param is not None:
            field = {"name": param.arg, "kind": kind, "required": False}
            if file and getattr(param, "lineno", None):
                field["source"] = dict(file=file, line=param.lineno)
            result.append(field)
    return result if len(result) <= 64 else None


class Discovery:
    def __init__(self, sources: dict[str, str]):
        self.sources = sources
        self.index = ModuleIndex.build(sources)
        self.trees = {}
        self.import_symbols = {}
        for file, source in sources.items():
            try:
                self.trees[file] = ast.parse(source)
            except (SyntaxError, ValueError, RecursionError):
                continue

    def definition(self, file, name):
        tree = self.trees.get(file)
        definitions = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name] if tree else []
        return definitions[0] if len(definitions) == 1 else None

    def resolve(self, file, expression):
        result = self.index.resolve(module_of_file(file), ast.unparse(expression))
        return result.definition

    def torch_symbol(self, file, expression):
        """Resolve ordinary torch imports; dynamic or shadowed bindings stay unknown."""
        parts = ast.unparse(expression).split(".")
        key = (file, parts[0])
        if key in self.import_symbols:
            prefix = self.import_symbols[key]
            return prefix + ("." + ".".join(parts[1:]) if len(parts) > 1 else "") if prefix else None
        tree = self.trees[file]
        bindings = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                bindings += [alias.name if alias.asname else alias.name.split(".")[0]
                             for alias in node.names if (alias.asname or alias.name.split(".")[0]) == parts[0]]
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                bindings += [node.module + "." + alias.name
                             for alias in node.names if (alias.asname or alias.name) == parts[0]]
            elif any(isinstance(child, ast.Name) and child.id == parts[0] and isinstance(child.ctx, ast.Store)
                     for child in ast.walk(node)):
                self.import_symbols[key] = None
                return None
        prefix = bindings[0] if len(bindings) == 1 else None
        self.import_symbols[key] = prefix
        return prefix + ("." + ".".join(parts[1:]) if len(parts) > 1 else "") if prefix else None

    def method(self, file, cls, name, seen=()):
        key = (file, cls)
        if key in seen or len(seen) >= 16:
            return None
        definition = self.definition(file, cls)
        if definition is None or not self.decorators_supported(file, definition) or definition.keywords:
            return None
        own = [node for node in definition.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
        if len(own) == 1:
            return _parameters(forward_declaration(definition) if name == "forward" and not seen else own[0],
                               decorators_known=self.decorators_supported(file, own[0], definition), file=file)
        if len(own) > 1 or len(definition.bases) != 1:
            return None
        base = definition.bases[0]
        if name == "__init__" and self.torch_symbol(file, base) == "torch.nn.Module":
            return []
        parent = self.resolve(file, base)
        return self.method(parent.file, parent.name, name, (*seen, key)) if parent else None

    def module_subclass(self, file, cls, seen=()):
        """Prove local inheritance where possible; external/dynamic bases stay unknown."""
        key = (file, cls)
        definition = self.definition(file, cls)
        if (key in seen or len(seen) >= 16 or definition is None
                or definition.decorator_list or definition.keywords):
            return None
        from services.ingest.module_discovery import TORCH_MODULE_BASES, EXTERNAL_MODULE_BASES
        record = self.index.modules[module_of_file(file)]
        bases = []
        for base in definition.bases:
            parent = self.resolve(file, base)
            if parent:
                bases.append(self.module_subclass(parent.file, parent.name, (*seen, key)))
            elif (isinstance(base, ast.Name) and base.id == 'object'
                  and 'object' not in record.bound_names and not record.unresolved_wildcard):
                bases.append(False)
            elif self.torch_symbol(file, base) in TORCH_MODULE_BASES | EXTERNAL_MODULE_BASES:
                bases.append(True)
            else:
                bases.append(None)
        if any(base is True for base in bases):
            return True
        return None if None in bases else False

    def decorators_supported(self, file, definition, owner=None):
        tree = self.trees[file]
        if owner is not None:
            # Method decorators are evaluated in the class namespace too.
            tree = ast.Module(body=[*tree.body, *owner.body], type_ignores=[])
        return known_decorators(tree, definition, package_of_file(file),
                                constructor=isinstance(definition, ast.FunctionDef) and definition.name == "__init__")

    def tensor(self, file, node):
        if not isinstance(node, ast.Call):
            return None
        symbol = self.torch_symbol(file, node.func)
        if symbol not in {"torch.randn", "torch.rand", "torch.zeros", "torch.ones", "torch.randint"}:
            return None
        positional = [_literal(arg) for arg in node.args]
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        if _MISSING in positional or set(kwargs) - {"dtype", "requires_grad"}:
            return None
        dtype = self.torch_symbol(file, kwargs["dtype"]) if "dtype" in kwargs else None
        if "dtype" in kwargs and dtype is None:
            return None
        dtype = dtype.removeprefix("torch.") if dtype else ("int64" if symbol == "torch.randint" else "float32")
        aliases = {"long": "int64", "int": "int32", "float": "float32", "double": "float64", "half": "float16"}
        dtype = aliases.get(dtype, dtype)
        if symbol == "torch.randint":
            if len(positional) != 3:
                return None
            low, high, shape = positional
            generator = {"kind": "randint", "low": low, "high": high, "seed": 0}
        else:
            shape = positional[0] if len(positional) == 1 and isinstance(positional[0], (list, tuple)) else positional
            generator = ({"kind": "normal", "mean": 0, "std": 1, "seed": 0} if symbol == "torch.randn"
                         else {"kind": "uniform", "low": 0, "high": 1, "seed": 0} if symbol == "torch.rand"
                         else {"kind": "full", "value": int(symbol == "torch.ones")})
        if not isinstance(shape, (list, tuple)) or not all(type(dim) is int and dim >= 0 for dim in shape):
            return None
        requires = _literal(kwargs["requires_grad"]) if "requires_grad" in kwargs else False
        from services.validation.contracts import TensorValue
        try:
            return TensorValue.model_validate(dict(kind="tensor", shape=list(shape), dtype=dtype,
                                                   requires_grad=requires, generator=generator)).model_dump()
        except ValueError:
            return None

    def binding(self, file, call, values=None):
        values = values or {}
        def value(node):
            if isinstance(node, ast.Name) and node.id in values:
                return values[node.id]
            tensor = self.tensor(file, node)
            if tensor is not None:
                return tensor
            literal = _literal(node)
            if literal is _MISSING:
                raise ValueError("Not a literal binding")
            return argument(literal)
        try:
            if any(kw.arg is None for kw in call.keywords):
                return None
            return {"args": [value(arg) for arg in call.args],
                    "kwargs": {kw.arg: value(kw.value) for kw in call.keywords}}
        except (ValueError, RecursionError):
            return None

    def examples(self, file, cls):
        examples, factories, parents = [], [], []
        def matches(origin, call):
            target = self.resolve(origin, call.func) if isinstance(call, ast.Call) else None
            return target and (target.file, target.name) == (file, cls)
        def factory(origin, owner):
            record = self.index.modules.get(module_of_file(origin))
            if (not isinstance(owner, ast.FunctionDef) or owner not in self.trees[origin].body
                    or record is None or owner.name in record.uncertain_names or record.unresolved_wildcard):
                return
            fields = _parameters(owner, method=False,
                                 decorators_known=self.decorators_supported(origin, owner), file=origin)
            if fields is not None:
                factories.append(dict(file=origin, callable=owner.name, fields=fields, line=owner.lineno))

        for origin, tree in self.trees.items():
            # Public builders often delegate through helpers rather than directly
            # returning Class(...). A resolved return annotation offers a factory;
            # execution still has to verify its actual result is the selected class.
            for owner in tree.body:
                annotation = owner.returns if isinstance(owner, ast.FunctionDef) else None
                if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
                    try:
                        annotation = ast.parse(annotation.value, mode="eval").body
                    except (SyntaxError, ValueError, RecursionError):
                        annotation = None
                target = self.resolve(origin, annotation) if isinstance(annotation, (ast.Name, ast.Attribute)) else None
                if target and (target.file, target.name) == (file, cls):
                    factory(origin, owner)
            for owner in ast.walk(tree):
                body = getattr(owner, "body", None)
                if not isinstance(body, list):
                    continue
                # Do not carry values between scopes or through branches/reassignments.
                models, values = {}, {}
                for statement in body:
                    if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                        destinations = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                        call = statement.value
                        for destination in destinations:
                            if isinstance(destination, ast.Name):
                                models.pop(destination.id, None)
                                values.pop(destination.id, None)
                                if matches(origin, call):
                                    binding = self.binding(origin, call)
                                    if binding is not None:
                                        models[destination.id] = binding
                                elif call is not None:
                                    tensor = self.tensor(origin, call)
                                    if tensor is not None:
                                        values[destination.id] = tensor
                            if isinstance(destination, ast.Attribute) and isinstance(destination.value, ast.Name) and destination.value.id == "self" and matches(origin, call):
                                enclosing = next((candidate for candidate in tree.body if isinstance(candidate, ast.ClassDef)
                                                  and candidate.lineno <= statement.lineno <= candidate.end_lineno), None)
                                if enclosing and (origin, enclosing.name) != (file, cls):
                                    parents.append(dict(file=origin, cls=enclosing.name, attribute=destination.attr, line=statement.lineno))
                    elif isinstance(statement, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.Delete, ast.AugAssign)):
                        models.clear()
                        values.clear()
                    # Only direct calls in this statement, not nested scopes.
                    expression = statement.value if isinstance(statement, (ast.Expr, ast.Assign, ast.AnnAssign, ast.Return)) else None
                    if isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name) and expression.func.id in models:
                        binding = self.binding(origin, expression, values)
                        if binding is not None:
                            examples.append(dict(file=origin, line=statement.lineno, init=models[expression.func.id], inputs=binding))
                    if isinstance(statement, ast.Return) and matches(origin, statement.value) and isinstance(owner, ast.FunctionDef):
                        factory(origin, owner)
                    # Unknown calls or object writes may mutate a model/input recipe.
                    if ((expression is not None and any(isinstance(call, ast.Call)
                         and not matches(origin, call) and self.tensor(origin, call) is None
                         and not (isinstance(call.func, ast.Name) and call.func.id in models)
                         for call in ast.walk(expression)))
                        or isinstance(statement, (ast.Assign, ast.AnnAssign)) and any(
                            not isinstance(target, ast.Name) for target in destinations)):
                        models.clear()
                        values.clear()
            for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
                # A freshly constructed instance can be called inline, including
                # inside print(...). Reuse only self-contained literal/tensor
                # bindings; surrounding calls cannot establish mutable aliases.
                if isinstance(call.func, ast.Call) and matches(origin, call.func):
                    init = self.binding(origin, call.func)
                    inputs = self.binding(origin, call)
                    if init is not None and inputs is not None:
                        examples.append(dict(file=origin, line=call.lineno, init=init, inputs=inputs))
                if not matches(origin, call):
                    continue
                binding = self.binding(origin, call)
                if binding is not None:
                    examples.append(dict(file=origin, line=call.lineno, init=binding))
        def unique(items):
            seen, result = set(), []
            for item in items:
                key = json.dumps(item, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    result.append(item)
            return result[:_MAX_CHOICES]
        return unique(examples), unique(factories), unique(parents)

    def configs(self, file, cls, init):
        candidates = {}
        for parameter in init or []:
            annotation = parameter.get("annotation")
            if annotation:
                owner_file = parameter.get("source", {}).get("file", file)
                if self.definition(owner_file, annotation) is not None:
                    candidates[(owner_file, annotation)] = parameter["name"]
                    continue
                target = self.index.resolve(module_of_file(owner_file), annotation).definition
                if target:
                    candidates[(target.file, target.name)] = parameter["name"]
        # Named config types are offered for selection, never constructed implicitly.
        for node in self.trees[file].body:
            if isinstance(node, ast.ClassDef) and (node.name.endswith(("Config", "Dimensions"))):
                candidates.setdefault((file, node.name), None)
        result = []
        for (origin, name), parameter in candidates.items():
            definition = self.definition(origin, name)
            if definition is None:
                continue
            constructor = next((node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"), None)
            fields = _parameters(constructor, decorators_known=self.decorators_supported(origin, constructor, definition), file=origin) if constructor else []
            dataclass = any(imported_symbol(self.trees[origin], node.func if isinstance(node, ast.Call) else node,
                                           package_of_file(origin)) in {"dataclasses.dataclass", "huggingface_hub.dataclasses.strict"}
                            for node in definition.decorator_list)
            if constructor is None:
                for node in definition.body:
                    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                        field = dict(name=node.target.id, kind="keyword", required=node.value is None,
                                     source=dict(file=origin, line=node.lineno))
                        value = _literal(node.value) if node.value is not None else _MISSING
                        if value is not _MISSING:
                            field.update(defaultValue=value, defaultText=ast.unparse(node.value))
                        fields.append(field)
            result.append(dict(file=origin, cls=name, parameter=parameter, fields=fields if fields is None or len(fields) <= 64 else None,
                               construction="call" if constructor is not None or dataclass else "attributes"))
        return result[:_MAX_CHOICES]

    def constraints(self, file, cls):
        """Only direct input uses establish dimensions; no batch or resolution guesses."""
        definition = self.definition(file, cls)
        if definition is None:
            return []
        constructor = next((node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"), None)
        forward = next((node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name == "forward"), None)
        if constructor is None or forward is None or forward.decorator_list:
            return []
        layers = {}
        for statement in constructor.body:
            if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call):
                for target in statement.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                        layers[target.attr] = statement.value
        inputs = {field["name"] for field in _parameters(forward) or []}
        result = []
        for statement in forward.body:
            # Control flow makes first-use facts uncertain; leave it for explicit setup.
            if not isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Return, ast.Expr)):
                break
            expression = statement.value
            for call in ast.walk(expression) if expression is not None else []:
                if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                        and isinstance(call.func.value, ast.Name) and call.func.value.id == "self"
                        and call.args and isinstance(call.args[0], ast.Name) and call.args[0].id in inputs):
                    continue
                layer = layers.get(call.func.attr)
                if layer is None:
                    continue
                symbol = self.torch_symbol(file, layer.func)
                axis, key = {"torch.nn.Linear": (-1, "in_features"), "torch.nn.Conv1d": (-2, "in_channels"),
                             "torch.nn.Conv2d": (-3, "in_channels"), "torch.nn.Conv3d": (-4, "in_channels")}.get(symbol, (None, None))
                if axis is None:
                    continue
                size = next((kw.value for kw in layer.keywords if kw.arg == key), layer.args[0] if layer.args else None)
                if size is None:
                    continue
                literal = _literal(size)
                fact = dict(input=call.args[0].id, axis=axis, file=file, line=layer.lineno)
                if type(literal) is int:
                    fact["size"] = literal
                elif isinstance(size, ast.Name) and size.id in {field["name"] for field in _parameters(constructor) or []}:
                    fact["initParameter"] = size.id
                else:
                    continue
                result.append(fact)
            destinations = statement.targets if isinstance(statement, ast.Assign) else [statement.target] if isinstance(statement, ast.AnnAssign) else []
            for target in destinations:
                inputs -= {node.id for node in ast.walk(target) if isinstance(node, ast.Name)}
        return result[:_MAX_CHOICES]


def discover(request: MultifileProjectionRequest, *, resolve_inputs=True, reject_unsupported=False) -> dict:
    if request.protocol_version != 1:
        raise ValueError("Unsupported source discovery version.")
    entries = validate_sources(request.sources)
    root = validate_root(request.root, entries)
    from services.jax_support.discovery import discover as discover_jax
    jax = discover_jax({file: entry.source for file, entry in entries.items()}, root.file, root.cls, root.source_digest)
    if jax is not None:
        jax['truncated'] = request.truncated
        return jax
    reader = Discovery({file: entry.source for file, entry in entries.items()})
    definition = reader.definition(root.file, root.cls)
    if definition is None:
        raise ValueError("The selected class needs valid, unambiguous Python source.")
    if reader.module_subclass(root.file, root.cls) is False:
        reason = (f"{root.cls} is not a torch.nn.Module subclass; "
                  "tensorViz captures PyTorch nn.Module (and Flax) models.")
        if reject_unsupported:
            raise UnsupportedModelClass(reason)
        model = f"{module_of_file(root.file)}:{root.cls}"
        profile = dict(kind='model', model=model, mode='eval', device='cpu', seed=0,
                       grad_enabled=False, init=dict(args=[], kwargs={}), inputs=dict(args=[], kwargs={}))
        # Use the normal setup response so hosts display this requirement even
        # when they replace HTTP error bodies with a generic transport message.
        return dict(model=model, init=None, inputs=None, examples=[], factories=[], configs=[], parents=[],
                    constraints=[], truncated=request.truncated, sourceDigest=root.source_digest,
                    resolution=dict(status='partial', draft=dict(profile=profile), evidence=[], unresolved=[reason]))
    init = reader.method(root.file, root.cls, "__init__")
    inputs = reader.method(root.file, root.cls, "forward")
    examples, factories, parents = reader.examples(root.file, root.cls) if resolve_inputs else ([], [], [])
    result = dict(model=f"{module_of_file(root.file)}:{root.cls}", init=init, inputs=inputs,
                examples=examples, factories=factories, configs=reader.configs(root.file, root.cls, init),
                parents=parents, constraints=reader.constraints(root.file, root.cls) if resolve_inputs else [], truncated=request.truncated,
                sourceDigest=root.source_digest)
    if resolve_inputs:
        from .graph_setup_resolution import resolve_setup
        result['resolution'] = resolve_setup(reader, root.file, root.cls, result)
    return result
