"""Non-executing argument and return-value bookkeeping for local helper calls."""
import ast
import inspect
from collections import Counter

from services.validation.signatures import _signature


UNKNOWN = object()


class Flow(list):
    """Tensor producers, optionally retaining a literal or container fields."""
    def __init__(self, nodes=(), *, fields=None, literal=UNKNOWN):
        super().__init__(nodes)
        self.fields = fields
        self.literal = literal

    @classmethod
    def container(cls, fields):
        return cls((node for value in fields.values() for node in value), fields=fields)

    @classmethod
    def merge(cls, values):
        if len(values) == 1:
            return values[0]
        fields = [value.fields if isinstance(value, cls) else None for value in values]
        if fields and all(field is not None for field in fields):
            keys = set(fields[0]).intersection(*(set(field) for field in fields[1:]))
            return cls((node for value in values for node in value), fields={
                key: cls.merge([field[key] for field in fields]) for key in keys})
        return cls(node for value in values for node in value)


def local_methods(definition):
    """Only ordinary, uniquely defined, non-rebound instance methods are callable."""
    if definition is None or definition.decorator_list:
        return {}
    counts = Counter(node.name for node in definition.body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
    if any(name in counts for name in ('__getattribute__', '__getattr__')):
        return {}
    rebound = {node.attr for node in ast.walk(definition)
               if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del))}
    rebound.update(child.id for statement in definition.body
                   if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                   for child in ast.walk(statement) if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)))
    if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
           and node.func.id in ('setattr', 'delattr', 'exec') for node in ast.walk(definition)):
        return {}
    result = {}
    for node in definition.body:
        if not isinstance(node, ast.FunctionDef) or counts[node.name] != 1 or node.name in rebound:
            continue
        positional = node.args.posonlyargs + node.args.args
        if not positional or positional[0].arg != 'self' or _signature(node) is None:
            continue
        if any(isinstance(child, (ast.Yield, ast.YieldFrom, ast.Await, ast.ListComp,
                                  ast.DictComp, ast.SetComp, ast.GeneratorExp, ast.Global, ast.Nonlocal))
               for child in ast.walk(node)):
            continue
        result[node.name] = node
    return result


def bind_helper(method, call, resolve):
    """Use Python's binder on symbolic values; never evaluate project code."""
    signature = _signature(method)
    if signature is None:
        return None
    args, kwargs = [], {}
    for argument in call.args:
        value = resolve(argument.value if isinstance(argument, ast.Starred) else argument)
        if isinstance(argument, ast.Starred):
            fields = value.fields if isinstance(value, Flow) else None
            if fields is None or list(fields) != list(range(len(fields))):
                return None
            args.extend(fields.values())
        else:
            args.append(value)
    for keyword in call.keywords:
        value = resolve(keyword.value)
        fields = value.fields if isinstance(value, Flow) else None
        additions = {keyword.arg: value} if keyword.arg is not None else fields
        if additions is None or any(not isinstance(key, str) or key in kwargs for key in additions):
            return None
        kwargs.update(additions)
    try:
        bound = signature.bind(Flow(), *args, **kwargs)
    except TypeError:
        return None
    positional = method.args.posonlyargs + method.args.args
    defaults = dict(zip([arg.arg for arg in positional][-len(method.args.defaults):], method.args.defaults))
    defaults.update({arg.arg: default for arg, default in zip(method.args.kwonlyargs, method.args.kw_defaults)
                     if default is not None})
    values = {}
    for name, parameter in signature.parameters.items():
        if name in bound.arguments:
            value = bound.arguments[name]
            if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                value = Flow.container(dict(enumerate(value)))
            elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
                value = Flow.container(value)
            values[name] = value
        elif name in defaults:
            # Defaults are definition-scope expressions, never caller locals.
            try:
                values[name] = Flow(literal=ast.literal_eval(defaults[name]))
            except (ValueError, TypeError):
                values[name] = Flow()
        else:
            values[name] = Flow.container({})
    return values
