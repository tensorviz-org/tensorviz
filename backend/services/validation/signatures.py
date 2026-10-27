"""Conservative argument preflight over supplied source; defaults are never evaluated.

Live signatures must still be checked by the approved worker. Decorators, inherited
methods and factories can change them; absence of a static finding is not proof.
"""
from __future__ import annotations

import ast
import copy
import inspect
import re

from .contracts import Binding, Requirement

_PRESENT = object()
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}\Z")


def _definition(body, name):
    definitions = [node for node in body
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                   and node.name == name]
    if len(definitions) != 1:
        return None
    # Rebinding is a live-object question, not a reason to use an earlier signature.
    for node in body:
        if node is not definitions[0] and any(
            (isinstance(child, ast.Name) and child.id == name and isinstance(child.ctx, (ast.Store, ast.Del)))
            or (isinstance(child, ast.Attribute) and isinstance(child.ctx, (ast.Store, ast.Del))
                and isinstance(child.value, ast.Name) and child.value.id == name)
            for child in ast.walk(node)
        ):
            return None
    return definitions[0]


def _signature(function):
    if not isinstance(function, ast.FunctionDef) or function.decorator_list:
        return None
    arguments = function.args
    positional = arguments.posonlyargs + arguments.args
    default_start = len(positional) - len(arguments.defaults)
    parameters = []
    for index, arg in enumerate(positional):
        kind = (inspect.Parameter.POSITIONAL_ONLY if index < len(arguments.posonlyargs)
                else inspect.Parameter.POSITIONAL_OR_KEYWORD)
        parameters.append(inspect.Parameter(arg.arg, kind, default=(
            _PRESENT if index >= default_start else inspect.Parameter.empty)))
    if arguments.vararg:
        parameters.append(inspect.Parameter(arguments.vararg.arg, inspect.Parameter.VAR_POSITIONAL))
    for arg, default in zip(arguments.kwonlyargs, arguments.kw_defaults):
        parameters.append(inspect.Parameter(arg.arg, inspect.Parameter.KEYWORD_ONLY,
                                           default=_PRESENT if default is not None else inspect.Parameter.empty))
    if arguments.kwarg:
        parameters.append(inspect.Parameter(arguments.kwarg.arg, inspect.Parameter.VAR_KEYWORD))
    try:
        return inspect.Signature(parameters)
    except ValueError:
        return None


def forward_declaration(definition):
    """Expose a direct local delegate's inputs without interpreting arbitrary Python.

    Only an unconditional first call forwarding unchanged *args and **kwargs is
    expanded. Wrapper keyword-only controls remain part of the public signature.
    Anything ambiguous keeps the original variadic declaration.
    """
    forward = _definition(definition.body, "forward")
    if not isinstance(forward, ast.FunctionDef) or _signature(forward) is None:
        return forward
    args = forward.args
    positional = args.posonlyargs + args.args
    if len(positional) != 1 or args.vararg is None or args.kwarg is None:
        return forward
    body = forward.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    first = body[0] if body else None
    call = first.value if isinstance(first, (ast.Assign, ast.AnnAssign, ast.Expr, ast.Return)) else None
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name) and call.func.value.id == positional[0].arg
            and len(call.args) == 1 and isinstance(call.args[0], ast.Starred)
            and isinstance(call.args[0].value, ast.Name) and call.args[0].value.id == args.vararg.arg
            and len(call.keywords) == 1 and call.keywords[0].arg is None
            and isinstance(call.keywords[0].value, ast.Name) and call.keywords[0].value.id == args.kwarg.arg):
        return forward
    # A second use or reassignment means this is not a transparent binding.
    for name in (args.vararg.arg, args.kwarg.arg):
        if sum(isinstance(node, ast.Name) and node.id == name for node in ast.walk(forward)) != 1:
            return forward
    delegate = _definition(definition.body, call.func.attr)
    if delegate is forward or _signature(delegate) is None:
        return forward
    delegated = delegate.args
    if delegated.vararg or delegated.kwarg or not (delegated.posonlyargs + delegated.args):
        return forward
    if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in
           {"__call__", "__getattr__", "__getattribute__"} for node in definition.body):
        return forward
    if any(isinstance(node, ast.Attribute) and node.attr == call.func.attr
           and isinstance(node.ctx, (ast.Store, ast.Del)) for node in ast.walk(definition)):
        return forward
    names = {param.arg for param in delegated.posonlyargs + delegated.args + delegated.kwonlyargs}
    if names.intersection(param.arg for param in args.kwonlyargs):
        return forward
    resolved = copy.copy(forward)
    resolved.args = copy.deepcopy(delegated)
    resolved.args.kwonlyargs.extend(args.kwonlyargs)
    resolved.args.kw_defaults.extend(args.kw_defaults)
    return resolved if _signature(resolved) is not None else forward


def _binding_requirements(function, binding: Binding, profile: str, field: str, method: bool):
    signature = _signature(function)
    if signature is None:
        return []
    prefix = ["profiles", profile, field]

    def problem(path, code, reason, remedy):
        # Python permits parameter names longer than diagnostic field segments.
        safe_path = path if all(not isinstance(p, str) or len(p) <= 128 for p in path) else prefix
        return Requirement(profile=profile, field=safe_path, phase="static", code=code,
                           reason=reason, remedy=remedy)

    args = [_PRESENT] * (len(binding.args) + int(method))
    kwargs = dict.fromkeys(binding.kwargs, _PRESENT)
    parameters = list(signature.parameters.values())
    positional = [p for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    has_kwargs = any(p.kind == p.VAR_KEYWORD for p in parameters)
    try:
        bound = signature.bind_partial(*args, **kwargs)
    except TypeError:
        # Never forward inspect's exception: use only structural names and fixed text.
        for parameter in positional[:len(args)]:
            if parameter.kind == parameter.POSITIONAL_OR_KEYWORD and parameter.name in kwargs:
                return [problem(prefix + ["kwargs", parameter.name], "signature",
                                "The parameter is supplied both positionally and by keyword.",
                                "Remove the duplicate keyword or positional argument.")]
        for name in kwargs:
            parameter = signature.parameters.get(name)
            if not has_kwargs and (parameter is None or parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_POSITIONAL
            )):
                return [problem(prefix + ["kwargs", name], "signature",
                                "This keyword cannot bind to the declared Python signature.",
                                "Remove this keyword; supply positional-only parameters in args.")]
        return [problem(prefix + ["args"], "signature", "Arguments do not bind to the Python declaration.",
                        "Match args to the declared positional parameters; remove extra arguments.")]
    result = []
    for parameter in parameters:
        if (parameter.name in bound.arguments or parameter.default is not inspect.Parameter.empty
                or parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)):
            continue
        if parameter.kind == parameter.POSITIONAL_ONLY or (
            parameter.kind == parameter.POSITIONAL_OR_KEYWORD and not _NAME.fullmatch(parameter.name)
        ):
            path = prefix + ["args", positional.index(parameter) - int(method)]
        else:
            path = prefix + ["kwargs", parameter.name]
        unsupported = parameter.kind == parameter.KEYWORD_ONLY and not _NAME.fullmatch(parameter.name)
        result.append(problem(path, "unsupported" if unsupported else "missing",
                              "A required Python parameter has no configured argument.",
                              "Use a script profile for this keyword name." if unsupported else
                              "Supply the required argument here. Use a script profile for complex objects; "
                              "only omitted parameters with Python defaults may remain unset."))
    return result


def check_bindings(source: bytes, qualified_name: str, profile: str, init: Binding | None,
                   inputs: Binding | None, *, framework=None) -> list[Requirement]:
    """Check plain declarations only; no imports, eval, default-value copying or calls."""
    try:
        module = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return [Requirement(profile=profile, field=["profiles", profile, "model"], phase="static",
                            code="invalid_source", reason="The supplied model source cannot be parsed.",
                            remedy="Fix and save the Python source, then refresh its host snapshot.")]
    body = module.body
    for part in qualified_name.split("."):
        target = _definition(body, part)
        if target is None or target.decorator_list or (isinstance(target, ast.ClassDef) and target.keywords):
            return []
        body = target.body
    if isinstance(target, ast.FunctionDef):
        if framework == 'jax':
            return _binding_requirements(target, inputs, profile, "inputs", False) if inputs is not None else []
        # A factory's return value and forward signature are runtime-only facts.
        return _binding_requirements(target, init, profile, "init", False) if init is not None else []
    if not isinstance(target, ast.ClassDef):
        return []
    constructor = _definition(body, "__init__")
    forward = forward_declaration(target)
    if framework == 'jax':
        forward = _definition(body, '__call__')
    # Custom construction/call hooks and signature overrides invalidate these inferences.
    altered = {node.name for node in body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if "__new__" in altered or any(isinstance(node, ast.Name) and node.id == "__signature__"
                                   for statement in body for node in ast.walk(statement)):
        constructor = None
    if "__call__" in altered and framework != 'jax':
        forward = None
    return ((_binding_requirements(constructor, init, profile, "init", True) if init is not None else [])
            + (_binding_requirements(forward, inputs, profile, "inputs", True) if inputs is not None else []))


def check_factory(source: bytes, name: str, profile: str, kwargs: dict) -> list[Requirement]:
    """Check only the factory declaration's binding; never execute defaults or imports."""
    try:
        module = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return [Requirement(profile=profile, field=["profiles", profile, "factory", "file"], phase="static",
                            code="invalid_source", reason="The supplied factory source cannot be parsed.",
                            remedy="Fix and save the factory source, then refresh its host snapshot.")]
    # Signature checking needs presence/positions, not materialized keyword values.
    binding = Binding(args=[], kwargs=dict.fromkeys(kwargs, 0))
    return _binding_requirements(_definition(module.body, name), binding, profile, "factory", False)
