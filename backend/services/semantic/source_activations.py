"""Closed activation signatures and observed mutation proof; never evaluate source."""
import ast

from .contracts import ActivationSourceOptions, ValueRef

NAMES = ["relu", "gelu", "silu", "sigmoid", "tanh"]
CLASSES = dict(relu="ReLU", gelu="GELU", silu="SiLU", sigmoid="Sigmoid", tanh="Tanh")
TARGETS = {name: f"aten.{name}.default" for name in NAMES}
INPLACE_TARGETS = {name: f"aten.{name}_.default" for name in ("relu", "silu")}


def source_options(node, expected, *, module=False):
    """Bind only known source signatures, retaining every original argument byte."""
    if any(isinstance(arg, ast.Starred) for arg in node.args) or any(kw.arg is None for kw in node.keywords):
        return None
    keywords = {kw.arg: kw.value for kw in node.keywords}
    if len(keywords) != len(node.keywords):
        return None
    positional = list(node.args)
    if not module:
        if positional:
            positional.pop(0)
            if "input" in keywords:
                return None
        elif "input" in keywords:
            keywords.pop("input")
        else:
            return None
    option = "inplace" if expected in ("relu", "silu") else "approximate" if expected == "gelu" else None
    if positional:
        # functional GELU's approximate is keyword-only; constructor accepts it positionally.
        if option is None or len(positional) != 1 or option in keywords or (expected == "gelu" and not module):
            return None
        keywords[option] = positional[0]
    if not keywords:
        return ActivationSourceOptions(), list(NAMES)
    if list(keywords) != [option]:
        return None
    value = keywords[option]
    if not isinstance(value, ast.Constant):
        return None
    if option == "inplace" and type(value.value) is bool:
        return ActivationSourceOptions(inplace=value.value), ["relu", "silu"]
    if option == "approximate" and type(value.value) is str and value.value in ("none", "tanh"):
        return ActivationSourceOptions(approximate=value.value), ["gelu"]
    return None


def invocation_input(arguments):
    if len(arguments.args) == 1 and not arguments.kwargs and isinstance(arguments.args[0], ValueRef):
        return arguments.args[0]
    if not arguments.args and list(arguments.kwargs) == ["input"] and isinstance(arguments.kwargs["input"], ValueRef):
        return arguments.kwargs["input"]
    return None


def observed_activation(op, expected, options, values):
    """Prove the single known kernel, including the exact retained write/alias chain."""
    inplace = options.inplace is True
    target = (INPLACE_TARGETS if inplace else TARGETS).get(expected)
    if (op.status != "completed" or op.target != target or len(op.arguments.args) != 1
            or not isinstance(op.arguments.args[0], ValueRef) or not isinstance(op.output, ValueRef)):
        return False
    kwargs = op.arguments.kwargs
    if kwargs:
        if (expected != "gelu" or list(kwargs) != ["approximate"]
                or kwargs["approximate"].model_dump(mode="json") != {
                    "kind": "str", "value": options.approximate or "none"}):
            return False
    elif options.approximate == "tanh":
        return False
    before, after = values[op.arguments.args[0].id], values[op.output.id]
    if (before.shape is None or before.shape != after.shape or before.dtype != after.dtype or before.device != after.device):
        return False
    if not inplace:
        return not op.effects
    writes = [effect for effect in op.effects if effect.kind == "write"]
    if (len(writes) != 1 or writes[0].beforeId != before.id or writes[0].after != op.output
            or before.storageId is None):
        return False
    for effect in op.effects:
        if effect.argumentPath != ["args", 0] or not isinstance(effect.after, ValueRef):
            return False
        old, new = values[effect.beforeId], values[effect.after.id]
        if (old.storageId != before.storageId or new.storageId != before.storageId
                or old.objectId != new.objectId or new.previousVersion != old.id
                or new.logicalVersion != old.logicalVersion + 1
                or (old.shape, old.dtype, old.device, old.stride, old.storageOffset)
                   != (new.shape, new.dtype, new.device, new.stride, new.storageOffset)):
            return False
    return True


def observed_activation_call(call, operations, children, expected, options, values):
    """One completed module call must expose its known kernel and exact boundaries."""
    if call.status != "completed" or children or len(operations) != 1 or invocation_input(call.arguments) is None:
        return False
    op = operations[0]
    return (observed_activation(op, expected, options, values)
        and op.arguments.args[0] == invocation_input(call.arguments) and op.output == call.output)
