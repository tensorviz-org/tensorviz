"""Bounded scalar parameter observations with no object protocols or tensor reads."""
import inspect
import json

from .arguments import ArgumentEncoder

MAX_RECORD_BYTES = 16384
MAX_TOTAL_BYTES = 1048576
MAX_PARAMETERS = 256
_MISSING = object()


def unavailable(reason):
    return dict(phase="initializer_entry", status="unavailable", reason=reason,
        positionalOnly=0, positional=[], keywordOnly=[], varArgs=None, varKwargs=None)


def scalar(value):
    kind = type(value)
    if value is None or kind is bool or kind is int or kind is float:
        # The existing scalar grammar handles integer bounds, nonfinite floats
        # and signed zero. Never pass arbitrary objects to its type inspection.
        result = ArgumentEncoder(None, None, lambda _reason: None).literal(value)
        if result["kind"] == "opaque":
            result["typeName"] = None
        return result
    return dict(kind="opaque", reason="unsupported_type", typeName=None)


def observe_parameters(frame, remaining_bytes):
    if remaining_bytes <= 0:
        return unavailable("byte_limit"), 0
    code = frame.f_code
    positional, keywords = code.co_argcount, code.co_kwonlyargcount
    if positional < 1 or positional - 1 + keywords > MAX_PARAMETERS:
        return unavailable("item_limit"), 0
    names = code.co_varnames
    parameter_names = names[1:positional + keywords]
    if any(len(name) > 128 for name in parameter_names):
        return unavailable("item_limit"), 0
    # Initializer entry has only Python-bound locals. Still avoid dictionary
    # probes against arbitrary keys and bound work independently of the body.
    local = {}
    for index, (name, value) in enumerate(frame.f_locals.items()):
        if index >= 8192:
            return unavailable("item_limit"), 0
        if type(name) is str:
            local[name] = value
    if any(name not in local for name in parameter_names):
        return unavailable("unsupported_type"), 0
    result = dict(phase="initializer_entry", status="observed", reason=None,
        positionalOnly=max(0, code.co_posonlyargcount - 1),
        positional=[dict(name=name, value=scalar(local[name])) for name in names[1:positional]],
        keywordOnly=[dict(name=name, value=scalar(local[name])) for name in names[positional:positional + keywords]],
        varArgs=None, varKwargs=None)
    cursor = positional + keywords
    if code.co_flags & inspect.CO_VARARGS:
        values = local.get(names[cursor], _MISSING)
        if type(values) is not tuple:
            return unavailable("unsupported_type"), 0
        if len(values) > MAX_PARAMETERS:
            return unavailable("item_limit"), 0
        result["varArgs"] = [scalar(value) for value in values]
        cursor += 1
    if code.co_flags & inspect.CO_VARKEYWORDS:
        values = local.get(names[cursor], _MISSING)
        if type(values) is not dict:
            return unavailable("unsupported_type"), 0
        if len(values) > MAX_PARAMETERS:
            return unavailable("item_limit"), 0
        encoded = {}
        for name, value in dict.items(values):
            if type(name) is not str or not 1 <= len(name) <= 128:
                return unavailable("unsupported_type"), 0
            try:
                name.encode("utf-8")
            except UnicodeEncodeError:
                return unavailable("unsupported_type"), 0
            encoded[name] = scalar(value)
        result["varKwargs"] = encoded
    amount = len(json.dumps(result, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode())
    if amount > min(remaining_bytes, MAX_RECORD_BYTES):
        return unavailable("byte_limit"), 0
    return result, amount


def source_argument(entry, path):
    """Resolve a non-expanded authored slot against actual Python bindings.

    The caller must prove the exact direct constructor and saved argument slot.
    A positional-only name in **kwargs is independent of the positional binding.
    """
    if entry is None or entry.status != "observed" or len(path) != 2:
        return None
    group, slot = path
    if group == "args" and type(slot) is int and slot >= 0:
        if slot < len(entry.positional):
            value = entry.positional[slot].value
        else:
            index = slot - len(entry.positional)
            value = entry.varArgs[index] if entry.varArgs is not None and index < len(entry.varArgs) else None
    elif group == "kwargs" and type(slot) is str:
        named = [item.value for item in [*entry.positional[entry.positionalOnly:], *entry.keywordOnly] if item.name == slot]
        value = named[0] if len(named) == 1 else (entry.varKwargs or {}).get(slot)
    else:
        return None
    return value if value is not None and value.kind != "opaque" else None
