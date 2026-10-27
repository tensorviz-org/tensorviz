"""Readable boundary names derived from recorded containers, never guessed roles."""
import json
import re

from .contracts import MappingArgument, SequenceArgument


def argument_names(argument, path=(), name=""):
    if isinstance(argument, SequenceArgument) and argument.items:
        for index, item in enumerate(argument.items):
            child = f"{name}[{index}]" if name else str(index + 1)
            yield from argument_names(item, (*path, "items", index), child)
    elif isinstance(argument, MappingArgument) and argument.entries:
        for index, entry in enumerate(argument.entries):
            key = entry.key
            if key.kind == "str" and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key.value):
                child = f"{name}.{key.value}" if name else key.value
            else:
                literal = "None" if key.kind == "none" else str(key.value) if key.kind in ("bool", "int") else json.dumps(key.value, ensure_ascii=False)
                child = f"{name}[{literal}]"
            yield from argument_names(entry.value, (*path, "entries", index, "value"), child)
    else:
        yield path, name


def port_names(arguments, output, field="arguments"):
    names = {}
    if arguments is not None:
        for index, argument in enumerate(arguments.args):
            names.update(argument_names(argument, (field, "args", index), str(index + 1)))
        for key, argument in arguments.kwargs.items():
            names.update(argument_names(argument, (field, "kwargs", key), key))
    if output is not None:
        names.update(argument_names(output, ("output",)))
    return names
