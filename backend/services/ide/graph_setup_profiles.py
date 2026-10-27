"""Prepare explicit setup files. The VS Code host reviews identity and writes them."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import tomllib

from pydantic import ValidationError

from services.validation.contracts import ModelProfile, ValidationConfig
from services.validation.signatures import check_bindings
from .graph_setup import argument, discover
from .module_index import module_of_file
from .multifile import MultifileProjectionRequest

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}\Z")


def _toml(value):
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) in (int, float) and math.isfinite(value):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{_toml(key)} = {_toml(item)}" for key, item in value.items()) + " }"
    raise ValueError("This value cannot be represented in TOML. Use a Python factory for null factory arguments.")


def _markers(source, name):
    """Find generated comments only outside TOML strings."""
    opening, closing = f"# tensorViz setup: {name}", f"# end tensorViz setup: {name}"
    marks, quote, triple, offset = [], None, False, 0
    for line in source.splitlines(keepends=True):
        if quote is None and line.rstrip("\r\n") in (opening, closing):
            marks.append((line.rstrip("\r\n"), offset, offset + len(line)))
        i = 0
        while i < len(line):
            char = line[i]
            if quote is None:
                if char == "#":
                    break
                if char in "\"'":
                    quote = char
                    triple = line[i:i + 3] == char * 3
                    i += 3 if triple else 1
                    continue
            elif char == "\\" and quote == '"':
                i += 2
                continue
            elif char == quote:
                if not triple or line[i:i + 3] == quote * 3:
                    i += 3 if triple else 1
                    quote = None
                    continue
            i += 1
        offset += len(line)
    if len(marks) == 2 and marks[0][0] == opening and marks[1][0] == closing:
        return marks[0][1], marks[1][2]
    return None


def save_profile_text(source: str, name: str, profile: dict) -> tuple[str, str]:
    if not _NAME.fullmatch(name):
        raise ValueError("Choose a profile name containing letters, digits and underscores, starting with a letter.")
    if len(source.encode()) > 262144:
        raise ValueError("tensorviz.toml exceeds the supported size.")
    existing = tomllib.loads(source) if source.strip() else {"version": 1, "profiles": {}}
    if existing.get("profiles"):
        ValidationConfig.model_validate(existing)
    elif set(existing) != {"version", "profiles"} or existing["version"] != 1:
        raise ValueError("Fix the existing tensorviz.toml before saving setup.")
    profiles = existing["profiles"]
    span = _markers(source, name)
    if name in profiles:
        if span is None:
            # Hand-authored profiles are preserved. The new selection is remembered by the host.
            base = name[:56]
            for index in range(2, 65):
                candidate = f"{base}_{index}"
                if candidate not in profiles:
                    name = candidate
                    break
            else:
                raise ValueError("Remove unused profiles before creating another setup.")
        elif tomllib.loads(source[slice(*span)]).get("profiles", {}) != {name: profiles[name]}:
            raise ValueError("This profile extends outside its generated section. Edit tensorviz.toml in VS Code.")
    elif span is not None:
        raise ValueError("The generated profile markers no longer match the configuration.")
    block = f"# tensorViz setup: {name}\n[profiles.{name}]\n"
    block += "".join(f"{key} = {_toml(value)}\n" for key, value in profile.items() if value is not None)
    block += f"# end tensorViz setup: {name}\n"
    if span is not None:
        result = source[:span[0]] + block + source[span[1]:]
    else:
        result = (source if source.strip() else "version = 1\n").rstrip() + "\n\n" + block
    if len(result.encode()) > 262144:
        raise ValueError("The resulting tensorviz.toml exceeds the supported size.")
    parsed = tomllib.loads(result)
    ValidationConfig.model_validate(parsed)
    if parsed["profiles"][name] != {key: value for key, value in profile.items() if value is not None}:
        raise ValueError("The generated profile did not round-trip exactly.")
    if any(parsed["profiles"].get(key) != value for key, value in profiles.items() if key != name):
        raise ValueError("An unrelated profile changed; nothing was written.")
    return name, result


def _python_literal(value):
    if value is None or type(value) in (bool, int, float, str):
        json.dumps(value, allow_nan=False)
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_python_literal(item) for item in value) + "]"
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return "{" + ", ".join(f"{key!r}: {_python_literal(item)}" for key, item in value.items()) + "}"
    raise ValueError("Configuration fields must contain JSON values.")


def _decode_argument(value):
    if not isinstance(value, dict):
        return repr(value)
    kind = value.get("kind")
    if kind == "none":
        return "None"
    if kind == "float":
        return repr(float(value["value"]))
    if kind in ("list", "tuple"):
        text = ", ".join(_decode_argument(item) for item in value["items"])
        return "[" + text + "]" if kind == "list" else "(" + text + ",)" if text else "()"
    if kind == "dict":
        return "{" + ", ".join(f"{key!r}: {_decode_argument(item)}" for key, item in value["items"].items()) + "}"
    raise ValueError("Use a custom factory for tensor constructor arguments.")


def config_factory(discovery, profile, selection):
    if set(selection) != {"index", "parameter", "values"}:
        raise ValueError("Choose a configuration type and its fields.")
    index, parameter, values = selection["index"], selection["parameter"], selection["values"]
    if type(index) is not int or not 0 <= index < len(discovery["configs"]):
        raise ValueError("The selected configuration type is unavailable. Refresh setup.")
    option = discovery["configs"][index]
    fields = option["fields"]
    if fields is None or not isinstance(values, dict):
        raise ValueError("Use an existing factory for this configuration type.")
    if set(values) - {field["name"] for field in fields}:
        raise ValueError("A configuration field is not declared by this type.")
    missing = [field["name"] for field in fields if field["required"] and field["name"] not in values]
    if missing:
        raise ValueError("Complete the required configuration fields: " + ", ".join(missing))
    parameters = {field["name"]: field for field in discovery["init"] or []}
    if parameter not in parameters or parameters[parameter]["kind"] not in ("keyword", "keyword_only"):
        raise ValueError("Choose a named model parameter for this configuration object.")
    init = profile.get("init")
    if not isinstance(init, dict) or parameter in init.get("kwargs", {}):
        raise ValueError("Remove the literal value for the configuration-object parameter.")
    # Validate the remaining constructor values before turning them into Python literals.
    validated = ModelProfile.model_validate({**profile, "init": {**init, "kwargs": {**init.get("kwargs", {}), parameter: 0}}})
    bindings = validated.init.model_dump()
    bindings["kwargs"].pop(parameter)
    model_module, cls = discovery["model"].split(":")
    config_module = module_of_file(option["file"])
    if not all(part.isidentifier() for part in (model_module + "." + config_module + "." + cls + "." + option["cls"]).split(".")):
        raise ValueError("Model and configuration must have importable Python names.")
    lines = ["# Generated by tensorViz. This factory constructs the selected model only.",
             f"from {model_module} import {cls} as _Model",
             f"from {config_module} import {option['cls']} as _Config", "", "", "def make_model():"]
    if option["construction"] == "call":
        args = ", ".join(f"{name}={_python_literal(value)}" for name, value in values.items())
        lines.append(f"    config = _Config({args})")
    else:
        lines.append("    config = _Config()")
        lines += [f"    config.{name} = {_python_literal(value)}" for name, value in values.items()]
    args = [_decode_argument(value) for value in bindings["args"]]
    args += [f"{name}={_decode_argument(value)}" for name, value in bindings["kwargs"].items()]
    args.append(f"{parameter}=config")
    lines.append(f"    return _Model({', '.join(args)})")
    source = "\n".join(lines) + "\n"
    digest = hashlib.sha256(source.encode()).hexdigest()[:12]
    file = f".tensorviz/setup_{cls}_{digest}.py"
    return {"file": file, "source": source}, {**profile, "init": None,
        "factory": {"file": file, "callable": "make_model", "kwargs": {}}}


def declared_config_factory(discovery, profile):
    model_module, cls = discovery['model'].split(':')
    if not all(part.isidentifier() for part in (model_module + '.' + cls).split('.')):
        raise ValueError('The model must have an importable Python name.')
    source = ('# Generated by tensorViz. These are editable example defaults, not recovered settings.\n'
              f'from {model_module} import {cls} as _Model\n\n\n'
              'def make_model():\n'
              '    config = _Model.config_class()\n'
              '    # Edit config fields here to choose another model configuration.\n'
              '    return _Model(config=config)\n')
    digest = hashlib.sha256(source.encode()).hexdigest()[:12]
    file = f'.tensorviz/setup_{cls}_{digest}.py'
    return dict(file=file, source=source), {**profile, 'init': None,
        'factory': dict(file=file, callable='make_model', kwargs={})}


def prepare_profile(request: MultifileProjectionRequest, config: str, profile: dict,
                    profile_name: str | None = None, config_object: dict | None = None, *, config_defaults: bool = False):
    """Return complete replacement bytes or actionable requirements; never write files."""
    discovery = discover(request, resolve_inputs=config_defaults)
    if config_defaults and (config_object is not None or profile.get('factory')
            or profile.get('init') != dict(args=[], kwargs={}) or discovery['init'] is not None
            or not discovery.get('resolution', {}).get('runtimeInputs')
            or discovery['resolution']['draft']['profile'].get('factory')):
        raise ValueError('Declared config defaults cannot replace a selected constructor or incomplete source discovery.')
    if profile.get("model") != discovery["model"] or profile.get("cwd", ".") != ".":
        raise ValueError("The profile must name the selected model in this workspace.")
    factory = None
    binding_profile = profile
    if config_object is not None:
        parameter = config_object.get("parameter")
        init = profile.get("init", {})
        binding_profile = {**profile, "init": {**init, "kwargs": {**init.get("kwargs", {}), parameter: 0}}}
    try:
        model = ModelProfile.model_validate(binding_profile)
    except ValidationError as error:
        return dict(ok=False, requirements=[dict(field=list(item["loc"]), reason=item["msg"])
                                           for item in error.errors(include_input=False, include_context=False)][:64])
    source = next(entry.source for entry in request.sources if entry.file == request.root.file)
    from ..validation.contracts import Binding
    requirements = check_bindings(source.encode(), request.root.cls, profile_name or "model", model.init,
                                  model.inputs if isinstance(model.inputs, Binding) else None, framework=model.framework)
    if requirements:
        return dict(ok=False, requirements=[dict(field=item.field[2:], reason=item.reason) for item in requirements])
    if config_object is not None:
        factory, profile = config_factory(discovery, profile, config_object)
        model = ModelProfile.model_validate(profile)
    if config_defaults:
        factory, profile = declared_config_factory(discovery, profile)
        model = ModelProfile.model_validate(profile)
    name = profile_name or ("tensorviz_" + request.root.cls)[:64]
    name, content = save_profile_text(config, name, model.model_dump(exclude_none=True))
    return dict(ok=True, profile=name, config=content, factory=factory)


def restore_config_profile(request: MultifileProjectionRequest, profile: dict, discovery=None):
    """Recover an editable form only from an exact, unchanged generated helper.

    Parse literals, then regenerate and compare every source byte and the content
    address. Custom factories remain factories; this never imports workspace code.
    """
    discovery = discovery or discover(request, resolve_inputs=False)
    try:
        saved = ModelProfile.model_validate(profile).model_dump(exclude_none=True)
        factory = saved.get("factory")
        if (saved["model"] != discovery["model"] or saved.get("cwd", ".") != "." or not factory
                or factory["callable"] != "make_model" or factory["kwargs"]
                or not re.fullmatch(r"\.tensorviz/setup_\w+_[0-9a-f]{12}\.py", factory["file"])):
            return None
        source = next((entry.source for entry in request.sources if entry.file == factory["file"]), None)
        if source is None or len(source.encode()) > 131072:
            return None
        tree = ast.parse(source)
        if (len(tree.body) != 3 or not all(isinstance(item, ast.ImportFrom) for item in tree.body[:2])
                or not isinstance(tree.body[2], ast.FunctionDef)):
            return None
        config_import, function = tree.body[1], tree.body[2]
        if len(config_import.names) != 1 or len(function.body) < 2:
            return None
        options = [(index, option) for index, option in enumerate(discovery["configs"])
                   if module_of_file(option["file"]) == config_import.module
                   and option["cls"] == config_import.names[0].name]
        if len(options) != 1:
            return None
        index, option = options[0]
        first, last = function.body[0], function.body[-1]
        if (not isinstance(first, ast.Assign) or not isinstance(first.value, ast.Call)
                or not isinstance(last, ast.Return) or not isinstance(last.value, ast.Call)):
            return None
        values = {kw.arg: ast.literal_eval(kw.value) for kw in first.value.keywords}
        if option["construction"] == "attributes":
            for statement in function.body[1:-1]:
                if (not isinstance(statement, ast.Assign) or len(statement.targets) != 1
                        or not isinstance(statement.targets[0], ast.Attribute)):
                    return None
                values[statement.targets[0].attr] = ast.literal_eval(statement.value)
        config_args = [kw for kw in last.value.keywords if isinstance(kw.value, ast.Name) and kw.value.id == "config"]
        if len(config_args) != 1 or config_args[0].arg is None:
            return None
        selection = dict(index=index, parameter=config_args[0].arg, values=values)
        direct = {key: value for key, value in saved.items() if key != "factory"}
        direct["init"] = dict(args=[argument(ast.literal_eval(value)) for value in last.value.args],
                              kwargs={kw.arg: argument(ast.literal_eval(kw.value)) for kw in last.value.keywords
                                      if kw is not config_args[0]})
        generated, _ = config_factory(discovery, direct, selection)
        if generated != dict(file=factory["file"], source=source):
            return None
        return dict(profile=direct, configObject=selection)
    except (ValueError, TypeError, KeyError, SyntaxError, RecursionError, OverflowError):
        return None
