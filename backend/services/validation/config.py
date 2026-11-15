"""Canonical TOML preflight. Host facts in, RunSpec or requirements out; no I/O.

See docs/pytorch-validation-resolver.md for the host snapshot and worker boundary.
This module never imports workspace code, inspects the filesystem or authorizes runs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath, PureWindowsPath
import re
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 model environments.
    import tomli as tomllib
from typing import Annotated, Literal

from pydantic import Field, StrictBool, StrictBytes, StrictStr, ValidationError

from . import contracts as c
from .signatures import check_bindings, check_factory


class PathFact(c.Contract):
    """Host stat/realpath result for one workspace-root-relative spelling.

    canonical_path is relative to the resolved root, None for an escape/unknown.
    file_id is the host's native file identity, shared by symlinks AND hard links.
    source/native roles mark dependencies; unmarked entries are only path inventory.
    """
    kind: Literal["file", "directory", "missing"]
    canonical_path: c.Path | Literal["."] | None
    file_id: c.Id | None = None
    digest: c.Digest | None = None
    document_version: c.Nat | None = None
    saved: StrictBool = True
    role: Literal["source", "native"] | None = None
    content: StrictBytes | None = None


class EnvironmentFact(c.Contract):
    """Already-probed machine-local environment, without credentials or env vars."""
    executable: Annotated[StrictStr, Field(min_length=1, max_length=4096)]
    resolved_executable: Annotated[StrictStr, Field(min_length=1, max_length=4096)]
    python_version: c.Id
    torch_version: c.Id | None
    selection_revision: c.Nat
    # Digest of the host's sanitized environment probe (e.g. installed package versions).
    probe_digest: c.Digest


class HostFacts(c.Contract):
    workspace_session_id: c.Id
    workspace_id: c.Id
    run_id: c.Id
    generation: c.Nat
    paths: dict[c.Path | Literal["."], PathFact]
    dependency_scope: Literal["complete", "partial"]
    environment: EnvironmentFact | None


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_digest(value) -> str:
    return _digest(json.dumps(value, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False, allow_nan=False).encode("utf-8"))


def _requirement(profile, field, code, reason, remedy, phase="static"):
    # Keep user keys only as bounded structural locations, never in prose.
    bounded = []
    for segment in field:
        if isinstance(segment, str) and len(segment) > 128:
            break
        bounded.append(segment)
    return c.Requirement(profile=profile, field=bounded[:32] or ["tensorviz.toml"],
                         phase=phase, code=code, reason=reason, remedy=remedy)


def _needs(requirements):
    if len(requirements) > 256:
        requirements = requirements[:255] + [_requirement(
            None, ["tensorviz.toml"], "invalid", "Further setup diagnostics exceed the display limit.",
            "Fix the reported fields and resolve again to see the remaining requirements.")]
    return c.NeedsSetup(status="needs_setup", requirements=requirements)


_REMEDIES = {
    "tensorviz.toml": "Use the documented version = 1 TOML structure with explicitly named script or model profiles.",
    "version": "Set version = 1 in the project-root tensorviz.toml.",
    "profiles": "Define 1–64 named profiles, each with an explicit script or model kind.",
    "kind": "Choose one supported kind and supply only that variant's fields.",
    "model": "Set an exact dotted.module:Qualified.Class or callable symbol; use a script for complex setup.",
    "script": "Set a saved workspace-relative .py regular file, using / separators and no traversal.",
    "capture": "Set capture.model to an exact class symbol and capture.call to a one-based invocation.",
    "call": "Set a positive one-based invocation number across all instances of the exact selected class.",
    "mode": "Set mode explicitly to train or eval.",
    "device": "Set device explicitly to cpu, mps, or cuda:N; availability is checked by the approved worker.",
    "grad_enabled": "Set grad_enabled explicitly to true or false.",
    "deterministic": "Set deterministic to true or false; the default is true.",
    "init": "Provide init.args and init.kwargs; omit only parameters with actual Python defaults.",
    "factory": "Provide factory.file, factory.callable and JSON-literal factory.kwargs instead of init; the factory must return the exact selected class.",
    "inputs": "Provide inputs.args and inputs.kwargs for the exact call; use a script for complex setup.",
    "args": "Provide an explicit args array, possibly empty, preserving positional order (64 model arguments or 256 script strings maximum).",
    "kwargs": "Provide an explicit kwargs table, possibly empty, with at most 64 ASCII Python identifier keys.",
    "shape": "Set shape to 0–32 nonnegative integer dimensions; [] is scalar and zero dimensions are valid.",
    "dtype": "Set dtype explicitly to bool, uint8, int8/16/32/64, float16/32/64, or bfloat16.",
    "requires_grad": "Set requires_grad explicitly; true requires a floating dtype.",
    "generator": "Choose normal, uniform, randint, full or values and specify all of its parameters.",
    "mean": "Supply an explicit finite mean for the normal generator.",
    "std": "Supply an explicit finite standard deviation greater than zero.",
    "low": "Supply an explicit lower bound compatible with the dtype and below the exclusive high bound.",
    "high": "Supply an explicit exclusive upper bound above low and within the dtype's range.",
    "seed": "Set an explicit nonnegative safe-integer seed for each random generator; no seed is inferred.",
    "values": "Supply at most 4096 compatible literals in flat row-major order, exactly matching the shape product.",
    "cwd": "Use . or an existing workspace-relative directory with / separators and no traversal.",
    "tracked_files": "List distinct existing workspace-relative regular files; do not use globs or aliases.",
    "limits": "Use timeout_ms, max_events, max_tensors, max_record_bytes and max_capture_bytes within the documented bounds.",
    "timeout_ms": "Set limits.timeout_ms to an integer between 100 and 86400000 (milliseconds).",
    "max_events": "Set an integer max_events between 1 and 100000.",
    "max_tensors": "Set an integer max_tensors between 1 and 100000.",
    "max_record_bytes": "Set an integer max_record_bytes between 1024 and 4194304 for validation records. For graph recording, use limits.max_capture_bytes (1024..16777216; default 16777216).",
    "max_capture_bytes": "Set an integer max_capture_bytes between 1024 and 16777216 for graph recording; the default is 16777216. This is independent of max_record_bytes for validation records.",
}
_VALUE_REMEDY = ("Use a safe integer, boolean, string or a tagged float/none/list/tuple/dict/tensor value. "
                 "Use a script profile for object construction, expressions or factories.")
_REMOVED = {"input_shape", "inputShape", "input_spec", "init_args", "declaredInit", "declared_input_shape",
            "declared_out_shape", "input", "output", "experiment", "experiments", "nodes", "edges", "sdk"}
# Match only our contract's fixed validator messages; never expose arbitrary error text.
_RELATIONAL = {
    "provide exactly one of init or factory": ("factory", "factory"),
    "requires_grad needs a floating dtype": ("requires_grad", "requires_grad"),
    "normal/uniform require a floating dtype": ("dtype", "dtype"),
    "randint requires integer dtype and low < high": ("generator", "Use randint only with an integer dtype and explicit integer bounds satisfying low < high."),
    "randint bounds exceed dtype range": ("generator", "Set both integer bounds within the dtype range: low is included, high is excluded."),
    "low must be less than high (exclusive)": ("high", "high"),
    "flat values count must equal the product of shape (scalar needs one)": (
        "generator", "Set values to exactly the shape product in flat row-major order; scalar needs one, empty needs zero."),
    "bool tensors require boolean literal values": ("generator", "Use only boolean literals for a bool tensor."),
    "integer literal outside dtype range": ("generator", "Use integer literals within the selected dtype's range."),
    "floating tensors require numeric literal values": ("generator", "Use finite numeric literals, not booleans, for floating tensors."),
    "tracked_files contains duplicates": ("tracked_files", "tracked_files"),
    "argument tree exceeds depth/node limit": (None, "Use at most eight container levels and 4096 argument nodes per binding; otherwise use a script."),
}


def _location(raw, location):
    """Strip Pydantic union branches without stripping real user keys of those names."""
    path, cursor, expect_tag = [], raw, False
    for segment in location:
        if segment == "[key]":
            continue
        if expect_tag and isinstance(cursor, dict) and cursor.get("kind") == segment:
            expect_tag = False
            continue
        if isinstance(cursor, dict) and segment in cursor:
            path.append(segment)
            expect_tag = ((len(path) == 2 and path[0] == "profiles")
                          or (cursor.get("kind") == "tensor" and segment == "generator"))
            cursor = cursor[segment]
        elif isinstance(cursor, list) and isinstance(segment, int) and segment < len(cursor):
            path.append(segment)
            cursor, expect_tag = cursor[segment], False
        elif isinstance(segment, str) and segment.startswith("tagged-union["):
            expect_tag = True
        elif isinstance(segment, str) and (segment.startswith(("literal[", "constrained-"))
                                          or segment in {"bool", "int", "str", "float"}):
            continue
        else:
            path.append(segment)
            cursor, expect_tag = None, False
    return path or ["tensorviz.toml"]


def _field_remedy(raw, field):
    cursor, parent = raw, None
    for segment in field[:-1]:
        parent = cursor
        if isinstance(cursor, dict):
            cursor = cursor.get(segment)
        elif isinstance(cursor, list) and isinstance(segment, int) and segment < len(cursor):
            cursor = cursor[segment]
        else:
            cursor = None
    key = field[-1]
    if key == "kind":
        if len(field) == 3:
            return "Set profile kind to script or model and supply only that variant's fields."
        if field[-2:-1] == ["generator"]:
            return "Set generator.kind to normal, uniform, randint, full or values and supply its parameters."
        return _VALUE_REMEDY
    if len(field) >= 5 and field[2] in {"init", "inputs"}:
        if isinstance(cursor, list) and isinstance(parent, dict) and parent.get("kind") in ("tensor", "values"):
            return _REMEDIES.get(field[-2], _VALUE_REMEDY)
        if not isinstance(cursor, dict) or cursor.get("kind") not in (
            "tensor", "normal", "uniform", "randint", "full", "values", "float", "none", "list", "tuple", "dict"
        ):
            return _VALUE_REMEDY
    if len(field) == 3 and key == "seed":
        return "Omit the model seed for no global RNG reset, or set an explicit nonnegative safe integer."
    return _REMEDIES.get(key, _VALUE_REMEDY)


def _config_errors(raw, error):
    requirements = []
    for item in error.errors(include_url=False, include_input=False):
        field = _location(raw, item["loc"])
        profile = field[1] if len(field) > 1 and field[0] == "profiles" else None
        if not isinstance(profile, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", profile) is None:
            profile = None
        code = "missing" if item["type"] in {"missing", "union_tag_not_found"} else "invalid"
        if item["type"] in {"union_tag_not_found", "union_tag_invalid"}:
            field.append("kind")
        key = field[-1]
        remedy = _field_remedy(raw, field)
        reason = "A required setting is missing." if code == "missing" else "The setting is malformed or incompatible."
        if item["type"] == "extra_forbidden":
            code, reason, remedy = "unknown_key", "This setting is not supported.", "Remove this key; use only the documented v1 profile fields."
            if key in _REMOVED:
                code, remedy = "removed_format", "Replace obsolete declarations with explicit v1 tensorviz.toml profiles; use a script for existing setup."
            if key == "timeout_ms" and len(field) == 3:
                remedy = "Move timeout_ms under limits; for example, limits = { timeout_ms = 30000 }."
            raw_profile = raw.get("profiles", {}).get(profile, {}) if profile else {}
            if len(field) == 3 and isinstance(raw_profile, dict) and key in (
                {"model", "mode", "device", "grad_enabled", "deterministic", "seed", "init", "factory", "inputs"}
                if raw_profile.get("kind") == "script" else {"script", "args", "capture"}
            ):
                code, reason, remedy = "ambiguous", "Script and direct-model settings cannot be mixed.", "Choose one profile kind and remove the fields belonging to the other kind."
        context_error = item.get("ctx", {}).get("error")
        args = getattr(context_error, "args", ())
        if len(args) == 1 and isinstance(args[0], str) and args[0] in _RELATIONAL:
            suffix, remedy_key = _RELATIONAL[args[0]]
            if suffix:
                field.append(suffix)
            remedy = _REMEDIES.get(remedy_key, remedy_key)
        if item["type"] == "string_pattern_mismatch" and (len(field) == 2 and field[0] == "profiles" or item["loc"][-1:] == ("[key]",)):
            remedy = "Use an ASCII Python identifier of 1–64 characters for this profile or keyword name."
        requirements.append(_requirement(profile, field, code, reason, remedy))
    # Scalar alternatives of the argument union are noise when a tagged error is more specific.
    ancestors = {tuple(r.field[:end]) for r in requirements for end in range(1, len(r.field))}
    specific = [r for r in requirements if tuple(r.field) not in ancestors]
    unique = {}
    for requirement in specific:
        unique.setdefault(tuple(requirement.field), requirement)
    return list(unique.values())


def _path_check(facts, path, profile, field, kind):
    fact = facts.paths.get(path)
    if fact is None or fact.kind == "missing":
        return _requirement(profile, field, "missing_file", "The referenced path is missing from the host snapshot.",
                            "Create or select the referenced workspace path, save it and refresh the snapshot.")
    if (fact.kind != kind or fact.canonical_path is None
            or any(ord(char) == 127 for char in path + (fact.canonical_path or ""))
            or (kind == "file" and (fact.canonical_path == "." or path == "."))):
        return _requirement(profile, field, "invalid", "The path is not the required kind inside the resolved workspace.",
                            "Select an existing contained regular file or directory; resolve symlinks in the host first.")
    if kind == "file":
        if not fact.saved:
            return _requirement(profile, field, "unsaved", "This run depends on an unsaved file.",
                                "Save the file in VS Code and refresh the saved snapshot before Run.")
        if fact.digest is None or fact.file_id is None:
            return _requirement(profile, field, "invalid", "The saved file identity is incomplete.",
                                "Refresh the host's content digest and native file identity.")
        if fact.content is not None and _digest(fact.content) != fact.digest:
            return _requirement(profile, field, "invalid", "Supplied content does not match its saved file digest.",
                                "Refresh the saved host snapshot; do not resolve mixed file revisions.")
    return None


def _absolute_executable(value):
    return (not any(ord(char) < 32 or ord(char) == 127 for char in value)
            and (PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()))


def resolve(config: bytes | None, selected_profile: str | None, facts: HostFacts,
            *, config_path: str = "tensorviz.toml") -> c.Resolution:
    """Resolve saved bytes with validated host facts. `resolved` means static preflight only.

    Always select a profile explicitly, even if only one exists. No source comment,
    sidecar, inferred shape, process environment or active editor is consulted.
    """
    if config_path != "tensorviz.toml":
        return _needs([_requirement(None, ["tensorviz.toml"], "removed_format", "Only project-root tensorviz.toml is supported.",
                                   "Move execution settings to version = 1 profiles in tensorviz.toml.")])
    if config is None:
        return _needs([_requirement(None, ["tensorviz.toml"], "missing", "The project has no saved validation configuration.",
                                   "Create and save project-root tensorviz.toml with version = 1 and an explicit profile.")])
    if len(config) > c.CONFIG_MAX_BYTES:
        return _needs([_requirement(None, ["tensorviz.toml"], "invalid", "The configuration exceeds 256 KiB.",
                                   "Reduce tensorviz.toml to at most 256 KiB; use a script for large literal data.")])
    try:
        raw = tomllib.loads(config.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, RecursionError):
        return _needs([_requirement(None, ["tensorviz.toml"], "invalid", "The configuration is not valid UTF-8 TOML.",
                                   "Fix TOML syntax and duplicate keys; replace JSON/YAML/SDK formats with v1 profiles.")])
    try:
        parsed = c.ValidationConfig.model_validate(raw)
    except ValidationError as error:
        return _needs(_config_errors(raw, error))
    config_error = _path_check(facts, config_path, None, ["tensorviz.toml"], "file")
    if config_error:
        return _needs([config_error])
    if selected_profile is None or selected_profile not in parsed.profiles:
        return _needs([_requirement(None, ["profiles"], "ambiguous" if selected_profile is None else "missing",
                                   "An existing profile must be selected explicitly.",
                                   "Select a named tensorviz.toml profile in the host; no profile is inferred from canvas focus.")])
    profile = parsed.profiles[selected_profile]
    prefix = ["profiles", selected_profile]
    requirements = []
    path_roles = {path: fact.role for path, fact in facts.paths.items() if fact.role is not None}
    path_roles["tensorviz.toml"] = "config"
    fields = {"tensorviz.toml": (None, ["tensorviz.toml"])}
    for index, path in enumerate(profile.tracked_files):
        path_roles.setdefault(path, "native")
        fields[path] = (selected_profile, prefix + ["tracked_files", index])
    if isinstance(profile, c.ScriptProfile):
        path_roles[profile.script] = "source"
        fields[profile.script] = (selected_profile, prefix + ["script"])
        if not profile.script.endswith(".py"):
            requirements.append(_requirement(selected_profile, prefix + ["script"], "invalid",
                                             "The script must be a saved .py file.", _REMEDIES["script"]))
    elif profile.factory is not None:
        path_roles[profile.factory.file] = "source"
        fields[profile.factory.file] = (selected_profile, prefix + ["factory", "file"])
    if isinstance(profile, c.ModelProfile) and isinstance(profile.inputs, c.InputFactory):
        path_roles[profile.inputs.factory.file] = 'source'
        fields[profile.inputs.factory.file] = (selected_profile, prefix + ['inputs', 'factory', 'file'])
    cwd_error = _path_check(facts, profile.cwd, selected_profile, prefix + ["cwd"], "directory")
    if cwd_error:
        requirements.append(cwd_error)
    # Refuse rather than truncate identities and claim completeness.
    if len(path_roles) > 100000:
        return _needs([_requirement(None, ["files"], "unsupported", "The dependency set exceeds 100000 files.",
                                   "Narrow the supplied dependency scope and disclose it as partial before resolving.")])
    files, physical, canonical = [], {}, {}
    for path, role in sorted(path_roles.items()):
        owner, field = fields.get(path, (selected_profile, ["files", path]))
        issue = _path_check(facts, path, owner, field, "file")
        if issue:
            requirements.append(issue)
            continue
        fact = facts.paths[path]
        if fact.file_id in physical or fact.canonical_path in canonical:
            requirements.append(_requirement(owner, field, "ambiguous", "Different paths identify the same file.",
                                             "Use one workspace-relative spelling per physical file; remove duplicate aliases."))
        physical[fact.file_id] = path
        canonical[fact.canonical_path] = path
        files.append(c.FileIdentity(path=path, digest=fact.digest, documentVersion=fact.document_version, role=role))
    config_fact = facts.paths.get("tensorviz.toml")
    if config_fact and config_fact.digest != _digest(config):
        requirements.append(_requirement(None, ["tensorviz.toml"], "invalid", "Configuration bytes differ from their saved identity.",
                                         "Save tensorviz.toml and refresh the host snapshot before resolving."))
    environment = facts.environment
    if environment is None:
        requirements.append(_requirement(None, ["interpreter"], "environment", "No project interpreter is selected.",
                                         "Select and probe the project Python environment in VS Code."))
    else:
        version = re.match(r"^(\d+)\.(\d+)(?:\.|$)", environment.python_version)
        if (not version or tuple(map(int, version.groups())) < (3, 10) or not environment.torch_version
                or not _absolute_executable(environment.executable)
                or not _absolute_executable(environment.resolved_executable)):
            requirements.append(_requirement(None, ["interpreter"], "environment", "The host environment facts are incomplete or unsupported.",
                                             "Select an absolute resolved Python 3.10+ executable with PyTorch and refresh its probe."))
    scope = facts.dependency_scope
    if isinstance(profile, c.ModelProfile):
        if isinstance(profile.inputs, c.InputFactory):
            input_factory = profile.inputs.factory
            fact = facts.paths.get(input_factory.file)
            if fact is not None and fact.content is not None:
                for requirement in check_factory(fact.content, input_factory.callable, selected_profile, input_factory.kwargs):
                    requirement.field = [*requirement.field[:2], 'inputs', *requirement.field[2:]]
                    requirements.append(requirement)
        if profile.factory is not None:
            factory_fact = facts.paths.get(profile.factory.file)
            if factory_fact is not None and factory_fact.content is not None:
                requirements.extend(check_factory(factory_fact.content, profile.factory.callable,
                                                  selected_profile, profile.factory.kwargs))
        module, qualname = profile.model.split(":", 1)
        base = ("" if profile.cwd == "." else profile.cwd + "/") + module.replace(".", "/")
        candidates = [path for path in (base + ".py", base + "/__init__.py")
                      if path in path_roles and path_roles[path] == "source"]
        if module == "__main__":
            requirements.append(_requirement(selected_profile, prefix + ["model"], "unsupported",
                                             "A direct profile needs an importable model symbol.",
                                             "Use a script profile with a __main__ capture selector or choose an importable module."))
        elif len(candidates) > 1:
            requirements.append(_requirement(selected_profile, prefix + ["model"], "ambiguous", "Multiple supplied sources map to this module.",
                                             "Remove the ambiguous module/package mapping; do not choose a source by declaration order."))
        elif candidates and facts.paths[candidates[0]].content is not None:
            requirements.extend(check_bindings(facts.paths[candidates[0]].content, qualname, selected_profile,
                                               profile.init, profile.inputs if isinstance(profile.inputs, c.Binding) else None,
                                               framework=profile.framework))
        else:
            scope = "partial"
    if requirements:
        return _needs(requirements)
    environment_digest = _json_digest(environment.model_dump(mode="json"))
    identity = c.RunIdentity(
        workspaceSessionId=facts.workspace_session_id, workspaceId=facts.workspace_id,
        runId=facts.run_id, generation=facts.generation, profile=selected_profile,
        configDigest=_digest(config), profileDigest=_json_digest(profile.model_dump(mode="json")),
        sourceDigest=_json_digest(sorted([file.path, file.role, file.digest] for file in files if file.role != "config")),
        environmentDigest=environment_digest)
    return c.Resolved(status="resolved", spec=c.RunSpec(
        version=1, identity=identity, profile=profile, files=files, dependencyScope=scope,
        # Launch the selected venv path: dereferencing its Python symlink loses
        # pyvenv.cfg and silently switches to the base interpreter/environment.
        interpreter=c.InterpreterIdentity(executable=environment.executable,
                                          version=environment.python_version, environmentDigest=environment_digest)))


def runtime_checks(spec: c.RunSpec) -> list[c.Requirement]:
    """Mandatory deferred checks, NOT failures or execution authorization.

    U04/U05 workers must perform these against the one approved execution and report
    unmet requirements using the same field/phase contract. Static resolution cannot
    certify live Python objects, allocation, dependencies or an observed capture.
    """
    name = spec.identity.profile
    checks = [(None, ["interpreter"], "environment", "Imports and the selected environment require runtime verification.",
               "In the approved worker, import real dependencies once; report failures without mocks or retries.")]
    prefix = ["profiles", name]
    if isinstance(spec.profile, c.ModelProfile):
        checks.extend([
            (name, prefix + ["model"], "unsupported", "Callable validity and its returned model are runtime-dependent.",
             "Resolve the exact callable once in the approved worker and require a real supported PyTorch model."),
            (name, prefix + ["factory" if spec.profile.factory is not None else "init"], "signature",
             "The live construction signature may differ from the declaration.",
             "Bind the configured construction arguments once; a factory must return the exact selected class."),
            (name, prefix + ["inputs"], "signature", "The live forward signature and tensor allocation require runtime checks.",
             "Bind the exact configured inputs and allocate without dtype, shape or generator substitutions or retries."),
            (name, prefix + ["device"], "environment", "Device and dtype availability depend on the selected environment.",
             "Use the explicit device and dtype; report unsupported allocation without falling back to another device."),
        ])
    else:
        checks.append((name, prefix + ["capture"], "unsupported", "Target reachability and capture support can only be observed at runtime.",
                       "Run the selected script once with its exact argv/context; report an unreached or unsupported capture as incomplete."))
    return [_requirement(*check, phase="runtime") for check in checks]
