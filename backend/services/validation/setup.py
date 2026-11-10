"""Shared setup grammar and execution context, safe under a private worker package.

Importing this module loads neither Torch nor workspace code. Runtime functions
are called only inside authorized workers; saved profile validation uses only the
bounded callable grammar.
"""
from __future__ import annotations

import importlib
from contextlib import contextmanager
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import sys
from typing import Any

MAX_FIXTURE_BYTES = 64 * 1024 * 1024

IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}\Z")


class ExecutionRequirement(ValueError):
    """A bounded setup fact; no arbitrary exception text enters the diagnostic."""
    def __init__(self, phase: str, code: str, field: list, remedy: str, names=()):
        messages = {
            "invalid_factory": "Setup must return an instance of the exact selected model class.",
            "unavailable_device": "The selected device is unavailable in this interpreter.",
        }
        super().__init__(messages.get(code, f"{phase}: {code}"))
        self.diagnostic = dict(id="setup", phase=phase, code=code, field=field,
            names=[name for name in names if isinstance(name, str)
                   and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name)][:64],
            sourceBindingId=None, remedy=remedy)


@contextmanager
def setup_stage(phase: str, field: list, remedy: str, code="execution_failed"):
    """Add fixed structural context without guessing from user exception text."""
    try:
        yield
    except ExecutionRequirement:
        raise
    except Exception as error:
        names = ()
        missing = next((cause for cause in exception_causes(error) if isinstance(cause, ModuleNotFoundError)), None)
        if missing is not None:
            phase, code, remedy = "import", "missing_dependency", "review_dependencies"
            name = missing.name
            if isinstance(name, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,255}", name):
                names = (name.split(".")[0],)
        elif isinstance(error, FileNotFoundError):
            code = "missing_file"
        elif isinstance(error, ImportError):
            phase, code, remedy = "import", "unresolved_import", "review_dependencies"
        elif phase == "import":
            # Executing package initializers can raise ordinary code/version
            # errors. Their presence is not evidence of an absent dependency.
            code, names = "execution_failed", (type(error).__name__,)
        raise ExecutionRequirement(phase, code, field, remedy, names) from error


def exception_causes(error: BaseException) -> list[BaseException]:
    """Follow Python's visible exception chain, bounded and cycle-safe."""
    causes = [error]
    while len(causes) < 8:
        last = causes[-1]
        cause = last.__cause__ or (None if last.__suppress_context__ else last.__context__)
        if cause is None or cause in causes:
            break
        causes.append(cause)
    return causes


def bind_call(target, args, kwargs, field: str):
    """Python decides validity; inspect parameter facts only to explain a refusal."""
    remedy = "provide_inputs" if field == "inputs" else "edit_profile"
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError) as error:
        raise ExecutionRequirement("binding", "unknown_signature", [field], remedy) from error
    try:
        signature.bind(*args, **kwargs)
    except TypeError as error:
        params = list(signature.parameters.values())
        positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        var_kwargs = any(p.kind == p.VAR_KEYWORD for p in params)
        keywords = {p.name for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
        unexpected = [key for key in kwargs if key not in keywords] if not var_kwargs else []
        duplicates = [p.name for p in positional[:len(args)] if p.kind == p.POSITIONAL_OR_KEYWORD and p.name in kwargs]
        excess = len(args) > len(positional) and not any(p.kind == p.VAR_POSITIONAL for p in params)
        occupied = {p.name for p in positional[:len(args)]} | (set(kwargs) & keywords)
        missing = [p.name for p in params if p.default is p.empty and p.name not in occupied
                   and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
        code, names = ("unexpected_argument", unexpected) if unexpected else (
            ("invalid_binding", duplicates) if duplicates or excess else ("missing_argument", missing))
        raise ExecutionRequirement("binding", code, [field], remedy, names) from error


def _object(value: Any, allowed: set[str], required: set[str] = frozenset()) -> dict:
    if not isinstance(value, dict) or not required <= value.keys() or value.keys() - allowed:
        raise ValueError("Unexpected or missing execution request fields.")
    return value


def relative_path(value: Any) -> str:
    if (not isinstance(value, str) or not 1 <= len(value) <= 512 or value.startswith("/")
            or any(c in value for c in ("\\", ":", "\0"))
            or any(p in ("", ".", "..") or p.startswith(".env") for p in value.split("/"))):
        raise ValueError("Use a workspace-relative path without traversal or environment files.")
    return value


def workspace_file(workspace: str, value: str, max_bytes: int | None = None) -> Path:
    root = Path(workspace).resolve(strict=True)
    path = (root / relative_path(value)).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Execution files must resolve inside the workspace.")
    if any(part.startswith(".env") for part in path.relative_to(root).parts):
        raise ValueError("Environment files cannot be execution files.")
    if max_bytes is not None and path.stat().st_size > max_bytes:
        raise ValueError("Execution file exceeds the size limit.")
    return path


def workspace_directory(workspace: str, value: str) -> Path:
    root = Path(workspace).resolve(strict=True)
    directory = root if value == "." else (root / relative_path(value)).resolve(strict=True)
    if not directory.is_relative_to(root) or not directory.is_dir():
        raise ValueError("The execution working directory must be inside the workspace.")
    if any(part.startswith(".env") for part in directory.relative_to(root).parts):
        raise ValueError("Environment directories cannot be execution working directories.")
    return directory


def _literals(value: Any) -> dict:
    if not isinstance(value, dict) or any(not isinstance(k, str) or not 1 <= len(k) <= 128 for k in value):
        raise ValueError("Configuration must be an object of JSON literals.")
    def check(item: Any, depth: int = 0) -> None:
        if depth > 8:
            raise ValueError("Configuration nesting exceeds eight levels.")
        if item is None or type(item) in (bool, str, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if isinstance(item, (list, dict)):
            if isinstance(item, dict) and any(not isinstance(k, str) for k in item):
                raise ValueError("JSON object keys must be strings.")
            for child in (item.values() if isinstance(item, dict) else item):
                check(child, depth + 1)
            return
        raise ValueError("Configuration must contain finite JSON literals.")
    check(value)
    if len(json.dumps(value, ensure_ascii=False, separators=(",", ":"))) > 8192:
        raise ValueError("Configuration exceeds 8 KiB.")
    return value


def callable_spec(value: Any) -> dict:
    value = _object(value, {"file", "callable", "kwargs"}, {"file", "callable"})
    file = relative_path(value["file"])
    if not file.endswith(".py") or not isinstance(value["callable"], str) or not IDENTIFIER.fullmatch(value["callable"]):
        raise ValueError("Name a Python file and a top-level callable.")
    return {**value, "kwargs": _literals(value.get("kwargs", {}))}


def package_root(file: str) -> str:
    """The directory ``file`` should be imported from.

    Walks out of the package: ``<ws>/models/resid.py`` next to a ``models/__init__.py``
    imports as ``models.resid`` from ``<ws>``, not as ``resid`` from ``<ws>/models``.
    Getting this wrong is not a detail — the fixture pins ``_import_path`` for
    ``MyBlock`` as ``models.blocks`` (task.md §7), and a sibling import inside the
    package fails outright from the wrong root.
    """
    root = os.path.dirname(os.path.abspath(file))
    for _ in range(32):
        if not os.path.isfile(os.path.join(root, "__init__.py")):
            return root
        parent = os.path.dirname(root)
        if parent == root:
            return root
        root = parent
    return root


def module_path_for(root: str, file: str) -> str:
    """The dotted module path of ``file`` inside ``root``.

    The IDE knows the file the user clicked in, so it does not go through
    ``scan.classes``, whose index is keyed by class NAME and collapses two classes
    called ``Net`` in different files onto the last one scanned. Asking by path is
    the only way a workspace with a repeated name opens the class the user meant.
    """
    rel = os.path.relpath(os.path.abspath(file), os.path.abspath(root))
    if rel.startswith(".."):
        raise ValueError(f"{file} is not inside {root}")
    if rel.endswith(".py"):
        rel = rel[:-3]
    parts = [p for p in rel.split(os.sep) if p]
    if parts and parts[-1] == "__init__":
        parts.pop()
    if not parts:
        raise ValueError(f"{file} does not name an importable module under {root}")
    return ".".join(parts)


def load_callable(workspace: str, spec: dict) -> Any:
    module = load_module(workspace_file(workspace, spec['file'], MAX_FIXTURE_BYTES))
    function = getattr(module, spec['callable'], None)
    if not callable(function):
        raise ExecutionRequirement("construction", "invalid_factory", ["factory", "callable"], "provide_factory")
    return function


def load_module(file: Path, *, root: Path | None = None) -> Any:
    root = str(root) if root is not None else package_root(str(file))
    if root not in sys.path:
        sys.path.insert(0, root)
    name = module_path_for(root, str(file))
    with setup_stage("import", ["model"], "review_dependencies", "unresolved_import"):
        module = importlib.import_module(name)
    if Path(getattr(module, '__file__', '')).resolve() != file.resolve():
        raise ValueError('The execution module name conflicts with an already imported module.')
    return module


def factory_model(workspace: str, setup: dict, target: Any, source: Path) -> torch.nn.Module:
    """Shared saved-profile/MCP factory path; class identity is checked, not its name."""
    import torch
    if not isinstance(target, type) or not issubclass(target, torch.nn.Module):
        raise ValueError('A model factory requires a selected PyTorch module class.')
    if Path(inspect.getfile(target)).resolve() != source.resolve():
        raise ValueError('The selected model class must be defined in the selected source file.')
    with setup_stage("construction", ["factory"], "provide_factory"):
        function = load_callable(workspace, setup)
        bind_call(function, (), setup['kwargs'], "factory")
        model = function(**setup['kwargs'])
    if type(model) is not target:
        raise ExecutionRequirement("construction", "invalid_factory", ["factory"], "provide_factory")
    return model


def configure_runtime(torch, spec: dict):
    """Apply the same device, RNG and deterministic policy before user imports."""
    device = torch.device(spec['device'])
    if device.type == 'cuda':
        if not torch.cuda.is_available() or (device.index is not None and device.index >= torch.cuda.device_count()):
            raise ExecutionRequirement("environment", "unavailable_device", ["device"], "select_environment")
        device = torch.device('cuda', torch.cuda.current_device() if device.index is None else device.index)
    if device.type == 'mps':
        if not torch.backends.mps.is_available():
            raise ExecutionRequirement("environment", "unavailable_device", ["device"], "select_environment")
        device = torch.device('mps', 0)
    if spec['seed'] is not None:
        random.seed(spec['seed'])
        torch.manual_seed(spec['seed'])
        try:
            import numpy as np
            np.random.seed(spec['seed'] % 2**32)
        except ImportError:
            pass
    torch.use_deterministic_algorithms(spec['deterministic'])
    torch.backends.cudnn.benchmark = False
    return device
