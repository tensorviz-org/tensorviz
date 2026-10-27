"""The standalone Python profile must remain closed over local services.

Imports inside functions count too: a capture that works at startup but fails
when the worker reaches a lazy import is still a broken installed plugin.
"""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess

BACKEND = Path(__file__).parent.resolve().parents[1]
REPO = BACKEND.parent
CONFIG = json.loads((REPO / "runtime/backend-profiles.json").read_text())
PROFILE = CONFIG["profiles"]["standalone"]


def _selected(name: str) -> bool:
    if name in PROFILE.get("excludedFiles", []):
        return False
    return name in PROFILE["files"] or (
        name.endswith(".py") and any(name.startswith(prefix) for prefix in PROFILE["pythonPrefixes"])
    )


def test_standalone_services_have_no_hidden_hosted_or_unshipped_imports():
    paths = subprocess.check_output(
        ["git", "ls-files", "-z", "backend"], cwd=REPO, text=True
    ).split("\0")
    modules = {}
    for name in paths:
        if not name.endswith(".py"):
            continue
        relative = name.removeprefix("backend/")
        module = relative.removesuffix(".py").replace("/", ".")
        modules[module.removesuffix(".__init__")] = relative
    errors = []
    for module, relative in modules.items():
        if not _selected(relative):
            continue
        package = module if relative.endswith("/__init__.py") else module.rpartition(".")[0]
        for node in ast.walk(ast.parse((BACKEND / relative).read_text())):
            imported = []
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    base = importlib.util.resolve_name("." * node.level + base, package)
                imported = [base, *(f"{base}.{alias.name}" for alias in node.names)]
            for name in imported:
                # Include package initializers, which Python executes along the way.
                for end in range(1, len(name.split(".")) + 1):
                    dependency = modules.get(".".join(name.split(".")[:end]))
                    if dependency and not _selected(dependency):
                        errors.append(f"{relative}:{node.lineno} imports unshipped {dependency}")
    assert not errors, "\n".join(sorted(set(errors)))


def test_standalone_does_not_package_hosted_authorities():
    for name in ("server_v2.py", "extension_environment.py", "auth.py", "services/project_store.py", "services/store_factory.py",
                 "services/user_history_store.py", "api/routers/projects.py"):
        assert not _selected(name), f"Hosted authority entered the standalone profile: {name}"
    for name in ("standalone_server.py", "services/semantic/contracts.py",
                 "services/validation/worker.py", "services/ide/local_browser_broker.py"):
        assert _selected(name), f"Shared local service missing from the standalone profile: {name}"
