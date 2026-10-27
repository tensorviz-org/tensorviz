"""`backend/research/` must never touch disk, a database, or the user's code.

This is enforced rather than documented because it is the kind of rule that rots: a
single `open()` added during a debugging session would quietly make the backend a
second reader of the workspace, and the extension would stop being the only component
that knows about trust and unsaved buffers.

The check is an AST scan of the package's own source, so it fails at review time
rather than at runtime on someone's machine.
"""

import ast
import os

import pytest


PACKAGE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "research")

#: Calls that would make this package read or write the world.
FORBIDDEN_CALLS = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "importlib.import_module",
    "os.walk",
    "os.listdir",
    "os.remove",
    "os.makedirs",
    "os.mkdir",
    "shutil.copy",
    "shutil.rmtree",
    "subprocess.run",
    "subprocess.Popen",
    "sqlite3.connect",
    "Path.read_text",
    "Path.write_text",
    "Path.open",
}

FORBIDDEN_IMPORTS = {"sqlite3", "subprocess", "importlib", "requests", "httpx", "torch"}


def research_modules():
    for name in sorted(os.listdir(PACKAGE)):
        if name.endswith(".py"):
            yield name, os.path.join(PACKAGE, name)


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


@pytest.mark.parametrize("name,path", list(research_modules()))
def test_module_performs_no_io(name, path):
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=name)

    offences = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            called = dotted(node.func)
            if called in FORBIDDEN_CALLS or (
                called and called.rsplit(".", 1)[-1] in {"read_text", "write_text"}
            ):
                offences.append(f"{name}:{node.lineno} calls {called}()")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_IMPORTS:
                    offences.append(f"{name}:{node.lineno} imports {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in FORBIDDEN_IMPORTS:
                offences.append(f"{name}:{node.lineno} imports from {node.module}")

    assert not offences, "research/ must stay pure:\n  " + "\n  ".join(offences)


def test_the_check_itself_can_fail(tmp_path):
    """A guard that cannot fail is not a guard. Prove this one detects an offence."""
    offender = tmp_path / "leaky.py"
    offender.write_text("def go():\n    return open('x').read()\n", encoding="utf-8")

    tree = ast.parse(offender.read_text(encoding="utf-8"))
    found = [
        n for n in ast.walk(tree) if isinstance(n, ast.Call) and dotted(n.func) in FORBIDDEN_CALLS
    ]
    assert found, "the purity scan would not have caught a bare open()"
