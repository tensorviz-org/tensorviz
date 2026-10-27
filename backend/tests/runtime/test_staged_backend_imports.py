"""The backend that ships inside the .vsix must be able to import itself.

``extension/scripts/stage-backend.mjs`` copies this package into the extension and drops a
few directories that exist only for development — ``tests/``, ``evals/`` and the caches. That
staging is what makes task.md P0 true ("open any folder, no canvas checkout"), and it has
one failure mode nobody sees until a user opens their first model: a module that is reachable
from ``server_v2`` importing one of the dropped trees at import time. The checkout imports
fine, every backend test passes, the .vsix builds, and the shipped backend dies with
``ModuleNotFoundError`` the moment the extension starts it.

So this reads the exclusion list out of the shared runtime profile (rather than repeating it,
which would let the two drift) and asserts statically that nothing staged imports an excluded
tree at module level. Static, because importing every module here would need the optional
dependencies and would be slower than the thing it guards.

Lazy imports inside functions are fine and are used deliberately elsewhere: the rule is only
that importing ``server_v2`` must not need a directory the user's copy will not have.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterator, List, Set, Tuple

BACKEND = Path(__file__).parent.resolve().parents[1]
REPO = BACKEND.parent
STAGE_SCRIPT = REPO / "extension" / "scripts" / "stage-backend.mjs"
PROFILE = REPO / "runtime" / "backend-profiles.json"


def _excluded_dirs() -> Set[str]:
    """Read the same exclusion data both package builders consume."""
    return set(json.loads(PROFILE.read_text())["excludedDirectories"])


def _staged_python_files(excluded: Set[str]) -> Iterator[Path]:
    """Every .py file that stage-backend.mjs would copy into the extension."""
    for path in BACKEND.rglob("*.py"):
        rel = path.relative_to(BACKEND)
        if any(part in excluded for part in rel.parts[:-1]):
            continue
        yield path


def _top_level_imports(tree: ast.Module) -> Iterator[Tuple[str, int]]:
    """Module-level imports only: the ones that run when the module is imported.

    An import nested in a function, a method or an ``if TYPE_CHECKING:`` block never executes
    on a plain import, which is exactly the escape hatch a staged module should use.
    """
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], node.lineno
        elif isinstance(node, ast.ImportFrom):
            # A relative import cannot reach a sibling tree that was dropped.
            if node.level == 0 and node.module:
                yield node.module.split(".")[0], node.lineno


def test_no_staged_module_imports_an_excluded_tree_at_import_time() -> None:
    excluded = _excluded_dirs()
    assert "evals" in excluded and "tests" in excluded, (
        f"Unexpected exclusion list {sorted(excluded)}; this test assumes the staging script "
        "still drops the development-only trees."
    )

    offenders: List[str] = []
    for path in _staged_python_files(excluded):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:  # a file the interpreter could not parse is its own bug
            offenders.append(f"{path.relative_to(REPO)}: could not parse ({exc})")
            continue
        for module, lineno in _top_level_imports(tree):
            if module in excluded:
                offenders.append(
                    f"{path.relative_to(REPO)}:{lineno} imports '{module}' at module level"
                )

    assert not offenders, (
        "These modules ship inside the .vsix but import a tree that staging drops, so the "
        "packaged backend cannot import and the extension can never start it:\n  "
        + "\n  ".join(offenders)
        + "\n\nFix by moving the import under `if TYPE_CHECKING:` (annotations only) or into "
        "the function that needs it — not by staging the tree, which is CI-only on purpose."
    )


def test_the_staging_script_still_drops_tests_and_evals() -> None:
    """If staging stops excluding these, the .vsix grows by the eval datasets silently."""
    excluded = _excluded_dirs()
    assert {"tests", "evals", "__pycache__"} <= excluded, sorted(excluded)


def test_staging_omits_developer_reports_and_keeps_runtime_assets(tmp_path: Path) -> None:
    """Exercise the copy filter: a local audit report must not ship to users."""
    extension = tmp_path / "extension"
    script = extension / "scripts" / STAGE_SCRIPT.name
    script.parent.mkdir(parents=True)
    shutil.copyfile(STAGE_SCRIPT, script)
    for name in ("stage-backend.mjs", "assets.mjs"):
        target = tmp_path / "tooling" / "runtime" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / "tooling" / "runtime" / name, target)
    (tmp_path / "runtime" / "python").mkdir(parents=True)
    (tmp_path / "runtime" / "licenses").mkdir()
    shutil.copyfile(PROFILE, tmp_path / "runtime" / PROFILE.name)
    (tmp_path / "runtime" / "licenses" / "tldraw-5.0.1.md").write_text("Fixture notice")
    backend = tmp_path / "backend"
    assets = {
        "server_v2.py": "app = object()\n",
        "requirements.txt": "fastapi\n",
        "requirements-cpu.txt": "torch\n",
        "requirements-test.txt": "pytest\n",
        "services/ide/runtime_asset.json": '{"keep": true}',
        "reports/nested/local-audit.md": "Developer-only report\n",
        ".env": "not a runtime asset",
        "services/.env.production": "not a runtime asset",
    }
    for name, content in assets.items():
        path = backend / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    app = extension / "media" / "app" / "index.html"
    app.parent.mkdir(parents=True)
    app.write_text("<html></html>")
    (tmp_path / "runtime" / "python" / "manifest.json").write_text(json.dumps({
        "platforms": {key: {"sha256": "a" * 64}
                      for key in ("darwin-arm64", "linux-x64", "win32-x64")},
    }))
    # Restaging must remove a report left by an earlier build as well.
    stale = extension / "backend" / "reports" / "old.md"
    stale.parent.mkdir(parents=True)
    stale.write_text("Old report")

    subprocess.run(["node", str(script)], check=True, capture_output=True, text=True)

    staged = extension / "backend"
    assert not (staged / "reports").exists()
    assert not (staged / ".env").exists()
    assert not (staged / "services/.env.production").exists()
    assert not (staged / "requirements-test.txt").exists()
    for name, content in assets.items():
        if not name.startswith("reports/") and ".env" not in name and name != "requirements-test.txt":
            assert (staged / name).read_text() == content
    assert (backend / "reports/nested/local-audit.md").read_text() == assets["reports/nested/local-audit.md"]


def test_the_staging_script_excludes_every_environment_file() -> None:
    """A developer credential must never become an installable VSIX asset."""

    config = json.loads(PROFILE.read_text())
    assert ".env" in config["excludedFileNames"]
    assert ".env." in config["excludedFilePrefixes"]
