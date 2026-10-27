"""The project-store seam: ``services.store_factory.get_project_store`` (task.md M9).

The store's location is about to become per-workspace. That decision used to be
copied into seven modules, where "switch them together" meant "find them all
again and hope". These tests hold the seam shut: the factory must resolve the
database the way the call sites did, resolve it *per call* (a cached singleton
would pin the first workspace a process saw onto every later one), and remain the
only place in shipped code that constructs a store.
"""

import os
import re
from pathlib import Path

from services.project_store import ProjectStore
from services.store_factory import get_project_store

_BACKEND = Path(__file__).parent.resolve().parents[1]


def test_factory_honours_the_configured_database(monkeypatch, tmp_path: Path):
    """Same contract the call sites had: USER_HISTORY_DB_PATH decides."""
    db = tmp_path / "workspace" / ".tensorviz" / "state.sqlite3"
    monkeypatch.setenv("USER_HISTORY_DB_PATH", str(db))

    store = get_project_store()
    store.create_project(owner_uid="workspace:abc", name="Sanity")

    assert db.exists()
    assert [p["name"] for p in store.list_projects("workspace:abc")] == ["Sanity"]


def test_explicit_path_beats_the_environment(monkeypatch, tmp_path: Path):
    """A caller that has already resolved a workspace must not be overruled by
    whatever the process happens to have in its environment."""
    monkeypatch.setenv("USER_HISTORY_DB_PATH", str(tmp_path / "env.sqlite3"))
    chosen = tmp_path / "explicit.sqlite3"

    get_project_store(str(chosen)).create_project(owner_uid="u", name="Explicit")

    assert chosen.exists()
    assert not (tmp_path / "env.sqlite3").exists()


def test_resolution_happens_per_call(monkeypatch, tmp_path: Path):
    """Two workspaces in one process must not share one store. A singleton cached
    at import would pass every other test in this file and fail here."""
    first, second = tmp_path / "a.sqlite3", tmp_path / "b.sqlite3"

    monkeypatch.setenv("USER_HISTORY_DB_PATH", str(first))
    get_project_store().create_project(owner_uid="u", name="Only in A")
    monkeypatch.setenv("USER_HISTORY_DB_PATH", str(second))
    store_b = get_project_store()

    assert second.exists()
    assert store_b.list_projects("u") == []


def test_factory_returns_a_project_store(monkeypatch, tmp_path: Path):
    """The seam is a swap point, not a rewrite: callers keep the same type."""
    monkeypatch.setenv("USER_HISTORY_DB_PATH", str(tmp_path / "state.sqlite3"))
    assert isinstance(get_project_store(), ProjectStore)


def test_no_shipped_module_constructs_a_store_directly():
    """The acceptance grep, as a test that can fail: a new ``ProjectStore()`` in a
    router or agent would silently opt that surface out of the seam."""
    offenders = []
    for root, dirs, files in os.walk(_BACKEND):
        dirs[:] = [d for d in dirs if d not in {"tests", "__pycache__", ".runtime"}]
        for name in files:
            if not name.endswith(".py") or name == "store_factory.py":
                continue
            path = Path(root) / name
            for number, line in enumerate(path.read_text().splitlines(), start=1):
                if re.search(r"(?<![\w.])ProjectStore\(", line) and not line.lstrip().startswith("#"):
                    offenders.append(f"{path.relative_to(_BACKEND)}:{number}: {line.strip()}")

    assert offenders == []
