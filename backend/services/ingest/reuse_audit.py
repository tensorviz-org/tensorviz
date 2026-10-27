"""Untracked-read detector for the execution cache (layer 0).

A cached worker result is valid only if everything the run read is part of the cache key.
The key covers the resolved plan (every workspace ``.py`` and the declared config,
fixture, checkpoint and setup files, by content digest), the interpreter and its
installed package versions, and the backend's content digest. Anything else a model
reads — a data file next to it, a file in ``~/.cache``, an editable install's source —
is invisible to the key, so a run that read it must not be stored.

``ReadAudit`` installs a ``sys.addaudithook`` hook (once per process; hooks cannot be
removed) and, while active, classifies every ``open`` event. ``open`` is raised by
``open()``, ``io.open``, ``os.open`` and ``io.open_code`` (which the import system uses
for source and bytecode), so it also sees imports. Filesystem mutations, process spawns
and socket connections are recorded as well.

The report holds a storable flag, closed-enum reasons and a count — never a path:
diagnostics travel to the host and must not leak file names.
"""
from __future__ import annotations

import os
import site
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

#: Reasons, as reported in ``diagnostics.reuse.reasons`` (closed set, sorted on output).
WRITE = "write"
WORKSPACE_READ = "untracked_workspace_read"
EXTERNAL_READ = "untracked_external_read"
SUBPROCESS = "subprocess"
NETWORK = "network"

#: Reads whose content cannot change a result. ``/dev/null`` reads as empty. Entropy
#: devices are deliberately absent: a model reading them is not reproducible.
BENIGN = frozenset({"/dev/null"})

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
_MUTATIONS = frozenset({"os.mkdir", "os.remove", "os.rename", "os.rmdir", "os.truncate", "os.link",
                        "os.symlink", "shutil.copyfile", "shutil.move", "shutil.rmtree"})
_SPAWNS = frozenset({"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork",
                     "os.forkpty", "pty.spawn"})
_NETWORK = frozenset({"socket.connect", "socket.sendto", "socket.bind"})
#: Everything the hook looks at. A run raises tens of thousands of audit events (mostly
#: ``object.__getattr__``, ``builtins.id`` and ``sys._getframe`` from the recorder), so
#: the hook's first and only work for those is one frozenset membership test.
_WATCHED = frozenset({"open"}) | _MUTATIONS | _SPAWNS | _NETWORK

#: The single per-process audit whose hook is installed; a later ``ReadAudit`` replaces it.
_active: Optional["ReadAudit"] = None
_installed = False


def _hook(event: str, args: tuple) -> None:
    if event not in _WATCHED:
        return
    audit = _active
    if audit is None or audit._busy:
        return
    # An audit hook that raises aborts the audited operation; this one must never.
    audit._busy = True
    try:
        audit._event(event, args)
    except Exception:  # noqa: BLE001 - classification failure cannot be proven benign
        audit.reasons.add(EXTERNAL_READ)
        audit._untracked.add(("<unclassified>", event))
    finally:
        audit._busy = False


def _real(path: str) -> str:
    return os.path.realpath(os.path.abspath(path))


def _covered_prefixes(backend: Path) -> List[str]:
    """The interpreter's own trees (stdlib and site-packages; covered by the plan's
    Python/torch/package versions) and the tensorViz backend (covered by its digest)."""
    roots: Set[str] = {str(backend)}
    for value in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix):
        if value:
            roots.add(value)
    try:
        roots.update(site.getsitepackages())
    except Exception:  # noqa: BLE001 - some embedded interpreters lack it
        pass
    if site.ENABLE_USER_SITE:
        try:
            roots.add(site.getusersitepackages())
        except Exception:  # noqa: BLE001
            pass
    resolved = set()
    for root in roots:
        try:
            resolved.add(_real(root))
        except (OSError, ValueError):
            continue
    return sorted(resolved)


class ReadAudit:
    """Records reads outside the covered set and any write, while active."""

    def __init__(self, workspace: str, covered_files: Iterable[str], backend: Optional[Path] = None,
                 extra_prefixes: Iterable[str] = ()) -> None:
        self.workspace = _real(workspace)
        self.files = {_real(os.path.join(self.workspace, f)) for f in covered_files}
        backend = backend if backend is not None else Path(__file__).resolve().parents[2]
        self.prefixes = tuple(sorted({*_covered_prefixes(backend), *(_real(p) for p in extra_prefixes)}))
        self.reasons: Set[str] = set()
        self._untracked: Set[tuple] = set()
        self._seen: Dict[str, bool] = {}
        self._busy = False

    @classmethod
    def for_plan(cls, plan: Any, workspace: str) -> "ReadAudit":
        return cls(workspace, (f.path for f in plan.source.files))

    def start(self) -> "ReadAudit":
        global _active, _installed
        if not _installed:
            sys.addaudithook(_hook)
            _installed = True
        _active = self
        return self

    def stop(self) -> Dict[str, Any]:
        global _active
        if _active is self:
            _active = None
        return self.report()

    def report(self) -> Dict[str, Any]:
        reads = sum(1 for _path, kind in self._untracked if kind == "read")
        return {"storable": not self.reasons, "reasons": sorted(self.reasons), "untrackedReads": reads}

    # -- classification -------------------------------------------------------------
    def _covered(self, real: str) -> bool:
        cached = self._seen.get(real)
        if cached is not None:
            return cached
        covered = real in BENIGN or real in self.files or any(
            real == prefix or real.startswith(prefix + os.sep) for prefix in self.prefixes)
        if not covered:
            parent = os.path.dirname(real)
            if os.path.basename(parent) == "__pycache__":
                # ``name.cpython-312.pyc`` beside a covered ``name.py``.
                stem = os.path.basename(real).split(".", 1)[0]
                covered = os.path.join(os.path.dirname(parent), stem + ".py") in self.files
        self._seen[real] = covered
        return covered

    def _event(self, event: str, args: tuple) -> None:
        if event in _SPAWNS:
            self.reasons.add(SUBPROCESS)
            return
        if event in _NETWORK:
            self.reasons.add(NETWORK)
            return
        if event in _MUTATIONS:
            self.reasons.add(WRITE)
            return
        path, mode, flags = (tuple(args) + (None, None, None))[:3]
        if isinstance(path, int):
            return  # an already-open descriptor; its open was classified when it happened
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        path = os.fspath(path)
        writes = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            mode is None and isinstance(flags, int) and bool(flags & _WRITE_FLAGS))
        real = _real(path)
        if writes:
            if real not in BENIGN:
                self.reasons.add(WRITE)
                self._untracked.add((real, "write"))
            return
        if self._covered(real):
            return
        inside = real == self.workspace or real.startswith(self.workspace + os.sep)
        self.reasons.add(WORKSPACE_READ if inside else EXTERNAL_READ)
        self._untracked.add((real, "read"))
