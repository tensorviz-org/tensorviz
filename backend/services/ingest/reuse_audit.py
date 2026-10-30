"""Untracked-read detector for the execution cache (layer 0).

A cached worker result is valid only if everything the run read is part of the cache key.
The key covers the resolved plan (every workspace ``.py`` and the declared config,
fixture, checkpoint and setup files, by content digest), the interpreter and its
installed package versions, and the backend's content digest. Anything else a model
reads — a data file next to it, a file in ``~/.cache``, the source a ``.pth`` file or an
editable install's import hook points to — is invisible to the key, so a run that read
it must not be stored.

``ReadAudit`` installs a ``sys.addaudithook`` hook (once per process; hooks cannot be
removed) and, while active, classifies every ``open`` event. ``open`` is raised by
``open()``, ``io.open``, ``os.open`` and ``io.open_code`` (which the import system uses
for source and bytecode), so it also sees imports. Directory listings (``os.listdir``,
``os.scandir``, and so ``glob`` and ``pathlib``) are classified the same way. Existence
checks raise no audit event, so ``os.stat``, ``os.lstat``, ``os.access`` and the
``os.path`` predicates are wrapped while the audit is active. Filesystem mutations,
process spawns and socket connections are recorded as well.

Native code opens files without an audit event (safetensors, tokenizers, h5py, cv2, a
compiled extension in the workspace), so a run that loads native code the audit does not
trust is not storable (``native_code``): an extension module first imported during the
run, other than the standard library's and torch's or numpy's own, or any ``ctypes``
library. This refuses some runs that read nothing; correctness comes first. Torch's own
path loaders (``torch.jit.load``, ``torch.package``) check the path from Python first,
which the audit sees. Not covered: ``torch.from_file``/``UntypedStorage.from_file``.

Reads under the interpreter's own trees (stdlib, site-packages) are keyed only by package
versions, which an in-place edit or a same-version reinstall keeps, so the audit records
each such file's stat fingerprint (``environment``) and the host rechecks it before
serving a stored result. A file reached through a link from those trees is part of the
installation wherever the link points (Debian's ``/usr/lib/python3.X/sitecustomize.py``
links to ``/etc``; Homebrew links each formula's package into site-packages from its
Cellar, and a symlinked editable install, ``flit install --symlink``, looks the same): it
is recorded under its path through the trees, whose stat follows the link. A link never
makes a workspace file part of the installation. Otherwise the path the read used decides,
by the deepest of its folders that resolves into the trees or into the workspace: a read
through the workspace's own link to a folder elsewhere stays untracked, while a read
through a link in the trees is installed even if that link reaches its file through such a
workspace link.

Modules imported before the audit starts (torch and what it imports, numpy, pydantic, the
worker's own imports) read their files while no hook was active. ``start`` therefore also
classifies the source and bytecode of every loaded module as an import inside the audit
would have been: an installed one is fingerprinted, and code from outside the
interpreter's trees, the workspace's keyed sources and the backend makes the run not
storable. Each fingerprint is taken the first time its module is seen loaded: inside
``start`` in a fresh worker, and in the pre-started worker right after it pre-loads,
before it waits for its request; a file edited during that wait is then refused rather
than recorded under the old code. On POSIX the pre-started worker also rechecks those
stamps when its request arrives (``preloaded_changed``) and, if any changed, becomes an
ordinary worker instead. Not covered: code whose module is no longer in ``sys.modules``
under its own file (``torch._VF`` and torch's ``torch.backends`` wrappers replace their
own entries), modules imported from a zip archive, other files read before the audit
(``.pth`` files, the ``entry_points.txt`` torch scans for device backends), native
libraries, and an edit that lands between a module's import and its first stamp: in a
fresh worker before ``start``, in the pre-started worker while it pre-loads.

The report holds a storable flag, closed-enum reasons, a count of untracked files read
and those fingerprints. An untracked path is never reported: diagnostics travel to the
host and must not leak file names. The fingerprinted files are the interpreter's own
installation, whose location the plan already carries.
"""
from __future__ import annotations

import functools
import importlib.machinery
import os
import site
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

#: Reasons, as reported in ``diagnostics.reuse.reasons`` (closed set, sorted on output).
WRITE = "write"
WORKSPACE_READ = "untracked_workspace_read"
EXTERNAL_READ = "untracked_external_read"
SUBPROCESS = "subprocess"
NETWORK = "network"
NATIVE = "native_code"

#: Reads whose content cannot change a result. ``/dev/null`` reads as empty. Entropy
#: devices are deliberately absent: a model reading them is not reproducible.
BENIGN = frozenset({"/dev/null"})

#: Packages whose extension modules are trusted not to read files behind the audit's back.
NATIVE_ALLOWLIST = frozenset({"torch", "numpy"})
_EXTENSION_SUFFIXES = tuple(importlib.machinery.EXTENSION_SUFFIXES)

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
_MUTATIONS = frozenset({"os.mkdir", "os.remove", "os.rename", "os.rmdir", "os.truncate", "os.link",
                        "os.symlink", "shutil.copyfile", "shutil.move", "shutil.rmtree"})
_SPAWNS = frozenset({"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork",
                     "os.forkpty", "pty.spawn"})
_NETWORK = frozenset({"socket.connect", "socket.sendto", "socket.bind"})
_LISTINGS = frozenset({"os.listdir", "os.scandir"})
#: Everything the hook looks at. A run raises tens of thousands of audit events (mostly
#: ``object.__getattr__``, ``builtins.id`` and ``sys._getframe`` from the recorder), so
#: the hook's first and only work for those is one frozenset membership test.
_WATCHED = frozenset({"open", "ctypes.dlopen", "sqlite3.connect"}) | _LISTINGS | _MUTATIONS | _SPAWNS | _NETWORK

#: How a path was observed, and what the classification found.
_READ, _LIST, _FIND, _STAT = "read", "list", "find", "stat"
_COVERED, _ENVIRONMENT, _WORKSPACE, _EXTERNAL = "covered", "environment", "workspace", "external"

#: The single per-process audit whose hook is installed; a later ``ReadAudit`` replaces it.
_active: Optional["ReadAudit"] = None
_installed = False
#: Set while this thread classifies an event, so the classifier's own stat calls are not
#: classified. Per thread: another thread's reads in the meantime still count.
_guard = threading.local()

_STAT_FN = os.stat
#: Existence checks raise no audit event: (module, name) wrapped while an audit is active.
#: On POSIX the ``os.path`` predicates call ``os.stat``; on Windows they are native.
_CHECKS = ((os, "stat"), (os, "lstat"), (os, "access"), (os.path, "exists"), (os.path, "lexists"),
           (os.path, "isfile"), (os.path, "isdir"), (os.path, "islink"))
_originals: Dict[tuple, Tuple[Callable, Callable]] = {}  # (module, name) -> (original, wrapper)
#: Callers test functions against these at call time (``shutil.copystat`` asks whether
#: ``os.stat in os.supports_follow_symlinks``), so a wrapper joins each set its original is in.
_SUPPORTS = (os.supports_dir_fd, os.supports_effective_ids, os.supports_fd, os.supports_follow_symlinks)


def _hook(event: str, args: tuple) -> None:
    if event not in _WATCHED:
        return
    audit = _active
    if audit is None or getattr(_guard, "busy", False):
        return
    # An audit hook that raises aborts the audited operation; this one must never.
    _guard.busy = True
    try:
        if event in _LISTINGS:
            # The import system lists each path entry once to find modules (FileFinder).
            caller = sys._getframe(1).f_code
            finder = caller.co_name == "_fill_cache" and caller.co_filename.endswith("importlib._bootstrap_external>")
            audit._listing(args[0] if args else None, finder)
        else:
            audit._event(event, args)
    except Exception:  # noqa: BLE001 - classification failure cannot be proven benign
        audit.reasons.add(EXTERNAL_READ)
        audit._untracked.add(("<unclassified>", event))
    finally:
        _guard.busy = False


def _existence_check(original: Callable) -> Callable:
    @functools.wraps(original)
    def check(path, *args, **kwargs):
        audit = _active
        if audit is not None and not getattr(_guard, "busy", False):
            _guard.busy = True
            try:
                audit._stat(path, kwargs.get("dir_fd"))
            except Exception:  # noqa: BLE001 - as in _hook
                audit.reasons.add(EXTERNAL_READ)
                audit._untracked.add(("<unclassified>", _STAT))
            finally:
                _guard.busy = False
        return original(path, *args, **kwargs)
    return check


def _wrap_checks() -> None:
    for module, name in _CHECKS:
        if (module, name) not in _originals:
            original = getattr(module, name)
            check = _existence_check(original)
            _originals[(module, name)] = (original, check)
            setattr(module, name, check)
            for supports in _SUPPORTS:
                if original in supports:
                    supports.add(check)


def _unwrap_checks() -> None:
    for (module, name), (original, check) in list(_originals.items()):
        if getattr(getattr(module, name), "__wrapped__", None) is original:
            setattr(module, name, original)
        for supports in _SUPPORTS:
            supports.discard(check)
        del _originals[(module, name)]


def _real(path: str) -> str:
    return os.path.realpath(os.path.abspath(path))


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def fingerprint(path: str) -> Optional[str]:
    """What the host compares before serving: ``size:mtime_ns`` (``:ctime_ns`` on POSIX,
    where every write, even one that restores mtime, updates it), or None when nothing is
    there. The host computes the same string with Node's ``fs.statSync``."""
    try:
        info = _STAT_FN(path)
    except (FileNotFoundError, NotADirectoryError):
        return None
    stamp = f"{info.st_size}:{info.st_mtime_ns}"
    return f"{stamp}:{info.st_ctime_ns}" if os.name == "posix" else stamp


def _covered_prefixes() -> List[str]:
    """The interpreter's own trees (stdlib and site-packages): reads under them are proven
    by the package versions in the key plus the fingerprints the host rechecks."""
    roots: Set[str] = set()
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


#: Module origin -> the (path the import system used, real path, fingerprint) of its source
#: and its bytecode, taken the first time ``remember_preloaded`` saw the module loaded.
_remembered: Dict[str, Tuple[Tuple[str, str, Optional[str]], ...]] = {}
#: Set once a loaded module's file could not be fingerprinted: no audit in this process can
#: then prove what that code was.
_unproven = False
_MODULE_SUFFIXES = (*importlib.machinery.SOURCE_SUFFIXES, *importlib.machinery.BYTECODE_SUFFIXES)


def _module_files(module: Any) -> Tuple[Any, Any]:
    """``(origin, cached)`` as the import system set them, read from the module's own
    namespace: an attribute lookup can run a lazily loaded module (``importlib.util``'s
    ``LazyLoader``) or a module class's ``__getattr__``."""
    try:
        namespace = object.__getattribute__(module, "__dict__")
    except AttributeError:
        return None, None
    return getattr(namespace.get("__spec__"), "origin", None) or namespace.get("__file__"), namespace.get("__cached__")


def _stamped(path: str, folders: Dict[str, str]) -> Tuple[str, str, Optional[str]]:
    """``(path, _real(path), fingerprint)``, resolving each folder once: a few thousand module
    files share a few hundred folders."""
    if os.path.islink(path):
        real = _real(path)
    else:
        head, tail = os.path.split(path)
        folder = folders.get(head)
        if folder is None:
            folder = folders[head] = _real(head)
        real = os.path.join(folder, tail)
    return path, real, fingerprint(real)


def remember_preloaded() -> Optional[List[Tuple[str, str, Optional[str]]]]:
    """The source and bytecode files of every module loaded now, as ``[path the import
    system used, real path, fingerprint]``, each fingerprint taken the first time its module
    was seen here; None once a file could not be fingerprinted.

    Modules imported before an audit starts (torch and what it imports, numpy, pydantic,
    the worker's own imports) raised their ``open`` events while no hook was active. The
    pre-started worker calls this right after it pre-loads, so a file edited while it waits
    keeps the fingerprint of the code it actually loaded and the host never serves that
    run's result. Extension modules raise no ``open`` event inside the audit either; a
    reinstall rewrites their package's Python files as well."""
    global _unproven
    folders: Dict[str, str] = {}
    files: List[Tuple[str, str, Optional[str]]] = []
    try:
        for module in list(sys.modules.values()):
            origin, cached = _module_files(module)
            if not isinstance(origin, str) or not origin.endswith(_MODULE_SUFFIXES) or not os.path.isabs(origin):
                continue  # built-in, frozen, namespace and extension modules
            entries = _remembered.get(origin)
            if entries is None:
                entries = (_stamped(origin, folders),)
                if isinstance(cached, str) and cached != origin:
                    bytecode = _stamped(cached, folders)
                    entries += (bytecode,) if bytecode[2] is not None else ()
                _remembered[origin] = entries
            files.extend(entries)
    except Exception:  # noqa: BLE001 - a file that cannot be fingerprinted cannot be proven unchanged
        _unproven = True
    return None if _unproven else files


def preloaded_changed() -> bool:
    """Whether a file ``remember_preloaded`` stamped has changed since, or one could not be
    stamped or checked: this process then no longer holds the code an ordinary launch would
    load. Each file is checked by the path the import system used, as that launch would find
    it: a link there re-pointed to another file (``brew upgrade``) changes it too. The
    pre-started worker asks once its request arrives (see standby.py)."""
    try:
        return _unproven or any(fingerprint(path) != stamp for entries in _remembered.values() for path, _real, stamp in entries)
    except Exception:  # noqa: BLE001 - as in remember_preloaded
        return True


class ReadAudit:
    """Records reads outside the covered set and any write, while active."""

    def __init__(self, workspace: str, covered_files: Iterable[str], backend: Optional[Path] = None,
                 extra_prefixes: Iterable[str] = (), covered_dirs: Iterable[str] = ()) -> None:
        from ..validation.host_snapshot import EXCLUDED
        self.workspace = _real(workspace)
        self.files = {_real(os.path.join(self.workspace, f)) for f in covered_files}
        backend = backend if backend is not None else Path(__file__).resolve().parents[2]
        self.backend = _real(str(backend))
        self.prefixes = tuple(sorted({*_covered_prefixes(), *(_real(p) for p in extra_prefixes)}))
        self.base = tuple(_real(p) for p in {sys.base_prefix, sys.base_exec_prefix} if p)
        self.excluded = EXCLUDED
        # Folders whose existence the key already implies: the workspace, the folders of
        # every covered file, the execution cwd, and the parents of all of these and of the
        # backend and the interpreter's trees.
        self.dirs: Set[str] = set()
        for start in (self.workspace, *(_real(os.path.join(self.workspace, d)) for d in covered_dirs)):
            self._add_parents(start, inclusive=True)
        for start in (*self.files, self.backend, *self.prefixes):
            self._add_parents(start, inclusive=False)
        self.reasons: Set[str] = set()
        self._untracked: Set[tuple] = set()
        self._environment: Dict[str, Optional[str]] = {}
        self._seen: Dict[tuple, Tuple[str, str]] = {}
        self._resolved: Dict[Any, Tuple[str, str]] = {}
        self._modules: Set[str] = set()

    def _add_parents(self, path: str, inclusive: bool) -> None:
        current = path if inclusive else os.path.dirname(path)
        while current not in self.dirs:
            self.dirs.add(current)
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent

    @classmethod
    def for_plan(cls, plan: Any, workspace: str) -> "ReadAudit":
        return cls(workspace, (f.path for f in plan.source.files), covered_dirs=[plan.execution.get("cwd", ".")])

    def start(self) -> "ReadAudit":
        global _active, _installed
        # Before the hook and the wrapped checks: these stats are the audit's, not the run's.
        self._preloaded()
        if not _installed:
            sys.addaudithook(_hook)
            _installed = True
        self._modules = set(sys.modules)
        _wrap_checks()
        _active = self
        return self

    def _preloaded(self) -> None:
        """Classify the files of modules loaded before the audit as an import inside it
        would have been: an installed file is fingerprinted (with its stamp from when it was
        first seen), and one the key does not cover is an untracked read. Never fails the run."""
        try:
            files = remember_preloaded()
            for path, real, stamp in files or ():
                verdict, key = self._reached(path, real, _READ)
                if verdict == _ENVIRONMENT:
                    self._environment.setdefault(key, stamp)
                elif verdict in (_WORKSPACE, _EXTERNAL):
                    self.reasons.add(WORKSPACE_READ if verdict == _WORKSPACE else EXTERNAL_READ)
                    self._untracked.add((real, _READ))
        except Exception:  # noqa: BLE001 - as in _hook
            files = None
        if files is None:
            self.reasons.add(EXTERNAL_READ)

    def stop(self) -> Dict[str, Any]:
        global _active
        if _active is self:
            _active = None
            _unwrap_checks()
        try:
            native = self._loaded_untrusted_native()
        except Exception:  # noqa: BLE001 - stop() runs in the worker's finally; a check that fails proves nothing
            native = True
        if native:
            self.reasons.add(NATIVE)
        return self.report()

    def report(self) -> Dict[str, Any]:
        reads = sum(1 for _path, kind in self._untracked if kind == _READ)
        storable = not self.reasons
        environment = sorted([path, stamp] for path, stamp in self._environment.items()) if storable else []
        return {"storable": storable, "reasons": sorted(self.reasons), "untrackedReads": reads,
                "environment": environment}

    # -- native code ------------------------------------------------------------------
    def _loaded_untrusted_native(self) -> bool:
        for name in set(sys.modules) - self._modules:
            origin, _ = _module_files(sys.modules.get(name))
            if isinstance(origin, str) and origin.endswith(_EXTENSION_SUFFIXES) and not self._trusted_native(name, _real(origin)):
                return True
        return False

    def _trusted_native(self, name: str, origin: str) -> bool:
        """The standard library's extension modules, from the base interpreter, torch's and
        numpy's own, from the installed package, and a compiled backend's (in the backend
        digest). Nothing from the workspace's own files; a ``.venv`` in it is the environment."""
        if _under(origin, self.workspace) and not self._in_environment(origin):
            return False
        if _under(origin, self.backend):
            return True
        top = name.partition(".")[0]
        if top in sys.stdlib_module_names:
            return any(_under(origin, base) for base in self.base)
        if top not in NATIVE_ALLOWLIST:
            return False
        package, _ = _module_files(sys.modules.get(top))
        return isinstance(package, str) and _under(origin, os.path.dirname(_real(package)))

    # -- classification -------------------------------------------------------------
    def _in_environment(self, real: str) -> bool:
        """Under the interpreter's trees. A workspace inside a prefix (/usr/src/app with
        sys.prefix /usr) is still the workspace; a prefix inside the workspace (.venv) is
        still the environment, also when both nest (/usr/src/app/.venv)."""
        prefixes = [p for p in self.prefixes if _under(real, p)]
        return bool(prefixes) and (not _under(real, self.workspace) or any(_under(p, self.workspace) for p in prefixes))

    def _resolve(self, path: Any) -> Tuple[str, str]:
        """``(absolute path, real path)``."""
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        path = os.fspath(path)
        key = path if os.path.isabs(path) else (os.getcwd(), path)
        resolved = self._resolved.get(key)
        if resolved is None:
            absolute = os.path.abspath(path)
            resolved = self._resolved[key] = (absolute, _real(absolute))
        return resolved

    def _bytecode_of_covered(self, real: str) -> bool:
        parent = os.path.dirname(real)
        if os.path.basename(parent) != "__pycache__":
            return False
        # ``name.cpython-312.pyc`` beside a covered ``name.py``.
        stem = os.path.basename(real).split(".", 1)[0]
        return os.path.join(os.path.dirname(parent), stem + ".py") in self.files

    def _inventoried(self, real: str) -> bool:
        """A workspace folder or ``.py`` path the snapshot walks: which sources exist there
        (and so whether this one does) is in the key."""
        if not _under(real, self.workspace):
            return False
        parts = os.path.relpath(real, self.workspace).split(os.sep)
        return not any(part in self.excluded for part in parts)

    def _classify(self, real: str, kind: str) -> str:
        if real in BENIGN or real in self.files or self._bytecode_of_covered(real) or _under(real, self.backend):
            return _COVERED
        if kind == _STAT and (real in self.dirs or (real.endswith(".py") and self._inventoried(real))):
            return _COVERED
        if self._in_environment(real):
            return _ENVIRONMENT
        if kind == _FIND and self._inventoried(real):
            # Which module an import finds there is seen through the file it then reads,
            # the .py inventory in the key and the native-code rule.
            return _COVERED
        return _WORKSPACE if _under(real, self.workspace) else _EXTERNAL

    def _reached(self, path: str, real: str, kind: str) -> Tuple[str, str]:
        """``_classify`` of ``real``, and the path to fingerprint it under. A file outside
        every covered tree that was reached through a link from a folder in the interpreter's
        trees (a distribution's ``sitecustomize.py`` linked from /etc, a Homebrew formula's
        package linked from its Cellar) is part of that installation: it is fingerprinted
        under its path through that folder, whose stat the host's recheck follows to the
        file, so the report names no path outside the installation. A file in the workspace
        keeps the workspace's rules. Otherwise the first folder up from the file that
        resolves into the environment or the workspace decides, so a path through the
        workspace's own link to a folder elsewhere stays external."""
        verdict = self._classify(real, kind)
        if verdict != _EXTERNAL:
            return verdict, real
        head, rest = os.path.abspath(path), []
        if head == real:
            return verdict, real  # no link on the way
        while True:
            # Up from the file, to the deepest folder whose real path is in the environment.
            head, tail = os.path.split(head)
            if not tail:
                return verdict, real
            rest.append(tail)
            folder = _real(head)
            if self._in_environment(folder):
                return _ENVIRONMENT, os.path.join(folder, *reversed(rest))
            if _under(folder, self.workspace):
                return verdict, real

    def _observe(self, path: Any, kind: str) -> None:
        absolute, real = self._resolve(path)
        # By the path used, not the file: whether a link in the environment reached it decides.
        seen = self._seen.get((absolute, kind))
        if seen is None:
            seen = self._seen[(absolute, kind)] = self._reached(absolute, real, kind)
            if seen[0] == _ENVIRONMENT:
                self._fingerprint(seen[1])
        verdict = seen[0]
        if verdict in (_WORKSPACE, _EXTERNAL):
            self.reasons.add(WORKSPACE_READ if verdict == _WORKSPACE else EXTERNAL_READ)
            self._untracked.add((real, kind if kind != _FIND else _LIST))

    def _fingerprint(self, path: str) -> None:
        # Taken before the read, so a later write shows as a different fingerprint.
        self._environment.setdefault(path, fingerprint(path))
        parent = os.path.dirname(path)
        if os.path.basename(parent) == "__pycache__":
            # The import system used this bytecode because its source's stat matched.
            source = os.path.join(os.path.dirname(parent), os.path.basename(path).split(".", 1)[0] + ".py")
            self._environment.setdefault(source, fingerprint(source))

    def _listing(self, path: Any, finder: bool) -> None:
        if isinstance(path, int):
            return  # a directory descriptor; its open was classified when it happened
        self._observe("." if path is None else path, _FIND if finder else _LIST)

    def _stat(self, path: Any, dir_fd: Any) -> None:
        if isinstance(path, int):
            return  # fstat of an open descriptor
        if dir_fd is not None and not os.path.isabs(os.fsdecode(os.fspath(path))):
            self.reasons.add(EXTERNAL_READ)  # relative to a folder this audit cannot name
            return
        self._observe(path, _STAT)

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
        if event == "ctypes.dlopen":
            self.reasons.add(NATIVE)
            return
        if event == "sqlite3.connect":
            database = args[0] if args else None
            if database not in (":memory:", "", b":memory:", b""):
                self._observe(database, _READ)
            return
        path, mode, flags = (tuple(args) + (None, None, None))[:3]
        if isinstance(path, int):
            return  # an already-open descriptor; its open was classified when it happened
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        path = os.fspath(path)
        writes = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            mode is None and isinstance(flags, int) and bool(flags & _WRITE_FLAGS))
        if writes:
            real = _real(path)
            if real not in BENIGN:
                self.reasons.add(WRITE)
                self._untracked.add((real, "write"))
            return
        self._observe(path, _READ)
