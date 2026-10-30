"""Pre-started, single-use capture worker: pay for ``import torch`` before the request.

Importing torch is roughly two thirds of an ordinary graph run. The host starts
this process ahead of the next run; it imports torch and tensorViz's own capture
internals (the private ``_tensorviz_capture_worker`` package and the modules a run
imports lazily, about 120-150 ms more), and then waits for exactly one request on
stdin. That request carries the argument list and the per-run
environment the host would otherwise have spawned ``bootstrap.py`` with. The
ordinary bootstrap then runs once, unchanged, as ``__main__``, and the process
exits. It is never reused. Pre-loading is best effort: if it fails, the run loads
the internals itself exactly as a fresh process would.

Single use is the whole point. A long-lived process that had imported the user's
model would serve the old module from ``sys.modules`` after an edit, and carry
seeds, thread counts and patched classes from one run into the next. Here the
user's code is still imported only after the request, into a process nothing else
has touched. Installed packages it pre-loaded can still change while it waits (an edit
in site-packages, a reinstall): when the request arrives it rechecks the stamps it took
of their files (``reuse_audit.remember_preloaded``) and, if any changed, replaces
itself with the ordinary bootstrap (``_continue_fresh``, POSIX only).

stdin is also the parent-death pipe (``worker_lifetime.guard_parent_pipe``). The
host writes one line and nothing more, so after the request has been read the
worker's own guard sees exactly what it would have seen: an open pipe that reaches
EOF only when the host goes away. EOF before any request means the host retired
this standby or exited; that is a clean exit, not an error. Nothing is written to
stdout here, because stdout carries the run's envelope.
"""
import builtins
import importlib.machinery
import json
import os
import sys
import types

MAX_REQUEST = 4 * 1024 * 1024
#: The environment this process started with, taken before torch or anything pre-loaded
#: could change it: what an ordinary launch of the request starts from (``_continue_fresh``).
STARTED_ENVIRON = dict(os.environ)


def _read_request():
    """One newline-terminated JSON object, or None if stdin closed first."""
    data = bytearray()
    while True:
        chunk = os.read(0, 65536)
        if not chunk:
            return None
        data += chunk
        if len(data) > MAX_REQUEST:
            raise ValueError('standby request too large')
        if data.endswith(b'\n'):
            break
        if b'\n' in data:
            # The host writes exactly one line. Bytes after it would be read by
            # nobody and could hide a protocol error, so refuse rather than guess.
            raise ValueError('standby request has trailing data')
    request = json.loads(data)
    argv, env = request.get('argv'), request.get('env')
    if (type(argv) is not list or not all(type(item) is str for item in argv)
            or type(env) is not dict or not all(type(k) is str and type(v) is str for k, v in env.items())):
        raise ValueError('malformed standby request')
    return argv, env


#: Imported by the worker inside functions during a run; pre-loading them is the
#: point. Each was audited for import-time side effects (none read the environment,
#: argv, cwd or clocks, or touch torch state). A name missing in some backend build is
#: simply skipped.
PRELOAD = (
    'ingest.service', 'ingest.normalise', 'ingest.spec', 'ingest.execution_spec', 'ingest.execution_sources',
    'ingest.tensor_inputs', 'ingest.worker_lifetime', 'ingest.execution_setup', 'ingest.execution_evidence',
    'ingest.execution_values', 'semantic.contracts', 'semantic.execution_plan', 'semantic.recorder',
    'semantic.source_bindings', 'semantic.construction', 'validation.setup',
    'validation.contracts', 'validation.config', 'validation.source', 'ingest.reuse_audit',
)


def _preload(bootstrap):
    """Load the private capture package, so the run that claims this standby finds it
    imported, and return its reuse audit module, which has stamped what is loaded. Best
    effort: any failure returns None and leaves the process as if nothing was
    pre-loaded, and the run imports what it needs itself, as it always has.

    ``bootstrap.py`` is loaded under a private name and removed again, so no module
    called ``bootstrap`` is left in ``sys.modules`` to shadow a workspace's own."""
    import importlib
    import importlib.util
    name = '_tensorviz_standby_bootstrap'
    try:
        spec = importlib.util.spec_from_file_location(name, bootstrap)
        loader = importlib.util.module_from_spec(spec)
        sys.modules[name] = loader
        try:
            spec.loader.exec_module(loader)
            loader.load_internals()
        finally:
            sys.modules.pop(name, None)
        for module in PRELOAD:
            try:
                importlib.import_module(f'{loader.PACKAGE}.{module}')
            except ImportError:
                pass
        # Fingerprint what is loaded now, before waiting: a file edited while this process
        # waits must keep the fingerprint of the code it loaded (see reuse_audit).
        audit = sys.modules[f'{loader.PACKAGE}.ingest.reuse_audit']
        audit.remember_preloaded()
    except Exception:  # noqa: BLE001 - pre-loading is an optimisation, never a failure
        return None
    return audit


def main():
    import torch  # noqa: F401  (the work this process exists to do early)
    bootstrap = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bootstrap.py')
    audit = _preload(bootstrap)
    try:
        request = _read_request()
    except Exception as error:
        sys.stderr.write(f'tensorViz standby could not read its request: {error}\n')
        return 2
    if request is None:
        return 0
    argv, env = request
    if audit is not None and os.name == 'posix' and audit.preloaded_changed():
        _continue_fresh(bootstrap, argv, env)
    os.environ.update(env)
    sys.argv = [bootstrap, *argv]
    _run_as_main(bootstrap)
    return 0


def _continue_fresh(bootstrap, argv, env):
    """Become the ordinary worker this request would have spawned. A file this process
    pre-loaded changed while it waited (an installed package edited or reinstalled), so
    the modules it holds are not what the run should execute.

    ``exec`` keeps the pid, the process group, the stdio pipes and the host's timers, so
    the host still cancels, kills and times the run and reads its envelope as before;
    stdin, the parent-death pipe, is already past the request line. The new process gets
    the environment this one started with plus the request's, and the arguments an
    ordinary launch has. POSIX only: on Windows exec starts another process, which the
    host would read as this one ending without output; there, and if exec fails, the run
    continues here and the cache never serves its result (the stamps it reports predate
    the edit)."""
    sys.stderr.write('tensorViz standby: pre-loaded files changed; continuing as a fresh worker\n')
    sys.stderr.flush()
    try:
        os.execve(sys.executable, [sys.orig_argv[0], bootstrap, *argv], {**STARTED_ENVIRON, **env})
    except Exception as error:  # noqa: BLE001 - exec failed and changed nothing: run here, as before
        sys.stderr.write(f'tensorViz standby could not restart: {error}\n')


def _run_as_main(script):
    """Run ``script`` the way ``python script`` does, so user code cannot tell a
    standby run from an ordinary one: a fresh ``__main__`` with the same
    ``__file__``, ``__loader__`` and ``__spec__`` CPython's file runner sets, and no
    extra modules (``runpy`` would be one). No ``bootstrap`` module is registered,
    so a workspace module of that name is not shadowed through ``sys.modules``."""
    main = types.ModuleType('__main__')
    main.__file__ = script
    main.__cached__ = None
    main.__builtins__ = builtins
    main.__loader__ = importlib.machinery.SourceFileLoader('__main__', script)
    main.__spec__ = None
    sys.modules['__main__'] = main
    with open(script, 'rb') as stream:
        code = compile(stream.read(), script, 'exec', dont_inherit=True)
    exec(code, main.__dict__)


if __name__ == '__main__':
    sys.exit(main())
