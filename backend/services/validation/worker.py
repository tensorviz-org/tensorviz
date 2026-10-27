"""Private-file bootstrap and bounded socket transport for approved workers."""
from __future__ import annotations

from ..python_compat import sha256_stream
import importlib.machinery
import json
import os
from pathlib import Path
import socket
import stat
import struct
import sys

from . import contracts as c

BOOTSTRAP_ENV = "TENSORVIZ_VALIDATION_BOOTSTRAP"


def consume_bootstrap():
    path = Path(os.environ.pop(BOOTSTRAP_ENV))
    try:
        with path.open("rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > c.BOOTSTRAP_MAX_BYTES:
                raise ValueError("invalid bootstrap")
            if os.name != "nt" and (info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600):
                raise ValueError("private bootstrap required")
            payload = stream.read(c.BOOTSTRAP_MAX_BYTES + 1)
        return c.WorkerBootstrap.model_validate_json(payload)
    finally:
        path.unlink(missing_ok=True)


def send_frame(channel, value, maximum):
    payload = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(payload) > maximum:
        raise ValueError("frame limit")
    channel.sendall(struct.pack("!I", len(payload)) + payload)


def dependencies_current(spec, workspace_root):
    """Reconcile observed local modules with the host's saved source superset."""
    root = Path(workspace_root).resolve()
    expected = {f.path: f.digest for f in spec.files if f.role == "source"}
    try:
        for module in tuple(sys.modules.values()):
            # No user __getattr__ or repr during reconciliation.
            namespace = vars(module) if type(module) is type(sys) else {}
            file = namespace.get("__file__")
            if not isinstance(file, str):
                continue
            candidate = Path(file).resolve()
            try:
                relative = candidate.relative_to(root).as_posix()
            except ValueError:
                continue
            # An excluded local environment is part of the separately probed
            # environment identity, not a claim of complete workspace coverage.
            if "site-packages" in candidate.parts:
                continue
            # The private cache prefix covers CPython's source-file loader only.
            # Do not certify sourceless, zip, native or custom workspace loaders
            # by hashing an unrelated .py file or silently ignoring their origin.
            if not file.endswith(".py") or type(namespace.get("__loader__")) is not importlib.machinery.SourceFileLoader:
                return False
            if relative not in expected:
                return False
            with candidate.open("rb") as stream:
                if sha256_stream(stream) != expected[relative]:
                    return False
        return True
    except (OSError, ValueError):
        return False


def main(script=None):
    try:
        bootstrap = consume_bootstrap()
    except BaseException:
        return 2
    from .source import SourceEvidence
    try:
        sources = SourceEvidence(bootstrap.spec, bootstrap.workspaceRoot)
    except Exception:
        sources = None  # Optional navigation evidence must not prevent execution.
    if isinstance(bootstrap.spec.profile, c.ScriptProfile):
        if script is None:
            return 2  # Script profiles require the namespace-isolated entrypoint.
        from .script_run import execute_script
        channel = socket.create_connection((bootstrap.resultChannel.host, bootstrap.resultChannel.port), timeout=5)
        send_frame(channel, dict(version=1, type="hello", runId=bootstrap.spec.identity.runId,
                                 token=bootstrap.resultChannel.token), c.WORKER_HELLO_MAX_BYTES)

        def publish(record):
            try:
                send_frame(channel, dict(version=1, type="record", record=record.model_dump(), dependenciesCurrent=dependencies_current(bootstrap.spec, bootstrap.workspaceRoot)),
                           min(c.WORKER_FRAME_MAX_BYTES, bootstrap.spec.profile.limits.max_record_bytes + 4096))
            finally:
                channel.close()

        try:
            execute_script(bootstrap.spec, script, publish, sources)
        finally:
            channel.close()
        return 0
    try:
        with socket.create_connection((bootstrap.resultChannel.host, bootstrap.resultChannel.port), timeout=5) as channel:
            send_frame(channel, dict(version=1, type="hello", runId=bootstrap.spec.identity.runId,
                                     token=bootstrap.resultChannel.token), c.WORKER_HELLO_MAX_BYTES)
            from .model_run import execute_model
            record = execute_model(bootstrap.spec, sources, workspace_root=bootstrap.workspaceRoot)
            send_frame(channel, dict(version=1, type="record", record=record.model_dump(), dependenciesCurrent=dependencies_current(bootstrap.spec, bootstrap.workspaceRoot)),
                       min(c.WORKER_FRAME_MAX_BYTES, bootstrap.spec.profile.limits.max_record_bytes + 4096))
            return 0 if record.process.state == "running" else 1
    except BaseException:
        # No project exception strings, bootstrap secrets or paths on either pipe.
        return 2


if __name__ == "__main__":
    sys.exit(main())
