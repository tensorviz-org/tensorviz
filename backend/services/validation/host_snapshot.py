"""Private host preflight adapter. Reads saved files, never imports project code.

Invoked with Python -I from the extension; stdin is host data, stdout is bounded
JSON. This is not an HTTP endpoint and cannot authorize execution.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 model environments.
    import tomli as tomllib

# -I deliberately omits both cwd and the script directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "services.validation"
from . import contracts as c
from .config import EnvironmentFact, HostFacts, PathFact, resolve

EXCLUDED = {".git", ".venv", "venv", "env", "node_modules", "__pycache__", ".tensorviz"}
MAX_SOURCE_BYTES = 32 * 1024 * 1024


def digest(data):
    return hashlib.sha256(data).hexdigest()


def environment_fact(executable, revision):
    packages = sorted((d.metadata.get("Name", ""), d.version) for d in importlib.metadata.distributions())
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = None
    return EnvironmentFact(executable=executable, resolved_executable=str(Path(sys.executable).resolve()),
        python_version=".".join(map(str, sys.version_info[:3])), torch_version=torch_version,
        selection_revision=revision, probe_digest=digest(json.dumps(packages, separators=(",", ":")).encode()))


def snapshot(request):
    if "replayPlan" in request:
        if set(request) - {"root", "documents", "replayPlan"}:
            raise ValueError("retained replay cannot substitute execution settings")
        from ..semantic.contracts import ResolvedExecutionPlan
        from ..semantic.candidate import verify_replay_base
        return verify_replay_base(ResolvedExecutionPlan.model_validate(request["replayPlan"]),
            request["root"], request.get("documents", {}))
    if "execution" in request and "captureRequest" in request:
        raise ValueError("choose a resolved invocation or automatic capture preparation")
    root = Path(request["root"]).resolve(strict=True)
    documents = request.get("documents", {})
    guards = {}
    budget = 0

    def fact(relative, role=None, content=False):
        nonlocal budget
        item = root / relative
        try:
            real = item.resolve(strict=True)
            canonical = real.relative_to(root).as_posix()
            info = real.stat()
            if real.is_dir():
                guards[relative] = [canonical, f"{info.st_dev}:{info.st_ino}"]
                return PathFact(kind="directory", canonical_path=canonical)
            if not real.is_file():
                raise ValueError("not regular")
            # Hash native data incrementally; never load checkpoints into memory.
            sha = hashlib.sha256()
            chunks = []
            with real.open("rb") as stream:
                while chunk := stream.read(65536):
                    sha.update(chunk)
                    if content:
                        budget += len(chunk)
                        if budget > MAX_SOURCE_BYTES:
                            raise ValueError("source snapshot limit")
                        chunks.append(chunk)
            after = real.stat()
            if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError("changed while reading")
            doc = documents.get(relative, {})
            file_id = f"{info.st_dev}:{info.st_ino}"
            guards[relative] = [canonical, file_id, sha.hexdigest(), doc.get("version"), bool(doc.get("dirty"))]
            return PathFact(kind="file", canonical_path=canonical, file_id=file_id,
                            digest=sha.hexdigest(), document_version=doc.get("version"),
                            saved=not doc.get("dirty", False), role=role,
                            content=b"".join(chunks) if content else None)
        except (OSError, ValueError):
            return PathFact(kind="missing", canonical_path=None, role=role)

    # Only lstat ENOENT establishes absence. A directory, unreadable file, escaped
    # or dangling symlink must not silently become a missing configuration.
    try:
        (root / "tensorviz.toml").lstat()
        config_present = True
    except FileNotFoundError:
        config_present = False
    config_fact = fact("tensorviz.toml", content=True)
    if config_present and config_fact.content is None:
        raise ValueError("configuration snapshot failed")
    config = config_fact.content
    parsed = None
    if config is not None and len(config) <= c.CONFIG_MAX_BYTES:
        try:
            parsed = c.ValidationConfig.model_validate(tomllib.loads(config.decode("utf-8")))
        except (ValueError, UnicodeError):
            pass
    profiles = list(parsed.profiles) if parsed else []
    paths = {".": fact("."), "tensorviz.toml": config_fact}
    selected = parsed.profiles.get(request.get("profile")) if parsed else None
    if selected:
        paths[selected.cwd] = fact(selected.cwd)
        for relative in selected.tracked_files:
            paths[relative] = fact(relative, "native")
        if isinstance(selected, c.ScriptProfile):
            paths[selected.script] = fact(selected.script, "source", True)
        elif selected.factory is not None:
            paths[selected.factory.file] = fact(selected.factory.file, "source", True)
        if isinstance(selected, c.ModelProfile) and isinstance(selected.inputs, c.InputFactory):
            paths[selected.inputs.factory.file] = fact(selected.inputs.factory.file, 'source', True)
    if "execution" in request or "captureRequest" in request:
        if any(doc.get("dirty") and (name == "tensorviz.toml" or name.endswith((".py", ".pyw")))
               for name, doc in documents.items()):
            raise ValueError("save configuration and Python source before resolving execution")
    if "execution" in request:
        from ..ingest.execution_spec import validate_execution_spec, validate_execution_job
        from ..semantic.execution_plan import execution_files
        execution = validate_execution_spec(request["execution"])
        job = validate_execution_job(request["job"]) if request.get("job") is not None else None
        cwd = execution["cwd"]
        paths[cwd] = fact(cwd)
        if paths[cwd].kind != "directory":
            raise ValueError("execution working directory is unavailable")
        for relative, role in execution_files(execution, job).items():
            if relative in paths and paths[relative].role not in (None, role):
                raise ValueError("competing execution file roles")
            paths[relative] = fact(relative, role, role == "source")
    # A conservative local source superset catches imports absent from the static
    # graph. Exclusions and dynamic/external imports make the scope explicitly partial.
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED and not (Path(directory) / d).is_symlink())
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            relative = (Path(directory) / name).relative_to(root).as_posix()
            if relative not in paths:
                paths[relative] = fact(relative, "source", True)
            if len(paths) > 100000:
                raise ValueError("file snapshot limit")
    environment = environment_fact(request["executable"], request["environmentRevision"])
    facts = HostFacts(workspace_session_id=request["session"], workspace_id=request["workspaceId"],
        run_id=request["runId"], generation=request["generation"], paths=paths,
        dependency_scope="partial", environment=environment)
    resolution = resolve(config, request.get("profile"), facts)
    if "captureRequest" in request:
        from ..semantic.execution_plan import resolve_preparation, applicability_digest
        preparation = resolve_preparation(request, facts, config, parsed)
        return {"preparation": preparation.model_dump(mode="json"),
                "applicabilityDigest": applicability_digest(preparation), "guards": guards}
    if "execution" in request:
        from ..semantic.execution_plan import resolve_plan, applicability_digest
        plan = resolve_plan(request, facts, config, resolution)
        return {"plan": plan.model_dump(mode="json"), "applicabilityDigest": applicability_digest(plan), "guards": guards}
    return {"profiles": profiles,
            "profileDefinitions": {name: profile.model_dump() for name, profile in parsed.profiles.items()} if parsed else {},
            "resolution": resolution.model_dump(), "guards": guards}


def main():
    if sys.argv[1:] == ["--serve"]:
        return serve()
    try:
        request = json.loads(sys.stdin.buffer.read(16 * 1024 * 1024 + 1))
        result = snapshot(request)
        print(json.dumps(result, separators=(",", ":")))
    except Exception:
        # Filesystem paths, source, package metadata and exception text stay private.
        print('{"error":"snapshot_failed"}')
        return 1
    return 0


MAX_REQUEST_LINE = 16 * 1024 * 1024


def serve():
    """Answer one snapshot per stdin line, one JSON line each, until stdin closes.

    The host keeps this process for a window so each snapshot skips interpreter
    start and the contract imports (most of a one-shot call's cost). Every request
    is still resolved from the files on disk: nothing is cached here, and this
    module never imports project code, so there is no state to go stale. Installed
    packages are re-listed per request (``environment_fact``) after dropping the
    import system's directory caches, so a ``pip install`` between two requests is
    seen. A failure answers that one request; the next is unaffected.
    """
    stdin, stdout = sys.stdin.buffer, sys.stdout
    while True:
        line = stdin.readline(MAX_REQUEST_LINE + 1)
        if not line:
            return 0
        try:
            if len(line) > MAX_REQUEST_LINE or not line.endswith(b"\n"):
                raise ValueError("snapshot request too large")
            importlib.invalidate_caches()
            answer = json.dumps(snapshot(json.loads(line)), separators=(",", ":"))
        except Exception:
            answer = '{"error":"snapshot_failed"}'
        stdout.write(answer + "\n")
        stdout.flush()


if __name__ == "__main__":
    sys.exit(main())
