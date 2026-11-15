"""Resolve shared worker plans from saved host facts; never import workspace code."""
from __future__ import annotations

import hashlib
from ..python_compat import sha256_stream
import json
import os
import stat
from pathlib import Path

from ..ingest.execution_spec import validate_execution_spec, validate_execution_job
from ..validation import contracts as v
from .contracts import CaptureLimits, CapturePreparation, ResolvedExecutionPlan, SourceSnapshot


def fingerprint(value) -> str:
    """Python owns this encoding; clients retain the returned digest verbatim."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def profile_execution(profile: v.ModelProfile) -> dict:
    """Lossless lowering to the existing explicit execution grammar."""
    inputs = profile.inputs.model_dump()
    construction = {"setup": profile.factory.model_dump()} if profile.factory is not None else {
        "constructorBinding": profile.init.model_dump()}
    return validate_execution_spec(dict(inputs=inputs if isinstance(profile.inputs, v.InputFactory) else {"binding": inputs}, **construction,
        **({'framework': 'jax'} if profile.framework == 'jax' else {}),
        device=profile.device, mode=profile.mode, seed=profile.seed,
        gradEnabled=profile.grad_enabled, deterministic=profile.deterministic, cwd=profile.cwd))


def execution_files(execution: dict, job: dict | None) -> dict[str, str]:
    if job is not None:
        raise ValueError("Diagnostic profiling and training-step jobs are not supported.")
    files = {}
    def add(file, role):
        if file in files and files[file] != role:
            raise ValueError("one execution file cannot have competing roles")
        files[file] = role
    if "setup" in execution:
        add(execution["setup"]["file"], "source")
    if "fixture" in execution["inputs"]:
        add(execution["inputs"]["fixture"], "native")
    if 'factory' in execution['inputs']:
        add(execution['inputs']['factory']['file'], 'source')
    if "checkpoint" in execution:
        add(execution["checkpoint"]["file"], "native")
    return files


def source_snapshot(request, facts, config):
    files = []
    physical = set()
    for name, fact in sorted(facts.paths.items()):
        role = "config" if name == "tensorviz.toml" else fact.role
        if role is None or name == "tensorviz.toml" and config is None:
            continue
        if fact.kind != "file" or not fact.saved or fact.digest is None or fact.file_id is None:
            raise ValueError("execution needs readable saved file identities")
        if fact.file_id in physical:
            raise ValueError("execution source aliases are ambiguous")
        physical.add(fact.file_id)
        files.append(v.FileIdentity(path=name, digest=fact.digest, documentVersion=fact.document_version, role=role))
    environment_digest = fingerprint(facts.environment.model_dump(mode="json"))
    return SourceSnapshot(version=1, id=f"source:{request['runId']}",
        workspaceSessionId=facts.workspace_session_id, workspaceId=facts.workspace_id,
        modelId=fingerprint([facts.workspace_id, request["modelFile"], request["className"]]),
        modelFile=request["modelFile"], className=request["className"], generation=facts.generation,
        files=files, dependencyScope=facts.dependency_scope,
        configDigest=hashlib.sha256(config).hexdigest() if config is not None else None,
        environmentDigest=environment_digest, sourceMode="saved")


def default_capture_limits():
    return CaptureLimits(operations=8192, invocations=4096, values=10000, bytes=v.CAPTURE_MAX_BYTES)


def resolve_plan(request: dict, facts, config: bytes | None, resolution) -> ResolvedExecutionPlan:
    execution = validate_execution_spec(request["execution"])
    job = validate_execution_job(request["job"]) if request.get("job") is not None else None
    selected = request.get("profile") is not None
    spec = None
    if selected:
        if not isinstance(resolution, v.Resolved) or not isinstance(resolution.spec.profile, v.ModelProfile):
            raise ValueError("the selected model profile did not resolve")
        spec = resolution.spec
        module, cls = spec.profile.model.split(":", 1)
        base = ("" if spec.profile.cwd == "." else spec.profile.cwd + "/") + module.replace(".", "/")
        if request["className"] != cls or request["modelFile"] not in (base + ".py", base + "/__init__.py"):
            raise ValueError("execution target differs from the selected profile")
        if execution != profile_execution(spec.profile):
            raise ValueError("execution differs from the selected saved profile")
        # Compare Python-owned applicability, never per-preflight request IDs.
        expected = request.get("expectedIdentity")
        if expected is None or any(expected.get(key) != getattr(spec.identity, key) for key in (
                "workspaceSessionId", "workspaceId", "profile", "sourceDigest",
                "configDigest", "profileDigest", "environmentDigest")):
            raise ValueError("saved profile identity changed before plan resolution")
    source = source_snapshot(request, facts, config)
    # Profile limits are never silently replaced with larger defaults.
    limits = spec.profile.limits if spec else None
    capture = CaptureLimits(operations=limits.max_events, invocations=limits.max_events,
        values=limits.max_tensors, bytes=limits.max_capture_bytes or limits.max_record_bytes) if limits else default_capture_limits()
    return ResolvedExecutionPlan(version=1, id=request["runId"], source=source,
        environment=facts.environment, interpreter=v.InterpreterIdentity(executable=facts.environment.executable,
            version=facts.environment.python_version, environmentDigest=source.environmentDigest),
        origin="profile" if selected else "explicit",
        configuration="selected" if selected else "absent" if config is None else "overridden",
        profileName=spec.identity.profile if spec else None, profileDigest=spec.identity.profileDigest if spec else None,
        execution=execution, job=job, timeoutMs=limits.timeout_ms if limits else request["timeoutMs"],
        captureLimits=capture)


def resolve_preparation(request, facts, config, parsed):
    if any(key in request for key in ("execution", "profile", "expectedIdentity", "job")):
        raise ValueError("automatic preparation cannot replace an explicit invocation")
    if config is not None:
        if parsed is None:
            raise ValueError("invalid configuration cannot become an automatic guess")
        for profile in parsed.profiles.values():
            symbol = profile.model if isinstance(profile, v.ModelProfile) else profile.capture.model
            module, cls = symbol.split(":", 1)
            base = ("" if profile.cwd == "." else profile.cwd + "/") + module.replace(".", "/")
            paths = [base + ".py", base + "/__init__.py"]
            if isinstance(profile, v.ScriptProfile) and module == "__main__":
                paths.append(profile.script)
            if cls == request["className"] and request["modelFile"] in paths:
                raise ValueError("a matching saved profile requires explicit selection")
    source = source_snapshot(request, facts, config)
    return CapturePreparation(version=1, id=request["runId"], source=source,
        interpreter=v.InterpreterIdentity(executable=facts.environment.executable,
            version=facts.environment.python_version, environmentDigest=source.environmentDigest),
        environment=facts.environment, configuration="absent" if config is None else "no_match",
        policy="legacy-input-search-v1",
        request=request["captureRequest"], timeoutMs=request["timeoutMs"], captureLimits=default_capture_limits())


def applicability_digest(plan: ResolvedExecutionPlan | CapturePreparation) -> str:
    data = plan.model_dump(mode="json")
    data.pop("id")
    for key in ("id", "generation", "workspaceSessionId"):
        data["source"].pop(key)
    for file in data["source"]["files"]:
        file.pop("documentVersion")
    return fingerprint(data)


def verify_saved_plan(plan: ResolvedExecutionPlan | CapturePreparation, workspace: str) -> None:
    """Last worker check before workspace imports; hashes are independent of mtimes."""
    from ..ingest.execution_spec import workspace_file, workspace_directory
    from ..validation.host_snapshot import environment_fact
    root = Path(workspace).resolve(strict=True)
    workspace_directory(workspace, plan.execution["cwd"] if isinstance(plan, ResolvedExecutionPlan) else ".")
    if environment_fact(plan.interpreter.executable, plan.environment.selection_revision) != plan.environment:
        raise ValueError("execution environment changed after plan resolution")
    for file in plan.source.files:
        location = workspace_file(workspace, file.path)
        with location.open("rb") as stream:
            if sha256_stream(stream) != file.digest:
                raise ValueError("saved execution file changed after plan resolution")
    if plan.configuration == "absent":
        try:
            (root / "tensorviz.toml").lstat()
        except FileNotFoundError:
            pass
        else:
            raise ValueError("configuration appeared after plan resolution")


def consume_private_json(environment_key, limit=v.BOOTSTRAP_MAX_BYTES):
    """Shared private host transport. The host owns unconditional file cleanup."""
    path = Path(os.environ.pop(environment_key))
    if path.is_symlink():
        raise ValueError("private execution plan required")
    with path.open("rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("execution plan exceeds its file budget")
        if os.name != "nt" and (info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600):
            raise ValueError("private execution plan required")
        data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError("execution plan exceeds its file budget")
        raw = json.loads(data)
    return path, raw


def _consume_execution_file(key, contract):
    """Read the private host envelope. The host owns unconditional file cleanup."""
    _, raw = consume_private_json("TENSORVIZ_EXECUTION_PLAN")
    if type(raw) is not dict or set(raw) != {"version", key, "applicabilityDigest"} or type(raw["version"]) is not int or raw["version"] != 1:
        raise ValueError("invalid execution plan envelope")
    plan = contract.model_validate(raw[key])
    if applicability_digest(plan) != raw["applicabilityDigest"]:
        raise ValueError("execution plan fingerprint mismatch")
    return plan, raw["applicabilityDigest"]


def consume_plan_file():
    return _consume_execution_file("plan", ResolvedExecutionPlan)


def consume_preparation_file():
    return _consume_execution_file("preparation", CapturePreparation)
