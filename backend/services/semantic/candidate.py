"""Prepare exact reviewed edits for the existing resolved-plan worker.

This host-only helper never imports workspace code. A copied workspace separates
source writes; it is not an OS sandbox for the explicitly authorized Python run.
The caller owns worker cancellation and must stop it before leaving the context.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from typing import Callable
from uuid import uuid4

from ..validation.host_snapshot import snapshot
from .contracts import (CandidateExecution, CaptureSnapshot, ResolvedExecutionPlan,
    ReviewedPythonSource, SourceCandidateRequest, SourceCandidateExecution)
from .execution_plan import applicability_digest, fingerprint, verify_saved_plan
from .source_edits import CapturedEditResolver, MAX_FILES, MAX_BYTES, MAX_FILE_BYTES
from .candidate_postconditions import (LiteralPostcondition, ActivationPostcondition, OperandPostcondition,
    DeletionPostcondition, ConstructorPostcondition, ConfigurationPostcondition, prepare_postconditions)

# Includes one maximum-size checkpoint, one fixture, and the saved source budget.
# All files count, including additional profile-tracked resources.
MAX_COPY_BYTES = (2048 + 64 + 32) * 1024 * 1024
MAX_COPY_FILES = 100000


class CandidateRefused(ValueError):
    """Fixed reasons only; source and arbitrary exception text stay private."""


class CandidateCancelled(CandidateRefused):
    pass


def verify_replay_base(plan: ResolvedExecutionPlan, workspace: str, documents: dict | None = None) -> dict:
    """Reuse preflight to catch added sources as well as changed saved files.

    The host supplies current buffer state; dirty sources/configuration refuse.
    Version continuity remains a host guard. This check runs in the selected
    interpreter, just like the original saved-plan preflight.
    """
    request = dict(root=workspace, executable=plan.interpreter.executable,
        environmentRevision=plan.environment.selection_revision,
        session=plan.source.workspaceSessionId, workspaceId=plan.source.workspaceId,
        runId=plan.id, generation=plan.source.generation,
        modelFile=plan.source.modelFile, className=plan.source.className,
        timeoutMs=plan.timeoutMs, execution=plan.execution, documents=documents or {})
    if plan.job is not None:
        request["job"] = plan.job
    try:
        if plan.origin == "profile":
            preflight = {key: value for key, value in request.items() if key not in ("execution", "job")}
            preflight["profile"] = plan.profileName
            resolution = snapshot(preflight)["resolution"]
            request.update(profile=plan.profileName, expectedIdentity=resolution["spec"]["identity"])
        result = snapshot(request)
        current = ResolvedExecutionPlan.model_validate(result["plan"])
        if applicability_digest(current) != applicability_digest(plan):
            raise ValueError("changed")
        return result
    except Exception:
        raise CandidateRefused("candidate_base_changed") from None


def _apply_reviewed_patch(before: str, patch: dict, reviewed: str) -> bytes:
    """Apply only resolver-owned UTF-16 ranges, then match the reviewed artifact."""
    after = before
    for edit in sorted(patch["edits"], key=lambda item: item["range"]["start"], reverse=True):
        start, end = edit["range"]["start"], edit["range"]["end"]
        if start[0] != end[0]:
            raise CandidateRefused("candidate_patch_unavailable")
        lines = after.split("\n")
        line = lines[start[0] - 1].encode("utf-16-le")
        lines[start[0] - 1] = (line[:start[1] * 2] + edit["newText"].encode("utf-16-le")
            + line[end[1] * 2:]).decode("utf-16-le")
        after = "\n".join(lines)
    data = after.encode("utf-8")
    if after != reviewed or hashlib.sha256(data).hexdigest() != patch["source_digest"]:
        raise CandidateRefused("candidate_review_mismatch")
    return data


def _regular_source(root: Path, relative: str) -> Path:
    location = root
    # Contract paths are normalized already. Reject internal aliases too: copying
    # them as ordinary files could change __file__/relative import semantics.
    for part in relative.split("/"):
        location = location / part
        if location.is_symlink():
            raise CandidateRefused("candidate_symlink_dependency")
    if not stat.S_ISREG(location.stat().st_mode):
        raise CandidateRefused("candidate_nonregular_dependency")
    return location


@dataclass(frozen=True)
class StagedCandidate:
    workspace: Path
    execution: CandidateExecution | SourceCandidateExecution
    postconditions: LiteralPostcondition | ActivationPostcondition | OperandPostcondition | DeletionPostcondition | ConstructorPostcondition | ConfigurationPostcondition | None


@contextmanager
def stage_candidate(workspace: str, plan: ResolvedExecutionPlan, capture: CaptureSnapshot,
                    intent: dict | None = None, reviewed_file: str | None = None, reviewed_source: str | None = None, *,
                    reviewed_sources: list[ReviewedPythonSource] | None = None,
                    cancelled: Callable[[], bool] = lambda: False, temporary_parent: str | None = None,
                    candidate_id: str | None = None):
    """Yield an independent saved candidate and remove it on every exit.

The exact retained plan/capture and reviewed sources come from the host, either
with a closed recipe or as a bounded source set. Staging is static; running the
returned plan requires separate explicit execution authority.
"""
    def check_cancelled():
        if cancelled():
            raise CandidateCancelled("candidate_cancelled")

    check_cancelled()
    # Revalidate mutable Pydantic instances before using their paths or identities.
    if not isinstance(plan, ResolvedExecutionPlan):
        raise CandidateRefused("candidate_requires_resolved_plan")
    plan = ResolvedExecutionPlan.model_validate(plan.model_dump(mode="json"))
    capture = CaptureSnapshot.model_validate(capture.model_dump(mode="json"))
    identifier = candidate_id if candidate_id is not None else f"candidate:{uuid4()}"
    if reviewed_sources is not None:
        if intent is not None or reviewed_file is not None or reviewed_source is not None:
            raise CandidateRefused("candidate_competing_edits")
        # Revalidate mutable records, including UTF-8 and aggregate byte limits.
        request = SourceCandidateRequest(version=2, id=identifier, plan=plan, capture=capture,
            reviewedSources=[item.model_dump(mode="json") if isinstance(item, ReviewedPythonSource) else item
                for item in reviewed_sources])
        reviewed_sources = request.reviewedSources
    elif (not isinstance(intent, dict) or not isinstance(reviewed_file, str)
            or not isinstance(reviewed_source, str) or len(reviewed_source.encode("utf-8")) > MAX_FILE_BYTES):
        raise CandidateRefused("candidate_source_limit")
    base_digest = applicability_digest(plan)
    if capture.planDigest != base_digest or capture.source != plan.source or capture.attemptId != plan.id:
        raise CandidateRefused("candidate_capture_plan_mismatch")
    if len(plan.source.files) > MAX_COPY_FILES:
        raise CandidateRefused("candidate_copy_limit")
    if sum(file.role == "source" for file in plan.source.files) > MAX_FILES:
        raise CandidateRefused("candidate_source_limit")
    verify_replay_base(plan, workspace)
    check_cancelled()
    root = Path(workspace).resolve(strict=True)
    # Source is held only within the existing edit resolver's limits. Native
    # fixtures/checkpoints are copied and hashed incrementally, never deserialized.
    sources = {}
    source_bytes = 0
    for file in plan.source.files:
        check_cancelled()
        if file.role != "source":
            continue
        location = _regular_source(root, file.path)
        with location.open("rb") as stream:
            data = stream.read(MAX_FILE_BYTES + 1)
        source_bytes += len(data)
        if len(data) > MAX_FILE_BYTES or source_bytes > MAX_BYTES:
            raise CandidateRefused("candidate_source_limit")
        if hashlib.sha256(data).hexdigest() != file.digest:
            raise CandidateRefused("candidate_base_changed")
        sources[file.path] = data.decode("utf-8")
    patch = None
    if reviewed_sources is None:
        patch = CapturedEditResolver(capture, sources).plan(intent)
        if not patch["edits"]:
            raise CandidateRefused("candidate_unchanged")
        if patch["file"] != reviewed_file:
            raise CandidateRefused("candidate_review_mismatch")
        replacements = {patch["file"]: _apply_reviewed_patch(sources[patch["file"]], patch, reviewed_source)}
        changes = [dict(file=patch["file"], beforeDigest=patch["expected_digest"], afterDigest=patch["source_digest"])]
    else:
        replacements, changes = {}, []
        tracked = {file.path: file for file in plan.source.files if file.role == "source"}
        for item in sorted(reviewed_sources, key=lambda item: item.file):
            if item.file not in tracked or item.beforeDigest != tracked[item.file].digest:
                raise CandidateRefused("candidate_review_mismatch")
            proposed = item.source.encode("utf-8")
            after_digest = hashlib.sha256(proposed).hexdigest()
            if after_digest == item.beforeDigest:
                raise CandidateRefused("candidate_unchanged")
            try:
                # Compilation performs syntax checks only; no import or execution.
                compile(proposed, item.file, "exec", dont_inherit=True)
            except Exception:
                raise CandidateRefused("candidate_syntax_invalid") from None
            replacements[item.file] = proposed
            changes.append(dict(file=item.file, beforeDigest=item.beforeDigest, afterDigest=after_digest))
    check_cancelled()
    with tempfile.TemporaryDirectory(prefix="tensorviz-candidate-", dir=temporary_parent) as temporary:
        directory = Path(temporary).resolve(strict=True)
        if os.name == "nt":
            # Same current-user-only ACL as the host's execution-plan transport.
            subprocess.run(["icacls", str(directory), "/inheritance:r", "/grant:r",
                f"{os.getlogin()}:(OI)(CI)F"], check=True, capture_output=True)
        else:
            directory.chmod(0o700)
        candidate_root = directory / "workspace"
        candidate_root.mkdir(mode=0o700)
        copied = 0
        physical = set()
        for file in plan.source.files:
            check_cancelled()
            location = _regular_source(root, file.path)
            destination = candidate_root / file.path
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            sha = hashlib.sha256()
            with location.open("rb") as source, destination.open("xb") as output:
                info = os.fstat(source.fileno())
                identity = (info.st_dev, info.st_ino)
                if not stat.S_ISREG(info.st_mode) or identity in physical:
                    raise CandidateRefused("candidate_aliased_dependency")
                physical.add(identity)
                if copied + info.st_size > MAX_COPY_BYTES:
                    raise CandidateRefused("candidate_copy_limit")
                while chunk := source.read(65536):
                    check_cancelled()
                    copied += len(chunk)
                    if copied > MAX_COPY_BYTES:
                        raise CandidateRefused("candidate_copy_limit")
                    sha.update(chunk)
                    output.write(chunk)
                final = os.fstat(source.fileno())
                if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
                        final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns) or sha.hexdigest() != file.digest:
                    raise CandidateRefused("candidate_base_changed")
            destination.chmod(0o600)
        # cwd may be an empty tracked directory. Never recreate symlinked cwd.
        cwd = plan.execution["cwd"]
        if cwd != ".":
            current = root
            for part in cwd.split("/"):
                current = current / part
                if current.is_symlink():
                    raise CandidateRefused("candidate_symlink_dependency")
            (candidate_root / cwd).mkdir(mode=0o700, parents=True, exist_ok=True)
        for file, proposed in replacements.items():
            (candidate_root / file).write_bytes(proposed)
        data = plan.model_dump(mode="json")
        data["id"] = identifier
        source = data["source"]
        source.update(id=f"source:{identifier}", workspaceId=identifier, generation=0,
            modelId=fingerprint([identifier, source["modelFile"], source["className"]]))
        digests = {change["file"]: change["afterDigest"] for change in changes}
        for file in source["files"]:
            file["documentVersion"] = None
            if file["path"] in digests:
                file["digest"] = digests[file["path"]]
        provenance = dict(id=identifier, baseCaptureId=capture.id,
            baseAttemptId=plan.id, baseSourceId=capture.source.id, baseWorkspaceId=capture.source.workspaceId,
            basePlanDigest=base_digest,
            plan=ResolvedExecutionPlan.model_validate(data))
        execution = (CandidateExecution(version=1, **provenance, intentDigest=fingerprint(intent), **changes[0])
            if patch is not None else SourceCandidateExecution(version=2, **provenance, changes=changes))
        # Copy races, new imports, configuration and package changes cannot turn a
        # stale base into a current proposal. The worker rechecks the staged plan.
        verify_replay_base(plan, workspace)
        verify_saved_plan(execution.plan, str(candidate_root))
        check_cancelled()
        yield StagedCandidate(candidate_root, execution, prepare_postconditions(
            execution.plan.source, patch["file"], patch, intent, reviewed_source) if patch is not None else None)
