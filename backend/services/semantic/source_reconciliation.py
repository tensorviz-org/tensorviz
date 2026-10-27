"""Nonexecuting round-trip checks over exact reviewed source and a later capture.

The host owns Apply receipts and live-source freshness. This service never reads
files, imports a model, runs Python, or promotes a candidate into the live graph.
"""
import hashlib

from .candidate import _apply_reviewed_patch
from .candidate_postconditions import prepare_postconditions, check_captured_postconditions
from .contracts import CaptureReconciliationRequest, CaptureReconciliation
from .source_edits import CapturedEditResolver, SourceEditRefused, MAX_FILE_BYTES


def reconcile_capture(payload):
    # Revalidate mutable model instances as well as JSON at this private boundary.
    if isinstance(payload, CaptureReconciliationRequest):
        payload = payload.model_dump(mode="json")
    request = CaptureReconciliationRequest.model_validate(payload)
    base, current = request.base, request.current
    reviewed = request.reviewedSource.encode("utf-8")
    if len(reviewed) > MAX_FILE_BYTES:
        raise SourceEditRefused("source_limit")
    intent = request.intent.model_dump(mode="json")
    resolver = CapturedEditResolver(base, request.sources)
    patch = resolver.plan(intent)
    if not patch["edits"] or patch["file"] != request.reviewedFile:
        raise SourceEditRefused("reviewed_edit_mismatch")
    _apply_reviewed_patch(request.sources[request.reviewedFile], patch, request.reviewedSource)
    reviewed_digest = hashlib.sha256(reviewed).hexdigest()
    fields = dict(version=1, baseCaptureId=base.id, captureId=current.id, sourceId=current.source.id,
        reviewedDigest=reviewed_digest,
        dependencyScope="complete" if base.source.dependencyScope == current.source.dependencyScope == "complete" else "partial",
        status="unavailable", inputs="unavailable", state="unavailable", postconditions=None)

    def unavailable(reason):
        return CaptureReconciliation(**fields, reason=reason)

    before, after = base.source, current.source
    if any(getattr(before, key) != getattr(after, key) for key in
            ("workspaceSessionId", "workspaceId", "modelId", "modelFile", "className")):
        return unavailable("different_model")
    if current.id == base.id or current.attemptId == base.attemptId:
        return unavailable("capture_not_new")
    if before.sourceMode != "saved" or after.sourceMode != "saved":
        return unavailable("source_unavailable")
    files = {file.path: file for file in after.files}
    edited = files.get(request.reviewedFile)
    if edited is None or edited.role != "source":
        return unavailable("source_unavailable")
    if edited.digest != reviewed_digest:
        return unavailable("source_reverted" if edited.digest == patch["expected_digest"] else "source_changed")
    # Source IDs, document versions and generations belong to each run. Exact
    # path/role/byte identities establish unchanged tracked dependencies across
    # runs. Incomplete inventories remain explicit, never a whole-program claim.
    expected = {file.path: (file.role, reviewed_digest if file.path == request.reviewedFile else file.digest) for file in before.files}
    if ({path: (file.role, file.digest) for path, file in files.items()} != expected
            or after.configDigest != before.configDigest):
        return unavailable("dependencies_changed")
    if before.environmentDigest != after.environmentDigest:
        return unavailable("environment_changed")
    proof = prepare_postconditions(after, request.reviewedFile, patch, intent, request.reviewedSource)
    postconditions = check_captured_postconditions(current, proof)
    def identity(left, right):
        return "unavailable" if left is None or right is None else "same" if left == right else "different"
    return CaptureReconciliation(**(fields | dict(status="checked",
        inputs=identity(base.inputDigest, current.inputDigest), state=identity(base.stateDigest, current.stateDigest),
        postconditions=postconditions)), reason="reviewed_source")
