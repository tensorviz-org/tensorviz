"""Replay approved canonical captures in hash-verified disposable model copies."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from scripts.model_corpus import BACKEND, MANIFEST, RECIPES, FACTORY, verified_copy


def project_source(workspace, recipe, envelope, capture, store_directory, document_directory=None):
    """Exercise the real multi-file API, with no writes to the user's cache."""
    from collections import Counter
    database = str(store_directory / "projection.sqlite3")
    previous_database = os.environ.get("USER_HISTORY_DB_PATH")
    os.environ["USER_HISTORY_DB_PATH"] = database
    try:
        from api.routers import ide
        from auth import CurrentUser
        from services.ide.multifile import MultifileProjectionRequest
        from services.store_factory import get_project_store
        from services.semantic.source_join import CaptureJoin
        from services.canonical_validation import CanonicalValidationService
    finally:
        if previous_database is None:
            os.environ.pop("USER_HISTORY_DB_PATH", None)
        else:
            os.environ["USER_HISTORY_DB_PATH"] = previous_database
    ide._store = get_project_store(database)
    sources = []
    for file in capture.source.files:
        if file.role == "source":
            text = (workspace / file.path).read_bytes().decode("utf-8")
            sources.append(dict(file=file.path, source=text, digest=hashlib.sha256(text.encode()).hexdigest()))
    digests = {entry["file"]: entry["digest"] for entry in sources}
    request = MultifileProjectionRequest(protocol_version=1,
        root=dict(file=recipe["file"], cls=recipe["class"], document_version=1,
                  source_digest=digests[recipe["file"]]), sources=sources)
    user = CurrentUser(uid="semantic-corpus")
    static = ide.multifile_project(request, user)
    request.trace_result = envelope
    projected = ide.multifile_project(request, user)
    if document_directory is not None:
        document_directory.mkdir(parents=True, exist_ok=True)
        (document_directory / (recipe["id"] + ".json")).write_text(json.dumps(projected["document"]))
    nodes = projected["document"]["semantic"]["nodes"]
    metadata = projected["document"]["semantic"]["metadata"]
    by_id = {fact["nodeId"]: fact for fact in metadata["sourceCorrespondence"]}
    joined = CaptureJoin(capture, model_file=recipe["file"], class_name=recipe["class"], source_digests=digests)
    expected = {}
    for node in static["document"]["semantic"]["nodes"]:
        fact = joined.correspondence(dict(id=node["node_id"], data=dict(
            parameters=node["params"], layerType=dict(id=node["op"])) ))
        if fact.status in ("observed", "failed"):
            expected[node["node_id"]] = fact.status
    mismatches = [id for id, status in expected.items()
        if by_id.get(id, {}).get("status") != status]
    from models.architecture_models import ArchitectureGraph
    validation = CanonicalValidationService().validate(ArchitectureGraph.model_validate(projected["document"]["semantic"]))
    false_unreached = [id for id, result in validation.node_results.items() if "run failed before" in (result.message or "")]
    coverage = Counter(fact["status"] for fact in by_id.values())
    operations = {node["metadata"]["captureRef"]["semanticId"] for node in nodes if node["metadata"]["captureRef"]["kind"] == "operation"}
    calls = {node["metadata"]["captureRef"]["semanticId"] for node in nodes if node["metadata"]["captureRef"]["kind"] == "invocation"}
    faithful = operations == {op.id for op in capture.operations} and calls == {call.id for call in capture.invocations}
    error_ids = {node["metadata"]["captureRef"]["semanticId"] for node in nodes if node["metadata"].get("error")}
    failed = {op.id for op in capture.operations if op.status == "failed"} | {call.id for call in capture.invocations if call.status == "failed"}
    by_call, by_operation = {call.id: call for call in capture.invocations}, {op.id: op for op in capture.operations}
    values = {value.id: value for value in capture.values}
    metric_mismatches = []
    for node in nodes:
        meta = node["metadata"]
        ref = meta["captureRef"]
        record = by_call.get(ref["semanticId"]) if ref["kind"] == "invocation" else by_operation.get(ref["semanticId"]) if ref["kind"] == "operation" else None
        if record is not None:
            value = values[record.output.id] if record.output is not None and record.output.kind == "value" else None
            expected_tensor = dict(shape=value.shape, dtype=value.dtype.removeprefix("torch."), logicalBytes=value.logicalBytes) if value is not None and value.logicalBytes is not None else None
            if meta.get("outputTensor") != expected_tensor:
                metric_mismatches.append(node["node_id"])
        if ref["kind"] == "invocation" and meta.get("parameterCount") != record.parameterCount:
            metric_mismatches.append(node["node_id"])
    return dict(nodes=len(nodes), edges=len(projected["document"]["semantic"]["edges"]),
        editCapabilities=dict(Counter(capability["recipe"] for node in nodes
            for capability in node["metadata"].get("sourceEditCapabilities", []))),
        calledSourceCards=len(expected), coverage=dict(coverage), missingOrWrongCoverage=sorted(mismatches),
        falseUnreached=sorted(false_unreached), operationNodes=len(operations), invocationNodes=len(calls),
        metricMismatches=metric_mismatches,
        measuredParameterNodes=sum("parameterCount" in node["metadata"] for node in nodes),
        measuredOutputNodes=sum("outputTensor" in node["metadata"] for node in nodes),
        documentBytes=len(json.dumps(projected["document"], separators=(",", ":")).encode()),
        matched=not mismatches and not metric_mismatches and not false_unreached and faithful and error_ids == failed and not validation.global_errors)


def project_semantic(capture):
    """Check full and collapsed views against the retained capture, not layout rank."""
    from services.semantic.projection import project_capture
    from services.semantic.contracts import ValueRef, walk_argument, walk_arguments
    full = project_capture(capture)
    collapsed = project_capture(capture, collapsed_invocations=[call.id for call in capture.invocations
                                                              if call.id != capture.rootInvocationId])
    operations = {node.semanticId for node in full.nodes if node.kind == "operation"}
    calls = {node.semanticId for node in full.nodes if node.kind == "invocation"}
    uses = sum(sum(isinstance(arg, ValueRef) for _, arg in walk_arguments(op.arguments))
               + len(op.effects) for op in capture.operations)
    results = 0 if capture.output is None else sum(isinstance(arg, ValueRef) for _, arg in walk_argument(capture.output))
    original = [(edge.id, edge.valueId, edge.sourcePortId, edge.targetPortId) for edge in full.edges]
    grouped = [(edge.id, edge.valueId, edge.sourcePortId, edge.targetPortId) for edge in collapsed.edges]
    inputs = {node.id for node in full.nodes if node.kind == "input"}
    return dict(operationNodes=len(operations), invocationNodes=len(calls), connections=len(full.edges),
        inputPorts=sum(port.nodeId in inputs for port in full.ports),
        collapsedVisibleNodes=sum(node.visible for node in collapsed.nodes),
        collapsedBoundaryPorts=sum(port.memberPortId is not None for port in collapsed.ports),
        restoredExact=project_capture(capture) == full,
        matched=(operations == {op.id for op in capture.operations} and calls == {call.id for call in capture.invocations}
                 and original == grouped and len(full.edges) == uses + results and project_capture(capture) == full))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--project-source", action="store_true", help="also verify source/capture correspondence through the multi-file API")
    parser.add_argument("--project-capture", action="store_true", help="also verify reversible semantic projections")
    parser.add_argument("--portable-capture", action="store_true", help="also verify detached portable computation payloads")
    parser.add_argument("--document-directory", type=Path, help="retain API rendering documents for browser QA (requires --project-source)")
    options = parser.parse_args()
    from services.validation.host_snapshot import snapshot
    from services.semantic.execution_plan import applicability_digest
    from services.semantic.contracts import ResolvedExecutionPlan, CaptureSnapshot

    recipes = json.loads(RECIPES.read_text())["recipes"]
    chosen = [recipe for recipe in recipes if recipe["id"] in options.case] if options.case else [
        recipe for recipe in recipes if "outputShape" in recipe]
    if not chosen or (options.case and {r["id"] for r in chosen} != set(options.case)):
        parser.error("choose known corpus cases")
    reference = {row["case"]: row for row in json.loads(
        (BACKEND / "tests/fixtures/four-model-reference.json").read_text())["results"]}
    results = []
    with tempfile.TemporaryDirectory(prefix="tensorviz-semantic-corpus-") as directory:
        temp = Path(directory)
        counts = {repo: verified_copy(options.models, repo, temp / repo)
                  for repo in sorted({recipe["repository"] for recipe in chosen})}
        if (temp / "whisper-main").exists():
            (temp / "whisper-main" / "audit_setup.py").write_bytes(FACTORY.read_bytes())
        for recipe in chosen:
            workspace = temp / recipe["repository"]
            plan = snapshot(dict(root=str(workspace), executable=sys.executable, environmentRevision=0,
                session="semantic-replay", workspaceId=recipe["repository"], runId=recipe["id"], generation=1,
                modelFile=recipe["file"], className=recipe["class"], timeoutMs=180000,
                execution=recipe["execution"]))["plan"]
            plan_file = temp / f"plan-{recipe['id']}.json"
            with os.fdopen(os.open(plan_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "w") as handle:
                json.dump(dict(version=1, plan=plan,
                    applicabilityDigest=applicability_digest(ResolvedExecutionPlan.model_validate(plan))), handle)
            started = time.monotonic()
            worker = subprocess.run([sys.executable, "-B", str(BACKEND / "services/ingest/bootstrap.py"),
                "--file", str(workspace / recipe["file"]), "--cls", recipe["class"], "--workspace", str(workspace),
                "--resolved-execution", "--tier", "real", "--json"], cwd=workspace,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "TENSORVIZ_EXECUTION_PLAN": str(plan_file)},
                capture_output=True, text=True, timeout=180)
            envelope = json.loads(worker.stdout)
            diagnostics = (envelope.get("result") or {}).get("diagnostics", {})
            payload = diagnostics.get("semantic_capture")
            row = dict(case=recipe["id"], pinnedFiles=counts[recipe["repository"]], exitCode=worker.returncode,
                workerWallSeconds=round(time.monotonic() - started, 3),
                captureError=diagnostics.get("semantic_capture_error"), hasCapture=payload is not None)
            if payload is not None:
                capture = CaptureSnapshot.model_validate(payload)
                values = {value.id: value for value in capture.values}
                shape = values[capture.output.id].shape if capture.output is not None and capture.output.kind == "value" else None
                observed = sorted(entry.modulePath for entry in capture.coverage.modules if entry.invocationIds)
                expected_paths = reference[recipe["id"]]["calledModules"]
                row.update(status=capture.status, omissions=capture.coverage.omissions,
                    operations=len(capture.operations), operationsSeen=capture.coverage.operationsSeen,
                    invocations=len(capture.invocations), invocationsSeen=capture.coverage.invocationsSeen,
                    sourceBindings=dict(total=len(capture.sourceBindings),
                        boundConstructors=sum(bool(item.constructorBindingIds) for item in capture.modules),
                        initializerArguments={status: sum(
                            (item.initializerArguments.status if item.initializerArguments else "absent") == status
                            for item in capture.modules) for status in ("observed", "unavailable", "absent")},
                        boundDefinitions=sum(bool(item.sourceBindingIds) for item in capture.definitions),
                        boundInvocations=sum(bool(item.sourceBindingIds) for item in capture.invocations),
                        boundOperations=sum(bool(item.sourceBindingIds) for item in capture.operations)),
                    values=len(capture.values), serializedBytes=len(capture.model_dump_json().encode()),
                    byteLimit=capture.limits.bytes, outputShape=shape,
                    rootInputs=[dict(shape=values[arg.id].shape, dtype=values[arg.id].dtype)
                                for arg in capture.inputs.args if arg.kind == "value"],
                    failureDiagnostics=[dict(code=d.code, field=d.field, expectedShape=d.expectedShape,
                        actualShape=d.actualShape, expectedDtype=d.expectedDtype, actualDtype=d.actualDtype)
                        for d in capture.diagnostics],
                    missingObservedPaths=sorted(set(expected_paths) - set(observed)),
                    unexpectedObservedPaths=sorted(set(observed) - set(expected_paths)))
                row["matched"] = (not row["missingObservedPaths"] and not row["unexpectedObservedPaths"]
                    and (worker.returncode == 0 and shape == recipe["outputShape"] and capture.status == "complete"
                         if "outputShape" in recipe else worker.returncode != 0 and capture.status == "partial"))
                if options.project_source:
                    row["sourceProjection"] = project_source(workspace, recipe, envelope, capture, temp, options.document_directory)
                    row["matched"] = row["matched"] and row["sourceProjection"]["matched"]
                if options.project_capture:
                    row["semanticProjection"] = project_semantic(capture)
                    row["matched"] = row["matched"] and row["semanticProjection"]["matched"]
                if options.portable_capture:
                    from services.semantic.portable import portable_capture
                    portable = portable_capture(capture)
                    if options.document_directory is not None:
                        options.document_directory.mkdir(parents=True, exist_ok=True)
                        (options.document_directory / (recipe["id"] + ".portable.json")).write_text(portable.model_dump_json())
                    exact = all(getattr(portable.capture, field) == getattr(capture, field) for field in
                        ("inputs", "output", "operations", "invocations", "values", "modules", "definitions", "diagnostics", "coverage"))
                    row["portableCapture"] = dict(sourceNames=portable.sourceNames,
                        bytes=len(portable.model_dump_json().encode()), exactComputation=exact)
                    row["matched"] = row["matched"] and exact
            else:
                row["matched"] = False
            results.append(row)
            print(json.dumps(row), flush=True)
    scope = "Canonical saved-plan graph workers"
    if options.project_source:
        scope += "; source correspondence through the multi-file API"
    if options.project_capture:
        scope += "; full and collapsed canonical projections"
    report = dict(version=1, manifestDigest=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        scope=scope + "; installed UI is not certified by this replay",
        python=sys.version.split()[0], results=results)
    options.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if all(row["matched"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
