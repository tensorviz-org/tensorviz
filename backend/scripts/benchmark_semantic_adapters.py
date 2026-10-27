"""Measure raw forward, current capture, and non-strict export in fresh CPU processes.

This is an explicit offline evaluation, never an editor fallback or adapter selector.
Only pinned corpus copies execute. No model outputs, exception strings or weights
are written to the report. Peak RSS is a process high-water mark, not tensor memory.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
from scripts.model_corpus import MANIFEST, RECIPES, FACTORY, verified_copy

ADAPTERS = ("raw", "runtime", "export_non_strict")


def peak_rss_bytes():
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def measured(call):
    started = time.perf_counter()
    result = call()
    return result, round((time.perf_counter() - started) * 1000, 3)


def semantic_edges(projection):
    # Display ports deliberately change when collapsing; the original uses must not.
    return [(edge.id, edge.kind, edge.valueId, edge.sourcePortId, edge.targetPortId) for edge in projection.edges]


def run_sample(workspace: Path, recipe: dict, adapter: str) -> dict:
    import torch
    from services.ingest.execution_setup import prepare_execution
    from services.ingest.execution_values import capture_output
    from services.ingest.trace import trace_model
    from services.semantic.contracts import CaptureSnapshot, ResolvedExecutionPlan
    from services.semantic.execution_plan import applicability_digest
    from services.semantic.projection import project_capture
    from services.semantic.recorder import CaptureContext
    from services.semantic.source_bindings import CapturedSourceEvidence
    from services.validation.host_snapshot import snapshot

    row = {"status": "failed", "phase": "setup", "torch": str(torch.__version__)}
    try:
        started = time.perf_counter()
        plan = ResolvedExecutionPlan.model_validate(snapshot(dict(root=str(workspace),
            executable=sys.executable, environmentRevision=0, session="adapter-benchmark",
            workspaceId=recipe["repository"], runId=recipe["id"], generation=1,
            modelFile=recipe["file"], className=recipe["class"], timeoutMs=120000,
            execution=recipe["execution"]))["plan"])
        sources = CapturedSourceEvidence(plan.source, str(workspace)) if adapter == "runtime" else None
        with sources.construction.observe() if sources else nullcontext():
            prepared = prepare_execution(str(workspace / recipe["file"]), recipe["class"],
                                         str(workspace), recipe["execution"])
        row.update(setupMs=round((time.perf_counter() - started) * 1000, 3),
                   threads=torch.get_num_threads(), setupPeakRssBytes=peak_rss_bytes(),
                   inputDigest=prepared.evidence["inputs"]["digest"],
                   stateDigest=prepared.evidence["state"]["digest"])
        model, args, kwargs = prepared.model, prepared.args, prepared.kwargs
        row["phase"] = "capture" if adapter != "raw" else "forward"
        with torch.no_grad():
            if adapter == "raw":
                output, row["measureMs"] = measured(lambda: model(*args, **kwargs))
                row["measurePeakRssBytes"] = peak_rss_bytes()
                evidence = capture_output(output, include_values=False)
            elif adapter == "runtime":
                context = CaptureContext(plan.source, applicability_digest(plan), plan.id, plan.captureLimits,
                    prepared.evidence["inputs"]["digest"], prepared.evidence["state"]["digest"], source_evidence=sources)
                traced, row["measureMs"] = measured(lambda: trace_model(model, args, kwargs, tier="real",
                    frame_roots=(str(workspace),), preserve_mode=True, record_evidence=True,
                    grad_enabled=False, semantic_context=context))
                row["measurePeakRssBytes"] = peak_rss_bytes()
                if traced.error or traced.semantic_capture is None:
                    row["failure"] = "forward_failed" if traced.error else "capture_unavailable"
                    return row
                capture = CaptureSnapshot.model_validate(traced.semantic_capture)
                full, projection_ms = measured(lambda: project_capture(capture))
                grouped, collapse_ms = measured(lambda: project_capture(capture,
                    collapsed_invocations=[call.id for call in capture.invocations if call.id != capture.rootInvocationId]))
                row.update(operations=len(capture.operations), invocations=len(capture.invocations),
                    values=len(capture.values), captureStatus=capture.status, omissions=capture.coverage.omissions,
                    captureBytes=len(capture.model_dump_json().encode()), limits=capture.limits.model_dump(),
                    projectionMs=projection_ms, collapseMs=collapse_ms,
                    projectedNodes=len(full.nodes), collapsedVisibleNodes=sum(node.visible for node in grouped.nodes),
                    projectedUses=len(full.edges),
                    losslessGrouping=semantic_edges(full) == semantic_edges(grouped))
                evidence = traced.evidence["outputs"]
            else:
                exported, row["measureMs"] = measured(lambda: torch.export.export(model, args, kwargs, strict=False))
                row["measurePeakRssBytes"] = peak_rss_bytes()
                nodes = list(exported.graph.nodes)
                calls = [node for node in nodes if node.op == "call_function"]
                row.update(operations=len(calls), graphNodes=len(nodes),
                    sourceMetadataOperations=sum(bool(node.meta.get("stack_trace")) for node in calls),
                    moduleMetadataOperations=sum(bool(node.meta.get("nn_module_stack")) for node in calls),
                    rangeConstraints=len(exported.range_constraints),
                    note="FX metadata is navigation evidence, not exact edit bindings or a canonical TensorViz capture.")
                row["phase"] = "exported_forward"
                output, row["exportedForwardMs"] = measured(lambda: exported.module()(*args, **kwargs))
                row["phase"] = "reference_comparison"
                reference = model(*args, **kwargs)
                if isinstance(output, torch.Tensor) and isinstance(reference, torch.Tensor):
                    row["comparison"] = dict(outputShape=list(output.shape), referenceShape=list(reference.shape),
                        outputDtype=str(output.dtype), referenceDtype=str(reference.dtype),
                        outputFinite=bool(torch.isfinite(output).all()), referenceFinite=bool(torch.isfinite(reference).all()),
                        tolerance="torch.testing.assert_close defaults")
                    if not row["comparison"]["outputFinite"] or not row["comparison"]["referenceFinite"]:
                        row["failure"] = "nonfinite_comparison"
                        return row
                    if output.shape == reference.shape:
                        difference = (output - reference).abs()
                        row["comparison"].update(maxAbsoluteDifference=float(difference.max()),
                            meanAbsoluteDifference=float(difference.mean()))
                torch.testing.assert_close(output, reference)
                row["matchesEagerReference"] = True
                evidence = capture_output(output, include_values=False)
        row.update(outputDigest=evidence["digest"], outputLeaves=[
            {key: leaf[key] for key in ("path", "kind", "shape", "dtype") if key in leaf}
            for leaf in evidence["leaves"]], outputIdentityComplete=evidence["digest"] is not None,
            outputFinite=evidence["allFinite"])
        shapes = [leaf.get("shape") for leaf in row["outputLeaves"] if leaf["kind"] == "tensor"]
        complete = adapter != "runtime" or row["captureStatus"] == "complete" and row["losslessGrouping"]
        row["status"] = "passed" if (shapes == [recipe["outputShape"]] and complete
            and row["outputIdentityComplete"] and row["outputFinite"] is True) else "failed"
        row["phase"] = "finished"
    except Exception as error:
        # Never retain raw model errors, absolute paths, source or tensor values.
        row["errorType"] = type(error).__name__
        from services.validation.setup import ExecutionRequirement
        if isinstance(error, ExecutionRequirement):
            row["setupDiagnostic"] = error.diagnostic
    finally:
        row["processPeakRssBytes"] = peak_rss_bytes()
    return row


def summarize(samples: list[dict]) -> list[dict]:
    result = []
    for case in dict.fromkeys(row["case"] for row in samples):
        group = [row for row in samples if row["case"] == case]
        raw = [row for row in group if row["adapter"] == "raw" and row["status"] == "passed"]
        raw_identities = {(row["inputDigest"], row["stateDigest"], row["outputDigest"]) for row in raw}
        summary = {"case": case, "adapters": {}}
        for adapter in ADAPTERS:
            selected = [row for row in group if row["adapter"] == adapter]
            passed = [row for row in selected if row["status"] == "passed"]
            item = {"passed": len(passed), "samples": len(selected)}
            if passed:
                item.update(medianMs=round(statistics.median(row["measureMs"] for row in passed), 3),
                    minimumMs=min(row["measureMs"] for row in passed), maximumMs=max(row["measureMs"] for row in passed),
                    maxMeasurePeakRssBytes=max(row["measurePeakRssBytes"] for row in passed))
                if adapter == "runtime":
                    item["matchesRawIdentity"] = bool(raw) and len(raw_identities) == 1 and all(
                        all(row[key] is not None for key in ("inputDigest", "stateDigest", "outputDigest")) and
                        (row["inputDigest"], row["stateDigest"], row["outputDigest"]) in raw_identities for row in passed)
                if adapter != "raw" and raw:
                    item["timeToRawRatio"] = round(item["medianMs"] / statistics.median(row["measureMs"] for row in raw), 2)
            summary["adapters"][adapter] = item
        result.append(summary)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--repeats", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--worker", nargs=3, metavar=("WORKSPACE", "RECIPE", "ADAPTER"), help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.worker:
        workspace, recipe_path, adapter = options.worker
        if adapter not in ADAPTERS:
            parser.error("unknown adapter")
        result = run_sample(Path(workspace), json.loads(Path(recipe_path).read_text()), adapter)
        options.output.write_text(json.dumps(result, allow_nan=False))
        return 0
    if options.models is None or options.timeout < 1 or options.timeout > 600:
        parser.error("--models and a timeout of 1–600 seconds are required")
    recipes = [recipe for recipe in json.loads(RECIPES.read_text())["recipes"] if "outputShape" in recipe]
    if options.case:
        recipes = [recipe for recipe in recipes if recipe["id"] in options.case]
        if {recipe["id"] for recipe in recipes} != set(options.case):
            parser.error("choose known successful corpus cases")
    samples = []
    with tempfile.TemporaryDirectory(prefix="tensorviz-adapter-benchmark-") as directory:
        root = Path(directory)
        for repo in sorted({recipe["repository"] for recipe in recipes}):
            verified_copy(options.models, repo, root / repo)
        if (root / "whisper-main").exists():
            (root / "whisper-main/audit_setup.py").write_bytes(FACTORY.read_bytes())
        for recipe in recipes:
            recipe_path = root / "recipe.json"
            recipe_path.write_text(json.dumps(recipe))
            for sample in range(options.repeats):
                # Rotate order to reduce systematic cache/thermal advantage.
                order = ADAPTERS[sample % 3:] + ADAPTERS[:sample % 3]
                for adapter in order:
                    output = root / "sample.json"
                    output.unlink(missing_ok=True)
                    started = time.perf_counter()
                    with tempfile.TemporaryFile() as discarded:
                        try:
                            child = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                                "--worker", str(root / recipe["repository"]), str(recipe_path), adapter,
                                "--output", str(output)], cwd=root / recipe["repository"],
                                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                                stdout=discarded, stderr=discarded, timeout=options.timeout)
                            row = json.loads(output.read_text()) if child.returncode == 0 and output.exists() else {
                                "status": "worker_failed", "exitCode": child.returncode}
                        except subprocess.TimeoutExpired:
                            row = {"status": "timeout"}
                    row.update(case=recipe["id"], adapter=adapter, sample=sample + 1,
                        processWallSeconds=round(time.perf_counter() - started, 3))
                    samples.append(row)
                    print(json.dumps({key: row[key] for key in ("case", "adapter", "sample", "status", "processWallSeconds")}), flush=True)
    report = dict(version=1, manifestDigest=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        factoryDigest=hashlib.sha256(FACTORY.read_bytes()).hexdigest(),
        python=platform.python_version(), platform=platform.platform(),
        method="One cold capture/forward per fresh process; fixed CPU recipes; setup excluded from measureMs. "
               "Process high-water RSS includes imports/setup; runtime measurement includes source/evidence recording "
               "and snapshot serialization. Export is strict=False once, with no fallback. No browser-render or GPU claim.",
        summaries=summarize(samples), samples=samples)
    options.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    # Unsupported export is an evaluation result. Broken eager/canonical baselines fail the gate.
    return 0 if all(row["status"] == "passed" for row in samples if row["adapter"] != "export_non_strict") and all(
        row["adapters"]["runtime"].get("matchesRawIdentity") for row in report["summaries"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
