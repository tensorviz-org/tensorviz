"""Recapture a reviewed-form in-place activation patch in verified disposable copies.

This checks backend round trips, not the native UI or installed VSIX. Original
model repositories are only read; model execution is explicit via this command.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from scripts.model_corpus import BACKEND, RECIPES, verified_copy
from services.semantic.contracts import CaptureSnapshot, ResolvedExecutionPlan
from services.semantic.execution_plan import applicability_digest
from services.semantic.source_edits import CapturedEditResolver, SourceEditRefused
from services.validation.host_snapshot import snapshot


def execute(workspace, recipe, directory, stage):
    plan = snapshot(dict(root=str(workspace), executable=sys.executable, environmentRevision=0,
        session="activation-options", workspaceId=recipe["repository"], runId=stage, generation=1 if stage == "before" else 2,
        modelFile=recipe["file"], className=recipe["class"], timeoutMs=180000, execution=recipe["execution"]))["plan"]
    plan_file = directory / (stage + ".json")
    with os.fdopen(os.open(plan_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "w") as handle:
        json.dump(dict(version=1, plan=plan,
            applicabilityDigest=applicability_digest(ResolvedExecutionPlan.model_validate(plan))), handle)
    worker = subprocess.run([sys.executable, "-B", str(BACKEND / "services/ingest/bootstrap.py"),
        "--file", str(workspace / recipe["file"]), "--cls", recipe["class"], "--workspace", str(workspace),
        "--resolved-execution", "--tier", "real", "--json"], cwd=workspace,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "TENSORVIZ_EXECUTION_PLAN": str(plan_file)},
        capture_output=True, text=True, timeout=180)
    if worker.returncode:
        raise RuntimeError("Corpus candidate execution failed")
    return CaptureSnapshot.model_validate(json.loads(worker.stdout)["result"]["diagnostics"]["semantic_capture"])


def replay(models, recipe):
    with tempfile.TemporaryDirectory(prefix="tensorviz-activation-options-") as folder:
        directory = Path(folder).resolve()
        workspace = directory / recipe["repository"]
        count = verified_copy(models, recipe["repository"], workspace)
        before = execute(workspace, recipe, directory, "before")
        sources = {file.path: (workspace / file.path).read_bytes().decode("utf-8") for file in before.source.files if file.role == "source"}
        resolver = CapturedEditResolver(before, sources)
        selected = None
        for call in before.invocations:
            try:
                cap = resolver.module_activation_capability(call.id)
            except SourceEditRefused:
                continue
            if cap.expected == "relu" and cap.options.inplace is True:
                selected = cap
                break
        if selected is None:
            raise RuntimeError("The pinned fixture no longer provides its in-place activation recipe")
        intent = dict(kind="replace_activation", capabilityId=selected.id, target=selected.target.model_dump(mode="json"),
                      expected="relu", value="silu", scope="source_expression")
        patch = resolver.plan(intent)
        assert patch["validation"] == "static_only" and len(patch["edits"]) == 1
        edit = patch["edits"][0]
        start, end = edit["range"]["start"], edit["range"]["end"]
        assert start[0] == end[0]
        lines = sources[patch["file"]].split("\n")
        line = lines[start[0] - 1].encode("utf-16-le")
        lines[start[0] - 1] = (line[:start[1] * 2] + edit["newText"].encode("utf-16-le") + line[end[1] * 2:]).decode("utf-16-le")
        candidate = "\n".join(lines)
        assert hashlib.sha256(candidate.encode()).hexdigest() == patch["source_digest"]
        (workspace / patch["file"]).write_bytes(candidate.encode())
        after = execute(workspace, recipe, directory, "after")
        assert before.status == after.status == "complete" and before.inputs == after.inputs and before.output == after.output
        affected = {module.id for module in before.modules if selected.expression.id in module.constructorBindingIds}
        changed_calls = {call.id for call in before.invocations if call.moduleId in affected}
        assert len(before.operations) == len(after.operations) and len(before.invocations) == len(after.invocations)
        changed = []
        for old, new in zip(before.operations, after.operations):
            expected = "aten.silu_.default" if old.invocationId in changed_calls else old.target
            assert new.target == expected
            if old.target != new.target:
                assert old.target == "aten.relu_.default"
                changed.append(old.id)
            assert old.model_dump(exclude={"target", "schemaSignature", "sourceBindingIds"}) == new.model_dump(
                exclude={"target", "schemaSignature", "sourceBindingIds"})
        assert before.values == after.values  # Includes all storage aliases, logical versions and metadata.
        assert [module.paths for module in before.modules] == [module.paths for module in after.modules]
        assert [call.model_dump(exclude={"sourceBindingIds"}) for call in before.invocations] == [
            call.model_dump(exclude={"sourceBindingIds"}) for call in after.invocations]
        assert len(changed) == selected.scope.observedOperations
        sources[patch["file"]] = candidate
        recaptured = CapturedEditResolver(after, sources).module_activation_capability(selected.target.semanticId)
        assert recaptured.expected == "silu" and recaptured.options == selected.options
        return dict(case=recipe["id"], pinnedFiles=count, sourceFile=patch["file"], patchCount=1,
            changedOperations=len(changed), affectedInstances=selected.scope.observedModuleInstances,
            options=selected.options.model_dump(), unchangedArgumentsEffectsAliasesAndMetadata=True,
            recapturedActivation=recaptured.expected, status=after.status, matched=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    cases = {"tensorsssss--explicit-correct", "Pytorch-UNet-master--explicit-correct"}
    recipes = [recipe for recipe in json.loads(RECIPES.read_text())["recipes"] if recipe["id"] in cases]
    assert {recipe["id"] for recipe in recipes} == cases
    results = [replay(options.models, recipe) for recipe in recipes]
    report = dict(version=1, scope="Saved-source in-place activation round trips in disposable copies; native UI not certified", results=results)
    options.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
