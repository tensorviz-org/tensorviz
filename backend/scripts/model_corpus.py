"""Verify and replay the pinned audit models in disposable, process-isolated copies.

No downloads, package installation, source edits, or execution happen on import.
The explicit CLI run is a raw-forward reference, not a graph-fidelity assertion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

BACKEND = Path(__file__).resolve().parents[1]
MANIFEST = BACKEND / "tests/fixtures/four-model-sources.json"
RECIPES = BACKEND / "tests/fixtures/four-model-corpus.json"
FACTORY = BACKEND / "tests/fixtures/corpus_whisper_factory.py"


def pinned_files(repository: str) -> dict[str, str]:
    prefix = repository + "/"
    files = {name[len(prefix):]: digest for name, digest in json.loads(MANIFEST.read_text()).items()
             if name.startswith(prefix)}
    if not files:
        raise ValueError("Repository is not part of the pinned corpus")
    return files


def verified_copy(models: Path, repository: str, destination: Path) -> int:
    """Copy only pinned bytes; never import from or write into the original tree."""
    source = (models / repository).resolve(strict=True)
    if destination.exists():
        raise ValueError("Corpus destination must be new")
    if destination.resolve().is_relative_to(models.resolve()):
        raise ValueError("Corpus copies must be outside the original model directory")
    files = pinned_files(repository)
    for relative, expected in files.items():
        parts = Path(relative)
        if parts.is_absolute() or ".." in parts.parts:
            raise ValueError("Invalid corpus manifest path")
        file = source / relative
        if not file.resolve(strict=True).is_relative_to(source) or file.is_symlink():
            raise ValueError("Corpus source must be a regular in-repository file")
        content = file.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Pinned corpus differs: {repository}/{relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return len(files)


def run_one(workspace: Path, recipe: dict) -> dict:
    # Only reached by the explicitly requested child process, after hash checks.
    sys.path.insert(0, str(BACKEND))
    import torch
    from services.ingest.execution_setup import prepare_execution

    called = set()
    hooks = []
    try:
        prepared = prepare_execution(str(workspace / recipe["file"]), recipe["class"],
                                     str(workspace), recipe["execution"])
        for name, module in prepared.model.named_modules():
            hooks.append(module.register_forward_pre_hook(lambda module, args, name=name: called.add(name)))
        with torch.no_grad():
            output = prepared.model(*prepared.args, **prepared.kwargs)
        shape = list(output.shape)
        finite = bool(torch.isfinite(output).all())
        success = recipe.get("outputShape") == shape and "errorTypes" not in recipe and finite
        return {"case": recipe["id"], "matched": success, "outcome": "completed",
                "outputShape": shape, "outputFinite": finite, "calledModules": sorted(called)}
    except Exception as error:
        # Do not copy arbitrary exception strings or model values into results.
        from services.validation.setup import ExecutionRequirement
        name = type(error).__name__
        underlying = error
        while isinstance(underlying, ExecutionRequirement) and underlying.__cause__ is not None:
            underlying = underlying.__cause__
        return {"case": recipe["id"], "matched": type(underlying).__name__ in recipe.get("errorTypes", []),
                "outcome": "failed", "errorType": name, "underlyingErrorType": type(underlying).__name__,
                "setupDiagnostic": error.diagnostic if isinstance(error, ExecutionRequirement) else None,
                "calledModules": sorted(called)}
    finally:
        for hook in hooks:
            hook.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--child", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        job = json.loads(args.child.read_text())
        result = run_one(Path(job["workspace"]), job["recipe"])
        Path(job["result"]).write_text(json.dumps(result, indent=2) + "\n")
        return
    if args.models is None:
        parser.error("--models must name the external pinned four-model directory")
    recipes = json.loads(RECIPES.read_text())["recipes"]
    selected = [r for r in recipes if not args.case or r["id"] in args.case]
    if not selected or set(args.case) - {r["id"] for r in recipes}:
        parser.error("Unknown corpus case")
    results = []
    with tempfile.TemporaryDirectory(prefix="tensorviz-model-corpus-") as directory:
        temp = Path(directory)
        for repository in sorted({r["repository"] for r in selected}):
            count = verified_copy(args.models, repository, temp / repository)
            print(f"Verified {repository}: {count} pinned files", flush=True)
        if args.verify_only:
            return
        for recipe in selected:
            workspace = temp / recipe["repository"]
            if "setup" in recipe["execution"]:
                (workspace / "audit_setup.py").write_bytes(FACTORY.read_bytes())
            job_file, result_file = temp / "job.json", temp / "result.json"
            result_file.unlink(missing_ok=True)
            job_file.write_text(json.dumps({"recipe": recipe, "workspace": str(workspace), "result": str(result_file)}))
            try:
                process = subprocess.run([sys.executable, "-B", __file__, "--child", str(job_file)],
                                         cwd=workspace, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
                result = (json.loads(result_file.read_text()) if process.returncode == 0 and result_file.exists()
                          else {"case": recipe["id"], "matched": False, "outcome": "worker_failed"})
            except subprocess.TimeoutExpired:
                result = {"case": recipe["id"], "matched": False, "outcome": "timed_out"}
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k != "calledModules"}), flush=True)
    report = {"version": 1, "manifestDigest": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
              "python": sys.version.split()[0], "results": results}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not all(result["matched"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
