"""Authored CPU oracle for conditional source-to-operation dimension checks.

Only this fixture generator executes its own models. Production source queries
parse supplied text and never execute the baseline or proposed program.
"""
import argparse
import hashlib
from fixture_json import dumps
from pathlib import Path
import platform
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def build():
    import torch
    from services.ingest.trace import trace_model
    from services.semantic.contracts import CaptureLimits, CaptureSnapshot, SourceSnapshot
    from services.semantic.recorder import CaptureContext
    from services.semantic.source_bindings import CapturedSourceEvidence
    from services.semantic.source_context import query_source_context

    cases = []
    for name, expression, derived in [
        ("variadic", "h.view(-1, WIDTH)", True),
        ("keyword_tuple", "torch.reshape(input=h, shape=(-1, WIDTH))", True),
        ("tensor_dimension", "h.view(h.shape[0], WIDTH)", True),
        ("unsupported_size_call", "h.view(h.size(0), WIDTH)", False),
        ("expanded_tuple", "h.view(*(-1, WIDTH))", False),
    ]:
        source = "import torch\nWIDTH = 4\nclass Model(torch.nn.Module):\n    def forward(self, h):\n        return " + expression + "\n"
        sources = [dict(file="model.py", source=source, digest=hashlib.sha256(source.encode()).hexdigest())]
        queries = [dict(file="model.py", symbol="WIDTH", direction="downstream", depth=8, limit=100, assumeValue=width) for width in (8, 3)]
        pages = [dict(query=query, page=query_source_context(dict(sources=sources, query=query))) for query in queries]
        with tempfile.TemporaryDirectory(prefix="tensorviz-source-shape-") as folder:
            root = Path(folder).resolve(); (root / "model.py").write_text(source)
            snapshot = SourceSnapshot(version=1, id="source:" + name, workspaceSessionId="session:oracle",
                workspaceId="workspace:oracle", modelId="model:" + name, modelFile="model.py", className="Model", generation=0,
                files=[dict(path="model.py", digest=sources[0]["digest"], documentVersion=1, role="source")],
                dependencyScope="complete", configDigest=None, environmentDigest="b" * 64, sourceMode="saved")
            evidence = CapturedSourceEvidence(snapshot, root)
            namespace = {"__name__": "source_shape_oracle"}
            exec(compile(source, str(root / "model.py"), "exec"), namespace)
            context = CaptureContext(snapshot, "c" * 64, "attempt:" + name,
                CaptureLimits(operations=100, invocations=100, values=100, bytes=1024 * 1024), source_evidence=evidence)
            result = trace_model(namespace["Model"](), (torch.ones(2, 4),), tier="real", semantic_context=context)
            assert result.error is None, result.error
            capture = CaptureSnapshot.model_validate(result.semantic_capture)
            assert capture.status == "complete"
            if derived: assert capture.operations[0].sourceShapeCall is not None
            observations = []
            for width in (4, 8, 3):
                # Compile the actual changed literal independently of the query
                # and capture. No predicted dimensions enter this execution.
                candidate = {}
                exec(compile(source.replace("WIDTH = 4", "WIDTH = " + str(width)), "candidate.py", "exec"), candidate)
                try:
                    output = candidate["Model"]()(torch.ones(2, 4))
                    observations.append(dict(width=width, forward="passed", outputShape=list(output.shape)))
                except RuntimeError:
                    observations.append(dict(width=width, forward="failed", outputShape=None))
            assert observations[0]["forward"] == "passed" and observations[-1]["forward"] == "failed"
        cases.append(dict(name=name, derivable=derived, sources=sources, queries=pages,
            capture=capture.model_dump(mode="json"), independentRuns=observations))
    return dict(version=1, python=platform.python_version(), torch=torch.__version__,
        scope="Authored CPU oracle; local fixed-input shape constraints, no provider-quality or installed claim", cases=cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()
    result = build()
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(dumps(result) + "\n")
    print(f"Wrote {len(result['cases'])} source/capture cases, ten conditional query pages and fifteen independent CPU forwards")
