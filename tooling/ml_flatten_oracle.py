"""Actual CPU flatten observations for source correspondence and dimension rules."""
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
    from services.semantic.portable import portable_capture
    from services.semantic.recorder import CaptureContext
    from services.semantic.source_bindings import CapturedSourceEvidence

    captures = []
    for expression, shape in [("h.flatten()", [2, 4]), ("torch.flatten(input=h, start_dim=1)", [2, 3, 4]),
                              ("h.flatten()", []), ("torch.flatten(h)", [0, 4]), ("h.flatten(-2, -1)", [2, 3, 4])]:
        source = "import torch\nclass Model(torch.nn.Module):\n    def forward(self, h):\n        return " + expression + "\n"
        with tempfile.TemporaryDirectory(prefix="tensorviz-flatten-") as directory:
            root = Path(directory).resolve(); (root / "model.py").write_text(source)
            snapshot = SourceSnapshot(version=1, id="source:flatten", workspaceSessionId="session:oracle",
                workspaceId="workspace:oracle", modelId="model:flatten", modelFile="model.py", className="Model", generation=0,
                files=[dict(path="model.py", digest=hashlib.sha256(source.encode()).hexdigest(), documentVersion=1, role="source")],
                dependencyScope="complete", configDigest=None, environmentDigest="b" * 64, sourceMode="saved")
            namespace = {"__name__": "flatten_oracle"}
            exec(compile(source, str(root / "model.py"), "exec"), namespace)
            context = CaptureContext(snapshot, "c" * 64, "attempt:flatten",
                CaptureLimits(operations=100, invocations=100, values=100, bytes=1024 * 1024),
                source_evidence=CapturedSourceEvidence(snapshot, root))
            result = trace_model(namespace["Model"](), (torch.ones(shape),), tier="real", semantic_context=context)
            assert result.error is None, result.error
            capture = CaptureSnapshot.model_validate(result.semantic_capture)
            assert capture.status == "complete" and capture.operations[0].sourceFlattenCall is not None
            captures.append(dict(source=source, inputShape=shape,
                portable=portable_capture(capture).model_dump(mode="json")))
    baseline = captures[0]["portable"]["capture"]
    op = baseline["operations"][0]
    binding = next(i for i, item in enumerate(baseline["sourceBindings"]) if item["id"] == op["sourceFlattenCall"]["bindingId"])
    output = next(i for i, item in enumerate(baseline["values"]) if item["id"] == op["output"]["id"])
    prefix = ["operations", 0, "sourceFlattenCall"]
    argument = dict(parameter="start_dim", sourcePath=["args", 0], value=0)
    cases = [dict(name=name, patches=[dict(path=path, value=value)]) for name, path, value in [
        ("unknown binding", prefix + ["bindingId"], "absent"),
        ("nonexact binding", ["sourceBindings", binding, "quality"], "candidate"),
        ("different operation", ["operations", 0, "target"], "aten.reshape.default"),
        ("nondefault omitted start", prefix + ["startDim"], 1),
        ("nondefault omitted end", prefix + ["endDim"], 1),
        ("out of range dimension", prefix + ["endDim"], 32),
        ("duplicate parameter", prefix + ["arguments"], [argument, argument]),
        ("missing first positional argument", prefix + ["arguments"], [dict(parameter="end_dim", sourcePath=["args", 1], value=-1)]),
        ("wrong source path", prefix + ["arguments"], [{**argument, "sourcePath": ["kwargs", "end_dim"]}]),
        ("wrong supplied value", prefix + ["arguments"], [{**argument, "value": 1}]),
        ("wrong callable signature", prefix + ["arguments"], [{**argument, "sourcePath": ["args", 1]}]),
        ("wrong dispatcher shape", ["operations", 0, "arguments", "args", 1, "items", 0, "value"], "7"),
        ("wrong observed output", ["values", output, "shape"], [4, 2]),
        ("contradictory reshape evidence", ["operations", 0, "sourceShapeCall"], dict(callable="torch.Tensor.view",
            bindingId=op["sourceFlattenCall"]["bindingId"], dimensions=[dict(sourcePath=["args", 0], dimension=0, value=dict(kind="int", value="8"))])),
    ]]
    dimensions = []
    for shape in [[], [0, 4], [2, 4], [2, 3, 4]]:
        for start, end in [(0, -1), (1, -1), (-2, -1), (0, 0), (1, 0), (4, -1), (-4, 2), (0, 1)]:
            try:
                output = torch.flatten(torch.ones(shape), start, end)
                result = dict(status="passed", outputShape=list(output.shape))
            except (RuntimeError, IndexError):
                result = dict(status="failed", outputShape=None)
            dimensions.append(dict(inputShape=shape, startDim=start, endDim=end, **result))
    return dict(version=1, python=platform.python_version(), torch=torch.__version__,
        scope="Authored CPU captures and independent flatten calls; no source-edit propagation, installed or quality claim",
        captures=captures, cases=cases, dimensions=dimensions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()
    result = build()
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(dumps(result) + "\n")
    print(f"Wrote {len(result['captures'])} observed captures, {len(result['cases'])} contradiction controls and {len(result['dimensions'])} independent flatten calls")
