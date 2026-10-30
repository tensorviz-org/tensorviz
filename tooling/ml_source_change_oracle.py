"""Frozen conditional source pages and an independent real PyTorch width oracle.

Only this developer fixture generator executes code. Production source queries
receive text and cannot import a model. No user model or credentials are read.
"""
import argparse
import hashlib
from fixture_json import dumps
import platform
from pathlib import Path
import sys
import tempfile
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from services.semantic.source_context import query_source_context

CONFIG = "_tensorviz_change_config"
SOURCES = {
    CONFIG + ".py": "WIDTH = 12\nHEADS = 3\n",
    "model.py": f"""import torch
from {CONFIG} import WIDTH, HEADS
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(8, WIDTH)
        self.tied = self.up
        self.norm = torch.nn.LayerNorm(WIDTH)
        self.down = torch.nn.Linear(WIDTH, 4)
    def forward(self, x):
        hidden = self.norm(self.up(x) + self.tied(x))
        split = hidden.reshape(x.shape[0], HEADS, WIDTH // HEADS)
        return self.down(split.flatten(1))
""",
}


def build(source_files=SOURCES, expected_forwards=("passed", "failed", "passed")):
    import torch
    from services.ingest.trace import trace_model
    from services.semantic.contracts import CaptureLimits, CaptureSnapshot, SourceSnapshot
    from services.semantic.recorder import CaptureContext
    from services.semantic.source_bindings import CapturedSourceEvidence

    sources = [dict(file=file, source=source, digest=hashlib.sha256(source.encode()).hexdigest()) for file, source in source_files.items()]
    queries = [dict(file=CONFIG + ".py", symbol="WIDTH", direction="downstream", depth=8, limit=100, assumeValue=value) for value in (16, 24)]
    pages = [dict(query=query, page=query_source_context(dict(sources=sources, query=query))) for query in queries]
    with tempfile.TemporaryDirectory(prefix="tensorviz-source-change-oracle-") as folder:
        root = Path(folder).resolve()
        for file, source in source_files.items(): (root / file).write_text(source)
        snapshot = SourceSnapshot(version=1, id="source:change-oracle", workspaceSessionId="session:oracle",
            workspaceId="workspace:oracle", modelId="model:change", modelFile="model.py", className="Model", generation=0,
            files=[dict(path=item["file"], digest=item["digest"], documentVersion=1, role="source") for item in sources],
            dependencyScope="complete", configDigest=None, environmentDigest="b" * 64, sourceMode="saved")
        evidence = CapturedSourceEvidence(snapshot, root)
        previous = sys.modules.get(CONFIG)
        module = types.ModuleType(CONFIG)
        observations = []
        try:
            exec(compile(source_files[CONFIG + ".py"], str(root / (CONFIG + ".py")), "exec"), module.__dict__)
            sys.modules[CONFIG] = module
            namespace = {"__name__": "source_change_oracle"}
            exec(compile(source_files["model.py"], str(root / "model.py"), "exec"), namespace)
            context = CaptureContext(snapshot, "c" * 64, "attempt:change-oracle",
                CaptureLimits(operations=200, invocations=100, values=200, bytes=1024 * 1024), source_evidence=evidence)
            torch.manual_seed(0)
            with evidence.construction.observe(): model = namespace["Model"]().eval()
            result = trace_model(model, (torch.ones(2, 8),), tier="real", semantic_context=context)
            assert result.error is None, result.error
            capture = CaptureSnapshot.model_validate(result.semantic_capture)
            assert capture.status == "complete"
            assert all(item.constructorClassVerified for item in capture.modules if item.paths != [""])
            for width in (12, 16, 24):
                namespace["WIDTH"] = width
                torch.manual_seed(0)
                candidate = namespace["Model"]().eval()
                entry = dict(width=width, parameterShapes={name: list(value.shape) for name, value in candidate.named_parameters()},
                    tiedObject=candidate.up.weight is candidate.tied.weight)
                try:
                    output = candidate(torch.ones(2, 8))
                    entry.update(forward="passed", outputShape=list(output.shape))
                except RuntimeError:
                    entry.update(forward="failed", outputShape=None)
                observations.append(entry)
            assert tuple(entry["forward"] for entry in observations) == expected_forwards
        finally:
            if previous is None: sys.modules.pop(CONFIG, None)
            else: sys.modules[CONFIG] = previous
    return dict(version=1, python=platform.python_version(), torch=torch.__version__,
        scope="Authored CPU oracle, conditional source arithmetic; no live-agent or installed acceptance", sources=sources,
        queries=pages, capture=capture.model_dump(mode="json"), independentRuns=observations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()
    result = build()
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(dumps(result) + "\n")
    print(f"Wrote {len(result['queries'])} conditional source pages, {len(result['capture']['operations'])} actual operations and three independent CPU forwards")
