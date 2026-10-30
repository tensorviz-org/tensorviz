"""Generate source-query pages and an actual CPU capture for fixed test sources.

Only this oracle executes the test model. The production source parser never
imports it. No user workspace model, checkpoint or agent is loaded.
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

SOURCES = {
    "_tensorviz_context_config.py": "WIDTH = 12\nHEADS = 3\n",
    "model.py": """import torch
from _tensorviz_context_config import WIDTH, HEADS
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.width = WIDTH
        self.heads = HEADS
        self.weight = torch.nn.Parameter(torch.eye(WIDTH))
    def forward(self, x):
        projected = x @ self.weight
        split = projected.reshape(x.shape[0], self.heads, self.width // self.heads)
        return split + split
""",
}


def build():
    import torch
    from services.ingest.trace import trace_model
    from services.semantic.contracts import CaptureLimits, CaptureSnapshot, SourceSnapshot
    from services.semantic.recorder import CaptureContext
    from services.semantic.source_bindings import CapturedSourceEvidence

    sources = [dict(file=file, source=source, digest=hashlib.sha256(source.encode()).hexdigest()) for file, source in SOURCES.items()]
    queries = [dict(file="_tensorviz_context_config.py", symbol=name, direction="downstream", depth=8, limit=100) for name in ("WIDTH", "HEADS")]
    queries.append(dict(file="model.py", symbol="forward", depth=0, limit=100))
    pages = [dict(query=query, page=query_source_context(dict(sources=sources, query=query))) for query in queries]
    with tempfile.TemporaryDirectory(prefix="tensorviz-source-oracle-") as folder:
        root = Path(folder).resolve()
        for file, source in SOURCES.items():
            (root / file).write_text(source)
        snapshot = SourceSnapshot(version=1, id="source:context-oracle", workspaceSessionId="session:oracle",
            workspaceId="workspace:oracle", modelId="model:context", modelFile="model.py", className="Model", generation=0,
            files=[dict(path=item["file"], digest=item["digest"], documentVersion=1, role="source") for item in sources],
            dependencyScope="complete", configDigest=None, environmentDigest="b" * 64, sourceMode="saved")
        evidence = CapturedSourceEvidence(snapshot, root)
        module_name = "_tensorviz_context_config"
        previous = sys.modules.get(module_name)
        module = types.ModuleType(module_name)
        try:
            exec(compile(SOURCES[module_name + ".py"], str(root / (module_name + ".py")), "exec"), module.__dict__)
            sys.modules[module_name] = module
            namespace = {"__name__": "context_oracle"}
            exec(compile(SOURCES["model.py"], str(root / "model.py"), "exec"), namespace)
            context = CaptureContext(snapshot, "c" * 64, "attempt:context-oracle",
                CaptureLimits(operations=100, invocations=100, values=100, bytes=1024 * 1024), source_evidence=evidence)
            result = trace_model(namespace["Model"](), (torch.ones(2, 12),), tier="real", semantic_context=context)
            assert result.error is None, result.error
            capture = CaptureSnapshot.model_validate(result.semantic_capture)
            assert capture.status == "complete" and all(op.sourceBindingIds for op in capture.operations), (
                capture.status, [(op.target, op.sourceBindingIds) for op in capture.operations])
        finally:
            if previous is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous
    return dict(version=1, python=platform.python_version(), torch=torch.__version__,
        scope="Fixed CPU source correspondence oracle; no live agent or installed acceptance", sources=sources,
        queries=pages, capture=capture.model_dump(mode="json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()
    result = build()
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(dumps(result) + "\n")
    print(f"Wrote {len(result['queries'])} source queries and {len(result['capture']['operations'])} actual operations with exact bindings")
