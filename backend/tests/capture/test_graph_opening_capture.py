"""Prepared setup reaches the real worker and canonical graph without input search."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from fastapi import HTTPException

from api.routers import ide
from auth import CurrentUser
from services.ide import crossfile
from services.ide.graph_setup_profiles import prepare_profile
from services.ide.multifile import MultifileProjectionRequest
from services.semantic.execution_plan import profile_execution
from services.validation.contracts import RunSpec
from services.validation.host_snapshot import snapshot
from services.project_store import ProjectStore


def tensor(shape, dtype="float32"):
    return dict(kind="tensor", shape=shape, dtype=dtype, requires_grad=False, generator=dict(kind="full", value=1))


@pytest.mark.parametrize("case", ["scalar", "config", "keyword", "functional_keyword", "delegated",
    "positional", "inherited", "fallback_alias", "import_alias", "module_alias", "reexport",
    "torch_alias", "module_container", "alias_chain", "variadic", "no_inputs", "identity", "branch", "shape_failure", "dependency",
    "constructor_default"])
def test_prepared_setup_captures_one_explicit_invocation(tmp_path, monkeypatch, case):
    prefix = "from torch import nn\nimport torch\n"
    source = prefix + '''class Model(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc = nn.Linear(width, 4)
    def forward(self, x):
        y = x + 1
        return self.fc(y)
'''
    config_object = None
    extra_sources = {}
    init = dict(args=[], kwargs=dict(width=8))
    inputs = dict(args=[tensor([2, 7] if case == "shape_failure" else [2, 8])], kwargs={})
    if case == "config":
        source = prefix + "from dataclasses import dataclass\n@dataclass\nclass Config:\n    width: int\n" + source[len(prefix):].replace("self, width", "self, config").replace("nn.Linear(width,", "nn.Linear(config.width,")
        init = dict(args=[], kwargs={})
        config_object = dict(index=0, parameter="config", values=dict(width=8))
    if case == "keyword":
        source = source.replace("forward(self, x)", "forward(self, x, *, mask)").replace("y = x + 1", "y = x * mask")
        inputs["kwargs"]["mask"] = tensor([2, 8])
    if case == "delegated":
        source = source.replace("def forward(self, x):", "def forward_features(self, x, masks=None):")
        source += '''    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        return ret
'''
        inputs = dict(args=[], kwargs=dict(x=tensor([2, 8]), is_training=False))
    if case == "functional_keyword":
        source = prefix + '''class Model(nn.Module):
    def forward(self, x, *, mask):
        return torch.sin(x) * mask
'''
        init = dict(args=[], kwargs={})
        inputs = dict(args=[], kwargs=dict(x=tensor([2, 8]), mask=tensor([2, 8])))
    if case == "positional":
        source = source.replace("forward(self, x)", "forward(self, x, /, *, scale)")
        source = source.replace("y = x + 1", "y = x * scale")
        inputs["kwargs"]["scale"] = 2
    if case == "inherited":
        source = source.replace("class Model(nn.Module):", "class Base(nn.Module):")
        source += "\nclass Model(Base):\n    pass\n"
    if case == "fallback_alias":
        source = source.replace("class Model(nn.Module):", "class Base(nn.Module):")
        source += ("\ntry:\n    from tensorviz_missing_optional_dependency import OptionalBase\n"
                   "except ImportError:\n    OptionalBase = Base\n"
                   "class Model(OptionalBase):\n    pass\n")
    if case in {"import_alias", "module_alias", "reexport"}:
        base = source.replace("class Model(nn.Module):", "class Base(nn.Module):")
        if case == "reexport":
            extra_sources = {"pkg/base.py": base, "pkg/__init__.py": "from .base import Base as Exported\n"}
            source = "from pkg import Exported\nclass Model(Exported): pass\n"
        else:
            extra_sources = {"base.py": base}
            source = ("from base import Base as Alias\nclass Model(Alias): pass\n" if case == "import_alias"
                      else "import base as blocks\nclass Model(blocks.Base): pass\n")
    if case == "constructor_default":
        # DINOv2's embed_layer=PatchEmbed: static expansion cannot follow the
        # parameter, but the capture runs the layer from its own file.
        extra_sources = {"pkg/__init__.py": "", "pkg/embed.py": "from torch import nn\n\n"
            "class PatchEmbed(nn.Module):\n    def __init__(self, width):\n        super().__init__()\n"
            "        self.proj = nn.Linear(width, width)\n    def forward(self, x):\n        return self.proj(x)\n"}
        source = (prefix + "from pkg.embed import PatchEmbed\n" + source[len(prefix):]).replace(
            "self, width)", "self, width, embed_layer=PatchEmbed)").replace(
            "super().__init__()\n", "super().__init__()\n        self.patch_embed = embed_layer(width)\n").replace(
            "y = x + 1", "y = self.patch_embed(x)")
    if case == "torch_alias":
        source = "from torch.nn import Module as TorchModule\n" + source.replace("class Model(nn.Module):", "class Model(TorchModule):")
    if case == "module_container":
        source = prefix + "class Model(nn.Sequential):\n    def __init__(self, width):\n        super().__init__(nn.Linear(width, 4))\n"
    if case == "alias_chain":
        source = source.replace("class Model(nn.Module):", "class Base(nn.Module):")
        source += "\nAlias = Base\nSecondAlias = Alias\nclass Model(SecondAlias): pass\n"
    if case == "variadic":
        source = source.replace("forward(self, x)", "forward(self, *values)")
        source = source.replace("y = x + 1", "y = values[0] + 1")
    if case in {"no_inputs", "identity"}:
        source = prefix + ('''class Model(nn.Module):
    def forward(self):
        return torch.ones(2, 8).sin()
''' if case == "no_inputs" else '''class Model(nn.Module):
    def forward(self, x):
        return x
''')
        init = dict(args=[], kwargs={})
        if case == "no_inputs":
            inputs = dict(args=[], kwargs={})
    if case == "branch":
        source = source.replace("y = x + 1", "y = x.sin() if x.sum() > 0 else x.cos()")
    if case == "dependency":
        source = "import tensorviz_missing_dependency_for_opening_test\n" + source
    (tmp_path / "model.py").write_text(source)
    for file, text in extra_sources.items():
        path = tmp_path / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    digest = hashlib.sha256(source.encode()).hexdigest()
    request = MultifileProjectionRequest(protocol_version=1,
        root=dict(file="model.py", cls="Model", source_digest=digest, document_version=1),
        sources=[dict(file="model.py", source=source, digest=digest),
                 *(dict(file=file, source=text, digest=hashlib.sha256(text.encode()).hexdigest())
                   for file, text in extra_sources.items())])
    if case == "functional_keyword":
        # A setup helper for another model is in the editor's candidate sources,
        # but this invocation never imports it.
        helper_source = "from model import Model\n\ndef make_model():\n    return Model()\n"
        helper_file = ".tensorviz/setup_Other.py"
        (tmp_path / ".tensorviz").mkdir()
        (tmp_path / helper_file).write_text(helper_source)
        request.sources.append(type(request.sources[0])(file=helper_file, source=helper_source,
            digest=hashlib.sha256(helper_source.encode()).hexdigest()))
    prepared = prepare_profile(request, "", dict(kind="model", model="model:Model", mode="eval", device="cpu",
        grad_enabled=False, init=init, inputs=inputs), config_object=config_object)
    assert prepared["ok"], prepared
    (tmp_path / "tensorviz.toml").write_text(prepared["config"])
    if prepared["factory"]:
        helper = tmp_path / prepared["factory"]["file"]
        helper.parent.mkdir()
        helper.write_text(prepared["factory"]["source"])
    preflight = dict(root=str(tmp_path), executable=sys.executable, environmentRevision=0,
        session="opening", workspaceId="workspace", runId=f"opening_{case}", generation=1,
        modelFile="model.py", className="Model", timeoutMs=30000, profile=prepared["profile"])
    first = snapshot(preflight)
    assert first["resolution"]["status"] == "resolved", first["resolution"]
    spec = RunSpec.model_validate(first["resolution"]["spec"])
    # Reopening resolves the same saved invocation without inference or another form.
    second = snapshot({**preflight, "runId": "reopen"})
    assert second["resolution"]["spec"]["profile"] == first["resolution"]["spec"]["profile"]
    resolved = snapshot({**preflight, "execution": profile_execution(spec.profile), "expectedIdentity": spec.identity.model_dump()})
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(dict(version=1, plan=resolved["plan"], applicabilityDigest=resolved["applicabilityDigest"])))
    plan.chmod(0o600)
    run = subprocess.run([sys.executable, "-B", "-m", "services.ingest.worker", "--resolved-execution",
        "--workspace", str(tmp_path), "--file", str(tmp_path / "model.py"), "--cls", "Model", "--json"],
        cwd=Path(__file__).parent.resolve().parents[1], env={**os.environ, "TENSORVIZ_EXECUTION_PLAN": str(plan)},
        text=True, capture_output=True, timeout=45)
    envelope = json.loads(run.stdout)
    if case == "dependency":
        assert run.returncode != 0
        assert envelope["error"]["missing_module"] == "tensorviz_missing_dependency_for_opening_test"
        assert envelope["error"]["diagnostic"]["code"] == "missing_dependency"
        return
    assert run.returncode == (4 if case == "shape_failure" else 0), envelope.get("error")
    capture = envelope["result"]["diagnostics"]["semantic_capture"]
    assert len([call for call in capture["invocations"] if call["parentId"] is None]) == 1
    assert envelope["result"]["diagnostics"]["execution"]["resolvedPlan"]["execution"]["inputs"]["binding"] == inputs
    if case == "scalar":
        # Unsupported static lowering must not prevent canonical capture display.
        def unsupported(*args, **kwargs):
            raise crossfile.ClassParseRefused("Unsupported static lowering")
        monkeypatch.setattr(crossfile, "project_workspace", unsupported)
    ide._store = ProjectStore(str(tmp_path / "graph.sqlite3"))
    if case == "fallback_alias":
        with pytest.raises(HTTPException) as unsupported:
            ide.multifile_project(request, CurrentUser(uid="opening-tests"))
        assert unsupported.value.status_code == 422, "an unresolved base does not mean the class was deleted"
    request.trace_result = envelope
    projected = ide.multifile_project(request, CurrentUser(uid="opening-tests"))
    assert projected["traced"] and projected["document"]["semantic"]["nodes"]
    refs = [node["metadata"]["captureRef"] for node in projected["document"]["semantic"]["nodes"]]
    assert {ref["semanticId"] for ref in refs if ref["kind"] == "invocation"} == {
        call["id"] for call in capture["invocations"]
    }, "every executed invocation must reach the graph instead of a static substitute"
    assert {ref["semanticId"] for ref in refs if ref["kind"] == "operation"} == {
        operation["id"] for operation in capture["operations"]
    }, "the graph must contain the operations from this exact forward call"
    if case == "identity":
        assert not capture["operations"], "a valid no-op model still has a captured invocation"
    assert bool(projected["trace_error"]) == (case == "shape_failure")
    if case == "shape_failure":
        assert capture["operations"], "the executed prefix survives the failing layer"
    if case == "constructor_default":
        dependencies = {entry["file"]: entry["digest"] for entry in projected["dependencies"]}
        spans = [span for span in projected["spanmap"].values() if span["file"] == "pkg/embed.py"]
        assert spans and all(span["focus"] for span in spans)
        assert all(dependencies.get("pkg/embed.py") == span["file_digest"] for span in spans), \
            "hosts open a span only when its file is a projection dependency with the same digest"
    if case == "functional_keyword":
        assert helper_file not in {entry["file"] for entry in projected["dependencies"]}
        changed = source + "\n# changed after capture\n"
        changed_digest = hashlib.sha256(changed.encode()).hexdigest()
        request.sources[0].source = changed
        request.sources[0].digest = changed_digest
        request.root.source_digest = changed_digest
        with pytest.raises(HTTPException) as stale:
            ide.multifile_project(request, CurrentUser(uid="opening-tests"))
        assert stale.value.status_code == 400, "unrelated files must not disable captured-source freshness checks"
