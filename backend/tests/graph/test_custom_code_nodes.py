"""Custom code as a first-class node (task.md M6).

The claim M6 makes is narrow and easy to fake, so each test here states the
half that could be faked and checks the other one:

* a user's own ``nn.Module`` becomes a node with the placeholder op
  ``custom_code`` — never an op named after their class, which no registry has
  and which ``graph_input_safety._LAYER_OPERATION`` could not carry anyway —
  plus the module path that would import the class back;
* that node is **valid, verifiable and generatable only because it carries
  evidence**. `_out_shape` is a measurement (the trace ran it) or a declaration
  (``# tensorviz: out_shape=`` in the class's own source). Without one, strict
  validation still declines, and it declines by saying *nobody has run this*
  rather than by quietly passing the input shape through — a passthrough there
  would put a number the model never produced in front of every node downstream;
* an opaque statement regenerates from its own source text, rebound to the
  variable the graph feeds it, and a statement that is no longer Python is
  reported rather than written.

No module-scope ``import torch`` (task.md §8.3); every tensor here is 2x16.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient


import server_v2  # noqa: E402
from models.graph_models import CodeParseRequest  # noqa: E402
from services.code_generation import CodeGenerationService  # noqa: E402
from services.ide import code_apply, projection  # noqa: E402
from utils.code_parser import CodeParserService  # noqa: E402

BACKEND = Path(__file__).parent.resolve().parent.parent
FIXTURE_REPO = BACKEND / "tests" / "fixtures" / "ide_repo"
CUSTOM = FIXTURE_REPO / "custom"


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(server_v2.app)


def trace_of(module: str, cls: str, *arguments: str) -> Dict[str, Any]:
    """The worker envelope for one fixture class, as the extension would get it."""
    completed = subprocess.run(
        [
            sys.executable, "-m", "services.ingest.worker",
            "--file", str(CUSTOM / f"{module}.py"), "--cls", cls, "--json", *arguments,
        ],
        cwd=str(BACKEND),
        capture_output=True,
        text=True,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
        timeout=300,
    )
    assert completed.stdout.strip(), f"worker printed nothing; stderr={completed.stderr[-2000:]}"
    envelope = json.loads(completed.stdout)
    assert completed.returncode == 0, f"exit {completed.returncode}: {envelope.get('error')}"
    return envelope


def params_of(envelope: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(node["id"]): ((node.get("data") or {}).get("parameters") or {})
        for node in envelope["result"]["graph"]["nodes"]
    }


def ops_of(envelope: Dict[str, Any]) -> Dict[str, str]:
    return {
        str(node["id"]): str(((node.get("data") or {}).get("layerType") or {}).get("id") or "")
        for node in envelope["result"]["graph"]["nodes"]
    }


def parsed_graph(module: str, cls: str):
    """The AST tier's graph for one fixture class."""
    source = (CUSTOM / f"{module}.py").read_text(encoding="utf-8")
    parsed = CodeParserService().parse_pytorch_code(
        CodeParseRequest(code=source, input_shape=[2, 16], file=f"custom/{module}.py")
    )
    return next(entry for entry in parsed["graphs"] if entry["name"] == cls)["graph"]


def canvas_graph(module: str, cls: str):
    """The flat graph that reaches ``/api/models/generate-code`` from the IDE.

    The canvas holds the PROJECTION, and the client rebuilds each node's
    ``layerType`` from the semantic ``op`` before exporting — so the payload the
    generator sees is the parsed graph with the projection's custom-class
    adoption applied to it. Reproduced here rather than hand-written, so a change
    to the adoption rule shows up in this test instead of silently passing it by.
    """
    from models.graph_models import Graph  # noqa: PLC0415

    source = (CUSTOM / f"{module}.py").read_text(encoding="utf-8")
    graph = parsed_graph(module, cls)
    raw = {
        "nodes": [n.model_dump() for n in graph.nodes],
        "edges": [e.model_dump() for e in graph.edges],
    }
    classes = projection.workspace_classes(source, f"custom/{module}.py")
    for entry in raw["nodes"]:
        projection.adopt_custom_class(entry, classes=classes)
    return Graph(**raw)


def node(node_id: str, node_type: str, layer_id: str, layer_name: str,
         parameters: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": node_id,
        "type": node_type,
        "position": {"x": 0.0, "y": 0.0},
        "data": {
            "layerType": {"id": layer_id, "name": layer_name, "parameters": []},
            "parameters": parameters,
        },
    }


def architecture_graph(params: Dict[str, Any]):
    """input -> one custom_code node -> output, as the canonical validator sees it."""
    from models.architecture_models import ArchitectureEdge, ArchitectureGraph, ArchitectureNode

    return ArchitectureGraph(
        graph_id="g", name="g", revision=0, inputs=[{"name": "in", "shape": "2,16"}],
        nodes=[
            ArchitectureNode(node_id="in", kind="input", op="Input",
                             params={"shape": "2,16"}, in_ports=[], out_ports=["y"]),
            ArchitectureNode(node_id="blk", kind="layer", op="custom_code",
                             params=params, in_ports=["x"], out_ports=["y"]),
            ArchitectureNode(node_id="out", kind="output", op="Output",
                             params={}, in_ports=["x"], out_ports=[]),
        ],
        edges=[
            ArchitectureEdge(edge_id="e0", source_node_id="in", source_port="y",
                             target_node_id="blk", target_port="x"),
            ArchitectureEdge(edge_id="e1", source_node_id="blk", source_port="y",
                             target_node_id="out", target_port="x"),
        ],
        modules=[], operations=[], metadata={},
    )


# ── the trace tier ───────────────────────────────────────────────────────────


def test_a_users_own_block_traces_to_custom_code_and_says_where_it_lives():
    """`self.blk = MyBlock(16)` is not an operation and must not pretend to be one.

    Before M6 this node carried `op: "MyBlock"` — a name no registry resolves, so
    every downstream plane (validate, verify, codegen) either refused it or
    invented an `nn.MyBlock`. It now carries the bare placeholder plus the two
    facts that make the class findable: which class, and the module that imports
    it. `_source_line` is checked against `inspect` rather than a literal so the
    assertion cannot drift when the fixture moves down a line.
    """
    import importlib.util  # noqa: PLC0415
    import inspect  # noqa: PLC0415 - kept out of module scope with torch (task.md §8.3)

    # Loaded by PATH, not by `import models.blocks`: backend has a `models`
    # package of its own and it is already imported, so the dotted name resolves
    # to the wrong one (the fixture's `models/__init__.py` says why).
    spec = importlib.util.spec_from_file_location(
        "m6_fixture_blocks", FIXTURE_REPO / "models" / "blocks.py"
    )
    module = importlib.util.module_from_spec(spec)
    # `inspect` resolves a class's file through `sys.modules[cls.__module__]`, so
    # the module has to be registered for the duration or it reads as builtin.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        defined_at = inspect.getsourcelines(module.MyBlock)[1]
    finally:
        sys.modules.pop(spec.name, None)

    envelope = trace_of("net", "Custom", "--depth", "1")
    blk = params_of(envelope)["blk"]

    assert ops_of(envelope)["blk"] == "custom_code"
    assert blk["_class"] == "MyBlock"
    assert blk["_import_path"] == "models.blocks"
    assert blk["_source_line"] == defined_at
    assert blk["_out_shape"] == "2x16"
    assert blk["_opaque"] is True
    assert blk["_shape_provenance"] == "observed"


def test_a_torch_layer_in_the_same_workspace_is_not_swallowed_by_custom_code():
    """The rule is "defined under the workspace", not "reached from it".

    `nn.Linear` is declared on a line of the user's own file and its class lives
    in site-packages, so a rule keyed on the declaration site would retype every
    torch layer in the workspace as opaque user code and lose its shape rule.
    """
    envelope = trace_of("net", "Custom", "--depth", "1")
    assert ops_of(envelope)["head"] == "linear"
    assert "_import_path" not in params_of(envelope)["head"]


def test_a_block_the_trace_never_reached_can_declare_its_own_shape():
    """`Branchy` takes neither branch, so nothing measured either block.

    A card with no shape makes every card after it unknown too. The class's own
    `# tensorviz: out_shape=` is the user's answer, and it is recorded as
    `declared` — never as `observed`, because the difference between "we ran it"
    and "you told us" is exactly what the canvas has to be able to show.
    """
    params = params_of(trace_of("declared", "Branchy", "--depth", "1"))

    assert params["declared"]["_shape_provenance"] == "declared"
    assert params["declared"]["_out_shape"] == "2x16"


def test_a_symbolic_declaration_is_recorded_and_still_yields_no_shape():
    """`out_shape=(B, D)` names the axes and no numbers.

    Recording it as a declaration is right; inventing a size for `B` would put a
    number the user never wrote in front of every node downstream, so the node
    keeps no `_out_shape` at all and validation goes on declining it.
    """
    symbolic = params_of(trace_of("declared", "Branchy", "--depth", "1"))["symbolic"]

    assert symbolic["_shape_provenance"] == "declared"
    assert symbolic["_declared_out_shape"] == "B, D"
    assert "_out_shape" not in symbolic


# ── validation and verification ──────────────────────────────────────────────


def test_the_traced_graph_validates_and_runs_a_dry_forward(client: TestClient):
    """The whole point of the milestone: a model built from the user's own class
    is a checked model, not one the product shrugs at.

    Both planes are asserted because they fail differently — `/api/models/validate`
    used to refuse the node outright, and `verify_model` used to die in
    `build_layer` with "Unknown layer type 'custom_code'" before either could
    report anything.
    """
    from services.verification_service import verify_model  # noqa: PLC0415

    graph = trace_of("net", "Custom", "--depth", "1")["result"]["graph"]

    response = client.post("/api/models/validate", json={"graph": graph})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["valid"] is True, body["errors"]

    result = verify_model(graph, input_shape=[2, 16])
    assert result["ok"] is True, result["errors"]
    assert result["shapes"]["blk"] == [2, 16]


def test_the_dry_forward_uses_the_measurement_instead_of_building_the_class():
    """A `custom_code` LAYER card — what the canvas holds once a projection is
    loaded — has no torch class to construct.

    Stage 2 used to hand `custom_code` to `build_layer`, which looked it up in
    `torch.nn` and took the whole verification down with "Unknown layer type"
    before any shape could be reported. Skipping the build is what lets the node
    fall through to the branch that trusts `_out_shape` — and the shape has to
    reach `head`, or the model is only checkable up to its first custom block.
    """
    from services.verification_service import verify_model  # noqa: PLC0415

    graph = {
        "nodes": [
            node("i", "inputNode", "input", "Input", {"shape": "2,16"}),
            node("blk", "layerNode", "custom_code", "custom_code",
                 {"_class": "MyBlock", "_import_path": "models.blocks", "_out_shape": "2x16"}),
            node("head", "layerNode", "linear", "Linear",
                 {"in_features": 16, "out_features": 4}),
        ],
        "edges": [
            {"id": "e1", "source": "i", "target": "blk"},
            {"id": "e2", "source": "blk", "target": "head"},
        ],
    }

    result = verify_model(graph, input_shape=[2, 16])

    assert result["ok"] is True, result["errors"]
    assert result["shapes"]["blk"] == [2, 16]
    assert result["shapes"]["head"] == [2, 4]


def test_validate_still_refuses_source_code_nobody_has_run(client: TestClient):
    """The M1 refusal has to survive M6, or the milestone reintroduces the false
    green it was written to stop.

    A `python` node with no observed and no declared shape is a statement nobody
    has executed. Blessing it would answer valid:true for a graph whose only
    unchecked part is the part the user wrote themselves — and the message still
    has to say WHERE, because `python_1` is not a name they chose.
    """
    graph = {
        "nodes": [
            node("i", "inputNode", "input", "Input", {"shape": "2,16"}),
            node("python_1", "layerNode", "python", "Python", {
                "_raw_call": "out.transpose(1, 2)",
                "_source_file": "models/resblock.py",
                "_lineno": 15,
            }),
        ],
        "edges": [{"id": "e1", "source": "i", "target": "python_1"}],
    }

    response = client.post("/api/models/validate", json={"graph": graph})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["valid"] is False, "a statement nobody has run is not a checked model"
    message = " | ".join(entry.get("message", "") for entry in body["errors"])
    assert "python_1" in message
    assert "models/resblock.py:15" in message


def test_strict_validation_declines_a_placeholder_with_no_evidence():
    """Strict mode must fail, and must fail for the right reason.

    "No deterministic rule" is a statement about knowledge; a shape error would
    be a statement about the graph being wrong, which it is not. And the node must
    come back with NO output shape: the passthrough lists in this module are
    deliberately not extended (task.md M6), because an opaque statement is not
    shape-preserving and reporting the input shape as its output is a number the
    model never produced.
    """
    from services.canonical_validation import CanonicalValidationService  # noqa: PLC0415

    result = CanonicalValidationService().validate(
        architecture_graph({"_class": "MyBlock"}), strict=True
    )

    assert result.valid is False
    blk = result.node_results["blk"]
    assert blk.status == "error"  # strict promotes the warning
    assert "source code, not an operation" in (blk.message or "")
    assert blk.output_shape is None, "never passthrough"


def test_strict_validation_accepts_the_same_node_once_it_carries_a_shape():
    """The evidence is the whole difference. Same graph, same op, one extra key."""
    from services.canonical_validation import CanonicalValidationService  # noqa: PLC0415

    result = CanonicalValidationService().validate(
        architecture_graph({"_class": "MyBlock", "_out_shape": "2x16"}), strict=True
    )

    assert result.valid is True, result.global_errors
    assert result.node_results["blk"].output_shape == [2, 16]
    # The shape has to reach the nodes AFTER it, or a model is only ever checkable
    # up to its first custom block.
    assert result.node_results["out"].output_shape == [2, 16]


# ── code generation ──────────────────────────────────────────────────────────


def test_generated_code_imports_the_users_class_and_the_file_runs():
    """A generated file that instantiates `MyBlock` without importing it is a
    NameError printed under a correct-looking architecture.

    `exec` with the fixture on `sys.path` is the check that matters: the import
    has to be the one that actually resolves, not a plausible-looking string.
    """
    code = CodeGenerationService().generate_flat_code(
        canvas_graph("net", "Custom"), [2, 16], "Regenerated"
    )

    assert "from models.blocks import MyBlock" in code
    assert "self.blk = MyBlock(" in code

    # Run in a fresh interpreter rooted at the fixture. In-process would mean
    # evicting backend's own `models` package from `sys.modules` for the
    # duration, in a suite of 1600 tests sharing one process — and a subprocess
    # additionally executes the file's usage example, so this proves the model
    # runs and not merely that the module body compiles.
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(FIXTURE_REPO),
        capture_output=True,
        text=True,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "Output shape: (2, 16)" in completed.stdout, completed.stdout


def test_a_traced_block_whose_arguments_are_unknown_is_reported_incomplete():
    """`MyBlock(16)` and `MyBlock()` are different models, and only one of them is
    the user's.

    A trace can see the constructed object and not the call that made it: a
    constructor argument that is never stored as an attribute is gone by the time
    the tracer arrives. The IDE's own path does not hit this — the projection
    merges the declaration text onto the traced node — but a caller that
    generates straight off a worker envelope does, and it has to be told rather
    than handed a plausible file.
    """
    from models.graph_models import Graph  # noqa: PLC0415

    graph = Graph(**trace_of("net", "Custom", "--depth", "1")["result"]["graph"])

    unsupported = CodeGenerationService().unreproducible_modules(graph.nodes)

    assert [entry["node_id"] for entry in unsupported] == ["blk"]
    assert "defaults" in unsupported[0]["reason"]


def test_an_opaque_statement_regenerates_from_its_own_source_text():
    """`x = x.transpose(1, 2)` has no op and no parameters — the only thing that
    can reproduce it is the text it was read from.

    It is the first statement of the fixture's forward, so the variable it reads
    is the one the generated forward binds and the line comes out verbatim.
    """
    code = CodeGenerationService().generate_flat_code(
        parsed_graph("net", "Transposed"), [2, 4, 16], "Regenerated"
    )

    assert "x.transpose(1, 2)" in code
    ast.parse(code)


def test_a_statement_read_after_another_layer_is_rebound_not_copied():
    """The text names ITS OWN variable. Copied verbatim after another layer, it
    would read whatever `x` is at that point — a different model that still runs,
    which is the worst thing this generator can emit."""
    source = (
        "import torch.nn as nn\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(4, 4)\n"
        "    def forward(self, x):\n"
        "        x = self.fc(x)\n"
        "        x = x.transpose(1, 2)\n"
        "        return x\n"
    )
    parsed = CodeParserService().parse_pytorch_code(
        CodeParseRequest(code=source, input_shape=[2, 4, 4], file="models/net.py")
    )
    graph = next(entry for entry in parsed["graphs"] if entry["is_top_level"])["graph"]

    code = CodeGenerationService().generate_flat_code(graph, [2, 4, 4], "Regenerated")

    assert "out_fc.transpose(1, 2)" in code
    assert "x.transpose(1, 2)" not in code


def test_a_statement_that_is_no_longer_python_is_reported_and_not_written():
    """`_raw_call` rides on the node through saves, AI turns and hand edits.

    A file that does not compile is worse than an incomplete one, so the text is
    parsed before being interpolated: the node lands in the caller's `unsupported`
    list — which is what makes `complete: false` — and the forward refuses at run
    time instead of binding the variable to something plausible.
    """
    from models.graph_models import Graph  # noqa: PLC0415

    broken = Graph(**{
        "nodes": [
            node("i", "inputNode", "input", "Input", {"shape": "2,16"}),
            node("python_0", "layerNode", "python", "Python",
                 {"_raw_call": "x.transpose(1,", "_source_file": "models/net.py", "_lineno": 8}),
        ],
        "edges": [{"id": "e1", "source": "i", "target": "python_0"}],
    })
    service = CodeGenerationService()

    unsupported = service.unreproducible_modules(broken.nodes)
    assert [entry["node_id"] for entry in unsupported] == ["python_0"]
    assert not (not unsupported), "complete:false is what this list is for"

    code = service.generate_flat_code(broken, [2, 16], "Regenerated")
    assert "raise NotImplementedError" in code
    assert "x.transpose(1," not in code
    ast.parse(code)


def test_a_custom_code_node_with_nowhere_to_import_from_is_still_refused():
    """The relaxation is evidence-shaped, not blanket. A class name with no module
    behind it cannot be written into a new file at all, and inventing a body for
    it would be a different model wearing the same word."""
    from core.graph_input_safety import UnsafeGraphInput  # noqa: PLC0415
    from models.graph_models import Graph  # noqa: PLC0415

    graph = Graph(**{
        "nodes": [
            node("i", "inputNode", "input", "Input", {"shape": "2,16"}),
            node("blk", "layerNode", "custom_code", "custom_code",
                 {"_class": "MyBlock", "_source_file": "models/net.py", "_lineno": 22}),
        ],
        "edges": [{"id": "e1", "source": "i", "target": "blk"}],
    })

    with pytest.raises(UnsafeGraphInput) as excinfo:
        CodeGenerationService().generate_flat_code(graph, [2, 16], "Regenerated")
    assert "models/net.py:22" in str(excinfo.value)


# ── the source edit ──────────────────────────────────────────────────────────


def test_adding_a_workspace_block_writes_its_import_too():
    """A declaration whose name is not bound is a NameError the moment the file
    runs, so the import is part of the same edit rather than a second thing for
    the user to remember. Three lines and nothing else touched."""
    source = (FIXTURE_REPO / "models" / "resid.py").read_text(encoding="utf-8")
    parsed = CodeParserService().parse_pytorch_code(
        CodeParseRequest(code=source, file="models/resid.py")
    )
    entry = next(item for item in parsed["graphs"] if item["name"] == "Net")
    graph = entry["graph"]
    projected = projection.project(
        {
            "nodes": [n.model_dump() for n in graph.nodes],
            "edges": [e.model_dump() for e in graph.edges],
        },
        cls="Net", rel_file="models/resid.py", source=source,
    )

    result = code_apply.apply_operations(
        source,
        projected.spanmap,
        [
            {"op": "add_layer", "op_id": "a1", "node_id": "blk", "layer_op": "custom_code",
             "params": {"_class": "MyBlock", "_import_path": "models.blocks", "width": 16}},
            {"op": "connect", "op_id": "c1", "source_node_id": "l1", "target_node_id": "blk"},
        ],
        cls="Net", file="models/resid.py",
    )

    assert result.refused == []
    added = [line for line in result.source.splitlines() if line not in source.splitlines()]
    assert "from models.blocks import MyBlock" in added
    assert "        self.blk = MyBlock(width=16)" in added
    assert "        h = self.blk(h)" in added
    assert len(added) == 3, added


def test_an_import_that_is_already_there_is_not_written_twice():
    """Idempotence is the difference between an edit and a mess: the same block
    added to a file that already imports it must change two lines, not three."""
    source = (CUSTOM / "net.py").read_text(encoding="utf-8")
    projected = projection.project(
        {
            "nodes": [n.model_dump() for n in parsed_graph("net", "Custom").nodes],
            "edges": [e.model_dump() for e in parsed_graph("net", "Custom").edges],
        },
        cls="Custom", rel_file="custom/net.py", source=source,
    )

    result = code_apply.apply_operations(
        source,
        projected.spanmap,
        [
            {"op": "add_layer", "op_id": "a1", "node_id": "second", "layer_op": "custom_code",
             "params": {"_class": "MyBlock", "_import_path": "models.blocks"}},
            {"op": "connect", "op_id": "c1", "source_node_id": "blk", "target_node_id": "second"},
        ],
        cls="Custom", file="custom/net.py",
    )

    assert result.refused == []
    assert result.source.count("from models.blocks import MyBlock") == 1


def test_the_workspace_palette_is_told_how_to_import_each_class(client: TestClient):
    """A palette entry that only knows a class NAME cannot write the import that
    makes the inserted declaration resolve."""
    response = client.post(
        "/api/ide/workspace/classes", json={"root": str(FIXTURE_REPO / "models")}
    )

    assert response.status_code == 200, response.text
    by_name = {entry["name"]: entry for entry in response.json()["classes"]}
    assert by_name["MyBlock"]["import_path"] == "models.blocks"


def test_a_flat_export_reports_the_statement_it_could_not_write(client: TestClient):
    """`/api/models/generate-code` on a FLAT graph now answers the question it
    was silently skipping.

    The hierarchical branch has always asked `unreproducible_modules` which nodes
    it failed to write. The flat branch never did: it left `unsupported` at the
    empty list the function initialised and returned `complete: true`, so a file
    missing a line looked exactly like a file that was whole. That is the failure
    mode this repository's own audit exists to catch, and it is worse here than a
    refusal would be — the user downloads the file and finds out when it runs.

    Driven over HTTP against a real parsed graph, because the bug was in the
    route, not in the service the route calls.
    """
    code = (
        "import torch.nn as nn\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(4, 4)\n"
        "    def forward(self, x):\n"
        "        x = self.fc(x)\n"
        "        x = x.transpose(1, 2)\n"
        "        return x\n"
    )
    parsed = client.post(
        "/api/code/parse",
        json={"code": code, "input_shape": [2, 4, 4], "file": "models/net.py"},
    )
    assert parsed.status_code == 200, parsed.text
    graph = next(e for e in parsed.json()["graphs"] if e.get("graph"))["graph"]
    statement = next(
        node for node in graph["nodes"] if node["data"]["layerType"]["id"] == "python"
    )
    # A shape it could only have got from a run: without one the route refuses at
    # validation and never reaches generation at all.
    statement["data"]["parameters"]["_out_shape"] = "2x4x4"

    whole = client.post(
        "/api/models/generate-code",
        json={"graph": graph, "input_shape": [2, 4, 4], "class_name": "GeneratedModel"},
    )
    assert whole.status_code == 200, whole.text
    assert whole.json()["complete"] is True
    assert whole.json()["unsupported"] == []

    # The same graph, with the one thing that could have regenerated that line
    # truncated. Everything else about the request is identical.
    statement["data"]["parameters"]["_raw_call"] = "x.transpose(1, "
    partial = client.post(
        "/api/models/generate-code",
        json={"graph": graph, "input_shape": [2, 4, 4], "class_name": "GeneratedModel"},
    )
    assert partial.status_code == 200, partial.text
    body = partial.json()
    assert body["complete"] is False
    assert len(body["unsupported"]) == 1, body["unsupported"]
    entry = body["unsupported"][0]
    assert entry["module_path"] == "models/net.py"
    assert "not valid Python" in entry["reason"]



def test_a_layer_that_cannot_be_constructed_is_refused_not_written():
    """`nn.Linear()` parses. It does not construct.

    That gap is the whole bug: the only check on a rendered edit was `ast.parse`,
    which asks whether the text is Python, so a constructor missing its required
    arguments was written into the user's file and reported as a successful edit.
    The editor underlined it, the model stopped running, and nothing in the
    response said so.

    The discriminator is the torch class's own signature — nothing is constructed
    to find out, because a caller can name a layer far too large to instantiate
    just to check that it is well formed. `nn.GELU()` is valid with no arguments
    and must still be written; `nn.Linear()` is not and must be refused, naming
    what is missing.
    """
    from services.ide import code_apply

    assert code_apply._missing_constructor_arguments("linear", {}) == [
        "in_features",
        "out_features",
    ]
    assert code_apply._missing_constructor_arguments(
        "linear", {"in_features": 8, "out_features": 4}
    ) == []
    # Ops that legitimately take nothing must not be caught by the same net.
    for op in ("gelu", "relu", "dropout", "flatten"):
        assert code_apply._missing_constructor_arguments(op, {}) == [], op
    # And an op the registry does not know refuses nothing: this check declines
    # what it is sure about, never what it cannot see.
    assert code_apply._missing_constructor_arguments("no_such_op", {}) == []
    assert code_apply._missing_constructor_arguments("custom_code", {}) == []


def test_the_refusal_reaches_the_caller_and_writes_nothing():
    """The check above, through the real apply path.

    Asserting the helper alone proved nothing about the product: the refusal has to
    reach `apply_operations`, come back as a refusal, and leave the source
    untouched. Both halves are here, because a check that refuses everything would
    pass a test that only looks at the refusing case.
    """
    source = (FIXTURE_REPO / "models" / "resid.py").read_text(encoding="utf-8")
    parsed = CodeParserService().parse_pytorch_code(
        CodeParseRequest(code=source, file="models/resid.py")
    )
    graph = next(item for item in parsed["graphs"] if item["name"] == "Net")["graph"]
    projected = projection.project(
        {
            "nodes": [n.model_dump() for n in graph.nodes],
            "edges": [e.model_dump() for e in graph.edges],
        },
        cls="Net", rel_file="models/resid.py", source=source,
    )

    refused = code_apply.apply_operations(
        source,
        projected.spanmap,
        [
            {"op": "add_layer", "op_id": "a1", "node_id": "n1", "layer_op": "linear", "params": {}},
            {"op": "connect", "op_id": "c1", "source_node_id": "l1", "target_node_id": "n1"},
        ],
        cls="Net", file="models/resid.py",
    )
    assert refused.refused, "a constructor that cannot be built was written"
    assert "in_features" in refused.refused[0]["message"]
    assert refused.source == source, "the file changed despite the refusal"

    # The same op with its arguments is written, so this refuses the broken case
    # rather than the operation.
    written = code_apply.apply_operations(
        source,
        projected.spanmap,
        [
            {"op": "add_layer", "op_id": "a1", "node_id": "n1", "layer_op": "linear",
             "params": {"in_features": 8, "out_features": 8}},
            {"op": "connect", "op_id": "c1", "source_node_id": "l1", "target_node_id": "n1"},
        ],
        cls="Net", file="models/resid.py",
    )
    assert written.refused == [], written.refused
    assert "nn.Linear(in_features=8, out_features=8)" in written.source
