"""Actual captured records cross the live document boundary without filtering."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from services.semantic.document import capture_document
from services.semantic.projection import project_capture
from tests.capture.test_semantic_recorder import context, capture


ROOT = Path(__file__).parent.resolve().parents[2]
FIXTURE = ROOT / "canvas/src/test/fixtures/canvas/capture-document.json"


def document(record, **kwargs):
    return capture_document(record, name="Model", graph_id="captured-model", model_file="model.py",
        source_digests={"model.py": "a" * 64}, source_graph={"nodes": [], "edges": []}, source_spans={}, **kwargs)[0]


def fixture_document():
    return document(json.loads((ROOT / "tests/fixtures/contracts/semantic-capture.json").read_text()))


def test_python_producer_matches_shared_browser_fixture():
    assert fixture_document().model_dump(mode="json", by_alias=True) == json.loads(FIXTURE.read_text())


def test_boundary_names_use_recorded_keys_without_rewriting_ports_or_values(context):
    class Net(nn.Module):
        def forward(self, x, *, attention_mask):
            return {"patch_tokens": x + 1, "pooled": (x.mean(1), attention_mask), "extra.key": {2: x}}

    _, record = capture(Net(), context, kwargs={"attention_mask": torch.ones(2, 4)})
    original = record.model_dump_json()
    result = document(record)
    nodes = {node.metadata["captureRef"]["kind"]: node for node in result.semantic.nodes}
    assert [p["displayName"] for p in nodes["input"].metadata["capturePorts"]] == ["1", "attention_mask"]
    expected = ["patch_tokens", "pooled[0]", "pooled[1]", '["extra.key"][2]']
    assert [p["displayName"] for p in nodes["output"].metadata["capturePorts"]] == expected
    assert [p["displayName"] for p in nodes["invocation"].metadata["capturePorts"] if p["direction"] == "out"] == expected
    projected = {p.id: p.model_dump(mode="json") for p in project_capture(record).ports}
    for node in result.semantic.nodes:
        for port in node.metadata["capturePorts"]:
            assert all(port[key] == value for key, value in projected[port["id"]].items())
    assert record.model_dump_json() == original


def test_slice_groups_use_exact_source_expressions_not_just_the_same_line(tmp_path):
    from tests.source_review.test_semantic_source_bindings import prepared, run

    source = '''import torch
class Model(torch.nn.Module):
    def forward(self, x):
        return x[:, :2][:, :1]
'''
    namespace, context = prepared(tmp_path, source)
    record = run(namespace["Model"](), context)
    options = dict(name="Model", graph_id="slices", model_file="model.py",
        source_digests={file.path: file.digest for file in context.source.files},
        source_graph={"nodes": [], "edges": []}, source_spans={})
    result = capture_document(record, **options, source_texts={"model.py": source})[0]
    slices = [node for node in result.semantic.nodes if node.op == "aten.slice.Tensor"]
    expressions = [node.metadata["captureSourceExpression"] for node in slices]
    # Torch 2.14 drops a slice that keeps the whole dimension, so `x[:, :2][:, :1]`
    # records the two dim-1 slices and not the identity dim-0 slices. The two
    # source subscripts on this one line still have to stay distinct.
    assert [node.metadata["captureSliceAxis"] for node in slices] == [1, 1]
    assert expressions[0] != expressions[1]
    for sources in (None, {"model.py": source.replace(":2", ":3")}):
        unverified = capture_document(record, **options, source_texts=sources)[0]
        assert all("captureSourceExpression" not in node.metadata for node in unverified.semantic.nodes)


def test_model_edits_and_helper_hint_survive_unused_files_exceeding_source_edit_budget(tmp_path):
    from services.semantic.source_bindings import CapturedSourceEvidence
    from services.semantic.source_edits import CapturedEditResolver
    from services.validation.source import MAX_FILES
    from tests.source_review.test_semantic_source_bindings import prepared, run

    source = '''import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = torch.nn.Linear(4, 3)
    def prepare_tokens(self, x):
        return self.patch_embed(x)
    def forward(self, x):
        return self.prepare_tokens(x)
'''
    namespace, context = prepared(tmp_path, source)
    model_file = context.source.files[0]
    sources = {"model.py": source}
    unrelated = []
    for index in range(MAX_FILES):
        name, text = f"aaa_unused_{index:03}.py", "unused = True\n"
        (tmp_path / name).write_text(text)
        sources[name] = text
        unrelated.append(model_file.model_copy(update={"path": name, "digest": hashlib.sha256(text.encode()).hexdigest()}))
    snapshot = context.source.model_copy(update={"files": [*unrelated, model_file]})
    evidence = CapturedSourceEvidence(snapshot, tmp_path)
    with evidence.construction.observe():
        model = namespace["Model"]()
    record = run(model, replace(context, source=snapshot, source_evidence=evidence))
    module = next(module for module in record.modules if module.paths == ["patch_embed"])
    call = next(call for call in record.invocations if call.moduleId == module.id)
    assert call.sourceBindingIds, "The selected model must not lose its call sites behind unrelated files."
    assert len(evidence.files) == MAX_FILES  # Keep the original verification budget.
    assert set(CapturedEditResolver(record, sources).sources) == {"model.py"}
    options = dict(name="Model", graph_id="helper-hints", model_file="model.py",
        source_digests={file.path: file.digest for file in snapshot.files},
        source_graph={"nodes": [], "edges": []}, source_spans={})
    result, spans = capture_document(record, **options, source_texts=sources)
    node = next(node for node in result.semantic.nodes if node.metadata.get("captureLabel") == "patch_embed")
    assert node.params["_out_shape"] == [2, 3]
    span = spans[node.node_id]
    assert {key: span[key] for key in ("file", "start", "end", "kind")} == {
        "file": "model.py", "start": [7, 15], "end": [7, 34], "kind": "call"}
    assert span["file_digest"] == model_file.digest
    assert node.metadata.get("sourceEditLimitation") != "source_limit"
    assert node.metadata["sourceEditCapabilities"]
    stale = {**sources, "model.py": source.replace("self.patch_embed(x)", "self.patch_embed(x + 1)")}
    _, stale_spans = capture_document(record, **options, source_texts=stale)
    assert node.node_id not in stale_spans, "A stale buffer must never receive an observed hint."


def test_every_original_node_port_and_connection_survives_document_handoff(context):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.Sequential(nn.Linear(4, 4), nn.GELU())
        def forward(self, x):
            a = self.block(x) + 1
            return a + a
    _, record = capture(Net(), context)
    before = record.model_dump_json()
    projected = project_capture(record)
    result = document(record)
    assert before == record.model_dump_json()
    assert {node.node_id for node in result.semantic.nodes} == {node.id for node in projected.nodes}
    assert {edge.edge_id for edge in result.semantic.edges} == {edge.id for edge in projected.edges}
    assert sum(len(node.metadata["capturePorts"]) for node in result.semantic.nodes) == len(projected.ports)
    for node in result.semantic.nodes:
        ref = node.metadata["captureRef"]
        assert ref["captureId"] == record.id and ref["sourceId"] == record.source.id
        assert node.in_ports == [port["id"] for port in node.metadata["capturePorts"] if port["direction"] == "in"]
        assert node.out_ports == [port["id"] for port in node.metadata["capturePorts"] if port["direction"] == "out"]
    root = next(node for node in result.semantic.nodes if node.metadata["captureRef"]["semanticId"] == record.rootInvocationId)
    assert root.node_id not in result.view.collapsed_modules
    assert root.metadata["capturedOperationCount"] == len(record.operations)
    assert len(result.view.collapsed_modules) == len(record.invocations) - 1
    assert "source" not in result.semantic.metadata["capture"]  # full identity stays in the retained capture


def test_large_measured_graph_crosses_document_and_wire_without_dropping_uses(context):
    from services.ide.local_browser_wire import decode_message, encode_message, MAX_WIRE_BYTES
    from services.semantic.document import validate_document_projection

    class Net(nn.Module):
        def forward(self, x):
            for _ in range(500):
                x = torch.stack([x] * 12).mean(dim=0)
            return x

    context.limits.operations = 2000
    context.limits.values = 2000
    _, record = capture(Net(), context)
    assert record.status == "complete"
    assert len(record.model_dump_json().encode()) < 4 * 1024 * 1024
    result = document(record)
    assert len(result.model_dump_json().encode()) > 6 * 1024 * 1024  # formerly refused here
    validate_document_projection(result.semantic)
    encoded = encode_message(result.model_dump(mode="json", by_alias=True))
    assert len(encoded.encode()) < MAX_WIRE_BYTES
    decoded = decode_message(encoded)
    assert decoded == result.model_dump(mode="json", by_alias=True)
    assert len(decoded["semantic"]["edges"]) == len(project_capture(record).edges)


def test_typed_multi_input_output_and_scalar_shapes_are_not_first_tensor_summaries(context):
    class Net(nn.Module):
        def forward(self, mel, tokens):
            return mel.sum(), tokens + 1
    _, record = capture(Net(), context, args=(torch.ones(1, 80, 32), torch.zeros(1, 4, dtype=torch.long)))
    result = document(record)
    input_node = next(node for node in result.semantic.nodes if node.kind == "input")
    output_node = next(node for node in result.semantic.nodes if node.kind == "output")
    assert "_out_shape" not in input_node.params and "_in_shape" not in output_node.params
    assert [(port["shape"], port["dtype"]) for port in input_node.metadata["capturePorts"]] == [
        ([1, 80, 32], "torch.float32"), ([1, 4], "torch.int64")]
    assert [port["shape"] for port in output_node.metadata["capturePorts"]] == [[], [1, 4]]


def test_operation_cards_keep_measured_tensor_ports_when_literals_prevent_single_input_summary(context):
    class Net(nn.Module):
        def forward(self, x):
            return x.reshape(4, 2).transpose(0, 1).clone(memory_format=torch.contiguous_format)
    _, record = capture(Net(), context)
    result = document(record)
    expected = {
        "aten.view.default": ([2, 4], [4, 2]),
        "aten.transpose.int": ([4, 2], [2, 4]),
        "aten.clone.default": ([2, 4], [2, 4]),
    }
    for target, (input_shape, output_shape) in expected.items():
        node = next(node for node in result.semantic.nodes if node.op == target)
        assert "_in_shape" not in node.params  # The card must read actual tensor ports.
        assert [p["shape"] for p in node.metadata["capturePorts"]
                if p["direction"] == "in" and p["argumentKind"] == "value"] == [input_shape]
        assert node.params["_out_shape"] == output_shape


def test_repeated_shared_calls_have_distinct_readable_labels(context):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.GELU()
            self.alias = self.act
        def forward(self, x):
            return self.alias(self.act(x))
    _, record = capture(Net(), context)
    result = document(record)
    shared = [node.metadata["captureLabel"] for node in result.semantic.nodes
              if node.metadata["captureLabel"].startswith("act")]
    assert shared == ["act · shared instance · call 1/2", "act · shared instance · call 2/2"]


def test_partial_failure_keeps_inputs_and_marks_each_failed_call(context):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3, 4)
        def forward(self, x):
            return self.fc(x)
    _, record = capture(Net(), context)
    result = document(record)
    failed_ids = {op.id for op in record.operations if op.status == "failed"} | {
        call.id for call in record.invocations if call.status == "failed"}
    assert {node.metadata["captureRef"]["semanticId"] for node in result.semantic.nodes if node.metadata.get("error")} == failed_ids
    output = next(node for node in result.semantic.nodes if node.kind == "output")
    assert not output.in_ports and "_in_shape" not in output.params
    assert result.semantic.metadata["capture"]["status"] == "partial"
    for node in result.semantic.nodes:
        if node.metadata.get("error"):
            assert "outputTensor" not in node.metadata
            assert node.metadata["captureDiagnostic"] == next(item.model_dump(mode="json")
                for item in record.diagnostics if item.id == node.metadata["captureDiagnostic"]["id"])
        if node.kind == "module":
            assert node.metadata["parameterCount"] == 16
    # A single failed operation propagates through its module callers.
    assert len({node.metadata["captureDiagnostic"]["id"] for node in result.semantic.nodes
                if node.metadata.get("error")}) == 1


def test_only_complete_direct_tensor_outputs_supply_overlay_bytes(context):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ReLU()
        def forward(self, x):
            return self.act(x).sum(), self.act(x[:0])
    _, record = capture(Net(), context)
    graph = document(record).semantic
    for node in graph.nodes:
        ref = node.metadata["captureRef"]
        if ref["semanticId"] == record.rootInvocationId or ref["kind"] == "output":
            assert "outputTensor" not in node.metadata
        if "outputTensor" in node.metadata:
            metric = node.metadata["outputTensor"]
            assert metric["shape"] == node.params["_out_shape"]
            assert metric["dtype"] == "float32"
    calls = [node for node in graph.nodes if node.metadata["captureLabel"].startswith("act")]
    assert [node.metadata["parameterCount"] for node in calls] == [0, 0]
    assert [node.metadata["outputTensor"]["logicalBytes"] for node in calls] == [32, 0]
    scalar = next(node for node in graph.nodes if node.op == "aten.sum.default")
    assert scalar.metadata["outputTensor"] == dict(shape=[], dtype="float32", logicalBytes=4)


def test_older_capture_has_no_guessed_metrics():
    payload = json.loads((ROOT / "tests/fixtures/contracts/semantic-capture.json").read_text())
    for value in payload["values"]:
        value.pop("logicalBytes", None)
    for call in payload["invocations"]:
        call.pop("parameterCount", None)
    graph = document(payload).semantic
    assert all("outputTensor" not in node.metadata and "parameterCount" not in node.metadata for node in graph.nodes)


def test_foreign_capture_and_display_byte_budget_refuse_without_lossy_fallback(monkeypatch):
    payload = json.loads((ROOT / "tests/fixtures/contracts/semantic-capture.json").read_text())
    before = deepcopy(payload)
    payload["source"]["files"][0]["digest"] = "b" * 64
    with pytest.raises(ValueError, match="snapshots differ"):
        document(payload)
    monkeypatch.setattr("services.semantic.document.MAX_DOCUMENT_BYTES", 1)
    with pytest.raises(ValueError, match="display transport budget"):
        document(before)


def test_captured_document_validation_never_runs_inference_or_runtime(monkeypatch):
    from services.canonical_validation import CanonicalValidationService
    service = CanonicalValidationService()
    monkeypatch.setattr(service, "_infer_shapes", lambda **_: pytest.fail("captured values must never be re-inferred"))
    graph = fixture_document().semantic
    result = service.validate(graph)
    assert result.valid and all(node.status == "ok" for node in result.node_results.values())
    graph.edges[0].target_port = graph.edges[-1].target_port
    result = service.validate(graph)
    assert not result.valid and result.global_errors
    for invalid in ("fabricated", [999, 999], [True]):
        graph = fixture_document().semantic
        graph.nodes[0].params["_out_shape"] = invalid
        result = service.validate(graph)
        assert not result.valid and result.global_errors


def test_real_multifile_api_uses_capture_even_when_legacy_graph_is_empty(tmp_path):
    from api.routers import ide
    from auth import CurrentUser
    from services.ide.multifile import MultifileProjectionRequest
    from services.project_store import ProjectStore
    from tests.source_review.test_semantic_source_join import SOURCE, record
    import hashlib
    source = SOURCE.replace("import torch\n", "import torch\nfrom torch import nn\n").replace("torch.nn.", "nn.")
    digest = hashlib.sha256(source.encode()).hexdigest()
    ide._store = ProjectStore(str(tmp_path / "capture-projection.sqlite3"))
    captured = record(source)
    envelope = dict(protocol_version=1, result=dict(graph=dict(nodes=[], edges=[]),
        diagnostics=dict(semantic_capture=captured)), spans={}, error=None)
    request = MultifileProjectionRequest(protocol_version=1,
        root=dict(file="model.py", cls="Model", source_digest=digest, document_version=1),
        sources=[dict(file="model.py", source=source, digest=digest)], trace_result=envelope)
    projected = ide.multifile_project(request, CurrentUser(uid="capture-tests"))
    nodes = projected["document"]["semantic"]["nodes"]
    assert sum(n["metadata"]["captureRef"]["kind"] == "operation" for n in nodes) == len(captured["operations"])
    assert sum(n["metadata"]["captureRef"]["kind"] == "invocation" for n in nodes) == len(captured["invocations"])
    assert projected["traced"] and projected["trace_error"] is None
    facts = projected["document"]["semantic"]["metadata"]["sourceCorrespondence"]
    assert next(f for f in facts if f["nodeId"] == "block.fc")["status"] == "observed"
    assert next(f for f in facts if f["nodeId"] == "unused")["status"] == "not_observed"
    fc = next(n for n in nodes if n["metadata"]["captureLabel"] == "block.fc")
    assert projected["spanmap"][fc["node_id"]]["file"] == "model.py"
    captured["source"]["files"][0]["digest"] = "b" * 64
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as refused:
        ide.multifile_project(request, CurrentUser(uid="capture-tests"))
    assert refused.value.detail["error"] == "trace_rejected"


def test_registered_capture_paths_remain_mappable_without_lowered_source_nodes():
    from tests.source_review.test_semantic_source_join import SOURCE, DIGEST, record
    document, _ = capture_document(record(), name="Model", graph_id="captured-model",
        model_file="model.py", source_digests={"model.py": DIGEST},
        source_graph={"nodes": [], "edges": []}, source_spans={}, source_texts={"model.py": SOURCE})
    facts = {fact["nodeId"]: fact for fact in document.semantic.metadata["sourceCorrespondence"]}
    assert facts["block.fc"]["basis"] == "registered_path"
    assert facts["block.fc"]["status"] == "observed"
    assert facts["block.fc"]["invocationIds"]
    assert facts["unused"]["status"] == "not_observed"
    assert not facts["unused"]["invocationIds"]


def test_single_source_projection_maps_failure_to_canonical_call():
    from services.ide.projection import project
    from tests.source_review.test_semantic_source_join import SOURCE, record
    captured = record(args=(torch.ones(2, 3),))
    envelope = dict(protocol_version=1, result=dict(graph=dict(nodes=[], edges=[]),
        diagnostics=dict(semantic_capture=captured)), spans={},
        error=dict(type="RuntimeError", message="Shape mismatch", node_id="block.fc", frames=[]))
    result = project(dict(nodes=[], edges=[]), envelope, cls="Model", rel_file="model.py", source=SOURCE)
    failed = next(n for n in result.document.semantic.nodes if n.node_id == result.error["node_id"])
    assert failed.metadata["captureRef"]["kind"] == "operation"
    assert failed.metadata["error"] and failed.params["_shape_provenance"] == "failed"


def test_declaration_hints_keep_their_anchor_when_capture_navigation_uses_the_call(tmp_path):
    from hashlib import sha256
    from tests.source_review.test_semantic_source_bindings import prepared, run

    source = """import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)
    def forward(self, x):
        return self.fc(x)
"""
    namespace, context = prepared(tmp_path, source)
    record = run(namespace["Model"](), context)
    declaration = dict(file="model.py", start=[5, 8], end=[5, 38], kind="decl")
    result, spans = capture_document(record, name="Model", graph_id="captured-model",
        model_file="model.py", source_digests={"model.py": sha256(source.encode()).hexdigest()},
        source_graph={"nodes": [], "edges": []}, source_spans={"fc": declaration},
        source_texts={"model.py": source})
    fc = next(node for node in result.semantic.nodes if node.metadata["captureLabel"] == "fc")
    assert fc.metadata["declarationSpan"] == declaration
    assert spans[fc.node_id]["kind"] == "call" and spans[fc.node_id]["start"][0] == 7
    assert fc.params["_out_shape"] == [2, 4]
    assert all("declarationSpan" not in node.metadata for node in result.semantic.nodes if node is not fc)


def test_parameter_navigation_uses_registered_identity_and_refuses_conflicting_aliases(tmp_path):
    from tests.source_review.test_semantic_source_bindings import prepared, run

    source = '''import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(4))
        self.tied = self.scale
    def forward(self, x):
        return x * self.scale
'''
    namespace, context = prepared(tmp_path, source)
    record = run(namespace["Model"](), context)
    declaration = dict(file="model.py", start=[5, 8], end=[5, 52], kind="decl")
    options = dict(name="Model", graph_id="state-source", model_file="model.py",
        source_digests={"model.py": hashlib.sha256(source.encode()).hexdigest()},
        source_graph={"nodes": [], "edges": []}, source_texts={"model.py": source})
    result, spans = capture_document(record, **options, source_spans={"scale": declaration, "tied": declaration})
    parameter = next(node for node in result.semantic.nodes if node.metadata.get("sourceKind") == "parameter")
    assert spans[parameter.node_id] == declaration
    result, spans = capture_document(record, **options, source_spans={"scale": declaration,
        "tied": dict(declaration, start=[6, 8], end=[6, 30])})
    assert parameter.node_id not in spans
