"""Coverage comes from canonical observations, never from rendered node depth."""
import hashlib

import pytest
import torch

from services.ide.projection import _merge, TraceRejected
from services.ingest.trace import trace_model
from services.semantic.contracts import CaptureLimits, SourceSnapshot, SourceCorrespondence
from services.semantic.recorder import CaptureContext
from services.semantic.source_join import CaptureJoin


SOURCE = '''import torch
class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)
        self.relu = torch.nn.ReLU()
    def forward(self, x): return self.relu(self.fc(x))
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block()
        self.unused = torch.nn.Linear(4, 4)
        self.offset = torch.nn.Parameter(torch.ones(4))
        self.untouched = torch.nn.Parameter(torch.zeros(4))
    def forward(self, x): return self.block(x) + self.offset
'''
DIGEST = hashlib.sha256(SOURCE.encode()).hexdigest()


def node(id, op="linear", kind="class", **params):
    return dict(id=id, type="layerNode", data=dict(layerType=dict(id=op, name=op),
        parameters=dict(_kind=kind, **params)))


def record(source=SOURCE, *, limit=None, args=None):
    namespace = {"__name__": "joined_model"}
    exec(compile(source, "/workspace/model.py", "exec"), namespace)
    context = CaptureContext(source=SourceSnapshot(version=1, id="source:0", workspaceSessionId="session:0",
        workspaceId="workspace:0", modelId="model:0", modelFile="model.py", className="Model", generation=0,
        files=[dict(path="model.py", digest=hashlib.sha256(source.encode()).hexdigest(), documentVersion=1, role="source")],
        dependencyScope="complete", configDigest=None, environmentDigest="b" * 64, sourceMode="saved"),
        plan_digest="c" * 64, attempt_id="attempt:0",
        limits=CaptureLimits(operations=1000, invocations=limit or 1000, values=1000, bytes=4 * 1024 * 1024))
    traced = trace_model(namespace["Model"](), args or (torch.ones(2, 4),), tier="real", semantic_context=context)
    assert traced.semantic_capture is not None, traced.semantic_error
    return traced.semantic_capture


def join(capture, source=SOURCE):
    return CaptureJoin(capture, model_file="model.py", class_name="Model",
                       source_digests={"model.py": hashlib.sha256(source.encode()).hexdigest()})


def test_calls_below_render_depth_and_uncalled_instances_keep_different_coverage():
    capture = record()
    joined = join(capture)
    deep = joined.decorate(node("block.fc", _shape_provenance="unreached"))
    params = deep["data"]["parameters"]
    assert params["_shape_provenance"] == "observed"
    assert params["_in_shape"] == params["_out_shape"] == "2x4"
    assert params["_capture_coverage"]["invocationIds"]
    assert joined.correspondence(node("unused")).status == "not_observed"
    assert joined.correspondence(node("not_constructed")).status == "static_only"
    assert joined.correspondence(node("python@Model#0", "python", "statement")).status == "unresolved"
    assert capture["status"] == "complete"


def test_partial_limits_cannot_establish_that_a_missing_call_was_not_executed():
    joined = join(record(limit=1))
    for path in ("block", "block.fc", "unused", "missing"):
        fact = joined.correspondence(node(path))
        assert fact.status == "capture_limited", path
        assert fact.reason == "invocation_limit"
        assert not fact.invocationIds


def test_parameter_use_is_value_evidence_not_an_invocation():
    joined = join(record())
    for name, used in (("offset", True), ("untouched", False)):
        decorated = joined.decorate(node(name, "parameter", _shape_provenance="unreached"))
        params = decorated["data"]["parameters"]
        fact = SourceCorrespondence.model_validate(params["_capture_coverage"])
        assert fact.status == "state" and fact.moduleId is None and not fact.invocationIds
        assert bool(fact.stateValueIds) is used
        assert (params.get("_out_shape") == "4") is used


def test_shared_instance_is_not_mistaken_for_a_unique_source_call():
    source = SOURCE.replace("self.unused =", "self.alias = self.block\n        self.unused =").replace(
        "self.block(x) + self.offset", "self.alias(self.block(x)) + self.offset")
    joined = join(record(source), source)
    first, alias = (joined.correspondence(node(path)) for path in ("block", "alias"))
    assert first.basis == alias.basis == "shared_instance"
    assert first.moduleId == alias.moduleId
    assert first.invocationIds == alias.invocationIds and len(first.invocationIds) == 2


def test_multi_output_and_differing_calls_never_inherit_a_first_tensor_shape():
    source = SOURCE.replace("return self.relu(self.fc(x))", "return (self.fc(x), x)").replace(
        "return self.block(x) + self.offset", "return self.block(x)[0] + self.offset")
    joined = join(record(source), source)
    decorated = joined.decorate(node("block", _shape_provenance="observed", _out_shape="2x4"))
    assert "_out_shape" not in decorated["data"]["parameters"]
    assert decorated["data"]["parameters"]["_capture_coverage"]["status"] == "observed"
    source = SOURCE.replace("return self.block(x) + self.offset", "a = self.block(x); return self.block(x[:1])")
    joined = join(record(source), source)
    params = joined.decorate(node("block"))["data"]["parameters"]
    assert params["_call_sites"] == 2
    assert "_in_shape" not in params and "_out_shape" not in params


def test_failed_module_keeps_its_measured_input_without_inventing_output():
    joined = join(record(args=(torch.ones(2, 3),)))
    fact = joined.correspondence(node("block.fc"))
    assert fact.status == "failed" and fact.invocationIds
    params = joined.decorate(node("block.fc", _out_shape="2x4", _declared_out_shape="2,4"))["data"]["parameters"]
    assert params["_in_shape"] == "2x3" and "_out_shape" not in params
    assert joined.correspondence(node("block.relu")).status == "not_observed"


def test_scalar_shape_and_existing_exact_output_metrics_survive_without_guessing():
    joined = join(record())
    item = node("block.fc")
    item["data"]["outputTensor"] = dict(shape=[2, 4], dtype="float32", logicalBytes=32)
    assert joined.decorate(item)["data"]["outputTensor"] == item["data"]["outputTensor"]
    item["data"]["outputTensor"]["dtype"] = "float64"
    assert "outputTensor" not in joined.decorate(item)["data"]
    source = SOURCE.replace("return self.relu(self.fc(x))", "return self.fc(x).sum()")
    scalar = join(record(source), source).decorate(node("block"))
    assert scalar["data"]["parameters"]["_out_shape"] == []


def test_join_refuses_foreign_source_and_malformed_canonical_evidence():
    captured = record()
    with pytest.raises(ValueError, match="snapshots differ"):
        join(captured, SOURCE + "# changed\n")
    with pytest.raises(ValueError, match="snapshots differ"):
        CaptureJoin(captured, model_file="model.py", class_name="Other", source_digests={"model.py": DIGEST})
    captured["invocations"][0]["moduleId"] = "absent"
    with pytest.raises(ValueError, match="known module"):
        join(captured)


def test_legacy_merge_joins_all_known_source_modules_even_when_trace_projection_is_shallow():
    capture = record()
    ast_graph = dict(nodes=[node("block"), node("block.fc"), node("block.relu"), node("unused"),
        node("offset", "parameter"), node("untouched", "parameter")], edges=[])
    trace = dict(protocol_version=1, result=dict(graph=dict(nodes=[node("block", _out_shape="2x4")], edges=[]),
        diagnostics=dict(semantic_capture=capture)), spans={}, error=None)
    graph, _, error = _merge(ast_graph, {}, trace, cls="Model", rel_file="model.py", source_digests={"model.py": DIGEST})
    assert error is None
    params = {n["id"]: n["data"]["parameters"] for n in graph["nodes"]}
    assert params["block.fc"]["_shape_provenance"] == "observed"
    assert params["block.relu"]["_shape_provenance"] == "observed"
    assert params["unused"]["_shape_provenance"] == "not_observed"
    assert params["offset"]["_shape_provenance"] == "state"
    assert not any(p["_shape_provenance"] == "unreached" for p in params.values())
    with pytest.raises(TraceRejected, match="saved source"):
        _merge(ast_graph, {}, trace, cls="Model", rel_file="model.py", source_digests={"model.py": "b" * 64})


def test_captured_child_cannot_replace_its_distinct_registered_parent():
    capture = record()
    trace = dict(protocol_version=1, result=dict(graph=dict(nodes=[node("block.fc")], edges=[]),
        diagnostics=dict(semantic_capture=capture)), spans={}, error=None)
    graph, _, _ = _merge(dict(nodes=[node("block"), node("block.fc")], edges=[]), {}, trace,
        cls="Model", rel_file="model.py", source_digests={"model.py": DIGEST})
    facts = {n["id"]: n["data"]["parameters"]["_capture_coverage"] for n in graph["nodes"]}
    assert facts["block"]["moduleId"] != facts["block.fc"]["moduleId"]
    assert facts["block"]["status"] == facts["block.fc"]["status"] == "observed"


def test_source_card_validation_never_turns_absent_evidence_into_a_shape_or_failed_run():
    from services.ingest.document import to_document
    from services.canonical_validation import CanonicalValidationService
    joined = join(record())
    nodes = [node("input", "input", shape="2,4"),
             joined.decorate(node("block.fc", in_features=4, out_features=4)),
             joined.decorate(node("unused", in_features=4, out_features=4)),
             node("output", "output")]
    edges = [dict(id=f"e{i}", source=source, target=target) for i, (source, target) in enumerate(
        (("input", "block.fc"), ("block.fc", "unused"), ("unused", "output")))]
    document = to_document("Model", dict(nodes=nodes, edges=edges))
    result = CanonicalValidationService().validate(document.semantic)
    assert result.node_results["block.fc"].output_shape == [2, 4]
    unused = result.node_results["unused"]
    assert unused.status == "warning" and unused.output_shape is None
    assert "No invocation" in unused.message and "run failed before" not in unused.message
    assert result.node_results["output"].output_shape is None


def test_observed_source_shapes_skip_derivation_and_runtime_fallback(monkeypatch):
    from services.ingest.document import to_document
    from services.canonical_validation import CanonicalValidationService
    joined = join(record())
    observed = joined.decorate(node("block.fc", "unknown_custom_op"))
    nodes = [node("input", "input", shape="2,4"), observed, node("output", "output")]
    edges = [dict(id=f"e{i}", source=source, target=target) for i, (source, target) in enumerate(
        (("input", "block.fc"), ("block.fc", "output")))]
    service = CanonicalValidationService()
    monkeypatch.setattr(service, "_infer_runtime_fallback", lambda *_: pytest.fail("joined source must never run fallback"))
    result = service.validate(to_document("Model", dict(nodes=nodes, edges=edges)).semantic)
    assert result.node_results["block.fc"].output_shape == [2, 4]
    assert result.node_results["output"].output_shape == [2, 4]


def test_source_correspondence_rejects_false_call_and_state_combinations():
    fact = join(record()).correspondence(node("block.fc")).model_dump(mode="json")
    for updates in (dict(invocationIds=[]), dict(moduleId=None), dict(stateValueIds=["tensor0"]),
                    dict(status="state"), dict(reason="operation_limit"),
                    dict(invocationIds=fact["invocationIds"] * 2)):
        with pytest.raises(ValueError):
            SourceCorrespondence.model_validate({**fact, **updates})
