"""Canonical records come from the real graph dispatch/module observer."""
from dataclasses import replace
import gc
import weakref

import pytest
import torch
from torch import nn

from services.ingest.trace import trace_model
from services.semantic.contracts import CaptureLimits, CaptureSnapshot, SourceSnapshot
from services.semantic.recorder import CaptureContext, SemanticRecorder


@pytest.fixture
def context():
    return CaptureContext(source=SourceSnapshot(version=1, id="source:0", workspaceSessionId="session:0",
        workspaceId="workspace:0", modelId="model:0", modelFile="model.py", className="Model", generation=0,
        files=[dict(path="model.py", digest="a" * 64, documentVersion=1, role="source")],
        dependencyScope="complete", configDigest=None, environmentDigest="b" * 64, sourceMode="saved"),
        plan_digest="c" * 64, attempt_id="attempt:0",
        limits=CaptureLimits(operations=1000, invocations=1000, values=1000, bytes=4 * 1024 * 1024))


def capture(model, context, args=None, kwargs=None, **options):
    result = trace_model(model, args if args is not None else (torch.ones(2, 4),), kwargs,
                         tier="real", semantic_context=context, record_evidence=True, **options)
    assert result.semantic_capture is not None, (result.semantic_error, result.error)
    parsed = CaptureSnapshot.model_validate(result.semantic_capture)
    assert parsed.attemptId == context.attempt_id
    assert parsed.source == context.source
    return result, parsed


def test_literals_repeated_operands_and_generic_functional_ops(context):
    class Net(nn.Module):
        def forward(self, h):
            a = torch.nn.functional.gelu(h + 1)
            return a + a
    result, record = capture(Net(), context)
    assert result.error is None
    assert record.status == "complete"
    assert [op.target for op in record.operations] == ["aten.add.Tensor", "aten.gelu.default", "aten.add.Tensor"]
    add1, gelu, add2 = record.operations
    assert add1.arguments.args[1].kind == "int" and add1.arguments.args[1].value == "1"
    assert gelu.arguments.args[0] == add1.output
    assert add2.arguments.args[0] == add2.arguments.args[1] == gelu.output
    assert add2.schemaSignature.startswith("aten::add.Tensor")
    assert record.output == add2.output


def test_unobserved_legacy_dispatch_cannot_claim_successful_capture(context, monkeypatch):
    # PyTorch 2.0 dispatches through this alias instead of the modern wrapper.
    original = nn.Module._call_impl
    monkeypatch.setattr(nn.Module, "__call__", original)
    class Net(nn.Module):
        def forward(self, x):
            return x + 1
    model = Net()
    result, record = capture(model, context)
    assert nn.Module.__call__ is original
    assert "_call_impl" not in model.__dict__
    assert record.rootInvocationId is None
    assert record.status == "failed"
    assert result.semantic_error == "recorder_error"
    assert "could not be recorded" in result.error


def test_private_worker_rejects_legacy_dispatch_before_loading_capture(monkeypatch):
    from services.ingest import bootstrap
    monkeypatch.setattr(nn.Module, "__call__", nn.Module._call_impl)
    with pytest.raises(bootstrap.CaptureRuntimeUnsupported, match="legacy module call dispatcher"):
        bootstrap.load_worker()


def test_canonical_budget_cannot_leave_an_unbounded_legacy_stream(context, monkeypatch):
    from services.ingest.trace import _Provenance

    def legacy(*args, **kwargs):
        pytest.fail("canonical dispatch must not enter the legacy operation recorder")

    monkeypatch.setattr(_Provenance, "_legacy_dispatch", legacy)
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.relu(x + 1))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = Block()

        def forward(self, x):
            for _ in range(100):
                x = self.block(x)
            return x

    limited = replace(context, limits=CaptureLimits(operations=8, invocations=8, values=32, bytes=100000))
    result, record = capture(Net(), limited)
    assert result.error is None
    assert record.status == "partial" and "operation_limit" in record.coverage.omissions
    assert len(record.operations) <= 8 and len(record.invocations) <= 8
    assert result.ops == [] and result.reuse_scopes == [] and result.input_ids == []
    assert result.evidence["outputs"]["allFinite"]


@pytest.mark.parametrize("snapshot_available", [True, False])
def test_canonical_ingestion_never_materializes_or_falls_back_to_legacy_graph(context, monkeypatch, snapshot_available):
    from services.ingest import service

    def legacy(*args, **kwargs):
        pytest.fail("canonical ingestion must not build the legacy tree or folded dataflow")

    monkeypatch.setattr(service, "build_tree", legacy)
    monkeypatch.setattr(service, "resolve_dataflow", legacy)
    if not snapshot_available:
        monkeypatch.setattr(SemanticRecorder, "snapshot", lambda self: None)
    graph, diagnostics, error = service._introspect_and_trace("Net", nn.ReLU(), 3,
        ((torch.ones(2, 4),), {}), {}, tier="real", strict_recipe=True,
        preserve_mode=True, semantic_context=context)
    assert error is None and graph == {"nodes": [], "edges": []}
    assert diagnostics.run_evidence["outputs"]["allFinite"]
    if snapshot_available:
        assert diagnostics.semantic_capture["status"] == "complete"
        assert diagnostics.aten_ops == 1
    else:
        assert diagnostics.semantic_capture is None
        assert diagnostics.semantic_capture_error == "byte_limit"


@pytest.mark.parametrize("tensor", [torch.ones(()), torch.empty(0, 3),
    torch.ones(2, 4, dtype=torch.float64).t(), torch.ones(1).expand(3, 4)])
def test_logical_bytes_measure_scalar_empty_noncontiguous_and_expanded_outputs(context, tensor):
    _, record = capture(nn.Identity(), context, args=(tensor,))
    value = next(value for value in record.values if value.id == record.output.id)
    assert value.logicalBytes == tensor.numel() * tensor.element_size()
    assert value.shape == list(tensor.shape)
    assert record.invocations[0].parameterCount == 0
    assert record.status == "complete"


def test_parameter_counts_are_unique_recursive_and_measured_per_call_entry(context):
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2))
            self.register_buffer("ignored", torch.ones(9))
        def forward(self, x):
            return x + self.weight.sum()
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = Block()
            self.alias = self.block
            self.next_weight = nn.Parameter(torch.ones(3))
            self.duplicate = self.next_weight
        def forward(self, x):
            x = self.block(x)
            self.block.weight = self.next_weight
            return self.alias(x)
    _, record = capture(Net(), context)
    assert record.invocations[0].parameterCount == 5
    assert [call.parameterCount for call in record.invocations[1:]] == [2, 3]
    assert len({call.moduleId for call in record.invocations[1:]}) == 1


def test_parameter_metric_budget_does_not_truncate_operation_capture(context, monkeypatch):
    monkeypatch.setattr("services.semantic.metrics.MAX_PARAMETER_ITEMS", 0)
    _, record = capture(nn.GELU(), context)
    assert record.invocations[0].parameterCount is None
    assert record.invocations[0].parameterCountUnavailable == "item_limit"
    assert record.status == "complete" and len(record.operations) == 1


def test_parameter_counter_avoids_user_iterators_cycles_and_partial_counts():
    from services.semantic.metrics import ParameterCounter
    class Net(nn.Module):
        def parameters(self, *args, **kwargs):
            pytest.fail("metadata must not execute an overridden module iterator")
        def children(self):
            pytest.fail("metadata must not execute an overridden module iterator")
    model = Net()
    model.weight = nn.Parameter(torch.ones(3))
    model.same = model.weight
    model.loop = model
    assert ParameterCounter().measure(model) == dict(parameterCount=3)
    model.lazy = nn.parameter.UninitializedParameter()
    assert ParameterCounter().measure(model) == dict(parameterCountUnavailable="unsupported_type")
    counter = ParameterCounter()
    counter.remaining = 2
    assert counter.measure(model) == dict(parameterCountUnavailable="item_limit")
    assert counter.remaining == 0
    empty = nn.Module()
    for i in range(10):
        empty.add_module(str(i), None)
    counter = ParameterCounter()
    counter.remaining = 2
    assert counter.measure(empty) == dict(parameterCountUnavailable="item_limit")


def test_ordered_cat_nested_kwargs_and_multi_output_keep_paths(context):
    class Net(nn.Module):
        def forward(self, x, *, bundle):
            return {"joined": torch.cat([bundle["skip"], x], dim=1), "pair": (x, None)}
    _, record = capture(Net(), context, kwargs={"bundle": {"skip": torch.zeros(2, 3)}})
    cat = record.operations[0]
    assert cat.arguments.args[0].items == [record.inputs.kwargs["bundle"].entries[0].value, record.inputs.args[0]]
    assert cat.arguments.args[1].value == "1"
    assert record.output.entries[0].value == cat.output
    assert record.output.entries[1].value.items[0] == record.inputs.args[0]
    assert record.output.entries[1].value.items[1].kind == "none"
    assert record.status == "complete"


def test_shared_module_aliases_repeated_calls_and_identical_instances_are_distinct(context):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.ReLU()
            self.alias = self.first
            self.second = nn.ReLU()
        def forward(self, x):
            return self.second(self.alias(self.first(x)))
    _, record = capture(Net(), context)
    shared = next(m for m in record.modules if "first" in m.paths)
    second = next(m for m in record.modules if "second" in m.paths)
    assert shared.paths == ["first", "alias"]
    assert shared.id != second.id and shared.definitionId == second.definitionId
    calls = [c for c in record.invocations if c.moduleId == shared.id]
    assert len(calls) == 2 and calls[0].id != calls[1].id
    assert record.operations[0].invocationId == calls[0].id
    assert record.operations[1].invocationId == calls[1].id
    assert record.status == "complete"


def test_views_mutations_and_state_values_have_actual_identity(context):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2, 4))
            self.register_buffer("bias", torch.ones(2, 4))
        def forward(self, x):
            h = x + self.weight
            view = h.view(-1)
            h.add_(self.bias)
            return h, view
    _, record = capture(Net(), context)
    assert record.status == "complete"
    values = {v.id: v for v in record.values}
    add = next(op for op in record.operations if op.target == "aten.add_.Tensor")
    before, after = values[add.arguments.args[0].id], values[add.output.id]
    assert after.previousVersion == before.id and after.version == before.version + 1
    assert after.producer.operationId == add.id
    alias = values[record.output.items[1].id]
    assert alias.storageId == after.storageId
    assert alias.producer.operationId == add.id and alias.producer.kind == "effect"
    assert [effect.kind for effect in add.effects] == ["write", "alias_update"]
    assert add.effects[0].after == add.output
    assert add.effects[1].after.id == alias.id
    assert values[record.operations[0].arguments.args[1].id].state[0].name == "weight"
    assert values[add.arguments.args[1].id].state[0].kind == "buffer"


def test_failed_operation_retains_attempted_arguments_and_no_output(context):
    class Net(nn.Module):
        def forward(self, x):
            return (x + 1) @ torch.ones(3, 2)
    result, record = capture(Net(), context)
    assert result.error is not None and record.status == "partial"
    assert record.output is None
    assert record.operations[-1].status == "failed"
    assert record.operations[-1].arguments.args[0] == record.operations[0].output
    assert record.invocations[0].status == "failed"
    assert record.coverage.modules[0].status == "failed"
    diagnostic = next(d for d in record.diagnostics if d.id == record.operations[-1].diagnosticId)
    assert diagnostic.code == "invalid_shape" and diagnostic.expectedShape == [None, 3]
    assert diagnostic.actualShape == [2, 4]
    assert record.invocations[0].diagnosticId == diagnostic.id
    assert {k: v for k, v in diagnostic.model_dump().items() if k != "id"} == {
        k: v for k, v in result.diagnostic.items() if k != "id"}


@pytest.mark.parametrize("wrapped,code", [(True, "invalid_shape"), (False, "execution_failed")])
def test_canonical_wrapped_errors_preserve_only_linked_failure_facts(context, wrapped, code):
    class Net(nn.Module):
        def forward(self, x):
            try:
                x @ torch.ones(3, 2)
            except RuntimeError as error:
                raise ValueError("PRIVATE_ERROR_TEXT") from error if wrapped else None
    result, record = capture(Net(), context)
    diagnostics = {d.id: d for d in record.diagnostics}
    root = diagnostics[record.invocations[0].diagnosticId]
    failed_op = next(op for op in record.operations if op.status == "failed")
    assert diagnostics[failed_op.diagnosticId].code == "invalid_shape"
    assert root.code == code and root.code == result.diagnostic["code"]
    assert "PRIVATE_ERROR_TEXT" not in record.model_dump_json()


def test_caught_error_does_not_become_the_terminal_assertion(context):
    class Net(nn.Module):
        def forward(self, x):
            try:
                x @ torch.ones(3, 2)
            except RuntimeError:
                pass
            assert False, "PRIVATE_ASSERTION"
    result, record = capture(Net(), context)
    diagnostics = {d.id: d for d in record.diagnostics}
    root = diagnostics[record.invocations[0].diagnosticId]
    assert root.code == "assertion_failed" and root.expectedShape is None
    failed_op = next(op for op in record.operations if op.status == "failed")
    assert diagnostics[failed_op.diagnosticId].code == "invalid_shape"
    assert failed_op.diagnosticId != root.id
    assert result.diagnostic["code"] == root.code
    assert "PRIVATE_ASSERTION" not in record.model_dump_json()


def test_canonical_dtype_failure_uses_the_shared_bounded_diagnostic(context):
    class Net(nn.Module):
        def forward(self, x):
            return torch.nn.functional.embedding(x, torch.ones(8, 2))
    result, record = capture(Net(), context)
    diagnostic = next(d for d in record.diagnostics if d.id == record.operations[-1].diagnosticId)
    assert diagnostic.code == "invalid_dtype" and diagnostic.field == ["args", 1]
    assert diagnostic.actualDtype == "torch.float32" and diagnostic.expectedDtype is None
    assert result.diagnostic["actualDtype"] == diagnostic.actualDtype


def test_diagnostic_byte_exhaustion_keeps_a_valid_failure_link(context, monkeypatch):
    original = SemanticRecorder.account
    def account(self, value, reserve=8):
        if isinstance(value, dict) and value.get("id", "").startswith("diagnostic:"):
            self.omit("byte_limit")
            return False
        return original(self, value, reserve)
    monkeypatch.setattr(SemanticRecorder, "account", account)
    class Net(nn.Module):
        def forward(self, x):
            assert False, "PRIVATE_PAYLOAD"
    result, record = capture(Net(), context)
    assert result.error is not None and record.status == "partial"
    assert record.coverage.omissions == ["byte_limit"]
    assert record.invocations[0].diagnosticId == "failure"
    assert record.diagnostics[0].id == "failure" and record.diagnostics[0].code == "execution_failed"


def test_caught_error_valid_none_output_and_dynamic_module(context):
    class Net(nn.Module):
        def forward(self, x):
            try:
                x @ torch.ones(3, 2)
            except RuntimeError:
                nn.Identity()(x)
            return None
    _, record = capture(Net(), context)
    assert record.status == "complete"
    assert record.output.kind == "none"
    dynamic = next(m for m in record.modules if not m.paths)
    assert any(c.moduleId == dynamic.id for c in record.invocations)
    assert any(op.status == "failed" for op in record.operations)


@pytest.mark.parametrize("limit,amount,reason,status", [("operations", 1, "operation_limit", "partial"),
    ("invocations", 1, "invocation_limit", "partial"), ("values", 1, "value_limit", "partial"),
    ("bytes", 4096, "byte_limit", "failed"), ("bytes", 6000, "byte_limit", "partial"),
    ("bytes", 8192, "byte_limit", "partial")])
def test_independent_budgets_leave_honest_bounded_records(context, limit, amount, reason, status):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.activation = nn.ReLU()
        def forward(self, x):
            for _ in range(10):
                x = self.activation(x + 1)
            return x
    context = replace(context, limits=context.limits.model_copy(update={limit: amount}))
    result, record = capture(Net(), context)
    if status == "failed":
        assert "could not be recorded" in result.error
    else:
        assert result.error is None
    assert record.status == status
    if status == "failed":
        # Successful execution is distinct from a budget too small to retain
        # even its first call after the immutable envelope/module inventory.
        assert record.rootInvocationId is None and not record.invocations and not record.operations
    assert reason in record.coverage.omissions
    if limit == "bytes":
        assert record.coverage.omissions == [reason], "A byte stop is not an exhausted invocation count."
    assert len(record.operations) <= record.limits.operations
    assert len(record.invocations) <= record.limits.invocations
    assert len(record.values) <= record.limits.values
    assert len(record.model_dump_json().encode()) <= record.limits.bytes


@pytest.mark.parametrize("inplace", [False, True])
def test_tensor_producers_fit_without_per_record_padding_exhausting_the_budget(context, inplace):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.activation = nn.ReLU(inplace=inplace)

        def forward(self, x):
            for _ in range(256):
                x = self.activation(x + 1)
            return x

    bounded = replace(context, limits=context.limits.model_copy(update={"bytes": 1024 * 1024}))
    result, record = capture(Net(), bounded)
    assert result.error is None
    assert record.status == "complete" and not record.coverage.omissions
    assert len(record.operations) == 512 and len(record.invocations) == 257
    assert record.output == record.operations[-1].output
    assert len(record.model_dump_json().encode()) <= bounded.limits.bytes


def test_output_producer_budget_refusal_keeps_a_valid_prefix(context, monkeypatch):
    original = SemanticRecorder.account
    def account(self, value, reserve=8):
        if isinstance(value, dict) and value.get("kind") == "tuple" and reserve > 8:
            self.omit("byte_limit")
            return False
        return original(self, value, reserve)
    monkeypatch.setattr(SemanticRecorder, "account", account)
    class Net(nn.Module):
        def forward(self, x):
            values, indices = torch.sort(x)
            return values, indices
    result, record = capture(Net(), context)
    assert result.error is None and record.status == "partial"
    assert record.coverage.omissions == ["byte_limit"]
    assert record.operations[0].output.kind == "opaque"
    assert all(value.producer is None for value in record.values)


def test_effect_producer_budget_refusal_keeps_a_valid_prefix(context, monkeypatch):
    original = SemanticRecorder.account
    def account(self, value, reserve=8):
        if isinstance(value, dict) and value.get("kind") == "write" and reserve > 8:
            self.omit("byte_limit")
            return False
        return original(self, value, reserve)
    monkeypatch.setattr(SemanticRecorder, "account", account)
    library = torch.library.Library("tensorviz_semantic_budget", "DEF")
    try:
        library.define("bump(Tensor(a!) x) -> ()")
        def bump(x):
            x.add_(1)
        library.impl("bump", bump, "CPU")
        class Net(nn.Module):
            def forward(self, x):
                torch.ops.tensorviz_semantic_budget.bump(x)
                return x
        result, record = capture(Net(), context)
    finally:
        library._destroy()
    assert result.error is None and record.status == "partial"
    assert record.coverage.omissions == ["byte_limit"]
    assert not record.operations[0].effects
    assert all(value.producer is None for value in record.values)


def test_recording_does_not_keep_intermediates_alive(context):
    class Net(nn.Module):
        def forward(self, x):
            temporary = x + 1
            reference = weakref.ref(temporary)
            output = temporary * 2
            del temporary
            gc.collect()
            assert reference() is None
            return output
    result, record = capture(Net(), context)
    assert result.error is None and record.status == "complete"


def test_unknown_custom_operator_is_a_generic_schema_record(context):
    library = torch.library.Library("tensorviz_semantic_test", "DEF")
    try:
        library.define("wave(Tensor x, float scale) -> Tensor")
        library.impl("wave", lambda x, scale: x * scale, "CPU")
        class Net(nn.Module):
            def forward(self, x):
                return torch.ops.tensorviz_semantic_test.wave(x, 2.5)
        _, record = capture(Net(), context)
    finally:
        library._destroy()
    assert len(record.operations) == 1
    op = record.operations[0]
    assert op.target == "tensorviz_semantic_test.wave.default"
    assert op.arguments.args[1].value == 2.5
    assert op.schemaSignature.startswith("tensorviz_semantic_test::wave(")
    assert record.status == "complete"


def test_function_mode_does_not_certify_escaped_mutable_storage(context):
    class Net(nn.Module):
        def forward(self, x):
            x.data_ptr()
            return x + 1
    _, record = capture(Net(), context)
    assert record.status == "partial"
    assert "unsupported_type" in record.coverage.omissions


def test_grad_metadata_is_read_after_autograd_and_python_failures_are_captured(context):
    class Net(nn.Module):
        def forward(self, x):
            y = x * 2
            try:
                x.add_(1)
            except RuntimeError:
                return y
    result, record = capture(Net(), context, args=(torch.ones(2, 4, requires_grad=True),), grad_enabled=True)
    assert result.error is None and record.status == "complete"
    assert record.invocations[0].gradEnabled
    assert all(value.requiresGrad for value in record.values)
    assert record.operations[-1].target == "python:add_"
    assert record.operations[-1].status == "failed"
    assert record.operations[-1].arguments.args[1].value == "1"


def test_too_small_envelope_budget_does_not_fail_the_model_or_claim_capture(context):
    context = replace(context, limits=context.limits.model_copy(update={"bytes": 1024}))
    result = trace_model(nn.Identity(), (torch.ones(4),), tier="real", semantic_context=context)
    assert result.error is None
    assert result.semantic_capture is None
    assert result.semantic_error == "byte_limit"


def test_recorder_exception_retains_an_honest_prefix_without_changing_forward(context, monkeypatch):
    real = SemanticRecorder.begin
    def broken(self, func, args, kwargs):
        if str(func) == "aten.mul.Tensor":
            raise RuntimeError("internal metadata failure")
        return real(self, func, args, kwargs)
    monkeypatch.setattr(SemanticRecorder, "begin", broken)
    class Net(nn.Module):
        def forward(self, x):
            return (x + 1) * 2
    result, record = capture(Net(), context)
    assert result.error is None and record.status == "partial"
    assert record.coverage.omissions == ["recorder_error"]
    assert record.operations[0].status == "completed"
    assert record.output.kind == "opaque"


def test_unused_module_and_called_stateless_module_have_different_coverage(context):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.called = nn.Identity()
            self.unused = nn.ReLU()
        def forward(self, x):
            return self.called(x)
    _, record = capture(Net(), context)
    by_path = {entry.modulePath: entry for entry in record.coverage.modules}
    assert by_path["called"].status == "observed" and by_path["called"].invocationIds
    assert by_path["unused"].status == "not_observed" and not by_path["unused"].invocationIds
    assert not record.operations and record.status == "complete"


def test_global_and_instance_wrappers_restore_after_forward_failure(context):
    class Net(nn.Module):
        def forward(self, x):
            raise RuntimeError("forward failed")
    model = Net()
    original = nn.Module._call_impl
    _, record = capture(model, context)
    assert record.status == "partial"
    assert nn.Module._call_impl is original
    assert "_call_impl" not in model.__dict__


def test_kernel_write_without_returned_version_has_an_explicit_effect(context):
    library = torch.library.Library("tensorviz_semantic_mutation", "DEF")
    try:
        library.define("bump(Tensor(a!) x) -> ()")
        def bump(x):
            x.add_(1)
        library.impl("bump", bump, "CPU")
        class Net(nn.Module):
            def forward(self, x):
                torch.ops.tensorviz_semantic_mutation.bump(x)
                return x + 1, x._version
        result, record = capture(Net(), context)
    finally:
        library._destroy()
    assert result.error is None and record.status == "complete"
    assert record.operations[0].output.kind == "none"
    effect = record.operations[0].effects[0]
    assert effect.argumentPath == ["args", 0]
    assert effect.beforeId == record.inputs.args[0].id
    assert effect.after == record.operations[1].arguments.args[0]
    changed = next(value for value in record.values if value.id == effect.after.id)
    assert changed.producer.kind == "effect" and changed.producer.path == [0, "after"]
    assert changed.previousVersion == effect.beforeId
    before = next(value for value in record.values if value.id == effect.beforeId)
    assert changed.version == int(record.output.items[1].value), "keep PyTorch's actual observed counter"
    assert changed.logicalVersion == before.logicalVersion + 1


def test_repeated_writable_operands_keep_their_exact_ports(context):
    class Net(nn.Module):
        def forward(self, x):
            torch._foreach_add_([x, x], 1)
            return x
    result, record = capture(Net(), context)
    assert result.error is None and record.status == "complete"
    op = record.operations[0]
    assert [effect.argumentPath for effect in op.effects] == [["args", 0, "items", 0], ["args", 0, "items", 1]]
    assert op.effects[0].beforeId == op.effects[1].beforeId
    assert op.effects[0].after == op.effects[1].after == record.output
    assert next(v for v in record.values if v.id == record.output.id).logicalVersion == 1


def test_keyword_out_and_independent_alias_counters_keep_storage_effects(context):
    # Passing an independently versioned alias as a boundary avoids escaping
    # storage inside the measured forward; both aliases are known beforehand.
    class Pair(nn.Module):
        def forward(self, x, alias):
            torch.add(x, 2, out=x)
            return x, alias
    x = torch.ones(2, 4)
    alias = torch.empty(0).set_(x.untyped_storage(), 0, x.shape, x.stride())
    _, record = capture(Pair(), context, args=(x, alias))
    assert record.status == "complete"
    op = record.operations[0]
    assert [effect.argumentPath for effect in op.effects] == [["kwargs", "out"], ["kwargs", "out"]]
    assert [effect.kind for effect in op.effects] == ["write", "alias_update"]
    values = {v.id: v for v in record.values}
    before = values[record.inputs.args[1].id]
    after = values[record.output.items[1].id]
    assert after.version == before.version and after.logicalVersion == before.logicalVersion + 1
    assert after.producer.operationId == op.id


def test_caught_failed_write_retains_effects_but_never_certifies_completeness(context):
    library = torch.library.Library("tensorviz_semantic_failed_write", "DEF")
    try:
        library.define("bump(Tensor(a!) x) -> ()")
        def bump(x):
            x.add_(1)
            raise RuntimeError("failed after write")
        library.impl("bump", bump, "CPU")
        class Net(nn.Module):
            def forward(self, x):
                try:
                    torch.ops.tensorviz_semantic_failed_write.bump(x)
                except RuntimeError:
                    return x + 2
        result, record = capture(Net(), context)
    finally:
        library._destroy()
    assert result.error is None and record.status == "partial"
    failed, following = record.operations
    assert failed.status == "failed" and failed.output is None
    assert failed.effects[0].after == following.arguments.args[0]
    assert "unsupported_type" in record.coverage.omissions


def test_mutation_value_budget_leaves_an_explicit_unknown_effect(context):
    context = replace(context, limits=context.limits.model_copy(update={"values": 1}))
    class Net(nn.Module):
        def forward(self, x):
            x.add_(1)
            return x
    result, record = capture(Net(), context)
    assert result.error is None and record.status == "partial"
    assert record.operations[0].effects[0].after.kind == "opaque"
    assert record.operations[0].effects[0].after.reason == "value_limit"
    assert len(record.values) == 1


def test_mutated_alias_inventory_does_not_keep_activations_alive(context):
    class Net(nn.Module):
        def forward(self, x):
            h = x + 1
            alias = h.view(-1)
            refs = weakref.ref(h), weakref.ref(alias)
            h.add_(2)
            del h, alias
            gc.collect()
            assert all(reference() is None for reference in refs)
            return x + 3
    result, record = capture(Net(), context)
    assert result.error is None and record.status == "complete"


def test_inference_tensor_writes_preserve_a_missing_torch_counter(context):
    class Net(nn.Module):
        def forward(self, x):
            x.add_(1)
            return x
    with torch.inference_mode():
        result, record = capture(Net(), context, args=(torch.ones(2, 4),))
    assert result.error is None and record.status == "complete"
    before = next(v for v in record.values if v.id == record.inputs.args[0].id)
    after = next(v for v in record.values if v.id == record.output.id)
    assert before.version is None and after.version is None
    assert after.logicalVersion == before.logicalVersion + 1
    assert after.previousVersion == before.id


def test_dynamic_parameter_state_is_visible_but_not_certified_as_saved_state(context):
    class Net(nn.Module):
        def forward(self, x):
            return nn.Linear(4, 2)(x)
    result, record = capture(Net(), context)
    assert result.error is None and record.status == "partial"
    assert "unsupported_type" in record.coverage.omissions
    assert any(not module.paths for module in record.modules)
