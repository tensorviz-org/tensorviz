"""Source locations follow verified executing code, never an incidental filename."""
from dataclasses import replace
import ast
import gc
import hashlib
import types
import weakref

import pytest
import torch

from services.ingest.trace import trace_model
from services.semantic.contracts import CaptureLimits, CaptureSnapshot, SourceSnapshot
from services.semantic.recorder import CaptureContext
from services.semantic.source_bindings import CapturedSourceEvidence, SourceIndex, _binding_tokens


def test_binding_tokens_do_not_depend_on_ast_dump_empty_field_defaults():
    # Pinned compact form used by the shared Python 3.13 capture fixtures.
    node = ast.parse("nn.ReLU()", mode="eval").body
    assert _binding_tokens(node) == "Call(func=Attribute(value=Name(id='nn', ctx=Load()), attr='ReLU', ctx=Load()))"
    # Missing arguments differ from an explicit empty list, None or AST-like text.
    variants = ["fn()", "fn([])", "fn(None)", "fn(', args=[]')", "fn(flag=False)"]
    assert len({_binding_tokens(ast.parse(source, mode="eval").body) for source in variants}) == len(variants)


def test_empty_optional_ast_fields_do_not_change_definition_tokens():
    node = ast.parse("def forward(x):\n    return x\n").body[0]
    before = _binding_tokens(node)
    # Python 3.12 added this empty field; Python 3.11 has the same semantics.
    node._fields = tuple(field for field in node._fields if field != "type_params")
    assert _binding_tokens(node) == before


def prepared(tmp_path, source, runtime_source=None):
    data = source.encode()
    path = tmp_path / "model.py"
    path.write_bytes(data)
    snapshot = SourceSnapshot(version=1, id="source:0", workspaceSessionId="session:0",
        workspaceId="workspace:0", modelId="model:0", modelFile="model.py", className="Model", generation=0,
        files=[dict(path="model.py", digest=hashlib.sha256(data).hexdigest(), documentVersion=1, role="source")],
        dependencyScope="complete", configDigest=None, environmentDigest="b" * 64, sourceMode="saved")
    sources = CapturedSourceEvidence(snapshot, tmp_path)
    namespace = {"__name__": "test_model"}
    exec(compile(runtime_source or data, str(path), "exec", dont_inherit=True), namespace)
    context = CaptureContext(snapshot, "c" * 64, "attempt:0",
        CaptureLimits(operations=1000, invocations=1000, values=1000, bytes=4 * 1024 * 1024), source_evidence=sources)
    return namespace, context


def run(model, context):
    result = trace_model(model, (torch.ones(2, 4),), tier="real", semantic_context=context)
    assert result.error is None, result.error
    record = CaptureSnapshot.model_validate(result.semantic_capture)
    assert record.status == "complete", record.coverage
    return record


def expressions(record):
    bindings = {binding.id: binding for binding in record.sourceBindings}
    return [(op, bindings[op.sourceBindingIds[0]].anchors[0]) for op in record.operations if op.sourceBindingIds]


def test_exact_multiline_calls_repeated_operands_and_shared_call_locations(tmp_path):
    source = '''import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = torch.nn.ReLU()
    def forward(self, h):
        a = h + 1
        b = self.activation(
            a + a
        )
        return self.activation(b)
'''
    namespace, context = prepared(tmp_path, source)
    record = run(namespace["Model"](), context)
    sites = expressions(record)
    assert [op.target for op, _ in sites] == [op.target for op in record.operations]
    assert [(a.startLine, a.endLine) for _, a in sites] == [(7, 7), (9, 9), (8, 10), (11, 11)]
    assert all(a.scope == "Model.forward" for _, a in sites)
    calls = record.invocations[1:]
    assert len(calls) == 2 and calls[0].moduleId == calls[1].moduleId
    assert calls[0].sourceBindingIds != calls[1].sourceBindingIds
    assert record.definitions[0].sourceBindingIds
    root_binding = next(b for b in record.sourceBindings if b.id == record.definitions[0].sourceBindingIds[0])
    assert root_binding.kind == "definition" and root_binding.anchors[0].startLine == 6
    assert root_binding.anchors[0].scope == "Model.forward"


def test_source_index_never_imports_or_evaluates_source_and_uses_utf16_columns(tmp_path):
    source = '''import torch
raise RuntimeError("must not execute")
class Model(torch.nn.Module):
    def forward(self, h):
        label = "😀\u2028"; return h + 1
'''
    snapshot = types.SimpleNamespace(id="s")
    file = types.SimpleNamespace(path="model.py", digest=hashlib.sha256(source.encode()).hexdigest())
    index = SourceIndex(snapshot.id, file, source.encode())
    binding = next(item[0] for key, item in index.expressions.items() if key[1] == 5)
    anchor = binding["anchors"][0]
    line = source.split("\n")[4]
    assert anchor["startColumn"] == len(line[:line.index("h + 1")].encode("utf-16-le")) // 2
    assert anchor["endColumn"] == len(line.encode("utf-16-le")) // 2


@pytest.mark.parametrize("runtime", [
    '''import torch
class Model(torch.nn.Module):
    def forward(self, h):
        return h + 2
''',
    '''import torch
class Model(torch.nn.Module):
    def forward(self, h):
        return h * 1
''',
])
def test_forged_or_changed_code_with_same_filename_never_gets_saved_bindings(tmp_path, runtime):
    saved = '''import torch
class Model(torch.nn.Module):
    def forward(self, h):
        return h + 1
'''
    namespace, context = prepared(tmp_path, saved, runtime)
    assert not run(namespace["Model"](), context).sourceBindings


def test_source_changed_after_index_is_never_reread_to_relabel_old_code(tmp_path):
    source = '''import torch
class Model(torch.nn.Module):
    def forward(self, h):
        return h + 1
'''
    namespace, context = prepared(tmp_path, source)
    (tmp_path / "model.py").write_text(source.replace("h + 1", "h + 2"))
    record = run(namespace["Model"](), context)
    assert expressions(record)[0][1].digest == context.source.files[0].digest
    assert all(binding.snapshotId == context.source.id for binding in record.sourceBindings)
    # The worker's existing post-run source check decides whether to retain this
    # old snapshot; mapping itself stays tied to the immutable pre-import source.


def test_unverified_helper_is_not_misattributed_to_saved_outer_call(tmp_path):
    source = '''import torch
class Model(torch.nn.Module):
    def forward(self, h):
        return external(h)
'''
    namespace, context = prepared(tmp_path, source)
    helper = {}
    exec(compile("def helper(h): return h + 1", str(tmp_path / "unknown.py"), "exec"), helper)
    namespace["external"] = helper["helper"]
    record = run(namespace["Model"](), context)
    assert record.operations and all(not op.sourceBindingIds for op in record.operations)


def test_derived_operations_share_one_authored_expression_without_invented_edit_locations(tmp_path):
    source = '''import torch
class Model(torch.nn.Module):
    def forward(self, h):
        return torch.nn.functional.layer_norm(h, (4,))
'''
    namespace, context = prepared(tmp_path, source)
    record = run(namespace["Model"](), context)
    assert record.operations and all(op.sourceBindingIds for op in record.operations)
    assert len({op.sourceBindingIds[0] for op in record.operations}) == 1
    assert expressions(record)[0][1].startLine == 4


def test_absent_pep657_position_and_location_budget_leave_computation_available(tmp_path, monkeypatch):
    source = '''import torch
class Model(torch.nn.Module):
    def forward(self, h):
        return h + 1
'''
    namespace, context = prepared(tmp_path, source)
    monkeypatch.setattr("services.semantic.source_bindings.MAX_CODES", 0)
    record = run(namespace["Model"](), context)
    assert not record.operations[0].sourceBindingIds
    assert record.operations[0].arguments.args[1].value == "1"


def test_binding_cache_does_not_retain_runtime_code(tmp_path):
    source = '''import torch
class Model(torch.nn.Module):
    def forward(self, h):
        return h + 1
'''
    namespace, context = prepared(tmp_path, source)
    cls = namespace["Model"]
    code = weakref.ref(cls.forward.__code__)
    run(cls(), context)
    del cls, namespace
    gc.collect()
    assert code() is None
    context.source_evidence.clear()
    assert not context.source_evidence.sites and not context.source_evidence.indexes and not context.source_evidence.source_buffers


def test_index_budget_and_unavailable_instruction_position_are_navigation_limits(tmp_path, monkeypatch):
    source = '''import torch
class Model(torch.nn.Module):
    def forward(self, h):
        return h + 1
'''
    namespace, context = prepared(tmp_path, source)
    code = namespace["Model"].forward.__code__
    assert context.source_evidence.expression(code, 0) is None
    assert context.source_evidence.expression(code, -1) is None
    monkeypatch.setattr("services.semantic.source_bindings.MAX_SITES", 0)
    context = replace(context, source_evidence=CapturedSourceEvidence(context.source, tmp_path))
    assert not run(namespace["Model"](), context).sourceBindings


def test_source_site_budget_is_shared_across_files(tmp_path, monkeypatch):
    source = '''import torch
class Model(torch.nn.Module):
    def forward(self, h):
        return h + 1
'''
    _, context = prepared(tmp_path, source)
    helper = b"def helper(x): return x + 2\n"
    (tmp_path / "helper.py").write_bytes(helper)
    snapshot = context.source.model_copy(update={"files": [*context.source.files,
        context.source.files[0].model_copy(update={"path": "helper.py", "digest": hashlib.sha256(helper).hexdigest()})]})
    monkeypatch.setattr("services.semantic.source_bindings.MAX_SITES", 3)
    sources = CapturedSourceEvidence(snapshot, tmp_path)
    assert not sources.indexes  # Unused repository files spend no AST-site budget.
    assert sources.source_index("model.py").site_count == 3
    assert sources.source_index("helper.py") is None
    assert sum(index.site_count for index in sources.indexes.values() if index is not None) == 3
