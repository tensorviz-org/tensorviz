"""Real observation trees must retain operands, types and order without payloads."""
import gc
import math
import weakref

from pydantic import TypeAdapter
import pytest
import torch

from services.semantic.arguments import ArgumentEncoder
from services.semantic.contracts import Argument, Arguments, walk_argument
from services.semantic.tensors import TensorMetadata


def encoder(limit=100):
    values, omissions = [], set()
    metadata = TensorMetadata(torch, values, limit, omissions.add, lambda _: None)

    def reference(tensor, path):
        fact = metadata.observe(tensor)
        return (dict(kind="value", id=fact["id"]) if fact is not None else
                dict(kind="opaque", reason=metadata.reason, typeName="torch.Tensor"))
    return ArgumentEncoder(torch, reference, omissions.add), metadata, omissions


def test_literals_repeated_operands_ordered_containers_and_kwargs():
    enc, metadata, omissions = encoder()
    h = torch.ones(2, 4)
    repeated = enc.arguments((h, h), {"alpha": 1})
    assert repeated["args"][0] == repeated["args"][1]
    assert len(metadata.values) == 1
    assert repeated["kwargs"] == {"alpha": dict(kind="int", value="1")}
    nested = enc.arguments(([h, h + 1],), {"dim": 1, "options": {False: None, "tag": (True, 2.0)}})
    parsed = Arguments.model_validate(nested)
    assert list(parsed.kwargs) == ["dim", "options"]
    assert nested["args"][0]["items"][0] == repeated["args"][0]
    assert nested["args"][0]["items"][1] != repeated["args"][0]
    assert nested["kwargs"]["options"]["entries"][0]["key"] == dict(kind="bool", value=False)
    assert not omissions


@pytest.mark.parametrize("value,kind,expected", [
    (None, "none", None), (False, "bool", False), (1, "int", "1"),
    (2**80, "int", str(2**80)), (1.0, "float", 1.0), (-0.0, "float", "-0"),
    (math.nan, "float", "nan"), (math.inf, "float", "inf"), (-math.inf, "float", "-inf"),
    ("hello", "str", "hello"), (torch.float32, "dtype", "torch.float32"),
    (torch.device("cpu"), "device", "cpu"), (torch.strided, "layout", "torch.strided"),
    (torch.contiguous_format, "memory_format", "torch.contiguous_format"),
])
def test_literal_types_survive_json(value, kind, expected):
    enc, _, omissions = encoder()
    result = enc.output(value)
    parsed = TypeAdapter(Argument).validate_json(TypeAdapter(Argument).dump_json(result, warnings=False))
    assert parsed.kind == kind
    if kind != "none":
        assert parsed.value == expected
    assert not omissions


def test_symbolic_integer_is_not_forced_to_a_concrete_value():
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    symbol = ShapeEnv().create_unbacked_symint()
    enc, _, omissions = encoder()
    result = enc.output(symbol)
    assert result == dict(kind="symbolic", type="int", expression=str(symbol))
    TypeAdapter(Argument).validate_python(result)
    assert not omissions


def test_unknown_objects_do_not_run_repr_or_iterators():
    class Secret:
        def __repr__(self):
            raise AssertionError("repr is not observation")
        def __iter__(self):
            raise AssertionError("iteration would execute user code")
    enc, _, omissions = encoder()
    result = enc.output(Secret())
    assert result["kind"] == "opaque"
    assert result["typeName"].endswith(".Secret")
    assert omissions == {"unsupported_type"}


def test_cycles_large_trees_long_strings_and_oversize_integers_are_explicit():
    enc, _, omissions = encoder()
    cyclic = []
    cyclic.append(cyclic)
    for source in (cyclic, "x" * 4097, 1 << 20000, [[0] * 4096] * 256):
        result = TypeAdapter(Argument).validate_python(enc.output(source))
        tree = list(walk_argument(result))
        assert len(tree) <= 16384
        assert max(len(path) for path, _ in tree) <= 64
        assert any(item.kind == "opaque" for _, item in tree)
    assert omissions == {"depth_limit", "item_limit"}


def test_value_budget_refuses_reference_without_dangling_identity():
    enc, metadata, omissions = encoder(limit=1)
    output = enc.output((torch.ones(1), torch.ones(1)))
    assert output["items"][0]["kind"] == "value"
    assert output["items"][1] == dict(kind="opaque", reason="value_limit", typeName="torch.Tensor")
    assert len(metadata.values) == 1
    assert omissions == {"value_limit"}


def test_shared_metadata_preserves_alias_versions_and_allows_collection():
    enc, metadata, omissions = encoder()
    tensor = torch.ones(2, 4)
    alias = tensor.view(4, 2)
    reference = weakref.ref(tensor)
    first = metadata.observe(tensor)
    view = metadata.observe(alias)
    tensor.add_(1)
    metadata.written(tensor)
    changed = metadata.observe(tensor)
    assert view["aliasOf"] == first["id"]
    assert changed["previousVersion"] == first["id"]
    assert changed["version"] == first["version"] + 1
    assert metadata.observe(tensor) is changed
    del tensor, alias
    gc.collect()
    assert reference() is None
    assert not metadata.objects and not metadata.storages
    assert not omissions


def test_independent_alias_version_counter_cannot_hide_a_storage_write():
    _, metadata, omissions = encoder()
    first = torch.ones(2)
    alias = torch.empty(0).set_(first.untyped_storage(), 0, (2,), (1,))
    metadata.observe(first)
    metadata.observe(alias)
    alias.add_(1)
    metadata.written(alias)
    assert metadata.observe(first) is None
    assert omissions == {"unsupported_type"}


class MutationModel(torch.nn.Module):
    def forward(self, x):
        a = x + 1
        view = a.view(-1)
        a.add_(2)
        return a, view


def test_existing_validation_dispatch_preserves_real_versions_after_shared_metadata():
    from services.validation.trace import TraceRecorder
    from services.validation.contracts import TensorFact
    record = dict(tensors=[], modules=[], invocations=[], edges=[], context=None,
                  capture=dict(rootInvocationId=None),
                  limits=dict(max_tensors=100, max_events=100, max_record_bytes=1000000))
    recorder = TraceRecorder(record, torch)
    output = recorder.run(MutationModel(), (torch.ones(2, 4),), {})
    assert torch.equal(output[0], torch.full((2, 4), 4.0))
    assert record["capture"]["state"] == "complete"
    for value in record["tensors"]:
        TensorFact.model_validate(value)
    writes = [item for item in record["invocations"] if item["name"] == "aten.add_.Tensor"]
    assert len(writes) == 1
    write = writes[0]
    values = {value["id"]: value for value in record["tensors"]}
    before = values[write["inputs"][0]["tensorId"]]
    after = values[write["output"]["tensorId"]]
    assert after["previousVersion"] == before["id"]
    assert after["version"] == before["version"] + 1


@pytest.mark.parametrize('ordered', [False, True])
def test_dictionary_subclass_storage_is_observed_without_custom_hooks(ordered):
    from collections import OrderedDict
    base = OrderedDict if ordered else dict
    class Output(base):
        def items(self): raise AssertionError('custom items must not run')
        def __iter__(self): raise AssertionError('custom iterator must not run')
        def __len__(self): raise AssertionError('custom length must not run')
        def __getitem__(self, key): raise AssertionError('custom lookup must not run')
        def __repr__(self): raise AssertionError('custom repr must not run')
    first, second = torch.ones(2, 3), torch.zeros(1)
    output = Output(first=first, second=second)
    if ordered:
        OrderedDict.move_to_end(output, 'first')
    enc, metadata, omissions = encoder()
    result = enc.output(output)
    assert result['kind'] == 'dict'
    assert [entry['key']['value'] for entry in result['entries']] == (['second', 'first'] if ordered else ['first', 'second'])
    assert all(entry['value']['kind'] == 'value' for entry in result['entries'])
    assert len(metadata.values) == 2 and not omissions
    TypeAdapter(Argument).validate_python(result)


def test_dictionary_subclass_cycles_and_budgets_remain_explicit():
    class Output(dict): pass
    cyclic = Output()
    cyclic['self'] = cyclic
    enc, _, omissions = encoder()
    for value in (cyclic, Output.fromkeys(range(4097), None)):
        TypeAdapter(Argument).validate_python(enc.output(value))
    assert omissions == {'depth_limit', 'item_limit'}
