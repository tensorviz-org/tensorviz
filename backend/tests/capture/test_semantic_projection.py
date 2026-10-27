"""Projection never deduplicates argument uses or deletes captured computation."""
from copy import deepcopy
import json
from pathlib import Path
import random

import pytest
import torch
from torch import nn

from services.semantic.contracts import CaptureProjection
from services.semantic.projection import project_capture
from tests.capture.test_semantic_recorder import context, capture  # shared actual recorder fixture


def original_records(projection):
    data = projection.model_dump(mode="json")
    return dict(nodes=[{key: value for key, value in node.items()
                       if key not in ("visible", "collapsed", "hiddenOperationCount")} for node in data["nodes"]],
                ports=[port for port in data["ports"] if port["memberPortId"] is None],
                edges=[{key: value for key, value in edge.items()
                       if key not in ("displaySourcePortId", "displayTargetPortId")} for edge in data["edges"]])


def ports_by_node(projection, kind, semantic_id=None):
    ids = {node.id for node in projection.nodes if node.kind == kind
           and (semantic_id is None or node.semanticId == semantic_id)}
    return [port for port in projection.ports if port.nodeId in ids and port.memberPortId is None]


def test_add_literal_and_repeated_operands_keep_distinct_ports():
    payload = json.loads((Path(__file__).parent.resolve().parents[2] / "tests/fixtures/contracts/semantic-capture.json").read_text())
    before = deepcopy(payload)
    projection = project_capture(payload)
    assert payload == before
    add1 = ports_by_node(projection, "operation", "add1")
    literal = next(port for port in add1 if port.argumentKind == "int")
    assert literal.paths == [["arguments", "args", 1]] and literal.valueId is None
    assert not any(edge.targetPortId == literal.id for edge in projection.edges)
    add2 = ports_by_node(projection, "operation", "add2")
    inputs = [port for port in add2 if port.direction == "in"]
    assert [port.paths for port in inputs] == [[["arguments", "args", 0]], [["arguments", "args", 1]]]
    edges = [edge for edge in projection.edges if edge.targetPortId in {p.id for p in inputs}]
    assert len(edges) == 2 and len({edge.id for edge in edges}) == 2
    assert len({edge.sourcePortId for edge in edges}) == 1
    assert not [n for n in projection.nodes if n.kind == "value"]
    assert len(projection.edges) == 4


def test_occurrence_ids_are_scoped_to_capture_not_reused_event_ordinals():
    payload = json.loads((Path(__file__).parent.resolve().parents[2] / "tests/fixtures/contracts/semantic-capture.json").read_text())
    first = project_capture(payload)
    payload["id"] = "capture:next"
    second = project_capture(payload)
    assert {node.semanticId for node in first.nodes if node.kind == "operation"} == {
        node.semanticId for node in second.nodes if node.kind == "operation"}
    assert not {node.id for node in first.nodes} & {node.id for node in second.nodes}
    assert not {edge.id for edge in first.edges} & {edge.id for edge in second.edges}


def test_nested_inputs_outputs_literals_and_input_aliases_stay_explicit(context):
    class Net(nn.Module):
        def forward(self, x, *, bundle):
            return {"scales": [torch.cat([x, bundle["skip"]], dim=1), x], "empty": (), "label": 7}
    tensor = torch.ones(2, 4)
    _, record = capture(Net(), context, args=(tensor,), kwargs={"bundle": {"skip": tensor}})
    projection = project_capture(record)
    inputs = ports_by_node(projection, "input")
    assert len(inputs) == 1 and inputs[0].argumentKind == "value"
    assert inputs[0].paths == [["inputs", "args", 0], ["inputs", "kwargs", "bundle", "entries", 0, "value"]]
    output = ports_by_node(projection, "output")
    assert [port.argumentKind for port in output] == ["value", "value", "tuple", "int"]
    assert len([edge for edge in projection.edges if edge.kind == "result"]) == 2
    assert len({tuple(path) for port in output for path in port.paths}) == 4


def test_whisper_shaped_named_boundaries_keep_separate_values_and_dtypes(context):
    class Net(nn.Module):
        def forward(self, mel, tokens):
            return mel.sum(), tokens + 1
    _, record = capture(Net(), context, args=(torch.ones(1, 80, 32), torch.zeros(1, 4, dtype=torch.long)))
    projection = project_capture(record)
    values = {value.id: value for value in record.values}
    inputs = ports_by_node(projection, "input")
    assert [(values[p.valueId].shape, values[p.valueId].dtype) for p in inputs] == [
        ([1, 80, 32], "torch.float32"), ([1, 4], "torch.int64")]
    output = ports_by_node(projection, "output")
    assert [values[p.valueId].shape for p in output] == [[], [1, 4]]


def test_reused_module_calls_and_every_repeated_block_survive_collapse(context):
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()
        def forward(self, x):
            return self.relu(x) + 1
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList(Block() for _ in range(12))
            self.shared = nn.GELU()
        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return self.shared(self.shared(x))
    _, record = capture(Net(), context)
    original = record.model_dump_json()
    full = project_capture(record)
    assert len([node for node in full.nodes if node.kind == "operation"]) == len(record.operations)
    assert len([node for node in full.nodes if node.kind == "invocation"]) == len(record.invocations)
    instances = {module.id: module.paths for module in record.modules}
    blocks = [call.id for call in record.invocations if any(path.startswith("blocks.") and path.count(".") == 1
                                                           for path in instances[call.moduleId])]
    shared = [call.id for call in record.invocations if "shared" in instances[call.moduleId]]
    assert len(blocks) == 12 and len(shared) == 2
    rng = random.Random(17)
    for _ in range(30):
        collapsed = [call.id for call in record.invocations if rng.choice([True, False])]
        grouped = project_capture(record, collapsed_invocations=collapsed)
        assert original_records(grouped) == original_records(full)
        assert record.model_dump_json() == original
        assert project_capture(record).model_dump() == full.model_dump()
    grouped = project_capture(record, collapsed_invocations=blocks + shared)
    regions = [node for node in grouped.nodes if node.collapsed]
    assert len(regions) == 14 and all(node.visible for node in regions)
    assert sum(node.hiddenOperationCount for node in regions) == len(record.operations)
    assert all(edge.displaySourcePortId is None and edge.displayTargetPortId is None or
               edge.displaySourcePortId is not None and edge.displayTargetPortId is not None for edge in grouped.edges)


def test_identity_modules_keep_interfaces_without_inventing_computation(context):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.identity = nn.Identity()
        def forward(self, x):
            return self.identity(self.identity(x))
    _, record = capture(Net(), context)
    projection = project_capture(record, collapsed_invocations=[call.id for call in record.invocations])
    assert not record.operations
    assert len([node for node in projection.nodes if node.kind == "invocation"]) == 3
    assert all(node.hiddenOperationCount == 0 for node in projection.nodes)
    assert len(projection.edges) == 1 and projection.edges[0].kind == "result"
    assert len(ports_by_node(projection, "invocation")) == 6


def test_disconnected_factory_operation_never_gets_a_rank_based_input_edge(context):
    class Net(nn.Module):
        def forward(self, unused):
            value = torch.ones(2, 4)
            return value + value
    _, record = capture(Net(), context)
    projection = project_capture(record)
    inputs = {port.id for port in ports_by_node(projection, "input")}
    assert inputs and not any(edge.sourcePortId in inputs for edge in projection.edges)
    factory = record.operations[0]
    assert factory.target == "aten.ones.default"
    factory_ports = {port.id for port in ports_by_node(projection, "operation", factory.id)}
    assert not any(edge.targetPortId in factory_ports for edge in projection.edges)
    add = next(op for op in record.operations if op.target == "aten.add.Tensor")
    add_inputs = {port.id for port in ports_by_node(projection, "operation", add.id) if port.direction == "in"}
    assert len([edge for edge in projection.edges if edge.targetPortId in add_inputs]) == 2


def test_private_state_hides_inside_its_use_but_shared_state_stays_shared(context):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(4, 4)
            self.second = nn.Linear(4, 4)
            self.second.weight = self.first.weight
        def forward(self, x):
            return self.second(self.first(x))
    _, record = capture(Net(), context)
    child_calls = [call.id for call in record.invocations if call.parentId == record.rootInvocationId]
    projection = project_capture(record, collapsed_invocations=child_calls)
    values = {value.id: value for value in record.values}
    nodes = {node.id: node for node in projection.nodes}
    state = [node for node in projection.nodes if node.kind == "value"]
    shared = [node for node in state if len(values[node.semanticId].state) > 1]
    assert len(shared) == 1 and shared[0].visible
    assert nodes[shared[0].parentId].semanticId == record.rootInvocationId
    private = [node for node in state if node not in shared]
    assert private and all(not node.visible and nodes[node.parentId].semanticId in child_calls for node in private)


def test_generic_ops_state_and_nonreturned_mutations_have_real_producers(context):
    class Write(nn.Module):
        def forward(self, x):
            x.add_(1)
            return None
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4))
            self.write = Write()
        def forward(self, x):
            self.write(x)
            return torch.sin(x) + self.weight
    _, record = capture(Net(), context)
    write_module = next(module for module in record.modules if "write" in module.paths)
    call = next(call for call in record.invocations if call.moduleId == write_module.id)
    projection = project_capture(record, collapsed_invocations=[call.id])
    nodes = {node.id: node for node in projection.nodes}
    ports = {port.id: port for port in projection.ports}
    assert len([node for node in projection.nodes if node.kind == "operation"]) == len(record.operations)
    state_nodes = [node for node in projection.nodes if node.kind == "value"]
    assert len(state_nodes) == 1
    assert next(value for value in record.values if value.id == state_nodes[0].semanticId).state[0].name == "weight"
    cross = [edge for edge in projection.edges if edge.displaySourcePortId is not None
             and ports[edge.displaySourcePortId].memberPortId is not None]
    assert cross and all(nodes[ports[e.displaySourcePortId].nodeId].semanticId == call.id for e in cross)
    assert any(edge.kind == "effect" for edge in projection.edges)
    assert ports_by_node(projection, "invocation", call.id)[-1].argumentKind == "none"


def test_failed_and_capture_limited_prefixes_never_gain_a_success_output(context):
    from dataclasses import replace
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3, 4)
        def forward(self, x):
            return self.fc(x)
    _, record = capture(Net(), context)
    projection = project_capture(record)
    assert record.status == "partial"
    assert not ports_by_node(projection, "output")
    assert not any(edge.kind == "result" for edge in projection.edges)
    class Limited(nn.Module):
        def forward(self, x):
            return (x + 1).cos()
    _, record = capture(Limited(), replace(context, limits=context.limits.model_copy(update={"operations": 1})))
    assert record.status == "partial" and "operation_limit" in record.coverage.omissions
    projection = project_capture(record)
    assert len([node for node in projection.nodes if node.kind == "operation"]) == len(record.operations) == 1
    # An omitted producer must never be drawn as if it were a known operation.
    produced = {value.id for value in record.values if value.producer is not None}
    for edge in projection.edges:
        source = next(port for port in projection.ports if port.id == edge.sourcePortId)
        node = next(node for node in projection.nodes if node.id == source.nodeId)
        assert (node.kind == "operation") == (edge.valueId in produced)


def test_maximum_depth_input_paths_and_empty_arguments_survive():
    payload = json.loads((Path(__file__).parent.resolve().parents[2] / "tests/fixtures/contracts/semantic-capture.json").read_text())
    value = payload["inputs"]["args"][0]
    for _ in range(32):
        value = dict(kind="dict", entries=[dict(key=dict(kind="str", value="next"), value=value)])
    payload["inputs"]["args"] = [value, dict(kind="list", items=[])]
    payload["invocations"][0]["arguments"] = deepcopy(payload["inputs"])
    projection = project_capture(payload)
    inputs = ports_by_node(projection, "input")
    assert {p.argumentKind for p in inputs} == {"list", "value"}
    assert max(len(path) for p in inputs for path in p.paths) == 99


def test_projection_refuses_foreign_collapse_and_fabricated_connections(context):
    class Net(nn.Module):
        def forward(self, x):
            return torch.cos(x + 1)
    _, record = capture(Net(), context)
    with pytest.raises(ValueError, match="outside this capture"):
        project_capture(record, collapsed_invocations=["absent"])
    full = project_capture(record)
    for mutate in (
        lambda p: p["edges"][0].update(valueId="absent"),
        lambda p: p["edges"][0].update(displaySourcePortId=None),
        lambda p: p["nodes"][-1].update(visible=False),
        lambda p: p["nodes"][2].update(hiddenOperationCount=99),
        lambda p: p["ports"][0].update(memberPortId="absent"),
    ):
        broken = full.model_dump(mode="json")
        mutate(broken)
        with pytest.raises(ValueError):
            CaptureProjection.model_validate(broken)
    grouped = project_capture(record, collapsed_invocations=[record.rootInvocationId])
    crossed = next(port for port in grouped.ports if port.memberPortId is not None)
    broken = grouped.model_dump(mode="json")
    next(port for port in broken["ports"] if port["id"] == crossed.id)["valueId"] = "absent"
    with pytest.raises(ValueError, match="exact value"):
        CaptureProjection.model_validate(broken)
