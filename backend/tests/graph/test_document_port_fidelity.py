"""One actual Python-converter document is consumed by browser and MCP tests."""
import json
from pathlib import Path

from services.ingest.document import to_document


FIXTURE = Path(__file__).parent.resolve().parents[2] / "canvas/src/test/fixtures/canvas/ported-document.json"


def fixture_graph():
    def node(id, kind, **data):
        return dict(id=id, type="moduleNode" if kind == "module" else "layerNode",
                    data=dict(layerType=dict(id=kind, name=kind), parameters={}, **data))
    return dict(nodes=[node("input", "input", outputPorts=["h", "tokens"]),
                       node("block", "module", isGroup=True, moduleMembers=["add"]),
                       node("add", "add", parentId="block", inputPorts=["args:0", "args:1", "alpha"], outputPorts=["sum"]),
                       node("factory", "ones", inputPorts=[], outputPorts=["value"]),
                       node("output", "output", inputPorts=["result"])],
                edges=[dict(id="operand-0", source="input", sourceHandle="h", target="add", targetHandle="args:0"),
                       dict(id="operand-1", source="input", sourceHandle="h", target="add", targetHandle="args:1"),
                       dict(id="result", source="add", target="output", data=dict(sourceHandle="sum", targetHandle="result"))])


def test_python_converter_matches_shared_browser_and_mcp_fixture():
    document = to_document("Ported", fixture_graph(), "ported").model_dump(mode="json", by_alias=True)
    assert json.loads(FIXTURE.read_text()) == document
    nodes = {node["node_id"]: node for node in document["semantic"]["nodes"]}
    assert nodes["input"]["out_ports"] == ["h", "tokens"]
    assert nodes["add"]["in_ports"] == ["args:0", "args:1", "alpha"]
    assert nodes["factory"]["in_ports"] == []
    assert nodes["add"]["metadata"]["parentId"] == "block"
    edges = document["semantic"]["edges"]
    assert [(edge["source_port"], edge["target_port"]) for edge in edges] == [
        ("h", "args:0"), ("h", "args:1"), ("sum", "result")]


def test_undeclared_legacy_ports_follow_real_connections_in_order():
    graph = fixture_graph()
    for node in graph["nodes"]:
        node["data"].pop("inputPorts", None)
        node["data"].pop("outputPorts", None)
    document = to_document("Ported", graph, "ported")
    nodes = {node.node_id: node for node in document.semantic.nodes}
    assert nodes["add"].in_ports == ["args:0", "args:1"]
    assert nodes["input"].out_ports == ["h"]
    assert nodes["factory"].in_ports == ["x"]  # legacy default, not a zero-arity claim
