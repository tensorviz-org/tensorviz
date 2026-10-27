"""F7: retain source-owned dotted modules without inventing helper layers.

The files in ide_dotted_repo are the installed DottedNet/ScaleBlock reproduction.
All checks parse supplied text; no fixture is imported or executed.
"""
import ast
from pathlib import Path

import pytest

from models.graph_models import CodeParseRequest
from services.ide import code_apply, multifile
from services.ide.module_index import ModuleIndex
from utils.code_parser import CodeParserService
from tests.graph.test_ide_multifile_projection import client, _entry, _project, _sha  # noqa: F401


FIXTURES = Path(__file__).parent.parent / "fixtures" / "ide_dotted_repo"
ROOT = (FIXTURES / "dotted_model.py").read_text()
BLOCKS = (FIXTURES / "blocks.py").read_text()


def project(client, root=ROOT, sources=None):
    sources = {"blocks.py": BLOCKS} if sources is None else sources
    return _project(client, {
        "protocol_version": multifile.IDE_PROJECTION_PROTOCOL_VERSION,
        "root": {"file": "dotted_model.py", "cls": "DottedNet",
                 "document_version": 1, "source_digest": _sha(root)},
        "sources": [_entry("dotted_model.py", root),
                    *[_entry(file, source) for file, source in sources.items()]],
    })


def semantic(data):
    graph = data["document"]["semantic"]
    return ({n["node_id"]: n for n in graph["nodes"]},
            {(e["source_node_id"], e["target_node_id"]) for e in graph["edges"]})


def test_parser_keeps_dotted_call_span_and_real_forward_flow():
    parsed = CodeParserService().parse_pytorch_code(CodeParseRequest(code=ROOT, file="dotted_model.py"))
    graph = next(g["graph"] for g in parsed["graphs"] if g["name"] == "DottedNet")
    nodes = {n.id: n for n in graph.nodes}
    assert set(nodes) == {"input", "block", "head", "output"}
    block = nodes["block"]
    assert block.data.layerType.id == "custom_code"
    params = block.data.parameters
    assert params["_class"] == "ScaleBlock"
    assert params["_raw_call"] == "blocks.ScaleBlock()"
    assert params["_source_file"] == "dotted_model.py"
    assert ROOT.splitlines()[params["_lineno"] - 1].strip() == "self.block = blocks.ScaleBlock()"
    assert {(e.source, e.target) for e in graph.edges} == {
        ("input", "block"), ("block", "head"), ("head", "output")}


@pytest.mark.parametrize("import_text,callee,sources", [
    ("import blocks", "blocks.ScaleBlock", {"blocks.py": BLOCKS}),
    ("import blocks as b", "b.ScaleBlock", {"blocks.py": BLOCKS}),
    ("import pkg.blocks", "pkg.blocks.ScaleBlock", {"pkg/__init__.py": "", "pkg/blocks.py": BLOCKS}),
    ("from pkg import blocks", "blocks.ScaleBlock", {"pkg/__init__.py": "", "pkg/blocks.py": BLOCKS}),
    ("import pkg", "pkg.Exported", {"pkg/__init__.py": "from .blocks import ScaleBlock as Exported\n", "pkg/blocks.py": BLOCKS}),
    ("import pkg", "pkg.lowercase", {"pkg/__init__.py": "from .blocks import ScaleBlock as lowercase\n", "pkg/blocks.py": BLOCKS}),
])
def test_workspace_dotted_imports_and_reexports_preserve_definition_and_edges(client, import_text, callee, sources):
    root = ROOT.replace("import blocks", import_text).replace("blocks.ScaleBlock", callee)
    data = project(client, root, sources)
    nodes, edges = semantic(data)
    assert set(nodes) == {"input", "block", "block.proj", "head", "output"}
    assert nodes["block"]["kind"] == "module"
    assert nodes["block"]["params"]["_class"] == "ScaleBlock"
    definition = "pkg/blocks.py" if "pkg/blocks.py" in sources else "blocks.py"
    assert nodes["block"]["params"]["_defined_in"] == definition
    assert not nodes["block"]["params"].get("_opaque")
    assert data["spanmap"]["block"]["file"] == "dotted_model.py"
    assert data["spanmap"]["block.proj"]["file"] == definition
    assert data["spanmap"]["block.proj"]["file_digest"] == _sha(BLOCKS)
    assert {("input", "block"), ("block", "head"), ("head", "output"),
            ("input", "block.proj"), ("block.proj", "head")} == edges
    assert not data["diagnostics"]
    # Empty package markers are not consulted; a re-export file is a dependency.
    assert {d["file"] for d in data["dependencies"]} == {
        "dotted_model.py", *(file for file, source in sources.items() if source)}


@pytest.mark.parametrize("sources,code", [
    ({}, "EXTERNAL_PACKAGE"),
    ({"blocks.py": "class Other: pass\n"}, "UNRESOLVED_IMPORT"),
    ({"blocks.py": "class ScaleBlock(\n"}, "DEPENDENCY_SYNTAX_ERROR"),
    ({"blocks.py": "from other import ScaleBlock\nfrom elsewhere import ScaleBlock\n"}, "AMBIGUOUS_BINDING"),
    ({"blocks.py": "from other import ScaleBlock\n", "other.py": "from blocks import ScaleBlock\n"}, "CYCLE_BOUNDARY"),
])
def test_unresolved_dotted_constructor_is_an_opaque_card_with_diagnostic(client, sources, code):
    data = project(client, sources=sources)
    nodes, edges = semantic(data)
    assert set(nodes) == {"input", "block", "head", "output"}
    assert nodes["block"]["op"] == "custom_code"
    assert nodes["block"]["params"]["_opaque"] is True
    assert edges == {("input", "block"), ("block", "head"), ("head", "output")}
    diagnostic = next(d for d in data["diagnostics"] if d.get("node") == "block")
    assert diagnostic["code"] == code
    assert diagnostic["file"] == "dotted_model.py"
    assert diagnostic["line"] == 8
    assert data["spanmap"]["block"]["start"] == [8, 8]


def test_helpers_and_non_module_classes_do_not_become_layers(client):
    helpers = """def make_value(): return 4
def UpperHelper(): return 4
class Settings: pass
"""
    root = ROOT.replace("import blocks", "import blocks\nimport helpers").replace(
        "        self.block =", "        self.value = helpers.make_value()\n"
        "        self.upper = helpers.UpperHelper()\n"
        "        self.settings = helpers.Settings()\n"
        "        self.block =")
    data = project(client, root, {"blocks.py": BLOCKS, "helpers.py": helpers})
    nodes, edges = semantic(data)
    assert set(nodes) == {"input", "block", "block.proj", "head", "output"}
    assert not data["diagnostics"]
    assert ("input", "block") in edges


def test_parser_does_not_invent_layers_from_ordinary_dotted_helpers():
    root = ROOT.replace("        self.block = blocks.ScaleBlock()", "        self.size = helpers.size()\n"
                        "        self.value = torch.ones(4)\n        self.ready = self.prepare()")
    parsed = CodeParserService().parse_pytorch_code(CodeParseRequest(code=root))
    assert {n.id for n in parsed["graphs"][0]["graph"].nodes} == {"input", "head", "output"}


def test_annotated_dotted_constructor_keeps_multiline_declaration_span(client):
    root = ROOT.replace("self.block = blocks.ScaleBlock()", "self.block: nn.Module = blocks.ScaleBlock(\n            width=4,\n        )")
    data = project(client, root)
    span = data["spanmap"]["block"]
    assert span["kind"] == "decl" and span["start"] == [8, 8] and span["end"] == [10, 9]
    assert semantic(data)[0]["block"]["params"]["width"] == 4


def test_dotted_repeated_instances_keep_distinct_ids_and_shared_definition_spans(client):
    root = ROOT.replace("        self.head =", "        self.second = blocks.ScaleBlock()\n        self.head =")
    root = root.replace("self.head(self.block(x))", "self.head(self.second(self.block(x)))")
    data = project(client, root)
    nodes, edges = semantic(data)
    assert nodes["second"]["params"]["_defined_in"] == "blocks.py"
    assert data["spanmap"]["second.proj"] == data["spanmap"]["block.proj"]
    assert data["spanmap"]["second"] != data["spanmap"]["block"]
    assert ("block", "second") in edges and ("block.proj", "second.proj") in edges


def test_workspace_projection_never_executes_constructor_or_module_source(client, tmp_path):
    marker = tmp_path / "executed"
    blocks = f"open({str(marker)!r}, 'w').write('executed')\n" + BLOCKS
    nodes, _ = semantic(project(client, sources={"blocks.py": blocks}))
    assert "block.proj" in nodes
    assert not marker.exists()


def test_unknown_lowercase_constructor_used_in_forward_stays_visible(client):
    data = project(client, ROOT.replace("blocks.ScaleBlock", "blocks.alias"), {})
    nodes, edges = semantic(data)
    assert nodes["block"]["params"]["_opaque"] is True
    assert any(d.get("node") == "block" for d in data["diagnostics"])
    assert edges == {("input", "block"), ("block", "head"), ("head", "output")}


def test_unknown_module_ancestry_does_not_erase_the_constructor(client):
    blocks = BLOCKS.replace("nn.Module", "unavailable.Base")
    data = project(client, sources={"blocks.py": blocks})
    nodes, _ = semantic(data)
    assert nodes["block"]["params"]["_opaque"] is True
    assert any(d.get("node") == "block" for d in data["diagnostics"])


def test_unused_dotted_block_has_no_invented_forward_edges(client):
    data = project(client, ROOT.replace("self.head(self.block(x))", "self.head(x)"))
    nodes, edges = semantic(data)
    assert "block" in nodes and "block.proj" in nodes
    assert edges == {("input", "head"), ("head", "output")}


def test_missing_forward_does_not_invent_a_chain_through_dotted_constructors(client):
    root = ROOT.split("    def forward", 1)[0]
    nodes, edges = semantic(project(client, root))
    assert "block" in nodes and "block.proj" in nodes
    assert edges == set()


@pytest.mark.parametrize("namespace", ["blocks", "torch"])
def test_dotted_class_colliding_with_torch_does_not_borrow_its_signature(client, namespace):
    root = ROOT.replace("import blocks", f"import blocks as {namespace}").replace(
        "blocks.ScaleBlock()", f"{namespace}.Linear(7, width=4)")
    blocks = BLOCKS.replace("ScaleBlock", "Linear")
    nodes, _ = semantic(project(client, root, {"blocks.py": blocks}))
    assert nodes["block"]["op"] == "custom_code"
    assert nodes["block"]["params"]["arg_0"] == 7
    assert "in_features" not in nodes["block"]["params"]


def test_dotted_wrapper_and_member_edit_their_own_declarations(client):
    root = ROOT.replace("blocks.ScaleBlock()", "blocks.ScaleBlock(width=4)  # keep custom call")
    data = project(client, root)
    for node, source, file, cls, param, value, before, after in [
        ("block", root, "dotted_model.py", "DottedNet", "width", 8, "width=4", "width=8"),
        ("block.proj", BLOCKS, "blocks.py", "ScaleBlock", "out_features", 8, "Linear(4, 4)", "Linear(4, 8)"),
    ]:
        local_id = node.split(".")[-1]
        result = code_apply.apply_operations(source, {local_id: data["spanmap"][node]},
            [{"op": "set_param", "op_id": "f7", "node_id": local_id, "param": param, "value": value}], cls=cls, file=file)
        assert result.refused == []
        assert result.source == source.replace(before, after)
        ast.parse(result.source)


@pytest.mark.parametrize("namespace", ["blocks", "torch"])
def test_dotted_custom_edit_never_guesses_a_torch_positional_slot(client, namespace):
    root = ROOT.replace("import blocks", f"import blocks as {namespace}").replace(
        "blocks.ScaleBlock()", f"{namespace}.Linear(7, 4)  # custom meanings")
    data = project(client, root, {"blocks.py": BLOCKS.replace("ScaleBlock", "Linear")})
    result = code_apply.apply_operations(root, {"block": data["spanmap"]["block"]},
        [{"op": "set_param", "op_id": "f7", "node_id": "block", "param": "out_features", "value": 8}],
        cls="DottedNet", file="dotted_model.py")
    assert result.refused[0]["reason"] == code_apply.UNSUPPORTED_OPERATION
    assert result.source == root and result.edits == []


@pytest.mark.parametrize("argument,refused", [("7", False), ("choose_width()", True)])
def test_custom_positional_edit_uses_source_position_and_preserves_python(client, argument, refused):
    root = ROOT.replace("blocks.ScaleBlock()", f"blocks.ScaleBlock({argument}, 4)  # keep")
    data = project(client, root)
    result = code_apply.apply_operations(root, {"block": data["spanmap"]["block"]},
        [{"op": "set_param", "op_id": "f7", "node_id": "block", "param": "arg_0", "value": 8}],
        cls="DottedNet", file="dotted_model.py")
    if refused:
        assert result.refused[0]["reason"] == code_apply.NON_LITERAL_VALUE
        assert result.source == root and result.edits == []
    else:
        assert result.refused == []
        assert result.source == root.replace("ScaleBlock(7, 4)", "ScaleBlock(8, 4)")


def test_custom_python_argument_is_refused_without_rewriting_source(client):
    root = ROOT.replace("blocks.ScaleBlock()", "blocks.ScaleBlock(width=choose_width())")
    data = project(client, root)
    result = code_apply.apply_operations(root, {"block": data["spanmap"]["block"]},
        [{"op": "set_param", "op_id": "f7", "node_id": "block", "param": "width", "value": 8}],
        cls="DottedNet", file="dotted_model.py")
    assert result.refused[0]["reason"] == code_apply.NON_LITERAL_VALUE
    assert result.source == root and result.edits == []


def test_custom_starred_positional_edit_is_refused(client):
    root = ROOT.replace("blocks.ScaleBlock()", "blocks.ScaleBlock(*widths, 4)")
    data = project(client, root)
    result = code_apply.apply_operations(root, {"block": data["spanmap"]["block"]},
        [{"op": "set_param", "op_id": "f7", "node_id": "block", "param": "arg_1", "value": 8}],
        cls="DottedNet", file="dotted_model.py")
    assert result.refused[0]["reason"] == code_apply.UNSUPPORTED_OPERATION
    assert result.source == root and result.edits == []


def test_index_identifies_reexported_helper_without_executing_it():
    index = ModuleIndex.build({"net.py": "import api\n", "api.py": "from helpers import UpperHelper\n",
                               "helpers.py": "raise RuntimeError('never execute')\ndef UpperHelper(): return 4\n"})
    visited = set()
    result = index.resolve("net", "api.UpperHelper", visited=visited)
    assert result.non_class is True
    assert result.definition is None
    assert visited == {"net.py", "api.py", "helpers.py"}
