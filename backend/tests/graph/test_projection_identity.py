"""Node identity under edits (task.md D6).

The pivot's promise is "edit either, the other follows". That only holds if a
node keeps its identity across the edits a person actually makes, because a node
id is what positions, AI-turn ops, debugger overlays and the ``.tensorviz/views``
sidecar are all keyed by. An id that moves when a blank line is inserted throws
away the user's layout every time they hit Enter.

So the two edits below are the contract, stated as a diff of id sets:

  * inserting a blank line above ``forward`` changes **no** id and shifts every
    ``call`` span by exactly one line, leaving ``decl`` spans alone;
  * renaming ``self.l1`` to ``self.proj`` removes exactly ``{'l1'}`` and adds
    exactly ``{'proj'}``.

These are asserted through ``services.ide.projection`` directly rather than over
HTTP: the HTTP contract is ``test_ide_router.py``'s job, and a failure here
should name the rule, not the route.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, Set

import pytest


from models.graph_models import CodeParseRequest  # noqa: E402
from services.ide import projection  # noqa: E402
from utils.code_parser import CodeParserService  # noqa: E402

RESID = Path(__file__).parent.resolve().parent / "fixtures" / "ide_repo" / "models" / "resid.py"


def _project(source: str, cls: str = "Net", rel_file: str = "models/resid.py"):
    parsed = CodeParserService().parse_pytorch_code(
        CodeParseRequest(code=source, file=rel_file)
    )
    entry = next(e for e in parsed["graphs"] if e["name"] == cls)
    graph = entry["graph"]
    return projection.project(
        {
            "nodes": [n.model_dump() for n in graph.nodes],
            "edges": [e.model_dump() for e in graph.edges],
        },
        cls=cls,
        rel_file=rel_file,
        source=source,
    )


def _ids(result) -> Set[str]:
    return {n.node_id for n in result.document.semantic.nodes}


def test_edited_unique_assignment_keeps_binding_evidence_but_changes_source_digest():
    source = """import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def forward(self, student_output, eps=1e-8):
        student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
        return student_output
"""
    before = _project(source)
    after = _project(source.replace("normalize(student_output,", "normalize(student_output.float(),"))
    op = "python@Net#0"
    left = next(n for n in before.document.semantic.nodes if n.node_id == op)
    right = next(n for n in after.document.semantic.nodes if n.node_id == op)
    assert left.params["_review_binding"] == right.params["_review_binding"] == ["Net", "forward", "student_output"]
    assert before.spanmap[op]["digest"] != after.spanmap[op]["digest"]
    assert left.params["_source_calls"] == ["normalize"]
    assert right.params["_source_calls"] == ["float", "normalize"]
    repeated = _project(source.replace("        return student_output", "        student_output = student_output.detach()\n        return student_output"))
    assert all("_review_binding" not in n.params for n in repeated.document.semantic.nodes)


def test_multiple_operations_in_one_assignment_are_not_given_one_binding_identity():
    source = """import torch.nn as nn
class Net(nn.Module):
    def forward(self, x):
        result = (x * 2) + x
        return result
"""
    result = _project(source)
    assert {"mul@Net#0", "add@Net#0"} <= _ids(result)
    assert all("_review_binding" not in n.params for n in result.document.semantic.nodes)


@pytest.mark.parametrize(("source", "calls"), [
    ("F.normalize(x.float())", ["float", "normalize"]),
    ("x.view(1).transpose(0, 1)", ["view", "transpose"]),
    ("torch.cat([x.clone(), y.clone()])", ["clone", "clone", "cat"]),
    ("F.normalize(x.float() if flag else x)", None),
    ("x.ready() and x.float()", None),
])
def test_call_labels_preserve_evaluation_order_without_claiming_conditional_execution(source, calls):
    params = CodeParserService._provenance({"call": ast.parse(source, mode="eval").body}, None)
    assert params.get("_source_calls") == calls


@pytest.mark.parametrize(("expression", "name"), [
    ("torch.where(masks.unsqueeze(-1), token.to(x.dtype).unsqueeze(0), x)", "where"),
    ("F.normalize(x.float())", "normalize"),
    ("x.view(1).transpose(0, 1)", "transpose"),
    ("finish(x.clone())", "finish"),
    ("F.normalize(x.float() if flag else x)", "normalize"),
])
def test_statement_title_names_the_result_call_not_nested_argument_preparation(expression, name):
    params = CodeParserService._provenance({"call": ast.parse(expression, mode="eval").body}, None)
    assert params["_source_result_call"] == {"name": name}


@pytest.mark.parametrize("imports, expression, target", [
    ("import torch", "torch.where(x > 0, x.unsqueeze(-1), x)", "torch.where"),
    ("import torch as th", "th.where(x > 0, x, x)", "torch.where"),
    ("import torch\nimport torch.utils.checkpoint", "torch.where(x > 0, x, x)", "torch.where"),
    ("import torch\nimport other as torch", "torch.where(x > 0, x, x)", None),
    ("from torch import where as choose", "choose(x > 0, x, x)", "torch.where"),
    ("from torch.nn.functional import normalize as norm", "norm(x.float())", "torch.nn.functional.normalize"),
    ("import torch\ntorch = other", "torch.where(x > 0, x, x)", None),
])
def test_result_call_labels_use_proven_imports_only(imports, expression, target):
    result = _project(f'{imports}\nfrom torch import nn\nclass Net(nn.Module):\n    def forward(self, x):\n        return {expression}\n')
    node = next(node for node in result.document.semantic.nodes if node.node_id.startswith('python@'))
    assert node.params['_source_result_call'].get('target') == target


def _spans_of_kind(spanmap: Dict[str, Dict[str, Any]], kind: str) -> Dict[str, int]:
    return {k: v["start"][0] for k, v in spanmap.items() if v["kind"] == kind}


@pytest.fixture()
def source() -> str:
    return RESID.read_text(encoding="utf-8")


def test_the_fixture_projects_to_the_seven_nodes_the_milestone_names(source: str):
    """M3's acceptance quotes ``nodeCount: 7`` for ``resid.py::Net``. If the
    fixture is edited so this changes, the Electron test's number is wrong too."""
    result = _project(source)
    assert len(_ids(result)) == 7


def test_inserting_a_blank_line_moves_spans_and_no_ids(source: str):
    before = _project(source)
    edited = source.replace("    def forward(self, x):", "\n    def forward(self, x):")
    assert edited != source
    after = _project(edited)

    assert _ids(after) == _ids(before)

    # Everything forward() produced moved down exactly one line…
    calls_before = _spans_of_kind(before.spanmap, "call")
    calls_after = _spans_of_kind(after.spanmap, "call")
    assert calls_before, "the fixture must contain at least one forward-produced node"
    assert calls_after == {node_id: line + 1 for node_id, line in calls_before.items()}

    # …and everything __init__ declared stayed where it was.
    assert _spans_of_kind(after.spanmap, "decl") == _spans_of_kind(before.spanmap, "decl")

    # The source changed, so the digest of the buffer must have.
    assert after.source_digest != before.source_digest


def test_renaming_an_attribute_is_one_removed_and_one_added(source: str):
    before = _project(source)
    edited = source.replace("self.l1", "self.proj")
    assert edited.count("self.proj") == 2, "rename must touch the declaration and the call"
    after = _project(edited)

    assert _ids(before) - _ids(after) == {"l1"}
    assert _ids(after) - _ids(before) == {"proj"}


def test_modulelist_member_edit_and_rename_preserve_source_and_flow_identity(source: str):
    before = _project(source)
    edited = _project(source.replace("depth=2", "depth=3"))
    assert _ids(edited) == _ids(before)
    assert edited.document.semantic.edges == before.document.semantic.edges
    assert edited.spanmap["blocks"] == before.spanmap["blocks"]
    container = next(n for n in edited.document.semantic.nodes if n.node_id == "blocks")
    assert container.kind == "module"
    assert len(container.params["_modules"]) == 3

    renamed = _project(source.replace("self.blocks", "self.stages"))
    assert _ids(before) - _ids(renamed) == {"blocks"}
    assert _ids(renamed) - _ids(before) == {"stages"}
    assert renamed.document.semantic.edges == before.document.semantic.edges
    assert renamed.spanmap["stages"]["start"] == before.spanmap["blocks"]["start"]
    assert renamed.document.view.layout["node_data"]["stages"]["moduleInstanceId"] == "stages"


def test_an_op_added_earlier_in_forward_does_not_renumber_a_different_op(source: str):
    """Why ``fn@owner#ordinal`` is per-``fn`` and not one counter.

    The parser numbers every synthesised node from a single counter, so inserting
    one call at the top of ``forward`` used to renumber every node after it —
    ``add_1`` became ``add_2`` — and the canvas lost the position of a card the
    user never touched.
    """
    before = _project(source)
    edited = source.replace(
        "        h = self.l1(x)",
        "        x = x.flatten(1)\n        h = self.l1(x)",
    )
    after = _project(edited)

    assert "flatten@Net#0" in _ids(after)
    assert _ids(after) - _ids(before) == {"flatten@Net#0"}
    assert _ids(before) - _ids(after) == set()


def test_the_span_digest_ignores_formatting_and_not_meaning(source: str):
    """The digest hashes ``ast.unparse`` of the statement, so reformatting a line
    is free — the node does not move and no downstream consumer is invalidated —
    while changing what the line computes is visible."""
    before = _project(source)

    reformatted = _project(source.replace("h = block(h)", "h = block( h )"))
    assert reformatted.spanmap["python@Net#0"]["digest"] == before.spanmap["python@Net#0"]["digest"]

    edited = _project(source.replace("h = block(h)", "h = block(h * 1)"))
    assert edited.spanmap["python@Net#0"]["digest"] != before.spanmap["python@Net#0"]["digest"]
    # Source projection now preserves scalar arithmetic. Existing identities stay
    # stable, and the authored multiplication has its own node and operand.
    assert _ids(edited) - _ids(before) == {"mul@Net#0"}
    assert _ids(before) - _ids(edited) == set()
    multiply = next(n for n in edited.document.semantic.nodes if n.node_id == "mul@Net#0")
    assert multiply.params["other"] == 1
    assert multiply.params["_scalar_side"] == "right"


def test_ops_carry_the_kind_that_says_how_much_of_them_is_modelled(source: str):
    result = _project(source)
    kinds = {
        n.node_id: n.params.get("_kind")
        for n in result.document.semantic.nodes
        if n.node_id not in projection.BOUNDARY_IDS
    }
    assert kinds["l1"] == "class"            # a declared submodule
    assert kinds["add@Net#0"] == "function"  # a torch op the parser typed
    assert kinds["python@Net#0"] == "statement"  # `block(h)`: deliberately unmodelled


def test_the_graph_id_is_derived_from_the_path_and_the_class(source: str):
    a = projection.graph_id_for("models/resid.py", "Net")
    assert a == projection.graph_id_for("models\\resid.py", "Net")
    assert a == projection.graph_id_for("/models/resid.py", "Net")
    assert a != projection.graph_id_for("models/resid.py", "Gate")
    assert a != projection.graph_id_for("other/resid.py", "Net")
    assert a.startswith("ide:") and a.endswith("#Net")


def test_projecting_a_class_the_file_does_not_define_is_refused(source: str):
    with pytest.raises(projection.ProjectionError):
        projection.init_attributes(source, "Missing")
