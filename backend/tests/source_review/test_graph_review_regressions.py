"""Regressions discovered by 100 real Git base/head comparisons."""
import hashlib
import pytest
import torch

from tests.graph.test_ide_multifile_projection import client
from tests.codegen.test_code_parser_dataflow import _graph, _by_op
from services.code_generation import CodeGenerationService
from services.verification_service import _apply_merge
from core.shape_inference import _apply_analytic_merge


def project(client, source, extra=None, **options):
    files = {"model.py": source, **(extra or {})}
    entries = [dict(file=file, source=text, digest=hashlib.sha256(text.encode()).hexdigest()) for file, text in files.items()]
    return client.post("/api/ide/multifile/project", json={"protocol_version": 1,
        "root": {"file": "model.py", "cls": "Net", "source_digest": entries[0]["digest"]}, "sources": entries, **options})


BASE = '''from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Dropout(0.1)
    def forward(self, x):
        return self.layer(x)
'''


def test_git_review_does_not_overwrite_or_advance_the_live_projection(client):
    from api.routers import ide
    live = project(client, BASE).json()
    stored = ide._store.get_graph('local-dev-user', live['graph_id'])
    assert stored is not None
    for source in [BASE, BASE.replace('Dropout(0.1)', 'Dropout(0.2)')]:
        review = project(client, source, review_only=True)
        assert review.status_code == 200, review.text
        assert review.json()['graph_id'] != live['graph_id']
        assert review.json()['revision'] == 0
        assert ide._store.get_graph('local-dev-user', review.json()['graph_id']) is None
        assert ide._store.get_graph('local-dev-user', live['graph_id']) == stored
    assert project(client, BASE, review_only=True, trace_result={}).status_code == 400


def test_constructor_defaults_survive_review_when_dynamic_layers_cannot_be_expanded(client):
    source = '''from torch import nn
raise RuntimeError("review must never execute source")
class Net(nn.Module):
    def __init__(self, ffn_layer="mlp", *, depth=12):
        """Constructor documentation."""
        super().__init__()
        self.blocks = nn.ModuleList(make_blocks(ffn_layer, depth))
    def forward(self, x):
        return self.blocks[0](x)
'''
    before = project(client, source).json()["document"]["semantic"]["metadata"]["reviewClass"]
    after = project(client, source.replace('ffn_layer="mlp"', 'ffn_layer="swiglu"')).json()["document"]["semantic"]["metadata"]["reviewClass"]
    assert before["defaults"] == {"ffn_layer": "'mlp'", "depth": "12"}
    assert after["defaults"]["ffn_layer"] == "'swiglu'"
    assert before["digest"] != after["digest"]
    documented = project(client, source.replace("Constructor documentation.", "Updated docs.") + "\n# comment\n").json()
    assert documented["document"]["semantic"]["metadata"]["reviewClass"]["digest"] == before["digest"]


def test_review_keeps_each_conditional_declaration_even_when_graph_lowering_omits_it(client):
    source = BASE.replace("def __init__(self):", "def __init__(self, project=True):").replace(
        "self.layer = nn.Dropout(0.1)",
        "self.layer = nn.Dropout(0.1)\n"
        "        self.output_projection = nn.Linear(4, 4) if project else nn.Identity()\n"
        "        self.output_linear = nn.Linear(4, 4) if project else nn.Identity()")
    result = project(client, source, review_only=True).json()
    members = result["document"]["semantic"]["metadata"]["reviewClass"]["members"]
    for name in ("output_projection", "output_linear"):
        assert members[name]["digest"]
        span = result["spanmap"][name]
        assert span["kind"] == "decl"
        assert f"self.{name} = nn.Linear" in source.splitlines()[span["start"][0] - 1]
    changed = source.replace("self.output_linear = nn.Linear(4, 4)", "self.output_linear = nn.Linear(4, 8)")
    after = project(client, changed, review_only=True).json()["document"]["semantic"]["metadata"]["reviewClass"]["members"]
    assert after["output_linear"]["digest"] != members["output_linear"]["digest"]
    assert after["output_projection"]["digest"] == members["output_projection"]["digest"]


def test_review_retains_snapshot_digests_beyond_the_statically_expanded_files(client):
    extra = {"factory.py": "# used during capture, not expanded by the source graph\n"}
    result = project(client, BASE, extra, review_only=True).json()
    assert {entry["file"]: entry["digest"] for entry in result["dependencies"]} == {
        file: hashlib.sha256(text.encode()).hexdigest() for file, text in {"model.py": BASE, **extra}.items()}


def test_review_declarations_keep_conditions_and_exclude_nested_scopes():
    from services.ide.projection import review_class
    source = BASE.replace("self.layer = nn.Dropout(0.1)",
        "if enabled:\n            self.layer = nn.Dropout(0.1)\n"
        "        def nested(self):\n            self.foreign = nn.Linear(4, 4)")
    before = review_class(source, "Net", "model.py")["members"]
    after = review_class(source.replace("if enabled:", "if not enabled:"), "Net", "model.py")["members"]
    assert set(before) == {"layer"}
    assert before["layer"]["digest"] != after["layer"]["digest"]
    assert before["layer"]["digest"] == review_class("\n\n" + source, "Net", "model.py")["members"]["layer"]["digest"]


def test_review_members_inside_try_with_and_match_blocks_keep_their_guards():
    from services.ide.projection import review_class
    source = BASE.replace("def __init__(self):", "def __init__(self, mode=8):").replace("self.layer = nn.Dropout(0.1)", "\n        ".join([
        "self.layer = nn.Dropout(0.1)",
        "try:", "    self.attn = FlashMHA(4)", "except ImportError:", "    self.attn = nn.MultiheadAttention(4, 1)",
        "else:", "    self.ready = True", "finally:", "    self.tried = True",
        'with torch.device("meta"):', "    self.head = nn.Linear(4, 2)",
        "match mode:", "    case 8:", "        self.out = nn.Identity()",
        "try:", "    pass", "except* ValueError:", "    self.fallback = nn.ReLU()",
    ]))
    members = review_class(source, "Net", "model.py")["members"]
    assert set(members) == {"layer", "attn", "ready", "tried", "head", "out", "fallback"}
    # Two alternative writes stay ambiguous, as an if/else pair does.
    assert members["attn"]["ambiguous"] and not members["head"]["ambiguous"]
    assert "self.head = nn.Linear(4, 2)" in source.split("\n")[members["head"]["span"]["start"][0] - 1]
    for before, after, name in [("except ImportError", "except Exception", "attn"),
                                ('torch.device("meta")', 'torch.device("cpu")', "head"),
                                ("case 8", "case 16", "out"), ("except* ValueError", "except* TypeError", "fallback")]:
        changed = review_class(source.replace(before, after), "Net", "model.py")["members"]
        assert [n for n in members if changed[n]["digest"] != members[n]["digest"]] == [name], before


@pytest.mark.parametrize("wrap", [
    'with torch.device("meta"):\n            {}',
    "try:\n            {}\n        except ImportError:\n            pass",
    "match mode:\n            case 8:\n                {}",
])
def test_review_navigates_to_unlowered_members_inside_compound_blocks(client, wrap):
    statement = "self.output_projection = nn.Linear(4, 4) if project else nn.Identity()"
    source = BASE.replace("def __init__(self):", "def __init__(self, project=True, mode=8):").replace(
        "self.layer = nn.Dropout(0.1)", "self.layer = nn.Dropout(0.1)\n        " + wrap.format(statement))
    result = project(client, source, review_only=True).json()
    assert result["document"]["semantic"]["metadata"]["reviewClass"]["members"]["output_projection"]["digest"]
    span = result["spanmap"]["output_projection"]
    assert span["kind"] == "decl" and statement in source.split("\n")[span["start"][0] - 1]


def test_aliases_and_keyword_equivalence_keep_constructor_meaning_and_source_spans(client):
    before = project(client, BASE).json()
    after_source = BASE.replace("from torch import nn", "from torch import nn as layers").replace("nn.", "layers.").replace("Dropout(0.1)", "Dropout(p=0.1)")
    after = project(client, after_source)
    assert after.status_code == 200, after.text
    before_layer = next(n for n in before["document"]["semantic"]["nodes"] if n["node_id"] == "layer")
    after_layer = next(n for n in after.json()["document"]["semantic"]["nodes"] if n["node_id"] == "layer")
    assert before_layer["params"]["_review_call"] == after_layer["params"]["_review_call"]
    assert after_layer["op"] == "Dropout"
    assert after.json()["diagnostics"] == []
    span = after.json()["spanmap"]["layer"]
    assert "self.layer = layers.Dropout(p=0.1)" in after_source.splitlines()[span["start"][0] - 1]


def test_alias_proof_does_not_admit_rebound_or_non_torch_names(client):
    for import_line in ["from imaginary import nn as layers", "from torch import nn as layers\nlayers = object()"]:
        source = BASE.replace("from torch import nn", import_line).replace("nn.", "layers.")
        assert project(client, source).status_code == 422


def test_imported_alias_module_expands_without_executing_source(client):
    root = BASE.replace("from torch import nn", "from torch import nn as layers\nfrom blocks import Block\nraise RuntimeError('must never execute')").replace("nn.", "layers.").replace("layers.Dropout(0.1)", "Block()")
    child = BASE.replace("class Net", "class Block").replace("from torch import nn", "import torch.nn as layers").replace("nn.", "layers.")
    result = project(client, root, {"blocks.py": child})
    assert result.status_code == 200, result.text
    assert "layer.layer" in result.json()["spanmap"]


def test_unknown_arguments_remain_in_comparison_instead_of_becoming_empty_parameters(client):
    a = project(client, BASE.replace("0.1", "probability(1)")).json()
    b = project(client, BASE.replace("0.1", "probability(2)")).json()
    def call(p):
        return next(n for n in p["document"]["semantic"]["nodes"] if n["node_id"] == "layer")["params"]["_review_call"]
    assert call(a) != call(b)


@pytest.mark.parametrize("operator,op,left", [("*", "mul", False), ("+", "add", False), ("-", "sub", True), ("/", "div", True)])
def test_literal_attribute_arithmetic_is_visible_and_both_runtime_paths_and_export_preserve_it(operator, op, left):
    expression = f"self.scale {operator} x" if left else f"x {operator} self.scale"
    code = BASE.replace("self.layer = nn.Dropout(0.1)", "self.scale = 2.0").replace("return self.layer(x)", f"return {expression}")
    graph = _graph(code, [2, 4])
    nodes = _by_op(graph, op)
    assert len(nodes) == 1
    params = nodes[0].data.parameters
    assert params["other"] == 2.0
    assert params["_scalar_side"] == ("left" if left else "right")
    x = torch.full((2, 4), 4.0)
    expected = getattr(torch, op)(2.0, x) if left else getattr(torch, op)(x, 2.0)
    for run in [_apply_merge, _apply_analytic_merge]:
        assert torch.equal(run(op, [x], params), expected)
    rhs = CodeGenerationService()._emit_op(nodes[0], ["x"])
    assert torch.equal(eval(rhs, {"torch": torch, "x": x}), expected)


def test_a_forward_write_prevents_assuming_the_constructor_literal_is_current():
    code = BASE.replace("self.layer = nn.Dropout(0.1)", "self.scale = 2.0").replace("return self.layer(x)", "self.scale = x.sum()\n        return x * self.scale")
    graph = _graph(code, [2, 4])
    assert not any(n.data.parameters.get("other") == 2.0 for n in graph.nodes)


def test_explicit_torch_defaults_do_not_change_constructor_meaning(client):
    def call(source):
        p = project(client, source).json()
        return next(n for n in p["document"]["semantic"]["nodes"] if n["node_id"] == "layer")["params"]["_review_call"]
    assert call(BASE) == call(BASE.replace("Dropout(0.1)", "Dropout(0.1, inplace=False)"))
    assert call(BASE) != call(BASE.replace("Dropout(0.1)", "Dropout(0.1, inplace=True)"))


def test_unpacked_constructor_arguments_do_not_receive_assumed_defaults(client):
    p = project(client, BASE.replace("Dropout(0.1)", "Dropout(**options)")).json()
    params = next(n for n in p["document"]["semantic"]["nodes"] if n["node_id"] == "layer")["params"]["_review_call"]["arguments"]
    assert "**0" in params
    assert "p" not in params and "inplace" not in params


@pytest.mark.parametrize("layer", ["LSTM", "GRU", "RNN"])
def test_recurrent_constructor_defaults_use_the_real_base_signature(client, layer):
    source = BASE.replace("nn.Dropout(0.1)", f"nn.{layer}(4, 8)")
    def arguments(code):
        graph = project(client, code).json()["document"]["semantic"]
        return next(n for n in graph["nodes"] if n["node_id"] == "layer")["params"]["_review_call"]["arguments"]
    implicit = arguments(source)
    assert implicit["bias"] is True
    assert implicit == arguments(source.replace("(4, 8)", "(4, 8, bias=True)"))
    assert arguments(source.replace("(4, 8)", "(4, 8, bias=False)"))["bias"] is False


@pytest.mark.parametrize("expression,op", [
    ("self.layer(x) + self.layer(x)", "add"),
    ("torch.add(self.layer(x), self.layer(x))", "add"),
    ("torch.mul(self.layer(x), self.layer(x))", "mul"),
    ("torch.cat([self.layer(x), self.layer(x)], dim=-1)", "cat"),
    ("torch.stack([self.layer(x), self.layer(x)], dim=1)", "stack"),
])
def test_repeated_tensor_operands_survive_as_two_value_uses(expression, op):
    source = "import torch\n" + BASE.replace("return self.layer(x)", "return " + expression)
    graph = _graph(source, [2, 4])
    additions = _by_op(graph, op)
    assert len(additions) == 1
    incoming = [edge for edge in graph.edges if edge.target == additions[0].id]
    assert len(incoming) == 2
    assert incoming[0].source == incoming[1].source == "layer"
    assert incoming[0].id != incoming[1].id


@pytest.mark.parametrize("placement", ["root", "child", "inherited"])
@pytest.mark.parametrize("operand", ["self.layer(x)", "x"])
def test_http_projection_keeps_repeated_operand_uses_through_composition(client, placement, operand):
    child = BASE.replace("return self.layer(x)", f"return {operand} + {operand}")
    if placement == "root":
        response = project(client, child)
    elif placement == "child":
        response = project(client, BASE.replace("from torch import nn", "from torch import nn\nfrom blocks import Block").replace("nn.Dropout(0.1)", "Block()"), {"blocks.py": child.replace("class Net", "class Block")})
    else:
        response = project(client, "from blocks import Block\nclass Net(Block):\n    pass\n", {"blocks.py": child.replace("class Net", "class Block")})
    assert response.status_code == 200, response.text
    graph = response.json()["document"]["semantic"]
    addition = next(n for n in graph["nodes"] if n["op"] == "torch.add")
    incoming = [e for e in graph["edges"] if e["target_node_id"] == addition["node_id"]]
    assert len(incoming) == 2
    assert incoming[0]["source_node_id"] == incoming[1]["source_node_id"]
    assert len({e["edge_id"] for e in graph["edges"]}) == len(graph["edges"])


def test_reachable_helper_implementation_changes_are_visible_without_execution():
    a = BASE.replace("return self.layer(x)", "return self.finish(self.layer(x))") + "\n    def finish(self, x):\n        return self.other(x)\n\n    def other(self, x):\n        return torch.relu(x)\n"
    b = a.replace("torch.relu(x)", "torch.sigmoid(x)")
    def helper(source):
        return [(n.data.layerType.name, n.data.parameters.get('_raw_call'))
                for n in _graph(source, [2, 4]).nodes if n.data.parameters.get('_review_scope')]
    assert any(call == 'torch.relu(x)' for _, call in helper(a))
    assert any(call == 'torch.sigmoid(x)' for _, call in helper(b))
    assert helper(a) != helper(b)
    assert helper(a) == helper(a + "\n    def unrelated(self, x):\n        return x * 7\n")


def test_changed_branch_predicate_is_retained_on_the_guarded_layer():
    a = BASE.replace("return self.layer(x)", "if self.training:\n            return self.layer(x)\n        return x")
    b = a.replace("if self.training:", "if not self.training:")
    def guard(source):
        return next(n for n in _graph(source, [2, 4]).nodes if n.id == "layer").data.parameters.get("_review_guards")
    assert guard(a) and guard(b) and guard(a) != guard(b)


@pytest.mark.parametrize("spelling", ["operator", "function"])
def test_attention_matrix_products_bound_dataflow_instead_of_accumulating_all_ancestors(spelling):
    depth = 4
    declarations = "\n".join(f"        self.{name}{i} = nn.Linear(4, 4)" for i in range(depth) for name in ["q", "k"])
    body = []
    for i in range(depth):
        expr = f"(self.q{i}(x) @ self.k{i}(x).transpose(-1, -2)) @ x" if spelling == "operator" else f"torch.matmul(torch.matmul(self.q{i}(x), self.k{i}(x).transpose(-1, -2)), x)"
        body.append("        x = " + expr)
    source = "import torch\nfrom torch import nn\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n" + declarations + "\n    def forward(self, x):\n" + "\n".join(body) + "\n        return x\n"
    graph = _graph(source, [2, 4])
    products = _by_op(graph, "matmul")
    assert len(products) == depth * 2
    assert len(graph.edges) <= depth * 9 + 2
    for node in products:
        assert len([e for e in graph.edges if e.target == node.id]) == 2
    x, y = torch.randn(2, 4), torch.randn(4, 3)
    expression = CodeGenerationService()._emit_op(products[0], ["x", "y"])
    assert torch.equal(eval(expression, {"torch": torch, "x": x, "y": y}), x @ y)


INHERITED = '''from torch import nn
from torchvision.models.resnet import ResNet, Bottleneck
raise RuntimeError("drafts must not execute the model")
class Net(ResNet):
    def __init__(self):
        super().__init__(block=Bottleneck, layers=[3, 8, 36, 3])
'''


def test_source_draft_shows_added_layer_after_opaque_inherited_forward(client):
    after = INHERITED + '''        self.fc2 = nn.Linear(self.fc.out_features, self.fc.out_features)
    def forward(self, x):
        x = super().forward(x)
        return self.fc2(x)
'''
    for source, added in [(INHERITED, False), (after, True)]:
        response = project(client, source, review_only=True)
        assert response.status_code == 200, response.text
        result = response.json()
        assert not result['traced']
        assert result['revision'] == 0
        graph = result['document']['semantic']
        inherited = next(node for node in graph['nodes'] if node['params'].get('_review_inherited'))
        assert inherited['node_id'] == '@inherited-forward'
        assert inherited['op'] == 'custom_code', 'the canvas must show the inherited callable, not a generic Custom title'
        assert inherited['params']['_opaque']
        assert inherited['params']['_class'] == 'ResNet.forward · inherited'
        edges = {(edge['source_node_id'], edge['target_node_id']) for edge in graph['edges']}
        assert ('input', inherited['node_id']) in edges
        if added:
            assert ('input', 'fc2') not in edges, 'the classifier consumes the inherited output, not raw image input'
            assert (inherited['node_id'], 'fc2') in edges
            assert ('fc2', 'output') in edges
            assert next(node for node in graph['nodes'] if node['node_id'] == 'fc2')['op'] == 'Linear'
            assert result['spanmap']['fc2']['kind'] == 'decl'
            assert 'fc2' in graph['metadata']['reviewClass']['members']
        else:
            assert (inherited['node_id'], 'output') in edges
        assert any(note['code'] == 'EXTERNAL_PACKAGE' for note in result['diagnostics'])
        assert all('_out_shape' not in node['params'] for node in graph['nodes'])
    assert project(client, INHERITED).status_code == 422, 'review admission must not silently change the live model path'


def test_inherited_draft_does_not_wire_uncalled_declarations_in_constructor_order(client):
    source = INHERITED + '        self.unused = nn.Linear(4, 4)\n'
    result = project(client, source, review_only=True).json()
    graph = result['document']['semantic']
    assert any(node['node_id'] == 'unused' for node in graph['nodes'])
    assert all(edge['source_node_id'] != 'unused' and edge['target_node_id'] != 'unused' for edge in graph['edges'])


def test_inherited_draft_with_full_forward_override_does_not_invent_base_execution(client):
    source = INHERITED + '''        self.fc2 = nn.Linear(4, 4)
    def forward(self, x):
        return self.fc2(x)
'''
    result = project(client, source, review_only=True).json()
    graph = result['document']['semantic']
    assert not any(node['params'].get('_review_inherited') for node in graph['nodes'])
    assert ('input', 'fc2') in {(edge['source_node_id'], edge['target_node_id']) for edge in graph['edges']}
