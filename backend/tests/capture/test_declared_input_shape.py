"""The input shape a file declares, obeyed by the tier that DRAWS the graph.

The trace worker has always obeyed ``# tensorviz: input=2,8``. The AST tier did
not, so a model whose input it could not infer was drawn against the image
default ``1,3,224,224`` — and a correct model opened as a canvas covered in shape
errors that the first save silently cleared. A user's first impression of their
own working model was a wall of red.

These tests pin the order of precedence and the one case the drawing tier must
NOT try to read, because reading it needs the torch this tier promises not to
import.
"""

from __future__ import annotations



from models.graph_models import CodeParseRequest  # noqa: E402
from utils.code_parser import CodeParserService, declared_input_shape  # noqa: E402

#: A model whose input cannot be inferred: the layer widths are a parameter the
#: file never gives a value to, and there is no example tensor anywhere in it.
#: The parameter carries no default on purpose — a default IS a value the file
#: states, and the drawing tier reads it, so `width=8` would make this guessable.
UNGUESSABLE = '''import torch.nn as nn


class Net(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.l1 = nn.Linear(width, width)

    def forward(self, x):
        return self.l1(x)


# tensorviz: input=2,8
'''


def _input_shape(code: str, **kwargs) -> str:
    parser = CodeParserService()
    result = parser.parse_pytorch_code(CodeParseRequest(code=code, **kwargs))
    assert result.get("success"), result
    graph = next(e["graph"] for e in result["graphs"] if e.get("graph"))
    node = next(n for n in graph.nodes if n.id == "input")
    return node.data.parameters["shape"]


def test_the_drawn_graph_uses_the_shape_the_file_declares():
    assert _input_shape(UNGUESSABLE) == "2,8"


def test_without_a_declaration_or_inference_the_input_stays_unknown():
    # No declaration, literal or example authorizes an image-shaped input.
    assert _input_shape(UNGUESSABLE.replace("# tensorviz: input=2,8", "")) == ""


def test_an_explicit_request_still_beats_the_file():
    # The caller asked for something specific. A comment in the file is a default
    # for when nobody asked, not an override of somebody who did.
    assert _input_shape(UNGUESSABLE, input_shape=[4, 8]) == "4,8"


def test_the_declaration_beats_an_inferred_shape():
    # A Linear whose first argument IS a literal gives the inference something to
    # work with. The declaration must still win: it is the only line where the
    # user states what the model is fed.
    literal = UNGUESSABLE.replace("nn.Linear(width, width)", "nn.Linear(16, 16)")
    assert _input_shape(literal.replace("input=2,8", "input=2,16")) == "2,16"
    assert _input_shape(literal.replace("# tensorviz: input=2,8", "")) == "1,16"


def test_the_declaration_beats_an_example_tensor_in_the_same_file():
    """The one case that separates "declared" from "discovered".

    A ``torch.randn(...)`` at module level is an example the user wrote for
    themselves; the comment is a sentence addressed to tensorViz. When a file has
    both and they disagree, the sentence addressed to us wins. Without this case
    the precedence is untestable, because a file with no example tensor gives the
    same answer whichever order the two are consulted in.
    """
    both = UNGUESSABLE.replace(
        "# tensorviz: input=2,8",
        "example = torch.randn(4, 3, 32, 32)\n# tensorviz: input=2,8",
    ).replace("import torch.nn as nn", "import torch\nimport torch.nn as nn")

    assert _input_shape(both) == "2,8"
    # And with the sentence removed, the example is what is discovered — which is
    # what proves the two sources are genuinely different and both are consulted.
    assert _input_shape(both.replace("# tensorviz: input=2,8", "")) == "4,3,32,32"


def test_the_expression_form_is_left_alone_rather_than_guessed_at():
    # `torch.randint(0, 50257, (1, 16))` is valid for the worker, which has torch.
    # This tier does not, so it must decline rather than half-read the line.
    assert declared_input_shape("# tensorviz: input=torch.randint(0, 5, (1, 16))") is None
    assert declared_input_shape("# tensorviz: input=") is None
    assert declared_input_shape("") is None


def test_both_tiers_read_declarations_with_one_stdlib_only_rule():
    """The drawing and the worker share one reader, so they cannot drift.

    The worker cannot import the parser. The worker runs with the user's own
    repository first on ``sys.path``, and importing ``utils.code_parser`` pulls in
    ``models.graph_models`` — which any workspace with a ``models/`` package of its
    own, meaning most of them, would shadow. That is not hypothetical: it was
    tried, and every worker test failed with the fixture's own ``models`` package
    shadowing the backend's.

    So both import ``input_declarations``, the worker relatively, and that module
    imports nothing but the standard library. If the two ever stopped sharing it,
    the graph that is drawn and the graph that is run could disagree about which
    class a declaration belongs to, which is the bug this whole file exists for.
    """
    import ast
    import sys
    from pathlib import Path

    from services.ingest import input_declarations, service
    from utils import code_parser

    assert service.input_declarations is input_declarations
    assert code_parser.applicable is input_declarations.applicable
    tree = ast.parse(Path(input_declarations.__file__).read_text(encoding="utf-8"))
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names}
    imported |= {node.module.split(".")[0] if node.level == 0 else "." for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom)}
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}, imported


# ── one model's declaration is not another's input ───────────────────────────
#
# Setup reads a declaration as belonging to the model class it sits above or
# inside. The drawing took the first declaration in the file for every class, and
# the worker gave a declaration outside every class to every class, so the same
# class could be drawn, saved and run with three different inputs.

#: Two models with different input widths, A ``Linear(8, 4)`` and B ``Linear(16, 4)``.
TWO_MODELS = '''import torch.nn as nn


{a}class A(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc(x)


{b}class B(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc(x)
'''
EACH = TWO_MODELS.format(a="# tensorviz: input=2,8\n", b="# tensorviz: input=3,16\n")
ONLY_A = TWO_MODELS.format(a="# tensorviz: input=2,8\n", b="")
#: Not directly above or inside either class, so it cannot say which one it is for.
STRAY = "# tensorviz: input=2,8\n" + TWO_MODELS.format(a="", b="")


def _drawn(code: str, cls: str, **kwargs) -> str:
    result = CodeParserService().parse_pytorch_code(CodeParseRequest(code=code, **kwargs))
    assert result.get("success"), result
    graph = next(e["graph"] for e in result["graphs"] if e["name"] == cls)
    return next(n for n in graph.nodes if n.id == "input").data.parameters["shape"]


def test_each_model_class_is_drawn_with_its_own_declaration():
    assert (_drawn(EACH, "A"), _drawn(EACH, "B")) == ("2,8", "3,16")
    assert declared_input_shape(EACH, "B") == [3, 16]


def test_a_class_without_a_declaration_of_its_own_is_not_drawn_with_another_classs():
    # B infers its own width instead of being fed A's eight.
    assert (_drawn(ONLY_A, "A"), _drawn(ONLY_A, "B")) == ("2,8", "1,16")
    assert declared_input_shape(ONLY_A, "B") is None


def test_a_declaration_that_belongs_to_no_class_is_drawn_for_none():
    assert (_drawn(STRAY, "A"), _drawn(STRAY, "B")) == ("1,8", "1,16")
    assert declared_input_shape(STRAY, "A") is None


def test_declaration_text_inside_a_string_is_not_a_declaration():
    docstring = '"""\n# tensorviz: input=5,8\n"""\n'
    assert declared_input_shape(docstring) is None
    assert _input_shape(docstring + UNGUESSABLE.replace("# tensorviz: input=2,8", "")) == ""


def test_carriage_return_line_breaks_are_read_as_python_reads_them():
    # Python reads a lone carriage return as a line break. Tokenizing the comments line
    # by line did not, so valid source was refused as a syntax error or not drawn.
    note = ONLY_A.replace("A(nn.Module):\n    def", "A(nn.Module):\n    # a note\r    def")
    old_mac = EACH.replace("\n", "\r")
    assert (_drawn(note, "A"), _drawn(note, "B")) == ("2,8", "1,16")
    assert (_drawn(old_mac, "A"), _drawn(old_mac, "B")) == ("2,8", "3,16")
    assert declared_input_shape(old_mac, "B") == [3, 16]


def _run_with(tmp_path, source: str, cls: str, encoding: str = "utf-8"):
    """The shapes the worker feeds ``cls`` when nothing else supplies an input."""
    from services.ingest import service

    model = tmp_path / "model.py"
    model.write_text(source, encoding=encoding)
    recipe = service._input_from_comment(str(model), cls)
    return None if recipe is None else [tuple(tensor.shape) for tensor in recipe[0]]


def test_the_worker_feeds_each_class_the_declaration_the_drawing_gives_it(tmp_path):
    assert (_run_with(tmp_path, EACH, "A"), _run_with(tmp_path, EACH, "B")) == ([(2, 8)], [(3, 16)])
    # It used to hand A's declaration, and a stray one, to B as well.
    assert _run_with(tmp_path, ONLY_A, "B") is None
    assert _run_with(tmp_path, STRAY, "B") is None
    # One model in the file: wherever the declaration sits, it is that model's.
    assert _run_with(tmp_path, UNGUESSABLE, "Net") == [(2, 8)]


def test_the_worker_reads_a_file_saved_with_a_byte_order_mark_by_the_same_rule(tmp_path):
    # Python imports a file saved as "UTF-8 with BOM", but a BOM left in the text is
    # a syntax error to ast, so the worker fell back to the file's first declaration
    # and fed B what A or no class declared.
    bom = dict(encoding="utf-8-sig")
    assert _run_with(tmp_path, EACH, "B", **bom) == [(3, 16)]
    assert _run_with(tmp_path, ONLY_A, "B", **bom) is None
    assert _run_with(tmp_path, STRAY, "B", **bom) is None


def test_the_worker_still_reads_a_declaration_in_a_file_python_cannot_parse(tmp_path):
    # A buffer mid-edit still deserves whatever the file declares.
    assert _run_with(tmp_path, "class B(:\n    # tensorviz: input=3,16\n", "B") == [(3, 16)]
    # So does source nested too deep for ast to build, which raises RecursionError.
    too_deep = ONLY_A + "print(" + "+".join(["1"] * 100_000) + ")\n"
    assert _run_with(tmp_path, too_deep, "B") == [(2, 8)]


def test_the_worker_reads_each_class_its_own_in_a_file_too_deep_to_prove_modules_in(tmp_path):
    # Python imports a module-level sum of 2,000 terms, but proving which classes are
    # modules walks it a frame per term and ran out of stack, so the trace failed.
    deep = "print(" + "+".join(["1"] * 2_000) + ")\n"
    assert (_run_with(tmp_path, EACH + deep, "A"), _run_with(tmp_path, EACH + deep, "B")) == ([(2, 8)], [(3, 16)])
    assert _run_with(tmp_path, ONLY_A + deep, "B") is None
