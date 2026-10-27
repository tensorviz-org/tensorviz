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


def test_a_comment_above_a_class_belongs_to_that_class_only():
    """A line written above one class is not the other class's example input."""
    source = '''import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 64)

    def forward(self, x):
        return self.fc(x)


# tensorviz: input=2,16,64
class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 64)

    def forward(self, x):
        return self.fc(x)
'''
    assert declared_input_shape(source, "TinyEncoder") == [2, 16, 64]
    assert declared_input_shape(source, "FeedForward") is None
    from services.ingest import service

    own = service._scoped_input_match(source, "TinyEncoder")
    assert own is not None and own.group("value").strip() == "2,16,64"
    assert service._scoped_input_match(source, "FeedForward") is None
    parsed = CodeParserService().parse_pytorch_code(CodeParseRequest(code=source))
    shapes = {}
    for entry in parsed["graphs"]:
        node = next(node for node in entry["graph"].nodes if node.id == "input")
        shapes[entry["name"]] = node.data.parameters["shape"]
    assert shapes["TinyEncoder"] == "2,16,64"
    assert shapes["FeedForward"] != "2,16,64"
    # An explicit caller shape still outranks the comment on every class.
    assert _input_shape(source, input_shape=[4, 8]) == "4,8"


def test_both_tiers_spell_the_declaration_the_same_way():
    """The two patterns are duplicated on purpose, and may not drift.

    The worker cannot import the parser's copy. The worker runs with the user's
    own repository first on ``sys.path``, and importing ``utils.code_parser``
    pulls in ``models.graph_models`` — which any workspace with a ``models/``
    package of its own, meaning most of them, would shadow. That is not
    hypothetical: it was tried, and every worker test failed with the fixture's
    own ``models`` package shadowing the backend's.

    So the duplication stays and this assertion guards it. If the two ever
    disagree, the graph that is drawn and the graph that is run disagree about
    what counts as a declaration, which is the bug this whole file exists for.
    """
    from services.ingest import service

    from utils.code_parser import INPUT_COMMENT

    assert service._INPUT_COMMENT.pattern == INPUT_COMMENT.pattern
