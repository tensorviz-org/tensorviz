"""Parser tests for imperfect single-file input: several classes, no forward, no shape.

Each case here was a reported failure of the "Import a single file of PyTorch code"
flow, where the importer reported success and produced a wrong model rather than
saying anything:

  * a file with two nn.Module classes kept whichever came FIRST in the source, so
    moving a helper class above the model changed what you got;
  * a class with no forward method left every layer after the first unwired, so the
    input tensor fed the last layer and validation failed with a mat1/mat2 mismatch;
  * a blank Input Shape always seeded (1, 3, 224, 224) even for an MLP whose first
    Linear states the width, though the field offers "best-effort detection".
"""

from __future__ import annotations



from models.graph_models import CodeParseRequest  # noqa: E402
from utils.code_parser import CodeParserService  # noqa: E402


def _parse(code: str, input_shape=None):
    return CodeParserService().parse_pytorch_code(
        CodeParseRequest(code=code, input_shape=input_shape)
    )


def _edges(graph):
    return [(e.source, e.target) for e in graph.edges]


def _input_shape(graph):
    # The Input node carries the canvas string form ("1,784") — the only form
    # core/shape_inference.py accepts. Read it back as the dims it names.
    node = next(n for n in graph.nodes if n.data.layerType.id == "input")
    return [int(part) for part in node.data.parameters["shape"].split(",")]


TWO_CLASSES = """import torch.nn as nn

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(8, 16, 3)

class ClassifierNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = Block()
        self.fc = nn.Linear(784, 10)
"""


def test_marks_the_class_no_other_class_instantiates_as_top_level():
    graphs = {g["name"]: g for g in _parse(TWO_CLASSES)["graphs"]}
    assert graphs["ClassifierNet"]["is_top_level"] is True
    # Block is a submodule of ClassifierNet, so it is not the model of the file.
    assert graphs["Block"]["is_top_level"] is False


def test_source_order_does_not_decide_which_class_is_top_level():
    # The whole point: the answer must not change when the classes are swapped.
    reordered = TWO_CLASSES.replace("class Block", "class ZBlock").replace(
        "self.b = Block()", "self.b = ZBlock()"
    )
    graphs = {g["name"]: g for g in _parse(reordered)["graphs"]}
    assert graphs["ClassifierNet"]["is_top_level"] is True
    assert graphs["ZBlock"]["is_top_level"] is False


def test_every_class_is_still_returned_so_callers_can_report_what_was_skipped():
    names = [g["name"] for g in _parse(TWO_CLASSES)["graphs"]]
    assert sorted(names) == ["Block", "ClassifierNet"]


NO_FORWARD = """import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
"""


def test_layers_are_chained_in_init_order_when_there_is_no_forward():
    graph = _parse(NO_FORWARD)["graphs"][0]["graph"]
    # fc1 -> fc2 is the edge that was missing: without it fc2 dangled and the
    # 784-wide input was multiplied against fc2's 128x10 weight.
    assert _edges(graph) == [("input", "fc1"), ("fc1", "fc2"), ("fc2", "output")]


def test_an_explicit_forward_still_wins_over_declaration_order():
    # Declaration order is a fallback, never an override: this forward runs the
    # layers in the reverse of the order they are declared in.
    code = """import torch.nn as nn

class Rev(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(10, 10)
        self.b = nn.Linear(10, 10)

    def forward(self, x):
        return self.a(self.b(x))
"""
    graph = _parse(code)["graphs"][0]["graph"]
    assert ("b", "a") in _edges(graph)
    assert ("a", "b") not in _edges(graph)


def test_infers_the_input_width_from_the_first_linear():
    graph = _parse(NO_FORWARD)["graphs"][0]["graph"]
    assert _input_shape(graph) == [1, 784]


def test_infers_channels_from_the_first_conv():
    code = """import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(3, 16, 3)
"""
    assert _input_shape(_parse(code)["graphs"][0]["graph"]) == [1, 3, 224, 224]


def test_an_explicit_input_shape_still_wins_over_inference():
    graph = _parse(NO_FORWARD, input_shape=[8, 784])["graphs"][0]["graph"]
    assert _input_shape(graph) == [8, 784]


def test_a_module_level_example_tensor_still_wins_over_inference():
    # torch.randn states the batch size too, which the layer signature cannot.
    code = NO_FORWARD + "\nx = torch.randn(8, 784)\n"
    assert _input_shape(_parse(code)["graphs"][0]["graph"]) == [8, 784]


def test_unknown_input_shape_stays_empty_instead_of_claiming_image_dimensions():
    code = """import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.ReLU()
"""
    graph = _parse(code)["graphs"][0]["graph"]
    assert next(n for n in graph.nodes if n.type == "inputNode").data.parameters["shape"] == ""


def test_feature_vector_loss_does_not_acquire_an_image_input_shape():
    code = """import torch.nn as nn
import torch.nn.functional as F
class KoLeoLoss(nn.Module):
    def forward(self, student_output, eps=1e-8):
        student_output = F.normalize(student_output.float(), eps=eps, p=2, dim=-1)
        return student_output
"""
    graph = _parse(code)["graphs"][0]["graph"]
    assert len(graph.nodes) > 2
    assert next(n for n in graph.nodes if n.type == "inputNode").data.parameters["shape"] == ""
    explicit = _parse(code, input_shape=[8, 384])["graphs"][0]["graph"]
    assert _input_shape(explicit) == [8, 384]


WITH_DEFAULTS = """import torch.nn as nn

class Net(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
"""


def _params(graph, node_id):
    # `_`-prefixed keys are provenance (file, span, raw call), not constructor
    # arguments; every plane strips them before they reach torch or codegen.
    node = next(n for n in graph.nodes if n.id == node_id)
    return {k: v for k, v in node.data.parameters.items() if not k.startswith("_")}


def test_resolves_layer_args_from_the_init_signature_defaults():
    # nn.Linear(784, hidden_dim) is only unreadable if nothing says what hidden_dim
    # is. A literal default in __init__ says exactly that, and dropping it imported
    # a Linear with an in_features and no out_features.
    graph = _parse(WITH_DEFAULTS)["graphs"][0]["graph"]
    assert _params(graph, "fc1") == {"in_features": 784, "out_features": 128}
    assert _params(graph, "fc2") == {"in_features": 128, "out_features": 10}


def test_resolves_keyword_only_defaults_too():
    code = """import torch.nn as nn

class Net(nn.Module):
    def __init__(self, *, width=64):
        super().__init__()
        self.fc = nn.Linear(width, 10)
"""
    assert _params(_parse(code)["graphs"][0]["graph"], "fc")["in_features"] == 64


def test_leaves_a_name_with_no_default_unset_rather_than_guessing():
    # Nothing in the file says what hidden_dim is. An empty field the user can see
    # and fill beats a plausible invention.
    code = WITH_DEFAULTS.replace("hidden_dim=128, num_classes=10", "hidden_dim, num_classes")
    graph = _parse(code)["graphs"][0]["graph"]
    assert _params(graph, "fc1") == {"in_features": 784}
    assert _params(graph, "fc2") == {}


def test_an_explicit_literal_still_wins_over_a_same_named_default():
    code = """import torch.nn as nn

class Net(nn.Module):
    def __init__(self, out=99):
        super().__init__()
        self.fc = nn.Linear(8, 4)
"""
    assert _params(_parse(code)["graphs"][0]["graph"], "fc")["out_features"] == 4
