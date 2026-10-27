from types import SimpleNamespace

from services.ide.code_focus import enrich_focus_spans


SOURCE = '''import torch
from torch import nn

class Block(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        x = x * self.scale
        return self.proj(x)

    def reset_scale(self):
        self.scale.fill_(1)

    def unrelated(self):
        return 2

def make():
    return Block()
'''


def focus(source=SOURCE, start=7, end=7, parameter=True):
    document = SimpleNamespace(semantic=SimpleNamespace(nodes=[SimpleNamespace(
        node_id="selected", op="State" if parameter else "Linear",
        metadata={"sourceKind": "parameter"} if parameter else {})]))
    result = enrich_focus_spans({"selected": dict(file="model.py", start=[start, 0], end=[end, 1])},
                               document, {"model.py": source})
    return result["selected"]["focus"]


def test_parameter_reveals_declaration_and_direct_uses_in_uncalled_methods():
    result = focus()
    assert result["class"] == [[4, 18]]
    assert result["function"] == [[5, 8]]
    assert result["parameter"]["declaration"] == [[7, 7]]
    assert result["parameter"]["uses"] == [[11, 11], [15, 15]]
    assert result["parameter"]["useFunctions"] == [[10, 12], [14, 15]]
    assert result["parameter"]["declarationFunctions"] == [[5, 8]]


def test_operation_context_is_its_calling_function_not_constructor():
    result = focus(start=12, end=12, parameter=False)
    assert result == {"class": [[4, 18]], "function": [[10, 12]], "node": [[12, 12]]}


def test_global_helper_does_not_borrow_a_model_class():
    result = focus(start=21, end=21, parameter=False)
    assert result["class"] == []
    assert result["function"] == [[20, 21]]


def test_receiver_name_is_not_assumed_to_be_self():
    result = focus(source=SOURCE.replace("self", "module"))
    assert result["parameter"]["uses"] == [[11, 11], [15, 15]]


def test_rebinding_receiver_never_highlights_another_objects_parameter():
    result = focus(source=SOURCE.replace("        x = x * self.scale", "        self = x; x = self.scale"))
    assert result["parameter"]["uses"] == [[15, 15]]


def test_parameter_reassignment_refuses_uses_instead_of_following_a_different_tensor():
    result = focus(source=SOURCE.replace("        x = x * self.scale", "        self.scale = x"))
    assert result["parameter"]["uses"] == []
    assert "ambiguous" in result["parameter"]["limitation"]


def test_ambiguous_declarations_do_not_pick_one_branch():
    result = focus(source=SOURCE.replace("        self.proj = nn.Linear(dim, dim)", "        self.scale = nn.Parameter(torch.zeros(dim))"))
    assert result["parameter"] is None


def test_nested_function_receiver_does_not_belong_to_outer_method():
    result = focus(source=SOURCE.replace("        self.scale.fill_(1)", "        def other(self): return self.scale"))
    assert result["parameter"]["uses"] == [[11, 11]]


def test_no_source_means_no_invented_focus_ranges():
    document = SimpleNamespace(semantic=SimpleNamespace(nodes=[]))
    spans = {"x": dict(file="library.py", start=[2, 0], end=[2, 9])}
    assert enrich_focus_spans(spans, document, {}) == spans
