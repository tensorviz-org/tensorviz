"""A block refactor must preserve execution and refuse ambiguous data flow."""
import ast
import hashlib

import pytest

from tests.source_review.test_code_apply import MLP, _apply
from tests.graph.test_ide_multifile_projection import client


def group(source=MLP, names=None, name="features"):
    return _apply(source, [{"op": "group_layers", "op_id": "group", "node_id": name,
                            "node_ids": names if names is not None else ["act", "fc1"]}])


def test_group_uses_source_order_and_preserves_unrelated_source_and_comments():
    source = MLP.replace("self.act = nn.ReLU()", "# activation\n        self.act = nn.ReLU()  # kept")
    result = group(source)
    assert not result.refused
    assert "self.features = FeaturesBlock()" in result.source
    assert "class FeaturesBlock(nn.Module):" in result.source
    assert "x = self.features(x)" in result.source
    assert "# activation" in result.source and "# kept" in result.source
    assert "self.fc2 = nn.Linear(128, 10)" in result.source
    assert '"""Three layers and a comment, so a rewrite of the file is visible."""' in result.source
    root = result.source.split("class FeaturesBlock")[0]
    assert "self.fc1" not in root and "self.act" not in root
    ast.parse(result.source)


def test_nested_chain_becomes_one_call():
    source = MLP.replace("x = self.fc1(x)\n        x = self.act(x)\n        x = self.fc2(x)\n        return x",
                         "return self.fc2(self.act(self.fc1(x)))")
    result = group(source)
    assert not result.refused
    assert "return self.fc2(self.features(x))" in result.source


def test_group_reprojects_as_a_block_with_children_and_valid_source_spans(client):
    source = group().source
    digest = hashlib.sha256(source.encode()).hexdigest()
    response = client.post("/api/ide/multifile/project", json={
        "protocol_version": 1,
        "sources": [{"file": "net.py", "source": source, "digest": digest}],
        "root": {"file": "net.py", "cls": "Net", "document_version": 2, "source_digest": digest},
    })
    assert response.status_code == 200, response.text
    result = response.json()
    nodes = result["document"]["view"]["layout"]["node_data"]
    assert nodes["features"]["moduleMembers"] == ["features.fc1", "features.act"]
    assert nodes["features.fc1"]["parentId"] == "features"
    assert result["spanmap"]["features.act"]["file_digest"] == digest
    assert result["diagnostics"] == []


@pytest.mark.parametrize("source,names,name", [
    (MLP, ["fc1", "fc2"], "features"),  # nonconsecutive declarations
    (MLP, ["fc1", "act"], "fc2"),  # collision
    (MLP, ["fc1", "fc1"], "features"),
    (MLP, [], "features"),
    (MLP, ["fc1", "act"], "class"),
    (MLP.replace("return x", "return self.fc1(x)"), ["fc1", "act"], "features"),
    (MLP.replace("x = self.act(x)", "x = self.act(x) + x"), ["fc1", "act"], "features"),
    (MLP.replace("x = self.act(x)", "if x is not None:\n            x = self.act(x)"), ["fc1", "act"], "features"),
    (MLP.replace("x = self.act(x)", "x = self.act(x, extra=True)"), ["fc1", "act"], "features"),
    (MLP + "\n    def helper(self):\n        return self.fc1\n", ["fc1", "act"], "features"),
    (MLP.replace("self.act = nn.ReLU()", "self.act = nn.ReLU(); print('keep')"), ["fc1", "act"], "features"),
    (MLP.replace("return x", "return getattr(self, 'fc1')(x)"), ["fc1", "act"], "features"),
    (MLP.replace("super().__init__()", "super().__init__()\n        nn = object()"), ["fc1", "act"], "features"),
    (MLP.replace("nn.Linear(784, 128)", "nn.Linear(784, width)"), ["fc1", "act"], "features"),
])
def test_ambiguous_group_is_refused_without_any_patch(source, names, name):
    result = group(source, names, name)
    assert result.refused
    assert result.source == source
    assert result.edits == []


def test_group_preserves_values_and_gradients_in_explicit_cpu_execution():
    # Execution here is deliberately test-only; grouping itself is entirely static.
    import torch
    source = MLP.replace("784, 128", "8, 4").replace("128, 10", "4, 2")
    result = group(source)
    assert not result.refused
    before, after = {}, {}
    exec(source, before)
    exec(result.source, after)
    torch.manual_seed(91)
    original = before["Net"]()
    torch.manual_seed(91)
    grouped = after["Net"]()
    x1 = torch.randn(3, 8, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    y1, y2 = original(x1), grouped(x2)
    torch.testing.assert_close(y1, y2, rtol=0, atol=0)
    y1.sum().backward()
    y2.sum().backward()
    torch.testing.assert_close(x1.grad, x2.grad, rtol=0, atol=0)
    for a, b in zip(original.parameters(), grouped.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


def test_add_layer_cannot_overwrite_an_existing_attribute():
    result = _apply(MLP, [
        {"op": "add_layer", "node_id": "fc2", "layer_op": "ReLU", "params": {}},
        {"op": "connect", "source_node_id": "fc1", "target_node_id": "fc2"},
    ])
    assert result.refused
    assert result.source == MLP
    assert not result.edits
