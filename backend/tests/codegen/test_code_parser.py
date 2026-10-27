"""Parser tests: functional/method forward-op capture (code->canvas fidelity)."""

from __future__ import annotations

import pytest


from models.graph_models import CodeParseRequest  # noqa: E402
from utils.code_parser import CodeParserService  # noqa: E402


def _parse(code: str, input_shape):
    res = CodeParserService().parse_pytorch_code(CodeParseRequest(code=code, input_shape=input_shape))
    assert res.get("graphs"), res
    return res["graphs"][0]["graph"]  # a pydantic Graph (nodes are Node objects)


def _ops(graph):
    return [n.data.layerType.name for n in graph.nodes]


CNN = """import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(4096, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = x.flatten(1)
        return self.fc(x)
"""


def test_functional_ops_in_forward_are_captured():
    """F.relu and x.flatten(1) written in forward must become nodes, wired in
    order between the conv/pool stack and the Linear — else the round-tripped CNN
    loses its flatten and the Linear mis-sizes."""
    ops = _ops(_parse(CNN, [1, 3, 32, 32]))
    # The dropped-before ops are now present.
    assert "Flatten" in ops, ops
    assert "ReLU" in ops, ops
    # Flatten sits after the pool and before the Linear.
    assert ops.index("MaxPool2d") < ops.index("Flatten") < ops.index("Linear"), ops


def test_plain_module_chain_unchanged():
    """A model with no functional ops must parse exactly as before (no phantom nodes)."""
    mlp = (
        "import torch.nn as nn\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc1 = nn.Linear(8, 16); self.relu = nn.ReLU(); self.fc2 = nn.Linear(16, 4)\n"
        "    def forward(self, x): return self.fc2(self.relu(self.fc1(x)))\n"
    )
    assert _ops(_parse(mlp, [1, 8])) == ["Input", "Linear", "ReLU", "Linear", "Output"]


@pytest.mark.parametrize('base', ['Module', 'nn.Module', 'torch.nn.Module'])
def test_direct_module_spellings_have_the_same_static_graph(base):
    code = CNN.replace('Net(nn.Module)', f'Net({base})')
    assert _ops(_parse(code, [1, 3, 32, 32])) == _ops(_parse(CNN, [1, 3, 32, 32]))


@pytest.mark.parametrize('base', ['other.Module', 'other.nn.Module', 'torch.Module', 'factory().Module'])
def test_unknown_module_looking_bases_are_not_assumed_to_be_torch(base):
    response = CodeParserService().parse_pytorch_code(CodeParseRequest(
        code=f'class Net({base}):\n    def forward(self, x): return x\n'))
    assert not response['success'] and not response['graphs']
