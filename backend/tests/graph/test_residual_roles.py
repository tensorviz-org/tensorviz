"""A dotted path must have a supported source relationship, not just an add."""
import pytest

from models.graph_models import CodeParseRequest
from services.ide.projection import project
from utils.code_parser import CodeParserService


def parse(body, args='x'):
    source = f'''import torch
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(8, 16)
        self.down = nn.Linear(16, 8)
        self.shortcut = nn.Linear(8, 8)
        self.other = nn.Linear(8, 8)
    def forward(self, {args}):
        {body}
'''
    result = CodeParserService().parse_pytorch_code(CodeParseRequest(code=source, input_shape=[1, 8]))
    graph = result['graphs'][0]['graph']
    assert graph is not None, result
    return source, graph


def residuals(graph):
    return [e for e in graph.edges if (e.data or {}).get('role') == 'residual']


@pytest.mark.parametrize('body', [
    'return self.down(self.up(x)) + x',
    'return torch.add(x, self.down(self.up(x)))',
    'identity = x\n        x = self.down(self.up(x))\n        x += identity\n        return x',
])
def test_identity_bypass_is_distinct_from_the_transformed_branch(body):
    source, graph = parse(body)
    marked = residuals(graph)
    assert len(marked) == 1
    assert marked[0].source == 'input'
    assert marked[0].data['roleEvidence'] == 'identity-bypass'
    document = project(graph.model_dump(), cls='Net', rel_file='net.py', source=source).document
    exported = [e for e in document.semantic.edges if e.metadata.get('role') == 'residual']
    assert len(exported) == 1
    assert exported[0].source_node_id == 'input'
    assert exported[0].target_node_id.startswith('add@Net#')


def test_explicit_projection_shortcut_shares_the_branch_origin():
    _, graph = parse('return self.down(self.up(x)) + self.shortcut(x)')
    assert [(e.source, e.data['roleEvidence']) for e in residuals(graph)] == [('shortcut', 'projection-shortcut')]


@pytest.mark.parametrize('body,args', [
    ('return self.shortcut(x) + self.other(x)', 'x'),  # both are single parallel projections
    ('return self.down(self.up(x)) + self.other(x)', 'x'),
    ('return torch.cat([x, self.down(self.up(x))], dim=1)', 'x'),
    ('return self.down(self.up(y)) + x', 'x, y'),
    ('return self.down(self.up(x)) + x * 2', 'x'),
    ('identity = x[:, :4]\n        return self.down(self.up(x)) + identity', 'x'),
    ('return self.down(self.up(x)) + mystery(x)', 'x'),
    ('if x is not None:\n            x = self.down(self.up(x)) + x\n        return x', 'x'),
])
def test_ambiguous_or_unsupported_relations_remain_ordinary(body, args):
    _, graph = parse(body, args)
    assert residuals(graph) == []
