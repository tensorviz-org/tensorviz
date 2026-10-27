"""Source review must preserve the tensor operations in DINOv2-style chains."""
import pytest

from models.graph_models import CodeParseRequest
from utils.code_parser import CodeParserService
from core.graph_input_safety import UnsafeGraphInput
from services.code_generation import CodeGenerationService


def parse(body, imports='import torch\nfrom torch import nn\n'):
    result = CodeParserService().parse_pytorch_code(CodeParseRequest(
        code=imports + 'class Net(nn.Module):\n    def forward(self, x):\n'
        + '\n'.join('        ' + line for line in body.splitlines()), input_shape=[1, 16, 8]))
    assert result['success'], result
    return result['graphs'][0]['graph']


def operations(graph):
    return [node for node in graph.nodes if node.type not in ('inputNode', 'outputNode')]


def test_dinov2_nested_layout_chain_keeps_each_operation_and_only_tensor_edges():
    graph = parse('''M = x.shape[1]
dim = x.shape[-1]
x = nn.functional.interpolate(x.reshape(1, M, M, dim).permute(0, 3, 1, 2), mode="bicubic", antialias=self.antialias, **kwargs)
return x.permute(0, 2, 3, 1).view(1, -1, dim)''')
    nodes = operations(graph)
    assert [node.data.layerType.name for node in nodes] == [
        'torch.reshape', 'torch.permute', 'interpolate', 'torch.permute', 'torch.reshape']
    assert nodes[0].data.parameters['_source_arguments'] == {'shape': '(1, M, M, dim)'}
    assert 'shape' not in nodes[0].data.parameters  # Symbolic expressions are not executable dimensions.
    assert nodes[1].data.parameters['dims'] == [0, 3, 1, 2]
    assert nodes[2].data.parameters['mode'] == 'bicubic'
    assert nodes[2].data.parameters['_source_arguments'] == {'antialias': 'self.antialias', '**kwargs': 'kwargs'}
    assert nodes[-1].data.parameters['_source_arguments'] == {'shape': '(1, -1, dim)'}
    ids = [graph.nodes[0].id, *(node.id for node in nodes), graph.nodes[-1].id]
    assert {(edge.source, edge.target) for edge in graph.edges} == set(zip(ids, ids[1:]))


@pytest.mark.parametrize('imports, expression', [
    ('import torch\nfrom torch import nn\n', 'torch.reshape(x, (1, 4, 4, 8))'),
    ('import torch as th\nfrom torch import nn\n', 'th.reshape(input=x, shape=(1, 4, 4, 8))'),
    ('from torch import nn, reshape as layout\n', 'layout(x, (1, 4, 4, 8))'),
    ('from torch import nn\n', 'x.view(1, 4, 4, 8)'),
    ('from torch import nn\n', 'x.reshape(shape=(1, 4, 4, 8))'),
])
def test_function_method_keyword_and_import_alias_forms(imports, expression):
    graph = parse('return ' + expression, imports)
    node, = operations(graph)
    assert node.data.layerType.name == 'torch.reshape'
    assert node.data.parameters['shape'] == [1, 4, 4, 8]


@pytest.mark.parametrize('imports, call', [
    ('import torch.nn.functional as functional\n', 'functional.interpolate'),
    ('from torch.nn.functional import interpolate as resize\n', 'resize'),
])
def test_interpolate_aliases_preserve_literal_options(imports, call):
    node, = operations(parse(f'return {call}(input=x, size=(32, 32), mode="bilinear", align_corners=False)',
                              'from torch import nn\n' + imports))
    assert node.data.layerType.name == 'interpolate'
    assert {key: value for key, value in node.data.parameters.items() if not key.startswith('_')} == {
        'size': [32, 32], 'mode': 'bilinear', 'align_corners': False}


def test_scalar_shape_expressions_do_not_become_tensor_inputs_or_nodes():
    graph = parse('return x.reshape(x.shape[0], x.shape[1] // 2, -1).permute(0, 2, 1)')
    assert len(operations(graph)) == 2
    assert len(graph.edges) == 3


def test_named_metadata_calculations_do_not_draw_tensor_arithmetic():
    graph = parse('''n = x.shape[1] - 1
dim = x.size(-1)
return x.reshape(1, n, dim)''')
    node, = operations(graph)
    assert node.data.layerType.name == 'torch.reshape'
    assert node.data.parameters['_source_arguments'] == {'shape': '(1, n, dim)'}
    assert len(graph.edges) == 2


def test_opaque_parent_does_not_claim_an_operation_already_drawn_as_a_card():
    nodes = operations(parse('return x.reshape(1, 4, 32).transpose(1, 2)'))
    assert [node.data.layerType.name for node in nodes] == ['torch.reshape', 'Python']
    assert nodes[-1].data.parameters['_source_calls'] == ['transpose']


def test_non_torch_interpolate_stays_opaque():
    node, = operations(parse('return other.functional.interpolate(x, size=(32, 32))'))
    assert node.data.layerType.id == 'python'


def test_shadowed_import_does_not_prove_interpolate():
    node, = operations(parse('return F.interpolate(x, size=(32, 32))',
                              'from torch import nn\nimport torch.nn.functional as F\nF = other\n'))
    assert node.data.layerType.id == 'python'


def test_export_does_not_silently_drop_a_symbolic_optional_argument():
    graph = parse('return nn.functional.interpolate(x, size=(32, 32), mode="bilinear", antialias=self.antialias)')
    with pytest.raises(UnsafeGraphInput, match='symbolic source arguments'):
        CodeGenerationService().generate_flat_code(graph, [1, 16, 8], 'Net')
