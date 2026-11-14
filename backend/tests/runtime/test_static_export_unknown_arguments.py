"""Unknown static constructor names must never become bogus export keywords."""
import pytest
from core.graph_input_safety import UnsafeGraphInput
from models import CodeParseRequest
from services.code_generation import CodeGenerationService
from utils.code_parser import CodeParserService


def project(arguments):
    source = f'''from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear({arguments})
    def forward(self, x):
        return self.fc(x)
'''
    parser = CodeParserService()
    # Explicitly model unavailable trusted metadata, even on Torch-enabled CI.
    parser._sig_cache['Linear'] = []
    return parser.parse_pytorch_code(CodeParseRequest(code=source))['graphs'][0]['graph']


def test_unknown_positional_names_refuse_export():
    graph = project('4, 3')
    with pytest.raises(UnsafeGraphInput, match='unresolved positional arguments'):
        CodeGenerationService().generate_flat_code(graph, [1, 4])


def test_named_arguments_roundtrip_without_signature():
    graph = project('in_features=4, out_features=3, bias=False')
    code = CodeGenerationService().generate_flat_code(graph, [1, 4])
    parsed = CodeParserService().parse_pytorch_code(CodeParseRequest(code=code))
    layer = next(n for n in parsed['graphs'][0]['graph'].nodes if n.data.layerType.name == 'Linear')
    assert layer.data.parameters['in_features'] == 4
    assert layer.data.parameters['out_features'] == 3
    assert layer.data.parameters['bias'] is False
    assert 'arg_0=' not in code
