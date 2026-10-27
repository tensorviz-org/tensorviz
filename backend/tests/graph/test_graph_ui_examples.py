"""Shipped UI examples must be executable and static edges must remain honest."""
import importlib.util
from pathlib import Path

import pytest
import torch

from models.graph_models import CodeParseRequest
from utils.code_parser import CodeParserService

EXAMPLES = Path(__file__).parent.resolve().parents[2] / 'examples' / 'graph_ui'


@pytest.mark.parametrize('file,cls,shape,result_shape', [
    ('residual', 'ResidualClassifier', (1, 3, 32, 32), (1, 10)),
    ('transformer', 'TinyEncoder', (2, 16, 64), (2, 16, 64)),
    ('branches', 'ParallelBranches', (2, 32), (2, 16)),
])
def test_examples_run_and_have_a_source_graph(file, cls, shape, result_shape):
    path = EXAMPLES / f'{file}.py'
    spec = importlib.util.spec_from_file_location(f'graph_ui_{file}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with torch.inference_mode():
        assert tuple(getattr(module, cls)().eval()(torch.zeros(shape)).shape) == result_shape
    parsed = CodeParserService().parse_pytorch_code(CodeParseRequest(code=path.read_text(), input_shape=list(shape)))
    assert parsed['graphs'], parsed
    assert all(item.get('graph') is not None for item in parsed['graphs']), parsed
    if file == 'branches':
        assert not any((edge.data or {}).get('role') == 'residual' for item in parsed['graphs'] for edge in item['graph'].edges)
