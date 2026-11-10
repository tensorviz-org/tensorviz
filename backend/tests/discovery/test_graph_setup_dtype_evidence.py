"""Source dtype evidence must agree with the actual PyTorch call."""
import pytest
from tests.discovery.test_graph_setup_resolution import LINEAR, resolved


@pytest.mark.parametrize('dtype,expected', [('float', 'float32'), ('double', 'float64'),
                                          ('half', 'float16'), ('bfloat16', 'bfloat16')])
def test_torch_dtype_aliases(dtype, expected):
    source = 'import torch\n' + LINEAR.replace('nn.Linear(width, 4)', f'nn.Linear(width, 4, dtype=torch.{dtype})')
    result = resolved(source)
    assert result['status'] == 'complete', result
    assert result['draft']['profile']['inputs']['kwargs']['x']['dtype'] == expected


@pytest.mark.parametrize('layer,call', [
    ('Linear', '8, 4, True, None, torch.float64'),
    ('Conv1d', "8, 4, 3, 1, 0, 1, 1, True, 'zeros', None, torch.float64"),
    ('Conv2d', "8, 4, 3, 1, 0, 1, 1, True, 'zeros', None, torch.float64"),
    ('Conv3d', "8, 4, 3, 1, 0, 1, 1, True, 'zeros', None, torch.float64"),
    ('ConvTranspose1d', "8, 4, 3, 1, 0, 0, 1, True, 1, 'zeros', None, torch.float64"),
    ('ConvTranspose2d', "8, 4, 3, 1, 0, 0, 1, True, 1, 'zeros', None, torch.float64"),
    ('ConvTranspose3d', "8, 4, 3, 1, 0, 0, 1, True, 1, 'zeros', None, torch.float64"),
    ('MultiheadAttention', '8, 2, 0., True, False, False, None, None, True, None, torch.float64'),
    ('TransformerEncoderLayer', "8, 2, 16, 0., 'relu', 1e-5, True, False, True, None, torch.float64"),
    ('TransformerDecoderLayer', "8, 2, 16, 0., 'relu', 1e-5, True, False, True, None, torch.float64"),
    ('Transformer', "8, 2, 1, 1, 16, 0., 'relu', None, None, 1e-5, True, False, True, None, torch.float64"),
    ('RNN', "8, 4, 1, 'tanh', True, True, 0., False, 0, None, torch.float64"),
    ('GRU', '8, 4, 1, True, True, 0., False, 0, None, torch.float64'),
    ('LSTM', '8, 4, 1, True, True, 0., False, 0, None, torch.float64'),
])
def test_positional_dtype_runs_with_inferred_input(layer, call):
    torch = pytest.importorskip('torch')
    source = 'import torch\n' + LINEAR.replace('nn.Linear(width, 4)', f'nn.{layer}({call})')
    if layer == 'MultiheadAttention':
        source = source.replace('self.fc(x)', 'self.fc(x, x, x)[0]')
    if layer in ('Transformer', 'TransformerDecoderLayer'):
        source = source.replace('self.fc(x)', 'self.fc(x, x)')
    result = resolved(source)
    assert result['status'] == 'complete', result
    spec = result['draft']['profile']['inputs']['kwargs']['x']
    assert spec['dtype'] == 'float64'
    namespace = {}
    exec(source, namespace)  # This test owns the synthetic fixture, not user source.
    with torch.no_grad():
        namespace['Net']().eval()(torch.randn(spec['shape'], dtype=getattr(torch, spec['dtype'])))


def test_unknown_dtype_does_not_authorize_a_float32_example():
    source = LINEAR.replace('nn.Linear(width, 4)', 'nn.Linear(width, 4, dtype=external_dtype)')
    result = resolved(source)
    assert result['status'] == 'partial', result
    assert result['draft']['profile']['inputs']['kwargs'] == {}


def test_embedding_weight_dtype_does_not_change_index_dtype():
    source = 'import torch\n' + LINEAR.replace('nn.Linear(width, 4)', 'nn.Embedding(width, 4, dtype=torch.double)')
    result = resolved(source)
    assert result['status'] == 'complete', result
    assert result['draft']['profile']['inputs']['kwargs']['x']['dtype'] == 'int64'


@pytest.mark.parametrize('dtype', ["'float64'", 'external_dtype', 'select_dtype()',
                                    'torch.complex64', 'torch.int32', 'False'])
def test_unsupported_dtype_evidence_stays_incomplete(dtype):
    source = 'import torch\n' + LINEAR.replace('nn.Linear(width, 4)', f'nn.Linear(width, 4, dtype={dtype})')
    result = resolved(source)
    assert result['status'] == 'partial', result
    assert result['draft']['profile']['inputs']['kwargs'] == {}


@pytest.mark.parametrize('args', ['width, 4, dtype=None', 'width, 4, True, None, None'])
def test_explicit_none_dtype_retains_default(args):
    result = resolved(LINEAR.replace('nn.Linear(width, 4)', f'nn.Linear({args})'))
    assert result['status'] == 'complete', result
    assert result['draft']['profile']['inputs']['kwargs']['x']['dtype'] == 'float32'


def test_dtype_expression_is_never_executed(tmp_path):
    marker = tmp_path / 'dtype-evaluation-must-not-run'
    source = f'from pathlib import Path\ndef select_dtype():\n    Path({str(marker)!r}).touch()\n    return None\n' + LINEAR.replace(
        'nn.Linear(width, 4)', 'nn.Linear(width, 4, True, None, select_dtype())')
    result = resolved(source)
    assert result['status'] == 'partial', result
    assert not marker.exists()


def test_imported_dtype_alias_is_preserved():
    source = 'from torch import double as precision\n' + LINEAR.replace(
        'nn.Linear(width, 4)', 'nn.Linear(width, 4, dtype=precision)')
    result = resolved(source)
    assert result['status'] == 'complete', result
    assert result['draft']['profile']['inputs']['kwargs']['x']['dtype'] == 'float64'
