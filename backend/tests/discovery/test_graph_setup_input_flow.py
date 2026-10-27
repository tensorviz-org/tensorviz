"""Input proposals follow known shape contracts, not names or arbitrary transforms."""
import pytest

from tests.discovery.test_graph_setup_resolution import LINEAR, resolved


NORM_ATTENTION = '''from torch import nn
class Net(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, 2, batch_first=True)
    def forward(self, x):
        h = self.norm(x)
        return self.attn(h, h, h)[0]
'''


@pytest.mark.parametrize('body', [
    'h = self.norm(x)\n        return self.attn(h, h, h)[0]',
    'return self.attn(self.norm(x), self.norm(x), self.norm(x))[0]',
    'return self.attn(query=self.norm(x), key=self.norm(x), value=self.norm(x))[0]',
])
def test_normalization_preserves_the_attention_input_shape(body):
    source = NORM_ATTENTION.replace('h = self.norm(x)\n        return self.attn(h, h, h)[0]', body)
    result = resolved(source)
    assert result['status'] == 'complete', result
    assert result['draft']['profile']['inputs']['kwargs']['x']['shape'] == [1, 8, 8]


@pytest.mark.parametrize('layer', ['nn.ReLU()', 'nn.GELU()', 'nn.SiLU()', 'nn.Dropout()', 'nn.Identity()', 'nn.LayerNorm(width)'])
def test_sequential_shape_preserving_prefix_uses_the_later_input_contract(layer):
    source = LINEAR.replace('nn.Linear(width, 4)', f'nn.Sequential({layer}, nn.Linear(width, 4))')
    result = resolved(source)
    assert result['status'] == 'complete', result
    assert result['draft']['profile']['inputs']['kwargs']['x']['shape'] == [1, 8]


@pytest.mark.parametrize('layer', ['nn.Flatten()', 'custom()', 'LayerNorm(width)'])
def test_unknown_or_shape_changing_prefix_does_not_inherit_later_dimensions(layer):
    result = resolved(NORM_ATTENTION.replace('nn.LayerNorm(width)', layer))
    assert result['status'] == 'partial'
    assert not result['draft']['profile']['inputs']['kwargs']


def test_known_shape_change_inside_a_known_preserving_layer_stays_unresolved():
    source = NORM_ATTENTION.replace('self.norm(x)', 'self.norm(x.flatten(1))')
    assert resolved(source)['status'] == 'partial'


def test_later_linear_boundary_does_not_conflict_with_attention_rank():
    source = NORM_ATTENTION.replace('self.norm =', 'self.fc = nn.Linear(width, width)\n        self.norm =').replace(
        'return self.attn(h, h, h)[0]', 'return self.attn(h, h, h)[0] + self.fc(h)')
    result = resolved(source)
    assert result['status'] == 'complete', result
    assert result['draft']['profile']['inputs']['kwargs']['x']['shape'] == [1, 8, 8]


def test_different_feature_widths_do_not_merge_just_because_linear_accepts_any_rank():
    source = NORM_ATTENTION.replace('self.norm =', 'self.fc = nn.Linear(7, 8)\n        self.norm =').replace(
        'return self.attn(h, h, h)[0]', 'return self.attn(h, h, h)[0] + self.fc(h)')
    assert resolved(source)['status'] == 'partial'
