"""No-configuration examples respect source relationships rather than names."""
import pytest

from tests.discovery.test_graph_setup_resolution import LINEAR, resolved


ATTENTION = '''from torch import nn
class Net(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
    def forward(self, x): return self.attn(x, x, x)[0]
'''


def test_renaming_does_not_change_inferred_dimensions_and_constructor_asserts_apply():
    source = LINEAR.replace('width=8', 'width').replace('self.fc =', 'assert width == 13\n        self.fc =')
    normal = resolved(source)
    renamed = resolved(source.replace('width', 'arbitrary_parameter'))
    assert normal['status'] == renamed['status'] == 'complete'
    assert normal['draft']['profile']['init']['kwargs'] == {'width': 13}
    assert renamed['draft']['profile']['init']['kwargs'] == {'arbitrary_parameter': 13}
    assert normal['draft']['profile']['inputs'] == renamed['draft']['profile']['inputs']


@pytest.mark.parametrize('source, expected', [
    (ATTENTION, {'width': 8, 'heads': 2}),
    (ATTENTION.replace('width, heads):', 'width, heads=3):'), {'width': 9}),
    (ATTENTION.replace('width, heads):', 'width=30, heads=None):') + '\nmodel = Net(width=30, heads=3)\n', {'width': 30, 'heads': 3}),
])
def test_attention_dimensions_are_jointly_compatible(source, expected):
    result = resolved(source)
    assert result['status'] == 'complete', result
    assert result['draft']['profile']['init']['kwargs'] == expected
    assert result['draft']['profile']['inputs']['kwargs']['x']['shape'] == [1, 8, expected['width']]


def test_grouped_convolution_can_infer_input_and_output_dimensions_without_class_names():
    source = LINEAR.replace('width=8', 'width, result_size').replace('nn.Linear(width, 4)', 'nn.Conv2d(width, result_size, 3, groups=3)')
    result = resolved(source)
    assert result['status'] == 'complete', result
    assert result['draft']['profile']['init']['kwargs'] == {'width': 9, 'result_size': 9}
    assert result['draft']['profile']['inputs']['kwargs']['x']['shape'] == [1, 9, 64, 64]


def test_source_arithmetic_is_solved_without_evaluating_python():
    source = LINEAR.replace('width=8', 'width').replace('nn.Linear(width, 4)', 'nn.Conv1d(width * 2, 12, 3, groups=3)')
    result = resolved(source)
    assert result['status'] == 'complete', result
    width = result['draft']['profile']['init']['kwargs']['width']
    assert width * 2 % 3 == 0
    assert result['draft']['profile']['inputs']['kwargs']['x']['shape'][1] == width * 2


def test_config_dimension_and_ordinary_constructor_dimension_can_both_be_missing():
    source = '''from dataclasses import dataclass
from torch import nn
@dataclass
class Config:
    vocabulary: int
class Net(nn.Module):
    def __init__(self, config: Config, width):
        super().__init__()
        self.emb = nn.Embedding(config.vocabulary, width)
    def forward(self, tokens): return self.emb(tokens)
'''
    result = resolved(source)
    assert result['status'] == 'complete', result
    assert result['draft']['configObject']['values'] == {'vocabulary': 32}
    assert result['draft']['profile']['init']['kwargs'] == {'width': 8}
    assert result['draft']['profile']['inputs']['kwargs']['tokens']['generator']['high'] == 32


def test_nested_modules_and_sequential_pass_dimension_evidence_back_to_the_root():
    source = LINEAR.replace('width=8', 'width, classes').replace('nn.Linear(width, 4)', 'Block(width, classes)')
    source += '''
class Block(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(incoming, outgoing), nn.ReLU())
    def forward(self, x): return self.layers(x)
'''
    result = resolved(source)
    assert result['status'] == 'complete', result
    assert result['draft']['profile']['init']['kwargs'] == {'width': 8, 'classes': 8}


def test_annotation_does_not_replace_a_concrete_source_argument():
    source = '''from dataclasses import dataclass
from torch import nn
@dataclass
class Config:
    count: int = 4
class Net(nn.Module):
    def __init__(self, config: Config, width):
        super().__init__()
        self.fc = nn.Linear(width, 4)
    def forward(self, x): return self.fc(x)
model = Net(config=7, width=args.width)
'''
    result = resolved(source)
    assert result['status'] == 'complete', result
    assert result['draft']['profile']['init']['kwargs'] == {'config': 7, 'width': 8}
    assert 'configObject' not in result['draft']


@pytest.mark.parametrize('source', [
    LINEAR.replace('width=8', 'path').replace('nn.Linear(width, 4)', 'load_layer(path)'),
    LINEAR.replace('width=8', 'width').replace('self.fc =', 'assert width > 300\n        self.fc ='),
    LINEAR.replace('width=8', 'width, nn=external'),
    LINEAR.replace('width=8', 'width').replace('self.fc =', 'width = custom(width)\n        self.fc ='),
])
def test_opaque_values_shadowing_and_out_of_budget_constraints_remain_unresolved(source):
    result = resolved(source)
    assert result['status'] == 'partial'
    assert not result['draft']['profile']['init']['kwargs']


def test_positional_only_dimensions_preserve_known_source_arguments():
    source = LINEAR.replace('width=8', 'width, outputs, /').replace('nn.Linear(width, 4)', 'nn.Linear(width, outputs)')
    result = resolved(source + '\nmodel = Net(args.width, 17)\n')
    assert result['status'] == 'complete', result
    assert result['draft']['profile']['init'] == {'args': [8, 17], 'kwargs': {}}


@pytest.mark.parametrize('kind, arguments, expected', [
    ('GRU', 'width, 8, 1, True, True', [1, 8, 8]),
    ('LSTM', 'width, 8, 1, True, False', [8, 1, 8]),
    ('RNN', 'width, 8, 1, "tanh", True, True', [1, 8, 8]),
    ('TransformerEncoderLayer', 'width, 2, 16, 0., "relu", 1e-5, True', [1, 8, 8]),
    ('MultiheadAttention', 'width, 2, 0., True, False, False, None, None, True', [1, 8, 8]),
])
def test_sequence_input_honors_each_public_batch_first_position(kind, arguments, expected):
    source = LINEAR.replace('width=8', 'width').replace('nn.Linear(width, 4)', f'nn.{kind}({arguments})')
    if kind == 'MultiheadAttention':
        source = source.replace('self.fc(x)', 'self.fc(x, x, x)[0]')
    result = resolved(source)
    assert result['status'] == 'complete', result
    assert result['draft']['profile']['inputs']['kwargs']['x']['shape'] == expected


def test_attention_key_value_dimensions_are_not_assumed_to_equal_the_query():
    source = ATTENTION.replace('width, heads):', 'width=8, heads=2):').replace(
        'batch_first=True', 'kdim=4, vdim=6, batch_first=True').replace('forward(self, x)', 'forward(self, x, key, value)').replace('self.attn(x, x, x)', 'self.attn(x, key, value)')
    result = resolved(source)
    assert result['status'] == 'complete', result
    inputs = result['draft']['profile']['inputs']['kwargs']
    assert [inputs[k]['shape'] for k in ('x', 'key', 'value')] == [[1, 8, 8], [1, 8, 4], [1, 8, 6]]


BLOCKS = '''from dataclasses import dataclass
from torch import nn
@dataclass
class Config:
    count: int
class Net(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.stem = nn.Linear(8, 8)
        self.blocks = nn.ModuleList([nn.Linear(8, 8) for _ in range(cfg.count)])
    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return x
'''


@pytest.mark.parametrize('count_name', ['count', 'arbitrary_parameter'])
def test_missing_block_count_uses_a_small_nonempty_example_independent_of_names(count_name):
    result = resolved(BLOCKS.replace('count', count_name))
    assert result['status'] == 'complete', result
    assert result['draft']['configObject']['values'] == {count_name: 1}
    assert any('range' in e['detail'] and e['kind'] == 'example' for e in result['evidence'])


def test_range_count_satisfies_explicit_minimum_instead_of_ignoring_it():
    source = BLOCKS.replace('self.stem =', 'assert cfg.count >= 3\n        self.stem =')
    result = resolved(source)
    assert result['draft']['configObject']['values'] == {'count': 3}


def test_explicit_builtin_alias_retains_range_semantics():
    result = resolved('from builtins import range as steps\n' + BLOCKS.replace('range(cfg.count)', 'steps(cfg.count)'))
    assert result['status'] == 'complete', result
    assert result['draft']['configObject']['values'] == {'count': 1}


@pytest.mark.parametrize('source', [
    BLOCKS + '\nrange = custom\n',
    BLOCKS + '\ndef range(count): return custom(count)\n',
    BLOCKS.replace('cfg: Config):', 'cfg: Config, range=external):'),
    BLOCKS.replace('self.stem =', 'range = external\n        self.stem ='),
    BLOCKS.replace('self.stem =', 'cfg.count = custom(cfg.count)\n        self.stem ='),
    BLOCKS.replace('range(cfg.count)', 'range(custom(cfg.count))'),
    BLOCKS.replace('range(cfg.count)', 'range(2, cfg.count)'),
])
def test_unknown_or_mutated_range_counts_remain_unresolved(source):
    result = resolved(source)
    assert result['status'] == 'partial', result
    assert result['draft']['configObject']['values'] == {}
