"""Configuration arguments and stored attributes are distinct source facts."""
import pytest

from tests.discovery.test_graph_setup_resolution import resolved


CONFIG = '''class Settings:
    def __init__(self, hidden_size=12, heads=3):
        width = hidden_size * 2
        self.width = width
        self.heads = heads
'''
MODEL = '''from torch import nn
from settings import Settings
class Net(nn.Module):
    def __init__(self, settings: Settings):
        super().__init__()
        self.attention = nn.MultiheadAttention(settings.width, settings.heads, batch_first=True)
    def forward(self, x):
        return self.attention(x, x, x)[0]
'''


def test_plain_configuration_constructor_defaults_produce_a_complete_example_without_execution(tmp_path):
    marker = tmp_path / 'must-not-import'
    config = f'from pathlib import Path\nPath({str(marker)!r}).touch()\n' + CONFIG
    result = resolved(MODEL, extra={'settings.py': config})
    assert not marker.exists()
    assert result['status'] == 'complete', result
    assert result['draft']['configObject'] == dict(index=0, parameter='settings', values={})
    assert result['draft']['profile']['inputs']['kwargs']['x']['shape'] == [1, 8, 24]


def test_explicit_configuration_values_keep_precedence_and_use_stored_attribute_arithmetic():
    result = resolved(MODEL + '\nmodel = Net(Settings(hidden_size=18))\n', extra={'settings.py': CONFIG})
    assert result['status'] == 'complete', result
    assert result['draft']['configObject']['values'] == {'hidden_size': 18}
    assert result['draft']['profile']['inputs']['kwargs']['x']['shape'] == [1, 8, 36]


def test_unused_parameter_does_not_masquerade_as_a_stored_attribute():
    result = resolved(MODEL, extra={'settings.py': CONFIG.replace('hidden_size=12', 'hidden_size=12, width=24').replace('self.width = width', 'self.channels = width')})
    assert result['status'] == 'partial'
    assert not result['draft']['profile']['inputs']['kwargs']


def test_class_default_remains_distinct_from_a_same_named_constructor_parameter():
    config = CONFIG.replace('class Settings:', 'class Settings:\n    width = 18').replace('self.width = width', 'self.channels = width')
    result = resolved(MODEL, extra={'settings.py': config})
    assert result['status'] == 'complete', result
    assert result['draft']['profile']['inputs']['kwargs']['x']['shape'] == [1, 8, 18]


@pytest.mark.parametrize('config', [
    CONFIG.replace('self.width = width', 'self.width = load_width(width)'),
    CONFIG.replace('self.width = width', 'if hidden_size > 8: self.width = width'),
    CONFIG.replace('class Settings:', 'class Settings(Base):'),
    CONFIG + '\n    @property\n    def width(self): return 7\n',
    CONFIG.replace('self.width = width', 'self = load_settings()'),
])
def test_unsupported_configuration_behavior_remains_unresolved(config):
    result = resolved(MODEL, extra={'settings.py': config})
    assert result['status'] == 'partial'
    assert 'configObject' not in result['draft']
    assert not result['draft']['profile']['inputs']['kwargs']
