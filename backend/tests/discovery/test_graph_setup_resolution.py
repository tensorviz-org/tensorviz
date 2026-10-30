import json

import pytest

from tests.discovery.test_graph_setup_discovery import inspect_model


LINEAR = '''from torch import nn
class Net(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.fc = nn.Linear(width, 4)
    def forward(self, x): return self.fc(x)
'''


def resolved(source=LINEAR, **kwargs):
    result = inspect_model(source, **kwargs)['resolution']
    json.dumps(result, allow_nan=False)
    return result


def test_a_class_comment_is_the_example_input_and_does_not_leak():
    source = '''import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Linear(32, 24)
    def forward(self, x):
        return self.left(x)

# tensorviz: input=2,32
class ParallelBranches(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Linear(32, 24)
    def forward(self, x):
        return self.left(x)
'''
    declared = resolved(source, cls='ParallelBranches')
    other = resolved(source, cls='FeedForward')
    assert declared['status'] == 'complete', declared
    assert declared['draft']['profile']['inputs']['args'][0]['shape'] == [2, 32]
    assert not declared.get('runtimeInputs')
    assert any('Explicit input declaration' in entry['detail'] for entry in declared['evidence'])
    assert other['draft']['profile']['inputs']['kwargs']['x']['shape'] == [1, 32]


def test_a_conv_comment_replaces_the_guessed_spatial_size():
    result = resolved('''import torch.nn as nn
# tensorviz: input=1,3,32,32
class ResidualClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 32, 3, padding=1)
    def forward(self, x):
        return self.stem(x)
''', cls='ResidualClassifier')
    assert result['status'] == 'complete', result
    assert result['draft']['profile']['inputs']['args'][0]['shape'] == [1, 3, 32, 32]
    assert not result.get('runtimeInputs')
    assert not any('example spatial size' in entry['detail'] for entry in result['evidence'])


def test_defaults_and_input_constraints_prepare_an_explicit_example_without_importing(tmp_path):
    marker = tmp_path / 'must-not-execute'
    result = resolved(f'from pathlib import Path\nPath({str(marker)!r}).touch()\n' + LINEAR)
    assert not marker.exists()
    assert result['status'] == 'complete'
    profile = result['draft']['profile']
    assert profile['init'] == dict(args=[], kwargs={}), 'Python defaults stay authoritative'
    assert profile['inputs']['kwargs']['x']['shape'] == [1, 8]
    assert profile['device'] == 'cpu' and profile['mode'] == 'eval'
    assert any(e['kind'] == 'example' and 'batch size 1' in e['detail'] for e in result['evidence'])


@pytest.mark.parametrize('shape', ['x.shape', 'x.size()'])
def test_linear_boundary_preserves_rank_required_by_direct_shape_unpacking(shape):
    result = resolved(LINEAR.replace('return self.fc(x)', f'batch, sequence, width = {shape}; return self.fc(x)'))
    assert result['status'] == 'complete'
    assert result['draft']['profile']['inputs']['kwargs']['x']['shape'] == [1, 8, 8]
    assert any('rank 3' in e['detail'] for e in result['evidence'])


def test_conflicting_direct_input_ranks_require_setup():
    result = resolved(LINEAR.replace('return self.fc(x)', 'a, b = x.shape; a, b, c = x.shape; return self.fc(x)'))
    assert result['status'] == 'partial'
    assert not result['draft']['profile']['inputs']['kwargs']


def test_repository_shape_is_never_replaced_by_a_synthetic_rank():
    result = resolved(LINEAR.replace('return self.fc(x)', 'b, s, w = x.shape; return self.fc(x)') + '''
import torch
model = Net()
model(torch.ones(2, 3, 8))
''')
    assert result['draft']['profile']['inputs']['args'][0]['shape'] == [2, 3, 8]


@pytest.mark.parametrize('patch, expected', [(14, [70, 70]), ((14, 12), [70, 72]), (128, [128, 128])])
def test_undeclared_spatial_extent_is_aligned_to_declared_patch_size(patch, expected):
    result = resolved(f'''from torch import nn
class Net(nn.Module):
    def __init__(self, patch_size={patch!r}):
        super().__init__()
        self.proj = nn.Conv2d(3, 8, kernel_size=patch_size, stride=patch_size)
    def forward(self, x): return self.proj(x)
''')
    assert result['status'] == 'complete', result
    assert result['draft']['profile']['inputs']['kwargs']['x']['shape'] == [1, 3, *expected]
    assert any('example spatial size' in e['detail'] for e in result['evidence'])


def test_incompatible_declared_image_size_is_not_silently_changed():
    result = resolved('''from torch import nn
class Net(nn.Module):
    def __init__(self, image_size=64, patch_size=14):
        self.proj = nn.Conv2d(3, 8, kernel_size=patch_size, stride=patch_size)
    def forward(self, x): return self.proj(x)
''')
    assert result['status'] == 'partial'
    assert not result['draft']['profile']['inputs']['kwargs']


def test_repository_call_takes_priority_over_synthetic_sizes():
    result = resolved(LINEAR + '''
import torch
model = Net(width=12)
model(torch.ones(3, 12))
''')
    assert result['status'] == 'complete'
    assert result['draft']['profile']['inputs']['args'][0]['shape'] == [3, 12]
    assert result['evidence'][0]['kind'] == 'repository'


def test_conflicting_examples_require_a_choice_and_never_pick_the_first():
    result = resolved(LINEAR + '''
import torch
model = Net()
model(torch.ones(3, 8))
model(torch.ones(7, 8))
''')
    assert result['status'] == 'partial'
    assert any('Several example calls' in item for item in result['unresolved'])


def test_constructor_call_supplies_a_required_parameter():
    result = resolved(LINEAR.replace('width=8', 'width') + '\nmodel = Net(width=12)\n')
    assert result['status'] == 'complete'
    assert result['draft']['profile']['init']['kwargs'] == {'width': 12}
    assert result['draft']['profile']['inputs']['kwargs']['x']['shape'] == [1, 12]


def test_missing_dimension_gets_an_explicit_editable_example():
    result = resolved(LINEAR.replace('width=8', 'width'))
    assert result['status'] == 'complete'
    assert result['draft']['profile']['init']['kwargs'] == {'width': 8}
    assert result['draft']['profile']['inputs']['kwargs']['x']['shape'] == [1, 8]
    assert any(e['field'] == 'init.width' and e['kind'] == 'example' for e in result['evidence'])


def test_unknown_source_override_can_use_the_declared_default_for_an_example():
    result = resolved(LINEAR + '\nmodel = Net(width=options.width)\n')
    assert result['status'] == 'complete', result
    assert result['draft']['profile']['init'] == dict(args=[], kwargs={})
    assert result['draft']['profile']['inputs']['kwargs']['x']['shape'] == [1, 8]
    assert any(e['field'] == 'init.width' and e['kind'] == 'example' for e in result['evidence'])


def test_unresolved_cli_dimension_gets_an_example_and_optional_flag_keeps_its_default():
    result = resolved(LINEAR.replace('width=8', 'width, bilinear=False') +
                      '\nmodel = Net(width=args.width, bilinear=args.bilinear)\n')
    assert result['status'] == 'complete'
    assert result['draft']['profile']['init']['kwargs'] == {'width': 8}
    assert not any('bilinear' in reason for reason in result['unresolved'])
    assert any(e['field'] == 'init.bilinear' and e['kind'] == 'example' for e in result['evidence'])


def test_known_constructor_value_wins_over_a_default_backed_source_call():
    result = resolved(LINEAR + '\na = Net(width=args.width)\nb = Net(width=12)\n')
    assert result['status'] == 'complete'
    assert result['draft']['profile']['init']['kwargs'] == {'width': 12}
    assert result['draft']['profile']['inputs']['kwargs']['x']['shape'] == [1, 12]


def test_nonliteral_default_is_not_an_automatic_replacement_for_missing_source_value():
    result = resolved(LINEAR.replace('width=8', 'width=load_width()') + '\nmodel = Net(width=args.width)\n')
    assert result['status'] == 'partial'
    assert any('init.width' in reason for reason in result['unresolved'])


def test_inherited_direct_boundary_resolves_without_model_name_rules():
    result = resolved(LINEAR.replace('class Net', 'class Base') + '\nclass Net(Base): pass\n')
    assert result['status'] == 'complete', result
    assert result['draft']['profile']['inputs']['kwargs']['x']['shape'] == [1, 8]


def test_conditional_layer_replacement_is_not_treated_as_a_fixed_width():
    result = resolved(LINEAR.replace('self.fc = nn.Linear(width, 4)', 'self.fc = nn.Linear(width, 4)\n        if flag:\n            self.fc = other()'))
    assert result['status'] == 'partial'


def test_shadowed_torch_namespace_does_not_establish_tensor_constraints():
    result = resolved(LINEAR.replace('width=8', 'width=8, nn=custom_layers'))
    assert result['status'] == 'partial'


def test_sequential_input_boundary_uses_the_first_operation():
    result = resolved(LINEAR.replace('nn.Linear(width, 4)', 'nn.Sequential(nn.Linear(width, 4), nn.ReLU())'))
    assert result['status'] == 'complete'
    assert result['draft']['profile']['inputs']['kwargs']['x']['shape'] == [1, 8]


def test_partial_multi_input_setup_keeps_the_resolved_input():
    result = resolved(LINEAR.replace('forward(self, x)', 'forward(self, x, *, context)'))
    assert result['status'] == 'partial'
    assert result['draft']['profile']['inputs']['kwargs']['x']['shape'] == [1, 8]
    assert result['unresolved'] == ['inputs.context: provide a value.']


def test_config_defaults_resolve_embedding_input_dtype_and_range():
    result = resolved('''from torch import nn
from dataclasses import dataclass
@dataclass
class Config:
    vocab_size: int = 32
    block_size: int = 6
class Net(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.tokens = nn.Embedding(config.vocab_size, 16)
    def forward(self, tokens): return self.tokens(tokens)
''')
    assert result['status'] == 'complete', result
    assert result['draft']['configObject'] == dict(index=0, parameter='config', values={})
    tensor = result['draft']['profile']['inputs']['kwargs']['tokens']
    assert tensor['shape'] == [1, 6] and tensor['dtype'] == 'int64'
    assert tensor['generator']['high'] == 32


def test_imported_config_defaults_resolve_a_complete_example():
    result = resolved('''from torch import nn
from settings import Dimensions
class Net(nn.Module):
    def __init__(self, config: Dimensions):
        super().__init__()
        self.fc = nn.Linear(config.width, 4)
    def forward(self, x): return self.fc(x)
''', extra={'settings.py': '''from dataclasses import dataclass
@dataclass
class Dimensions:
    width: int = 12
'''})
    assert result['status'] == 'complete', result
    assert result['draft']['configObject'] == dict(index=0, parameter='config', values={})
    assert result['draft']['profile']['inputs']['kwargs']['x']['shape'] == [1, 12]


def test_documented_model_retains_its_direct_input_constraint():
    result = resolved(LINEAR.replace('class Net', '@auto_docstring\nclass Net').replace(
        '    def forward', '    @auto_docstring\n    def forward').replace(
        'from torch import nn', 'from torch import nn\nfrom transformers.utils import auto_docstring'))
    assert result['status'] == 'complete', result
    assert result['draft']['profile']['inputs']['kwargs']['x']['shape'] == [1, 8]


def test_config_instance_from_repository_is_reused_without_running_its_factory():
    result = resolved('''from torch import nn
from dataclasses import dataclass
@dataclass
class Config:
    width: int
class Net(nn.Module):
    def __init__(self, config: Config):
        self.fc = nn.Linear(config.width, 4)
    def forward(self, x): return self.fc(x)
config = Config(width=12)
model = Net(config)
''')
    assert result['status'] == 'complete', result
    assert result['draft']['configObject']['values'] == {'width': 12}
    assert result['draft']['profile']['inputs']['kwargs']['x']['shape'] == [1, 12]


def test_delegated_forward_follows_a_nested_input_module_and_declared_image_size():
    result = resolved('''from torch import nn
class Patch(nn.Module):
    def __init__(self, img_size=224, in_chans=3, patch_size=16):
        self.proj = nn.Conv2d(in_chans, 32, kernel_size=patch_size, stride=patch_size)
    def forward(self, x): return self.proj(x)
class Net(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_layer=Patch):
        self.patch = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans)
    def features(self, x, mask=None): return self.patch(x)
    def forward(self, *args, is_training=False, **kwargs):
        return self.features(*args, **kwargs)
''')
    assert result['status'] == 'complete', result
    assert result['draft']['profile']['inputs']['kwargs']['x']['shape'] == [1, 3, 224, 224]
    assert any('declared spatial size' in item['detail'] for item in result['evidence'])


@pytest.mark.parametrize('body', ['x = x.flatten(1)\n        return self.fc(x)',
    'if flag:\n            x = other(x)\n        return self.fc(x)'])
def test_transformed_inputs_cannot_inherit_the_later_layer_shape(body):
    result = resolved(LINEAR.replace('return self.fc(x)', '\n        ' + body))
    assert result['status'] == 'partial'
    assert not result['draft']['profile']['inputs']['kwargs']


def test_unknown_variadic_forward_does_not_become_an_empty_call():
    result = resolved('from torch import nn\nclass Net(nn.Module):\n    def forward(self, *args, **kwargs): return external(*args, **kwargs)\n')
    assert result['status'] == 'partial'
    assert any('dynamic arguments' in item for item in result['unresolved'])


def test_declared_zero_input_forward_can_open_without_setup():
    result = resolved('from torch import nn\nclass Net(nn.Module):\n    def forward(self): return 1\n')
    assert result['status'] == 'complete'
    assert result['draft']['profile']['inputs'] == dict(args=[], kwargs={})


def test_oversized_synthetic_tensors_require_explicit_setup():
    result = resolved('''from torch import nn
class Net(nn.Module):
    def __init__(self, image_size=100000): self.conv = nn.Conv2d(3, 4, 3)
    def forward(self, x): return self.conv(x)
''')
    assert result['status'] == 'partial'


def test_concrete_constructor_example_beats_unknown_cli_options():
    result = resolved(LINEAR.replace('width=8', 'width') + '\na = Net(width=args.width)\nb = Net(width=12)\n')
    assert result['status'] == 'complete'
    assert result['draft']['profile']['inputs']['kwargs']['x']['shape'] == [1, 12]


def test_sequential_subclass_boundary_is_resolved():
    source = LINEAR.replace('nn.Linear(width, 4)', 'Block(width)') + '''
class Block(nn.Sequential):
    def __init__(self, width):
        super().__init__(nn.Linear(width, 4), nn.ReLU())
'''
    result = resolved(source)
    assert result['status'] == 'complete'
    assert result['draft']['profile']['inputs']['kwargs']['x']['shape'] == [1, 8]


def test_recursive_sequential_boundary_is_bounded():
    source = LINEAR.replace('nn.Linear(width, 4)', 'nn.Sequential(Net())')
    assert resolved(source)['status'] == 'partial'


def test_annotated_config_with_missing_dimensions_gets_an_explicit_example():
    source = '''from torch import nn
from dataclasses import dataclass
@dataclass
class Dimensions:
    width: int
class Net(nn.Module):
    def __init__(self, dims: Dimensions): self.fc = nn.Linear(dims.width, 4)
    def forward(self, x): return self.fc(x)
'''
    result = resolved(source)
    assert result['status'] == 'complete'
    assert result['draft']['configObject'] == dict(index=0, parameter='dims', values={'width': 8})
    assert any(e['field'] == 'config.width' and e['kind'] == 'example' for e in result['evidence'])


@pytest.mark.parametrize('mutation', ['model.double()', 'modify(model)', 'model.fc = external()', 'x.resize_(2, 16)'])
def test_mutated_repository_invocations_are_not_treated_as_complete_recipes(mutation):
    result = resolved(LINEAR + f'\nimport torch\nmodel = Net()\nx = torch.ones(2, 8)\n{mutation}\nmodel(x)\n')
    assert not any(item['field'] == 'call' for item in result['evidence'])


def test_automatic_limit_counts_all_inputs_together():
    source = LINEAR.replace('forward(self, x)', 'forward(self, x, y)') + '''
import torch
model = Net()
model(torch.ones(300000, 8), torch.ones(300000, 8))
'''
    result = resolved(source)
    assert result['status'] == 'partial'
    assert any('size limit' in item for item in result['unresolved'])


def test_unrelated_source_scopes_cannot_exhaust_model_binding_resolution(monkeypatch):
    from services.ide.graph_setup_resolution import Resolver
    def limited(self):
        self.steps += 1
        return self.steps <= 30
    monkeypatch.setattr(Resolver, 'available', limited)
    result = resolved(LINEAR, extra={'training.py': '\n'.join(f'def task{i}():\n    x = work({i})' for i in range(20))})
    assert result['status'] == 'complete'


def test_unsupported_constructor_spread_is_not_silently_replaced_with_defaults():
    result = resolved(LINEAR + '\nmodel = Net(**options)\n')
    assert result['status'] == 'partial'
    assert any('source constructor call' in item for item in result['unresolved'])


@pytest.mark.parametrize('mutation', ['if condition:\n            width = 12', 'width += 4'])
def test_mutated_constructor_locals_do_not_establish_a_fixed_input_width(mutation):
    result = resolved(LINEAR.replace('self.fc = nn.Linear', mutation + '\n        self.fc = nn.Linear'))
    assert result['status'] == 'partial'
    assert not result['draft']['profile']['inputs']['kwargs']


def test_sequential_subclass_modified_after_initialization_needs_explicit_setup():
    source = LINEAR.replace('nn.Linear(width, 4)', 'Block(width)') + '''
class Block(nn.Sequential):
    def __init__(self, width):
        super().__init__(nn.Linear(width, 4))
        self[0] = other_layer()
'''
    assert resolved(source)['status'] == 'partial'


def test_unresolved_factory_alternatives_do_not_override_the_selected_class_defaults():
    result = resolved(LINEAR + '\ndef alternative(**options):\n    return Net(**options)\n')
    assert result['status'] == 'complete'
    assert result['draft']['profile']['init'] == dict(args=[], kwargs={})
    assert any(item['field'] == 'init.width' and item['kind'] == 'source' for item in result['evidence'])


def test_dotted_torch_import_retains_its_python_binding():
    result = resolved(LINEAR.replace('from torch import nn', 'import torch.nn').replace('nn.', 'torch.nn.'))
    assert result['status'] == 'complete'
    assert result['draft']['profile']['inputs']['kwargs']['x']['shape'] == [1, 8]


def test_unique_factory_resolves_custom_construction_but_does_not_guess_its_input():
    source = LINEAR.replace('width=8', 'width') + '\ndef make_model():\n    return Net(int("12"))\n'
    result = resolved(source)
    assert result['draft']['profile']['factory'] == dict(file='model.py', callable='make_model', kwargs={})
    assert 'init' not in result['draft']['profile']
    assert result['draft']['profile']['inputs'] == dict(args=[], kwargs={})
    assert result['unresolved'] == ['inputs.x: provide a value.']
    assert any(item['field'] == 'factory' and item['kind'] == 'example' for item in result['evidence'])
    assert not any(item['field'].startswith(('init', 'inputs.')) for item in result['evidence'])


@pytest.mark.parametrize('signature', ['width', '*, width'])
def test_factory_requiring_arguments_or_dynamic_setup_is_not_selected(signature):
    source = LINEAR.replace('width=8', 'width') + f'\ndef make_model({signature}):\n    return Net(int("12"))\n'
    result = resolved(source)
    assert result['status'] == 'partial'
    assert 'factory' not in result['draft']['profile']


def test_multiple_factories_offer_the_first_example_variant_for_runtime_verification():
    source = LINEAR.replace('width=8', 'width') + '''
def small(): return Net(int("8"))
def large(): return Net(int("16"))
'''
    result = resolved(source)
    assert result['status'] == 'partial'
    assert result['draft']['profile']['factory']['callable'] == 'small'
    assert result['runtimeInputs'] is True


def test_factory_cannot_override_a_complete_repository_recipe():
    source = LINEAR.replace('width=8', 'width') + '''
import torch
model = Net(width=8)
model(torch.ones(2, 8))
def make_model(): return Net(int("12"))
'''
    result = resolved(source)
    assert result['status'] == 'complete'
    assert 'factory' not in result['draft']['profile']
    assert result['draft']['profile']['inputs']['args'][0]['shape'] == [2, 8]


def test_factory_removes_input_hints_from_unresolved_direct_construction():
    source = LINEAR.replace('width=8', 'width, config').replace('nn.Linear(width, 4)', 'nn.Linear(8, 4)')
    result = resolved(source + '\ndef make_model():\n    return Net(int("12"), object())\n')
    assert 'factory' in result['draft']['profile']
    assert result['draft']['profile']['inputs'] == dict(args=[], kwargs={})
    assert result['unresolved'] == ['inputs.x: provide a value.']


def test_zero_input_model_can_use_a_unique_factory_without_a_setup_question(tmp_path):
    marker = tmp_path / 'factory-executed'
    source = LINEAR.replace('width=8', 'width').replace('forward(self, x): return self.fc(x)', 'forward(self): return 1')
    source += f'\ndef make_model():\n    open({str(marker)!r}, "w").close()\n    return Net(int("8"))\n'
    result = resolved(source)
    assert result['status'] == 'complete'
    assert result['draft']['profile']['factory']['callable'] == 'make_model'
    assert not marker.exists(), 'discovery must never execute the factory'


def test_packaging_setup_is_never_an_automatic_factory():
    result = resolved(LINEAR.replace('width=8', 'width'), extra={
        'setup.py': 'from model import Net\ndef make_model():\n    return Net(int("8"))\n'})
    assert result['status'] == 'partial'
    assert 'factory' not in result['draft']['profile']


@pytest.mark.parametrize('signature', ['*args', '**kwargs'])
def test_optional_variadic_factory_is_a_candidate_not_a_success(signature):
    result = resolved(LINEAR.replace('width=8', 'width') + f'\ndef make_model({signature}):\n    return Net(int("12"))\n')
    assert result['status'] == 'partial' and result['runtimeInputs'] is True
    assert result['draft']['profile']['factory']['callable'] == 'make_model'


@pytest.mark.parametrize('factory', ['def small(): return Net(int("8"), object())',
    'def small() -> Net:\n    if True:\n        return Net(int("8"), object())'])
def test_factory_cannot_discard_known_source_constructor_values(factory):
    source = LINEAR.replace('width=8', 'width, config') + '''
model = Net(width=12, config=unknown)
''' + factory + '\n'
    result = resolved(source)
    assert result['draft']['profile']['init']['kwargs']['width'] == 12
    assert 'factory' not in result['draft']['profile']
    assert not result.get('runtimeInputs')


def test_runtime_input_preparation_only_fills_an_empty_input_binding():
    dynamic = LINEAR.replace('self.fc = nn.Linear(width, 4)',
                            'layers = []\n        for _ in range(1): layers.append(nn.Linear(width, 4))\n        self.fc = nn.Sequential(*layers)')
    result = resolved(dynamic)
    assert result['status'] == 'partial' and result['runtimeInputs'] is True
    assert resolved(dynamic.replace('forward(self, x)', 'forward(self, x, context)'))['runtimeInputs'] is True
    assert not resolved(LINEAR).get('runtimeInputs'), 'a complete source recipe needs no constructor probe'


def test_optional_inputs_get_runtime_contract_check_before_an_empty_capture():
    result = resolved(LINEAR.replace('forward(self, x)', 'forward(self, x=None)'))
    assert result['status'] == 'partial' and result['runtimeInputs'] is True
    assert result['draft']['profile']['inputs'] == dict(args=[], kwargs={})


def test_inherited_library_signature_can_be_prepared_with_a_workspace_factory():
    result = resolved('''from library import PretrainedModel
class Net(PretrainedModel): pass

def make_model() -> Net:
    return Net(make_config())
''')
    assert result['status'] == 'partial' and result['runtimeInputs'] is True
    assert result['draft']['profile']['factory']['callable'] == 'make_model'


def test_multi_input_source_without_constructor_values_cannot_start_a_probe():
    result = resolved(LINEAR.replace('width=8', 'config').replace('nn.Linear(width, 4)', 'config.layer()')
                      .replace('forward(self, x)', 'forward(self, x=None, context=None)'))
    assert not result.get('runtimeInputs')


def test_existing_optional_input_example_is_never_replaced():
    result = resolved(LINEAR.replace('forward(self, x)', 'forward(self, x=None)') + '''
import torch
model = Net()
model(torch.ones(2, 8))
''')
    assert result['status'] == 'complete' and not result.get('runtimeInputs')
    assert result['draft']['profile']['inputs']['args'][0]['shape'] == [2, 8]
