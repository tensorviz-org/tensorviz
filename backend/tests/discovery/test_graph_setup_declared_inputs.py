"""Explicit numeric input comments must not be replaced by inferred examples."""
import pytest
import tomllib

from tests.discovery.test_graph_setup_resolution import LINEAR, resolved


TWO_MODELS = '''from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.fc = nn.Linear(8, 4)
    def forward(self, x): return self.fc(x)
class Block(nn.Module):
    def __init__(self):
        super().__init__(); self.fc = nn.Linear(16, 16)
    def forward(self, x): return self.fc(x)
'''
NET, BLOCK = 'class Net(nn.Module):\n', 'class Block(nn.Module):\n'
#: A module base without a forward of its own, and the model built on it.
BASE = '''import torch.nn as nn
class Base(nn.Module):
    def __init__(self):
        super().__init__(); self.fc = nn.Linear(8, 8)
class Net(Base):
    def forward(self, x): return self.fc(x)
'''
#: A model on a base this file cannot prove, and its subclass without forward.
UNPROVEN_SUBCLASS = '''# tensorviz: input=5,8
from vendor import Base
class Net(Base):
    def forward(self, x): return x
class Wide(Net):
    pass
'''


def declared(value, source=LINEAR):
    return resolved(f'# tensorviz: input={value}\n' + source)


def shapes(result):
    inputs = result['draft']['profile']['inputs']
    return [value['shape'] for value in [*inputs['args'], *inputs['kwargs'].values()]]


def test_declared_shape_wins_over_synthetic_dimensions():
    result = declared('5,8')
    assert result['status'] == 'complete'
    assert result['draft']['profile']['inputs']['args'][0]['shape'] == [5, 8]
    assert not result['draft']['profile']['inputs']['kwargs']
    assert any(e['field'] == 'inputs' and e['kind'] == 'source' and e['line'] == 1 for e in result['evidence'])


def test_declaration_replaces_example_inputs_but_preserves_its_constructor():
    result = declared('5,12', LINEAR + '\nimport torch\nmodel = Net(width=12)\nmodel(torch.ones(3,12))\n')
    assert result['status'] == 'complete'
    assert result['draft']['profile']['init']['kwargs'] == {'width': 12}
    assert result['draft']['profile']['inputs']['args'][0]['shape'] == [5, 12]


def test_multiple_declared_tensors_keep_positional_order_and_optional_defaults():
    source = LINEAR.replace('forward(self, x)', 'forward(self, x, y, scale=1)').replace('return self.fc(x)', 'return self.fc(x + y) * scale')
    result = declared('2, 8; 2,8', source)
    assert result['status'] == 'complete'
    assert [value['shape'] for value in result['draft']['profile']['inputs']['args']] == [[2, 8], [2, 8]]
    assert not result['draft']['profile']['inputs']['kwargs']


def test_declared_inputs_survive_factory_selection():
    source = LINEAR.replace('width=8', 'width') + '\ndef make_model():\n    return Net(int("8"))\n'
    result = declared('5,8', source)
    assert result['status'] == 'complete'
    assert result['draft']['profile']['factory']['callable'] == 'make_model'
    assert result['draft']['profile']['inputs']['args'][0]['shape'] == [5, 8]


@pytest.mark.parametrize('value', ['', 'torch.ones(5,8)', '-1,8', '0,8', '1,8388608', ',8',
                                   '5,8;', ','.join(['1'] * 33), '1;' * 64 + '1'])
def test_unsupported_declarations_require_setup_without_inference_or_runtime_probe(value):
    result = declared(value)
    assert result['status'] == 'partial'
    assert result['draft']['profile']['inputs'] == {'args': [], 'kwargs': {}}
    assert not result.get('runtimeInputs')
    assert any('declaration' in reason for reason in result['unresolved'])


def test_conflicting_declarations_require_a_choice():
    result = declared('5,8', '# tensorviz: input=3,8\n' + LINEAR)
    assert result['status'] == 'partial'
    assert result['draft']['profile']['inputs'] == {'args': [], 'kwargs': {}}
    assert not result.get('runtimeInputs')


@pytest.mark.parametrize('prefix', ['note = "# tensorviz: input=5,8"\n', '"""\n# tensorviz: input=5,8\n"""\n'])
def test_declaration_text_inside_strings_is_not_an_input_directive(prefix):
    result = resolved(prefix + LINEAR)
    assert result['draft']['profile']['inputs']['kwargs']['x']['shape'] == [1, 8]


@pytest.mark.parametrize('signature,value', [('x, *, mask', '5,8'), ('x', '5,8;5,8')])
def test_incompatible_argument_counts_cannot_complete_setup(signature, value):
    result = declared(value, LINEAR.replace('forward(self, x)', f'forward(self, {signature})'))
    assert result['status'] == 'partial'
    assert any(reason.startswith('inputs') for reason in result['unresolved'])
    assert not result.get('runtimeInputs')


def test_unknown_forward_keeps_explicit_inputs_without_requesting_a_probe():
    result = declared('5,8', 'from torch import nn\nclass Net(nn.Linear):\n    pass\n')
    assert result['status'] == 'partial'
    assert result['draft']['profile']['inputs']['args'][0]['shape'] == [5, 8]
    assert not result.get('runtimeInputs')


@pytest.mark.parametrize('comment', ['5,8', 'torch.ones(5,8)'])
def test_explicit_profile_values_remain_authoritative_over_comments(comment):
    from services.ide.graph_setup_profiles import prepare_profile
    from tests.discovery.test_graph_setup_profiles import profile, snapshot
    selected = profile()
    prepared = prepare_profile(snapshot(f'# tensorviz: input={comment}\n' + LINEAR), '', selected)
    assert prepared['ok']
    saved = tomllib.loads(prepared['config'])['profiles'][prepared['profile']]
    assert saved['inputs'] == selected['inputs']
    assert saved['init'] == selected['init']


def test_file_top_declaration_is_not_another_model_classes_input():
    # 0.1.32 gave Block the [5, 8] written for Net as a complete setup. Complete
    # setup is saved to tensorviz.toml, stays authoritative and fails every capture.
    result = resolved('# tensorviz: input=5,8\n' + TWO_MODELS, cls='Block')
    assert result['status'] == 'partial'
    assert [5, 8] not in shapes(result)
    assert any('several model classes' in reason for reason in result['unresolved'])


@pytest.mark.parametrize('cls', ['Net', 'Block'])
@pytest.mark.parametrize('source', [
    '# tensorviz: input=5,8\n' + TWO_MODELS,
    TWO_MODELS.replace(BLOCK, '\n# tensorviz: input=5,8\n\n' + BLOCK),
    TWO_MODELS + '# tensorviz: input=5,8\n',
    'from dataclasses import dataclass\n'
    + TWO_MODELS.replace(BLOCK, '# tensorviz: input=5,8\n@dataclass\nclass Config:\n    width: int = 8\n' + BLOCK),
    TWO_MODELS.replace(BLOCK, 'def helper():\n    return 1\n    # tensorviz: input=5,8\n' + BLOCK)],
    ids=['file top', 'between classes', 'file end', 'above a dataclass', 'end of a function body'])
def test_module_level_declaration_with_several_model_classes_requires_setup(cls, source):
    # Its class is unknown: neither apply it to a guessed class nor replace it
    # with an inferred example or a runtime probe.
    result = resolved(source, cls=cls)
    assert result['status'] == 'partial'
    assert result['draft']['profile']['inputs'] == {'args': [], 'kwargs': {}}
    assert not result.get('runtimeInputs')
    assert any(reason.startswith('inputs:') and 'several model classes' in reason for reason in result['unresolved'])


@pytest.mark.parametrize('source', [
    TWO_MODELS.replace(NET, '# tensorviz: input=5,8\n# A small classifier.\n' + NET)
              .replace(BLOCK, '# tensorviz: input=3,16\n' + BLOCK),
    TWO_MODELS.replace(NET, NET + '    # tensorviz: input=5,8\n')
              .replace(BLOCK, BLOCK + '    # tensorviz: input=3,16\n'),
    TWO_MODELS.replace(NET, '# tensorviz: input=5,8\n' + NET)
              .replace(BLOCK, BLOCK + '    # tensorviz: input=3,16\n'),
    'from transformers.utils import auto_docstring\n'
    + TWO_MODELS.replace(NET, '# tensorviz: input=5,8\n' + NET)
                .replace(BLOCK, '# tensorviz: input=3,16\n@auto_docstring\n' + BLOCK),
    TWO_MODELS.replace(BLOCK, '    # tensorviz: input=5,8\n# tensorviz: input=3,16\n' + BLOCK),
    TWO_MODELS.replace(BLOCK, '    # tensorviz: input=5,8\n\n\n# tensorviz: input=3,16\n' + BLOCK)],
    ids=['above', 'inside', 'above and inside', 'above a decorator', 'end of body', 'end of body, blank lines'])
def test_declaration_above_or_inside_a_class_belongs_to_that_class_only(source):
    # The other class's declaration is not a second, conflicting declaration.
    net, block = resolved(source, cls='Net'), resolved(source, cls='Block')
    assert (net['status'], shapes(net)) == ('complete', [[5, 8]])
    assert (block['status'], shapes(block)) == ('complete', [[3, 16]])


def test_old_mac_line_breaks_keep_each_class_its_own_declaration():
    # Python reads a lone carriage return as a line break; the comment reader refused it.
    source = (TWO_MODELS.replace(NET, '# tensorviz: input=5,8\n' + NET)
              .replace(BLOCK, '# tensorviz: input=3,16\n' + BLOCK).replace('\n', '\r'))
    net, block = resolved(source, cls='Net'), resolved(source, cls='Block')
    assert (net['status'], shapes(net)) == ('complete', [[5, 8]])
    assert (block['status'], shapes(block)) == ('complete', [[3, 16]])


def test_class_without_its_own_declaration_keeps_its_inferred_example():
    result = resolved(TWO_MODELS.replace(NET, '# tensorviz: input=5,8\n' + NET), cls='Block')
    assert (result['status'], shapes(result)) == ('complete', [[1, 16]])
    assert not any(e['field'] == 'inputs' and e['kind'] == 'source' for e in result['evidence'])


def test_several_declarations_for_one_class_still_require_a_choice_only_for_that_class():
    source = TWO_MODELS.replace(NET, '# tensorviz: input=5,8\n' + NET + '    # tensorviz: input=3,8\n')
    net = resolved(source, cls='Net')
    assert net['status'] == 'partial'
    assert net['draft']['profile']['inputs'] == {'args': [], 'kwargs': {}}
    assert any('Several input declarations' in reason for reason in net['unresolved'])
    assert shapes(resolved(source, cls='Block')) == [[1, 16]]


def test_a_declaration_outside_every_class_does_not_block_a_class_with_its_own():
    # The stray one applies to no class, so it is not a second declaration for Net.
    source = TWO_MODELS.replace(NET, '# tensorviz: input=5,8\n' + NET) + '# tensorviz: input=3,16\n'
    net, block = resolved(source, cls='Net'), resolved(source, cls='Block')
    assert (net['status'], shapes(net)) == ('complete', [[5, 8]])
    assert (block['status'], shapes(block)) == ('partial', [])
    assert any('several model classes' in reason for reason in block['unresolved'])


@pytest.mark.parametrize('helper', ['from dataclasses import dataclass\n@dataclass\nclass Config:\n    width: int = 8\n',
                                    'class Registry:\n    names = []\n',
                                    'from transformers import PretrainedConfig\nclass NetConfig(PretrainedConfig):\n    model_type = "net"\n',
                                    'import torch\nclass Twice(torch.autograd.Function):\n'
                                    '    @staticmethod\n    def forward(ctx, x): return x * 2\n'],
                         ids=['dataclass', 'plain class', 'config subclass', 'autograd function'])
def test_classes_that_are_not_models_do_not_make_a_module_level_declaration_ambiguous(helper):
    result = declared('5,8', LINEAR + helper)
    assert (result['status'], shapes(result)) == ('complete', [[5, 8]])


@pytest.mark.parametrize('source', [
    LINEAR.replace(NET, 'from dataclasses import dataclass\n# tensorviz: input=5,8\n@dataclass\nclass Config:\n    width: int = 8\n\n' + NET),
    TWO_MODELS.replace(NET, NET + '    class Config:\n        # tensorviz: input=5,8\n        width = 8\n')],
    ids=['above a dataclass', 'inside a nested class'])
def test_a_class_that_is_not_a_model_does_not_take_the_declaration(source):
    # Dropping it would save the inferred [1, 8] as the authoritative setup.
    result = resolved(source)
    assert (result['status'], shapes(result)) == ('complete', [[5, 8]])


@pytest.mark.parametrize('cls', ['Net', 'Block'])
@pytest.mark.parametrize('source,file', [
    ('from vendor import Base\n' + TWO_MODELS.replace('(nn.Module)', '(Base)'), 'model.py'),
    ('from .layers import Base\n' + TWO_MODELS.replace('(nn.Module)', '(Base)'), 'pkg/model.py'),
    ('from vendor import Base\n' + TWO_MODELS.replace(NET, 'class Net(Base):\n'), 'model.py')],
    ids=['library base', 'unsupplied sibling module', 'one unproven'])
def test_models_the_source_index_cannot_prove_still_count(cls, source, file):
    # Open as Graph offers each of them, so none may take a file-level declaration alone.
    result = resolved('# tensorviz: input=5,8\n' + source, cls=cls, file=file)
    assert (result['status'], shapes(result)) == ('partial', [])
    assert any('several model classes' in reason for reason in result['unresolved'])


@pytest.mark.parametrize('source', [
    '# tensorviz: input=5,8\n' + TWO_MODELS,
    TWO_MODELS.replace(NET, '# tensorviz: input=5,8\n' + NET).replace(BLOCK, '# tensorviz: input=3,16\n' + BLOCK)],
    ids=['outside every class', 'one per class'])
def test_counting_model_classes_reuses_the_setup_module_index(monkeypatch, source):
    # Telling which model a declaration belongs to reads the index setup already built,
    # rather than indexing the whole workspace a second time.
    from services.ide.module_index import ModuleIndex
    build, builds = ModuleIndex.build.__func__, []
    monkeypatch.setattr(ModuleIndex, 'build', classmethod(lambda cls, sources: builds.append(sources) or build(cls, sources)))
    resolved(TWO_MODELS)
    undeclared, builds[:] = len(builds), []
    resolved(source)
    assert len(builds) == undeclared


def test_selected_class_counts_as_a_model_when_the_source_index_cannot_prove_its_base():
    # Open as Graph may select a library subclass; the worker checks its type.
    result = declared('5,8', 'from vendor import Base\nclass Net(Base):\n    def forward(self, x): return x\n')
    assert shapes(result) == [[5, 8]]
    assert not any('several model classes' in reason for reason in result['unresolved'])


@pytest.mark.parametrize('source', [
    '# tensorviz: input=5,8\n' + TWO_MODELS,
    TWO_MODELS + '# tensorviz: input=5,8\n',
    TWO_MODELS.replace(NET, '# tensorviz: input=5,8\n' + NET).replace(BLOCK, BLOCK + '    # tensorviz: input=3,16\n'),
    TWO_MODELS.replace(NET, '# tensorviz: input=5,8\n' + NET) + '# tensorviz: input=3,16\n',
    TWO_MODELS.replace(BLOCK, '    # tensorviz: input=5,8\n\n\n# tensorviz: input=3,16\n' + BLOCK),
    TWO_MODELS.replace(NET, '# tensorviz: input=5,8\n' + NET + '    # tensorviz: input=3,8\n'),
    TWO_MODELS.replace(NET, NET + '    class Config:\n        # tensorviz: input=5,8\n        width = 8\n'),
    '# tensorviz: input=5,8\n' + LINEAR + 'class Wide(Net):\n    pass\n',
    '# tensorviz: input=5,8\n' + LINEAR + 'from dataclasses import dataclass\n@dataclass\nclass Config:\n    width: int = 8\n',
    'from vendor import Base\n# tensorviz: input=5,8\n' + TWO_MODELS.replace('(nn.Module)', '(Base)'),
    '# tensorviz: input=5,8\n' + BASE,
    BASE.replace('class Base', '# tensorviz: input=5,8\nclass Base'),
    LINEAR + '# tensorviz: input=5,8\nclass Stem(nn.Sequential):\n    def __init__(self):\n        super().__init__(nn.Linear(8, 8))\n',
    '# tensorviz: input=5,8\n' + LINEAR + 'class Head(nn.Module):\n    def __init__(self):\n        super().__init__(); self.fc = nn.Linear(4, 2)\n',
    '# tensorviz: input=5,8\nfrom transformers import PreTrainedModel\nclass Pretrained(PreTrainedModel):\n    config_class = None\n'
    'class Net(Pretrained):\n    def forward(self, x): return x\n',
    # Only setup can tell whether vendor is a workspace module. The worker counts Wide,
    # a subclass of the model Net, as setup does when vendor.py proves Base is a module
    # (the across-files test below); for a library base outside the catalog setup does not.
    pytest.param(UNPROVEN_SUBCLASS, marks=pytest.mark.xfail(strict=True, reason='needs the file that defines Base')),
    '# tensorviz: input=5,8\nimport torch\nimport flax.linen as nn\nclass Dense(nn.Module):\n    def __call__(self, x): return x\n'
    'class Net(torch.nn.Module):\n    def forward(self, x): return x\n'],
    ids=['file top', 'file end', 'above and inside', 'own and stray', 'end of body', 'several for one class',
         'inside a nested class', 'subclass of a model in the file', 'beside a dataclass', 'unproven bases',
         'base without forward', 'above a base without forward', 'above a Sequential without forward',
         'helper without forward', 'library base without forward', 'subclass of an unproven model', 'beside a flax module'])
def test_readers_without_the_workspace_index_choose_the_declaration_setup_chooses(source):
    # The capture worker has no module index: it proves models from this file alone.
    # Setup proves them with the index; each class must get the same answer. The line
    # alone is not the answer: a declaration setup refuses as belonging to no class
    # sits on the same line the worker would run it from.
    for cls, setup, worker in _verdicts({'model.py': source}, 'model.py'):
        assert worker == setup, cls


#: A module base in one workspace file, and the models built on it in another: Net,
#: and Wide, which has no forward of its own and only changes the width.
WORKSPACE_BASE = '''import torch.nn as nn
class Base(nn.Module):
    def __init__(self, width=8):
        super().__init__(); self.fc = nn.Linear(width, width)
'''
ON_BASE = '''from models.base import Base
class Net(Base):
    def forward(self, x): return self.fc(x)
# tensorviz: input=2,16
class Wide(Net):
    def __init__(self):
        super().__init__(width=16)
'''


@pytest.mark.parametrize('sources', [
    {'models/base.py': WORKSPACE_BASE, 'models/net.py': ON_BASE},
    {'models/__init__.py': '', 'models/base.py': WORKSPACE_BASE,
     'models/net.py': ON_BASE.replace('from models.base', 'from .base')},
    {'helpers.py': 'def scale(x): return x\n', 'models/net.py': '''import torch.nn as nn
from helpers import *
class Net(nn.Module):
    def __init__(self, width=8):
        super().__init__(); self.fc = nn.Linear(width, width)
    def forward(self, x): return self.fc(x)
# tensorviz: input=2,16
class Wide(Net):
    def __init__(self):
        super().__init__(width=16)
'''},
    {'vendor.py': WORKSPACE_BASE, 'models/net.py': UNPROVEN_SUBCLASS},
    # Known gaps. A class without forward whose module base only another file proves,
    # and which subclasses no model in this file, counts for setup and not for the
    # worker, so the worker gives its declaration to the file's one model.
    pytest.param({'models/base.py': WORKSPACE_BASE, 'models/net.py': '''import torch.nn as nn
from models.base import Base
class Net(nn.Module):
    def forward(self, x): return x
# tensorviz: input=2,16
class Wide(Base):
    def __init__(self):
        super().__init__(width=16)
'''}, marks=pytest.mark.xfail(strict=True, reason='the worker cannot read models/base.py')),
    pytest.param({'helpers.py': 'def scale(x): return x\n', 'models/net.py': '''import torch.nn as nn
from helpers import *
# tensorviz: input=2,16
class Base(nn.Module):
    def __init__(self, width=16):
        super().__init__(); self.fc = nn.Linear(width, width)
class Net(Base):
    def forward(self, x): return self.fc(x)
'''}, marks=pytest.mark.xfail(strict=True, reason='the worker cannot read what helpers.py exports'))],
    ids=['imported base', 'relative import', 'wildcard import', 'library base in the workspace',
         'imported base without forward', 'base without forward after a wildcard import'])
def test_readers_without_the_workspace_index_choose_the_declaration_setup_chooses_across_files(sources):
    # Setup proves Net is a module from the other file; the worker cannot read it. Wide
    # subclasses Net in this file, so both count it and keep Wide's declaration from Net
    # rather than feed Net a width its layers do not take.
    for cls, setup, worker in _verdicts(sources, 'models/net.py'):
        assert worker == setup, cls


def _verdicts(sources, file):
    """``(class, setup's verdict, the worker's)`` for each class in ``file``."""
    import ast
    from services.ide.graph_setup import Discovery
    from services.ide.graph_setup_resolution import Resolver
    from services.ingest.input_declarations import applicable

    def verdict(setup):
        if setup is None:
            return None
        _inputs, problem, line = setup
        if problem is None:
            return 'applies', line
        # Several of the class's own: setup asks which, the worker runs the first.
        return ('several' if problem.startswith('Several input declarations') else
                'refused' if 'several model classes' in problem else problem), line

    source = sources[file]
    for cls in [node.name for node in ast.parse(source).body if isinstance(node, ast.ClassDef)]:
        setup = Resolver(Discovery(sources), file, cls, None).declared_inputs()
        own, stray = applicable(source, cls)
        worker = (('applies' if len(own) == 1 else 'several'), own[0][1]) if own else None if stray is None else ('refused', stray)
        yield cls, verdict(setup), worker


@pytest.mark.parametrize('source', [
    'import torch.nn as nn\nclass A(nn.Module): pass\nclass B(A): pass\nclass C(B): pass\n',
    'from torch import nn\nclass A(nn.Sequential): pass\n',
    'import torch\nclass A(torch.nn.Module): pass\nclass B(torch.nn.Linear): pass\nclass C(torch.nn.Parameter): pass\n',
    'import torch.nn\nclass A(torch.nn.ModuleList): pass\n',
    'from torch.nn import Module, Sequential as Seq\nclass A(Module): pass\nclass B(Seq): pass\n',
    'from torch.nn.modules.module import Module\nclass A(Module): pass\n',
    'class A(nn.Module): pass\nclass B(Module): pass\n',
    'import flax.linen as nn\nclass A(nn.Module): pass\n',
    'from flax.linen import Module\nclass A(Module): pass\n',
    'from transformers import PreTrainedModel\nclass A(PreTrainedModel): pass\nclass B(A): pass\n',
    'import torch.nn as nn\nclass A(nn.Module): pass\nclass A(object): pass\n',
    'import torch.nn as nn\nclass A(nn.Module): pass\nA = None\n',
    'import torch.nn as nn\nnn = None\nclass A(nn.Module): pass\n',
    'import torch.nn as nn\nBase = nn.Module\nclass A(Base): pass\n',
    'import torch.nn as nn\nnn.Module = object\nclass A(nn.Module): pass\n',
    'import torch.nn as nn\ndef nn(): pass\nclass A(nn.Module): pass\n',
    'import torch.nn as nn\nclass A(nn.ModuleDict[str]): pass\n',
    'try:\n    import torch.nn as nn\nexcept ImportError:\n    nn = None\nclass A(nn.Module): pass\n',
    'try:\n    import flax.linen as nn\nexcept ImportError:\n    nn = None\nclass A(nn.Module): pass\n',
    'import torch.nn as nn\nwhile False:\n    import flax.linen as nn\nclass A(nn.Module): pass\n',
    'import torch.nn as nn\nfor nn in range(3):\n    pass\nclass A(nn.Module): pass\n',
    'import torch.nn as nn\nwith open("f") as nn:\n    pass\nclass A(nn.Module): pass\n',
    'import torch.nn as nn\nif True:\n    class Base(nn.Module): pass\nclass A(Base): pass\n',
    'import torch.nn as nn\nfrom helpers import *\nclass A(nn.Module): pass\n',
    'import torch.nn as nn\nwith open("f"):\n    from helpers import *\nclass A(nn.Module): pass\n',
    'from .layers import Block\nclass A(Block): pass\n'],
    ids=['chain', 'from torch import nn', 'torch attribute', 'import torch.nn', 'from torch.nn import', 'module path',
         'shorthand', 'flax', 'flax Module', 'library base', 'redefined', 'rebound class', 'rebound nn', 'alias',
         'attribute store', 'function', 'subscript', 'try torch', 'try flax', 'loop branch', 'loop target', 'with target',
         'class in a branch', 'wildcard import', 'nested wildcard import', 'relative import'])
def test_the_worker_proves_the_modules_discovery_proves_from_one_file(source):
    # Setup and the drawing count a class without forward() as a model when the module
    # index proves it is one. The worker has no index, so it proves them from the file's
    # own imports the way discovery does from a one-file workspace.
    import ast
    from services.ingest import input_declarations
    from services.ingest.module_discovery import module_declarations

    expected = {(line, name) for _file, line, name in module_declarations({'model.py': source})}
    assert input_declarations._file_modules(ast.parse(source)) == expected


def test_the_worker_counts_the_framework_bases_discovery_counts():
    # The worker cannot import module_discovery, which reads the op registry.
    from services.ingest.input_declarations import MODULE_BASES
    from services.ingest.module_discovery import EXTERNAL_MODULE_BASES, TORCH_MODULE_BASES

    assert MODULE_BASES == TORCH_MODULE_BASES | EXTERNAL_MODULE_BASES
