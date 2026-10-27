"""Real worker execution with actual batch/checkpoint identity and keyword inputs."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from services.ingest.execution_spec import validate_execution_spec, workspace_file

BACKEND = Path(__file__).parent.resolve().parent.parent
TENSOR = {'shape': [2, 3], 'dtype': 'float32', 'fill': 'ones'}
BASE = {'inputs': {'args': [TENSOR]}}


@pytest.mark.parametrize('change', [
    {'device': 'auto'}, {'seed': True}, {'mode': 'test'}, {'deterministic': 'yes'},
    {'gradEnabled': 1}, {'cwd': '../outside'}, {'cwd': None},
    {'inputs': {'fixture': '../secret.pt'}}, {'inputs': {'fixture': '.env'}},
    {'inputs': {'fixture': 'a.pt', 'args': []}},
    {'inputs': {'kwargs': {'x': {**TENSOR, 'shape': [10000, 10000]}}}},
    {'constructorArgs': {}, 'setup': {'file': 'model.py', 'callable': 'build'}},
    {'constructorBinding': {'args': [], 'kwargs': {}}, 'setup': {'file': 'model.py', 'callable': 'build'}},
    {'constructorArgs': {'x': float('nan')}}, {'checkpoint': {'file': '/tmp/weights.pt'}},
    {'setup': {'file': 'model.py', 'callable': 'build()'}}, {'shell': 'echo'},
])
def test_rejects_ambiguous_or_unsafe_requests(change):
    with pytest.raises(ValueError):
        validate_execution_spec({**BASE, **change})


def test_symlink_must_resolve_inside_workspace(tmp_path):
    outside = tmp_path / 'outside.pt'
    outside.write_bytes(b'not a tensor')
    ws = tmp_path / 'ws'
    ws.mkdir()
    (ws / 'batch.pt').symlink_to(outside)
    with pytest.raises(ValueError, match='inside the workspace'):
        workspace_file(str(ws), 'batch.pt')


def test_working_directory_cannot_escape_through_a_symlink(tmp_path):
    from services.ingest.execution_spec import workspace_directory
    ws = tmp_path / 'ws'
    ws.mkdir()
    (ws / 'escaped').symlink_to(tmp_path, target_is_directory=True)
    assert workspace_directory(str(ws), '.') == ws
    with pytest.raises(ValueError, match='inside the workspace'):
        workspace_directory(str(ws), 'escaped')


def test_invalid_request_does_not_import_torch_or_user_code(tmp_path):
    model = tmp_path / 'model.py'
    marker = tmp_path / 'ran'
    model.write_text(f'from pathlib import Path\nPath({str(marker)!r}).touch()\n')
    process = subprocess.run([sys.executable, '-S', '-m', 'services.ingest.worker',
        '--file', str(model), '--cls', 'Net', '--workspace', str(tmp_path),
        '--execution', json.dumps({**BASE, 'inputs': {'fixture': '../bad.pt'}})],
        cwd=BACKEND, capture_output=True, text=True, timeout=30)
    assert process.returncode == 2 and json.loads(process.stdout)['error']['type'] == 'UsageError'
    assert not marker.exists()


MODEL = '''import torch
from torch import nn
# tensorviz: input=(_ for _ in ()).throw(RuntimeError("must not evaluate"))
class Net(nn.Module):
    def __init__(self, width=3):
        super().__init__()
        self.fc = nn.Linear(width, 2, bias=False)
    def forward(self, *, tokens, scale):
        assert tokens.dtype == torch.float32 and scale == 2
        return self.fc(tokens) * scale

def build(width=3):
    model = Net(width)
    with torch.no_grad():
        model.fc.weight.fill_(7)
    model.eval()
    return model
'''


def run_worker(tmp_path, spec, source=MODEL):
    model = tmp_path / 'net.py'
    model.write_text(source)
    process = subprocess.run([sys.executable, '-m', 'services.ingest.worker', '--file', str(model),
        '--cls', 'Net', '--workspace', str(tmp_path), '--execution', json.dumps(spec)],
        cwd=BACKEND, env={**os.environ, 'KMP_DUPLICATE_LIB_OK': 'TRUE', 'PYTHONDONTWRITEBYTECODE': '1'},
        capture_output=True, text=True, timeout=90)
    return process, json.loads(process.stdout)


def fixture(tmp_path):
    torch.save({'args': (), 'kwargs': {'tokens': torch.arange(6).float().reshape(2, 3), 'scale': 2},
                'batch': {'labels': torch.tensor([0, 1])}}, tmp_path / 'batch.pt')


def test_fixture_keyword_inputs_checkpoint_and_mode_are_recorded(tmp_path):
    fixture(tmp_path)
    torch.save({'weights': {'fc.weight': torch.full((2, 3), 0.25)}}, tmp_path / 'weights.pt')
    spec = {'inputs': {'fixture': 'batch.pt'}, 'constructorArgs': {'width': 3},
            'checkpoint': {'file': 'weights.pt', 'key': 'weights'}, 'mode': 'train', 'seed': 17}
    process, envelope = run_worker(tmp_path, spec)
    assert process.returncode == 0, envelope['error']
    execution = envelope['result']['diagnostics']['execution']
    assert execution['state']['digest'] and execution['inputs']['fixtureDigest']
    assert execution['setup']['checkpoint']['missingKeys'] == []
    assert execution['moduleModes'] == {'': True, 'fc': True}
    assert execution['mode'] == 'train' and execution['seed'] == 17
    assert envelope['result']['diagnostics']['seed'] == execution['seed']
    assert execution['environment']['device'] == 'cpu'
    assert execution['environment']['torch'] == str(torch.__version__)
    assert execution['inputs']['description']['items']['kwargs']['items']['tokens']['shape'] == [2, 3]
    assert any(n['type'] == 'output' or n['id'] == 'output_node' for n in envelope['result']['graph']['nodes'])
    _, repeat = run_worker(tmp_path, spec)
    again = repeat['result']['diagnostics']['execution']
    assert again['inputs']['digest'] == execution['inputs']['digest']
    assert again['state']['digest'] == execution['state']['digest']
    torch.save({'weights': {'fc.weight': torch.full((2, 3), 0.5)}}, tmp_path / 'weights.pt')
    _, changed = run_worker(tmp_path, spec)
    assert changed['result']['diagnostics']['execution']['state']['digest'] != execution['state']['digest']


def test_model_factory_preserves_custom_state_and_rejects_wrong_class(tmp_path):
    fixture(tmp_path)
    spec = {'inputs': {'fixture': 'batch.pt'}, 'setup': {'file': 'net.py', 'callable': 'build'}, 'mode': 'preserve'}
    process, envelope = run_worker(tmp_path, spec)
    assert process.returncode == 0, envelope['error']
    execution = envelope['result']['diagnostics']['execution']
    assert execution['moduleModes'] == {'': False, 'fc': False}
    assert execution['setup']['factoryDigest'] and execution['state']['digest']
    bad_source = MODEL.replace('return model\n', 'return nn.Identity()\n')
    process, envelope = run_worker(tmp_path, spec, bad_source)
    assert process.returncode == 3 and 'selected model class' in envelope['error']['message']


def test_keyword_tensor_descriptions_and_train_mode_are_actually_used(tmp_path):
    source = '''import torch
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
    def forward(self, *, x):
        assert self.training and self.dropout.training
        assert x.shape == (2, 3)
        return self.dropout(x)
'''
    process, envelope = run_worker(tmp_path, {'inputs': {'kwargs': {'x': TENSOR}}, 'mode': 'train'}, source)
    assert process.returncode == 0, envelope['error']
    assert envelope['result']['diagnostics']['execution']['moduleModes']['dropout'] is True


def test_unavailable_device_fails_instead_of_silently_using_cpu(tmp_path):
    if torch.cuda.device_count() >= 100:
        pytest.fail('Test assumes fewer than 100 CUDA devices; choose an unavailable index.')
    process, envelope = run_worker(tmp_path, {**BASE, 'device': 'cuda:99'})
    assert process.returncode == 3 and 'unavailable' in envelope['error']['message']


def test_non_strict_checkpoint_discloses_missing_keys(tmp_path):
    fixture(tmp_path)
    torch.save({}, tmp_path / 'empty.pt')
    process, envelope = run_worker(tmp_path, {'inputs': {'fixture': 'batch.pt'},
        'checkpoint': {'file': 'empty.pt', 'strict': False}})
    assert process.returncode == 0, envelope['error']
    assert envelope['result']['diagnostics']['execution']['setup']['checkpoint']['missingKeys'] == ['fc.weight']


def test_input_identity_preserves_aliases_and_keyword_order():
    from services.ingest.execution_setup import describe_tree, tree_digest
    tensor = torch.arange(4).float()
    shared = describe_tree({'left': tensor[:2], 'right': tensor[2:]})
    separate = describe_tree({'left': tensor[:2].clone(), 'right': tensor[2:].clone()})
    assert shared['items']['left']['storageGroup'] == shared['items']['right']['storageGroup']
    assert tree_digest(shared) != tree_digest(separate)
    reverse = describe_tree({'right': tensor[2:], 'left': tensor[:2]})
    assert tree_digest(reverse) != tree_digest(shared)


def test_fixture_rejects_custom_pickle_objects(tmp_path):
    fixture(tmp_path)
    torch.save({'args': (Path('unexpected-object'),)}, tmp_path / 'batch.pt')
    process, envelope = run_worker(tmp_path, {'inputs': {'fixture': 'batch.pt'}})
    assert process.returncode == 3 and envelope['result'] is None


def test_empty_argument_model_does_not_search_for_synthetic_inputs(tmp_path):
    source = """import torch
from torch import nn
class Net(nn.Module):
    def forward(self):
        return torch.ones(3)
"""
    process, envelope = run_worker(tmp_path, {'inputs': {'args': []}}, source)
    assert process.returncode == 0, envelope['error']
    assert envelope['result']['diagnostics']['timings_ms']['input_candidates_tried'] == 1


@pytest.mark.parametrize('enabled', [True, False])
def test_explicit_gradient_context_is_the_context_actually_traced(tmp_path, enabled):
    source = f'''import torch
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 2)
    def forward(self, x):
        assert torch.is_grad_enabled() is {enabled!r}
        return self.fc(x)
'''
    process, envelope = run_worker(tmp_path, {**BASE, 'gradEnabled': enabled}, source)
    assert process.returncode == 0, envelope['error']
    assert envelope['result']['diagnostics']['execution']['gradEnabled'] is enabled


def test_selected_working_directory_applies_to_construction_and_forward(tmp_path):
    (tmp_path / 'experiment').mkdir()
    (tmp_path / 'experiment/width.txt').write_text('3')
    source = '''from pathlib import Path
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(int(Path('width.txt').read_text()), 2)
    def forward(self, x):
        assert Path('width.txt').read_text() == '3'
        return self.fc(x)
'''
    process, envelope = run_worker(tmp_path, {**BASE, 'cwd': 'experiment'}, source)
    assert process.returncode == 0, envelope['error']
    assert envelope['result']['diagnostics']['execution']['cwd'] == 'experiment'


def test_explicit_null_seed_does_not_become_the_automatic_trace_seed(tmp_path, monkeypatch):
    from services.ingest.service import ingest_source
    from services.ingest.spec import ModelRef
    file = tmp_path / 'null_seed_model.py'
    file.write_text('''import torch
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        assert torch.initial_seed() == 12345
    def forward(self, x):
        assert torch.initial_seed() == 12345
        return x + 1
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(12345)
            state = torch.get_rng_state().clone()
            result = ingest_source(ModelRef(file=str(file), cls='Net'), workspace=str(tmp_path),
                                   execution={**BASE, 'seed': None}, tier='real')
            assert result.diagnostics.execution['seed'] is None
            assert torch.equal(torch.get_rng_state(), state)
    finally:
        sys.modules.pop('null_seed_model', None)


def test_seed_accepts_the_saved_profile_safe_integer_range(tmp_path):
    source = '''import torch
from torch import nn
class Net(nn.Module):
    def forward(self, x):
        assert torch.initial_seed() == 9007199254740991
        return x + 1
'''
    process, envelope = run_worker(tmp_path, {**BASE, 'seed': 2**53 - 1}, source)
    assert process.returncode == 0, envelope['error']


def test_explicit_execution_cannot_mix_in_legacy_backward_flags(tmp_path):
    process = subprocess.run([sys.executable, '-S', '-m', 'services.ingest.worker',
        '--file', str(tmp_path / 'unimported.py'), '--cls', 'Net', '--workspace', str(tmp_path),
        '--execution', json.dumps(BASE), '--capture', 'grads', '--loss', 'out.sum()'],
        cwd=BACKEND, capture_output=True, text=True, timeout=30)
    assert process.returncode == 2 and json.loads(process.stdout)['error']['type'] == 'UsageError'


def test_randomness_identity_tracks_python_numpy_and_torch_separately():
    import random
    from services.ingest.execution_setup import rng_identity
    device = torch.device('cpu')
    baseline = rng_identity(device)
    random.random()
    python_changed = rng_identity(device)
    assert baseline['pythonDigest'] != python_changed['pythonDigest']
    assert baseline['torchCpuDigest'] == python_changed['torchCpuDigest']
    torch.rand(1)
    torch_changed = rng_identity(device)
    assert torch_changed['torchCpuDigest'] != python_changed['torchCpuDigest']
    assert torch_changed['complete'] is True
    np = sys.modules.get('numpy')
    if np is not None:
        np.random.rand()
        numpy_changed = rng_identity(device)
        assert numpy_changed['numpyDigest'] != torch_changed['numpyDigest']
        assert numpy_changed['torchCpuDigest'] == torch_changed['torchCpuDigest']


def test_worker_tracks_actual_helper_source_instead_of_timestamp_valid_bytecode(tmp_path):
    import hashlib
    import py_compile
    helper = tmp_path / 'recipe_helper.py'
    helper.write_text('WIDTH = 3\n')
    stamp = helper.stat().st_mtime
    py_compile.compile(str(helper), doraise=True)
    helper.write_text('WIDTH = 4\n')  # Same size and timestamp: normal imports use stale pyc.
    os.utime(helper, (stamp, stamp))
    (tmp_path / 'recipes.py').write_text('from net import Net\nfrom recipe_helper import WIDTH\ndef build(): return Net(WIDTH)\n')
    model = MODEL.replace("assert tokens.dtype == torch.float32 and scale == 2", "assert tokens.shape[-1] == self.fc.in_features")
    spec = {'inputs': {'kwargs': {'tokens': {**TENSOR, 'shape': [2, 4]}, 'scale': {'shape': [], 'dtype': 'float32', 'fill': 'ones'}}},
            'setup': {'file': 'recipes.py', 'callable': 'build'}}
    process, envelope = run_worker(tmp_path, spec, model)
    assert process.returncode == 0, envelope['error']
    imports = {entry['file']: entry for entry in envelope['result']['diagnostics']['execution']['importedSources']}
    assert {'net.py', 'recipes.py', 'recipe_helper.py'} <= imports.keys()
    assert imports['recipe_helper.py']['digest'] == hashlib.sha256(helper.read_bytes()).hexdigest()


def test_import_loader_is_restored_on_failure_and_rejects_workspace_bytecode(tmp_path):
    import importlib.machinery
    import py_compile
    from services.ingest.execution_sources import workspace_imports
    helper = tmp_path / 'only_bytecode.py'
    helper.write_text('WIDTH = 3\n')
    bytecode = tmp_path / 'only_bytecode.pyc'
    py_compile.compile(str(helper), cfile=str(bytecode), doraise=True)
    original = importlib.machinery.SourceFileLoader.get_code
    with pytest.raises(ValueError, match='bytecode-only'):
        with workspace_imports(str(tmp_path)):
            importlib.machinery.SourcelessFileLoader('only_bytecode', str(bytecode)).get_code('only_bytecode')
    assert importlib.machinery.SourceFileLoader.get_code is original


def test_aliased_workspace_tracks_both_paths_and_still_rejects_escape(tmp_path):
    import importlib.machinery
    from services.ingest.execution_sources import workspace_imports
    actual = tmp_path / 'actual'
    actual.mkdir()
    alias = tmp_path / 'workspace'
    alias.symlink_to(actual, target_is_directory=True)
    (actual / 'helper.py').write_text('VALUE = 1\n')
    outside = tmp_path / 'outside.py'
    outside.write_text('VALUE = 2\n')
    (actual / 'escape.py').symlink_to(outside)
    original = importlib.machinery.SourceFileLoader.get_code
    with workspace_imports(str(alias)) as sources:
        for root in (actual, alias):
            importlib.machinery.SourceFileLoader('helper', str(root / 'helper.py')).get_code('helper')
        assert [item['file'] for item in sources] == ['helper.py', 'helper.py']
        assert sources[0]['digest'] == sources[1]['digest']
        with pytest.raises(ValueError, match='escapes'):
            importlib.machinery.SourceFileLoader('escape', str(alias / 'escape.py')).get_code('escape')
    assert importlib.machinery.SourceFileLoader.get_code is original


FACTORY_MODEL = """import torch
from torch import nn
class Net(nn.Module):
    def forward(self, noisy_mel, timestep, tokens, lengths):
        assert noisy_mel.shape == (1, 64, 80) and noisy_mel.dtype == torch.float32
        assert noisy_mel[0, 0, 1].item() == 1
        assert timestep.tolist() == [0.5]
        assert tokens.dtype == torch.int64 and tokens.tolist() == [list(range(17))]
        assert lengths.dtype == torch.int64 and lengths.tolist() == [17]
        return noisy_mel * timestep[:, None, None]
def graph_inputs():
    return dict(noisy_mel=torch.arange(5120).float().reshape(1, 64, 80),
                timestep=torch.tensor([0.5]), tokens=torch.arange(17).reshape(1, 17),
                lengths=torch.tensor([17]))
"""


def test_python_inputs_preserve_all_four_dit_tensors_and_source_identity(tmp_path):
    spec = {'inputs': {'factory': {'file': 'net.py', 'callable': 'graph_inputs'}}}
    process, envelope = run_worker(tmp_path, spec, FACTORY_MODEL)
    assert process.returncode == 0, envelope['error']
    inputs = envelope['result']['diagnostics']['execution']['inputs']
    assert inputs['factoryDigest'] and inputs['factory']['callable'] == 'graph_inputs'
    assert inputs['description']['items']['kwargs']['items']['noisy_mel']['shape'] == [1, 64, 80]


def test_input_factory_explicit_positional_binding_preserves_aliases(tmp_path):
    from services.ingest.execution_setup import factory_inputs
    (tmp_path / 'inputs.py').write_text('import torch\ndef make():\n    x = torch.ones(3)\n    return {"args": (x,), "kwargs": {"alias": x}}\n')
    args, kwargs = factory_inputs(str(tmp_path), {'file': 'inputs.py', 'callable': 'make'}, torch.device('cpu'))
    assert args[0] is kwargs['alias']


@pytest.mark.parametrize('body, error', [
    ('return torch.ones(3)', 'must return'),
    ('return {"args": 1, "kwargs": {}}', 'args must'),
    ('return {1: torch.ones(3)}', 'string keys'),
    ('return {"x": object()}', 'tensors and JSON literals'),
    ('return {"x": torch.ones(4_194_305)}', 'tensor-element budget'),
    ('from pathlib import Path; Path(__file__).write_text("changed"); return {}', 'source changed'),
])
def test_input_factory_rejects_unusable_or_changed_examples(tmp_path, body, error, monkeypatch):
    monkeypatch.delitem(sys.modules, "inputs", raising=False)
    from services.ingest.execution_setup import factory_inputs
    (tmp_path / 'inputs.py').write_text('import torch\ndef make():\n    ' + body + '\n')
    with pytest.raises(ValueError, match=error):
        factory_inputs(str(tmp_path), {'file': 'inputs.py', 'callable': 'make'}, torch.device('cpu'))


def test_input_factory_cannot_mix_with_bindings_or_escape_workspace(tmp_path):
    with pytest.raises(ValueError):
        validate_execution_spec({'inputs': {'factory': {'file': 'net.py', 'callable': 'graph_inputs'}, 'args': []}})
    with pytest.raises(ValueError):
        validate_execution_spec({'inputs': {'factory': {'file': '../outside.py', 'callable': 'make'}}})
    from services.ingest.execution_setup import factory_inputs
    outside = tmp_path / 'outside.py'
    outside.write_text('raise AssertionError("must not execute")')
    ws = tmp_path / 'ws'; ws.mkdir()
    (ws / 'inputs.py').symlink_to(outside)
    with pytest.raises(ValueError, match='workspace'):
        factory_inputs(str(ws), {'file': 'inputs.py', 'callable': 'make'}, torch.device('cpu'))
