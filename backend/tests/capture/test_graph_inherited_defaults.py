"""Missing workspace signatures must not hide usable inherited Python defaults."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from services.ide.graph_setup import discover
from services.ide.graph_setup_profiles import prepare_profile
from services.semantic.execution_plan import profile_execution
from services.semantic.projection import project_capture
from services.validation.contracts import ModelProfile
from services.validation.host_snapshot import snapshot
from tests.discovery.test_graph_setup_profiles import snapshot as sources
from tests.discovery.test_graph_setup_resolution import resolved


SOURCE = 'from library import Base\nclass Net(Base):\n    pass\n'


@pytest.mark.parametrize('body', ['pass', 'def forward(self): return 1'])
def test_absent_inherited_constructor_is_checked_in_a_worker_before_saving(body):
    result = resolved(SOURCE.replace('pass', body))
    assert result['status'] == 'partial'
    assert result.get('runtimeInputs') is True
    assert result['draft']['profile']['init'] == dict(args=[], kwargs={})
    assert any(item['field'] == 'init' and item['kind'] == 'example' for item in result['evidence'])
    assert not any(reason.startswith('init:') for reason in result['unresolved'])


@pytest.mark.parametrize('call', [
    'model = Net(width=12)',
    'Alias = Net\nmodel = Alias(width=12)',
    'model = build(Net, width=12)',
    'model = Net(config=unknown)',
    'models = [Net(config=unknown)]',
    'def run():\n    model = Net(config=unknown)',
    'def run():\n    models = [Net(config=unknown)]',
])
def test_existing_constructor_intent_never_becomes_an_empty_default_call(call):
    assert not resolved(SOURCE + call + '\n').get('runtimeInputs')


@pytest.mark.parametrize('source', [
    SOURCE.replace('pass', 'def __init__(self, config): pass'),
    SOURCE.replace('pass', '@decorator\n    def __init__(self): pass'),
    SOURCE.replace('class Net(Base)', '@decorator\nclass Net(Base)'),
    SOURCE.replace('class Net(Base)', 'class Net(Base, metaclass=Meta)'),
    SOURCE.replace('class Net(Base)', 'class Net(Base, Mixin)'),
])
def test_unsupported_local_construction_stays_in_setup(source):
    assert not resolved(source).get('runtimeInputs')


def test_incomplete_sources_do_not_authorize_an_inherited_default_probe():
    request = sources(SOURCE)
    request.truncated = True
    assert not discover(request)['resolution'].get('runtimeInputs')


def _worker(tmp_path, workspace, profile, *, probe, config_defaults=False, timeout_seconds=30):
    execution = profile_execution(ModelProfile.model_validate(profile))
    raw = snapshot(dict(root=str(workspace), executable=sys.executable, environmentRevision=0,
        session='inherited-defaults', workspaceId='inherited-defaults', runId='inherited-defaults', generation=1,
        modelFile='model.py', className='Net', timeoutMs=timeout_seconds * 1000, execution=execution))
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'version': 1, 'plan': raw['plan'], 'applicabilityDigest': raw['applicabilityDigest']}))
    plan.chmod(0o600)
    backend = Path(__file__).parent.resolve().parents[1]
    result = subprocess.run([sys.executable, '-B', str(backend/'services/ingest/bootstrap.py'),
        '--file', str(workspace/'model.py'), '--cls', 'Net', '--workspace', str(workspace),
        '--resolved-execution', *(['--prepare-inputs'] if probe else []),
        *(['--prepare-config-defaults'] if config_defaults else []), '--json'], cwd=backend,
        env={**os.environ, 'PYTHONPATH': '', 'TENSORVIZ_EXECUTION_PLAN': str(plan)},
        capture_output=True, text=True, timeout=timeout_seconds + 10)
    return result.returncode, json.loads(result.stdout)


@pytest.mark.parametrize('module,family', [
    ('mobilenetv2', 'MobileNetV2'), ('squeezenet', 'SqueezeNet'), ('densenet', 'DenseNet'),
])
def test_no_factory_library_defaults_open_a_real_graph(tmp_path, module, family):
    pytest.importorskip('torchvision')
    source = f'from torchvision.models.{module} import {family}\nclass Net({family}): pass\n'
    request = sources(source)
    resolution = discover(request)['resolution']
    assert resolution.get('runtimeInputs') is True
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace/'model.py').write_text(source)
    profile = resolution['draft']['profile']
    assert 'factory' not in profile
    code, envelope = _worker(tmp_path, workspace, profile, probe=True)
    assert code == 0, envelope
    assert 'graph' not in envelope['result']
    profile['inputs'] = envelope['result']['graphExample']['binding']
    assert profile['inputs']['kwargs']['x']['shape'] == [1, 3, 64, 64]
    prepared = prepare_profile(request, '', profile)
    assert prepared['ok'], prepared
    (workspace/'tensorviz.toml').write_text(prepared['config'])
    import tomllib
    saved = tomllib.loads(prepared['config'])['profiles'][prepared['profile']]
    code, envelope = _worker(tmp_path, workspace, saved, probe=False)
    assert code == 0, envelope
    capture = envelope['result']['diagnostics']['semantic_capture']
    projected = project_capture(capture)
    assert len(projected.nodes) > 10
    assert capture['status'] == 'complete'
    output = capture['output']
    assert output['kind'] == 'value'
    tensor = next(value for value in capture['values'] if value['id'] == output['id'])
    assert tensor['shape'] == [1, 1000]
    print(json.dumps(dict(family=family, nodes=len(projected.nodes), coverage=capture['status'])))


def test_inherited_required_parameters_fail_before_model_construction(tmp_path):
    source = 'from torch.nn import Linear\nclass Net(Linear): pass\n'
    resolution = discover(sources(source))['resolution']
    assert resolution.get('runtimeInputs') is True
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace/'model.py').write_text(source)
    code, envelope = _worker(tmp_path, workspace, resolution['draft']['profile'], probe=True)
    assert code != 0 and envelope['result'] is None
    assert not (workspace/'tensorviz.toml').exists()
