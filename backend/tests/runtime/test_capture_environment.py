"""Run this suite in a minimal Python 3.10 environment as well as the tool runtime."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from services.ide import environment_setup
from services.semantic.contracts import ResolvedExecutionPlan
from services.semantic.execution_plan import applicability_digest, profile_execution
from services.validation.contracts import RunSpec
from services.validation.host_snapshot import snapshot

BACKEND = Path(__file__).resolve().parents[2]


def test_capture_requirements_do_not_require_the_server(monkeypatch):
    monkeypatch.setattr(environment_setup, 'inventory', lambda: {
        'torch': '2.6.0', 'pydantic': '2.12.0', 'packaging': '25.0', 'numpy': '2.2.0', 'tomli': '2.2.1',
    })
    result = environment_setup.inspect_environment(BACKEND / 'requirements-capture.txt')
    assert result['missing'] == result['incompatible'] == []
    assert result['pip'] is False


def test_real_capture_probe_never_imports_server_libraries():
    script = '''import sys
class RejectServerImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'fastapi', 'uvicorn', 'websockets', 'libcst', 'dotenv', 'httpx', 'requests'}:
            raise AssertionError('Capture imported a server dependency: ' + fullname)
sys.meta_path.insert(0, RejectServerImports())
from capture_environment import probe_capture
assert probe_capture()['torch']
'''
    result = subprocess.run([sys.executable, '-B', '-c', script], cwd=BACKEND,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_saved_factory_profile_captures_with_the_private_worker(tmp_path):
    workspace = tmp_path / 'project'
    workspace.mkdir()
    (workspace / 'model.py').write_text('''import torch
class Net(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    def forward(self, x):
        return self.layer(x).relu()
''')
    (workspace / 'setup_model.py').write_text('''from torch import nn
from model import Net
def build():
    return Net(nn.Linear(8, 4))
''')
    (workspace / 'tensorviz.toml').write_text('''version = 1
[profiles.small]
kind = "model"
model = "model:Net"
mode = "eval"
device = "cpu"
grad_enabled = false
factory = {file="setup_model.py", callable="build", kwargs={}}
[profiles.small.inputs]
args = [{kind="tensor", shape=[2,8], dtype="float32", requires_grad=false, generator={kind="full", value=1}}]
kwargs = {}
''')
    params = dict(root=str(workspace), executable=sys.executable, environmentRevision=0,
                  session='capture-env', workspaceId='capture-env', runId='capture-env-run', generation=1,
                  modelFile='model.py', className='Net', timeoutMs=30000, profile='small')
    resolved = snapshot(params)['resolution']
    assert resolved['status'] == 'resolved', resolved
    spec = RunSpec.model_validate(resolved['spec'])
    plan = ResolvedExecutionPlan.model_validate(snapshot(dict(params,
        expectedIdentity=spec.identity.model_dump(), execution=profile_execution(spec.profile)))['plan'])
    plan_file = tmp_path / 'plan.json'
    plan_file.write_text(json.dumps(dict(version=1, plan=plan.model_dump(mode='json'),
                                         applicabilityDigest=applicability_digest(plan))))
    plan_file.chmod(0o600)
    result = subprocess.run([sys.executable, '-B', str(BACKEND / 'services/ingest/bootstrap.py'),
        '--resolved-execution', '--workspace', str(workspace), '--file', str(workspace / 'model.py'),
        '--cls', 'Net', '--json'], cwd=BACKEND, env={**os.environ,
        'TENSORVIZ_EXECUTION_PLAN': str(plan_file), 'PYTHONPATH': str(BACKEND)},
        capture_output=True, text=True, timeout=60)
    assert result.stdout, result.stderr
    envelope = json.loads(result.stdout)
    assert result.returncode == 0, envelope.get('error')
    assert envelope['worker']['python'] == sys.version.split()[0]
    diagnostics = envelope['result']['diagnostics']
    assert diagnostics['execution']['planCurrent'] is True
    assert diagnostics['semantic_capture']['status'] == 'complete'
    assert diagnostics['semantic_capture']['operations']


@pytest.mark.parametrize('target_version,needs_tomli', [('3.10.22', True), ('3.12.15', False)])
def test_capture_markers_use_the_selected_python_not_the_tool_runtime(monkeypatch, target_version, needs_tomli):
    monkeypatch.setattr(environment_setup, 'inventory', lambda: {
        'torch': '2.14.1', 'pydantic': '2.14.0', 'packaging': '26.0', 'numpy': '2.2.0',
    })
    monkeypatch.setattr(environment_setup, 'environment_info', lambda installed: {
        'python': target_version, 'executable': '/project/python', 'prefix': '/project',
    })
    result = environment_setup.inspect_environment(environment_setup.CAPTURE_REQUIREMENTS)
    assert result['incompatible'] == []
    assert bool(result['missing']) is needs_tomli
    if needs_tomli:
        assert result['missing'] == ['tomli>=2.0; python_version < "3.11"']
