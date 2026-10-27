"""One saved invocation reaches both entry points with no identity/recipe substitution."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from services.semantic.contracts import ResolvedExecutionPlan
from services.semantic.execution_plan import applicability_digest, consume_plan_file, profile_execution, verify_saved_plan
from services.validation.contracts import RunSpec
from services.validation.host_snapshot import snapshot

MODEL = '''from pathlib import Path
import torch
Path(__file__).with_suffix('.executed').touch()
class Model(torch.nn.Module):
    def forward(self, x): return torch.sin(x) + 1
'''
CONFIG = '''version = 1
[profiles.small]
kind = "model"
model = "model:Model"
mode = "eval"
device = "cpu"
grad_enabled = false
seed = 7
init = {args = [], kwargs = {}}
inputs = {args = [{kind="tensor", shape=[2,4], dtype="float32", requires_grad=false, generator={kind="full", value=1}}], kwargs = {}}
[profiles.small.limits]
timeout_ms = 1000
max_events = 40
max_tensors = 50
max_record_bytes = 100000
'''


@pytest.fixture
def plan_request(tmp_path):
    (tmp_path / 'model.py').write_text(MODEL)
    return dict(root=str(tmp_path), executable=sys.executable, environmentRevision=0,
                session='session', workspaceId='workspace', runId='plan_request', generation=1,
                modelFile='model.py', className='Model', timeoutMs=30000,
                execution={'inputs': {'args': [{'shape': [2, 4], 'dtype': 'float32', 'fill': 'ones'}]}})


def parsed(plan_request):
    return ResolvedExecutionPlan.model_validate(snapshot(plan_request)['plan'])


def test_explicit_plan_is_static_saved_and_normalized(plan_request):
    plan = parsed(plan_request)
    assert plan.configuration == 'absent' and plan.origin == 'explicit'
    assert plan.execution['gradEnabled'] is False and plan.execution['seed'] == 0
    assert plan.source.files[0].path == 'model.py'
    assert plan.source.dependencyScope == 'partial'
    assert not (Path(plan_request['root']) / 'model.executed').exists()
    verify_saved_plan(plan, plan_request['root'])


def test_profile_preserves_exact_bindings_limits_and_identity(plan_request):
    root = Path(plan_request['root'])
    (root / 'tensorviz.toml').write_text(CONFIG)
    preflight = {k: v for k, v in plan_request.items() if k != 'execution'} | {'profile': 'small'}
    spec = RunSpec.model_validate(snapshot(preflight)['resolution']['spec'])
    plan_request.update(profile='small', expectedIdentity=spec.identity.model_dump(), execution=profile_execution(spec.profile))
    plan = parsed(plan_request)
    assert plan.configuration == 'selected' and plan.profileDigest == spec.identity.profileDigest
    assert plan.timeoutMs == 1000 and plan.captureLimits.model_dump() == dict(operations=40, invocations=40, values=50, bytes=100000)
    assert plan.execution['inputs']['binding'] == spec.profile.inputs.model_dump()
    plan_request['className'] = 'Other'
    with pytest.raises(ValueError, match='target differs'):
        parsed(plan_request)
    plan_request['className'] = 'Model'
    plan_request['execution']['seed'] = 42
    with pytest.raises(ValueError, match='differs'):
        parsed(plan_request)
    plan_request['execution'] = profile_execution(spec.profile)
    (root / 'tensorviz.toml').write_text(CONFIG + '# changed\n')
    with pytest.raises(ValueError, match='identity changed'):
        parsed(plan_request)


def test_large_graph_budget_is_explicit_and_does_not_enlarge_validation_records(plan_request):
    root = Path(plan_request['root'])
    (root / 'tensorviz.toml').write_text(CONFIG + 'max_capture_bytes = 16777216\n')
    preflight = {k: v for k, v in plan_request.items() if k != 'execution'} | {'profile': 'small'}
    spec = RunSpec.model_validate(snapshot(preflight)['resolution']['spec'])
    plan_request.update(profile='small', expectedIdentity=spec.identity.model_dump(), execution=profile_execution(spec.profile))
    plan = parsed(plan_request)
    assert plan.captureLimits.bytes == 16777216
    assert spec.profile.limits.max_record_bytes == 100000
    from services.validation.contracts import Limits
    with pytest.raises(ValueError):
        Limits(max_capture_bytes=16777217)
    with pytest.raises(ValueError):
        Limits(max_record_bytes=16777216)


@pytest.mark.parametrize('change', ['config', 'source', 'new_helper', 'recipe', 'mode', 'device', 'grad', 'seed', 'cwd', 'timeout', 'environment'])
def test_applicability_tracks_execution_relevant_changes(plan_request, change):
    before = parsed(plan_request)
    root = Path(plan_request['root'])
    if change == 'config': (root / 'tensorviz.toml').write_text('# intentionally overridden')
    if change == 'source': (root / 'model.py').write_text(MODEL + '# changed\n')
    if change == 'new_helper': (root / 'helper.py').write_text('WIDTH = 4\n')
    if change == 'recipe': plan_request['execution']['inputs']['args'][0]['shape'] = [3, 4]
    if change == 'mode': plan_request['execution']['mode'] = 'train'
    if change == 'device': plan_request['execution']['device'] = 'cuda:0'
    if change == 'grad': plan_request['execution']['gradEnabled'] = True
    if change == 'seed': plan_request['execution']['seed'] = None
    if change == 'cwd':
        (root / 'sub').mkdir()
        plan_request['execution']['cwd'] = 'sub'
    if change == 'timeout': plan_request['timeoutMs'] = 20000
    if change == 'environment': plan_request['environmentRevision'] = 1
    after = parsed(plan_request)
    assert applicability_digest(before) != applicability_digest(after)


@pytest.mark.parametrize('job', [
    {'kind': 'profile', 'warmup': 1, 'iterations': 2},
    {'kind': 'training', 'loss': {'file': 'loss.py', 'callable': 'loss'}},
])
def test_profiling_and_training_jobs_are_rejected(plan_request, job):
    plan_request['job'] = job
    with pytest.raises(ValueError, match='not supported'):
        parsed(plan_request)


def test_occurrence_identity_is_not_the_applicability_key(plan_request):
    a = parsed(plan_request)
    plan_request.update(runId='other', generation=2, documents={'model.py': {'version': 10, 'dirty': False}})
    b = parsed(plan_request)
    assert a.id != b.id and a.source.id != b.source.id
    assert applicability_digest(a) == applicability_digest(b)


@pytest.mark.parametrize('change', ['dirty', 'new_dirty_config', 'deleted', 'escape', 'cwd_escape'])
def test_unknown_or_unsaved_source_never_becomes_a_resolved_plan(plan_request, change, tmp_path):
    root = Path(plan_request['root'])
    if change == 'dirty': plan_request['documents'] = {'model.py': {'version': 2, 'dirty': True}}
    if change == 'new_dirty_config': plan_request['documents'] = {'tensorviz.toml': {'version': 1, 'dirty': True}}
    if change == 'deleted': (root / 'model.py').unlink()
    if change == 'escape': plan_request['modelFile'] = '../outside.py'
    if change == 'cwd_escape': plan_request['execution']['cwd'] = '../outside'
    with pytest.raises(ValueError): parsed(plan_request)


def test_declared_native_and_factory_files_have_required_roles(plan_request):
    root = Path(plan_request['root'])
    (root / 'recipe.py').write_text('raise RuntimeError("do not import")\n')
    (root / 'state.pt').write_bytes(b'not loaded during preflight')
    plan_request['execution'].update(setup={'file': 'recipe.py', 'callable': 'build'}, checkpoint={'file': 'state.pt'})
    plan = parsed(plan_request)
    roles = {file.path: file.role for file in plan.source.files}
    assert roles == {'model.py': 'source', 'recipe.py': 'source', 'state.pt': 'native'}
    broken = plan.model_dump()
    broken['source']['files'] = [f for f in broken['source']['files'] if f['path'] != 'state.pt']
    with pytest.raises(ValueError, match='matching saved identities'):
        ResolvedExecutionPlan.model_validate(broken)


@pytest.mark.parametrize('change', ['source', 'config', 'environment'])
def test_worker_rechecks_saved_plan_before_importing(plan_request, change):
    plan = parsed(plan_request)
    if change == 'source': (Path(plan_request['root']) / 'model.py').write_text(MODEL + '# changed\n')
    if change == 'config': (Path(plan_request['root']) / 'tensorviz.toml').write_text('# new')
    if change == 'environment':
        # An environment's package fingerprint cannot be relabelled as current.
        plan.environment.probe_digest = '0' * 64
    with pytest.raises(ValueError): verify_saved_plan(plan, plan_request['root'])
    assert not (Path(plan_request['root']) / 'model.executed').exists()


def test_private_plan_transport_rejects_permissions_and_forged_digest(plan_request, monkeypatch, tmp_path):
    plan = parsed(plan_request)
    file = tmp_path / 'plan.json'
    body = {'version': 1, 'plan': plan.model_dump(), 'applicabilityDigest': applicability_digest(plan)}
    file.write_text(json.dumps(body))
    file.chmod(0o600)
    monkeypatch.setenv('TENSORVIZ_EXECUTION_PLAN', str(file))
    assert consume_plan_file()[0] == plan
    assert 'TENSORVIZ_EXECUTION_PLAN' not in os.environ
    body['applicabilityDigest'] = '0' * 64
    file.write_text(json.dumps(body))
    monkeypatch.setenv('TENSORVIZ_EXECUTION_PLAN', str(file))
    with pytest.raises(ValueError, match='fingerprint'): consume_plan_file()
    if os.name != 'nt':
        file.chmod(0o644)
        monkeypatch.setenv('TENSORVIZ_EXECUTION_PLAN', str(file))
        with pytest.raises(ValueError, match='private'): consume_plan_file()


@pytest.mark.parametrize('stale', [False, True])
def test_real_worker_uses_plan_and_refuses_stale_source_before_model_import(plan_request, stale, tmp_path):
    resolved = snapshot(plan_request)
    file = tmp_path / 'plan.json'
    file.write_text(json.dumps({'version': 1, 'plan': resolved['plan'], 'applicabilityDigest': resolved['applicabilityDigest']}))
    file.chmod(0o600)
    root = Path(plan_request['root'])
    if stale: (root / 'model.py').write_text(MODEL + '# new revision\n')
    backend = Path(__file__).parent.resolve().parents[1]
    process = subprocess.run([sys.executable, '-B', '-m', 'services.ingest.worker', '--resolved-execution',
        '--workspace', str(root), '--file', str(root / 'model.py'), '--cls', 'Model', '--json'],
        cwd=backend, env={**os.environ, 'TENSORVIZ_EXECUTION_PLAN': str(file)}, capture_output=True, text=True, timeout=30)
    result = json.loads(process.stdout)
    assert process.returncode == (2 if stale else 0), process.stderr
    assert (root / 'model.executed').exists() != stale
    if not stale:
        execution = result['result']['diagnostics']['execution']
        assert execution['resolvedPlan'] == resolved['plan']
        assert execution['planDigest'] == resolved['applicabilityDigest']
        assert execution['planCurrent'] is True


@pytest.mark.parametrize('cwd', ['.', 'src'])
def test_saved_profile_namespace_imports_follow_its_working_directory(plan_request, cwd, tmp_path):
    package = tmp_path / 'src' / 'namespace'
    package.mkdir(parents=True)
    (package / 'model.py').write_text('from .constant import OFFSET\n' + MODEL.replace(' + 1', ' + OFFSET'))
    (package / 'constant.py').write_text('OFFSET = 1\n')
    module = 'src.namespace.model' if cwd == '.' else 'namespace.model'
    (tmp_path / 'tensorviz.toml').write_text(CONFIG.replace('model:Model', module + ':Model').replace(
        'kind = "model"', 'kind = "model"\ncwd = ' + json.dumps(cwd)))
    request = {k: v for k, v in plan_request.items() if k != 'execution'}
    request.update(modelFile='src/namespace/model.py', profile='small')
    spec = RunSpec.model_validate(snapshot(request)['resolution']['spec'])
    request.update(expectedIdentity=spec.identity.model_dump(), execution=profile_execution(spec.profile))
    resolved = snapshot(request)
    plan_file = tmp_path / 'plan.json'
    plan_file.write_text(json.dumps(dict(version=1, plan=resolved['plan'], applicabilityDigest=resolved['applicabilityDigest'])))
    plan_file.chmod(0o600)
    process = subprocess.run([sys.executable, '-B', '-m', 'services.ingest.worker', '--resolved-execution',
        '--workspace', str(tmp_path), '--file', str(package / 'model.py'), '--cls', 'Model', '--json'],
        cwd=Path(__file__).parent.resolve().parents[1], env={**os.environ, 'TENSORVIZ_EXECUTION_PLAN': str(plan_file)},
        capture_output=True, text=True, timeout=30)
    envelope = json.loads(process.stdout)
    assert process.returncode == 0, envelope.get('error')
    assert envelope['result']['diagnostics']['execution']['planCurrent'] is True
    assert envelope['result']['diagnostics']['semantic_capture']['status'] == 'complete'
    assert (package / 'model.executed').exists()
    assert not (tmp_path / 'model.executed').exists(), 'the root lookalike must not run'
