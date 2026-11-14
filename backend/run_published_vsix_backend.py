#!/usr/bin/env python3
"""Accept a SHA-verified published VSIX's bundled backend in existing isolated envs.

No installation, dependency, client config, account or security changes. Services
bind loopback and use the normal local launch token. Raw logs and synthetic
fixtures remain in output/private; only the allowlisted JSON is publishable.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
import zipfile

p = argparse.ArgumentParser()
p.add_argument('--vsix', type=Path, required=True)
p.add_argument('--sha256', required=True)
p.add_argument('--version', required=True)
p.add_argument('--release-source-sha', required=True)
p.add_argument('--test-source', type=Path, required=True)
p.add_argument('--no-torch-python', type=Path, required=True)
p.add_argument('--broken-torch-python', type=Path, required=True)
p.add_argument('--capture-python', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
assert hashlib.sha256(a.vsix.read_bytes()).hexdigest() == a.sha256, 'VSIX SHA256 mismatch'
private = a.output.absolute() / 'private'
private.mkdir(parents=True, exist_ok=True, mode=0o700)
os.chmod(private, 0o700)
backend = private / 'extracted-backend'
backend.mkdir(exist_ok=True)
with zipfile.ZipFile(a.vsix) as archive:
    package = json.loads(archive.read('extension/package.json'))
    assert package['version'] == a.version
    for info in archive.infolist():
        if not info.filename.startswith('extension/backend/') or info.is_dir():
            continue
        relative = Path(info.filename).relative_to('extension/backend')
        assert '..' not in relative.parts and not relative.is_absolute()
        target = backend / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(info))
home = private / 'home'
tmp = private / 'tmp'
home.mkdir(exist_ok=True)
tmp.mkdir(exist_ok=True)
env = {'PATH': '/usr/bin:/bin', 'HOME': str(home), 'TMPDIR': str(tmp),
       'XDG_CACHE_HOME': str(home / 'cache'), 'PYTHONNOUSERSITE': '1',
       'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
       'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1'}
report = {
    'artifact': 'Published TensorViz VSIX bundled backend', 'version': a.version,
    'vsix_sha256': a.sha256, 'release_source_sha': a.release_source_sha,
    'test_source_sha': subprocess.run(['git', '-C', str(a.test_source), 'rev-parse', 'HEAD'],
        capture_output=True, text=True, check=True).stdout.strip(),
    'tested_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope': 'Actual published artifact backend code; local subprocess and loopback HTTP acceptance. No extension activation or graphical acceptance is implied.',
    'backend_file_sha256': {name: hashlib.sha256((backend / name).read_bytes()).hexdigest()
        for name in ['server_v2.py', 'standalone_server.py', 'capture_environment.py', 'services/ingest/bootstrap.py']},
    'environments': {}, 'checks': [],
}

def invoke(name, python, code, args=(), timeout=120, extra_env=None):
    command = [str(python.absolute()), '-I', '-B', '-c', code, *map(str, args)]
    started = time.monotonic()
    child = subprocess.Popen(command, cwd=private, env={**env, **(extra_env or {})},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = child.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGTERM)
        stdout, stderr = child.communicate(timeout=10)
        raise RuntimeError(f'{name} timed out; isolated child group terminated')
    (private / f'{name}.stdout.log').write_text(stdout)
    (private / f'{name}.stderr.log').write_text(stderr)
    result = {'name': name, 'status': 'passed' if child.returncode == 0 else 'failed',
              'exit_code': child.returncode, 'duration_seconds': round(time.monotonic() - started, 2)}
    if child.returncode == 0:
        result['result'] = json.loads(stdout)
    else:
        # Never export full exception traces, environment, tokens or local paths.
        result['failure_summary'] = 'Child failed; raw details retained privately.'
    return result

version_code = """import importlib.metadata as m, importlib.util, json, platform, sys
print(json.dumps({'python':sys.version.split()[0], 'architecture':platform.machine(),
'macos':platform.mac_ver()[0], 'torch_package':next((d.version for d in m.distributions() if d.metadata['Name'].lower()=='torch'),None),
'torch_importable_spec':importlib.util.find_spec('torch') is not None,
'packages':{n:m.version(n) for n in ['pydantic','numpy','fastapi','uvicorn'] if importlib.util.find_spec(n)}}))
"""
for label, interpreter in [('no_torch', a.no_torch_python), ('synthetic_broken_torch', a.broken_torch_python), ('healthy_torch', a.capture_python)]:
    result = invoke('version_' + label, interpreter, version_code)
    assert result['status'] == 'passed'
    report['environments'][label] = result['result']
assert report['environments']['no_torch']['torch_importable_spec'] is False

service_code = r'''
import importlib.util, json, pathlib, secrets, sys
backend, test_file, service, failure = sys.argv[1:]
spec=importlib.util.spec_from_file_location('artifact_acceptance',test_file)
tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
tests.BACKEND=pathlib.Path(backend)
tests.TOKEN=secrets.token_urlsafe(32)
if service=='extension': tests._run(failure)
else: tests.test_static_service_without_model_framework(failure)
print(json.dumps({'service':service,'framework_failure':failure,'startup':'passed',
'health':'passed','static_projection':'passed','static_discovery':'passed','static_setup':'passed',
'model_executed_by_static_routes':False,
'additional_checks':(['parse','workspace classes','buffer classes','repo scan','templates','structural validation','code generation','eight torch-required requests return HTTP503 pytorch_unavailable reason '+failure,'IDE importer refuses in-process execution'] if service=='extension' else ['unauthorized health request rejected HTTP401'])}))
'''
for service, test_name in [('extension', 'test_server_v2_without_torch.py'), ('standalone', 'test_service_without_torch.py')]:
    test_file = a.test_source.absolute() / 'backend/tests/runtime' / test_name
    for failure in ['missing', 'broken']:
        result = invoke(f'{service}_{failure}', a.no_torch_python, service_code,
            [backend, test_file, service, failure], extra_env={'TENSORVIZ_SERVICE_TEST_BACKEND': str(backend)})
        result['command'] = '<NO_TORCH_PYTHON> run_published_vsix_backend.py (published backend; pinned source test harness; ' + service + ' ' + failure + ')'
        result['simulation'] = 'Missing/broken framework imports prohibited by test finder; genuinely no Torch installed in interpreter.'
        report['checks'].append(result)

probe_code = r'''
import json, subprocess, sys
backend=sys.argv[1]
result=subprocess.run([sys.executable,'-I','-B',backend+'/capture_environment.py','probe-capture'],capture_output=True,text=True,timeout=60)
print(json.dumps({'helper_exit_code':result.returncode,'response':json.loads(result.stdout)}))
'''
for label, interpreter, expected_code in [('missing',a.no_torch_python,'MISSING_DEPENDENCIES'),('broken',a.broken_torch_python,'CAPTURE_RUNTIME_UNSUPPORTED'),('healthy',a.capture_python,None)]:
    result=invoke('capture_probe_'+label,interpreter,probe_code,[backend])
    if result['status']=='passed':
        response=result['result']['response']
        if label=='healthy': assert response['ok'] is True
        else: assert response['ok'] is False and response['code']==expected_code
    result['command']='<'+label.upper()+'_TORCH_PYTHON> -I -B <PUBLISHED_BACKEND>/capture_environment.py probe-capture'
    if label=='broken': result['fixture']='Separate venv with capture dependencies and synthetic torch2.14.1 metadata whose import raises an incompatible-architecture OSError; not real PyTorch.'
    report['checks'].append(result)

capture_code=r'''
import hashlib, json, os, pathlib, subprocess, sys
backend=pathlib.Path(sys.argv[1]);workspace=pathlib.Path(sys.argv[2]);workspace.mkdir(exist_ok=True)
sys.path.insert(0,str(backend))
from services.semantic.execution_plan import profile_execution
from services.semantic.projection import project_capture
from services.validation.contracts import ModelProfile
source="""from torch import nn
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(8, 8)
    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(x)))
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(4, 8)
        self.block = Block()
        self.head = nn.Linear(8, 3)
    def forward(self, x):
        return self.head(self.block(self.stem(x)))
"""
(workspace/'model.py').write_text(source)
profile=ModelProfile.model_validate({'kind':'model','model':'model:Net','init':{'args':[],'kwargs':{}},'mode':'eval','device':'cpu','grad_enabled':False,'seed':0,'inputs':{'args':[],'kwargs':{'x':{'kind':'tensor','shape':[2,4],'dtype':'float32','requires_grad':False,'generator':{'kind':'full','value':1}}}}})
request={'root':str(workspace),'executable':sys.executable,'environmentRevision':0,'session':'published-acceptance','workspaceId':'published-acceptance','runId':'published-acceptance','generation':1,'modelFile':'model.py','className':'Net','timeoutMs':30000,'execution':profile_execution(profile)}
preflight=subprocess.run([sys.executable,'-I','-B',str(backend/'services/validation/host_snapshot.py')],input=json.dumps(request),text=True,capture_output=True,timeout=30)
assert preflight.returncode==0,preflight.stderr
prepared=json.loads(preflight.stdout)
plan=workspace/'execution-plan.json';plan.write_text(json.dumps({'version':1,'plan':prepared['plan'],'applicabilityDigest':prepared['applicabilityDigest']}));plan.chmod(0o600)
result=subprocess.run([sys.executable,'-I','-B',str(backend/'services/ingest/bootstrap.py'),'--file',str(workspace/'model.py'),'--cls','Net','--workspace',str(workspace),'--resolved-execution','--json'],cwd=backend,env={**os.environ,'TENSORVIZ_EXECUTION_PLAN':str(plan)},capture_output=True,text=True,timeout=60)
(workspace/'capture.raw.json').write_text(result.stdout);(workspace/'capture.stderr.log').write_text(result.stderr)
assert result.returncode==0,result.stderr
capture=json.loads(result.stdout)['result']['diagnostics']['semantic_capture'];projection=project_capture(capture)
assert capture['status']=='complete' and projection.nodes and projection.edges
invocation=next(x for x in capture['invocations'] if x['id']==capture['rootInvocationId'])
assert invocation['status']=='completed'
output=next(x for x in capture['values'] if x['id']==capture['output']['id'])
assert output['shape']==[2,3]
print(json.dumps({'model':'Synthetic Linear(4,8) -> residual Block(8) -> Linear(8,3)',
'model_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
'capture_status':capture['status'],'input_shape':[2,4],'output_shape':output['shape'],
'parameters':211,'graph_nodes':len(projection.nodes),'graph_edges':len(projection.edges),
'root_invocation_status':invocation['status'],
'graph_inspection':'Programmatic semantic projection and output-shape assertions; no UI inspection.'}))
'''
result=invoke('healthy_real_model_capture',a.capture_python,capture_code,[backend,private/'synthetic-model'])
result['command']='<CAPTURE_PYTHON> -I -B <PUBLISHED_BACKEND>/services/validation/host_snapshot.py; <CAPTURE_PYTHON> -I -B <PUBLISHED_BACKEND>/services/ingest/bootstrap.py --file <SYNTHETIC_WORKSPACE>/model.py --cls Net --workspace <SYNTHETIC_WORKSPACE> --resolved-execution --json'
report['checks'].append(result)
report['status']='passed' if all(x['status']=='passed' for x in report['checks']) else 'failed'
report['cleanup']='All owned short-lived service and capture children exited. Synthetic model and raw logs retained only under output/private.'
path=a.output.absolute()/('published-'+a.version+'-backend-results.json')
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
