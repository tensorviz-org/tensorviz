#!/usr/bin/env python3
"""MCP stdio acceptance without granting plugin permission or running providers."""
import argparse
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time

p = argparse.ArgumentParser()
p.add_argument('--evidence', required=True)
p.add_argument('--runtime', required=True)
p.add_argument('--label', required=True)
args = p.parse_args()
base = Path(args.evidence).resolve()
root = base / 'private' / (args.label + '-native')
root.mkdir(parents=True, mode=0o700, exist_ok=False)
project = root / 'synthetic-project'
project.mkdir()
(project / 'model.py').write_text('from torch import nn\nclass Tiny(nn.Module):\n def __init__(self):\n  super().__init__()\n  self.fc = nn.Linear(4, 3)\n def forward(self, x):\n  return self.fc(x)\n')
(project / 'tensorviz.toml').write_text('version=1\n[profiles.tiny]\nkind="model"\nmodel="model:Tiny"\nmode="eval"\ndevice="cpu"\ngrad_enabled=false\ninit={args=[],kwargs={}}\ninputs={args=[{kind="tensor",shape=[2,4],dtype="float32",requires_grad=false,generator={kind="full",value=1.0}}],kwargs={}}\n')
env = {'PATH': '/usr/bin:/bin', 'TENSORVIZ_DATA_DIR': str(root / 'state'),
       'TENSORVIZ_RUNTIME_CACHE_DIR': str(root / 'runtime-cache'),
       'TMPDIR': str(root), 'TORCHINDUCTOR_CACHE_DIR': str(root / 'inductor')}
report = {'label': args.label, 'classification': 'native stdio; no signed-in client or GUI',
          'path': 'system utilities only; no project Python or Torch supplied', 'checks': [],
          'permission_granted_by_test': False, 'provider_calls': False}
raw = open(root / 'protocol.jsonl', 'w')
err = open(root / 'stderr.txt', 'w')
proc = subprocess.Popen([str(Path(args.runtime).resolve() / 'tensorviz-host')],
                        cwd=project, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=err, text=True, bufsize=1, start_new_session=True)
messages = queue.Queue()
def reader():
    for line in proc.stdout:
        raw.write(line); raw.flush()
        try:
            messages.put(json.loads(line))
        except ValueError:
            messages.put({'unparseable': True})
threading.Thread(target=reader, daemon=True).start()
next_id = 0

def request(method, params, timeout=30):
    global next_id
    next_id += 1
    proc.stdin.write(json.dumps({'jsonrpc': '2.0', 'id': next_id, 'method': method, 'params': params}) + '\n')
    proc.stdin.flush()
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            m = messages.get(timeout=max(0.01, end-time.monotonic()))
        except queue.Empty:
            return {'harness_status': 'timeout'}
        if m.get('method') == 'elicitation/create':
            return {'harness_status': 'blocked_permission', 'method': m['method']}
        if m.get('id') == next_id:
            return m
    return {'harness_status': 'timeout'}

try:
    initialized = request('initialize', {'protocolVersion': '2024-11-05',
        'capabilities': {'elicitation': {'form': {}}},
        'clientInfo': {'name': 'tensorviz-mac-isolated-acceptance', 'version': '1'}})
    version = initialized.get('result', {}).get('serverInfo', {})
    report['server'] = {k: version[k] for k in ['name', 'version'] if k in version}
    report['checks'].append({'name': 'initialize', 'status': 'passed' if version else 'failed'})
    proc.stdin.write(json.dumps({'jsonrpc': '2.0', 'method': 'notifications/initialized'}) + '\n')
    proc.stdin.flush()
    tools = request('tools/list', {})
    names = sorted(t['name'] for t in tools.get('result', {}).get('tools', []))
    report['tools'] = names
    report['checks'].append({'name': 'tools-list', 'status': 'passed' if names else 'failed', 'count': len(names)})
    for name in ['list_models', 'get_model_context', 'find_computation']:
        if name not in names:
            report['checks'].append({'name': name, 'status': 'not_advertised'})
            continue
        tool_args = {'list_models': {},
                     'get_model_context': {'modelId': 'not-opened', 'revision': '0' * 64},
                     'find_computation': {'modelId': 'not-opened', 'captureId': 'none', 'sourceId': 'none', 'text': 'fc'}}[name]
        params = {'name': name, 'arguments': tool_args}
        result = request('tools/call', params)
        value = result.get('result', {})
        structured = value.get('structuredContent') or {}
        error = structured.get('error') or {}
        report['checks'].append({'name': name, 'status': 'responded' if 'result' in result else result.get('harness_status', 'failed'),
            'isError': value.get('isError', False), 'errorCode': error.get('code'),
            'modelCount': len(structured['models']) if isinstance(structured.get('models'), list) else None})
    opening = request('tools/call', {'name': 'open_graph', 'arguments': {
        'workspaceRoot': str(project), 'file': 'model.py', 'cls': 'Tiny', 'waitMs': 0}}, timeout=30)
    structured = opening.get('result', {}).get('structuredContent') or {}
    for _ in range(10):
        if opening.get('harness_status') or structured.get('status') not in ['opening', 'pending', 'running']:
            break
        rid = structured.get('requestId')
        if not rid:
            break
        opening = request('tools/call', {'name': 'open_graph', 'arguments': {'requestId': rid, 'waitMs': 1000}}, timeout=10)
        structured = opening.get('result', {}).get('structuredContent') or {}
    report['checks'].append({'name': 'open-graph-without-granting-permission',
        'status': opening.get('harness_status', structured.get('status', 'responded')),
        'nextAction': (structured.get('nextAction') or {}).get('kind'),
        'errorCode': (structured.get('error') or {}).get('code')})
    report['capture'] = 'not_run: no permission was granted by this harness'
finally:
    try:
        proc.stdin.close()
        proc.wait(timeout=5)
    except (BrokenPipeError, subprocess.TimeoutExpired):
        try: os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        try: proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL); proc.wait()
    raw.close(); err.close()
    report['permission_record_created'] = (root / 'state' / 'permissions' / 'standalone.json').exists()
    out = base / 'public' / (args.label + '-native-report.json')
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
