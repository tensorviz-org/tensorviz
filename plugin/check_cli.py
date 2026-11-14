#!/usr/bin/env python3
"""Disposable-profile CLI acceptance; never grants TensorViz permission.

Inputs: --evidence TASK_DIR --catalog EXTRACTED_CATALOG [--label published-0.1.44].
The public report is allowlisted; raw output stays under private/. The input
catalog is copied before task-only MCP state/cache environment overrides.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--evidence', required=True)
parser.add_argument('--catalog', required=True)
parser.add_argument('--label', default='published-0.1.44')
args = parser.parse_args()
base = Path(args.evidence).resolve()
private = base / 'private' / args.label
private.mkdir(parents=True, exist_ok=True, mode=0o700)
catalog = private / 'install-catalog'
if catalog.exists():
    raise SystemExit('Refusing to reuse the test installation catalog')
shutil.copytree(Path(args.catalog).resolve(), catalog)
data = private / 'state'
cache = private / 'runtime-cache'
temp = private / 'temp'
temp.mkdir(mode=0o700)
for relative in ['plugins/tensorviz/mcp.json', 'plugins/tensorviz/.mcp.json']:
    manifest = catalog / relative
    doc = json.loads(manifest.read_text())
    server = doc['mcpServers']['tensorviz']
    server['env'] = {**server.get('env', {}), 'TENSORVIZ_DATA_DIR': str(data),
                     'TENSORVIZ_RUNTIME_CACHE_DIR': str(cache)}
    manifest.write_text(json.dumps(doc, indent=2) + '\n')

def snapshot():
    result = {}
    # Metadata only: never read account credential contents.
    for relative in ['.codex/config.toml', '.codex/auth.json', '.claude.json',
                     '.claude/settings.json', '.claude/.credentials.json']:
        p = Path.home() / relative
        if p.exists():
            s = p.stat()
            result[relative] = [s.st_size, s.st_mtime_ns, s.st_ino]
        else:
            result[relative] = None
    return result

before = snapshot()
env = {k: os.environ[k] for k in ['PATH', 'HOME', 'USER', 'LOGNAME', 'LANG'] if k in os.environ}
env.update({'TMPDIR': str(temp), 'NO_COLOR': '1', 'CI': '1',
            'CODEX_HOME': str(private / 'codex-profile'),
            'CLAUDE_CONFIG_DIR': str(private / 'claude-profile'),
            'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1',
            'ANTHROPIC_API_KEY': 'test-no-provider-calls', 'CLAUDE_CODE_OAUTH_TOKEN': '',
            'TENSORVIZ_DATA_DIR': str(data), 'TENSORVIZ_RUNTIME_CACHE_DIR': str(cache)})
for key in ['CODEX_HOME', 'CLAUDE_CONFIG_DIR']:
    Path(env[key]).mkdir(mode=0o700)
report = {'label': args.label, 'catalog_changes': 'MCP state/cache paths only',
          'provider_calls': False, 'permission_granted_by_test': False, 'checks': []}

def run(name, command):
    started = time.monotonic()
    process = subprocess.Popen(command, env=env, cwd=private, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=60)
        status = 'passed' if process.returncode == 0 else 'failed'
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        status = 'timeout'
    (private / (name + '.stdout')).write_text(stdout)
    (private / (name + '.stderr')).write_text(stderr)
    item = {'name': name, 'status': status, 'exit_code': process.returncode,
            'seconds': round(time.monotonic() - started, 3)}
    report['checks'].append(item)
    print(json.dumps(item), flush=True)
    try:
        parsed = json.loads(stdout)
    except ValueError:
        parsed = None
    return process.returncode, parsed

codex = shutil.which('codex')
claude = shutil.which('claude')
run('codex-marketplace-add', [codex, 'plugin', 'marketplace', 'add', str(catalog), '--json'])
code, installed = run('codex-plugin-add', [codex, 'plugin', 'add', 'tensorviz@tensorviz-standalone', '--json'])
if code == 0:
    run('codex-plugin-list', [codex, 'plugin', 'list', '--marketplace', 'tensorviz-standalone', '--json'])
    run('codex-mcp-list', [codex, 'mcp', 'list', '--json'])
run('codex-plugin-remove', [codex, 'plugin', 'remove', 'tensorviz@tensorviz-standalone', '--json'])
run('codex-plugin-list-after-remove', [codex, 'plugin', 'list', '--marketplace', 'tensorviz-standalone', '--json'])
run('claude-marketplace-add', [claude, 'plugin', 'marketplace', 'add', str(catalog)])
run('claude-plugin-install', [claude, 'plugin', 'install', 'tensorviz@tensorviz-standalone', '--scope', 'user'])
run('claude-plugin-list', [claude, 'plugin', 'list', '--json'])
run('claude-plugin-uninstall', [claude, 'plugin', 'uninstall', 'tensorviz@tensorviz-standalone', '--scope', 'user'])
run('claude-plugin-list-after-remove', [claude, 'plugin', 'list', '--json'])
report['normal_account_file_metadata_unchanged'] = before == snapshot()
expected_version = json.loads((catalog / 'plugins/tensorviz/plugin.json').read_text())['version']
try:
    def read_result(name): return json.loads((private / (name + '.stdout')).read_text())
    codex_install = read_result('codex-plugin-add')
    codex_plugins = read_result('codex-plugin-list')['installed']
    codex_servers = read_result('codex-mcp-list')
    claude_plugins = read_result('claude-plugin-list')
    checks = {'codex_installed_version': codex_install['version'],
        'codex_enabled': len(codex_plugins) == 1 and codex_plugins[0]['enabled'],
        'codex_exactly_one_mcp_server': len(codex_servers) == 1 and codex_servers[0]['name'] == 'tensorviz',
        'codex_removed': read_result('codex-plugin-list-after-remove')['installed'] == [],
        'claude_installed_version': claude_plugins[0]['version'],
        'claude_enabled': len(claude_plugins) == 1 and claude_plugins[0]['enabled'],
        'claude_removed': read_result('claude-plugin-list-after-remove') == []}
    checks['expected_versions'] = checks['codex_installed_version'] == checks['claude_installed_version'] == expected_version
    report['verified_installation_results'] = checks
except (ValueError, KeyError, IndexError, TypeError):
    report['verified_installation_results'] = {'status': 'could_not_verify'}
report['limitations'] = ['No signed-in desktop rendering or provider/model turn',
                         'No permission helper, permission records or native approval modified']
(base / 'public' / (args.label + '-cli-report.json')).write_text(json.dumps(report, indent=2) + '\n')
