#!/usr/bin/env python3
"""Run focused source tests after local npm dependencies and compilation exist.

This does not install dependencies or use a logged-in desktop. The repository tests
use disposable fixtures; installer tests use local synthetic wheels and no index.
Raw test output stays under evidence/private. Only small receipts are public.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--no-torch-python', required=True)
    parser.add_argument('--capture-python', required=True)
    parser.add_argument('--reproduce-original-no-torch-suite', action='store_true',
                        help='Repeat the initial harness mistake for the mixed installer/capture suite.')
    args = parser.parse_args()
    source = args.source.absolute()
    evidence = args.evidence.absolute()
    private, public = evidence / 'private', evidence / 'public'
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    public.mkdir(exist_ok=True)
    for name in ('tmp', 'home'):
        (private / name).mkdir(exist_ok=True)
    env = {
        'PATH': os.environ['PATH'], 'LANG': 'en_US.UTF-8',
        'HOME': str(private / 'home'), 'TMPDIR': str(private / 'tmp'),
        'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
        'PIP_NO_INDEX': '1', 'PIP_DISABLE_PIP_VERSION_CHECK': '1',
    }
    pin = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    metadata_script = ('import json,platform,sys,importlib.metadata as m,importlib.util as u; '
                       'names=["pytest","torch","fastapi","websockets","pydantic"]; '
                       'installed={x.metadata["Name"].lower():x.version for x in m.distributions()}; '
                       'print(json.dumps(dict(python=platform.python_version(),arch=platform.machine(),'
                       'packages={n:installed.get(n) for n in names},'
                       'torchImportable=u.find_spec("torch") is not None)))')
    runtimes = {}
    for label, python in [('noTorch', args.no_torch_python), ('capture', args.capture_python)]:
        # Retain the venv executable spelling: resolving its symlink loses venv context.
        runtimes[label] = json.loads(subprocess.check_output([python, '-B', '-c', metadata_script], env=env, text=True))
    jobs = [
        ('extension-runtime-preflight-lifecycle', source / 'extension', [
            'node', 'node_modules/mocha/bin/mocha.js', '--ui', 'tdd', '--require',
            './out/test/bootstrap.js', 'out/test/unit/runtime.test.js',
            'out/test/unit/runtimeLifecycle.test.js', 'out/test/unit/preflight.test.js']),
        ('backend-service-and-static', source / 'backend', [
            args.no_torch_python, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
            'tests/runtime/test_service_provisioning.py', 'tests/runtime/test_service_without_torch.py']),
        ('backend-additive-environment', source / 'backend', [
            args.no_torch_python if args.reproduce_original_no_torch_suite else args.capture_python,
            '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
            'tests/runtime/test_environment_setup.py']),
        ('backend-capture-separation', source / 'backend', [
            args.capture_python, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
            'tests/runtime/test_capture_environment.py']),
    ]
    receipts = []
    for label, cwd, command in jobs:
        started = time.monotonic()
        output = private / (label + '.log')
        with output.open('w') as log:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900)
        raw = output.read_text()
        counts = re.findall(r'\b\d+ (?:passing|pending|failing|passed|failed|skipped|error|errors)\b', raw)
        safe_command = [('NO_TORCH_PYTHON' if part == args.no_torch_python else
                         'CAPTURE_PYTHON' if part == args.capture_python else part) for part in command]
        receipt = dict(label=label, command=safe_command, cwd=str(cwd.relative_to(source)),
                       exitCode=result.returncode, counts=counts[-5:],
                       durationSeconds=round(time.monotonic()-started, 2),
                       rawLogSha256=hashlib.sha256(output.read_bytes()).hexdigest())
        receipts.append(receipt)
        report = dict(sourceCommit=pin, recordedAt=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      runtimes=runtimes, checks=receipts, scope='Focused local source tests; no desktop UI or release gate claim.')
        (public / 'source-test-receipts.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(receipt), flush=True)
    return int(any(item['exitCode'] for item in receipts))


if __name__ == '__main__':
    raise SystemExit(main())
