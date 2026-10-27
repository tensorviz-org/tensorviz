#!/usr/bin/env python3
"""Run real interpreter gates; provision dependencies separately, never here."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
CAPTURE_TESTS = [
    'backend/tests/runtime/test_capture_environment.py',
    'backend/tests/runtime/test_environment_setup.py',
    'backend/tests/capture/test_capture_preparation.py',
    'backend/tests/validation/test_setup_requirements.py',
    'backend/tests/validation/test_validation_source_recovery.py',
]
SERVICE_TESTS = [
    'backend/tests/api/test_graph_opening_broker.py',
    'backend/tests/api/test_local_browser_protocol.py',
    'backend/tests/api/test_local_browser_validation.py',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kind', choices=['capture', 'service'], default='capture')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {'kind': args.kind, 'python': sys.version, 'executable': sys.executable,
              'os': platform.system(), 'release': platform.release(),
              'platform': platform.platform(), 'architecture': platform.machine(),
              'packages': dict(sorted((dist.metadata['Name'], dist.version)
                                     for dist in importlib.metadata.distributions())),
              'excluded': [], 'steps': []}
    if Path('/etc/os-release').is_file():
        report['osRelease'] = Path('/etc/os-release').read_text()
    report_file = output / 'result.json'

    def run(label, command):
        started = time.monotonic()
        with (output / (label + '.log')).open('w') as log:
            try:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, timeout=600)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
        report['steps'].append({'name': label, 'exitCode': code,
                                'seconds': round(time.monotonic() - started, 2)})
        report_file.write_text(json.dumps(report, indent=2))
        print(label, 'passed' if code == 0 else 'FAILED', flush=True)
        return code == 0

    helper = 'capture_environment.py' if args.kind == 'capture' else 'extension_environment.py'
    passed = run('probe', [sys.executable, '-B', str(ROOT / 'backend' / helper), 'probe'])
    tests = list(CAPTURE_TESTS if args.kind == 'capture' else SERVICE_TESTS)
    selection = []
    if args.kind == 'capture':
        if sys.version_info >= (3, 11):
            tests.append('backend/tests/source_review/test_semantic_source_bindings.py')
        else:
            report['excluded'].append('Expression source-edit bindings require Python 3.11+.')
    junit = output / 'pytest.xml'
    passed = run('pytest', [sys.executable, '-m', 'pytest', *tests, *selection,
                           '-q', '-ra', '--junitxml=' + str(junit)]) and passed
    if junit.is_file():
        suites = ET.parse(junit).getroot()
        report['tests'] = {key: sum(int(suite.get(key, 0)) for suite in suites.iter('testsuite'))
                           for key in ['tests', 'failures', 'errors', 'skipped']}
        if report['tests']['skipped']:
            passed = False  # Unexpected skips must be reviewed, not counted as a pass.
    report['passed'] = passed
    report_file.write_text(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
