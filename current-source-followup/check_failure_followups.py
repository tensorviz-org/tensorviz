#!/usr/bin/env python3
"""Reproduce interpreter/path limitations without editing the repository tests."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--capture-python', required=True)
    args = parser.parse_args()
    source, evidence = args.source.absolute(), args.evidence.absolute()
    private, public = evidence / 'private', evidence / 'public'
    env = dict(PATH=os.environ['PATH'], HOME=str(private/'home'), TMPDIR=str(private/'tmp'),
               LANG='en_US.UTF-8', PYTHONPATH=str(source/'backend'), PYTHONNOUSERSITE='1',
               PYTHONDONTWRITEBYTECODE='1', PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1')
    test_file = source/'backend/tests/runtime/test_environment_setup.py'
    trace_test = 'test_model_import_diagnostic_bounds_traceback_frames_without_source_or_locals'
    capture_test = 'test_private_capture_worker_executes_without_importing_server_dependencies'
    rows = []
    # A short directory is needed to distinguish the independent 12,000-character
    # cap from the 64-frame cap. This verified copy is removed after the checks.
    with tempfile.TemporaryDirectory(prefix='tv-test-', dir='/tmp') as temp:
        copied = Path(temp)/test_file.name
        shutil.copyfile(test_file, copied)
        source_hash = hashlib.sha256(test_file.read_bytes()).hexdigest()
        assert hashlib.sha256(copied.read_bytes()).hexdigest() == source_hash
        jobs = [
            ('capture-correct-interpreter', test_file, capture_test),
            ('traceback-original-path', test_file, trace_test),
            ('traceback-short-path-identical-file', copied, trace_test),
        ]
        for label, path, test in jobs:
            command = [args.capture_python, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', str(path)+'::'+test]
            start = time.monotonic()
            log_path = private/(label+'.log')
            with log_path.open('w') as log:
                result = subprocess.run(command, cwd=source/'backend', env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
            raw = log_path.read_text()
            rows.append(dict(label=label, test=test, exitCode=result.returncode,
                             counts=re.findall(r'\b\d+ (?:passed|failed|skipped|error|errors)\b', raw)[-3:],
                             durationSeconds=round(time.monotonic()-start, 2),
                             rawLogSha256=hashlib.sha256(log_path.read_bytes()).hexdigest()))
    report = dict(sourceCommit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
                  recordedAt=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  testFileSha256=source_hash, checks=rows,
                  interpretation=['The full environment suite needs Torch for its real capture test.',
                                  'The source traceback test expects exactly 64 frames, while the production function also caps text at 12000 characters.',
                                  'The unmodified test fails in the deep checkout and passes from an identical file at a short temporary path.',
                                  'This follow-up does not erase or relabel the original failed full-suite run.'])
    (public/'failure-followups.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
