"""Execution-cache untracked-read detector (services/ingest/reuse_audit.py).

A resolved-plan run reports ``diagnostics.reuse`` = {storable, reasons, untrackedReads}.
Anything the cache key cannot see — a data file, a module outside the workspace and the
interpreter, a write, a subprocess — must make the run not storable. A plain model whose
imports are all workspace sources and installed packages must stay storable.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from services.semantic.execution_plan import fingerprint  # noqa: F401  (backend importable)
from services.validation.host_snapshot import snapshot

BACKEND = Path(__file__).parent.resolve().parents[1]

PLAIN = '''import torch
from helpers import width
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, width())
    def forward(self, x): return torch.relu(self.fc(x))
'''
HELPERS = 'def width():\n    return 3\n'


def _workspace(root: Path, model: str, files=None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / 'model.py').write_text(model)
    (root / 'helpers.py').write_text(HELPERS)
    for name, content in (files or {}).items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(content)
    return root


def _plan(root: Path, tmp: Path, seed=0):
    request = dict(root=str(root), executable=sys.executable, environmentRevision=0, session='session', workspaceId='w',
                   runId='reuse', generation=1, modelFile='model.py', className='Model', timeoutMs=60000,
                   execution={'seed': seed, 'inputs': {'args': [{'shape': [2, 4], 'dtype': 'float32', 'fill': 'ones'}]}})
    resolved = snapshot(request)
    file = tmp / f'plan-{os.getpid()}-{len(list(tmp.glob("plan-*")))}.json'
    file.write_text(json.dumps({'version': 1, 'plan': resolved['plan'], 'applicabilityDigest': resolved['applicabilityDigest']}))
    file.chmod(0o600)
    return file


def _run(root: Path, tmp: Path, env=None, audit=True):
    plan = _plan(root, tmp)
    process = subprocess.run([sys.executable, '-B', str(BACKEND / 'services/ingest/bootstrap.py'), '--resolved-execution',
                              '--workspace', str(root), '--file', str(root / 'model.py'), '--cls', 'Model', '--json',
                              *(['--reuse-audit'] if audit else [])],
                             cwd=BACKEND, env={**os.environ, 'KMP_DUPLICATE_LIB_OK': 'TRUE', **(env or {}),
                                               'TENSORVIZ_EXECUTION_PLAN': str(plan)},
                             capture_output=True, text=True, timeout=120)
    envelope = json.loads(process.stdout)
    assert process.returncode == 0, (envelope.get('error'), process.stderr[-2000:])
    assert envelope['result']['diagnostics']['execution']['planCurrent'] is True
    return envelope['result']['diagnostics']['reuse']


def test_plain_workspace_model_is_storable(tmp_path):
    root = _workspace(tmp_path / 'ws', PLAIN)
    assert _run(root, tmp_path) == {'storable': True, 'reasons': [], 'untrackedReads': 0}


def test_reading_a_data_file_in_the_workspace_is_not_storable(tmp_path):
    model = PLAIN.replace('        super().__init__()\n',
                          '        super().__init__()\n        self.scale = float(open(__file__.replace("model.py", "scale.txt")).read())\n')
    root = _workspace(tmp_path / 'ws', model, {'scale.txt': '2.0\n'})
    assert _run(root, tmp_path) == {'storable': False, 'reasons': ['untracked_workspace_read'], 'untrackedReads': 1}


def test_numpy_load_of_a_data_file_is_not_storable(tmp_path):
    np = pytest.importorskip('numpy')
    root = _workspace(tmp_path / 'ws', PLAIN.replace(
        '        super().__init__()\n',
        '        super().__init__()\n        import numpy\n        self.table = numpy.load(__file__.replace("model.py", "data/table.npy"))\n'))
    (root / 'data').mkdir()
    np.save(root / 'data' / 'table.npy', np.arange(4))
    report = _run(root, tmp_path)
    assert report['storable'] is False and 'untracked_workspace_read' in report['reasons']
    assert report['untrackedReads'] >= 1


def test_reading_a_file_outside_the_workspace_is_not_storable(tmp_path):
    outside = tmp_path / 'cache' / 'config.json'
    outside.parent.mkdir()
    outside.write_text('{"width": 3}')
    model = PLAIN.replace('from helpers import width\n', f'import json\nwidth = lambda: json.load(open({str(outside)!r}))["width"]\n')
    root = _workspace(tmp_path / 'ws', model)
    assert _run(root, tmp_path) == {'storable': False, 'reasons': ['untracked_external_read'], 'untrackedReads': 1}


def test_writing_a_file_is_not_storable(tmp_path):
    model = 'from pathlib import Path\nPath(__file__).with_suffix(".executed").touch()\n' + PLAIN
    root = _workspace(tmp_path / 'ws', model)
    report = _run(root, tmp_path)
    assert report['storable'] is False and report['reasons'] == ['write']


def test_writing_outside_the_workspace_is_not_storable(tmp_path):
    target = tmp_path / 'elsewhere.log'
    model = PLAIN.replace('    def forward(self, x):', f'    def forward(self, x):\n        open({str(target)!r}, "a").write("x")\n        return torch.relu(self.fc(x))\n    def unused(self, x):')
    root = _workspace(tmp_path / 'ws', model)
    report = _run(root, tmp_path)
    assert report['storable'] is False and report['reasons'] == ['write']


def test_importing_a_module_outside_workspace_and_site_packages_is_not_storable(tmp_path):
    lib = tmp_path / 'lib'
    lib.mkdir()
    (lib / 'outside_helpers.py').write_text(HELPERS)
    model = PLAIN.replace('from helpers import width', 'from outside_helpers import width')
    root = _workspace(tmp_path / 'ws', model)
    report = _run(root, tmp_path, env={'PYTHONPATH': str(lib)})
    # The module and the import system's probe for its cached bytecode.
    assert report == {'storable': False, 'reasons': ['untracked_external_read'], 'untrackedReads': 2}


def test_editable_install_like_source_path_is_not_storable(tmp_path):
    # An editable install is a path entry (from a .pth) pointing at a source tree outside
    # site-packages: its files change without any version change the plan could see.
    src = tmp_path / 'project' / 'src' / 'mylib'
    src.mkdir(parents=True)
    (src / '__init__.py').write_text('')
    (src / 'layers.py').write_text(HELPERS)
    model = f'import sys\nsys.path.insert(0, {str(src.parent)!r})\n' + PLAIN.replace('from helpers import width', 'from mylib.layers import width')
    root = _workspace(tmp_path / 'ws', model)
    report = _run(root, tmp_path)
    assert report['storable'] is False and report['reasons'] == ['untracked_external_read']
    assert report['untrackedReads'] == 4  # package __init__ and module, each with its bytecode probe


def test_spawning_a_process_is_not_storable(tmp_path):
    model = f'import subprocess\nsubprocess.run([{sys.executable!r}, "-c", "pass"])\n' + PLAIN
    root = _workspace(tmp_path / 'ws', model)
    report = _run(root, tmp_path)
    assert report['storable'] is False and 'subprocess' in report['reasons']


def test_without_the_flag_nothing_is_audited(tmp_path):
    # The host sends --reuse-audit only while the cache is on: with it off, the worker
    # does no extra work and reports no block, even for a run that reads a data file.
    model = PLAIN.replace('        super().__init__()\n',
                          '        super().__init__()\n        self.scale = float(open(__file__.replace("model.py", "scale.txt")).read())\n')
    root = _workspace(tmp_path / 'ws', model, {'scale.txt': '2.0\n'})
    assert _run(root, tmp_path, audit=False) is None


def test_legacy_runs_report_no_reuse_block(tmp_path):
    root = _workspace(tmp_path / 'ws', PLAIN)
    process = subprocess.run([sys.executable, '-B', str(BACKEND / 'services/ingest/bootstrap.py'), '--file', str(root / 'model.py'),
                              '--cls', 'Model', '--input', '2,4', '--json'], cwd=BACKEND,
                             env={**os.environ, 'KMP_DUPLICATE_LIB_OK': 'TRUE', 'PYTHONPATH': str(root)},
                             capture_output=True, text=True, timeout=120)
    envelope = json.loads(process.stdout)
    assert envelope['result']['diagnostics']['reuse'] is None


def _strip(value):
    timing = {'elapsed_ms', 'build_ms', 'trace_ms'}
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k not in timing}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def test_the_detector_changes_nothing_but_the_reuse_block(tmp_path, monkeypatch, capsys):
    """Same plan, in process, with and without --reuse-audit: identical envelopes apart
    from ``diagnostics.reuse`` (and the per-capture uuid and timings)."""
    from services.ingest import worker
    root = _workspace(tmp_path / 'ws', PLAIN)
    argv = ['--resolved-execution', '--workspace', str(root), '--file', str(root / 'model.py'), '--cls', 'Model', '--json']
    monkeypatch.syspath_prepend(str(root))

    def once(enabled):
        monkeypatch.setenv('TENSORVIZ_EXECUTION_PLAN', str(_plan(root, tmp_path)))
        assert worker.main(argv + (['--reuse-audit'] if enabled else [])) == 0
        monkeypatch.undo()
        monkeypatch.syspath_prepend(str(root))
        for name in ('model', 'helpers'):
            sys.modules.pop(name, None)
        return json.loads(capsys.readouterr().out)

    on, off = once(True), once(False)
    assert on['result']['diagnostics'].pop('reuse') == {'storable': True, 'reasons': [], 'untrackedReads': 0}
    assert off['result']['diagnostics'].pop('reuse') is None
    for envelope in (on, off):
        envelope['result']['diagnostics']['semantic_capture'].pop('id')
    assert _strip(on) == _strip(off)


def test_audit_never_raises_into_the_audited_operation(tmp_path):
    from services.ingest.reuse_audit import ReadAudit
    audit = ReadAudit(str(tmp_path), []).start()
    try:
        audit._event = None  # classification itself fails
        (tmp_path / 'x.txt').write_text('still works')
        assert (tmp_path / 'x.txt').read_text() == 'still works'
    finally:
        report = audit.stop()
    assert report['storable'] is False
