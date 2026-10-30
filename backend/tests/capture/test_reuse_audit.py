"""Execution-cache untracked-read detector (services/ingest/reuse_audit.py).

A resolved-plan run reports ``diagnostics.reuse`` = {storable, reasons, untrackedReads}.
Anything the cache key cannot see — a data file, a module outside the workspace and the
interpreter, a write, a subprocess — must make the run not storable. A plain model whose
imports are all workspace sources and installed packages must stay storable.
"""
import importlib.machinery
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from services.semantic.execution_plan import fingerprint  # noqa: F401  (backend importable)
from services.validation.host_snapshot import snapshot

BACKEND = Path(__file__).parent.resolve().parents[1]
EXTENSION = importlib.machinery.EXTENSION_SUFFIXES[0]

PLAIN = '''import torch
from helpers import width
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, width())
    def forward(self, x): return torch.relu(self.fc(x))
'''
HELPERS = 'def width():\n    return 3\n'


@pytest.fixture(autouse=True)
def _only_modules_a_worker_could_have_loaded(monkeypatch):
    """An in-process audit also classifies every module this pytest process loaded before
    it, and refuses a run whose interpreter loaded code from outside its own trees. Earlier
    test files leave such modules behind (their workspaces' models, and the module
    ``torch.distributed.nn`` generates in a temporary folder); a worker has none. Hidden
    for the duration of each test."""
    from services.ingest.reuse_audit import _covered_prefixes, _under
    roots = (*_covered_prefixes(), os.path.realpath(BACKEND))
    for name, module in list(sys.modules.items()):
        try:
            namespace = object.__getattribute__(module, '__dict__')  # as the audit reads it
        except AttributeError:
            continue
        origin = getattr(namespace.get('__spec__'), 'origin', None) or namespace.get('__file__')
        if isinstance(origin, str) and os.path.isabs(origin) and not any(_under(os.path.realpath(origin), root) for root in roots):
            monkeypatch.delitem(sys.modules, name)


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


def _run(root: Path, tmp: Path, env=None, audit=True, full=False):
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
    report = envelope['result']['diagnostics']['reuse']
    # The installed-environment fingerprints are checked by the tests that are about them.
    return report if full or report is None else {k: v for k, v in report.items() if k != 'environment'}


def test_plain_workspace_model_is_storable(tmp_path):
    root = _workspace(tmp_path / 'ws', PLAIN)
    assert _run(root, tmp_path) == {'storable': True, 'reasons': [], 'untrackedReads': 0}


GENERATED = {
    # fx's GraphModule.forward is exec'd too, under an '<eval_with_key>.N ...' file name.
    'exec': ('GENERATED = {}\nexec("def double(x):\\n    return x * 2\\n", GENERATED)\n',
             'GENERATED["double"](torch.relu(self.fc(x)))'),
    # transformers' ModelOutput.__init__ is a dataclass-generated method; here the field's
    # default factory runs an op inside it.
    'dataclass': ('import dataclasses, functools\n@dataclasses.dataclass\nclass Output:\n    hidden: torch.Tensor\n'
                  '    mask: torch.Tensor = dataclasses.field(default_factory=functools.partial(torch.ones, 2))\n',
                  'Output(torch.relu(self.fc(x))).hidden'),
}


@pytest.mark.parametrize('kind', sorted(GENERATED))
def test_a_model_running_generated_code_is_storable(tmp_path, kind):
    """Generated code's co_filename is '<string>', which names no file: the per-op frame
    capture must not look for '<cwd>/<string>', an existence check the audit cannot prove."""
    prelude, output = GENERATED[kind]
    model = PLAIN.replace('class Model', prelude + 'class Model').replace('torch.relu(self.fc(x))', output)
    root = _workspace(tmp_path / 'ws', model)
    assert _run(root, tmp_path) == {'storable': True, 'reasons': [], 'untrackedReads': 0}


def test_a_storable_run_lists_the_installed_files_it_read_with_their_current_fingerprints(tmp_path):
    """The key holds package versions, not file contents: the host re-stats these files before
    serving, so an in-place edit of an installed module is caught. ``colorsys`` (stdlib) is
    imported by the model itself, inside the audited window."""
    import colorsys
    from services.ingest.reuse_audit import fingerprint
    root = _workspace(tmp_path / 'ws', 'import colorsys\n' + PLAIN)
    report = _run(root, tmp_path, full=True)
    assert report['storable'] is True
    environment = dict(report['environment'])
    assert os.path.realpath(colorsys.__file__) in environment, sorted(environment)[:20]
    workspace = os.path.realpath(root)
    for path, stamp in environment.items():
        assert not path.startswith(workspace + os.sep), 'workspace files are keyed by content, never listed here'
        assert stamp == fingerprint(path), path


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
    from services.ingest.reuse_audit import _covered_prefixes, _under
    root = _workspace(tmp_path / 'ws', PLAIN)
    argv = ['--resolved-execution', '--workspace', str(root), '--file', str(root / 'model.py'), '--cls', 'Model', '--json']
    # The import path a worker has: the interpreter's trees, the backend and the workspace.
    # pytest also puts the repository root first; an import in the run that lists it after
    # anything there was added or removed would be an untracked read no worker makes.
    roots = (*_covered_prefixes(), os.path.realpath(BACKEND))
    path = [entry for entry in sys.path if any(_under(os.path.realpath(entry or '.'), r) for r in roots)]

    def once(enabled):
        monkeypatch.setattr(sys, 'path', [str(root), *path])
        monkeypatch.setenv('TENSORVIZ_EXECUTION_PLAN', str(_plan(root, tmp_path)))
        assert worker.main(argv + (['--reuse-audit'] if enabled else [])) == 0
        monkeypatch.undo()
        for name in ('model', 'helpers'):
            sys.modules.pop(name, None)
        return json.loads(capsys.readouterr().out)

    on, off = once(True), once(False)
    reuse = on['result']['diagnostics'].pop('reuse')
    assert reuse.pop('environment') and reuse == {'storable': True, 'reasons': [], 'untrackedReads': 0}
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


@pytest.mark.parametrize('access', ['read', 'existence check'])
def test_a_read_on_one_thread_is_seen_while_another_thread_is_being_classified(tmp_path, access):
    """The re-entrancy guard is per thread: a background thread classifying its own covered
    read must not hide the main thread's untracked one."""
    import threading
    from services.ingest.reuse_audit import ReadAudit
    workspace = _workspace(tmp_path / 'ws', PLAIN, {'scale.txt': '2.0\n'})
    audit = ReadAudit(str(workspace), ['model.py'])
    classify, classifying, release = audit._event, threading.Event(), threading.Event()

    def held(event, args):
        if threading.current_thread() is other:
            classifying.set()
            release.wait(10)
        classify(event, args)

    audit._event = held
    other = threading.Thread(target=(workspace / 'model.py').read_text)
    audit.start()
    try:
        other.start()
        assert classifying.wait(10)
        try:
            if access == 'read':
                (workspace / 'scale.txt').read_text()
            else:
                os.path.exists(workspace / 'ckpt.pt')
        finally:
            release.set()
            other.join(10)
    finally:
        report = audit.stop()
    assert report == {'storable': False, 'reasons': ['untracked_workspace_read'],
                      'untrackedReads': 1 if access == 'read' else 0, 'environment': []}


def test_wrapped_checks_keep_the_os_capabilities_of_the_originals(tmp_path):
    """``shutil.copystat(..., follow_symlinks=False)`` asks ``os.stat in os.supports_follow_symlinks``
    at call time: a wrapper missing from that set made it stat nothing and crash."""
    import shutil
    from services.ingest.reuse_audit import ReadAudit
    names = ('supports_dir_fd', 'supports_effective_ids', 'supports_fd', 'supports_follow_symlinks')
    before = {name: set(getattr(os, name)) for name in names}
    originals = (os.stat, os.lstat, os.access)
    (tmp_path / 'target').write_text('x')
    for link in ('src', 'dst'):
        (tmp_path / link).symlink_to(tmp_path / 'target')
    audit = ReadAudit(str(tmp_path / 'ws'), []).start()
    try:
        shutil.copystat(tmp_path / 'src', tmp_path / 'dst', follow_symlinks=False)
        for name in names:
            for check, original in zip((os.stat, os.lstat, os.access), originals):
                assert check is not original and (check in getattr(os, name)) == (original in before[name]), (name, check)
    finally:
        audit.stop()
    assert {name: set(getattr(os, name)) for name in names} == before


# -- reads the audit cannot see as ``open`` events -------------------------------------

def test_a_native_file_reader_is_not_storable(tmp_path):
    """safetensors opens its file in Rust, so no Python audit event names the weights. The
    run must not be stored: retraining would overwrite the file and the key would not change."""
    pytest.importorskip('safetensors')
    import torch
    from safetensors.torch import save_file
    model = PLAIN.replace('        self.fc = torch.nn.Linear(4, width())\n',
                          '        self.fc = torch.nn.Linear(4, width())\n'
                          '        from safetensors.torch import load_file\n'
                          '        self.load_state_dict(load_file(__file__.replace("model.py", "weights.safetensors")))\n')
    root = _workspace(tmp_path / 'ws', model)
    save_file({'fc.weight': torch.ones(3, 4), 'fc.bias': torch.zeros(3)}, str(root / 'weights.safetensors'))
    assert _run(root, tmp_path) == {'storable': False, 'reasons': ['native_code'], 'untrackedReads': 0}


def test_a_compiled_extension_in_the_workspace_is_not_storable(tmp_path):
    # A rebuilt extension changes no .py file, and loading one raises no ``open`` event.
    np = pytest.importorskip('numpy')
    import glob
    import shutil
    package = os.path.dirname(np.__file__)
    candidates = glob.glob(os.path.join(package, '_core', '_multiarray_tests.*')) + glob.glob(
        os.path.join(package, 'core', '_multiarray_tests.*'))
    if not candidates:
        pytest.skip('this numpy ships no _multiarray_tests extension to copy')
    root = _workspace(tmp_path / 'ws', 'import _multiarray_tests\n'
                      'assert _multiarray_tests.__file__.startswith(__file__.rsplit("/", 1)[0])\n' + PLAIN)
    shutil.copy(candidates[0], root / os.path.basename(candidates[0]))
    assert _run(root, tmp_path) == {'storable': False, 'reasons': ['native_code'], 'untrackedReads': 0}


def test_loading_a_shared_library_is_not_storable(tmp_path):
    # torch.ops.load_library and ctypes load native code that reads what it likes.
    root = _workspace(tmp_path / 'ws', 'import ctypes\nctypes.CDLL(None)\n' + PLAIN)
    assert _run(root, tmp_path) == {'storable': False, 'reasons': ['native_code'], 'untrackedReads': 0}


def test_torch_jit_load_of_a_workspace_file_is_not_storable(tmp_path):
    """torch.jit.load reads the archive in C++; its Python side only checks that the path
    exists, which is the one thing the audit can see."""
    import torch
    root = _workspace(tmp_path / 'ws', PLAIN.replace(
        '        self.fc = torch.nn.Linear(4, width())\n',
        '        self.fc = torch.nn.Linear(4, width())\n'
        '        self.scale = float(torch.jit.load(__file__.replace("model.py", "inner.pt")).weight.sum())\n'))
    torch.jit.save(torch.jit.script(torch.nn.Linear(2, 2)), str(root / 'inner.pt'))
    report = _run(root, tmp_path)
    assert report['storable'] is False and report['reasons'] == ['untracked_workspace_read']


LISTINGS = {
    'os.listdir': 'len(os.listdir(classes))',
    'os.scandir': 'len(list(os.scandir(classes)))',
    'glob.glob': 'len(glob.glob(os.path.join(classes, "*")))',
    'Path.iterdir': 'len(list(pathlib.Path(classes).iterdir()))',
}


@pytest.mark.parametrize('listing', sorted(LISTINGS))
def test_listing_a_workspace_directory_is_not_storable(tmp_path, listing):
    """`nn.Linear(512, len(os.listdir("data/classes")))`: adding a class folder changes the
    model, while no file the key covers changes."""
    model = 'import glob, os, pathlib\n' + PLAIN.replace(
        '        self.fc = torch.nn.Linear(4, width())\n',
        '        classes = __file__.replace("model.py", "data/classes")\n'
        f'        self.fc = torch.nn.Linear(4, {LISTINGS[listing]})\n')
    root = _workspace(tmp_path / 'ws', model)
    for name in ('cat', 'dog'):
        (root / 'data' / 'classes' / name).mkdir(parents=True)
    assert _run(root, tmp_path) == {'storable': False, 'reasons': ['untracked_workspace_read'], 'untrackedReads': 0}


EXISTENCE = {
    'os.path.exists': 'os.path.exists(checkpoint)',
    'os.path.isfile': 'os.path.isfile(checkpoint)',
    'os.path.lexists': 'os.path.lexists(checkpoint)',
    'Path.exists': 'pathlib.Path(checkpoint).exists()',
    'os.stat': 'stat_or_none(checkpoint)',
    'os.access': 'os.access(checkpoint, os.F_OK)',
}
#: The usual direct probe, which does not go through ``os.path``.
STAT_OR_NONE = 'def stat_or_none(path):\n    try:\n        return os.stat(path)\n    except FileNotFoundError:\n        return None\n'


@pytest.mark.parametrize('check', sorted(EXISTENCE))
def test_an_existence_check_in_the_workspace_is_not_storable(tmp_path, check):
    """`if os.path.exists("ckpt.pt"): load it`: the first run finds nothing and opens nothing,
    so only the existence check shows that saving a checkpoint later changes the result."""
    model = 'import os, pathlib\n' + STAT_OR_NONE + PLAIN.replace(
        '        super().__init__()\n',
        '        super().__init__()\n'
        '        checkpoint = __file__.replace("model.py", "ckpt.pt")\n'
        f'        self.resumed = bool({EXISTENCE[check]})\n')
    root = _workspace(tmp_path / 'ws', model)
    assert _run(root, tmp_path) == {'storable': False, 'reasons': ['untracked_workspace_read'], 'untrackedReads': 0}


def test_an_existence_check_outside_the_workspace_is_not_storable(tmp_path):
    flag = tmp_path / 'elsewhere' / 'flag'
    model = f'import os\nWIDE = os.path.exists({str(flag)!r})\n' + PLAIN
    root = _workspace(tmp_path / 'ws', model)
    assert _run(root, tmp_path) == {'storable': False, 'reasons': ['untracked_external_read'], 'untrackedReads': 0}


def test_checking_workspace_sources_and_their_folders_stays_storable(tmp_path):
    # Which .py files exist, and so which folders hold them, is already in the key.
    model = 'import os\n' + PLAIN.replace(
        '        super().__init__()\n',
        '        super().__init__()\n'
        '        here = os.path.dirname(__file__)\n'
        '        assert os.path.isdir(here) and os.path.exists(os.path.join(here, "helpers.py"))\n'
        '        assert not os.path.exists(os.path.join(here, "missing_module.py"))\n')
    root = _workspace(tmp_path / 'ws', model)
    assert _run(root, tmp_path) == {'storable': True, 'reasons': [], 'untrackedReads': 0}


def test_a_sqlite_database_is_a_read_and_a_folder_relative_check_is_unproven(tmp_path):
    """sqlite opens its database in C, so the ``sqlite3.connect`` event is the only sign that
    a model's labels come from a file. A check relative to a folder descriptor names no
    path the audit can classify. The wrapped checks are the originals again afterwards."""
    import sqlite3
    from services.ingest.reuse_audit import ReadAudit
    workspace = tmp_path / 'ws'
    workspace.mkdir()
    sqlite3.connect(workspace / 'labels.db').close()
    stat, exists = os.stat, os.path.exists

    def audited(operation):
        audit = ReadAudit(str(workspace), []).start()
        try:
            assert os.stat is not stat and os.path.exists is not exists
            operation()
        finally:
            report = audit.stop()
        assert os.stat is stat and os.path.exists is exists
        return {k: v for k, v in report.items() if k != 'environment'}

    assert audited(lambda: sqlite3.connect(':memory:').close()) == {'storable': True, 'reasons': [], 'untrackedReads': 0}
    assert audited(lambda: sqlite3.connect(workspace / 'labels.db').close()) == {
        'storable': False, 'reasons': ['untracked_workspace_read'], 'untrackedReads': 1}
    folder = os.open(workspace, os.O_RDONLY)
    try:
        assert audited(lambda: os.stat('labels.db', dir_fd=folder)) == {
            'storable': False, 'reasons': ['untracked_external_read'], 'untrackedReads': 0}
    finally:
        os.close(folder)


# -- the installed environment -----------------------------------------------------------

def test_an_installed_module_is_fingerprinted_and_an_in_place_edit_changes_the_fingerprint(tmp_path, monkeypatch):
    """Reads under the interpreter's trees are keyed by package version only, which an
    in-place edit (or a reinstall of the same version) keeps. The audit records each such
    file's stat fingerprint; the host compares it with the file before serving. Here the
    environment is a `.venv` inside the workspace, which must count as environment."""
    from services.ingest.reuse_audit import ReadAudit, fingerprint
    workspace = tmp_path / 'ws'
    site = workspace / '.venv' / 'lib' / 'site-packages'
    (site / 'fakelib').mkdir(parents=True)
    module = site / 'fakelib' / '__init__.py'
    module.write_text('WIDTH = 3\n')
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    audit = ReadAudit(str(workspace), [], extra_prefixes=[str(site)]).start()
    try:
        import fakelib  # noqa: F401
    finally:
        report = audit.stop()
        sys.modules.pop('fakelib', None)
    assert report['storable'] is True, report
    recorded = dict(report['environment'])
    real = os.path.realpath(module)
    assert recorded[real] is not None and recorded[real] == fingerprint(real)
    module.write_text('WIDTH = 30\n')  # what a user debugging the library does
    assert fingerprint(real) != recorded[real]


def test_a_workspace_inside_an_interpreter_prefix_still_tracks_its_data_files(tmp_path):
    # A system Python with sys.prefix '/usr' and a project in /usr/src/app.
    from services.ingest.reuse_audit import ReadAudit
    prefix = tmp_path / 'usr'
    workspace = prefix / 'src' / 'app'
    workspace.mkdir(parents=True)
    (workspace / 'scale.txt').write_text('2.0\n')
    audit = ReadAudit(str(workspace), [], extra_prefixes=[str(prefix)]).start()
    try:
        assert (workspace / 'scale.txt').read_text() == '2.0\n'
    finally:
        report = audit.stop()
    assert report == {'storable': False, 'reasons': ['untracked_workspace_read'], 'untrackedReads': 1, 'environment': []}


def test_a_link_in_a_workspace_inside_an_interpreter_prefix_still_reads_outside_it(tmp_path):
    # The same project linking a folder from elsewhere: the link is the project's, so what
    # it reaches is not part of the installation that encloses the project.
    from services.ingest.reuse_audit import ReadAudit
    prefix = tmp_path / 'usr'
    workspace = prefix / 'src' / 'app'
    workspace.mkdir(parents=True)
    vendor = tmp_path / 'opt' / 'vendor'
    vendor.mkdir(parents=True)
    (vendor / 'scale.txt').write_text('2.0\n')
    (workspace / 'vendor').symlink_to(vendor, target_is_directory=True)
    audit = ReadAudit(str(workspace), [], extra_prefixes=[str(prefix)]).start()
    try:
        assert (workspace / 'vendor' / 'scale.txt').read_text() == '2.0\n'
    finally:
        report = audit.stop()
    assert report == {'storable': False, 'reasons': ['untracked_external_read'], 'untrackedReads': 1, 'environment': []}


def test_a_package_linked_into_site_packages_is_fingerprinted_when_the_run_imports_it(tmp_path, monkeypatch):
    """Homebrew links each formula's Python package into its site-packages from the Cellar
    (``numpy`` -> ``../../../Cellar/numpy/<version>/...``). The run imports it through that
    link, so it is part of the installation: fingerprinted under the path it was found at,
    whose stat the host's recheck follows to the file, and never reported by its target."""
    from services.ingest.reuse_audit import ReadAudit, fingerprint
    site = tmp_path / 'lib' / 'site-packages'
    site.mkdir(parents=True)
    cellar = tmp_path / 'Cellar' / 'linkedpkg' / '1.0' / 'linkedpkg'
    cellar.mkdir(parents=True)
    (cellar / '__init__.py').write_text('')
    (cellar / 'layers.py').write_text(HELPERS)
    (site / 'linkedpkg').symlink_to(cellar, target_is_directory=True)
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    audit = ReadAudit(str(tmp_path / 'ws'), [], extra_prefixes=[str(site)]).start()
    try:
        from linkedpkg.layers import width
        assert width() == 3
    finally:
        report = audit.stop()
        sys.modules.pop('linkedpkg.layers', None)
        sys.modules.pop('linkedpkg', None)
    assert report['storable'] is True, report
    recorded = dict(report['environment'])
    linked = Path(os.path.realpath(site)) / 'linkedpkg'
    for name in ('__init__.py', 'layers.py'):
        assert recorded.get(str(linked / name)) == fingerprint(str(cellar / name)) is not None, name
    assert not any(path.startswith(str(cellar.parent)) for path in recorded)
    (cellar / 'layers.py').write_text('def width():\n    return 30\n')
    assert fingerprint(str(linked / 'layers.py')) != recorded[str(linked / 'layers.py')]


def test_a_linked_package_s_file_read_by_its_own_path_is_still_an_untracked_read(tmp_path):
    # The link decides, so a verdict holds for the path a read used, not for the file: the
    # read through site-packages must not vouch for a later read of its Cellar path.
    from services.ingest.reuse_audit import ReadAudit
    site = tmp_path / 'lib' / 'site-packages'
    site.mkdir(parents=True)
    cellar = tmp_path / 'Cellar' / 'linkedpkg' / '1.0' / 'linkedpkg'
    cellar.mkdir(parents=True)
    (cellar / 'scale.txt').write_text('2.0\n')
    (site / 'linkedpkg').symlink_to(cellar, target_is_directory=True)
    audit = ReadAudit(str(tmp_path / 'ws'), [], extra_prefixes=[str(site)]).start()
    try:
        assert (site / 'linkedpkg' / 'scale.txt').read_text() == '2.0\n'
        assert (cellar / 'scale.txt').read_text() == '2.0\n'
    finally:
        report = audit.stop()
    assert report == {'storable': False, 'reasons': ['untracked_external_read'], 'untrackedReads': 1, 'environment': []}


def test_a_venv_in_a_workspace_inside_an_interpreter_prefix_is_still_environment(tmp_path):
    # The same project with its own .venv: /usr is the outer prefix, but the venv's
    # site-packages is the interpreter's, so its files are fingerprinted, not refused.
    from services.ingest.reuse_audit import ReadAudit, _under, fingerprint
    prefix = tmp_path / 'usr'
    site = prefix / 'src' / 'app' / '.venv' / 'lib' / 'site-packages'
    (site / 'fakelib').mkdir(parents=True)
    module = site / 'fakelib' / '__init__.py'
    module.write_text('WIDTH = 3\n')
    audit = ReadAudit(str(prefix / 'src' / 'app'), [], extra_prefixes=[str(prefix), str(site)]).start()
    try:
        assert module.read_text() == 'WIDTH = 3\n'
    finally:
        report = audit.stop()
    real = os.path.realpath(module)
    # The modules this pytest process loaded before the audit are environment too; only the venv's file is this test's.
    environment = [entry for entry in report.pop('environment') if _under(entry[0], os.path.realpath(prefix))]
    assert report == {'storable': True, 'reasons': [], 'untrackedReads': 0} and environment == [[real, fingerprint(real)]]


def test_a_venv_inside_the_workspace_keeps_its_own_numpy_trusted(tmp_path):
    """VS Code's Create Environment, ``uv venv`` and Poetry put ``.venv`` in the project. The
    worker itself first imports numpy.random in the audited window (it seeds every run), so
    distrusting the environment's own numpy there would leave nothing ever storable."""
    script = '''import json, os, sys
import numpy
from services.ingest.reuse_audit import ReadAudit
project = os.path.dirname(os.path.realpath(sys.prefix))  # holds this interpreter's environment
inside = os.path.realpath(numpy.__file__).startswith(project + os.sep) and "numpy.random" not in sys.modules
audit = ReadAudit(project, []).start()
import numpy.random
numpy.random.seed(0)
report = audit.stop()
print(json.dumps({"inside": inside, "storable": report["storable"], "reasons": report["reasons"]}))
'''
    process = subprocess.run([sys.executable, '-B', '-c', script], cwd=BACKEND, capture_output=True, text=True, timeout=120,
                             env={**os.environ, 'KMP_DUPLICATE_LIB_OK': 'TRUE', 'PYTHONPATH': str(BACKEND)})
    assert process.returncode == 0, process.stderr[-2000:]
    result = json.loads(process.stdout)
    if not result.pop('inside'):
        pytest.skip('numpy is not installed under this interpreter\'s prefix, or is already imported at startup')
    assert result == {'storable': True, 'reasons': []}


def test_only_the_interpreter_s_own_extension_modules_are_trusted(tmp_path):
    """Native code can read files without an audit event. The standard library's,
    torch's/numpy's own and a compiled backend's modules are trusted; the same names loaded
    from anywhere else are not."""
    from services.ingest.reuse_audit import ReadAudit
    import numpy
    workspace = tmp_path / 'ws'
    site = workspace / '.venv' / 'lib' / 'site-packages'
    audit = ReadAudit(str(workspace), [], backend=tmp_path / 'backend', extra_prefixes=[str(site)])
    assert audit._trusted_native('_tensorviz_backend', str(tmp_path / 'backend' / ('_tensorviz_backend' + EXTENSION)))
    # Module origins arrive resolved, as the audit resolves them (Homebrew links its prefix).
    numpy_native = os.path.join(os.path.realpath(os.path.dirname(numpy.__file__)), '_core', 'x' + EXTENSION)
    base_native = os.path.join(os.path.realpath(sys.base_prefix), 'lib', 'lib-dynload', '_bz2' + EXTENSION)
    assert audit._trusted_native('numpy._core.x', numpy_native)
    assert audit._trusted_native('_bz2', base_native)
    assert not audit._trusted_native('_bz2', str(workspace / ('_bz2' + EXTENSION)))
    assert not audit._trusted_native('numpy', str(workspace / ('numpy' + EXTENSION)))
    assert not audit._trusted_native('safetensors._safetensors_rust', os.path.join(sys.prefix, 'lib', 'x' + EXTENSION))
    # An environment inside the workspace is still the environment, and still not trusted
    # beyond the standard library's and torch's and numpy's own.
    assert not audit._trusted_native('safetensors._safetensors_rust', str(site / 'safetensors' / ('x' + EXTENSION)))


def test_a_module_lazily_loaded_during_the_run_is_not_run_by_the_native_check(tmp_path):
    """``importlib.util.LazyLoader`` runs a module on its first attribute lookup. The model
    never touches this one, so checking the run's new modules for native code must not run
    it either: here it would raise, and the audit stops in the worker's ``finally``."""
    model = ('import importlib.util, sys\n'
             'spec = importlib.util.find_spec("optional_backend")\n'
             'spec.loader = importlib.util.LazyLoader(spec.loader)\n'
             'optional_backend = sys.modules["optional_backend"] = importlib.util.module_from_spec(spec)\n'
             'spec.loader.exec_module(optional_backend)\n' + PLAIN)
    root = _workspace(tmp_path / 'ws', model, {'optional_backend.py': 'raise RuntimeError("optional backend unavailable")\n'})
    assert _run(root, tmp_path) == {'storable': True, 'reasons': [], 'untrackedReads': 0}


def test_a_failing_native_check_refuses_the_run_instead_of_failing_it(tmp_path):
    from services.ingest.reuse_audit import ReadAudit
    audit = ReadAudit(str(tmp_path), []).start()
    audit._loaded_untrusted_native = None  # the check itself fails
    assert audit.stop() == {'storable': False, 'reasons': ['native_code'], 'untrackedReads': 0, 'environment': []}


# -- modules loaded before the audit started ----------------------------------------------

def _installed(site: Path, name: str) -> Path:
    (site / name).mkdir(parents=True)
    module = site / name / '__init__.py'
    module.write_text(HELPERS)
    return module


def test_an_installed_module_imported_before_the_audit_is_fingerprinted(tmp_path, monkeypatch):
    """torch and everything it imports are loaded before the audit starts, so their reads
    raised no event the audit saw. Their files must be rechecked before serving all the same."""
    from services.ingest.reuse_audit import ReadAudit, fingerprint
    workspace = tmp_path / 'ws'
    site = workspace / '.venv' / 'lib' / 'site-packages'
    module = _installed(site, 'preloadedlib')
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    import preloadedlib
    try:
        audit = ReadAudit(str(workspace), [], extra_prefixes=[str(site)]).start()
        try:
            assert preloadedlib.width() == 3
        finally:
            report = audit.stop()
    finally:
        sys.modules.pop('preloadedlib', None)
    assert report['storable'] is True, report
    real = os.path.realpath(module)
    recorded = dict(report['environment'])
    assert recorded.get(real) == fingerprint(real) is not None
    module.write_text('def width():\n    return 30\n')
    assert fingerprint(real) != recorded[real]


def test_a_module_remembered_before_an_edit_keeps_its_older_stamp(tmp_path, monkeypatch):
    """The pre-started worker imports torch, then waits for its request. A file edited while
    it waits must keep the stamp from before the edit: the process still runs the old code."""
    from services.ingest.reuse_audit import ReadAudit, fingerprint, remember_preloaded
    workspace = tmp_path / 'ws'
    site = workspace / '.venv' / 'lib' / 'site-packages'
    real = os.path.realpath(_installed(site, 'waitinglib'))
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    import waitinglib  # noqa: F401
    try:
        remember_preloaded()
        before = fingerprint(real)
        Path(real).write_text('def width():\n    return 30\n')
        report = ReadAudit(str(workspace), [], extra_prefixes=[str(site)]).start().stop()
    finally:
        sys.modules.pop('waitinglib', None)
    assert dict(report['environment'])[real] == before != fingerprint(real)


def test_a_module_imported_before_the_audit_from_outside_the_interpreter_is_not_storable(tmp_path, monkeypatch):
    """An editable install or a PYTHONPATH folder that torch's imports reach: package versions
    say nothing about it, and its path may not be reported."""
    from services.ingest.reuse_audit import ReadAudit
    lib = tmp_path / 'lib'
    _installed(lib, 'outsidelib')
    monkeypatch.syspath_prepend(str(lib))
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    import outsidelib  # noqa: F401
    try:
        report = ReadAudit(str(tmp_path / 'ws'), []).start().stop()
    finally:
        sys.modules.pop('outsidelib', None)
    assert report == {'storable': False, 'reasons': ['untracked_external_read'], 'untrackedReads': 1, 'environment': []}


def test_a_module_linked_into_the_interpreter_s_tree_from_outside_it_is_fingerprinted(tmp_path, monkeypatch):
    """Debian and Ubuntu ship ``/usr/lib/python3.X/sitecustomize.py`` as a link to
    ``/etc/python3.X/sitecustomize.py``, which ``site`` imports in every worker before the
    audit starts. It was found in the interpreter's tree, so it is part of that installation,
    not a reason to refuse every run on those systems."""
    from services.ingest.reuse_audit import ReadAudit, fingerprint
    prefix = tmp_path / 'usr'
    stdlib = prefix / 'lib' / 'python3.12'
    stdlib.mkdir(parents=True)
    target = tmp_path / 'etc' / 'python3.12' / 'linkedcustomize.py'
    target.parent.mkdir(parents=True)
    target.write_text(HELPERS)
    (stdlib / 'linkedcustomize.py').symlink_to(target)
    monkeypatch.syspath_prepend(str(stdlib))
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    import linkedcustomize  # noqa: F401  (as site imports sitecustomize)
    try:
        report = ReadAudit(str(tmp_path / 'ws'), [], extra_prefixes=[str(prefix)]).start().stop()
    finally:
        sys.modules.pop('linkedcustomize', None)
    assert report['storable'] is True, report
    link = os.path.join(os.path.realpath(stdlib), 'linkedcustomize.py')
    recorded = dict(report['environment'])
    assert recorded.get(link) == fingerprint(str(target)) is not None and str(target) not in recorded
    target.write_text('def width():\n    return 30\n')
    assert fingerprint(link) != recorded[link]


@pytest.mark.parametrize('when', ['before the audit', 'during the audit'])
@pytest.mark.parametrize('linked', ['module', 'package'])
def test_a_link_from_the_interpreter_s_tree_into_the_workspace_keeps_the_workspace_rules(tmp_path, monkeypatch, linked, when):
    """A link in site-packages to the project (a symlinked development install) does not make
    the project's files part of the installation: a module there that the key does not cover
    is still an untracked workspace read."""
    from services.ingest.reuse_audit import ReadAudit
    workspace = tmp_path / 'ws'
    module = _installed(workspace, 'linkedwslib')
    site = tmp_path / 'site'
    site.mkdir()
    if linked == 'module':
        (site / 'linkedwslib.py').symlink_to(module)
    else:
        (site / 'linkedwslib').symlink_to(module.parent, target_is_directory=True)
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    try:
        if when == 'before the audit':
            import linkedwslib  # noqa: F401
        audit = ReadAudit(str(workspace), [], extra_prefixes=[str(site)]).start()
        try:
            import linkedwslib  # noqa: F401,F811
        finally:
            report = audit.stop()
    finally:
        sys.modules.pop('linkedwslib', None)
    assert report['storable'] is False and report['reasons'] == ['untracked_workspace_read'], report


def test_a_lazily_loaded_module_is_fingerprinted_without_being_run(tmp_path, monkeypatch):
    """``importlib.util.LazyLoader`` runs a module on its first attribute lookup. The audit
    reads what the import system set without one, so it never runs code the model did not."""
    import importlib.util
    import types
    from services.ingest.reuse_audit import ReadAudit, fingerprint
    workspace = tmp_path / 'ws'
    site = workspace / '.venv' / 'lib' / 'site-packages'
    module = _installed(site, 'lazylib')
    module.write_text('RAN = True\n')
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    spec = importlib.util.spec_from_file_location('lazylib', module)
    spec.loader = importlib.util.LazyLoader(spec.loader)
    lazy = importlib.util.module_from_spec(spec)
    sys.modules['lazylib'] = lazy
    try:
        spec.loader.exec_module(lazy)
        report = ReadAudit(str(workspace), [], extra_prefixes=[str(site)]).start().stop()
        assert type(lazy) is not types.ModuleType and 'RAN' not in object.__getattribute__(lazy, '__dict__')
    finally:
        sys.modules.pop('lazylib', None)
    assert report['storable'] is True, report
    real = os.path.realpath(module)
    assert dict(report['environment']).get(real) == fingerprint(real) is not None


def test_a_loaded_module_that_cannot_be_fingerprinted_makes_runs_not_storable(tmp_path, monkeypatch):
    """A loaded module's file the audit cannot stat cannot be proven unchanged, by this run or
    by any later run of the process (a later stamp could postdate an edit). It refuses the
    run; it never fails it."""
    from services.ingest import reuse_audit
    from services.ingest.reuse_audit import ReadAudit
    workspace = tmp_path / 'ws'
    site = workspace / '.venv' / 'lib' / 'site-packages'
    real = os.path.realpath(_installed(site, 'unstatlib'))
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    monkeypatch.setattr(reuse_audit, '_unproven', False)  # per process: restored after this test
    original = reuse_audit.fingerprint

    def fingerprint(path):
        if path == real:
            raise PermissionError(13, 'Permission denied', path)
        return original(path)

    monkeypatch.setattr(reuse_audit, 'fingerprint', fingerprint)
    import unstatlib  # noqa: F401
    try:
        first = ReadAudit(str(workspace), [], extra_prefixes=[str(site)]).start().stop()
        monkeypatch.setattr(reuse_audit, 'fingerprint', original)
        later = ReadAudit(str(workspace), [], extra_prefixes=[str(site)]).start().stop()
    finally:
        sys.modules.pop('unstatlib', None)
    assert first == later == {'storable': False, 'reasons': ['untracked_external_read'], 'untrackedReads': 0, 'environment': []}


def test_a_worker_run_fingerprints_torch_files_it_imported_before_the_audit(tmp_path):
    """Conv2d's constructor runs ``torch.nn.modules.utils._pair``: torch code that no forward
    frame names, imported long before the audit started."""
    import torch
    import torch.nn.modules.utils as torch_utils
    from services.ingest.reuse_audit import fingerprint
    model = PLAIN.replace('self.fc = torch.nn.Linear(4, width())',
                          'self.conv = torch.nn.Conv2d(1, 2, kernel_size=3)\n        self.fc = torch.nn.Linear(4, width())')
    root = _workspace(tmp_path / 'ws', model)
    report = _run(root, tmp_path, full=True)
    assert report['storable'] is True
    environment = dict(report['environment'])
    for path in (torch_utils.__file__, torch.__file__):
        real = os.path.realpath(path)
        assert environment.get(real) == fingerprint(real) is not None, real
