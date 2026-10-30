"""The pre-started worker's own contract, driven as a real subprocess.

The host suite proves the handoff end to end; these pin the parts only the
process itself can show: what happens on EOF, on a malformed request, that stdout
stays clean, that the run equals an ordinary launch, and that the parent-death
pipe still kills the run once the request has been consumed.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[2]
STANDBY = BACKEND / 'services' / 'ingest' / 'standby.py'
BOOTSTRAP = BACKEND / 'services' / 'ingest' / 'bootstrap.py'
MODEL = """import torch
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 4)
    def forward(self, x):
{body}
        return self.a(x)
"""


def env(**extra):
    base = {**os.environ, 'KMP_DUPLICATE_LIB_OK': 'TRUE', 'PYTHONUNBUFFERED': '1'}
    base.pop('TENSORVIZ_PARENT_PIPE', None)
    return {**base, **extra}


def start(**extra):
    return subprocess.Popen([sys.executable, str(STANDBY)], cwd=BACKEND, env=env(**extra),
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            start_new_session=True)


def argv(file):
    return ['--file', str(file), '--cls', 'Model', '--input', '2,4', '--depth', '2', '--json']


@pytest.fixture
def model(tmp_path):
    file = tmp_path / 'model.py'
    file.write_text(MODEL.format(body='        pass'))
    return file


def test_eof_before_a_request_is_a_clean_silent_exit():
    process = start()
    out, err = process.communicate(b'', timeout=60)
    assert process.returncode == 0 and out == b''


def test_malformed_request_refuses_and_runs_nothing():
    for line in (b'{"argv": "not a list", "env": {}}\n', b'not json\n', b'{"argv": [], "env": {"A": 1}}\n'):
        process = start()
        out, err = process.communicate(line, timeout=60)
        assert process.returncode == 2 and out == b'', (line, err)


def test_trailing_data_after_the_request_is_refused():
    process = start()
    out, _ = process.communicate(b'{"argv": [], "env": {}}\n{"argv": []}\n', timeout=60)
    assert process.returncode == 2 and out == b''


def test_a_request_runs_exactly_the_ordinary_bootstrap(model):
    ordinary = subprocess.run([sys.executable, str(BOOTSTRAP), *argv(model)], cwd=BACKEND, env=env(),
                              capture_output=True, timeout=120)
    process = start()
    time.sleep(0.5)
    request = json.dumps({'argv': argv(model), 'env': {}}) + '\n'
    process.stdin.write(request.encode()); process.stdin.flush()
    out = process.stdout.read()
    process.wait(timeout=120)
    process.stdin.close()
    assert process.returncode == ordinary.returncode == 0
    standby, fresh = json.loads(out), json.loads(ordinary.stdout)
    assert standby['error'] is None and fresh['error'] is None
    assert standby['result']['graph'] == fresh['result']['graph']
    assert out.count(b'\n') == ordinary.stdout.count(b'\n'), 'stdout carries only the envelope'


def test_main_module_matches_an_ordinary_script_launch(model, tmp_path):
    """User code that inspects __main__ or sys.modules sees an ordinary launch."""
    marker = tmp_path / 'main.json'
    probe = (f'        import sys, json; m = sys.modules["__main__"]; json.dump(dict('
             f'loader=type(m.__loader__).__name__, spec=repr(m.__spec__), file=m.__file__, '
             f'runpy="runpy" in sys.modules), open({str(marker)!r}, "w"))')
    model.write_text(MODEL.format(body=probe))
    subprocess.run([sys.executable, str(BOOTSTRAP), *argv(model)], cwd=BACKEND, env=env(), capture_output=True, timeout=120)
    ordinary = json.loads(marker.read_text())
    process = start()
    out, _ = process.communicate((json.dumps({'argv': argv(model), 'env': {}}) + '\n').encode(), timeout=120)
    assert json.loads(out)['error'] is None
    assert json.loads(marker.read_text()) == ordinary
    # Torch imports runpy while loading, so a script launch is no longer a
    # process that has never imported it. The invariant is that standby matches
    # the ordinary launch, including whatever runpy that launch already has.


def test_request_environment_reaches_the_worker(model, tmp_path):
    marker = tmp_path / 'seen.txt'
    model.write_text(MODEL.format(body=f'        open({str(marker)!r}, "w").write(__import__("os").environ.get("TV_DELTA", "missing"))'))
    process = start()
    request = json.dumps({'argv': argv(model), 'env': {'TV_DELTA': 'delivered'}}) + '\n'
    out, _ = process.communicate(request.encode(), timeout=120)
    assert json.loads(out)['error'] is None
    assert marker.read_text() == 'delivered'


@pytest.mark.skipif(os.name == 'nt', reason='process-group kill is POSIX here')
def test_parent_death_after_the_request_still_kills_the_run(model):
    """The request consumes one line of stdin, which is also the death pipe. After
    it, closing stdin must still make the worker kill its own group, exactly as an
    ordinary TENSORVIZ_PARENT_PIPE launch does."""
    model.write_text(MODEL.format(body='        import time; time.sleep(60)'))
    process = start(TENSORVIZ_PARENT_PIPE='1')
    process.stdin.write((json.dumps({'argv': argv(model), 'env': {}}) + '\n').encode())
    process.stdin.flush()
    time.sleep(3)  # inside forward()
    assert process.poll() is None
    process.stdin.close()  # the host went away
    process.wait(timeout=15)
    assert process.returncode in (1, -signal.SIGKILL)


def test_workspace_module_named_bootstrap_is_not_shadowed(model, tmp_path):
    """run_path must not register `bootstrap` in sys.modules, so a user's own
    module of that name still imports."""
    (tmp_path / 'bootstrap.py').write_text('VALUE = 7\n')
    model.write_text(MODEL.format(body='        import sys; sys.path.insert(0, %r); import bootstrap; assert bootstrap.VALUE == 7' % str(tmp_path)))
    process = start()
    out, err = process.communicate((json.dumps({'argv': argv(model), 'env': {}}) + '\n').encode(), timeout=120)
    assert json.loads(out)['error'] is None, err.decode()[-2000:]


# --- Pre-loaded capture internals -------------------------------------------------

PROBE = ('        import sys, json; m = sys.modules["__main__"]; json.dump(dict('
         'private=sorted(k for k in sys.modules if k.startswith("_tensorviz_capture_worker")), '
         'package=id(sys.modules["_tensorviz_capture_worker"]), '
         'worker_package=id(sys.modules["_tensorviz_capture_worker.ingest.worker"].__package__ and '
         'sys.modules["_tensorviz_capture_worker"]), '
         'user=sorted(k for k in sys.modules if not k.startswith("_tensorviz")), '
         'path=sys.path, main=[type(m.__loader__).__name__, repr(m.__spec__), m.__file__], '
         'threads=__import__("torch").get_num_threads()), open({marker!r}, "w"))')


def _probe(model, marker, how):
    model.write_text(MODEL.format(body=PROBE.format(marker=str(marker))))
    if how == 'fresh':
        done = subprocess.run([sys.executable, str(BOOTSTRAP), *argv(model)], cwd=BACKEND, env=env(), capture_output=True, timeout=120)
        out = done.stdout
    else:
        process = start()
        time.sleep(3)  # long enough for torch and the internals to be pre-loaded
        out, _ = process.communicate((json.dumps({'argv': argv(model), 'env': {}}) + '\n').encode(), timeout=120)
    assert json.loads(out)['error'] is None
    return json.loads(marker.read_text())


def test_standby_preloads_internals_and_the_run_reuses_that_package(model, tmp_path):
    """The run must use the package the standby loaded, not a second copy of it."""
    process = start()
    time.sleep(3)
    marker = tmp_path / 'seen.json'
    model.write_text(MODEL.format(body=(
        '        import sys, json; p = sys.modules["_tensorviz_capture_worker"]; '
        'i = sys.modules["_tensorviz_capture_worker.ingest"]; '
        # A second package object would not have the already-imported subpackage
        # bound as an attribute: binding happens on the object the import ran against.
        'json.dump(dict(marker=getattr(p, "__tensorviz_loaded_from__", None), '
        'same=getattr(p, "ingest", None) is i, '
        'service="_tensorviz_capture_worker.ingest.service" in sys.modules), open(%r, "w"))' % str(marker))))
    out, err = process.communicate((json.dumps({'argv': argv(model), 'env': {}}) + '\n').encode(), timeout=120)
    assert json.loads(out)['error'] is None, err.decode()[-2000:]
    seen = json.loads(marker.read_text())
    assert seen['marker'] == str(BACKEND / 'services' / '__init__.py')
    assert seen['same'] is True
    assert seen['service'] is True


def test_what_user_code_sees_matches_a_fresh_launch(model, tmp_path):
    """Pre-loading may only change which tensorViz-private modules are loaded and
    when. Everything user code can observe stays as a fresh launch has it."""
    fresh = _probe(model, tmp_path / 'fresh.json', 'fresh')
    standby = _probe(model, tmp_path / 'standby.json', 'standby')
    for key in ('user', 'path', 'main', 'threads'):
        assert standby[key] == fresh[key], key
    assert set(fresh['private']) <= set(standby['private'])


def test_a_failing_preload_still_runs_correctly(model, tmp_path):
    """Pre-loading is best effort. If it raises, the run loads the internals itself."""
    broken = tmp_path / 'sitecustomize.py'
    broken.write_text(
        'import importlib.util as u\n'
        '_orig = u.spec_from_file_location\n'
        'def _fail(name, *a, **k):\n'
        '    if name == "_tensorviz_standby_bootstrap":\n'
        '        raise RuntimeError("preload broken for this test")\n'
        '    return _orig(name, *a, **k)\n'
        'u.spec_from_file_location = _fail\n')
    expected = subprocess.run([sys.executable, str(BOOTSTRAP), *argv(model)], cwd=BACKEND, env=env(), capture_output=True, timeout=120)
    process = start(PYTHONPATH=str(tmp_path))
    out, err = process.communicate((json.dumps({'argv': argv(model), 'env': {}}) + '\n').encode(), timeout=120)
    assert process.returncode == 0, err.decode()[-2000:]
    assert json.loads(out)['result']['graph'] == json.loads(expected.stdout)['result']['graph']
    assert out.count(b'\n') == expected.stdout.count(b'\n'), 'stdout still carries only the envelope'


def test_a_foreign_package_under_the_private_name_is_replaced_not_reused(tmp_path):
    """Only a package that load_internals itself created may be reused."""
    script = tmp_path / 'check.py'
    script.write_text(
        'import sys, types, importlib.util\n'
        f'sys.path.insert(0, {str(BACKEND / "services" / "ingest")!r})\n'
        'impostor = types.ModuleType("_tensorviz_capture_worker")\n'
        'sys.modules["_tensorviz_capture_worker"] = impostor\n'
        'import bootstrap\n'
        'worker = bootstrap.load_internals()\n'
        'package = sys.modules["_tensorviz_capture_worker"]\n'
        'assert package is not impostor\n'
        f'assert package.__tensorviz_loaded_from__ == {str(BACKEND / "services" / "__init__.py")!r}\n'
        'again = bootstrap.load_internals()\n'
        'assert sys.modules["_tensorviz_capture_worker"] is package and again is worker\n'
        'print("ok")\n')
    done = subprocess.run([sys.executable, str(script)], cwd=BACKEND, env=env(), capture_output=True, timeout=120)
    assert done.stdout.decode().strip() == 'ok', done.stderr.decode()[-2000:]


def test_preload_prints_nothing_and_eof_still_exits_cleanly():
    process = start()
    time.sleep(3)
    out, err = process.communicate(b'', timeout=60)
    assert process.returncode == 0
    assert out == b''


# --- Execution-cache audit of what was pre-loaded ---------------------------------

def test_standby_fingerprints_what_it_preloaded_before_it_waits():
    """A file edited while the standby waits must keep its pre-edit stamp: the run that
    claims this process executes the code loaded before the edit."""
    import torch
    from services.ingest.reuse_audit import fingerprint
    script = ('import importlib.util, json, sys, torch\n'
              f'spec = importlib.util.spec_from_file_location("standby_under_test", {str(STANDBY)!r})\n'
              'standby = importlib.util.module_from_spec(spec); spec.loader.exec_module(standby)\n'
              f'assert standby._preload({str(BOOTSTRAP)!r})\n'
              'audit = sys.modules["_tensorviz_capture_worker.ingest.reuse_audit"]\n'
              'print(json.dumps(audit._remembered.get(torch.__spec__.origin)))\n')
    process = subprocess.run([sys.executable, '-B', '-c', script], cwd=BACKEND, env=env(), capture_output=True, text=True, timeout=120)
    assert process.returncode == 0, process.stderr[-2000:]
    real = os.path.realpath(torch.__file__)
    assert json.loads(process.stdout)[0] == [torch.__spec__.origin, real, fingerprint(real)]


def test_standby_and_fresh_runs_report_the_same_installed_files(tmp_path):
    """The host compares a stored result with the next run, which may be either kind: the
    installed files they report (including those loaded before the audit) must agree."""
    import torch
    from tests.capture.test_reuse_audit import PLAIN, _plan, _workspace
    root = _workspace(tmp_path / 'ws', PLAIN)
    request = ['--resolved-execution', '--workspace', str(root), '--file', str(root / 'model.py'), '--cls', 'Model',
               '--json', '--reuse-audit']
    fresh = subprocess.run([sys.executable, str(BOOTSTRAP), *request], cwd=BACKEND, capture_output=True, timeout=120,
                           env=env(TENSORVIZ_EXECUTION_PLAN=str(_plan(root, tmp_path))))
    process = start()
    time.sleep(3)  # pre-loaded, and waiting
    out, err = process.communicate((json.dumps({'argv': request, 'env': {
        'TENSORVIZ_EXECUTION_PLAN': str(_plan(root, tmp_path))}}) + '\n').encode(), timeout=120)
    assert process.returncode == fresh.returncode == 0, err.decode()[-2000:]
    standby, ordinary = (json.loads(text)['result']['diagnostics']['reuse'] for text in (out, fresh.stdout))
    assert standby == ordinary and standby['storable'] is True
    assert os.path.realpath(torch.__file__) in dict(standby['environment'])


@pytest.mark.skipif(os.name == 'nt', reason='the hand-off to a fresh worker is POSIX only')
@pytest.mark.parametrize('change, launch', [('unchanged', 'legacy'), ('edited', 'legacy'), ('relinked', 'legacy'),
                                            ('edited', 'plan')])
def test_a_standby_whose_preloaded_files_changed_continues_as_a_fresh_worker(model, tmp_path, change, launch):
    """A module the standby loaded before it began to wait is edited, or its link re-pointed
    (``brew upgrade`` links the new keg and may keep the old one), before its request
    arrives: the run must execute the new code, as an ordinary launch would. The process
    replaces itself, so the host's pid, process group and pipes still reach the run. The host
    claims a standby only for a resolved plan, whose file it names in the request's
    environment alone (the standby started without it), so the new process must get that too."""
    site = tmp_path / 'site'
    site.mkdir()
    (site / 'sitecustomize.py').write_text('import preloadedwidth\n')  # loaded before the standby waits
    library = site / 'preloadedwidth.py'
    kegs = [tmp_path / 'Cellar' / version / 'preloadedwidth.py' for version in ('1.0', '2.0')]
    if change == 'relinked':
        for keg, width in zip(kegs, (3, 30)):
            keg.parent.mkdir(parents=True)
            keg.write_text(f'def width():\n    return {width}\n')
        library.symlink_to(kegs[0])
    else:
        library.write_text('def width():\n    return 3\n')
    while_waiting = {'unchanged': 'pass',
                     'edited': f'pathlib.Path({str(library)!r}).write_text("def width():\\n    return 30\\n")',
                     'relinked': f'os.remove({str(library)!r}); os.symlink({str(kegs[1])!r}, {str(library)!r})'}[change]
    marker = tmp_path / 'seen.json'
    model.write_text(MODEL.format(body='        import os, json, preloadedwidth; json.dump([preloadedwidth.width(), '
                                       f'os.getpid(), os.getpgid(0)], open({str(marker)!r}, "w"))'))
    request = {'argv': argv(model), 'env': {}}
    if launch == 'plan':  # as traceWorkerCore's claimStandby sends it
        from tests.capture.test_reuse_audit import _plan, _workspace
        root = _workspace(tmp_path / 'ws', model.read_text())
        request = {'argv': ['--resolved-execution', '--workspace', str(root), '--file', str(root / 'model.py'), '--cls', 'Model',
                            '--json'], 'env': {'TENSORVIZ_EXECUTION_PLAN': str(_plan(root, tmp_path))}}
    # The standby itself, with the edit landing while it waits for its request.
    script = ('import importlib.util, os, pathlib, sys\n'
              f'spec = importlib.util.spec_from_file_location("standby_under_test", {str(STANDBY)!r})\n'
              'standby = importlib.util.module_from_spec(spec); spec.loader.exec_module(standby)\n'
              'read = standby._read_request\n'
              'def changed_while_waiting():\n'
              f'    {while_waiting}\n'
              '    return read()\n'
              'standby._read_request = changed_while_waiting\n'
              'sys.exit(standby.main())\n')
    process = subprocess.Popen([sys.executable, '-c', script], cwd=BACKEND, env=env(PYTHONPATH=str(site)),
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    out, err = process.communicate((json.dumps(request) + '\n').encode(), timeout=120)
    envelope = json.loads(out)
    assert process.returncode == 0 and envelope['error'] is None, (envelope['error'], err.decode()[-2000:])
    assert json.loads(marker.read_text()) == [3 if change == 'unchanged' else 30, process.pid, process.pid]
    assert (b'continuing as a fresh worker' in err) is (change != 'unchanged')
    if launch == 'plan':
        assert envelope['result']['diagnostics']['execution']['planCurrent'] is True
