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
