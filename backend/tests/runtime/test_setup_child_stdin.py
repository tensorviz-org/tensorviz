"""Noninteractive setup children must not borrow the parent-lifetime pipe."""
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
from types import SimpleNamespace
import venv

import pytest

from services.ide import environment_setup as setup


@pytest.mark.parametrize('operation', [
    'snapshot', 'pip', 'requirements', 'model-probe', 'capture-probe',
])
def test_setup_subprocesses_use_devnull(operation, monkeypatch, tmp_path):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        assert kwargs.get('stdin') == subprocess.DEVNULL
        assert kwargs['capture_output'] is True
        return SimpleNamespace(returncode=0, stdout=json.dumps({
            'ok': True, 'identity': {'packages': {}},
        }), stderr='')

    monkeypatch.setattr(setup.subprocess, 'run', run)
    monkeypatch.setattr(setup, 'TARGET_PYTHON', sys.executable)
    if operation == 'snapshot':
        setup._target_snapshot()
    elif operation == 'pip':
        setup.pip_run(['--version'])
    elif operation == 'requirements':
        setup.installed_requirements()
    elif operation == 'model-probe':
        setup.probe_model_import(tmp_path, tmp_path / 'model.py')
    else:
        setup.probe_capture()
    assert len(calls) == 1


def test_venv_bootstrap_uses_devnull_without_disabling_lifetime_guard(monkeypatch, tmp_path, capsys):
    calls = []
    def check_output(command, **kwargs):
        calls.append((command, kwargs))
        assert kwargs['stdin'] == subprocess.DEVNULL
        return b''

    def create(builder, destination):
        assert builder.with_pip
        assert Path(destination) == tmp_path
        context = SimpleNamespace(env_exec_cmd=sys.executable, env_exe=sys.executable,
                                  env_dir=str(tmp_path), bin_path=str(tmp_path))
        builder._call_new_python(context, '-m', 'ensurepip', '--upgrade', '--default-pip')

    # No package installation or network access: exercise the actual subclass
    # and superclass launch path with only the OS spawn replaced.
    monkeypatch.setattr(subprocess, 'check_output', check_output)
    monkeypatch.setattr(venv.EnvBuilder, 'create', create)
    monkeypatch.setattr(sys, 'argv', ['environment_setup.py', 'create-environment', str(tmp_path)])
    setup.main()
    assert len(calls) == 1
    assert 'ensurepip' in calls[0][0]
    assert json.loads(capsys.readouterr().out)['ok'] is True


def test_real_setup_children_start_with_blocking_parent_pipe():
    program = '''
import ast, json, os, pathlib, runpy, sys, threading, types, venv
source = pathlib.Path(sys.argv[1])
namespace = runpy.run_path(str(source))
namespace['venv'] = venv
builder = next(n for n in ast.walk(ast.parse(source.read_text()))
               if isinstance(n, ast.ClassDef) and n.name == 'SetupEnvBuilder')
exec(compile(ast.Module(body=[builder], type_ignores=[]), str(source), 'exec'), namespace)
threading.Thread(target=lambda: os.read(0, 1024), daemon=True).start()
context = types.SimpleNamespace(env_exec_cmd=sys.executable, env_dir=sys.prefix)
namespace['SetupEnvBuilder']()._call_new_python(context, '-c', 'print("started")', timeout=5)
namespace['pip_run'](['--version'])
snapshot = namespace['_target_snapshot']
snapshot.__globals__['TARGET_PYTHON'] = sys.executable
assert snapshot()['identity']['executable'] == sys.executable
print(json.dumps({'ok': True}), flush=True)
'''
    with subprocess.Popen([sys.executable, '-I', '-B', '-c', program, str(Path(setup.__file__).resolve())],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          start_new_session=os.name != 'nt') as child:
        output = queue.Queue()
        threading.Thread(target=lambda: output.put(child.stdout.readline()), daemon=True).start()
        try:
            # Hold stdin open, exactly as the owner does while its helper runs.
            assert json.loads(output.get(timeout=30))['ok'] is True
            assert child.wait(timeout=5) == 0
        finally:
            if child.poll() is None:
                if os.name == 'nt':
                    subprocess.run([str(Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32' / 'taskkill.exe'),
                                    '/PID', str(child.pid), '/T', '/F'],
                                   stdin=subprocess.DEVNULL, capture_output=True, timeout=10)
                else:
                    os.killpg(child.pid, signal.SIGKILL)
            child.stdin.close()
