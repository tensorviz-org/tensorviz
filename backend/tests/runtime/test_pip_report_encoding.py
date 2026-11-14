"""pip reports are UTF-8 even when Windows defaults to CP1252."""
import json
from pathlib import Path
import pytest
import importlib
import importlib.util
import os
import sys
if os.environ.get('TENSORVIZ_ENCODING_BACKEND'):
    sys.path.insert(0, os.environ['TENSORVIZ_ENCODING_BACKEND'])
    setup = importlib.import_module('_tensorviz_backend.services.ide.environment_setup')
elif os.environ.get('TENSORVIZ_ENCODING_BASELINE'):
    spec = importlib.util.spec_from_file_location('encoding_baseline', os.environ['TENSORVIZ_ENCODING_BASELINE'])
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
else:
    from services.ide import environment_setup as setup

@pytest.mark.parametrize('replacement', [False, True])
@pytest.mark.parametrize('payload,error', [
    (json.dumps({'install': [], 'description': 'caf\u00e9 \u8def\u5f84 \u270f\ufe0f'}, ensure_ascii=False).encode('utf-8'), None),
    (b'{"install":[],"description":"\xff"}', UnicodeDecodeError),
    (b'{not-json}', json.JSONDecodeError),
])
def test_pip_report_encoding(monkeypatch, tmp_path, replacement, payload, error):
    directory = tmp_path / 'caf\u00e9-\u8def\u5f84'
    directory.mkdir()
    requirements = directory / 'requirements.txt'
    requirements.write_text('demo==1\n', encoding='utf-8')
    monkeypatch.setattr(setup, 'identity', lambda: {'packages': {}})
    monkeypatch.setattr(setup, 'check_install_target', lambda: None)
    monkeypatch.setattr(setup, 'project_requirements', lambda root: {'demo': 'demo==1'})
    monkeypatch.setattr(setup, 'installed_requirements', lambda: {})
    monkeypatch.setattr(setup, 'broken_dependents', lambda *args: [])
    seen=[]
    monkeypatch.setattr(setup, 'validate_wheels', lambda report, baseline: seen.append(report) or [])
    def pip_run(args):
        Path(args[args.index('--report')+1]).write_bytes(payload)
    monkeypatch.setattr(setup, 'pip_run', pip_run)
    # Simulate Windows' locale default even on UTF-8 test hosts.
    read_text = Path.read_text
    def windows_read(path, encoding=None, errors=None):
        return read_text(path, encoding=encoding or 'cp1252', errors=errors)
    monkeypatch.setattr(Path, 'read_text', windows_read)
    def plan():
        return setup.plan_replacement(directory, ['demo==1'], directory) if replacement else setup.plan(directory, requirements)
    if error:
        with pytest.raises(error): plan()
        assert not seen
        assert not (directory/'plan.json').exists()
    else:
        assert plan()['packages'] == []
        assert seen[0]['description'] == 'caf\u00e9 \u8def\u5f84 \u270f\ufe0f'
        assert json.loads((directory/'plan.json').read_bytes())['wheels'] == []
