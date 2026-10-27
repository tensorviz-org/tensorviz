"""One read-only inspection reads the target environment once (``one_snapshot``).

The inspections used to start one target Python per fact they needed: three
identical processes for ``inspect-capture`` and two for ``inspect-model``. These
tests pin the reuse and its limits: only inside the read-only actions, only for a
snapshot that already answered the requested modules, and never across requests or
around actions that change the environment.
"""
import json
import sys

import pytest

from services.ide import environment_setup as setup


@pytest.fixture
def snapshots(monkeypatch):
    """Count target processes; each returns a distinct snapshot so reuse is visible."""
    calls = []

    def fake(modules=None):
        calls.append(list(modules or []))
        return {'identity': {'executable': '/t/python', 'prefix': '/t', 'python': '3.12.13', 'platform': 'darwin',
                             'architecture': 'arm64', 'implementation': 'cpython', 'abi': 'cpython-312-darwin',
                             'packages': {'torch': '2.7.1', 'pip': '25.0'}, 'call': len(calls)},
                'modules': {name: True for name in modules or []}, 'stdlib': ['os', 'sys'],
                'basePrefix': '/base', 'conda': False, 'externallyManaged': False, 'writable': True}

    monkeypatch.setattr(setup, '_target_snapshot', fake)
    monkeypatch.setattr(setup, 'TARGET_PYTHON', '/t/python')
    return calls


def test_outside_an_inspection_every_call_reads_the_target_again(snapshots):
    assert setup.target_snapshot()['identity']['call'] == 1
    assert setup.target_snapshot()['identity']['call'] == 2


def test_inside_one_inspection_the_target_is_read_once(snapshots):
    with setup.one_snapshot():
        first = setup.target_snapshot()
        assert setup.inventory() == first['identity']['packages']
        assert setup.identity() is first['identity']
        assert setup.environment_info()['torch'] == '2.7.1'
    assert len(snapshots) == 1


def test_a_snapshot_answers_only_for_modules_it_was_asked_about(snapshots):
    with setup.one_snapshot():
        wide = setup.target_snapshot(['torch', 'numpy'])
        assert setup.target_snapshot(['numpy']) is wide        # subset: reused
        assert setup.target_snapshot() is wide                 # identity only: reused
        other = setup.target_snapshot(['einops'])              # not answered: read again
        assert other is not wide and other['modules'] == {'einops': True}
    assert snapshots == [['numpy', 'torch'], ['einops']]


def test_the_scope_ends_with_the_inspection_even_on_error(snapshots):
    with pytest.raises(RuntimeError):
        with setup.one_snapshot():
            setup.target_snapshot()
            raise RuntimeError('inspection failed')
    assert setup._SNAPSHOTS is None
    setup.target_snapshot()
    assert len(snapshots) == 2


@pytest.mark.parametrize('action,expected', [('inspect-capture', 1), ('inspect', 1), ('inspect-runtime', 1)])
def test_inspection_actions_start_one_target_process(snapshots, monkeypatch, capsys, action, expected):
    monkeypatch.setattr(sys, 'argv', ['environment_setup.py', '--target-python', '/t/python', action])
    setup.main()
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])['ok'] is True
    assert len(snapshots) == expected


def test_model_inspection_starts_one_target_process(snapshots, monkeypatch, capsys, tmp_path):
    model = tmp_path / 'model.py'
    model.write_text('import torch\nimport numpy\nfrom torch import nn\nclass Net(nn.Module):\n    pass\n')
    monkeypatch.setattr(sys, 'argv', ['environment_setup.py', '--target-python', '/t/python', 'inspect-model', str(tmp_path), str(model)])
    setup.main()
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])['ok'] is True
    assert snapshots == [['numpy', 'torch']]


def test_candidate_inspection_checks_capture_and_model_from_one_target_process(snapshots, monkeypatch, capsys, tmp_path):
    model = tmp_path / 'model.py'
    model.write_text('import torch\nimport numpy\nfrom torch import nn\nclass Net(nn.Module):\n    pass\n')
    monkeypatch.setattr(sys, 'argv', ['environment_setup.py', '--target-python', '/t/python', 'inspect-candidate', str(tmp_path), str(model)])
    setup.main()
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert report['ok'] is True and report['torch'] == '2.7.1'
    assert report['model'] == {'missingModules': [], 'incompatible': []}
    # The model's modules were asked first, so the capture check reused that snapshot.
    assert snapshots == [['numpy', 'torch']]


def test_candidate_inspection_reports_what_the_model_still_needs(monkeypatch, tmp_path):
    (tmp_path / 'requirements.txt').write_text('transformers>=5\neinops\n')
    model = tmp_path / 'model.py'
    model.write_text('import torch\nimport einops\nfrom transformers import AutoModel\n')
    packages = {'torch': '2.7.1', 'transformers': '4.57.6', 'pydantic': '2.12.0', 'packaging': '25.0', 'numpy': '2.3.1'}
    monkeypatch.setattr(setup, 'TARGET_PYTHON', '/t/python')
    monkeypatch.setattr(setup, '_target_snapshot', lambda modules=None: {
        'identity': {'executable': '/t/python', 'prefix': '/t', 'python': '3.12.13', 'platform': 'darwin', 'architecture': 'arm64',
                     'implementation': 'cpython', 'abi': 'cpython-312-darwin', 'packages': packages},
        'modules': {name: name != 'einops' for name in modules or []}, 'stdlib': ['os', 'sys']})
    with setup.one_snapshot():
        report = setup.inspect_candidate(tmp_path, model)
    assert report['missing'] == [] and report['incompatible'] == []
    assert report['model']['missingModules'] == [{'module': 'einops', 'requirement': 'einops'}]
    assert report['model']['incompatible'] == ['transformers 4.57.6 (needs >=5)']


def test_candidate_inspection_keeps_environment_facts_when_the_model_cannot_be_checked(snapshots, tmp_path):
    outside = tmp_path.parent / f'{tmp_path.name}-outside.py'
    outside.write_text('import torch\n')
    try:
        with setup.one_snapshot():
            report = setup.inspect_candidate(tmp_path, outside)
    finally:
        outside.unlink()
    assert 'model' not in report and report['torch'] == '2.7.1' and report['environment']['executable'] == '/t/python'


def test_candidate_inspection_fails_only_for_an_unavailable_environment(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(setup, 'TARGET_PYTHON', None)
    model = tmp_path / 'model.py'
    model.write_text('import torch\n')

    def unavailable(modules=None):
        raise setup.SetupError('The selected Python environment could not be inspected. Choose another environment.', 'ENVIRONMENT_UNAVAILABLE')
    monkeypatch.setattr(setup, '_target_snapshot', unavailable)
    monkeypatch.setattr(sys, 'argv', ['environment_setup.py', '--target-python', '/t/python', 'inspect-candidate', str(tmp_path), str(model)])
    with pytest.raises(SystemExit):
        setup.main()
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert report == {'ok': False, 'message': 'The selected Python environment could not be inspected. Choose another environment.',
                      'code': 'ENVIRONMENT_UNAVAILABLE'}


def test_actions_that_change_the_environment_still_compare_fresh_snapshots(snapshots):
    """plan/install take a baseline identity and compare it after their change; a
    reused snapshot would hide the change, so they never run inside the scope."""
    baseline = setup.identity()
    assert setup.identity() != baseline   # a second read is a new process (call 2)
    assert setup._SNAPSHOTS is None
