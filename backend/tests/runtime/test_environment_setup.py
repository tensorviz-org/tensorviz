"""Additive installation contracts; local wheel/source fixtures, no network/Torch download."""
import json
import io
import re
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace
import zipfile

import pytest

from services.ide import environment_setup as setup


def test_model_import_failure_retains_native_cause_and_observed_platform(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(setup, 'TARGET_PYTHON', None)
    monkeypatch.setattr(setup.platform, 'machine', lambda: 'arm64')
    monkeypatch.setattr(setup.sys, 'platform', 'darwin')
    monkeypatch.setattr(setup.metadata, 'version', lambda name: '2.7.1')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(version=SimpleNamespace(cuda=None),
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True))))
    model = tmp_path / 'native_failure_fixture.py'
    model.write_text('try:\n    raise OSError("dlopen: library libcudart.so was not found")\n'
                     'except OSError as cause:\n    raise ImportError("flash_attn native extension could not load") from cause\n')
    result = setup.probe_model_import(tmp_path, model)
    assert result['ok'] is False
    assert result['code'] == 'MODEL_IMPORT_FAILED'
    assert 'libcudart.so' in result['message']
    diagnostic = result['diagnostic']
    assert diagnostic['exceptionType'] == 'OSError'
    assert [item['exceptionType'] for item in diagnostic['causes']] == ['ExecutionRequirement', 'ImportError', 'OSError']
    assert 'flash_attn native extension' in diagnostic['traceback']
    assert str(model) in diagnostic['traceback']
    assert diagnostic['runtime'] == dict(platform='darwin', architecture='arm64', python=sys.version.split()[0],
        torch='2.7.1', cuda=None, cudaAvailable=False, mpsAvailable=True)
    assert not diagnostic['truncated']


def test_model_import_diagnostic_reports_missing_module_without_importing_torch(monkeypatch):
    import builtins
    original_import = builtins.__import__
    monkeypatch.delitem(sys.modules, 'torch', raising=False)
    def no_torch(name, *args, **kwargs):
        assert name != 'torch', 'diagnostics must not start another import'
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_torch)
    monkeypatch.setattr(setup.metadata, 'version', lambda _: '2.7.1')
    diagnostic = setup.model_import_diagnostic(ModuleNotFoundError("No module named 'flash_attn_2_cuda'", name='flash_attn_2_cuda'))
    assert diagnostic['causes'][0]['module'] == 'flash_attn_2_cuda'
    assert diagnostic['runtime']['cudaAvailable'] is None
    assert diagnostic['runtime']['mpsAvailable'] is None
    assert diagnostic['runtime']['torch'] == '2.7.1'


def test_model_import_diagnostic_redacts_credentials_and_omits_source_and_locals(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(setup, 'TARGET_PYTHON', None)
    monkeypatch.delitem(sys.modules, 'torch', raising=False)
    model = tmp_path / 'private_failure_fixture.py'
    model.write_text('private_literal = "DO_NOT_EXPOSE_LOCAL"\n'
                     'raise ImportError("fetch failed: https://person:private_password@example.org/pkg?token=private_token&api_key=private_key")\n')
    diagnostic = setup.probe_model_import(tmp_path, model)['diagnostic']
    encoded = json.dumps(diagnostic)
    for private in ['DO_NOT_EXPOSE_LOCAL', 'private_literal', 'private_password', 'private_token', 'private_key']:
        assert private not in encoded
    assert '[redacted]' in diagnostic['message']
    assert 'example.org/pkg' in diagnostic['message']
    assert 'line 2' in diagnostic['traceback']


def test_model_import_diagnostic_bounds_messages_and_exception_cycles():
    error = ValueError('x' * 20000)
    error.__cause__ = error
    diagnostic = setup.model_import_diagnostic(error)
    assert diagnostic['truncated']
    assert len(diagnostic['message']) == 2048
    assert len(diagnostic['traceback']) <= 12000
    assert len(diagnostic['causes']) == 1


def test_model_import_diagnostic_bounds_unicode_for_the_browser_contract():
    diagnostic = setup.model_import_diagnostic(ImportError('\U0001f642' * 2048))
    assert len(diagnostic['message'].encode('utf-16-le')) == 4096
    assert diagnostic['truncated']


def test_model_import_diagnostic_bounds_escaped_json_for_the_helper_transport():
    error = ImportError('\u4e2d' * 2048)
    for _ in range(7):
        outer = ImportError('\u4e2d' * 2048)
        outer.__cause__ = error
        error = outer
    diagnostic = setup.model_import_diagnostic(error)
    assert len(json.dumps(diagnostic)) <= 64000
    assert len(diagnostic['causes']) == 8
    assert diagnostic['message'] == diagnostic['causes'][-1]['message']
    assert diagnostic['truncated']


def test_model_import_diagnostic_bounds_traceback_frames_without_source_or_locals():
    def nested(depth):
        private_literal = 'DO_NOT_EXPOSE_LOCAL'
        if depth:
            return nested(depth - 1)
        raise ImportError('Native extension unavailable.')
    try:
        nested(80)
    except ImportError as error:
        diagnostic = setup.model_import_diagnostic(error)
    assert diagnostic['traceback'].count('  File ') == 64
    assert diagnostic['truncated']
    assert 'DO_NOT_EXPOSE_LOCAL' not in diagnostic['traceback']
    assert 'private_literal' not in diagnostic['traceback']


def test_target_model_probe_preserves_returned_diagnostic(tmp_path, monkeypatch):
    result = dict(ok=False, code='MODEL_IMPORT_FAILED', message='Missing native module.',
                  diagnostic=setup.model_import_diagnostic(ImportError('Native extension missing.')))
    monkeypatch.setattr(setup, 'TARGET_PYTHON', '/target/python')
    calls = []
    def run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return SimpleNamespace(stdout='model progress output\n' + json.dumps(result) + '\n')
    monkeypatch.setattr(setup.subprocess, 'run', run)
    assert setup.probe_model_import(tmp_path, tmp_path / 'model.py') == result
    assert calls[0][0][:3] == ['/target/python', '-I', '-B']
    assert calls[0][1]['timeout'] == 90


def test_fresh_model_recipe_preserves_pinned_git_and_skips_optional_imports(tmp_path, monkeypatch):
    revision = 'a' * 40
    (tmp_path / 'pyproject.toml').write_text('[project]\ndependencies = ["torch_remat @ git+https://github.com/example/remat.git@' + revision + '", "datasets>=3"]\n')
    monkeypatch.setattr(setup, 'model_dependencies', lambda *_: {'missingModules': [
        {'module': 'torch_remat', 'requirement': None}, {'module': 'datasets', 'requirement': 'datasets>=3', 'optional': True}]})
    recipe = setup.model_environment_requirements(tmp_path, tmp_path / 'model.py')
    assert [item.replace(' ', '') for item in recipe['sourcePackages']] == [
        'torch_remat@git+https://github.com/example/remat.git@' + revision]
    assert 'torch' in recipe['requirements']
    assert not any('datasets' in requirement for requirement in recipe['requirements'])
    assert setup.project_requirements(tmp_path)['torch-remat'] is None, 'ordinary additive installs still reject sources'


@pytest.mark.parametrize('source', ['git+https://github.com/example/remat.git@main',
                                   'git+https://secret@github.com/example/remat.git@' + 'a' * 40,
                                   'file:///tmp/remat.whl'])
def test_fresh_model_recipe_refuses_unpinned_private_or_local_sources(tmp_path, monkeypatch, source):
    (tmp_path / 'requirements.txt').write_text('helper @ ' + source)
    monkeypatch.setattr(setup, 'model_dependencies', lambda *_: {'missingModules': [{'module': 'helper', 'requirement': None}]})
    with pytest.raises(setup.SetupError, match='public HTTPS Git'):
        setup.model_environment_requirements(tmp_path, tmp_path / 'model.py')


def test_new_model_planner_never_targets_an_existing_service_or_project_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, 'identity', lambda: {'prefix': str(tmp_path / 'environment'), 'packages': {'torch': '2.7.1'}})
    monkeypatch.setattr(setup, 'pip_run', lambda _: pytest.fail('protected environment must not reach pip'))
    with pytest.raises(setup.SetupError, match='fresh model environment'):
        setup.plan_model_environment(tmp_path, tmp_path, tmp_path / 'model.py', setup.project_manifest_digest(tmp_path))


def test_new_model_plan_freezes_project_sources_and_refuses_changed_declarations(tmp_path, monkeypatch):
    project = tmp_path / 'project'; project.mkdir(); manifest = project / 'requirements.txt'; manifest.write_text('helper==1')
    monkeypatch.setattr(setup, 'identity', lambda: {'prefix': str(tmp_path / 'environment'), 'packages': {}})
    monkeypatch.setattr(setup, 'model_environment_requirements', lambda *_: {'requirements': ['helper==1'], 'sourcePackages': []})
    def freeze(directory, requirements, **_):
        (directory / 'plan.json').write_text(json.dumps({'baseline': setup.identity(), 'wheels': []}))
        return {'digest': 'old', 'packages': [], 'builtFromSource': []}
    monkeypatch.setattr(setup, 'plan', freeze)
    resolved = setup.plan_model_environment(tmp_path, project, project / 'model.py', setup.project_manifest_digest(project))
    saved = json.loads((tmp_path / 'plan.json').read_bytes())
    assert saved['modelProject'] == str(project)
    assert resolved['digest'] == setup.digest((tmp_path / 'plan.json').read_bytes())
    manifest.write_text('helper==2')
    monkeypatch.setattr(setup, 'pip_run', lambda _: pytest.fail('changed project must not install'))
    with pytest.raises(setup.SetupError, match='Project requirements changed after review'):
        setup.install(tmp_path, resolved['digest'])


def test_named_model_planning_requires_the_exact_fresh_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, 'identity', lambda: {'prefix': str(tmp_path / 'kimi-k3'), 'packages': {}})
    monkeypatch.setattr(setup, 'model_environment_requirements', lambda *_: {'requirements': ['helper==1'], 'sourcePackages': []})
    def freeze(directory, requirements, **_):
        (directory / 'plan.json').write_text('{}')
        return {'packages': [], 'builtFromSource': []}
    monkeypatch.setattr(setup, 'plan', freeze)
    setup.plan_model_environment(tmp_path, tmp_path, tmp_path / 'model.py', setup.project_manifest_digest(tmp_path), environment_name='kimi-k3')
    for name in ('other', '../kimi-k3', 'CON'):
        with pytest.raises(setup.SetupError, match='fresh model environment'):
            setup.plan_model_environment(tmp_path, tmp_path, tmp_path / 'model.py', setup.project_manifest_digest(tmp_path), environment_name=name)


def test_source_build_availability_reads_metadata_and_names_the_unavailable_requirement(monkeypatch):
    def missing(arguments):
        assert arguments == ['index', 'versions', 'triton'], 'do not install or execute source to check availability'
        raise setup.SetupError('No distribution.', 'PACKAGE_UNAVAILABLE', 'triton')
    monkeypatch.setattr(setup, 'pip_run', missing)
    with pytest.raises(setup.SetupError, match='compatible environment') as failure:
        setup.check_package_availability('triton>=3.5.0')
    assert failure.value.requirement == 'triton>=3.5.0'
    assert failure.value.code == 'PACKAGE_UNAVAILABLE'
    monkeypatch.setattr(setup, 'pip_run', lambda _: pytest.fail('direct URLs must not reach the index check'))
    with pytest.raises(setup.SetupError, match='indexed package'):
        setup.check_package_availability('helper @ https://example.org/helper.tar.gz')


def test_pinned_git_model_build_is_not_executed_before_source_approval(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, 'identity', lambda: {'prefix': str(tmp_path / 'environment'), 'packages': {}})
    monkeypatch.setattr(setup, 'model_environment_requirements', lambda *_: {'requirements': ['helper @ git+https://example.org/helper.git@' + 'a' * 40],
                                                                         'sourcePackages': ['helper @ git+https://example.org/helper.git@' + 'a' * 40]})
    monkeypatch.setattr(setup, 'pip_run', lambda _: pytest.fail('source build needs explicit approval'))
    with pytest.raises(setup.SetupError, match='Review source execution first'):
        setup.plan_model_environment(tmp_path, tmp_path, tmp_path / 'model.py', setup.project_manifest_digest(tmp_path))


def test_changed_git_pin_cannot_execute_after_reviewing_a_different_project_declaration(tmp_path, monkeypatch):
    manifest = tmp_path / 'requirements.txt'
    manifest.write_text('helper @ git+https://example.org/helper.git@' + 'a' * 40)
    reviewed = setup.project_manifest_digest(tmp_path)
    manifest.write_text('helper @ git+https://example.org/helper.git@' + 'b' * 40)
    monkeypatch.setattr(setup, 'identity', lambda: {'prefix': str(tmp_path / 'environment'), 'packages': {}})
    monkeypatch.setattr(setup, 'pip_run', lambda _: pytest.fail('a changed Git pin must not execute build code'))
    with pytest.raises(setup.SetupError, match='changed after source review'):
        setup.plan_model_environment(tmp_path, tmp_path, tmp_path / 'model.py', reviewed, allow_source=True)


def test_backend_probe_is_supplied_by_the_host(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["environment_setup.py", "probe"])
    with pytest.raises(SystemExit):
        setup.main()
    assert json.loads(capsys.readouterr().out)["message"] == "Unknown environment setup action."
    setup.main(probe_backend=lambda: {"torch": "test-runtime", "cuda": None})
    assert json.loads(capsys.readouterr().out) == {"ok": True, "torch": "test-runtime", "cuda": None}


def test_cli_keeps_sanitized_unavailable_requirement_for_creation_recovery(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['environment_setup.py', 'inspect'])
    def fail(*_):
        raise setup.SetupError('No distribution satisfies these requirements.', 'PACKAGE_UNAVAILABLE', 'helper==1')
    monkeypatch.setattr(setup, 'inspect_environment', fail)
    with pytest.raises(SystemExit):
        setup.main()
    result = json.loads(capsys.readouterr().out)
    assert result['requirement'] == 'helper==1'
    assert result['code'] == 'PACKAGE_UNAVAILABLE'


def wheel(directory, name, version, requires=()):
    path = directory / f"{name}-{version}-py3-none-any.whl"
    info = f"{name}-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{info}/METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n" +
                         "".join(f"Requires-Dist: {item}\n" for item in requires))
        archive.writestr(f"{info}/WHEEL", "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr(f"{name}.py", f"__version__ = '{version}'\n")
        archive.writestr(f"{info}/RECORD", "")
    return path


def source_package(directory, marker):
    name, version = 'tv_source_helper', '1.0'
    backend = f'''import pathlib, zipfile
def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    pathlib.Path({str(marker)!r}).write_text('build ran')
    filename = '{name}-{version}-py3-none-any.whl'
    with zipfile.ZipFile(pathlib.Path(wheel_directory) / filename, 'w') as archive:
        archive.writestr('{name}-{version}.dist-info/METADATA', 'Metadata-Version: 2.1\\nName: {name}\\nVersion: {version}\\nRequires-Dist: torch>=2\\n')
        archive.writestr('{name}-{version}.dist-info/WHEEL', 'Wheel-Version: 1.0\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n')
        archive.writestr('{name}-{version}.dist-info/RECORD', '')
        archive.writestr('{name}.py', 'VALUE = 1\\n')
    return filename
'''
    target = directory / f'{name}-{version}.tar.gz'
    files = {'build_backend.py': backend, 'pyproject.toml':
             '[build-system]\nrequires = []\nbuild-backend = "build_backend"\nbackend-path = ["."]\n'}
    with tarfile.open(target, 'w:gz') as archive:
        for filename, content in files.items():
            data = content.encode(); info = tarfile.TarInfo(f'{name}-{version}/{filename}'); info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return target


@pytest.fixture
def environment(tmp_path):
    subprocess.run([sys.executable, "-m", "venv", str(tmp_path / "venv")], check=True)
    python = tmp_path / "venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    torch = wheel(wheels, "torch", "2.7.1+cu128")
    subprocess.run([str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(torch)],
                   check=True, capture_output=True)
    return python, wheels


def run_helper(python, code, tmp_path, requirements, monkeypatch):
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    monkeypatch.setenv("PIP_FIND_LINKS", str(tmp_path / "wheels"))
    script = ("import importlib.util, pathlib, json; "
              f"spec=importlib.util.spec_from_file_location('setup', {str(Path(setup.__file__))!r}); "
              "s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s); "
              f"directory=pathlib.Path({str(tmp_path)!r}); requirements=pathlib.Path({str(requirements)!r}); " + code)
    return subprocess.run([str(python), "-c", script], capture_output=True, text=True)


def test_local_install_adds_only_reviewed_missing_packages_and_preserves_cuda(environment, tmp_path, monkeypatch):
    python, wheels = environment
    helper = wheel(wheels, "tv_helper", "1.0", ["torch>=2", "tv_dependency>=1"])
    dependency = wheel(wheels, "tv_dependency", "1.0")
    req = tmp_path / "requirements.txt"
    req.write_text(f"torch\n{helper.as_uri()}\n{dependency.as_uri()}\n")
    result = run_helper(python, "before=s.identity(); p=s.plan(directory, requirements); "
                        "assert [w['name'] for w in p['packages']] == ['tv-helper','tv-dependency']; "
                        "assert s.identity()==before; "
                        "s.install(directory, p['digest']); "
                        "assert s.inventory()['torch']=='2.7.1+cu128'; "
                        "assert s.inventory()['tv-dependency']=='1.0'", tmp_path, req, monkeypatch)
    assert result.returncode == 0, result.stderr


@pytest.fixture
def pipless_environment(environment, tmp_path):
    manager, wheels = environment
    subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(tmp_path / 'pipless')], check=True)
    target = tmp_path / 'pipless' / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    torch = wheel(wheels, 'torch', '2.6.0+cu124')
    subprocess.run([str(manager), '-m', 'pip', '--python', str(target), 'install', '--no-index', '--no-deps', str(torch)],
                   check=True, capture_output=True)
    return manager, target, wheels


def test_external_manager_adds_hashed_wheels_without_adding_pip_or_replacing_target_cuda(pipless_environment, tmp_path, monkeypatch):
    manager, target, wheels = pipless_environment
    helper = wheel(wheels, 'tv_helper', '1.0', ['torch>=2'])
    req = tmp_path / 'requirements.txt'; req.write_text(f'{helper.as_uri()}\n')
    result = run_helper(manager, f"""
manager_before = s.identity()
s.TARGET_PYTHON = {str(target)!r}
before = s.identity()
assert before['packages'] == {{'torch': '2.6.0+cu124'}}
p = s.plan(directory, requirements)
assert p['packages'] == [{{'name': 'tv-helper', 'version': '1.0'}}]
assert s.identity() == before, 'Planning must not install into the target'
assert json.loads((directory / 'plan.json').read_text())['baseline'] == before
s.install(directory, p['digest'])
assert s.inventory() == {{'torch': '2.6.0+cu124', 'tv-helper': '1.0'}}
assert s.environment_info()['pip'] is None
assert s.environment_info()['executable'] == {str(target)!r}
s.TARGET_PYTHON = None
assert s.identity() == manager_before, 'The installer environment must not be modified'
""", tmp_path, req, monkeypatch)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('change', ['different_target', 'venv_config', 'requirements'])
def test_external_review_cannot_be_applied_to_a_changed_target_or_request(pipless_environment, tmp_path, monkeypatch, change):
    manager, target, wheels = pipless_environment
    helper = wheel(wheels, 'tv_helper', '1.0')
    req = tmp_path / 'requirements.txt'; req.write_text(f'{helper.as_uri()}\n')
    if change == 'different_target':
        subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(tmp_path / 'other')], check=True)
    change_code = {
        'different_target': f"s.TARGET_PYTHON = {str(tmp_path / 'other' / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python'))!r}",
        'venv_config': f"cfg = pathlib.Path({str(target.parent.parent / 'pyvenv.cfg')!r}); cfg.write_text(cfg.read_text() + '\\n# changed after review\\n')",
        'requirements': "requirements.write_text('torch>=3')",
    }[change]
    result = run_helper(manager, f"""
s.TARGET_PYTHON = {str(target)!r}
p = s.plan(directory, requirements)
{change_code}
try:
    s.install(directory, p['digest'])
    raise AssertionError('Changed target/request must require a fresh review')
except s.SetupError as error:
    assert 'changed' in str(error)
s.TARGET_PYTHON = {str(target)!r}
assert s.inventory() == {{'torch': '2.6.0+cu124'}}
""", tmp_path, req, monkeypatch)
    assert result.returncode == 0, result.stderr


def test_dependency_presence_and_versions_come_from_target_not_manager(pipless_environment, tmp_path, monkeypatch):
    manager, target, wheels = pipless_environment
    manager_only = wheel(wheels, 'tv_manager_only', '1.0')
    target_only = wheel(wheels, 'tv_target_only', '1.0')
    subprocess.run([str(manager), '-m', 'pip', 'install', '--no-index', '--no-deps', str(manager_only)], check=True, capture_output=True)
    subprocess.run([str(manager), '-m', 'pip', '--python', str(target), 'install', '--no-index', '--no-deps', str(target_only)], check=True, capture_output=True)
    req = tmp_path / 'requirements.txt'; req.write_text('torch<2.7\ntv-manager-only>=1\n')
    (tmp_path / 'model.py').write_text('import torch\nimport tv_manager_only\nimport tv_target_only\nimport torchmetrics\n')
    result = run_helper(manager, f"""
s.TARGET_PYTHON = {str(target)!r}
report = s.model_dependencies(directory, directory / 'model.py')
assert report['incompatible'] == [], 'Manager Torch 2.7 must not override project Torch 2.6'
assert report['missingModules'] == [{{'module': 'torchmetrics', 'requirement': 'torchmetrics'}},
                                   {{'module': 'tv_manager_only', 'requirement': 'tv-manager-only>=1'}}]
assert report['environment']['executable'] == {str(target)!r}
assert report['environment']['pip'] is None
assert s.inspect_environment(requirements)['missing'] == ['tv-manager-only>=1']
""", tmp_path, req, monkeypatch)
    assert result.returncode == 0, result.stderr


def test_external_target_rejects_a_transitive_torch_replacement(pipless_environment, tmp_path, monkeypatch):
    manager, target, wheels = pipless_environment
    helper = wheel(wheels, 'tv_helper', '1.0', ['torch>=3'])
    newer = wheel(wheels, 'torch', '3.0')
    req = tmp_path / 'requirements.txt'; req.write_text(f'{helper.as_uri()}\n{newer.as_uri()}\n')
    result = run_helper(manager, f"""
s.TARGET_PYTHON = {str(target)!r}
before = s.identity()
try:
    s.plan(directory, requirements)
    raise AssertionError('Target Torch replacement must not resolve')
except s.SetupError as error:
    assert error.code == 'DEPENDENCY_CONFLICT'
assert s.identity() == before
assert not (directory / 'plan.json').exists()
""", tmp_path, req, monkeypatch)
    assert result.returncode == 0, result.stderr


def test_cli_capture_inspection_needs_no_server_libraries_or_target_pip(pipless_environment):
    manager, target, _ = pipless_environment
    result = subprocess.run([str(manager), str(Path(setup.__file__)), '--target-python', str(target), 'inspect-capture'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['ok'] is True
    assert report['environment']['executable'] == str(target)
    assert report['torch'] == '2.6.0+cu124'
    assert report['pip'] is False
    assert report['missing'] == ['pydantic>=2.12,<3', 'packaging', 'numpy'] + (
        ['tomli>=2.0; python_version < "3.11"'] if sys.version_info < (3, 11) else [])
    assert report['incompatible'] == []
    assert re.fullmatch('[a-f0-9]{64}', report['fingerprint'])


@pytest.mark.parametrize('change', ['packages', 'venv_config'])
def test_capture_metadata_fingerprint_changes_with_the_target_environment(pipless_environment, change):
    manager, target, wheels = pipless_environment

    def inspect():
        result = subprocess.run([str(manager), str(Path(setup.__file__)), '--target-python', str(target), 'inspect-capture'],
                                capture_output=True, text=True, check=True)
        return json.loads(result.stdout)

    before = inspect()
    assert inspect()['fingerprint'] == before['fingerprint']
    if change == 'packages':
        added = wheel(wheels, 'tv_new_library', '1.0')
        subprocess.run([str(manager), '-m', 'pip', '--python', str(target), 'install', '--no-index', '--no-deps', str(added)],
                       capture_output=True, text=True, check=True)
    else:
        config = target.parent.parent / 'pyvenv.cfg'
        config.write_text(config.read_text() + '\n# environment changed\n')
    after = inspect()
    assert after['fingerprint'] != before['fingerprint']
    assert after['torch'] == before['torch']
    assert after['pip'] is False


def test_private_capture_worker_executes_without_importing_server_dependencies():
    backend = Path(setup.__file__).resolve().parents[2]
    bootstrap = backend / 'services' / 'ingest' / 'bootstrap.py'
    model = backend / 'tests' / 'fixtures' / 'ide_repo' / 'models' / 'resid.py'
    script = f"""
import importlib.abc, importlib.util, sys
class NoServerLibraries(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {{'fastapi', 'uvicorn', 'websockets', 'libcst', 'dotenv', 'httpx', 'requests'}}:
            raise AssertionError('Capture imported a tool-server library: ' + fullname)
sys.meta_path.insert(0, NoServerLibraries())
spec = importlib.util.spec_from_file_location('bootstrap', {str(bootstrap)!r})
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
sys.argv = ['worker', '--file', {str(model)!r}, '--cls', 'Net', '--input', '2,8', '--json']
raise SystemExit(module.main())
"""
    result = subprocess.run([sys.executable, '-I', '-B', '-c', script], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    envelope = json.loads(result.stdout)
    assert envelope['error'] is None
    assert envelope['result']['graph']['nodes']


def test_external_manager_repairs_a_pipless_compatibility_copy_not_itself_or_original(pipless_environment, tmp_path, monkeypatch):
    manager, target, wheels = pipless_environment
    original = [wheel(wheels, 'transformers', '5.17.0'), wheel(wheels, 'huggingface_hub', '1.16.1')]
    subprocess.run([str(manager), '-m', 'pip', '--python', str(target), 'install', '--no-index', '--no-deps', *map(str, original)],
                   check=True, capture_output=True)
    wheel(wheels, 'transformers', '4.57.6', ['huggingface-hub<1', 'torch>=2'])
    wheel(wheels, 'huggingface_hub', '0.36.2')
    req = tmp_path / 'requirements.txt'; req.write_text('transformers>=4.38.2\n')
    result = run_helper(manager, f"""
manager_before = s.identity()
s.TARGET_PYTHON = {str(target)!r}
original_before = s.identity()
candidate = s.clone_compatibility_environment(directory / 'candidate', ['transformers==4.57.6'])['interpreter']
s.TARGET_PYTHON = candidate
old = s.pip_run
s.pip_run = lambda args: old(args + ['--no-index', '--find-links', str(directory / 'wheels')])
p = s.plan_compatibility(directory, ['transformers==4.57.6'], directory)
s.install(directory, p['digest'])
assert s.inventory() == {{'torch': '2.6.0+cu124', 'transformers': '4.57.6', 'huggingface-hub': '0.36.2'}}
assert s.environment_info()['pip'] is None
s.TARGET_PYTHON = {str(target)!r}
assert s.identity() == original_before
s.TARGET_PYTHON = None
assert s.identity() == manager_before
""", tmp_path, req, monkeypatch)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('writable,externally_managed', [(False, False), (True, True)])
def test_external_manager_refuses_unsafe_install_targets(monkeypatch, writable, externally_managed):
    monkeypatch.setattr(setup, 'TARGET_PYTHON', '/system/python')
    monkeypatch.setattr(setup, 'target_snapshot', lambda: {
        'identity': {'prefix': '/system'}, 'basePrefix': '/system', 'conda': False,
        'writable': writable, 'externallyManaged': externally_managed})
    with pytest.raises(setup.SetupError, match='writable project virtual environment'):
        setup.check_install_target()


def test_managed_update_uses_cpu_requirements_and_is_additive_and_repeatable(environment, tmp_path, monkeypatch):
    python, wheels = environment
    websocket = wheel(wheels, "websockets", "15.0.1")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(f"torch==2.7.1\n{websocket.as_uri()}\n")
    cpu_requirements = tmp_path / "requirements-cpu.txt"
    cpu_requirements.write_text("--no-index\n-r requirements.txt\n")
    result = run_helper(python, "before=s.inventory(); "
                        "assert s.update_managed(directory, requirements)=={'installed':1}; "
                        "assert s.inventory()==dict(before, websockets='15.0.1'); "
                        "assert s.update_managed(directory, requirements)=={'installed':0}; "
                        "assert s.inventory()==dict(before, websockets='15.0.1')",
                        tmp_path, cpu_requirements, monkeypatch)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("action", ["plan", "update_managed"])
def test_conflicting_dependency_cannot_replace_torch(environment, tmp_path, monkeypatch, action):
    python, wheels = environment
    newer_torch = wheel(wheels, "torch", "3.0")
    helper = wheel(wheels, "tv_helper", "1.0", ["torch>=3"])
    req = tmp_path / "requirements.txt"
    req.write_text(f"{helper.as_uri()}\n{newer_torch.as_uri()}\n")
    result = run_helper(python, f"s.{action}(directory, requirements)", tmp_path, req, monkeypatch)
    assert result.returncode != 0
    assert not (tmp_path / "plan.json").exists()
    result = run_helper(python, "assert s.inventory()['torch']=='2.7.1+cu128'; "
                        "assert 'tv-helper' not in s.inventory()", tmp_path, req, monkeypatch)
    assert result.returncode == 0


def test_replacement_plan_and_unhashed_artifacts_are_refused():
    entry = {"metadata": {"name": "torch", "version": "3.0"},
             "download_info": {"url": "https://example.test/torch.whl", "archive_info": {}}}
    with pytest.raises(ValueError, match="replace"):
        setup.validate_wheels({"install": [entry]}, {"torch": "2.7.1+cu128"})
    with pytest.raises(ValueError, match="verified wheel"):
        setup.validate_wheels({"install": [entry]}, {})


@pytest.mark.parametrize('archive,valid', [
    ({'hash': 'sha256=' + 'a' * 64}, True),
    ({'hashes': {'sha256': 'a' * 64}}, True),
    ({'hash': 'sha256=' + 'A' * 64, 'hashes': {'sha256': 'a' * 64}}, True),
    ({'hash': 'sha256=' + 'a' * 64, 'hashes': {'sha256': 'b' * 64}}, False),
    ({'hash': 'sha256=' + 'a' * 64, 'hashes': {'sha256': ''}}, False),
    ({'hash': 'md5=' + 'a' * 32}, False),
    ({'hash': 'sha256=not-a-checksum'}, False),
    ({}, False),
])
def test_pip_report_hash_formats_keep_artifact_verification(archive, valid):
    entry = {'metadata': {'name': 'tv-helper', 'version': '1.0'},
             'download_info': {'url': 'https://example.test/tv_helper-1.0-py3-none-any.whl',
                               'archive_info': archive}}
    if valid:
        assert setup.validate_wheels({'install': [entry]}, {})[0]['sha256'].lower() == 'a' * 64
    else:
        with pytest.raises(ValueError, match='verified wheel'):
            setup.validate_wheels({'install': [entry]}, {})


def test_changed_environment_or_plan_requires_new_review(tmp_path, monkeypatch):
    baseline = {"packages": {"torch": "2.7.1+cu128"}}
    data = json.dumps({"baseline": baseline, "wheels": []}).encode()
    (tmp_path / "plan.json").write_bytes(data)
    with pytest.raises(ValueError, match="plan changed"):
        setup.install(tmp_path, "not-the-approved-digest")
    monkeypatch.setattr(setup, "identity", lambda: {"packages": {"torch": "2.8"}})
    with pytest.raises(ValueError, match="environment changed"):
        setup.install(tmp_path, setup.digest(data))


def test_inspection_names_incompatible_versions_without_touching_torch(tmp_path, monkeypatch):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("torch\npydantic>=2.12,<3\nlibcst>=1.4,<2\n")
    monkeypatch.setattr(setup, "inventory", lambda: {"torch": "2.5.1+cu121", "pydantic": "1.10", "pip": "26"})
    result = setup.inspect_environment(requirements)
    assert result["torch"] == "2.5.1+cu121"
    assert result["missing"] == ["libcst>=1.4,<2"]
    assert result["incompatible"] == ([f"Python {sys.version.split()[0]} (needs 3.11 or newer)"]
                                      if sys.version_info < (3, 11) else []) + ["pydantic 1.10 (needs <3,>=2.12)"]
    assert result["environment"]["executable"] == sys.executable
    assert result["environment"]["prefix"] == sys.prefix
    assert result["environment"]["torch"] == "2.5.1+cu121"
    assert result["environment"]["pip"] == "26"


def test_environment_inspection_requires_the_dependency_parser(monkeypatch):
    # A fresh managed runtime used to trace successfully but fail model import
    # inspection because packaging was present only in development environments.
    installed = setup.inventory()
    installed.pop("packaging", None)
    monkeypatch.setattr(setup, "inventory", lambda: installed)
    result = setup.inspect_environment()
    assert "packaging" in result["missing"]
    assert result["torch"] == installed.get("torch")


def test_managed_requirements_include_the_project_backend_dependencies():
    # A managed install must support the same live canvas as an existing venv.
    # HTTP-only uvicorn used to pass startup while /local/ws returned 404.
    from packaging.requirements import Requirement
    backend = Path(__file__).parent.resolve().parents[1]
    def requirements(name):
        return [Requirement(line.strip()) for line in (backend / name).read_text().splitlines()
                if line.strip() and not line.lstrip().startswith('#')]
    managed = {setup.canonical(req.name): req for req in requirements('requirements.txt')}
    for required in requirements('requirements-project.txt'):
        assert setup.canonical(required.name) in managed, f'Managed runtime is missing {required.name}'
        for pin in managed[setup.canonical(required.name)].specifier:
            if pin.operator == '==':
                assert required.specifier.contains(pin.version), f'Incompatible managed pin: {required.name}'


def test_runtime_identity_remains_readable_when_full_inventory_is_inconsistent(monkeypatch):
    monkeypatch.setattr(setup, 'inventory', lambda: pytest.fail('Runtime discovery must not need a valid install plan'))
    monkeypatch.setattr(setup.metadata, 'version', lambda name: {'torch': '2.7.1', 'pip': '26'}[name])
    result = setup.environment_info()
    assert result['executable'] == sys.executable
    assert result['prefix'] == sys.prefix
    assert result['torch'] == '2.7.1'


@pytest.mark.parametrize("requirement", ["--index-url https://evil", "../local", "thing @ https://evil/a.whl",
                                      "git+https://evil/repo", "torch\n--upgrade", "thing; os_name=='nt'"])
def test_agent_package_requests_refuse_pip_directives_and_external_code(requirement):
    with pytest.raises(setup.SetupError):
        setup.package_requirements([requirement])


def test_agent_package_requests_preserve_version_constraints():
    assert setup.package_requirements(["einops>=0.8,<1", "transformers[torch]==4.50.0"]) == [
        "einops<1,>=0.8", "transformers[torch]==4.50.0"]


@pytest.mark.parametrize(('stderr', 'code'), [
    ('No matching distribution found for fairscale', 'NO_COMPATIBLE_WHEEL'),
    ('ResolutionImpossible: conflicting dependencies', 'DEPENDENCY_CONFLICT'),
    ('Failed building wheel for helper', 'SOURCE_BUILD_FAILED'),
    ('ProxyError accessing https://secret:password@example.test', 'PACKAGE_INDEX_UNREACHABLE'),
])
def test_pip_failure_reports_a_safe_specific_reason_without_leaking_output(monkeypatch, stderr, code):
    monkeypatch.setattr(setup.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=1, stderr=stderr))
    with pytest.raises(setup.SetupError) as error:
        setup.pip_run(['install', '--only-binary=:all:'])
    assert error.value.code == code
    assert 'secret' not in str(error.value) and 'password' not in str(error.value)
    assert 'Choose another environment' not in str(error.value)


def test_source_build_cannot_change_dependencies_after_resolution(tmp_path, monkeypatch):
    report = {'install': [{'metadata': {'name': 'tv_helper', 'version': '1.0', 'requires_dist': ['torch>=2']},
                          'download_info': {'url': 'https://example.test/tv_helper.tar.gz',
                                            'archive_info': {'hashes': {'sha256': 'a' * 64}}}}]}
    monkeypatch.setattr(setup, 'pip_run', lambda args: wheel(Path(args[args.index('--wheel-dir') + 1]),
                                                          'tv_helper', '1.0', ['torch>=3']))
    with pytest.raises(setup.SetupError, match='dependencies.*changed'):
        setup.build_source_wheels(report, tmp_path, {})


@pytest.mark.parametrize('tamper', [False, True])
def test_reviewed_source_build_freezes_a_wheel_and_preserves_installed_torch(environment, tmp_path, monkeypatch, tamper):
    python, wheels = environment
    marker = tmp_path / 'build-ran'
    source_package(wheels, marker)
    req = tmp_path / 'requirements.txt'; req.write_text('tv_source_helper==1.0\n')
    code = """
old = s.pip_run
s.pip_run = lambda args: old(args + ['--no-index', '--find-links', str(directory / 'wheels')])
before = s.identity()
try:
    s.plan(directory, requirements)
    raise AssertionError('source-only package must not build before approval')
except s.SetupError as error:
    assert error.code == 'NO_COMPATIBLE_WHEEL', str(error)
assert not (directory / 'build-ran').exists()
p = s.plan(directory, requirements, allow_source=True)
assert p['builtFromSource'] == ['tv-source-helper']
assert p['packages'] == [{'name': 'tv-source-helper', 'version': '1.0'}]
assert s.identity() == before
assert (directory / 'build-ran').exists()
"""
    if tamper:
        code += """
artifact = next((directory / 'built').rglob('*.whl'))
artifact.write_bytes(artifact.read_bytes() + b'changed')
try:
    s.install(directory, p['digest'])
    raise AssertionError('changed wheel must not install')
except s.SetupError:
    pass
assert s.identity() == before
"""
    else:
        code += "s.install(directory, p['digest']); assert s.inventory()['tv-source-helper']=='1.0'; assert s.inventory()['torch']=='2.7.1+cu128'"
    result = run_helper(python, code, tmp_path, req, monkeypatch)
    assert result.returncode == 0, result.stderr


def test_static_dependency_report_reads_local_imports_but_never_executes_them(tmp_path, monkeypatch):
    (tmp_path / "model.py").write_text("from layers import Layer\nimport tv_missing\nraise RuntimeError('never execute')\n")
    (tmp_path / "layers.py").write_text("import tv_other\nimport json\n")
    (tmp_path / "requirements.txt").write_text("tv-missing>=1\n-r private.txt\n")
    monkeypatch.setattr(setup.importlib.util, 'find_spec', lambda module: None)
    result = setup.model_dependencies(tmp_path, tmp_path / 'model.py')
    assert result['filesChecked'] == 2
    assert result['missingModules'] == [
        {'module': 'tv_missing', 'requirement': 'tv-missing>=1'},
        {'module': 'tv_other', 'requirement': None}]


def test_dependency_report_does_not_follow_files_outside_workspace(tmp_path):
    outside = tmp_path / 'outside.py'
    outside.write_text('import unknown')
    root = tmp_path / 'workspace'
    root.mkdir()
    (root / 'model.py').symlink_to(outside)
    with pytest.raises(setup.SetupError, match='belong'):
        setup.model_dependencies(root, root / 'model.py')


def test_dependency_report_follows_relative_imports_and_discloses_size_limits(tmp_path, monkeypatch):
    (tmp_path / 'model.py').write_text('from . import layers\nimport oversized\n')
    (tmp_path / 'layers.py').write_text('import missing_relative_dep\n')
    (tmp_path / 'oversized.py').write_text('#' * 262145)
    monkeypatch.setattr(setup.importlib.util, 'find_spec', lambda module: None)
    result = setup.model_dependencies(tmp_path, tmp_path / 'model.py')
    assert result['missingModules'] == [{'module': 'missing_relative_dep', 'requirement': None}]
    assert result['truncated'] is True


def test_model_library_names_use_declarations_then_known_mappings_not_guessed_packages(tmp_path, monkeypatch):
    (tmp_path / 'model.py').write_text('import transformers\nimport PIL\nimport private_model_lib\n')
    (tmp_path / 'pyproject.toml').write_text('[project]\ndependencies = ["transformers>=4.40,<5"]\n')
    monkeypatch.setattr(setup.importlib.util, 'find_spec', lambda module: None)
    report = setup.model_dependencies(tmp_path, tmp_path / 'model.py')
    assert report['missingModules'] == [
        {'module': 'PIL', 'requirement': 'pillow'},
        {'module': 'private_model_lib', 'requirement': None},
        {'module': 'transformers', 'requirement': 'transformers==4.57.6'},
    ]


def test_optional_and_deferred_imports_only_become_required_with_runtime_evidence(tmp_path, monkeypatch):
    (tmp_path / 'model.py').write_text('''import torch
try:
    import optional_layers
except ImportError:
    pass
if False:
    import timm
def forward():
    import einops
''')
    (tmp_path / 'optional_layers.py').write_text('import transformers\n')
    monkeypatch.setattr(setup.importlib.util, 'find_spec', lambda module: None if module != 'torch' else object())
    report = setup.model_dependencies(tmp_path, tmp_path / 'model.py')
    assert all(item.get('optional') for item in report['missingModules'])
    report = setup.model_dependencies(tmp_path, tmp_path / 'model.py', ['einops', 'diffusers'])
    assert [item['module'] for item in report['missingModules'] if not item.get('optional')] == ['diffusers', 'einops']


@pytest.mark.parametrize('requirement', ['transformers<5; python_version >= "3.11"', 'transformers @ https://example.invalid/model.whl'])
def test_unsupported_explicit_requirements_never_fall_back_to_unconstrained_packages(tmp_path, monkeypatch, requirement):
    (tmp_path / 'model.py').write_text('import transformers\n')
    (tmp_path / 'requirements.txt').write_text(requirement)
    monkeypatch.setattr(setup.importlib.util, 'find_spec', lambda module: None)
    assert setup.model_dependencies(tmp_path, tmp_path / 'model.py')['missingModules'] == [
        {'module': 'transformers', 'requirement': None}]


@pytest.mark.parametrize('manifest', ['requirements.txt', 'pyproject.toml'])
def test_missing_git_dependency_is_recognized_as_already_declared_without_installing(tmp_path, monkeypatch, manifest):
    (tmp_path / 'model.py').write_text('from layers import Block\n')
    (tmp_path / 'layers.py').write_text('import torch_remat\nimport unknown_model_library\n')
    requirement = 'torch_remat @ git+https://github.com/meta-pytorch/remat.git@' + 'a' * 40
    (tmp_path / manifest).write_text('[project]\ndependencies = ' + json.dumps([requirement])
                                      if manifest.endswith('.toml') else requirement)
    monkeypatch.setattr(setup.importlib.util, 'find_spec', lambda module: None)
    report = setup.model_dependencies(tmp_path, tmp_path / 'model.py')
    assert report['unsupportedDeclaredModules'] == ['torch_remat']
    assert report['missingModules'] == [
        {'module': 'torch_remat', 'requirement': None},
        {'module': 'unknown_model_library', 'requirement': None}]


def test_dependency_scan_follows_src_packages_submodules_and_sibling_imports(tmp_path, monkeypatch):
    (tmp_path / 'src' / 'model_package').mkdir(parents=True)
    (tmp_path / 'model.py').write_text('from model_package import layers\n')
    (tmp_path / 'src' / 'model_package' / '__init__.py').write_text('import einops\n')
    (tmp_path / 'src' / 'model_package' / 'layers.py').write_text('import helpers\n')
    (tmp_path / 'src' / 'model_package' / 'helpers.py').write_text('import transformers\n')
    monkeypatch.setattr(setup.importlib.util, 'find_spec', lambda module: None)
    report = setup.model_dependencies(tmp_path, tmp_path / 'model.py')
    assert report['filesChecked'] == 4
    assert report['missingModules'] == [
        {'module': 'einops', 'requirement': 'einops'}, {'module': 'transformers', 'requirement': 'transformers==4.57.6'}]


def test_selected_module_checks_parent_package_initializers_before_graph_import(tmp_path, monkeypatch):
    package = tmp_path / 'models'
    package.mkdir()
    (package / 'model.py').write_text('import torch\n')
    (package / '__init__.py').write_text('from .config import Config\n')
    (package / 'config.py').write_text('import transformers\n')
    monkeypatch.setattr(setup.importlib.util, 'find_spec', lambda module: None)
    report = setup.model_dependencies(tmp_path, package / 'model.py')
    assert report['filesChecked'] == 3
    assert {'module': 'transformers', 'requirement': 'transformers==4.57.6'} in report['missingModules']


@pytest.mark.parametrize('constraint,expected', [('transformers>=4.38.2', ['transformers==4.57.6']),
                                              ('Transformers>=4.38.2', ['transformers==4.57.6']), ('transformers>=5', [])])
def test_compatibility_candidates_respect_project_constraints(tmp_path, monkeypatch, constraint, expected):
    (tmp_path / 'model.py').write_text('import transformers\n')
    (tmp_path / 'requirements.txt').write_text(constraint)
    monkeypatch.setattr(setup.importlib.util, 'find_spec', lambda module: object())
    monkeypatch.setattr(setup.metadata, 'version', lambda name: '5.17.0' if name == 'transformers' else '1.0')
    report = setup.model_dependencies(tmp_path, tmp_path / 'model.py')
    assert report['missingModules'] == []
    assert report['compatibilityPackages'] == expected


def test_existing_incompatible_versions_are_not_treated_as_ready(tmp_path, monkeypatch):
    (tmp_path / 'model.py').write_text('import transformers\n')
    (tmp_path / 'requirements.txt').write_text('transformers<5')
    monkeypatch.setattr(setup.importlib.util, 'find_spec', lambda module: object())
    monkeypatch.setattr(setup.metadata, 'version', lambda name: '5.17.0' if name == 'transformers' else '1.0')
    report = setup.model_dependencies(tmp_path, tmp_path / 'model.py')
    assert report['incompatible'] == ['transformers 5.17.0 (needs <5)']
    # The exact declaration, so a host can try installed versions or replace them as declared.
    assert report['incompatibleRequirements'] == ['transformers<5']


def test_replacement_plans_cannot_run_in_the_original_environment(tmp_path):
    with pytest.raises(setup.SetupError, match='disposable'):
        setup.plan_compatibility(tmp_path, ['transformers==4.57.6'], tmp_path)


@pytest.mark.parametrize('case', ['success', 'torch_conflict', 'project_conflict', 'dependent_conflict', 'changed_plan', 'changed_requirements'])
def test_compatibility_copy_resolves_reviewed_versions_without_changing_original(environment, tmp_path, monkeypatch, case):
    python, wheels = environment
    original = [wheel(wheels, 'transformers', '5.17.0'), wheel(wheels, 'huggingface_hub', '1.16.1')]
    if case == 'dependent_conflict':
        original.append(wheel(wheels, 'tv_dependent', '1.0', ['transformers>=5']))
    subprocess.run([str(python), '-m', 'pip', 'install', '--no-index', '--no-deps', *map(str, original)], check=True, capture_output=True)
    wheel(wheels, 'transformers', '4.57.6', ['huggingface-hub<1', 'torch>=3' if case == 'torch_conflict' else 'torch>=2'])
    wheel(wheels, 'huggingface_hub', '0.36.2')
    req = tmp_path / 'requirements.txt'
    req.write_text('transformers>=4.38.2\n' + ('huggingface-hub>=1\n' if case == 'project_conflict' else ''))
    copied = tmp_path / 'candidate'
    original_config = (python.parent.parent / 'pyvenv.cfg').read_text()
    result = run_helper(python, f"before=s.identity(); s.clone_compatibility_environment(pathlib.Path({str(copied)!r}), ['transformers==4.57.6']); assert s.identity()==before", tmp_path, req, monkeypatch)
    assert result.returncode == 0, result.stderr
    assert (python.parent.parent / 'pyvenv.cfg').read_text() == original_config
    candidate = copied / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    code = """
old = s.pip_run
s.pip_run = lambda args: old(args + ['--no-index', '--find-links', str(directory / 'wheels')])
before = s.identity()
"""
    if case in ('torch_conflict', 'project_conflict', 'dependent_conflict'):
        code += """
try:
    s.plan_compatibility(directory, ['transformers==4.57.6'], directory)
    raise AssertionError('conflicting versions must not resolve')
except s.SetupError:
    pass
assert s.identity() == before
assert not (directory / 'plan.json').exists()
"""
    else:
        code += "p = s.plan_compatibility(directory, ['transformers==4.57.6'], directory)\nassert s.identity() == before\n"
        if case == 'success':
            code += """
s.install(directory, p['digest'])
assert s.inventory()['transformers'] == '4.57.6'
assert s.inventory()['huggingface-hub'] == '0.36.2'
assert s.inventory()['torch'] == '2.7.1+cu128'
"""
        else:
            code += ("(directory / 'plan.json').write_text('{}')\n" if case == 'changed_plan'
                     else "requirements.write_text('transformers>=5')\n")
            code += """
try:
    s.install(directory, p['digest'])
    raise AssertionError('changed review must not install')
except s.SetupError:
    pass
assert s.identity() == before
"""
    result = run_helper(candidate, code, tmp_path, req, monkeypatch)
    assert result.returncode == 0, result.stderr
    result = run_helper(python, "assert s.inventory()['transformers']=='5.17.0'; assert s.inventory()['huggingface-hub']=='1.16.1'; assert s.inventory()['torch']=='2.7.1+cu128'", tmp_path, req, monkeypatch)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('modules', [['--index-url'], ['a.b'], ['x'] * 33, 'transformers'])
def test_runtime_dependency_names_are_bounded_identifiers(tmp_path, modules):
    (tmp_path / 'model.py').write_text('')
    with pytest.raises(setup.SetupError, match='module names'):
        setup.model_dependencies(tmp_path, tmp_path / 'model.py', modules)


INDEXED = """
old = s.pip_run
s.pip_run = lambda args: old(args + ['--no-index', '--find-links', str(directory / 'wheels')])
"""


@pytest.mark.parametrize('case', ['success', 'changed_requirements'])
def test_in_place_replacement_changes_pinned_versions_and_their_dependents_after_review(pipless_environment, tmp_path, monkeypatch, case):
    manager, target, wheels = pipless_environment
    vision = wheel(wheels, 'tv_vision', '2.0', ['torch==2.6.0+cu124'])
    subprocess.run([str(manager), '-m', 'pip', '--python', str(target), 'install', '--no-index', '--no-deps', str(vision)],
                   check=True, capture_output=True)
    wheel(wheels, 'torch', '2.0.0')
    wheel(wheels, 'tv_vision', '1.0', ['torch==2.0.0'])
    req = tmp_path / 'requirements.txt'; req.write_text('torch==2.0.0\n')
    result = run_helper(manager, INDEXED + f"""
manager_before = s.identity()
s.TARGET_PYTHON = {str(target)!r}
before = s.identity()
p = s.plan_replacement(directory, ['torch==2.0.0'], directory)
assert p['replacesTorch'] is True
# torchvision-style dependents move with the pin instead of being left broken.
assert sorted(p['packages'], key=lambda w: w['name']) == [
    {{'name': 'torch', 'version': '2.0.0', 'previous': '2.6.0+cu124'}},
    {{'name': 'tv-vision', 'version': '1.0', 'previous': '2.0'}}], p
assert s.identity() == before, 'Planning must not change the environment'
if {case!r} == 'changed_requirements':
    (directory / 'requirements.txt').write_text('torch==2.1.0\\n')
    try:
        s.install(directory, p['digest'])
        raise AssertionError('a changed project pin must require a new review')
    except s.SetupError as error:
        assert 'changed after review' in str(error)
    assert s.identity() == before
else:
    s.install(directory, p['digest'])
    assert s.inventory() == {{'torch': '2.0.0', 'tv-vision': '1.0'}}
s.TARGET_PYTHON = None
assert s.identity() == manager_before, 'The installer environment must not be modified'
""", tmp_path, req, monkeypatch)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('case', ['no_build', 'stuck_dependent'])
def test_in_place_replacement_explains_why_no_plan_exists(pipless_environment, tmp_path, monkeypatch, case):
    manager, target, wheels = pipless_environment
    if case == 'stuck_dependent':
        vision = wheel(wheels, 'tv_vision', '2.0', ['torch==2.6.0+cu124'])
        subprocess.run([str(manager), '-m', 'pip', '--python', str(target), 'install', '--no-index', '--no-deps', str(vision)],
                       check=True, capture_output=True)
        wheel(wheels, 'torch', '2.0.0')
    req = tmp_path / 'requirements.txt'; req.write_text('torch==2.0.0\n')
    result = run_helper(manager, INDEXED + f"""
s.TARGET_PYTHON = {str(target)!r}
before = s.identity()
try:
    s.plan_replacement(directory, ['torch==2.0.0'], directory)
except s.SetupError as error:
    print(json.dumps([error.code, str(error)]))
assert s.identity() == before
""", tmp_path, req, monkeypatch)
    assert result.returncode == 0, result.stderr
    code, message = json.loads(result.stdout)
    if case == 'no_build':
        assert code == 'NO_MATCHING_BUILD'
        assert message.startswith(f'torch 2.0.0 has no build for Python {sys.version_info.major}.{sys.version_info.minor} on '), message
    else:
        assert code == 'DEPENDENCY_CONFLICT'
        assert 'together with the installed tv-vision' in message, message


def test_in_place_replacement_accepts_only_the_projects_own_declarations(tmp_path):
    (tmp_path / 'requirements.txt').write_text('torch==2.0.0\n')
    for requested in (['torch==1.13.0'], ['numpy==1.26.4']):
        with pytest.raises(setup.SetupError, match="project's own requirements"):
            setup.plan_replacement(tmp_path, requested, tmp_path)


def test_unavailable_requirement_is_named_without_forwarding_pip_output(monkeypatch):
    stderr = ('ERROR: Could not find a version that satisfies the requirement torch==2.0.0 (from versions: 2.2.0)\n'
              'ERROR: No matching distribution found for torch==2.0.0\nLooking in https://user:secret@index.example/simple')
    monkeypatch.setattr(setup.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=1, stderr=stderr))
    with pytest.raises(setup.SetupError) as error:
        setup.pip_run(['install', '--only-binary=:all:'])
    assert error.value.requirement == 'torch==2.0.0'
    assert 'secret' not in str(error.value)
