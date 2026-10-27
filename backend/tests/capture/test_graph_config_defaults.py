"""Declared library configs are example recipes, never recovered user settings."""
import json
import tomllib

import pytest
import torch

from services.ide.graph_setup import discover
from services.ide.graph_setup_profiles import prepare_profile
from services.ingest.graph_example import propose_constructor
from services.semantic.projection import project_capture
from tests.capture.test_graph_inherited_defaults import _worker
from tests.discovery.test_graph_setup_profiles import snapshot as sources


class Config:
    def __init__(self, width=7):
        self.width = width


class Configured(torch.nn.Module):
    config_class = Config

    def __init__(self, config, extra=9):
        super().__init__()
        self.config, self.extra = config, extra


def test_declared_config_defaults_fill_only_the_required_config():
    kwargs = propose_constructor(Configured)
    assert set(kwargs) == {'config'}
    assert type(kwargs['config']) is Config and kwargs['config'].width == 7
    assert Configured(**kwargs).extra == 9


def test_optional_config_preserves_its_actual_constructor_default():
    class Optional(Configured):
        def __init__(self, config=None):
            pass
    assert propose_constructor(Optional) == {}


@pytest.mark.parametrize('case', ['no_declaration', 'callable', 'required_config_field', 'other_required', 'positional_config', 'descriptor'])
def test_unsupported_config_contract_is_not_guessed(case):
    class Target(Configured):
        pass
    if case == 'no_declaration':
        Target.config_class = None
    elif case == 'callable':
        Target.config_class = lambda: Config()
    elif case == 'required_config_field':
        class Required:
            def __init__(self, width):
                raise AssertionError('An incomplete config must not be constructed.')
        Target.config_class = Required
    elif case == 'other_required':
        Target.__init__ = lambda self, config, width: None
    elif case == 'positional_config':
        exec('def init(self, config, /): pass', namespace := {})
        Target.__init__ = namespace['init']
    else:
        class Descriptor:
            def __get__(self, instance, owner):
                raise AssertionError('Do not execute config_class descriptors.')
        Target.config_class = Descriptor()
    with pytest.raises((TypeError, ValueError)):
        propose_constructor(Target)


def test_config_default_factory_is_fixed_editable_source_and_round_trips():
    request = sources('from transformers import BertModel\nclass Net(BertModel): pass\n')
    profile = discover(request)['resolution']['draft']['profile']
    prepared = prepare_profile(request, '', profile, config_defaults=True)
    assert prepared['ok'], prepared
    helper = prepared['factory']
    assert 'config = _Model.config_class()' in helper['source']
    assert 'return _Model(config=config)' in helper['source']
    saved = tomllib.loads(prepared['config'])['profiles'][prepared['profile']]
    assert saved['factory']['file'] == helper['file']
    assert 'init' not in saved
    assert '_Model(' not in request.sources[0].source


@pytest.mark.parametrize('change', ['known_init', 'factory', 'config_object', 'source_call', 'truncated'])
def test_config_defaults_never_replace_a_selected_recipe(change):
    source = 'from transformers import BertModel\nclass Net(BertModel): pass\n'
    request = sources(source + ('model = Net(unknown)\n' if change == 'source_call' else ''))
    profile = discover(request)['resolution']['draft']['profile']
    kwargs = dict(config_defaults=True)
    if change == 'known_init':
        profile['init']['kwargs']['config'] = 2
    elif change == 'factory':
        profile.pop('init')
        profile['factory'] = dict(file='factory.py', callable='make_model', kwargs={})
    elif change == 'config_object':
        kwargs['config_object'] = dict(index=0, parameter='config', values={})
    elif change == 'truncated':
        request.truncated = True
    with pytest.raises(ValueError):
        prepare_profile(request, '', profile, **kwargs)


@pytest.mark.parametrize('family', ['BertModel', 'T5Model', 'ViTModel', 'Dinov2Model'])
def test_library_config_defaults_pass_a_saved_real_capture(tmp_path, family):
    pytest.importorskip('transformers')
    source = f'from transformers import {family}\nclass Net({family}): pass\n'
    request = sources(source)
    profile = discover(request)['resolution']['draft']['profile']
    workspace = tmp_path/'workspace'
    workspace.mkdir()
    (workspace/'model.py').write_text(source)
    # Existing no-config execution must still fail unless explicitly authorized.
    code, envelope = _worker(tmp_path, workspace, profile, probe=True)
    assert code != 0 and envelope['result'] is None
    code, envelope = _worker(tmp_path, workspace, profile, probe=True, config_defaults=True)
    assert code == 0, envelope
    proposal = envelope['result']['graphExample']
    assert proposal['configDefaults'] is True and 'graph' not in envelope['result']
    profile['inputs'] = proposal['binding']
    prepared = prepare_profile(request, '', profile, config_defaults=True)
    assert prepared['ok'], prepared
    helper = workspace/prepared['factory']['file']
    helper.parent.mkdir()
    helper.write_text(prepared['factory']['source'])
    (workspace/'tensorviz.toml').write_text(prepared['config'])
    saved = tomllib.loads(prepared['config'])['profiles'][prepared['profile']]
    code, envelope = _worker(tmp_path, workspace, saved, probe=False)
    assert code == 0, envelope
    capture = envelope['result']['diagnostics']['semantic_capture']
    graph = project_capture(capture)
    assert len(graph.nodes) > 100
    assert capture['output'] is not None
    print(json.dumps(dict(family=family, nodes=len(graph.nodes), coverage=capture['status'], inputs=profile['inputs'])))


@pytest.mark.parametrize('override', [
    {'constructorBinding': {'args': [], 'kwargs': {'config': 2}}},
    {'constructorArgs': {'config': 2}},
    {'setup': {'file': 'factory.py', 'callable': 'make_model', 'kwargs': {}}},
])
def test_private_config_preparation_refuses_explicit_construction_before_import(tmp_path, override):
    from services.ingest.execution_setup import prepare_execution
    from services.semantic.execution_plan import profile_execution
    from services.validation.contracts import ModelProfile
    request = sources('from transformers import BertModel\nclass Net(BertModel): pass\n')
    profile = discover(request)['resolution']['draft']['profile']
    spec = {**profile_execution(ModelProfile.model_validate(profile)), **override}
    if 'constructorArgs' in override or 'setup' in override:
        spec.pop('constructorBinding')
    with pytest.raises(ValueError, match='Config preparation cannot replace'):
        prepare_execution(str(tmp_path/'missing.py'), 'Net', str(tmp_path), spec,
                          input_proposal=lambda model: {}, constructor_proposal=propose_constructor)


@pytest.mark.parametrize('candidate', [False, True])
def test_config_defaults_cannot_bypass_normal_or_candidate_capture(tmp_path, candidate):
    # An invalid mode must exit before trying to read or import missing.py.
    import subprocess
    import sys
    from pathlib import Path
    backend = Path(__file__).parent.resolve().parents[1]
    result = subprocess.run([sys.executable, '-B', str(backend/'services/ingest/bootstrap.py'),
        '--file', str(tmp_path/'missing.py'), '--cls', 'Net', '--workspace', str(tmp_path),
        '--prepare-config-defaults', *(['--candidate-execution'] if candidate else []), '--json'],
        cwd=backend, capture_output=True, text=True, timeout=10)
    envelope = json.loads(result.stdout)
    assert result.returncode == 2
    assert envelope['error']['type'] == 'UsageError' and envelope['result'] is None
