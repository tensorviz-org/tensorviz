import hashlib
import tomllib

import pytest

from services.ide.graph_setup_profiles import prepare_profile, save_profile_text
from services.ide.multifile import MultifileProjectionRequest


SOURCE = '''from torch import nn
class Net(nn.Module):
    def __init__(self, width): pass
    def forward(self, x, *, mask=None): return x
'''


def snapshot(source=SOURCE):
    digest = hashlib.sha256(source.encode()).hexdigest()
    return MultifileProjectionRequest(protocol_version=1, root=dict(file="model.py", cls="Net", source_digest=digest),
        sources=[dict(file="model.py", source=source, digest=digest)])


def profile(**overrides):
    return dict(kind="model", model="model:Net", mode="eval", device="cpu", grad_enabled=False,
        init={"args": [], "kwargs": {"width": 8}},
        inputs={"args": [dict(kind="tensor", shape=[2, 8], dtype="float32", requires_grad=False,
                             generator=dict(kind="normal", mean=0, std=1, seed=0))], "kwargs": {}}, **overrides)


def test_complete_profile_roundtrips_and_preserves_unrelated_text():
    original = '# User notes\nversion = 1\n\n[profiles.existing]\nkind = "model"\nmodel = "other:Net"\nmode = "eval"\ndevice = "cpu"\ngrad_enabled = false\ninit = {args=[],kwargs={}}\ninputs = {args=[],kwargs={}}\n'
    result = prepare_profile(snapshot(), original, profile())
    assert result["ok"]
    assert result["config"].startswith(original.rstrip())
    parsed = tomllib.loads(result["config"])
    assert parsed["profiles"]["existing"] == tomllib.loads(original)["profiles"]["existing"]
    assert parsed["profiles"][result["profile"]]["inputs"]["args"][0]["shape"] == [2, 8]


def test_generated_profile_updates_in_place_without_accumulating_variants():
    first = prepare_profile(snapshot(), "", profile())
    changed = profile()
    changed["inputs"]["args"][0]["shape"] = [4, 8]
    second = prepare_profile(snapshot(), first["config"], changed, first["profile"])
    assert second["profile"] == first["profile"]
    assert list(tomllib.loads(second["config"])["profiles"]) == [first["profile"]]
    assert "[4, 8]" in second["config"]


def test_handwritten_profile_is_preserved_as_new_setup_is_added():
    first = prepare_profile(snapshot(), "", profile())
    original = "\n".join(line for line in first["config"].splitlines() if not line.startswith("#"))
    second = prepare_profile(snapshot(), original, profile(), first["profile"])
    assert second["profile"] == first["profile"] + "_2"
    assert second["config"].startswith(original)


def test_markers_inside_multiline_strings_cannot_replace_other_content():
    first = prepare_profile(snapshot(), "", profile())
    original = first["config"].replace("# tensorViz setup: tensorviz_Net\n", "").replace("# end tensorViz setup: tensorviz_Net\n", "")
    original = original.replace('"width" = 8', '''"width" = """
# tensorViz setup: tensorviz_Net
# end tensorViz setup: tensorviz_Net
"""''')
    result = prepare_profile(snapshot(), original, profile(), first["profile"])
    assert result["profile"] == "tensorviz_Net_2"
    assert result["config"].startswith(original.rstrip())
    assert tomllib.loads(result["config"])["profiles"]["tensorviz_Net"] == tomllib.loads(original)["profiles"]["tensorviz_Net"]


def test_missing_constructor_and_input_fields_are_reported_together():
    pending = profile()
    pending["init"] = {"args": [], "kwargs": {}}
    pending["inputs"] = {"args": [], "kwargs": {}}
    result = prepare_profile(snapshot(), "", pending)
    assert not result["ok"]
    assert {tuple(item["field"]) for item in result["requirements"]} == {
        ("init", "kwargs", "width"), ("inputs", "kwargs", "x")}


@pytest.mark.parametrize('dtype,generator', [
    ('int64', dict(kind='randint', low=0, high=16, seed=0)),
    ('float32', dict(kind='normal', mean=0, std=1, seed=0)),
])
def test_omitted_requires_grad_saves_an_explicit_false_default(dtype, generator):
    pending = profile()
    tensor = pending['inputs']['args'][0]
    tensor.update(dtype=dtype, generator=generator)
    del tensor['requires_grad']
    result = prepare_profile(snapshot(), '', pending)
    assert result['ok'], result
    saved = tomllib.loads(result['config'])['profiles'][result['profile']]
    assert saved['inputs']['args'][0]['requires_grad'] is False
    assert 'requires_grad' not in tensor, 'validation must not mutate the draft'


def test_tagged_tensor_errors_only_describe_the_selected_variant():
    pending = profile()
    pending['inputs'] = dict(args=[], kwargs={'input_ids': dict(
        kind='tensor', dtype='int64', generator=dict(kind='randint', low=0, seed=0))})
    result = prepare_profile(snapshot(), '', pending)
    assert result == dict(ok=False, requirements=[
        dict(field=['inputs', 'input_ids', 'shape'], reason='field required'),
        dict(field=['inputs', 'input_ids', 'generator', 'high'], reason='field required')])


@pytest.mark.parametrize('name', ['input_ids', 'bool', 'str', 'Binding'])
def test_nested_tagged_values_keep_readable_paths_and_user_field_names(name):
    pending = profile()
    pending['inputs'] = dict(args=[], kwargs={name: dict(kind='dict', items={
        'tokens': dict(kind='list', items=[dict(kind='tensor', dtype='float32',
            generator=dict(kind='normal', mean=0, std=1, seed=0))])})})
    result = prepare_profile(snapshot(), '', pending)
    assert result == dict(ok=False, requirements=[dict(
        field=['inputs', name, 'items', 'tokens', 'items', 0, 'shape'], reason='field required')])


def test_float_value_tag_is_not_confused_with_a_scalar_union_branch():
    pending = profile()
    pending['inputs']['args'] = [dict(kind='float')]
    assert prepare_profile(snapshot(), '', pending) == dict(ok=False, requirements=[dict(
        field=['inputs', 0, 'value'], reason='field required')])


@pytest.mark.parametrize('generator,field,reason', [
    ({}, 'kind', 'field required'),
    (dict(kind='unknown'), 'kind', 'unknown kind; choose one of'),
    (dict(kind='normal', mean=0, std=-1.5, seed=0), 'std', 'Input should be greater than 0'),
])
def test_generator_validation_omits_union_labels(generator, field, reason):
    pending = profile()
    pending['inputs']['args'][0]['generator'] = generator
    result = prepare_profile(snapshot(), '', pending)
    assert not result['ok']
    assert len(result['requirements']) == 1
    requirement = result['requirements'][0]
    assert requirement['field'] == ['inputs', 0, 'generator', field]
    assert requirement['reason'].startswith(reason)


def test_invalid_input_factory_does_not_report_binding_alternatives():
    pending = profile()
    pending['inputs'] = dict(factory=dict(file='inputs.py'))
    result = prepare_profile(snapshot(), '', pending)
    assert result == dict(ok=False, requirements=[dict(
        field=['inputs', 'factory', 'callable'], reason='field required')])


def test_draft_errors_are_capped_to_five_readable_requirements():
    pending = profile()
    pending['inputs']['args'] = [dict(kind='tensor', dtype='int64',
        generator=dict(kind='randint', low=0, high=16, seed=0)) for _ in range(12)]
    result = prepare_profile(snapshot(), '', pending)
    assert not result['ok']
    assert len(result['requirements']) == 5
    assert ['.'.join(map(str, item['field'])) + ': ' + item['reason']
            for item in result['requirements']] == [f'inputs.{index}.shape: field required' for index in range(5)]


def test_signature_requirements_also_have_a_five_message_cap():
    source = SOURCE.replace('x, *, mask=None', ', '.join(f'input_{index}' for index in range(12)))
    pending = profile()
    pending['inputs'] = dict(args=[], kwargs={})
    result = prepare_profile(snapshot(source), '', pending)
    assert not result['ok']
    assert len(result['requirements']) == 5
    assert [item['field'][-1] for item in result['requirements']] == [f'input_{index}' for index in range(5)]


def test_model_level_errors_name_setup_and_omit_pydantic_prefixes():
    pending = profile()
    del pending['init']
    assert prepare_profile(snapshot(), '', pending) == dict(ok=False, requirements=[dict(
        field=['setup'], reason='provide exactly one of init or factory')])


@pytest.mark.parametrize("binding", [{"args": [], "kwargs": {}}, {"args": [], "kwargs": {"is_training": False}}])
def test_empty_delegated_input_cannot_be_saved_as_a_complete_profile(binding):
    source = '''from torch import nn
class Net(nn.Module):
    def forward_features(self, x, masks=None): return x
    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        return ret
'''
    pending = profile()
    pending["init"] = {"args": [], "kwargs": {}}
    pending["inputs"] = binding
    result = prepare_profile(snapshot(source), "", pending)
    assert not result["ok"]
    assert result["requirements"][0]["field"] == ["inputs", "kwargs", "x"]
    assert "config" not in result


def test_invalid_existing_config_never_becomes_absence():
    with pytest.raises(ValueError):
        prepare_profile(snapshot(), "version = broken", profile())


def test_profile_cannot_retarget_the_opening_request():
    changed = profile()
    changed["model"] = "other:Different"
    with pytest.raises(ValueError):
        prepare_profile(snapshot(), "", changed)


def test_object_config_produces_model_specific_factory_without_executing_it(tmp_path):
    source = '''from torch import nn
from dataclasses import dataclass
@dataclass
class Config:
    width: int
class Net(nn.Module):
    def __init__(self, config, classes): pass
    def forward(self, x): return x
'''
    pending = profile()
    pending["init"] = {"args": [], "kwargs": {"classes": 4}}
    result = prepare_profile(snapshot(source), "", pending, config_object={
        "index": 0, "parameter": "config", "values": {"width": 16}})
    assert result["ok"]
    assert "config = _Config(width=16)" in result["factory"]["source"]
    assert "return _Model(classes=4, config=config)" in result["factory"]["source"]
    saved = tomllib.loads(result["config"])["profiles"][result["profile"]]
    assert saved["factory"]["file"] == result["factory"]["file"]
    assert "init" not in saved
    compile(result["factory"]["source"], "factory.py", "exec")


def test_object_config_does_not_hide_other_required_model_arguments():
    source = '''from torch import nn
class Config:
    width: int = 8
class Net(nn.Module):
    def __init__(self, config, classes): pass
    def forward(self, x): return x
'''
    pending = profile()
    pending["init"] = {"args": [], "kwargs": {}}
    result = prepare_profile(snapshot(source), "", pending,
        config_object={"index": 0, "parameter": "config", "values": {}})
    assert not result["ok"]
    assert result["requirements"][0]["field"] == ["init", "kwargs", "classes"]


def test_user_strings_in_factory_remain_literal_data():
    source = '''from torch import nn
class Config:
    path: str = ""
class Net(nn.Module):
    def __init__(self, config): pass
    def forward(self, x): return x
'''
    pending = profile()
    pending["init"] = {"args": [], "kwargs": {}}
    result = prepare_profile(snapshot(source), "", pending, config_object={
        "index": 0, "parameter": "config", "values": {"path": "'); __import__('os').system('bad'); #"}})
    import ast
    parsed = ast.parse(result["factory"]["source"])
    assignment = parsed.body[-1].body[1]
    assert isinstance(assignment.value, ast.Constant)
    assert assignment.value.value == "'); __import__('os').system('bad'); #"


def test_saving_a_selected_recipe_never_repeats_automatic_resolution(monkeypatch):
    from services.ide.graph_setup_resolution import Resolver
    def forbidden(_self):
        raise AssertionError('Saving must validate the selected recipe, not rediscover alternatives')
    monkeypatch.setattr(Resolver, 'resolve', forbidden)
    assert prepare_profile(snapshot(), '', profile())['ok']


@pytest.mark.parametrize('construction', ['call', 'attributes'])
def test_saved_generated_config_restores_exact_editable_values(construction):
    from services.ide.graph_setup_profiles import restore_config_profile
    decorator = '@dataclass\n' if construction == 'call' else ''
    source = f'''from torch import nn
from dataclasses import dataclass
{decorator}class Config:
    width: int = 8
class Net(nn.Module):
    def __init__(self, config: Config, classes): pass
    def forward(self, x): return x
'''
    pending = profile(seed=17)
    pending['init'] = {'args': [], 'kwargs': {'classes': 4}}
    selection = {'index': 0, 'parameter': 'config', 'values': {'width': 16}}
    prepared = prepare_profile(snapshot(source), '', pending, config_object=selection)
    saved = tomllib.loads(prepared['config'])['profiles'][prepared['profile']]
    helper = prepared['factory']
    data = snapshot(source).model_dump()
    data['sources'].append(dict(file=helper['file'], source=helper['source'],
                               digest=hashlib.sha256(helper['source'].encode()).hexdigest()))
    request = MultifileProjectionRequest.model_validate(data)
    restored = restore_config_profile(request, saved)
    assert restored is not None, 'reopening a generated factory must retain editable configuration fields'
    assert restored['configObject'] == selection
    assert restored['profile']['init'] == pending['init']
    assert 'factory' not in restored['profile']
    for field in ('inputs', 'seed', 'device', 'mode', 'limits'):
        assert restored['profile'][field] == saved[field]
    restored['configObject']['values']['width'] = 24
    changed = prepare_profile(request, prepared['config'], restored['profile'], prepared['profile'],
                              restored['configObject'])
    assert changed['ok']
    expected = 'config.width = 24' if construction == 'attributes' else 'width=24'
    assert expected in changed['factory']['source']
    assert saved['factory']['file'] == helper['file'], 'restoration must not mutate the saved profile'


@pytest.mark.parametrize('mutation', ['edited_helper', 'renamed_helper', 'missing_helper', 'factory_kwargs', 'other_model', 'other_cwd'])
def test_custom_or_changed_factory_is_not_replaced_by_inferred_config(mutation):
    from services.ide.graph_setup_profiles import restore_config_profile
    source = '''from torch import nn
from dataclasses import dataclass
@dataclass
class Config:
    width: int = 8
class Net(nn.Module):
    def __init__(self, config: Config): pass
    def forward(self, x): return x
'''
    pending = profile()
    pending['init'] = {'args': [], 'kwargs': {}}
    prepared = prepare_profile(snapshot(source), '', pending,
                               config_object={'index': 0, 'parameter': 'config', 'values': {'width': 16}})
    saved = tomllib.loads(prepared['config'])['profiles'][prepared['profile']]
    helper = prepared['factory']
    if mutation == 'edited_helper': helper['source'] += '\nprint("user addition")\n'
    if mutation == 'renamed_helper': saved['factory']['file'] = helper['file'] = 'my_factory.py'
    if mutation == 'factory_kwargs': saved['factory']['kwargs'] = {'width': 24}
    if mutation == 'other_model': saved['model'] = 'other:Net'
    if mutation == 'other_cwd': saved['cwd'] = 'src'
    data = snapshot(source).model_dump()
    if mutation != 'missing_helper':
        data['sources'].append(dict(file=helper['file'], source=helper['source'],
                                    digest=hashlib.sha256(helper['source'].encode()).hexdigest()))
    assert restore_config_profile(MultifileProjectionRequest.model_validate(data), saved) is None


def test_input_factory_saves_and_reopens_without_fabricating_forward_bindings():
    draft = profile()
    draft['inputs'] = {'factory': {'file': 'model.py', 'callable': 'graph_inputs', 'kwargs': {}}}
    source = SOURCE + '\ndef graph_inputs(): return {"x": None}\n'
    first = prepare_profile(snapshot(source), '', draft)
    assert first['ok'], first
    saved = tomllib.loads(first['config'])['profiles'][first['profile']]
    assert saved['inputs'] == draft['inputs']
    second = prepare_profile(snapshot(source), first['config'], saved, first['profile'])
    assert second['ok'] and second['profile'] == first['profile']
