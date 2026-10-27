import hashlib

import pytest

from services.ide.graph_setup import discover
from services.ide.multifile import MultifileProjectionRequest


def inspect_model(source, cls="Net", extra=None, file="model.py"):
    sources = {file: source, **(extra or {})}
    entries = [dict(file=path, source=text, digest=hashlib.sha256(text.encode()).hexdigest())
               for path, text in sources.items()]
    request = MultifileProjectionRequest(protocol_version=1, root=dict(
        file=file, cls=cls, source_digest=entries[0]["digest"]), sources=entries)
    return discover(request)


def test_setup_collects_constructor_and_all_forward_arguments_without_execution(tmp_path):
    marker = tmp_path / "executed"
    result = inspect_model(f'''
from torch import nn
from pathlib import Path
Path({str(marker)!r}).touch()
class Net(nn.Module):
    def __init__(self, channels, classes, bilinear=False): pass
    def forward(self, x, /, context, *, mask, optional=None): pass
''')
    assert not marker.exists()
    assert [field["name"] for field in result["init"]] == ["channels", "classes", "bilinear"]
    assert [field["required"] for field in result["init"]] == [True, True, False]
    assert result["init"][2]["defaultValue"] is False
    assert [(field["name"], field["kind"], field["required"]) for field in result["inputs"]] == [
        ("x", "positional", True), ("context", "keyword", True),
        ("mask", "keyword_only", True), ("optional", "keyword_only", False)]


def test_variadic_forward_exposes_inputs_of_its_direct_delegate_without_guessing_shapes():
    result = inspect_model('''
raise RuntimeError("discovery must not import this file")
from torch import nn
class Net(nn.Module):
    def forward_features(self, x, masks=None): return x
    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        return self.head(ret)
''')
    assert [(f["name"], f["kind"], f["required"]) for f in result["inputs"]] == [
        ("x", "keyword", True), ("masks", "keyword", False), ("is_training", "keyword_only", False)]
    assert result["inputs"][1]["defaultValue"] is None
    assert result["inputs"][2]["defaultValue"] is False
    assert result["constraints"] == []
    assert result["examples"] == []


@pytest.mark.parametrize("change", [
    lambda s: s.replace("ret = self.features(*args, **kwargs)", "if kwargs:\n            return self.features(*args, **kwargs)\n        ret = None"),
    lambda s: s.replace("ret = self.features", "kwargs.pop('optional', None)\n        ret = self.features"),
    lambda s: s.replace("self.features(*args, **kwargs)", "self.features(*args, x=1, **kwargs)"),
    lambda s: s.replace("return ret", "return self.features(*args, **kwargs)"),
    lambda s: s.replace("def features", "@changes_signature\n    def features"),
    lambda s: s.replace("x, mask=None", "x, is_training=False"),
    lambda s: s.replace("x, mask=None", "*values, **options"),
    lambda s: s.replace("def forward", "def __init__(self): self.features = another_method\n    def forward"),
    lambda s: s.replace("def forward", "def __getattribute__(self, name): return other\n    def forward"),
    lambda s: s.replace("self.features(*args, **kwargs)", "self.forward(*args, **kwargs)"),
])
def test_ambiguous_forwarding_keeps_variadic_inputs(change):
    source = '''from torch import nn
class Net(nn.Module):
    def features(self, x, mask=None): return x
    def forward(self, *args, is_training=False, **kwargs):
        ret = self.features(*args, **kwargs)
        return ret
'''
    inputs = inspect_model(change(source))["inputs"]
    assert not any(field["name"] == "x" for field in inputs)
    assert any(field["kind"] == "varargs" for field in inputs)


def test_delegated_positional_only_and_keyword_only_inputs_keep_their_binding_kinds():
    result = inspect_model('''from torch import nn
class Net(nn.Module):
    def features(this, x, /, *, mask): return x
    def forward(self, *args, is_training=False, **kwargs):
        """The docstring does not change the delegated signature."""
        return self.features(*args, **kwargs)
''')
    assert [(field["name"], field["kind"], field["required"]) for field in result["inputs"]] == [
        ("x", "positional", True), ("mask", "keyword_only", True), ("is_training", "keyword_only", False)]


def test_only_tensors_passed_to_the_selected_instance_are_examples():
    result = inspect_model('''
import torch
class Net(torch.nn.Module):
    def __init__(self, width): pass
    def forward(self, x): return x
unrelated = torch.randn(99, 100)
model = Net(width=8)
x = torch.randn(2, 8)
output = model(x)
''')
    full = [example for example in result["examples"] if "inputs" in example]
    assert len(full) == 1
    assert full[0]["init"] == {"args": [], "kwargs": {"width": 8}}
    assert full[0]["inputs"]["args"][0]["shape"] == [2, 8]
    assert full[0]["file"] == "model.py"


@pytest.mark.parametrize('expression', [
    'Net(width=12)(torch.ones(3, 12))',
    'print(Net(width=12)(torch.ones(3, 12)).shape)',
    'output = Net(width=12)(x=torch.ones(3, 12))',
])
def test_inline_model_calls_preserve_their_complete_literal_example(expression):
    from tests.discovery.test_graph_setup_resolution import LINEAR
    result = inspect_model('import torch\n' + LINEAR + '\n' + expression)['resolution']
    assert result['status'] == 'complete', result
    profile = result['draft']['profile']
    assert profile['init']['kwargs'] == {'width': 12}
    tensors = [*profile['inputs']['args'], *profile['inputs']['kwargs'].values()]
    assert tensors[0]['shape'] == [3, 12]
    assert any(e['field'] == 'call' and e['kind'] == 'repository' for e in result['evidence'])


def test_conflicting_inline_calls_remain_a_choice_instead_of_selecting_one():
    from tests.discovery.test_graph_setup_resolution import LINEAR
    result = inspect_model('import torch\n' + LINEAR + '''
print(Net(width=12)(torch.ones(3, 12)))
print(Net(width=16)(torch.ones(4, 16)))
''')['resolution']
    assert result['status'] == 'partial'
    assert any('Several example calls' in reason for reason in result['unresolved'])


@pytest.mark.parametrize('expression', [
    'print(Net(width=12).double()(torch.ones(3, 12)))',
    'print(Net(width=12)(torch.ones(3, 12).reshape(1, 36)))',
    'print(Net(width=12)(external_input))',
])
def test_transformed_or_unknown_inline_values_do_not_become_complete_examples(expression):
    from tests.discovery.test_graph_setup_resolution import LINEAR
    result = inspect_model('import torch\n' + LINEAR + '\n' + expression)
    assert not any('inputs' in example for example in result['examples'])


def test_call_site_provenance_resolves_import_aliases_and_keeps_alternatives():
    result = inspect_model('''
import torch
class Net(torch.nn.Module):
    def __init__(self, width): pass
    def forward(self, x): return x
''', extra={"examples.py": '''
from model import Net as Example
one = Example(width=8)
two = Example(width=32)
'''})
    assert [item["init"]["kwargs"]["width"] for item in result["examples"]] == [8, 32]
    assert all(item["file"] == "examples.py" for item in result["examples"])


def test_reassignment_does_not_bind_an_input_to_an_old_model():
    result = inspect_model('''
import torch
class Net(torch.nn.Module):
    def forward(self, x): return x
model = Net()
model = something_else()
model(torch.randn(7, 9))
''')
    assert result["init"] == []
    assert not any("inputs" in item for item in result["examples"])


def test_object_configuration_and_existing_factory_are_choices():
    result = inspect_model('''
import torch
from dataclasses import dataclass
@dataclass
class GPTConfig:
    width: int = 128
    vocabulary: int = 1000
class Net(torch.nn.Module):
    def __init__(self, config: GPTConfig): pass
    def forward(self, tokens): return tokens
def make_model(size=128):
    return Net(GPTConfig(width=size))
''')
    assert result["configs"] == [dict(file="model.py", cls="GPTConfig", parameter="config",
        construction="call", fields=[
            dict(name="width", kind="keyword", required=False, defaultValue=128, defaultText="128", source=dict(file="model.py", line=6)),
            dict(name="vocabulary", kind="keyword", required=False, defaultValue=1000, defaultText="1000", source=dict(file="model.py", line=7))])]
    assert result["factories"][0]["callable"] == "make_model"
    assert result["factories"][0]["fields"][0]["defaultValue"] == 128


def test_parent_instance_is_offered_instead_of_inventing_context_inputs():
    result = inspect_model('''
import torch
class Net(torch.nn.Module):
    def __init__(self, width): pass
    def forward(self, x, context, *, mask): pass
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Net(64)
''')
    assert result["parents"] == [dict(file="model.py", cls="Model", attribute="attention", line=9)]


def test_inherited_arguments_are_read_from_workspace_source():
    result = inspect_model('''
from base import Base
class Net(Base): pass
''', extra={"base.py": '''
import torch
class Base(torch.nn.Module):
    def __init__(self, width): pass
    def forward(self, x): return x
'''})
    assert result["init"][0]["name"] == "width"
    assert result["inputs"][0]["name"] == "x"


def test_unknown_or_decorated_signatures_remain_unknown():
    result = inspect_model('''
from external import Base
class Net(Base):
    @changes_signature
    def forward(self, x): return x
''')
    assert result["init"] is None
    assert result["inputs"] is None


def test_imported_dataclass_config_retains_fields_and_defaults():
    result = inspect_model('''from torch import nn
from settings import Dimensions as Config
class Net(nn.Module):
    def __init__(self, config: Config): pass
    def forward(self, x): return x
''', extra={"settings.py": '''from dataclasses import dataclass as config_type
@config_type
class Dimensions:
    width: int = 32
    classes: int = 2
'''})
    assert len(result["configs"]) == 1
    option = result["configs"][0]
    assert (option["file"], option["cls"], option["parameter"], option["construction"]) == (
        "settings.py", "Dimensions", "config", "call")
    assert [(f["name"], f["defaultValue"]) for f in option["fields"]] == [("width", 32), ("classes", 2)]


@pytest.mark.parametrize("decorator", ["auto_docstring", "auto_docstring(custom_intro='A model.')"])
def test_documentation_decorators_preserve_model_signatures(decorator):
    result = inspect_model(f'''from torch import nn
from transformers.utils import auto_docstring
@{decorator}
class Net(nn.Module):
    def __init__(self, width=32): pass
    @auto_docstring
    def forward(self, x, *, mask=None): return x
''')
    assert result["init"][0]["defaultValue"] == 32
    assert [f["name"] for f in result["inputs"]] == ["x", "mask"]


def test_diffusers_configuration_decorator_preserves_constructor_signature():
    result = inspect_model('''from torch import nn
from diffusers.configuration_utils import register_to_config
class Net(nn.Module):
    @register_to_config
    def __init__(self, channels=4): pass
    def forward(self, sample, timestep, context): return sample
''')
    assert result["init"][0]["defaultValue"] == 4


@pytest.mark.parametrize("shadow", [
    "def auto_docstring(cls): return other\n",
    "auto_docstring = other\n",
    "if condition:\n    auto_docstring = other\n",
    "from elsewhere import *\n",
])
def test_shadowed_known_decorator_does_not_establish_a_signature(shadow):
    result = inspect_model('''from torch import nn
from transformers.utils import auto_docstring
''' + shadow + '''@auto_docstring
class Net(nn.Module):
    def __init__(self, width=32): pass
    def forward(self, x): return x
''')
    assert result["init"] is None and result["inputs"] is None


def test_class_local_decorator_shadow_stays_unknown():
    result = inspect_model('''from torch import nn
from transformers.utils import auto_docstring
class Net(nn.Module):
    auto_docstring = changes_signature
    @auto_docstring
    def forward(self, x): return x
''')
    assert result["inputs"] is None


def test_relative_library_imports_and_strict_configuration_keep_declared_fields():
    result = inspect_model('''from torch import nn
from ...utils import auto_docstring, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from .configuration_net import NetConfig
@auto_docstring
class Net(nn.Module):
    def __init__(self, config: NetConfig): pass
    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(self, x=None, *, mask=None, **kwargs): return x
''', file='transformers/models/net/modeling_net.py', extra={
        'transformers/models/net/configuration_net.py': '''from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig
@strict
class NetConfig(PreTrainedConfig):
    width: int = 32
'''})
    assert result['init'][0]['name'] == 'config'
    assert [f['name'] for f in result['inputs']] == ['x', 'mask', 'kwargs']
    assert result['configs'][0]['construction'] == 'call'
    assert result['configs'][0]['fields'][0]['defaultValue'] == 32
    assert result['resolution']['status'] == 'partial', 'source metadata is not a validated library setup'


def test_default_expressions_are_never_evaluated():
    result = inspect_model('''
import torch
class Net(torch.nn.Module):
    def __init__(self, config=download_config()): pass
    def forward(self, x): return x
''')
    assert result["init"][0]["required"] is False
    assert "defaultValue" not in result["init"][0]
    assert result["init"][0]["defaultText"] == "download_config()"


def test_invalid_root_or_source_digest_refuses_discovery():
    with pytest.raises(ValueError):
        inspect_model("not valid python")


def test_layer_constraint_does_not_invent_batch_or_spatial_dimensions():
    result = inspect_model('''
from torch import nn
class Net(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc = nn.Linear(width, 4)
    def forward(self, x): return self.fc(x)
''')
    assert result["constraints"] == [dict(input="x", axis=-1, initParameter="width", file="model.py", line=6)]
    assert not result["examples"]


def test_dimensions_after_input_transformation_are_not_input_constraints():
    result = inspect_model('''
from torch import nn
class Net(nn.Module):
    def __init__(self):
        self.fc = nn.Linear(16, 4)
    def forward(self, x):
        x = x.flatten(1)
        return self.fc(x)
''')
    assert result["constraints"] == []


@pytest.mark.parametrize('annotation', ['Net', '"Net"'])
def test_factory_annotation_discovers_indirect_library_builder_without_execution(annotation, tmp_path):
    marker = tmp_path / 'not-imported'
    result = inspect_model(f'''from torch import nn
from pathlib import Path
Path({str(marker)!r}).touch()
class Net(nn.Module):
    def __init__(self, blocks): pass
    def forward(self, x): return x
from torchvision.models._api import register_model
from torchvision.models._utils import handle_legacy_interface
@register_model()
@handle_legacy_interface(weights=("pretrained", None))
def small(*, weights=None, progress=True, **kwargs) -> {annotation}:
    return build([1, 1], weights, progress, **kwargs)
''')
    assert not marker.exists()
    assert [f['callable'] for f in result['factories']] == ['small']
    assert [f['name'] for f in result['factories'][0]['fields']] == ['weights', 'progress', 'kwargs']


@pytest.mark.parametrize('suffix', [
    '\nsmall = other\n', '\ndef small(): return other\n',
    '\nif condition:\n    small = other\n', '\nfrom elsewhere import *\n',
])
def test_rebound_or_ambiguous_factory_is_not_offered(suffix):
    result = inspect_model('''from torch import nn
class Net(nn.Module):
    def forward(self, x): return x
def small() -> Net: return Net()
''' + suffix)
    assert result['factories'] == []


def test_unknown_decorated_factory_signature_is_not_trusted():
    result = inspect_model('''from torch import nn
class Net(nn.Module):
    def forward(self, x): return x
@arbitrary_decorator
def small() -> Net: return Net()
''')
    assert result['factories'] == []


def test_qualified_imported_factory_annotation_resolves_selected_class():
    result = inspect_model('''from torch import nn
class Net(nn.Module):
    def forward(self, x): return x
''', extra={'builders.py': '''import model as models
def small() -> models.Net: return helper()
'''})
    assert [(f['file'], f['callable']) for f in result['factories']] == [('builders.py', 'small')]


def test_declaration_sources_follow_inheritance_and_config_fields_without_importing():
    result = inspect_model("from base import Base\nclass Net(Base): pass\n", extra={"base.py": """raise RuntimeError('never execute')
from torch import nn
from dataclasses import dataclass
@dataclass
class ModelConfig:
    width: int = 8
class Base(nn.Module):
    def __init__(self, config: ModelConfig): pass
    def forward(self, x): return x
"""})
    assert result["init"][0]["source"] == {"file": "base.py", "line": 8}
    assert result["inputs"][0]["source"] == {"file": "base.py", "line": 9}
    assert result["configs"][0]["fields"][0]["source"] == {"file": "base.py", "line": 6}
