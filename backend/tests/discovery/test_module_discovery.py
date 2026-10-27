"""Model discovery must follow bindings, never global class-name coincidences."""
import builtins
import pytest
from services.ingest.module_discovery import EXTERNAL_MODULE_BASES
from services.ingest.repo_scan import scan_repo, scan_source


@pytest.mark.parametrize('source, expected', [
    ('from torch.nn import Module as TorchModule\nclass Model(TorchModule): pass\n', {'Model'}),
    ('import torch.nn as layers\nAlias = layers.Module\nclass Model(Alias): pass\n', {'Model'}),
    ('from torch import nn\nAlias = nn\nclass Model(Alias.Module): pass\n', {'Model'}),
    ('from torch import nn\nclass Model(nn.Sequential): pass\n', {'Model'}),
    ('from torch.nn import Linear as Dense\nclass Model(Dense): pass\n', {'Model'}),
    ('from torch import nn\nclass Base(nn.Module): pass\nAlias = Base\nclass Child(Alias): pass\nclass Leaf(Child): pass\n', {'Base', 'Child', 'Leaf'}),
    ('from torch import nn\nclass Base(nn.Module): pass\nAlias = Base\nclass Base: pass\nclass Model(Alias): pass\n', {'Model'}),
    ('from torch import nn\nclass Base: pass\nAlias = Base\nclass Base(nn.Module): pass\nclass Helper(Alias): pass\n', {'Base'}),
    ('from torch import nn\nclass Base(nn.Module): pass\nAlias = Base\nAlias = object\nclass Helper(Alias): pass\n', {'Base'}),
    ('from torch import nn\nclass Base(nn.Module): pass\nAlias = Base\ndel Alias\nclass Helper(Alias): pass\n', {'Base'}),
    ('from torch import nn\nclass Base(nn.Module): pass\nAlias = Base\nAlias, other = object, 1\nclass Helper(Alias): pass\n', {'Base'}),
    ('from torch import nn\nclass Module: pass\nclass Helper(Module): pass\nclass Real(nn.Module): pass\n', {'Real'}),
    ('from unrelated import Module\nclass Helper(Module): pass\n', set()),
    ('import unrelated as nn\nclass Helper(nn.Module): pass\n', set()),
    ('from torch import nn\nclass Base(nn.Module): pass\nif optional:\n    Alias = Base\nelse:\n    from unavailable import Alias\nclass Model(Alias): pass\n', {'Base', 'Model'}),
    ('from torch import nn\nclass Base(nn.Module): pass\nif optional:\n    from unavailable import Alias\nelse:\n    Alias = Base\nclass Model(Alias): pass\n', {'Base', 'Model'}),
    ('from torch import nn\nclass Base(nn.Module): pass\ndef configure():\n    Alias = Base\nclass Helper(Alias): pass\n', {'Base'}),
    ('from torch import nn\nclass Base(nn.Module): pass\nclass Config:\n    Alias = Base\nclass Helper(Alias): pass\n', {'Base'}),
    ('from torch import nn\nclass Base(nn.Module): pass\nAlias = factory(Base)\nclass Helper(Alias): pass\n', {'Base'}),
    ('from torch import nn\nclass Base(nn.Module): pass\nclass Model(Mixin, Base): pass\n', {'Base', 'Model'}),
    ('from torch import nn\nclass Base(nn.Module): pass\nclass Model(Base[T]): pass\n', {'Base', 'Model'}),
    ('Alias = Other\nOther = Alias\nclass Helper(Alias): pass\n', set()),
])
def test_buffer_and_workspace_bindings_agree(tmp_path, source, expected):
    source = 'raise RuntimeError("source-only discovery")\n' + source
    (tmp_path / 'model.py').write_text(source)
    assert {entry['name'] for entry in scan_source(source)} == expected
    scan = scan_repo(str(tmp_path))
    actual = {entry['name'] for entry in scan.class_occurrences
              if entry.get('is_module', entry['name'] in scan.module_classes)}
    assert actual == expected


@pytest.mark.parametrize('statement, base', [
    ('from transformers import PreTrainedModel', 'PreTrainedModel'),
    ('from transformers.modeling_utils import PreTrainedModel as Parent', 'Parent'),
    ('import transformers as hf', 'hf.PreTrainedModel'),
    ('import transformers.modeling_utils', 'transformers.modeling_utils.PreTrainedModel'),
    ('from diffusers import ModelMixin', 'ModelMixin'),
    ('from diffusers.models.modeling_utils import ModelMixin as Parent', 'Parent'),
    ('import lightning as L', 'L.LightningModule'),
    ('from lightning.pytorch import LightningModule', 'LightningModule'),
    ('from lightning.pytorch.core.module import LightningModule as Parent', 'Parent'),
    ('import pytorch_lightning as pl', 'pl.LightningModule'),
    ('from pytorch_lightning.core.module import LightningModule as Parent', 'Parent'),
    ('from lightning.pytorch.core import LightningModule', 'LightningModule'),
    ('from pytorch_lightning.core import LightningModule', 'LightningModule'),
    ('from transformers import BertPreTrainedModel', 'BertPreTrainedModel'),
    ('from transformers.models.gpt2 import GPT2PreTrainedModel', 'GPT2PreTrainedModel'),
    ('from transformers.models.llama.modeling_llama import LlamaPreTrainedModel as Parent', 'Parent'),
    ('from transformers import Qwen2PreTrainedModel', 'Qwen2PreTrainedModel'),
    ('from transformers import Qwen3PreTrainedModel', 'Qwen3PreTrainedModel'),
    ('from transformers import T5PreTrainedModel', 'T5PreTrainedModel'),
    ('from transformers import ViTPreTrainedModel', 'ViTPreTrainedModel'),
    ('from transformers import WhisperPreTrainedModel', 'WhisperPreTrainedModel'),
    ('from transformers import Wav2Vec2PreTrainedModel', 'Wav2Vec2PreTrainedModel'),
    ('from transformers import Dinov2PreTrainedModel', 'Dinov2PreTrainedModel'),
    ('from diffusers.models.modeling_utils import LegacyModelMixin', 'LegacyModelMixin'),
    ('from diffusers import UNet2DConditionModel', 'UNet2DConditionModel'),
    ('from diffusers.models import UNet2DModel', 'UNet2DModel'),
    ('from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL', 'AutoencoderKL'),
    ('from diffusers import Transformer2DModel', 'Transformer2DModel'),
    ('from diffusers import FluxTransformer2DModel', 'FluxTransformer2DModel'),
    ('from diffusers import SD3Transformer2DModel', 'SD3Transformer2DModel'),
    ('from peft import PeftModel', 'PeftModel'),
    ('from peft import PeftModelForCausalLM', 'PeftModelForCausalLM'),
    ('from peft.peft_model import PeftModelForSeq2SeqLM', 'PeftModelForSeq2SeqLM'),
    ('from timm.models.vision_transformer import VisionTransformer', 'VisionTransformer'),
    ('from timm.models.resnet import ResNet', 'ResNet'),
    ('from torchvision.models import ResNet', 'ResNet'),
    ('from torchvision.models.vision_transformer import VisionTransformer', 'VisionTransformer'),
    ('from torch_geometric.nn import MessagePassing', 'MessagePassing'),
    ('from mmengine.model import BaseModule', 'BaseModule'),
    ('from mmengine.model.base_model import BaseModel', 'BaseModel'),
])
def test_external_pytorch_bases_follow_imports_without_loading_libraries(tmp_path, statement, base):
    source = (statement + '\nraise RuntimeError("discovery must not import libraries or model source")\n'
              'class Mixin: pass\n'
              f'class Model(Mixin, {base}): pass\n'
              'Alias = Model\nclass Child(Alias): pass\n')
    (tmp_path / 'model.py').write_text(source)
    assert {entry['name'] for entry in scan_source(source)} == {'Model', 'Child'}
    (tmp_path / 'exported.py').write_text('from model import Child as Parent\n')
    (tmp_path / 'leaf.py').write_text('from exported import Parent\nclass Leaf(Parent): pass\n')
    assert scan_repo(str(tmp_path)).module_classes == {'Model', 'Child', 'Leaf'}


@pytest.mark.parametrize('qualified', sorted(EXTERNAL_MODULE_BASES))
def test_catalog_paths_preserve_aliases_without_importing_optional_packages(monkeypatch, qualified):
    original_import = builtins.__import__
    external_packages = {path.split('.')[0] for path in EXTERNAL_MODULE_BASES}

    def no_optional_import(name, *args, **kwargs):
        assert name.split('.')[0] not in external_packages, f'Discovery imported {name}'
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', no_optional_import)
    module, _, name = qualified.rpartition('.')
    source = (f'import {module} as library\n'
              f'Alias = library.{name}\n'
              f'library.{name} = object\n'
              'class Kept(Alias): pass\n'
              f'class Rebound(library.{name}): pass\n'
              f'from unrelated import {name} as Other\n'
              'class Unrelated(Other): pass\n')
    assert {entry['name'] for entry in scan_source(source)} == {'Kept'}


@pytest.mark.parametrize('source', [
    'class Helper(PreTrainedModel): pass',
    'from unrelated import PreTrainedModel\nclass Helper(PreTrainedModel): pass',
    'class PreTrainedModel: pass\nclass Helper(PreTrainedModel): pass',
    'from transformers import PreTrainedModel\nPreTrainedModel = object\nclass Helper(PreTrainedModel): pass',
    'import transformers as hf\nhf.PreTrainedModel = object\nclass Helper(hf.PreTrainedModel): pass',
    'from transformers import *\nclass Helper(PreTrainedModel): pass',
    'from transformers import PreTrainedConfig\nclass Helper(PreTrainedConfig): pass',
    'from transformers import TFPreTrainedModel\nclass Helper(TFPreTrainedModel): pass',
    'from transformers import FlaxPreTrainedModel\nclass Helper(FlaxPreTrainedModel): pass',
    'from diffusers import DiffusionPipeline\nclass Helper(DiffusionPipeline): pass',
    'from lightning import LightningDataModule\nclass Helper(LightningDataModule): pass',
    'from transformers import UnknownPreTrainedModel\nclass Helper(UnknownPreTrainedModel): pass',
    'from transformers import AutoModel\nclass Helper(AutoModel): pass',
    'from transformers import Trainer\nclass Helper(Trainer): pass',
    'from diffusers import ConfigMixin\nclass Helper(ConfigMixin): pass',
    'from diffusers import SchedulerMixin\nclass Helper(SchedulerMixin): pass',
    'from peft import PeftConfig\nclass Helper(PeftConfig): pass',
    'from peft import AutoPeftModel\nclass Helper(AutoPeftModel): pass',
    'from timm.models import create_model\nclass Helper(create_model): pass',
    'from torchvision.models import ResNet50_Weights\nclass Helper(ResNet50_Weights): pass',
    'from torch_geometric.data import Data\nclass Helper(Data): pass',
    'from mmengine.runner import Runner\nclass Helper(Runner): pass',
])
def test_external_class_spelling_does_not_imply_a_pytorch_module(source):
    assert scan_source(source) == []


@pytest.mark.parametrize('package, base', [
    ('transformers', 'PreTrainedModel'), ('diffusers', 'ModelMixin'),
    ('lightning', 'LightningModule'), ('pytorch_lightning', 'LightningModule'),
    ('peft', 'PeftModel'), ('timm.models.vision_transformer', 'VisionTransformer'),
    ('torchvision.models', 'ResNet'), ('torch_geometric.nn', 'MessagePassing'),
    ('mmengine.model', 'BaseModule'),
])
@pytest.mark.parametrize('module_base', ['object', 'nn.Module'])
def test_workspace_definitions_override_external_framework_catalog(tmp_path, package, base, module_base):
    local = tmp_path / (package.replace('.', '/') + '.py')
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(f'from torch import nn\nclass {base}({module_base}): pass\n')
    (tmp_path / 'model.py').write_text(f'from {package} import {base}\nclass Model({base}): pass\n')
    assert scan_repo(str(tmp_path)).module_classes == ({base, 'Model'} if module_base == 'nn.Module' else set())


@pytest.mark.parametrize('statement, base', [
    ('from .base import Block', 'Block'),
    ('from .base import Block as Renamed', 'Renamed'),
    ('from pkg.base import Block as Renamed', 'Renamed'),
    ('import pkg.base as blocks', 'blocks.Block'),
    ('import pkg.base', 'pkg.base.Block'),
    ('from pkg import Exported', 'Exported'),
    ('from pkg import Alias', 'Alias'),
    ('from .base import *', 'Block'),
    ('from .base import Block\nAlias = Block', 'Alias'),
])
def test_cross_file_imports_and_reexports(tmp_path, statement, base):
    sources = {
        'pkg/base.py': 'from torch import nn\nclass Block(nn.Module): pass\n',
        'pkg/__init__.py': 'from .base import Block as Exported\nAlias = Exported\n',
        'pkg/child.py': statement + f'\nclass Child({base}): pass\n',
        'other.py': 'class Block: pass\nclass Helper(Block): pass\n',
        'unbound.py': 'class Unbound(Block): pass\n',
    }
    for name, source in sources.items():
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(source)
    scan = scan_repo(str(tmp_path))
    actual = {(entry['file'], entry['name']) for entry in scan.class_occurrences
              if entry.get('is_module', entry['name'] in scan.module_classes)}
    assert actual == {('pkg/base.py', 'Block'), ('pkg/child.py', 'Child')}


def test_torch_module_catalog_contains_only_actual_modules():
    import torch
    from services.ingest.module_discovery import TORCH_MODULE_BASES
    for name in TORCH_MODULE_BASES:
        value = torch
        for segment in name.split('.')[1:]:
            value = getattr(value, segment)
        assert isinstance(value, type) and issubclass(value, torch.nn.Module), name


def test_missing_and_shadowed_bindings_are_not_recovered_by_name(tmp_path):
    sources = {
        'base.py': 'from torch import nn\nclass Base(nn.Module): pass\n',
        'late.py': 'from base import *\nclass Helper(Later): pass\nfrom base import Base as Later\n',
        'mutated.py': 'from torch import nn\nnn.Module = object\nclass Helper(nn.Module): pass\n',
        'private.py': 'class Helper(Unknown.Module): pass\n',
        'cycle_a.py': 'from cycle_b import B\nclass A(B): pass\n',
        'cycle_b.py': 'from cycle_a import A\nclass B(A): pass\n',
    }
    for name, source in sources.items():
        (tmp_path / name).write_text(source)
    scan = scan_repo(str(tmp_path))
    assert {(c['file'], c['name']) for c in scan.class_occurrences if c['is_module']} == {('base.py', 'Base')}


def test_supplied_torch_namespace_wins_over_framework_spelling(tmp_path):
    (tmp_path / 'torch').mkdir()
    (tmp_path / 'torch/nn.py').write_text('class Module: pass\n')
    (tmp_path / 'model.py').write_text('from torch import nn\nclass Helper(nn.Module): pass\n')
    assert not scan_repo(str(tmp_path)).module_classes


def test_cross_file_alias_cycles_and_long_inheritance_terminate(tmp_path):
    (tmp_path / 'a.py').write_text('from b import Alias\n')
    (tmp_path / 'b.py').write_text('from a import Alias\n')
    source = 'from a import Alias\nclass Missing(Alias): pass\nfrom torch import nn\nclass Base(nn.Module): pass\n'
    source += '\n'.join(f'class C{i}({"Base" if i == 0 else "C" + str(i - 1)}): pass' for i in range(1100))
    (tmp_path / 'model.py').write_text(source)
    found = scan_repo(str(tmp_path)).module_classes
    assert len(found) == 1101 and 'C1099' in found and 'Missing' not in found


def test_nested_definitions_do_not_hide_or_become_top_level_targets(tmp_path):
    source = ('from torch import nn\nclass Model(nn.Module): pass\n'
              'def factory():\n    class Model: pass\n    class Inner(nn.Module): pass\n')
    assert [(c['name'], c['line']) for c in scan_source(source)] == [('Model', 2)]
    (tmp_path / 'model.py').write_text(source)
    scan = scan_repo(str(tmp_path))
    assert [(c['name'], c['line']) for c in scan.class_occurrences if c['is_module']] == [('Model', 2)]
