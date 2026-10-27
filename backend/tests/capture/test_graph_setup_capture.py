"""Automatically resolved recipes reach the real worker without input search."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

from services.ide.graph_setup import discover
from services.ide.graph_setup_profiles import prepare_profile
from services.semantic.contracts import ResolvedExecutionPlan
from services.semantic.execution_plan import applicability_digest, profile_execution
from services.semantic.projection import project_capture
from services.validation.contracts import RunSpec
from services.validation.host_snapshot import snapshot
from tests.discovery.test_graph_setup_profiles import snapshot as sources
from tests.discovery.test_graph_setup_resolution import LINEAR


@pytest.mark.parametrize('fail', [False, True])
@pytest.mark.parametrize('construction', ['default', 'factory', 'imported_config', 'sequence', 'patch',
                                        'required', 'nested', 'attention', 'required_config',
                                        'recurrent', 'grouped_conv', 'transposed_conv',
                                        'normalized_attention', 'keyword_attention', 'sequential_norm',
                                        'inline_example', 'block_count', 'namespace_config', 'package_config', 'constructor_config'])
def test_resolved_recipe_captures_once_and_preserves_forward_failure(tmp_path, fail, construction):
    factory = construction == 'factory'
    source = LINEAR
    if construction == 'sequence':
        source = source.replace('return self.fc(x)', 'batch, sequence, width = x.shape; return self.fc(x)')
    elif construction == 'patch':
        source = source.replace('width=8', 'width=8, patch_size=14').replace(
            'nn.Linear(width, 4)', 'nn.Conv2d(3, width, patch_size, stride=patch_size)').replace(
            'return self.fc(x)', 'assert x.shape[-1] % 14 == 0; return self.fc(x)')
    elif construction in ('required', 'nested', 'attention', 'required_config', 'recurrent', 'grouped_conv',
                          'transposed_conv', 'normalized_attention', 'keyword_attention', 'sequential_norm'):
        source = source.replace('width=8', 'width')
        if construction == 'nested':
            source = source.replace('width):', 'width, classes):').replace('nn.Linear(width, 4)', 'Block(width, classes)')
            source += '''
class Block(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(incoming, outgoing), nn.ReLU())
    def forward(self, x): return self.layers(x)
'''
        elif construction in ('attention', 'normalized_attention', 'keyword_attention'):
            source = source.replace('width):', 'width, heads):').replace(
                'nn.Linear(width, 4)', 'nn.MultiheadAttention(width, heads, batch_first=True)')
            if construction != 'attention':
                source = source.replace('self.fc =', 'self.norm = nn.LayerNorm(width)\n        self.fc =')
        elif construction == 'sequential_norm':
            source = source.replace('nn.Linear(width, 4)', 'nn.Sequential(nn.LayerNorm(width), nn.GELU(), nn.Linear(width, 4))')
        elif construction == 'recurrent':
            source = source.replace('nn.Linear(width, 4)', 'nn.GRU(width, 8, batch_first=True)')
        elif construction == 'grouped_conv':
            source = source.replace('width):', 'width, classes):').replace(
                'nn.Linear(width, 4)', 'nn.Conv2d(width, classes, 3, groups=3)')
        elif construction == 'transposed_conv':
            source = source.replace('nn.Linear(width, 4)', 'nn.ConvTranspose1d(width, 4, 3)')
    source = source.replace('return self.fc(x)', 'y = self.fc(x); return y' if not fail
                            else 'y = self.fc(x); raise RuntimeError("deliberate failure")')
    if construction == 'attention':
        source = source.replace('self.fc(x)', 'self.fc(x, x, x)[0]')
    elif construction == 'normalized_attention':
        source = source.replace('self.fc(x)', 'self.fc(self.norm(x), self.norm(x), self.norm(x))[0]')
    elif construction == 'keyword_attention':
        source = source.replace('self.fc(x)', 'self.fc(query=self.norm(x), key=self.norm(x), value=self.norm(x))[0]')
    elif construction == 'inline_example':
        source = 'import torch\n' + source + '\nif __name__ == "__main__":\n    print(Net(width=12)(torch.ones(3, 12)).shape)\n'
    if factory:
        source = source.replace('width=8', 'width') + '\ndef make_model():\n    return Net(int("8"))\n'
    config_source = 'from dataclasses import dataclass\n@dataclass\nclass Dimensions:\n    width: int = 8\n'
    if construction in ('imported_config', 'required_config', 'namespace_config', 'package_config', 'constructor_config'):
        source = 'from settings import Dimensions\n' + source.replace('width=8', 'config: Dimensions').replace(
            'nn.Linear(width, 4)', 'nn.Linear(config.width, 4)')
        if construction == 'constructor_config':
            config_source = 'class Dimensions:\n    def __init__(self, hidden_size=12):\n        self.width = hidden_size * 2\n'
        if construction == 'required_config':
            source = source.replace('width):', 'config: Dimensions):')
            config_source = config_source.replace('width: int = 8', 'width: int')
    elif construction == 'block_count':
        config_source = config_source.replace('width: int = 8', 'depth: int\n    width: int = 8')
        source = 'from settings import Dimensions\n' + source.replace('width=8', 'config: Dimensions').replace(
            'nn.Linear(width, 4)', 'nn.Linear(config.width, config.width)').replace(
            'self.fc =', 'self.blocks = nn.ModuleList([nn.Linear(config.width, config.width) for _ in range(config.depth)])\n        self.fc =')
        source = source.replace('return y', 'return self.blocks[0](y)')
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    model_file = 'models/model.py' if construction in ('namespace_config', 'package_config') else 'model.py'
    if model_file.startswith('models/'):
        (workspace / 'models').mkdir()
        source = source.replace('from settings import', 'from .settings import')
    (workspace / model_file).write_text(source)
    request = sources(source)
    request.root.file = model_file
    request.sources[0].file = model_file
    if construction in ('imported_config', 'required_config', 'block_count', 'namespace_config', 'package_config', 'constructor_config'):
        config_file = str(Path(model_file).parent / 'settings.py')
        (workspace / config_file).write_text(config_source)
        from services.ide.multifile import MultifileSourceEntry
        request.sources.append(MultifileSourceEntry(file=config_file, source=config_source,
                                          digest=hashlib.sha256(config_source.encode()).hexdigest()))
        if construction == 'package_config':
            (workspace / 'models/__init__.py').write_text('')
            request.sources.append(MultifileSourceEntry(file='models/__init__.py', source='',
                                                digest=hashlib.sha256(b'').hexdigest()))
    resolution = discover(request)['resolution']
    assert resolution['status'] == ('partial' if factory else 'complete')
    if construction == 'inline_example':
        assert resolution['draft']['profile']['init']['kwargs'] == {'width': 12}
        assert resolution['draft']['profile']['inputs']['args'][0]['shape'] == [3, 12]
    elif construction == 'block_count':
        assert resolution['draft']['configObject']['values'] == {'depth': 1}
    elif construction == 'constructor_config':
        assert resolution['draft']['configObject']['values'] == {}
        assert resolution['draft']['profile']['inputs']['kwargs']['x']['shape'] == [1, 24]
    if factory:
        assert resolution['draft']['profile']['factory']['callable'] == 'make_model'
        assert resolution['unresolved'] == ['inputs.x: provide a value.']
        resolution['draft']['profile']['inputs']['kwargs']['x'] = dict(
            kind='tensor', shape=[1, 8], dtype='float32', requires_grad=False,
            generator=dict(kind='normal', mean=0, std=1, seed=0))
    prepared = prepare_profile(request, '', resolution['draft']['profile'],
                               config_object=resolution['draft'].get('configObject'))
    assert prepared['ok']
    (workspace / 'tensorviz.toml').write_text(prepared['config'])
    if prepared['factory']:
        helper = workspace / prepared['factory']['file']
        helper.parent.mkdir(exist_ok=True)
        helper.write_text(prepared['factory']['source'])
    params = dict(root=str(workspace), executable=sys.executable, environmentRevision=0,
                  session='auto-setup-test', workspaceId='auto-setup-test', runId='auto-run', generation=1,
                  modelFile=model_file, className='Net', timeoutMs=30000, profile=prepared['profile'])
    spec = RunSpec.model_validate(snapshot(params)['resolution']['spec'])
    plan = ResolvedExecutionPlan.model_validate(snapshot(dict(params, expectedIdentity=spec.identity.model_dump(),
                                                  execution=profile_execution(spec.profile)))['plan'])
    plan_file = tmp_path / 'plan.json'
    plan_file.write_text(json.dumps(dict(version=1, plan=plan.model_dump(mode='json'), applicabilityDigest=applicability_digest(plan))))
    plan_file.chmod(0o600)
    backend = Path(__file__).parent.resolve().parents[1]
    process = subprocess.run([sys.executable, '-B', str(backend / 'services/ingest/bootstrap.py'), '--resolved-execution',
                              '--workspace', str(workspace), '--file', str(workspace / model_file), '--cls', 'Net', '--json'],
                             # Match the extension host: its backend on PYTHONPATH must
                             # not shadow the workspace's namespace package `models`.
                             cwd=backend, env={**os.environ, 'TENSORVIZ_EXECUTION_PLAN': str(plan_file),
                                               'PYTHONPATH': str(backend) + os.pathsep + os.environ.get('PYTHONPATH', '')},
                             capture_output=True, text=True, timeout=45)
    envelope = json.loads(process.stdout)
    assert process.returncode == (4 if fail else 0), envelope.get('error')
    diagnostics = envelope['result']['diagnostics']
    assert diagnostics['execution']['planCurrent'] is True
    assert diagnostics['timings_ms']['input_candidates_tried'] == 1
    capture = diagnostics['semantic_capture']
    assert capture['status'] == ('partial' if fail else 'complete')
    projection = project_capture(capture)
    assert projection.nodes, 'a successful worker must produce a measured graph'
    assert (workspace / 'tensorviz.toml').read_text() == prepared['config']
