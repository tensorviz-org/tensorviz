"""Optional-library acceptance: real constructors, saved recipes, capture and projection.

Run with transformers installed; no network or pretrained weights are needed.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip('transformers')

from services.ide.graph_setup import discover
from services.ide.graph_setup_profiles import prepare_profile
from services.semantic.execution_plan import profile_execution
from services.semantic.projection import project_capture
from services.validation.contracts import ModelProfile
from services.validation.host_snapshot import snapshot
from tests.discovery.test_graph_setup_profiles import snapshot as sources


@pytest.mark.parametrize('family,config,keys', [
    ('Bert', 'vocab_size=256, hidden_size=16, num_hidden_layers=1, num_attention_heads=4, intermediate_size=32', ['input_ids']),
    ('T5', 'vocab_size=256, d_model=16, d_ff=32, d_kv=4, num_layers=1, num_decoder_layers=1, num_heads=4',
     ['decoder_input_ids', 'input_ids', 'decoder_attention_mask']),
    ('ViT', 'image_size=32, patch_size=8, hidden_size=16, num_hidden_layers=1, num_attention_heads=4, intermediate_size=32', ['pixel_values']),
    ('Dinov2', 'image_size=28, patch_size=14, hidden_size=16, num_hidden_layers=1, num_attention_heads=4, intermediate_size=32', ['pixel_values']),
])
def test_library_factory_gets_usable_inputs_and_a_measured_graph(tmp_path, family, config, keys):
    source = f'''from transformers import {family}Model, {family}Config
class Net({family}Model): pass

def make_model() -> Net:
    return Net({family}Config({config}))
'''
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'model.py').write_text(source)
    request = sources(source)
    resolution = discover(request)['resolution']
    assert resolution['status'] == 'partial' and resolution['runtimeInputs'] is True
    profile = resolution['draft']['profile']
    assert profile['factory']['callable'] == 'make_model'
    assert profile['inputs'] == dict(args=[], kwargs={})
    backend = Path(__file__).parent.resolve().parents[1]

    def run(probe):
        execution = profile_execution(ModelProfile.model_validate(profile))
        raw = snapshot(dict(root=str(workspace), executable=sys.executable, environmentRevision=0,
            session='library-example', workspaceId='library-example', runId='library-example', generation=1,
            modelFile='model.py', className='Net', timeoutMs=30000, execution=execution))
        plan = tmp_path / 'plan.json'
        plan.write_text(json.dumps({'version': 1, 'plan': raw['plan'], 'applicabilityDigest': raw['applicabilityDigest']}))
        plan.chmod(0o600)
        result = subprocess.run([sys.executable, '-B', str(backend/'services/ingest/bootstrap.py'),
            '--file', str(workspace/'model.py'), '--cls', 'Net', '--workspace', str(workspace),
            '--resolved-execution', *(['--prepare-inputs'] if probe else []), '--json'], cwd=backend,
            env={**os.environ, 'PYTHONPATH': '', 'TENSORVIZ_EXECUTION_PLAN': str(plan),
                 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'},
            capture_output=True, text=True, timeout=40)
        envelope = json.loads(result.stdout)
        assert result.returncode == 0, envelope.get('error')
        return envelope['result']

    proposal = run(True)
    assert 'graph' not in proposal
    profile['inputs'] = proposal['graphExample']['binding']
    assert set(profile['inputs']['kwargs']) == set(keys)
    prepared = prepare_profile(request, '', profile)
    assert prepared['ok'], prepared
    (workspace/'tensorviz.toml').write_text(prepared['config'])
    # Reopen the saved profile, so serialization is part of the acceptance path.
    saved = snapshot(dict(root=str(workspace), executable=sys.executable, environmentRevision=0,
        session='library-example', workspaceId='library-example', runId='reopen', generation=2,
        modelFile='model.py', className='Net', timeoutMs=30000, profile=prepared['profile']))
    profile = saved['resolution']['spec']['profile']
    result = run(False)
    diagnostics = result['diagnostics']
    assert diagnostics['execution']['planCurrent'] is True
    assert diagnostics['timings_ms']['input_candidates_tried'] == 1
    capture = diagnostics['semantic_capture']
    (tmp_path/'capture.json').write_text(json.dumps(capture))
    root = next(call for call in capture['invocations'] if call['id'] == capture['rootInvocationId'])
    assert root['status'] == 'completed'
    output = {entry['key']['value']: entry['value'] for entry in capture['output']['entries']}
    assert output['last_hidden_state']['kind'] == 'value'
    if family == 'T5':
        # Cache objects are still opaque; preserve that limit instead of changing
        # the model's use_cache default to make the acceptance fixture easier.
        assert capture['status'] == 'partial' and capture['coverage']['omissions'] == ['unsupported_type']
        assert output['encoder_last_hidden_state']['kind'] == 'value'
        assert output['past_key_values'] == dict(kind='opaque', reason='unsupported_type',
                                                typeName='transformers.cache_utils.EncoderDecoderCache')
    else:
        assert capture['status'] == 'complete', capture['coverage']['omissions']
    projection = project_capture(capture)
    assert projection.nodes
    print(json.dumps({'family': family, 'inputs': {key: value['shape'] for key, value in profile['inputs']['kwargs'].items()},
                      'nodes': len(projection.nodes), 'capture': capture['status']}))
