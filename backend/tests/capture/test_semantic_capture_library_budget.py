"""A valid library graph must not truncate solely because of unused padding."""
import inspect
import json

import pytest

from services.ide.graph_setup import discover
from services.semantic.projection import project_capture
from tests.capture.test_graph_inherited_defaults import _worker
from tests.discovery.test_graph_setup_profiles import snapshot as sources


def test_full_default_conditional_unet_keeps_its_output_within_the_existing_budget(tmp_path):
    pytest.importorskip('diffusers')
    from diffusers import UNet2DConditionModel

    defaults = inspect.signature(UNet2DConditionModel.__init__).parameters
    channels = defaults['in_channels'].default
    width = defaults['cross_attention_dim'].default
    source = 'from diffusers import UNet2DConditionModel\nclass Net(UNet2DConditionModel): pass\n'
    profile = discover(sources(source))['resolution']['draft']['profile']

    def tensor(shape):
        return dict(kind='tensor', shape=shape, dtype='float32', requires_grad=False,
                    generator=dict(kind='normal', mean=0, std=1, seed=0))

    # Explicit valid input, with no constructor overrides or learned fallback.
    profile['inputs'] = dict(args=[], kwargs=dict(sample=tensor([1, channels, 64, 64]),
        timestep=1, encoder_hidden_states=tensor([1, 8, width])))
    workspace = tmp_path/'workspace'
    workspace.mkdir()
    (workspace/'model.py').write_text(source)
    # This budget assertion concerns retained capture bytes, not CPU startup time.
    code, envelope = _worker(tmp_path, workspace, profile, probe=False, timeout_seconds=180)
    assert code == 0, envelope
    capture = envelope['result']['diagnostics']['semantic_capture']
    assert capture['status'] == 'complete' and not capture['coverage']['omissions']
    assert len(capture['operations']) == capture['coverage']['operationsSeen']
    assert len(capture['invocations']) == capture['coverage']['invocationsSeen']
    output_id = capture['output']['entries'][0]['value']['id']
    output = next(value for value in capture['values'] if value['id'] == output_id)
    assert output['shape'] == [1, channels, 64, 64]
    assert capture['limits']['bytes'] == 4 * 1024 * 1024
    assert len(json.dumps(capture, separators=(',', ':'), ensure_ascii=False).encode()) <= capture['limits']['bytes']
    assert len(project_capture(capture).nodes) > 1000
