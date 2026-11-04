"""Setup routes preserve actionable, source-proven errors without leaking exceptions."""
import pytest
from fastapi import HTTPException

from api.routers.ide import (GraphSetupDiscoveryRequest, GraphSetupProfileRequest,
                             discover_graph_setup, prepare_graph_profile)
from tests.discovery.test_graph_setup_profiles import profile, snapshot


@pytest.mark.parametrize('route', ['discover', 'profile'])
def test_non_pytorch_class_error_reaches_the_api_response(route):
    sources = snapshot('''class Module: pass
class Net(Module):
    def __init__(self, nin, nouts): pass
    def __call__(self, x): return x
''')
    reason = ('Net is not a torch.nn.Module subclass; '
              'tensorViz captures PyTorch nn.Module (and Flax) models.')
    if route == 'discover':
        result = discover_graph_setup(GraphSetupDiscoveryRequest(**sources.model_dump()))
        assert result['resolution']['unresolved'] == [reason]
        assert result['resolution']['status'] == 'partial'
    else:
        result = prepare_graph_profile(GraphSetupProfileRequest(sources=sources, config='', profile=profile()))
        assert result == dict(ok=False, requirements=[dict(field=['model'], reason=reason)])


@pytest.mark.parametrize('route', ['discover', 'profile'])
def test_other_setup_value_errors_remain_redacted(monkeypatch, route):
    from services.ide import graph_setup

    def fail(*args, **kwargs):
        raise ValueError('private exception details')

    monkeypatch.setattr(graph_setup, 'discover', fail)
    if route == 'profile':
        from services.ide import graph_setup_profiles
        monkeypatch.setattr(graph_setup_profiles, 'discover', fail)
    with pytest.raises(HTTPException) as error:
        if route == 'discover':
            discover_graph_setup(GraphSetupDiscoveryRequest(**snapshot().model_dump()))
        else:
            prepare_graph_profile(GraphSetupProfileRequest(sources=snapshot(), config='', profile=profile()))
    assert error.value.status_code == 400
    assert 'private' not in error.value.detail


def test_tensor_draft_errors_reach_the_api_as_short_field_requirements():
    pending = profile()
    pending['inputs'] = dict(args=[], kwargs={'input_ids': dict(
        kind='tensor', dtype='int64', generator=dict(kind='randint', low=0, high=16, seed=0))})
    result = prepare_graph_profile(GraphSetupProfileRequest(sources=snapshot(), config='', profile=pending))
    assert result == dict(ok=False, requirements=[dict(
        field=['inputs', 'input_ids', 'shape'], reason='field required')])
