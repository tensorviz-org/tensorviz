"""Checkpoint metadata has narrower claims than the actual PyTorch loader."""
import json

import pytest
import torch

from services.ingest.checkpoint_state import checkpoint_comparison

DESCRIPTOR = {'file': 'weights.pt', 'digest': 'b' * 64, 'strict': True}


def compare(model, state):
    return checkpoint_comparison(model, state, DESCRIPTOR, 'a' * 64)


def rows(report):
    return {row['name']: row for row in report['entries']}


def test_width_mismatch_is_independent_of_dtype_conversion():
    model = torch.nn.Linear(3, 2, bias=False)
    report = compare(model, {'weight': torch.ones(2, 4, dtype=torch.float64)})
    row = rows(report)['weight']
    assert row['shape'] == row['dtype'] == 'different'
    assert report['loadStatus'] == 'not_run' and report['complete']
    with pytest.raises(RuntimeError, match='size mismatch'):
        model.load_state_dict({'weight': torch.ones(2, 4)})
    converted = {'weight': torch.ones(2, 3, dtype=torch.float64)}
    row = rows(compare(model, converted))['weight']
    assert row['shape'] == 'same' and row['dtype'] == 'different'
    model.load_state_dict(converted)
    assert model.weight.dtype == torch.float32


def test_names_and_nonpersistent_buffers_are_observed_without_inventing_keys():
    model = torch.nn.Module()
    model.register_parameter('weight', torch.nn.Parameter(torch.ones(3)))
    model.register_buffer('persistent', torch.ones(1))
    model.register_buffer('temporary', torch.ones(2), persistent=False)
    model.register_parameter('unused', None)
    row = rows(compare(model, {'extra': torch.ones(1), 'weight': torch.ones(3)}))
    assert set(row) == {'weight', 'persistent', 'extra'}
    assert row['persistent']['presence'] == 'registered_only'
    assert row['extra']['presence'] == 'checkpoint_only'
    assert row['weight']['shape'] == 'same'


def test_tied_objects_and_storage_views_have_distinct_relationships():
    model = torch.nn.Module()
    model.a = torch.nn.Parameter(torch.arange(6).float())
    model.tied = model.a
    model.view = torch.nn.Parameter(model.a[:3])
    state = {'a': torch.arange(6).float()}
    state['tied'] = state['a']
    state['view'] = state['a'][3:]
    row = rows(compare(model, state))
    assert row['a']['registered']['objectGroup'] == row['tied']['registered']['objectGroup']
    assert row['a']['registered']['objectGroup'] != row['view']['registered']['objectGroup']
    assert row['a']['registered']['storageGroup'] == row['view']['registered']['storageGroup']
    assert row['view']['registered']['storageOffset'] == 0
    assert row['view']['stored']['storageOffset'] == 3
    assert row['a']['stored']['objectGroup'] == row['tied']['stored']['objectGroup']
    assert row['a']['stored']['objectGroup'] != row['view']['stored']['objectGroup']


def test_equal_shapes_never_infer_sharing():
    model = torch.nn.Module()
    model.a = torch.nn.Parameter(torch.ones(3))
    model.b = torch.nn.Parameter(torch.ones(3))
    row = rows(compare(model, {'a': torch.ones(3), 'b': torch.ones(3)}))
    for inventory in ('registered', 'stored'):
        assert row['a'][inventory]['objectGroup'] != row['b'][inventory]['objectGroup']
        assert row['a'][inventory]['storageGroup'] != row['b'][inventory]['storageGroup']


def test_metadata_never_calls_state_dict_forward_or_load_hooks():
    class Custom(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
        def state_dict(self, *args, **kwargs):
            pytest.fail('state_dict is not a metadata reader')
        def forward(self, *args):
            pytest.fail('forward is not a metadata reader')
        def load_state_dict(self, *args, **kwargs):
            pytest.fail('comparison is not load authorization')
    model = Custom()
    model.register_state_dict_pre_hook(lambda *_: pytest.fail('serialization hook ran'))
    model.register_load_state_dict_pre_hook(lambda *_: pytest.fail('loading hook ran'))
    report = compare(model, {'weight': torch.ones(1)})
    assert report['loaderSemantics'] == 'not_inferred'
    assert report['complete'] and report['entries'][0]['shape'] == 'same'


@pytest.mark.parametrize('tensor', [torch.ones(2, device='meta'), torch.ones(2).to_sparse(),
                                   torch.quantize_per_tensor(torch.ones(2), 0.1, 0, torch.qint8)])
def test_unsupported_storage_does_not_become_known_state(tensor):
    report = compare(torch.nn.Linear(2, 1), {'weight': tensor})
    assert not report['complete'] and 'unsupported_tensor' in report['omissions']
    assert rows(report)['weight']['stored']['storageGroup'] is None


def test_extra_state_and_unsupported_mappings_are_explicit():
    report = compare(torch.nn.Module(), {'_extra_state': {'private': 'do not expose'}})
    assert not report['complete'] and report['entries'][0]['stored']['unavailable'] == 'non_tensor_state'
    assert 'do not expose' not in json.dumps(report)
    assert compare(torch.nn.Module(), [])['omissions'] == ['unsupported_mapping']


def test_limits_do_not_claim_complete_inventory(monkeypatch):
    import services.ingest.checkpoint_state as module
    monkeypatch.setattr(module, 'MAX_ENTRIES', 2)
    report = compare(torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)), {})
    assert len(report['entries']) == 2 and 'entry_limit' in report['omissions']
    assert not report['complete']
    report = compare(torch.nn.Module(), {str(i): torch.ones(1) for i in range(3)})
    assert len(report['entries']) == 2 and 'entry_limit' in report['omissions']


def test_truncated_inventory_does_not_falsely_claim_a_key_is_missing(monkeypatch):
    import services.ingest.checkpoint_state as module
    monkeypatch.setattr(module, 'MAX_ENTRIES', 1)
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.ones(1))
    report = compare(model, {'extra': torch.ones(1), 'weight': torch.ones(1)})
    assert report['registeredComplete'] and not report['storedComplete']
    assert rows(report)['weight']['presence'] == 'unknown'
    assert not report['complete']


def test_byte_limits_and_cycles_are_bounded(monkeypatch):
    import services.ingest.checkpoint_state as module
    monkeypatch.setattr(module, 'MAX_BYTES', 9000)
    report = compare(torch.nn.Module(), {('long' * 400) + str(i): torch.ones(1) for i in range(4)})
    assert not report['entries'] and 'byte_limit' in report['omissions']
    model = torch.nn.Module()
    model.child = model
    assert compare(model, {})['omissions'] == ['cyclic_registration']


def test_bad_registration_names_are_never_formatted():
    class Name:
        def __str__(self):
            pytest.fail('custom name formatting ran')
    model = torch.nn.Module()
    model._parameters[Name()] = torch.nn.Parameter(torch.ones(1))
    model._modules[Name()] = torch.nn.Module()
    report = compare(model, {})
    assert report['omissions'] == ['name_limit']


def test_metadata_failure_is_explicit_and_does_not_prevent_the_requested_load(monkeypatch):
    import services.ingest.checkpoint_state as module
    model = torch.nn.Linear(2, 1, bias=False)
    saved = {'weight': torch.ones(1, 2)}
    def unavailable(*args):
        raise RuntimeError('private metadata error')
    monkeypatch.setattr(module, '_metadata', unavailable)
    report = compare(model, saved)
    assert report['omissions'] == ['metadata_error'] and not report['complete']
    assert not report['registeredComplete'] and not report['storedComplete']
    assert report['entries'] == [] and 'private' not in json.dumps(report)
    model.load_state_dict(saved)
    assert torch.equal(model.weight, saved['weight'])
