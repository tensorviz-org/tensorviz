"""Capture identity must not extend activation lifetimes or confuse reused addresses."""
import gc
import weakref

import pytest
import torch
from torch import nn

from services.semantic.identity import weak_get, weak_identity, weak_set
from services.ingest.trace import trace_model
from services.validation.trace import TraceRecorder


class Item:
    pass


def test_weak_identity_is_stable_live_unique_after_release_and_cleans_metadata():
    table, counters = {}, {}
    identities = []
    for _ in range(100):
        item = Item()
        identity = weak_identity(table, counters, item, 'value:')
        assert weak_identity(table, counters, item, 'value:') == identity
        identities.append(identity)
        reference = weakref.ref(item)
        del item
        assert reference() is None
        assert not table, 'released objects cannot accumulate metadata entries'
    assert len(set(identities)) == 100


def test_an_old_weak_callback_cannot_delete_a_new_entry_at_the_same_key():
    old, new = Item(), Item()
    table = {}
    weak_set(table, old, 'old')
    key = id(old)
    callback = table[key][0]
    # Simulate a replacement between registration and the old callback.
    replacement = (weakref.ref(new), 'new')
    table[key] = replacement
    del old
    assert callback() is None
    assert table[key] is replacement
    assert weak_get(table, new) is None, 'a foreign address does not establish identity'


@pytest.mark.parametrize('record_evidence', [False, True])
def test_forward_can_release_an_intermediate_while_capture_is_running(record_evidence):
    class Net(nn.Module):
        def forward(self, x):
            temporary = x + 1
            reference = weakref.ref(temporary)
            output = temporary * 2
            del temporary
            gc.collect()
            self.released_during_forward = reference() is None
            return output + x
    model = Net()
    traced = trace_model(model, (torch.ones(4),), tier='real', record_evidence=record_evidence)
    assert traced.error is None
    assert model.released_during_forward, 'provenance must not pin all intermediate tensors'
    assert traced.ops[0]['in_ids'] == traced.input_ids
    assert traced.ops[1]['in_ids'] == traced.ops[0]['out_ids']
    assert traced.ops[2]['in_ids'] == [*traced.ops[1]['out_ids'], *traced.input_ids]


def test_temporary_outputs_never_reuse_a_previous_producer_identity():
    class Net(nn.Module):
        def forward(self, x):
            for _ in range(200):
                temporary = x + 1
                del temporary
            return x
    traced = trace_model(Net(), (torch.ones(1),), tier='real')
    assert traced.error is None
    assert len(traced.ops) == 200
    identities = [operation['out_ids'][0] for operation in traced.ops]
    assert len(set(identities)) == len(identities)
    assert all(operation['in_ids'] == traced.input_ids for operation in traced.ops)


def test_validation_identity_keeps_storage_alias_and_mutation_facts_without_tensors():
    record = {'tensors': [], 'limits': {'max_tensors': 20, 'max_record_bytes': 100000}}
    recorder = TraceRecorder(record, torch)
    original = torch.ones(2, 4)
    alias = original.view(4, 2)
    original_ref, alias_ref = weakref.ref(original), weakref.ref(alias)
    before = recorder.tensor(original)['tensorId']
    view = recorder.tensor(alias)['tensorId']
    original.add_(1)
    after = recorder.tensor(original)['tensorId']
    facts = {item['id']: item for item in record['tensors']}
    assert facts[view]['aliasOf'] == before
    assert facts[view]['storageId'] == facts[before]['storageId']
    assert facts[after]['objectId'] == facts[before]['objectId']
    assert facts[after]['previousVersion'] == before
    del original, alias
    gc.collect()
    assert original_ref() is alias_ref() is None
    assert not recorder.objects and not recorder.storages
    assert len(record['tensors']) == 3, 'immutable metadata outlives tensor storage'
