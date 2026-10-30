"""Metadata observations inside an already authorized checkpoint load.

No file reads, state_dict hooks, tensor values, model calls or loader decisions.
The ordinary loader remains authoritative, including custom serialization hooks.
"""
from __future__ import annotations

from collections import OrderedDict
from itertools import islice
import json

import torch

from ..validation.setup import ExecutionRequirement

MAX_ENTRIES = 4096
MAX_BYTES = 1024 * 1024
MAX_MODULES = 10000


class CheckpointLoadFailure(ExecutionRequirement):
    def __init__(self, report):
        super().__init__('materialization', 'invalid_config', ['checkpoint'], 'edit_profile')
        self.checkpoint_compatibility = report


def _metadata(value, kind, objects, storages):
    result = dict(kind=kind, shape=None, dtype=None, objectGroup=None,
                  storageGroup=None, storageOffset=None, stride=None, unavailable=None)
    if type(value) not in (torch.Tensor, torch.nn.Parameter):
        result['unavailable'] = 'unsupported_tensor' if isinstance(value, torch.Tensor) else 'non_tensor_state'
        return result
    try:
        shape = list(value.shape)
        if len(shape) > 32 or any(type(n) is not int or not 0 <= n <= 9007199254740991 for n in shape):
            raise ValueError('shape limit')
        result.update(shape=shape, dtype=str(value.dtype),
                      objectGroup=objects.setdefault(id(value), len(objects)))
        if value.layout != torch.strided or value.is_quantized or value.device.type == 'meta':
            result['unavailable'] = 'unsupported_tensor'
            return result
        storage = (str(value.device), value.untyped_storage()._cdata)
        result.update(storageGroup=storages.setdefault(storage, len(storages)),
                      storageOffset=value.storage_offset(), stride=list(value.stride()))
    except (ValueError, TypeError, RuntimeError):
        result['unavailable'] = 'metadata_error'
    return result


def checkpoint_comparison(model, state, descriptor, source_digest):
    report = dict(version=1, sourceDigest=source_digest, checkpoint=dict(descriptor),
                  scope='registered_state_before_load', loaderSemantics='not_inferred',
                  loadStatus='not_run', complete=False, registeredComplete=False,
                  storedComplete=False, omissions=[], entries=[])
    omissions = set()
    registered, stored = {}, {}
    objects, storages = {}, {}
    try:
        pending = [(model, '', frozenset())]
        visited = 0
        while pending:
            module, prefix, ancestors = pending.pop()
            visited += 1
            if visited > MAX_MODULES:
                omissions.add('module_limit')
                break
            if id(module) in ancestors:
                omissions.add('cyclic_registration')
                continue
            attrs = object.__getattribute__(module, '__dict__')
            nonpersistent = attrs.get('_non_persistent_buffers_set', set())
            if type(nonpersistent) not in (set, frozenset):
                omissions.add('unsupported_registration')
                continue
            for kind, field in (('parameter', '_parameters'), ('buffer', '_buffers')):
                table = attrs.get(field, {})
                if type(table) not in (dict, OrderedDict):
                    omissions.add('unsupported_registration')
                    continue
                if len(table) > MAX_ENTRIES:
                    omissions.add('entry_limit')
                for name, value in islice(table.items(), MAX_ENTRIES):
                    if type(name) is not str:
                        omissions.add('name_limit')
                        continue
                    if value is None or kind == 'buffer' and name in nonpersistent:
                        continue
                    full = f'{prefix}.{name}' if prefix else name
                    if type(full) is not str or len(full) > 2048:
                        omissions.add('name_limit')
                        continue
                    if len(registered) >= MAX_ENTRIES:
                        omissions.add('entry_limit')
                        break
                    registered[full] = _metadata(value, kind, objects, storages)
                if 'entry_limit' in omissions:
                    break
            if 'entry_limit' in omissions:
                break
            children = attrs.get('_modules', {})
            if type(children) not in (dict, OrderedDict):
                omissions.add('unsupported_registration')
                continue
            for name, child in islice(children.items(), MAX_MODULES):
                if type(name) is not str:
                    omissions.add('name_limit')
                    continue
                child_prefix = f'{prefix}.{name}' if prefix else name
                if type(child_prefix) is not str or len(child_prefix) > 2048:
                    omissions.add('name_limit')
                elif child is not None:
                    if len(pending) >= MAX_MODULES:
                        omissions.add('module_limit')
                        break
                    pending.append((child, child_prefix, ancestors | {id(module)}))
            if len(children) > MAX_MODULES:
                omissions.add('module_limit')
        report['registeredComplete'] = not omissions
        objects, storages = {}, {}
        if type(state) not in (dict, OrderedDict):
            omissions.add('unsupported_mapping')
        else:
            report['storedComplete'] = len(state) <= MAX_ENTRIES
            if len(state) > MAX_ENTRIES:
                omissions.add('entry_limit')
            for name, value in islice(state.items(), MAX_ENTRIES):
                if type(name) is not str or len(name) > 2048:
                    omissions.add('name_limit')
                    report['storedComplete'] = False
                    continue
                stored[name] = _metadata(value, 'stored', objects, storages)
        size = 0
        for name in dict.fromkeys([*registered, *stored]):
            expected, saved = registered.get(name), stored.get(name)
            def compare(field):
                if expected is None or saved is None or expected[field] is None or saved[field] is None:
                    return 'unknown'
                return 'same' if expected[field] == saved[field] else 'different'
            row = dict(name=name, registered=expected, stored=saved,
                       presence='both' if expected is not None and saved is not None
                       else ('registered_only' if report['storedComplete'] else 'unknown') if expected is not None
                       else 'checkpoint_only' if report['registeredComplete'] else 'unknown',
                       shape=compare('shape'), dtype=compare('dtype'))
            for metadata in (expected, saved):
                if metadata and metadata['unavailable']:
                    omissions.add(metadata['unavailable'])
            size += len(json.dumps(row, ensure_ascii=True).encode())
            if size > MAX_BYTES - 8192:
                omissions.add('byte_limit')
                break
            report['entries'].append(row)
    except Exception:
        # Metadata must not replace or prevent the explicitly requested load.
        omissions.add('metadata_error')
        report.update(registeredComplete=False, storedComplete=False, entries=[])
    report.update(complete=not omissions, omissions=sorted(omissions))
    return report
