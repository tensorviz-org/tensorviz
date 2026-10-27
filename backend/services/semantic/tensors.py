"""Tensor metadata shared by capture adapters, with no activation keepalive."""
from __future__ import annotations

from .identity import weak_identity


class TensorMetadata:
    def __init__(self, torch, values, limit, omit, account):
        self.torch, self.values, self.limit = torch, values, limit
        self.omit, self.account = omit, account
        self.objects, self.storages, self.counters = {}, {}, {}
        self.snapshots, self.latest, self.storage_latest = {}, {}, {}
        self.storage_epochs, self.snapshot_epochs, self.metadata = {}, {}, {}
        self.reason = None
        from torch._subclasses.fake_tensor import FakeTensor
        # FakeTensors carry the same shape/stride/storage metadata and never values.
        self.dense = (torch.Tensor, torch.nn.Parameter, FakeTensor)

    def identity(self, value, table, prefix):
        return weak_identity(table, self.counters, value, prefix)

    def unavailable(self, reason):
        self.reason = reason
        self.omit(reason)
        return None

    def snapshot_key(self, oid, version):
        return (oid, version)

    def observe(self, value):
        self.reason = None
        if type(value) not in self.dense or value.layout != self.torch.strided:
            return self.unavailable("unsupported_type")
        oid = self.identity(value, self.objects, "object")
        try:
            version = value._version
        except RuntimeError:
            version = None
        storage = value.untyped_storage()
        sid = self.identity(storage, self.storages, "storage")
        shape, stride, offset = list(value.shape), list(value.stride()), value.storage_offset()
        # Symbolic/fake/subclass tensors need their own adapter, not int coercion.
        if (len(shape) > 32 or any(type(n) is not int or not 0 <= n <= 2**53 - 1 for n in shape)
                or any(type(n) is not int or abs(n) > 2**53 - 1 for n in stride)
                or type(offset) is not int or not 0 <= offset <= 2**53 - 1):
            return self.unavailable("unsupported_type")
        metadata = (shape, stride, offset, value.requires_grad, sid)
        key = self.snapshot_key(oid, version)
        if key in self.snapshots:
            if (self.metadata[key] != metadata
                    or self.snapshot_epochs[key] != self.storage_epochs.get(sid, 0)):
                # A write through a separate alias counter/inference tensor cannot
                # be invented as another version of the same logical value.
                return self.unavailable("unsupported_type")
            return self.snapshots[key]
        if len(self.values) >= self.limit:
            return self.unavailable("value_limit")
        previous = self.latest.get(oid)
        tid = f"tensor{len(self.values)}"
        alias = self.storage_latest.get(sid)
        fact = dict(id=tid, objectId=oid, storageId=sid, version=version, shape=shape,
                    dtype=str(value.dtype), device=str(value.device), requiresGrad=value.requires_grad,
                    aliasOf=alias if alias != previous else None,
                    previousVersion=previous if version is not None else None)
        fact = self.decorate(fact, value)
        if self.account(fact) is False:
            return self.unavailable("byte_limit")
        self.values.append(fact)
        self.snapshots[key] = fact
        self.metadata[key] = metadata
        self.snapshot_epochs[key] = self.storage_epochs.get(sid, 0)
        self.latest[oid] = self.storage_latest[sid] = tid
        return fact

    def decorate(self, fact, value):
        return fact

    def written(self, value):
        if type(value) in self.dense:
            if value.layout == self.torch.strided:
                sid = self.identity(value.untyped_storage(), self.storages, "storage")
                self.storage_epochs[sid] = self.storage_epochs.get(sid, 0) + 1
        elif type(value) in (tuple, list):
            for item in value[:4096]:
                self.written(item)
