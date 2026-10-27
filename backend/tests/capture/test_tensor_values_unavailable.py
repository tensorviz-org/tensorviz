"""Fake and meta tensors have no values: fingerprints refuse them before reading memory.

`tensor_bytes` copies a tensor's bytes with `ctypes.string_at(data_ptr(), n)`. A
FakeTensor's data_ptr is not the address of any values, so on `main` that read
segfaulted the worker (llama_1b) or silently hashed unrelated memory (mlp_tiny).
Real tensors must keep exactly the digests they had before the guard.
"""
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from services.ingest import execution_setup
from services.ingest.execution_setup import (TensorValuesUnavailable, describe_tree, state_identity,
                                             tensor_bytes, tensor_digest, values_unavailable)

BACKEND = Path(__file__).parent.resolve().parent.parent


@pytest.fixture
def no_memory_reads(monkeypatch):
    """Any raw-pointer read fails the test, so a refusal must happen before it."""
    def refuse(*_args, **_kwargs):
        pytest.fail("ctypes.string_at must not read a fake or meta tensor")
    monkeypatch.setattr(execution_setup.ctypes, "string_at", refuse)


def fake(factory):
    with FakeTensorMode():
        return factory()


def expected_digest(dtype: str, shape: list, raw: bytes) -> str:
    """The pre-guard algorithm, written out independently: header, then raw bytes."""
    digest = hashlib.sha256()
    digest.update(json.dumps([dtype, shape]).encode())
    digest.update(raw)
    return digest.hexdigest()


@pytest.mark.parametrize("tensor, dtype, raw", [
    (torch.tensor([1.0, -2.5, 3.0]), "torch.float32", struct.pack("<3f", 1.0, -2.5, 3.0)),
    (torch.tensor([[1, 2], [3, 4]], dtype=torch.int64).t(), "torch.int64", struct.pack("<4q", 1, 3, 2, 4)),
    (torch.tensor([True, False]), "torch.bool", b"\x01\x00"),
    (torch.tensor([1 + 2j]).conj(), "torch.complex64", struct.pack("<2f", 1.0, -2.0)),
    (torch.tensor(7.0), "torch.float32", struct.pack("<f", 7.0)),
    (torch.empty(0, 3), "torch.float32", b""),
])
def test_real_tensor_digests_and_bytes_are_unchanged(tensor, dtype, raw):
    assert values_unavailable(tensor) is None
    assert tensor_bytes(tensor) == raw
    assert tensor_digest(tensor) == expected_digest(dtype, list(tensor.shape), raw)


def test_large_real_tensor_still_streams_every_chunk():
    tensor = torch.arange(600_000, dtype=torch.float32)
    raw = struct.pack(f"<{tensor.numel()}f", *tensor.tolist())
    assert tensor_digest(tensor) == expected_digest("torch.float32", [600_000], raw)


@pytest.mark.parametrize("factory", [
    lambda: torch.ones(4, 3),
    lambda: torch.ones(4, 3).t(),                    # non-contiguous view
    lambda: torch.zeros(0, 5),                       # empty: still no value digest
    lambda: torch.tensor(2.0),                       # 0-dim
    lambda: torch.ones(2, dtype=torch.complex64).conj(),
    lambda: torch.ones(3, dtype=torch.int64),
    lambda: nn.Parameter(torch.ones(2, 2)),
])
def test_fake_tensors_are_refused_without_reading_memory(factory, no_memory_reads):
    tensor = fake(factory)
    assert values_unavailable(tensor) == "fake_tensor"
    with pytest.raises(TensorValuesUnavailable, match="^fake_tensor$"):
        tensor_bytes(tensor)
    with pytest.raises(TensorValuesUnavailable, match="^fake_tensor$"):
        tensor_digest(tensor)


@pytest.mark.parametrize("shape", [(4, 3), (0,), ()])
def test_meta_tensors_are_refused_with_their_own_reason(shape, no_memory_reads):
    tensor = torch.empty(shape, device="meta")
    assert values_unavailable(tensor) == "meta_tensor"
    with pytest.raises(TensorValuesUnavailable, match="^meta_tensor$"):
        tensor_bytes(tensor)
    with pytest.raises(TensorValuesUnavailable, match="^meta_tensor$"):
        tensor_digest(tensor)


def test_a_real_tensor_copied_inside_an_active_fake_mode_is_refused(no_memory_reads):
    real = torch.ones(8)
    with FakeTensorMode(allow_non_fake_inputs=True):
        # The CPU copy inside tensor_bytes becomes a FakeTensor under the mode.
        with pytest.raises(TensorValuesUnavailable, match="^fake_tensor$"):
            tensor_bytes(real)
        with pytest.raises(TensorValuesUnavailable, match="^fake_tensor$"):
            tensor_digest(real)


def test_sparse_and_quantized_keep_their_existing_refusal():
    sparse = torch.eye(2).to_sparse()
    quantized = torch.quantize_per_tensor(torch.ones(2), 0.1, 0, torch.quint8)
    for tensor in (sparse, quantized):
        with pytest.raises(ValueError, match="dense, unquantized") as raised:
            tensor_digest(tensor)
        assert type(raised.value) is ValueError


def test_state_identity_names_why_a_fake_or_meta_state_has_no_digest(no_memory_reads):
    fake_model = fake(lambda: nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4)))
    assert state_identity(fake_model) == {"digest": None, "parametersAndBuffers": None, "unavailable": "fake_tensor"}
    meta_model = nn.Linear(4, 4, device="meta")
    assert state_identity(meta_model)["unavailable"] == "meta_tensor"


def test_one_fake_tensor_in_a_real_state_leaves_no_partial_digest():
    mixed = nn.Linear(2, 2)
    mixed.register_buffer("shadow", fake(lambda: torch.ones(2)))
    assert state_identity(mixed)["unavailable"] == "fake_tensor"


def test_real_state_identity_is_unchanged():
    model = nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0]]))
        model.bias.fill_(0.5)
    identity = state_identity(model)
    assert identity["unavailable"] is None
    weight, bias = identity["parametersAndBuffers"]["items"]["weight"], identity["parametersAndBuffers"]["items"]["bias"]
    assert weight["digest"] == expected_digest("torch.float32", [1, 2], struct.pack("<2f", 1.0, 2.0))
    assert bias["digest"] == expected_digest("torch.float32", [1], struct.pack("<f", 0.5))
    # Any other exception type is still reported by name, as before.
    assert state_identity(_Unreadable())["unavailable"] == "RuntimeError"


class _Unreadable(nn.Module):
    def state_dict(self, *args, **kwargs):
        raise RuntimeError("boom")


def test_fake_inputs_cannot_be_fingerprinted(no_memory_reads):
    inputs = fake(lambda: {"args": [torch.ones(2, 3)], "kwargs": {}})
    with pytest.raises(TensorValuesUnavailable):
        describe_tree(inputs)


def test_large_fake_state_does_not_crash_the_process():
    # Regression for the llama_1b worker crash: main exits with SIGSEGV here.
    script = ("import torch\nfrom torch._subclasses.fake_tensor import FakeTensorMode\n"
              "from services.ingest.execution_setup import state_identity\n"
              "with FakeTensorMode():\n    model = torch.nn.Linear(1024, 1024)\n"
              "print(state_identity(model)['unavailable'])\n")
    run = subprocess.run([sys.executable, "-c", script], cwd=BACKEND, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.strip() == "fake_tensor"
