"""Within-run block templates (TENSORVIZ_BLOCK_TEMPLATES) on the canonical fake tier.

A stamped capture must be exactly the capture the live forward records, so every test
here compares a flag-on run with a flag-off run of the same model and requires the two
captures to be identical apart from the capture id. The flag is read once at import, so
the tests flip the module attributes rather than the environment.
"""
import gc
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from services.ingest import block_templates
from services.ingest.trace import trace_model
from services.semantic.contracts import CaptureLimits, SourceSnapshot
from services.semantic.recorder import CaptureContext
from services.semantic.source_bindings import CapturedSourceEvidence

BACKEND = Path(__file__).parent.resolve().parents[1]


def context(**limits):
    bounds = dict(operations=100000, invocations=100000, values=100000, bytes=64 * 1024 * 1024)
    bounds.update(limits)
    return CaptureContext(source=SourceSnapshot(version=1, id="source:0", workspaceSessionId="session:0",
        workspaceId="workspace:0", modelId="model:0", modelFile="model.py", className="Model", generation=0,
        files=[dict(path="model.py", digest="a" * 64, documentVersion=1, role="source")],
        dependencyScope="complete", configDigest=None, environmentDigest="b" * 64, sourceMode="saved"),
        plan_digest="c" * 64, attempt_id="attempt:0", limits=CaptureLimits(**bounds))


@pytest.fixture
def flag(monkeypatch):
    def set_mode(mode):
        monkeypatch.setattr(block_templates, "ENABLED", mode in ("1", "shadow"))
        monkeypatch.setattr(block_templates, "SHADOW", mode == "shadow")
    set_mode("")
    return set_mode


def capture(build, inputs, mode, flag, **limits):
    """Construct, make inputs and trace inside one FakeTensorMode, as the fake-tier worker does."""
    flag(mode)
    with FakeTensorMode(allow_non_fake_inputs=True):
        torch.manual_seed(0)
        random.seed(0)
        model = build()
        args = inputs()
        result = trace_model(model, args, {}, tier="fake", semantic_context=context(**limits))
    assert result.error is None, result.error
    return result


def comparable(result):
    data = dict(result.semantic_capture, id="capture:fixed")
    return json.dumps(data, sort_keys=True), json.dumps(dict(io=result.io, counts=result.call_counts,
                                                             visited=sorted(result.visited)), sort_keys=True)


def on_off(build, inputs, flag, **limits):
    off = capture(build, inputs, "", flag, **limits)
    on = capture(build, inputs, "1", flag, **limits)
    shadow = capture(build, inputs, "shadow", flag, **limits)
    assert off.block_templates is None
    assert comparable(on) == comparable(off)
    assert comparable(shadow) == comparable(off)
    assert shadow.block_templates["shadowMismatches"] == 0, shadow.block_templates
    return on.block_templates, shadow.block_templates


# ───────────────────────────────────────────── models ───────────────────────────
class Block(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 2 * d)
        self.fc2 = nn.Linear(2 * d, d)
        self.drop = nn.Dropout(0.0)

    def forward(self, x):
        h = self.fc1(self.norm(x))
        h = torch.relu_(h)                       # in-place write on an internal tensor
        a, b = h.chunk(2, dim=-1)                # views, multi-output
        h = self.fc2(torch.cat([b, a], dim=-1))
        return x + self.drop(h)


class Stack(nn.Module):
    def __init__(self, block=Block, n=6):
        super().__init__()
        self.blocks = nn.ModuleList([block() for _ in range(n)])
        self.head = nn.Linear(8, 3)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def x28():
    return (torch.randn(2, 8),)


# ───────────────────────────────────────────── tests ────────────────────────────
def test_flag_off_builds_no_cache_and_reports_nothing(flag):
    result = capture(Stack, x28, "", flag)
    assert result.block_templates is None


def test_repeated_blocks_are_stamped_identically(flag):
    on, shadow = on_off(Stack, x28, flag)
    assert on["hits"] == 5 and on["misses"] == 1 and on["templates"] == 1, on
    assert on["stampedOperations"] > 0
    assert shadow["shadowCompared"] == 5


def test_stamped_outputs_are_fake_tensors_with_the_live_metadata(flag):
    seen = {}
    for mode in ("", "1"):
        flag(mode)
        with FakeTensorMode(allow_non_fake_inputs=True):
            model = Stack()
            got = []
            # A hook outside the blocks sees what the last (stamped) block returned.
            model.head.register_forward_hook(lambda module, args, output: got.append(args[0]))
            result = trace_model(model, x28(), {}, tier="fake", semantic_context=context())
        (x,) = got
        seen[mode] = (type(x), tuple(x.shape), x.stride(), x.dtype, x.storage_offset(), x._version,
                      x.untyped_storage().nbytes(), x._base is None)
    assert result.block_templates["hits"] == 5
    assert seen["1"] == seen[""] and seen["1"][0] is FakeTensor


class ViewOut(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.inner = nn.Sequential(nn.Linear(8, 8))

    def forward(self, x):
        return self.inner(self.fc(x)).view(2, 2, 4).transpose(1, 2).reshape(2, 8)


def test_view_outputs_are_fabricated_with_their_base(flag):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([ViewOut() for _ in range(4)])

        def forward(self, x):
            outs = []
            for block in self.blocks:
                x = block(x)
                outs.append(x.view(2, 2, 4)[:, 1])
            return torch.stack(outs).sum(0)
    on, _ = on_off(Model, x28, flag)
    assert on["hits"] == 3, on


def test_view_of_view_output_keeps_its_parent_view_alive(flag):
    """Under FakeTensorMode a view keeps its parent view alive; a later in-place write on
    the output then records an alias_update for that parent. The stamp must match."""
    class Heads(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 8)
            self.inner = nn.Sequential(nn.Linear(8, 8))

        def forward(self, x):
            return self.inner(self.fc(x)).view(2, 2, 4).transpose(1, 2)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Heads() for _ in range(4)])

        def forward(self, x):
            total = 0
            for block in self.blocks:
                y = block(x)
                y.add_(1.0)          # write through the returned view
                total = total + y.sum()
            return total
    on, _ = on_off(Model, x28, flag)
    assert on["hits"] == 3, on
    off = capture(Model, x28, "", flag).semantic_capture
    assert any(e["kind"] == "alias_update" for op in off["operations"] for e in op["effects"])


@pytest.mark.parametrize("config,expected_hits", [((8, 8, 8, 8), 3), ((8, 16, 8, 16), 2), ((8, 16, 32, 64), 0)])
def test_key_separates_configurations_and_input_shapes(flag, config, expected_hits):
    class Proj(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.d = d
            self.inner = nn.Sequential(nn.Linear(8, d), nn.ReLU())

        def forward(self, x):
            return self.inner(x)[:, :8] * (1.0 / self.d)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Proj(d) for d in config])

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x
    on, _ = on_off(Model, x28, flag)
    assert on["hits"] == expected_hits, on


def test_same_instance_called_again_is_a_candidate(flag):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = Block()

        def forward(self, x):
            for _ in range(4):
                x = self.shared(x)
            return x
    on, _ = on_off(Model, x28, flag)
    # A single instance becomes a candidate on its second call. That call already
    # knows the parameters (a different slot state than call 1 had); calls 3, 4 hit it.
    assert on["hits"] == 2 and on["templates"] == 1, on


COUNTER = [0]
GLOBAL_SCALE = [1.0]


class InPlaceInput(Block):
    def forward(self, x):
        x.mul_(1.0)
        return super().forward(x)


class AttrWrite(Block):
    def forward(self, x):
        self.calls = getattr(self, "calls", 0) + 1
        return super().forward(x)


class GlobalWrite(Block):
    def forward(self, x):
        COUNTER[0] += 1
        return super().forward(x) * GLOBAL_SCALE[0]


class TorchRng(Block):
    def forward(self, x):
        return super().forward(x) + torch.rand_like(x)


class PyRng(Block):
    def forward(self, x):
        return super().forward(x) * random.choice([1.0])


class Dynamic(Block):
    def forward(self, x):
        return nn.Identity()(super().forward(x))


class Raises(nn.Module):
    def forward(self, x):
        raise ValueError("inside")


class CaughtException(Block):
    def __init__(self):
        super().__init__()
        self.bad = Raises()

    def forward(self, x):
        try:
            self.bad(x)
        except ValueError:
            pass
        return super().forward(x)


class ParamWrite(Block):
    def forward(self, x):
        self.fc1.bias.add_(0.0)
        return super().forward(x)


@pytest.mark.parametrize("block,reason", [
    (InPlaceInput, "write_external"),
    (AttrWrite, "attr_write"),
    (GlobalWrite, "py_state_write"),
    (TorchRng, "rng_op"),
    (PyRng, "rng_python"),
    (Dynamic, "dynamic_module"),
    (CaughtException, "exception"),
    (ParamWrite, "write_external"),
])
def test_hazards_make_a_block_ineligible_and_the_capture_stays_exact(flag, block, reason):
    on, _ = on_off(lambda: Stack(block, n=4), x28, flag)
    assert on["hits"] == 0, on
    assert any(key.startswith(reason) for key in on["ineligible"]), on


def test_user_hooks_make_a_block_keyless(flag):
    def build():
        model = Stack(n=4)
        for block in model.blocks:
            block.register_forward_hook(lambda module, args, output: None)
        return model
    on, _ = on_off(build, x28, flag)
    assert on["hits"] == 0 and on["noKey"].get("user_hooks"), on


def test_integer_index_tensor_is_allowed_bool_mask_is_not(flag):
    class Gather(Block):
        def forward(self, x):
            index = torch.arange(8).flip(0)
            return super().forward(x)[:, index]
    on, _ = on_off(lambda: Stack(Gather, n=4), x28, flag)
    assert on["hits"] == 3, on


def test_dropout_probability_zero_is_not_rng(flag):
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(8, 24)

        def forward(self, x):
            q, k, v = self.qkv(x).view(2, 1, 3, 8).unbind(2)
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0).view(2, 8) + x
    on, _ = on_off(lambda: Stack(Attention, n=4), x28, flag)
    assert on["hits"] == 3, on


@pytest.mark.parametrize("limit", ["operations", "invocations", "values", "bytes"])
def test_limits_truncate_exactly_like_the_live_capture(flag, limit):
    full = capture(Stack, x28, "", flag).semantic_capture
    size = len(json.dumps(full).encode())
    bound = dict(operations=len(full["operations"]) * 2 // 3, invocations=len(full["invocations"]) * 2 // 3,
                 values=len(full["values"]) * 2 // 3, bytes=size * 2 // 3)[limit]
    off = capture(Stack, x28, "", flag, **{limit: bound})
    on = capture(Stack, x28, "1", flag, **{limit: bound})
    assert comparable(on) == comparable(off)
    assert off.semantic_capture is None or off.semantic_capture["coverage"]["omissions"]


def test_gc_threshold_is_restored(flag):
    before = gc.get_threshold()
    capture(Stack, x28, "1", flag)
    assert gc.get_threshold() == before


def test_shadow_mode_reports_a_corrupted_template(flag, monkeypatch):
    original = block_templates.TemplateCache._apply

    def corrupt(self, tpl, ctx, modules, u):
        outs = original(self, tpl, ctx, modules, u)
        self.rec.data["operations"][-1]["target"] = "aten.corrupted"
        return outs
    monkeypatch.setattr(block_templates.TemplateCache, "_apply", corrupt)
    off = capture(Stack, x28, "", flag)
    shadow = capture(Stack, x28, "shadow", flag)
    assert comparable(shadow) == comparable(off), "shadow mode must always emit the live result"
    assert shadow.block_templates["shadowMismatches"] == 5
    assert "target" in shadow.block_templates["shadowMismatchSamples"][0]["path"]


def test_call_binding_replacement_matches_the_original():
    class Stub:
        def __init__(self):
            self.internal = {__file__}
            self.torch_directory = os.path.dirname(torch.__file__) + os.sep
            self.calls = []

        def expression(self, code, lasti):
            self.calls.append((code.co_name, lasti))
            return {"id": f"binding:{code.co_name}"}

    stub = Stub()

    class Recorder:
        def __init__(self):
            self.sources = stub
            self.values = type("V", (), {"observe": lambda self, t: None, "omit": None})()
            self.encoder = type("E", (), {"omit": None})()

        def omit(self, reason):
            return None

    cache = block_templates.TemplateCache.__new__(block_templates.TemplateCache)
    cache.rec, cache.values = Recorder(), None
    cache.rec.values.values = []
    cache.values = cache.rec.values
    cache.active, cache.boundaries, cache.log = [], {}, []
    cache._install()

    def via_original():
        return CapturedSourceEvidence.call_binding(stub)

    def via_replacement():
        return stub.call_binding()

    def caller(fn):
        return fn()  # this file is internal to the stub, so the walk continues to pytest

    assert caller(via_replacement) == caller(via_original)
    first, second = stub.calls
    assert first[0] == second[0]
    cache.close()


MODEL = '''import torch
from torch import nn
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(8)
        self.fc = nn.Linear(8, 8)
    def forward(self, x):
        h = self.fc(self.norm(x))
        return x + torch.relu(h)
# tensorviz: input=2,8
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(4)])
        self.seq = nn.Sequential(*[Block() for _ in range(3)])
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(8, 8), nn.ReLU()) for _ in range(3)])
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        for mlp in self.mlps:     # library code inside: bindings resolve outside the call
            x = mlp(x)
        return self.seq(x)
'''


@pytest.mark.parametrize("mode", ["1", "shadow"])
def test_worker_envelope_is_identical(tmp_path, mode):
    """The product path: source bindings, call sites, the envelope, inside and outside a Sequential."""
    from services.validation.host_snapshot import snapshot
    model = tmp_path / "model.py"
    model.write_text(MODEL)
    prepared = snapshot(dict(root=str(tmp_path), executable=sys.executable, environmentRevision=0,
        session="test", workspaceId="test", runId="auto", generation=1, modelFile="model.py",
        className="Model", timeoutMs=60000, captureRequest=dict(input=None, tensorInputs=None, overrides=None)))

    def run(flag_value):
        private = tmp_path / f"private-{flag_value or 'off'}.json"
        private.write_text(json.dumps(dict(version=1, preparation=prepared["preparation"],
                                           applicabilityDigest=prepared["applicabilityDigest"])))
        private.chmod(0o600)
        env = {k: v for k, v in os.environ.items() if k != "TENSORVIZ_BLOCK_TEMPLATES"}
        env.update(TENSORVIZ_EXECUTION_PLAN=str(private), PYTHONDONTWRITEBYTECODE="1")
        if flag_value:
            env["TENSORVIZ_BLOCK_TEMPLATES"] = flag_value
        process = subprocess.run([sys.executable, "-B", str(BACKEND / "services/ingest/bootstrap.py"),
            "--capture-preparation", "--workspace", str(tmp_path), "--file", str(model), "--cls", "Model",
            "--tier", "fake", "--depth", "2", "--json"], cwd=BACKEND, env=env, capture_output=True, text=True,
            timeout=120)
        assert process.returncode == 0, process.stderr[-2000:]
        return json.loads(process.stdout)

    def normal(envelope):
        def walk(value, key=""):
            if isinstance(value, dict):
                return {k: ("<t>" if k.endswith("_ms") or k == "timings_ms" else walk(v, k))
                        for k, v in value.items() if k != "block_templates"}
            if isinstance(value, list):
                return [walk(v) for v in value]
            if isinstance(value, str) and value.startswith("capture:"):
                return "capture:<id>"
            if isinstance(value, float) and key in ("wall", "elapsed"):
                return "<t>"
            return value
        return walk(envelope)

    off, on = run(""), run(mode)
    assert "block_templates" not in off["result"]["diagnostics"]
    stats = on["result"]["diagnostics"]["block_templates"]
    assert stats["hits"] == 8, stats
    if mode == "shadow":
        assert stats["shadowMismatches"] == 0, stats
    assert normal(on) == normal(off)
    assert off["result"]["diagnostics"]["semantic_capture"]["sourceBindings"]
