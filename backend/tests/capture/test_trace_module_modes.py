"""Inference capture must not leak temporary mode changes into the user's model."""
import pytest
import torch
from torch import nn

from services.ingest.trace import trace_model


class MixedModeModel(nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.norm = nn.BatchNorm1d(4)
        self.dropout = nn.Dropout()
        self.fail = fail
        self.seen_modes = None

    def forward(self, x):
        self.seen_modes = [module.training for module in self.modules()]
        result = self.dropout(self.norm(x))
        if self.fail:
            raise RuntimeError("intentional forward failure")
        return result


@pytest.mark.parametrize("tier", ["real", "fake"])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("root_training", [False, True])
def test_inference_capture_restores_mixed_modes(tier, fail, root_training):
    model = MixedModeModel(fail).train(root_training)
    # Exercise both frozen children in a training model and active children in
    # an eval model. A batch of one proves BatchNorm actually ran in eval mode.
    model.norm.train(not root_training)
    before = [module.training for module in model.modules()]
    running_mean = model.norm.running_mean.clone()
    result = trace_model(model, (torch.ones(1, 4),), tier=tier)
    assert model.seen_modes == [False, False, False]
    assert [module.training for module in model.modules()] == before
    assert torch.equal(model.norm.running_mean, running_mean)
    assert bool(result.error) == fail
    if fail:
        assert "intentional forward failure" in result.error
    for module in model.modules():
        assert not module._forward_hooks
        assert not module._forward_pre_hooks


class CustomMode(nn.Module):
    def __init__(self):
        super().__init__()
        self.mode_label = "train"
        self.transitions = []

    def train(self, mode=True):
        super().train(mode)
        self.mode_label = "train" if mode else "eval"
        self.transitions.append(mode)
        return self

    def forward(self, x):
        return x + 1


@pytest.mark.parametrize("tier", ["real", "fake"])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("root_training", [False, True])
def test_inference_capture_restores_custom_train_state(tier, fail, root_training):
    class CustomModel(CustomMode):
        def __init__(self):
            super().__init__()
            self.active = CustomMode()
            self.frozen = CustomMode().eval()

        def forward(self, x):
            assert all(module.mode_label == "eval" for module in self.modules())
            x = self.frozen(self.active(x))
            if fail:
                raise RuntimeError("intentional forward failure")
            return x

    model = CustomModel().train(root_training)
    model.active.train(not root_training)
    model.frozen.eval()
    before = [(module.training, module.mode_label) for module in model.modules()]
    result = trace_model(model, (torch.ones(1, 4),), tier=tier)
    assert bool(result.error) == fail
    assert [(module.training, module.mode_label) for module in model.modules()] == before
    for module in model.modules():
        assert not module._forward_hooks
        assert not module._forward_pre_hooks


@pytest.mark.parametrize("tier", ["real", "fake"])
def test_eval_capture_does_not_reenter_custom_train(tier):
    model = CustomMode().eval()
    before = list(model.transitions)
    result = trace_model(model, (torch.ones(1, 4),), tier=tier)
    assert result.error is None
    assert model.transitions == before


def test_uniform_capture_does_not_repeat_custom_train():
    child = CustomMode()
    model = nn.Sequential(nn.Sequential(nn.Sequential(child)))
    result = trace_model(model, (torch.ones(1),), tier="real")
    assert result.error is None
    assert child.transitions == [False, True]


@pytest.mark.parametrize("tier", ["real", "fake"])
def test_shared_child_restores_after_both_parents(tier):
    model = nn.Module()
    model.left = nn.Sequential(CustomMode())
    model.right = nn.Sequential(model.left[0])
    model.left.eval()
    model.forward = lambda x: model.right(model.left(x))
    before = [(module.training, getattr(module, "mode_label", None)) for module in model.modules()]
    result = trace_model(model, (torch.ones(1, 4),), tier=tier)
    assert result.error is None
    assert [(module.training, getattr(module, "mode_label", None)) for module in model.modules()] == before


@pytest.mark.parametrize("tier", ["real", "fake"])
def test_failed_custom_restore_reports_error_and_removes_hooks(tier):
    class RefusesRestore(CustomMode):
        def train(self, mode=True):
            super().train(mode)
            if mode:
                raise RuntimeError("mode restoration failed")
            return self

    model = nn.Sequential(RefusesRestore(), CustomMode())
    result = trace_model(model, (torch.ones(1, 4),), tier=tier)
    assert "mode restoration failed" in result.error
    assert all(module.training for module in model.modules())
    assert model[1].mode_label == "train"
    for module in model.modules():
        assert not module._forward_hooks
        assert not module._forward_pre_hooks


def test_fake_limitation_keeps_mode_restoration_failure():
    from services.semantic.contracts import CaptureLimits, SourceSnapshot
    from services.semantic.recorder import CaptureContext

    context = CaptureContext(source=SourceSnapshot(version=1, id="source:0", workspaceSessionId="session:0",
        workspaceId="workspace:0", modelId="model:0", modelFile="model.py", className="Model", generation=0,
        files=[dict(path="model.py", digest="a" * 64, documentVersion=1, role="source")],
        dependencyScope="complete", configDigest=None, environmentDigest="b" * 64, sourceMode="saved"),
        plan_digest="c" * 64, attempt_id="attempt:0",
        limits=CaptureLimits(operations=10000, invocations=10000, values=10000, bytes=4 * 1024 * 1024))

    class RefusesRestore(CustomMode):
        def train(self, mode=True):
            if mode:
                raise RuntimeError("mode restoration failed")
            return super().train(mode)

        def forward(self, x):
            return x + x.sum().item()

    model = RefusesRestore()
    result = trace_model(model, (torch.ones(1),), tier="fake", semantic_context=context)
    assert result.fake_limitation is not None
    assert "FakeTensorLimitation" in result.error
    assert "Mode restoration failed" in result.error
    assert "mode restoration failed" in result.error
    assert model.training
    assert not model._forward_hooks
    assert not model._forward_pre_hooks


@pytest.mark.parametrize("tier", ["real", "fake"])
def test_explicit_preserve_mode_keeps_mixed_modes(tier):
    model = MixedModeModel().train()
    model.norm.eval()
    before = [module.training for module in model.modules()]
    result = trace_model(model, (torch.ones(1, 4),), tier=tier, preserve_mode=True)
    assert result.error is None
    assert model.seen_modes == before
    assert [module.training for module in model.modules()] == before


def test_failing_eval_restores_modes_and_removes_hooks():
    class RefusesEval(MixedModeModel):
        def train(self, mode=True):
            super().train(mode)
            if not mode:
                raise RuntimeError("eval setup failed")
            return self

    model = RefusesEval()
    model.norm.eval()
    before = [module.training for module in model.modules()]
    result = trace_model(model, (torch.ones(1, 4),), tier="real")
    assert "eval setup failed" in result.error
    assert [module.training for module in model.modules()] == before
    for module in model.modules():
        assert not module._forward_hooks
        assert not module._forward_pre_hooks
