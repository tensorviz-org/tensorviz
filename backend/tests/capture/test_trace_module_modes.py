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
