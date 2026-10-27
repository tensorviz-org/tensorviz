"""Loss callables receive the real output, fixture batch and prepared model."""
import torch


def mse(output, batch, model):
    return (output - batch['targets']).square().mean()


def zero(output, batch, model):
    return output.sum() * 0


def nonfinite(output, batch, model):
    return output.sum() * torch.tensor(float('nan'), device=output.device)


def detached(output, batch, model):
    return mse(output, batch, model).detach()
