"""Held-out evaluation. No gradients, model in eval mode, one pass."""
import torch
import torch.nn.functional as F

from training.metrics import summarise, token_accuracy


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    losses, accuracies = [], []
    for inputs, targets in loader:
        logits = model(inputs)
        losses.append(
            F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)).item()
        )
        accuracies.append(token_accuracy(logits, targets))

    report = summarise(losses)
    report["accuracy"] = sum(accuracies) / max(len(accuracies), 1)
    return report
