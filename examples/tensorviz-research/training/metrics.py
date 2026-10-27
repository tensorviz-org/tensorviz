"""What we report. Perplexity is the headline; accuracy is the sanity check."""
import math

import torch


def token_accuracy(logits, targets):
    """Fraction of positions where the argmax matched."""
    predicted = logits.argmax(dim=-1)
    return (predicted == targets).float().mean().item()


def perplexity(mean_loss: float) -> float:
    """exp of mean cross-entropy, clamped so a diverged run prints a number."""
    return math.exp(min(mean_loss, 20.0))


@torch.no_grad()
def summarise(losses):
    mean_loss = sum(losses) / max(len(losses), 1)
    return {"loss": mean_loss, "perplexity": perplexity(mean_loss)}
