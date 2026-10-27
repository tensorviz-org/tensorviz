"""The other half. `from .cycle_a import CycleA` closes the cycle."""
import torch.nn as nn

from .cycle_a import CycleA  # noqa: F401


class CycleB(nn.Module):
    def __init__(self, width: int = 8):
        super().__init__()
        self.inner = CycleA(width)

    def forward(self, x):
        return self.inner(x)
