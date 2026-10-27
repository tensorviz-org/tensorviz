"""One half of an import cycle. Never executed: the pair exists so static
resolution can prove it terminates instead of recursing forever."""
import torch.nn as nn

from .cycle_b import CycleB  # noqa: F401


class CycleA(nn.Module):
    def __init__(self, width: int = 8):
        super().__init__()
        self.inner = CycleB(width)

    def forward(self, x):
        return self.inner(x)
