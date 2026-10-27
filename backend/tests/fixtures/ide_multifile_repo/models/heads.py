"""The second `Block`. Nothing about `from models.heads import Block` may resolve to
blocks.py's class, and `WideBlock` exercises local inheritance through a binding."""
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, width: int = 8):
        super().__init__()
        self.proj = nn.Linear(width, width)

    def forward(self, x):
        return self.proj(x)


class WideBlock(Block):
    def __init__(self, width: int = 8):
        super().__init__(width * 2)
        self.narrow = nn.Linear(width * 2, width)

    def forward(self, x):
        return self.narrow(super().forward(x))
