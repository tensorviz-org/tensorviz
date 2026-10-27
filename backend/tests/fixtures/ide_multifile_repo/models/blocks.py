"""The first of two classes named `Block` in this workspace (models/heads.py has the other).

Static resolution must follow the import binding on the call site, never the bare
name, so the two stay distinct definitions.
"""
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, width: int = 8):
        super().__init__()
        self.fc = nn.Linear(width, width)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.fc(x))
