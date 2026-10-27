"""A plain block, imported as `models.blocks` (task.md §7 pins that path)."""
import torch.nn as nn


class MyBlock(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.fc = nn.Linear(width, width)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.fc(x))


# tensorviz: input=2,8
