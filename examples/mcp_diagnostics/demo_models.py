"""Small models for the manual MCP execution checks in docs/mcp.md."""
import torch
from torch import nn


# tensorviz: input=torch.ones(2, 3)
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 4)
        self.act = nn.ReLU()
        self.head = nn.Linear(4, 2)
        self.unused = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.head(self.act(self.fc(x)))


# tensorviz: input=torch.ones(1, 1, 56, 56)
class BrokenResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.AvgPool2d(2)
        self.tail = nn.Conv2d(1, 2, 1)

    def forward(self, x):
        small = self.down(x)
        return self.tail(small + x)
