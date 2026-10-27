"""Parallel branches are ordinary data flow, not inferred residual shortcuts."""
import torch
import torch.nn as nn


# tensorviz: input=2,32
class ParallelBranches(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Linear(32, 24)
        self.right = nn.Linear(32, 40)
        self.fuse = nn.Linear(64, 16)
        self.output = nn.Sigmoid()

    def forward(self, x):
        left = self.left(x)
        right = self.right(x)
        return self.output(self.fuse(torch.cat([left, right], dim=1)))
