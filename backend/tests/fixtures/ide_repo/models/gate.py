"""Data-dependent control flow: the case torch.export refuses (task.md D3)."""
import torch.nn as nn


class Gate(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.up = nn.Linear(width, width)
        self.down = nn.Linear(width, width)

    def forward(self, x):
        if x.sum() > 0:
            x = self.up(x)
        return self.down(x)


# A negative sum on purpose: `up` must stay on the untaken side of the branch, so
# the projection has something to render as unreached. A random input would make
# which layer is dimmed a coin toss (task.md M4).
# tensorviz: input=torch.full((2, 8), -1.0)
