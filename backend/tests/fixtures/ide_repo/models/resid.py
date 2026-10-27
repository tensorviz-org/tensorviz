"""A residual net with a ModuleList loop.

Fixture for the IDE projection tests (task.md §7). Bodies are deliberately tiny
— every dimension here is 8 — because the backend suite runs in one process and
these classes are instantiated by the M4 trace worker.

The AST tier cannot type `block(h)`: `block` is a loop variable, not a
`self.<attr>`, so it lands as an opaque `python` node carrying its source text
rather than vanishing. That is the point of having it here.
"""
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, width=8, depth=2):
        super().__init__()
        self.l1 = nn.Linear(width, width)
        self.blocks = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.l2 = nn.Linear(width, width)

    def forward(self, x):
        h = self.l1(x)
        for block in self.blocks:
            h = block(h)
        return x + self.l2(h)


# The example input the trace worker uses, in the file the model lives in
# (task.md M4). The worker's shape search knows images and token ids, so a net
# whose first layer is Linear(8, 8) is unguessable — and a graph with no shapes
# is what "unguessable" looks like on the canvas.
# tensorviz: input=2,8
