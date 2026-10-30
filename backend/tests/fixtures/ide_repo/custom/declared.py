"""Blocks that say their own output shape (task.md M6).

A trace can only report what it ran. ``Branchy`` deliberately does not run
either block — the example input is all negative, so ``x.sum() > 0`` is false —
which is exactly the case where a card would otherwise carry no shape at all and
everything downstream of it would go unknown.

``# tensorviz: out_shape=`` is the user's answer to that: the class states what
it produces, and the node carries ``_shape_provenance='declared'`` so the canvas
never presents a declaration as a measurement. Only an all-integer declaration
yields a usable ``_out_shape``; ``(B, D)`` names the axes and no numbers, so it
is recorded as declared and still has no shape to give — guessing one would put
a number the user never wrote in front of every node downstream.
"""
import torch.nn as nn


class Declared(nn.Module):
    # tensorviz: out_shape=(2, 16)
    def __init__(self, width=16):
        super().__init__()
        self.fc = nn.Linear(width, width)

    def forward(self, x):
        return self.fc(x)


class Symbolic(nn.Module):
    # tensorviz: out_shape=(B, D)
    def __init__(self, width=16):
        super().__init__()
        self.fc = nn.Linear(width, width)

    def forward(self, x):
        return self.fc(x)


# A negative sum on purpose: neither block runs, so the only thing that can say
# what they produce is the declaration in their own source.
# tensorviz: input=torch.full((2, 16), -1.0)
class Branchy(nn.Module):
    def __init__(self):
        super().__init__()
        self.declared = Declared(16)
        self.symbolic = Symbolic(16)
        self.head = nn.Linear(16, 16)

    def forward(self, x):
        if x.sum() > 0:
            x = self.declared(x)
            x = self.symbolic(x)
        return self.head(x)
