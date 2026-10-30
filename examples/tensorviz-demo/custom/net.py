"""A model assembled from the user's OWN block (task.md M6).

``MyBlock`` lives in ``models/blocks.py``, so the node for ``self.blk`` has to
carry where the class is (``_import_path``), not only what it is called: "add
this block to another file" cannot write an import from a class name alone.

``Transposed`` is the other half of the same milestone. ``x.transpose(1, 2)`` is
a statement the AST tier cannot type, so it lands as an opaque ``python`` node
carrying its own source text — and that text is the only thing that can
regenerate it. It is the FIRST statement of its forward on purpose: the text
reads ``x``, and a generator that emitted it after another layer would silently
transpose a tensor the user never wrote that line for.
"""
import torch.nn as nn

from models.blocks import MyBlock


# tensorviz: input=2,16
class Custom(nn.Module):
    def __init__(self):
        super().__init__()
        self.blk = MyBlock(16)
        # MyBlock here ends Linear(10, 20), so the head takes 20 — not the 16 its
        # ide_repo twin takes, whose MyBlock is Linear(width, width).
        self.head = nn.Linear(20, 16)

    def forward(self, x):
        return self.head(self.blk(x))


class Transposed(nn.Module):
    # tensorviz: input=2,8,4
    # Its own: Custom's declaration above is 2-D, and `x.transpose(1, 2)` needs
    # three dimensions.
    def __init__(self, width=8):
        super().__init__()
        self.mix = nn.Linear(width, width)

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.mix(x)
