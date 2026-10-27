"""`import models.blocks as blocks` followed by `blocks.Block(...)` — the module-as-value form."""
import models.blocks as blocks


class Encoder(blocks.Block):
    def __init__(self, width: int = 8):
        super().__init__(width)

    def forward(self, x):
        return super().forward(x) + x
