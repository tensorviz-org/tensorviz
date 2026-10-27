from torch import nn
import blocks

# tensorviz: input=2,4
class DottedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = blocks.ScaleBlock()
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        return self.head(self.block(x))
