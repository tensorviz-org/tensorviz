from torch import nn
from .stages import Trunk
from .heads import Head


class ComplexNet(nn.Module):
    """Eight residual blocks, four nesting levels and distinct dotted/flat ids."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, padding=1)
        self.trunk1 = Trunk()
        self.trunk2 = Trunk()
        self.branch = Head()
        self.branch_fc = nn.Linear(10, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.trunk1(x)
        x = self.trunk2(x)
        x = self.branch(x)
        return self.branch_fc(x)
