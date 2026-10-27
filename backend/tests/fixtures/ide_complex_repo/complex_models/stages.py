from torch import nn
from .blocks import ResidualBlock


class Stage(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.block1 = ResidualBlock(width)
        self.block2 = ResidualBlock(width)

    def forward(self, x):
        return self.block2(self.block1(x))


class Trunk(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.stage1 = Stage(width)
        self.stage2 = Stage(width)

    def forward(self, x):
        return self.stage2(self.stage1(x))
