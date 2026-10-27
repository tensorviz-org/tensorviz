"""The root of the multi-file fixture.

Every supported import form lands here once:
  - absolute import              `from models.heads import Block as HeadBlock`
  - import alias                 both aliased names above and below
  - package re-export            `from models import CoreBlock`
  - relative imports             `from .mixer import Mixer` and friends
  - duplicate class names        heads.py and blocks.py both define `Block`
  - repeated instances           `self.enc1` / `self.enc2` of CoreBlock
  - cycle                        CycleA <-> CycleB, left opaque at the boundary
  - dynamic unresolved           DynamicStage's computed getattr
"""
import torch.nn as nn

from models import CoreBlock
from models.heads import Block as HeadBlock, WideBlock

from .cycle_a import CycleA
from .dynamic import DynamicStage
from .enc import Encoder
from .mixer import Mixer


class Net(nn.Module):
    def __init__(self, width: int = 8):
        super().__init__()
        self.inp = nn.Linear(width, width)
        self.enc1 = CoreBlock(width)
        self.enc2 = CoreBlock(width)
        self.head = HeadBlock(width)
        self.wide = WideBlock(width)
        self.mix = Mixer(width)
        self.stack = Encoder(width)
        self.loop = CycleA(width)
        self.dyn = DynamicStage(width)
        self.out = nn.Linear(width, width)

    def forward(self, x):
        x = self.inp(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.head(x)
        x = self.wide(x)
        x = self.mix(x)
        x = self.stack(x)
        x = self.loop(x)
        x = self.dyn(x)
        return self.out(x)


# tensorviz: input=8
