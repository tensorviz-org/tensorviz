"""A config dataclass and the module that reads it."""
from typing import Any


from torch.nn.modules.linear import Linear


from dataclasses import dataclass

import torch.nn as nn


@dataclass
class Cfg:
    width: int = 8
    hidden: int = 16


# How to build a model whose `__init__` takes a config OBJECT. No literal says
# "a default Cfg", so the declaration names the class and the worker builds it
# out of this module — the same module it is about to build Configured from.
# tensorviz: init=Configured(cfg=Cfg())
class Configured(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.proj = nn.Linear(in_features=cfg.width, out_features=cfg.hidden)
        self.out = nn.Linear(in_features=cfg.hidden, out_features=cfg.width)

    def forward(self, x) -> Any:
        return self.out(self.proj(x))

cfg = Cfg()
conf = Configured(cfg)

