"""A config dataclass and the module that reads it."""
from typing import Any


from torch.nn.modules.linear import Linear


from dataclasses import dataclass

import torch.nn as nn


@dataclass
class Cfg:
    width: int = 8
    hidden: int = 16


# The configured profile in tensorviz.toml uses make_model below.
class Configured(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.proj = nn.Linear(in_features=cfg.width, out_features=cfg.hidden)
        self.out = nn.Linear(in_features=cfg.hidden, out_features=cfg.width)

    def forward(self, x) -> Any:
        return self.out(self.proj(x))

cfg = Cfg()
conf = Configured(cfg)


def make_model():
    """Construct the model for the saved TensorViz profile."""
    return Configured(Cfg())
