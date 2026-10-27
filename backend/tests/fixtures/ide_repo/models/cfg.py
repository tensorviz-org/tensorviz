"""A config dataclass and the module that reads it."""
from dataclasses import dataclass

import torch.nn as nn


@dataclass
class Cfg:
    width: int = 8
    hidden: int = 16


class Configured(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.proj = nn.Linear(cfg.width, cfg.hidden)
        self.out = nn.Linear(cfg.hidden, cfg.width)

    def forward(self, x):
        return self.out(self.proj(x))
