"""A model built from a config object, which is how real models are built.

A class whose constructor takes a config cannot be opened by declaring numbers:
there is no literal that means "a default ModelConfig". So the declaration below
names the class to build, and the worker builds it out of this module — the same
module it is about to build the model from, so nothing runs that was not going to.

`depth` carries no default on purpose: it is what shows that the call takes its own
arguments. Deleting `depth=2` from the declaration is a good thing to try, because
the refusal then names ModelConfig rather than the model, and hands back this exact
line to put back.
"""
from dataclasses import dataclass

import torch.nn as nn


@dataclass
class ModelConfig:
    depth: int
    width: int = 8


# tensorviz: init=ConfigDrivenNet(cfg=ModelConfig(depth=2))
class ConfigDrivenNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.stem = nn.Linear(cfg.width, cfg.width)
        self.blocks = nn.ModuleList(
            [nn.Linear(cfg.width, cfg.width) for _ in range(cfg.depth)]
        )
        self.head = nn.Linear(cfg.width, cfg.width)

    def forward(self, x):
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h)
