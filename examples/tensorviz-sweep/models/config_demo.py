"""A model built from a config object, which is how real models are built.

The configured profile in tensorviz.toml uses make_model below to construct
ModelConfig explicitly. TOML stores the factory reference rather than a Python
expression. Change depth=2 in the factory to explore a different block count.
"""
from dataclasses import dataclass

import torch.nn as nn


@dataclass
class ModelConfig:
    depth: int
    width: int = 8


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


def make_model():
    """Construct the model for the saved TensorViz profile."""
    return ConfigDrivenNet(ModelConfig(depth=2))
